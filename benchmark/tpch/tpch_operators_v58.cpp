/**
 * ThunderDuck TPC-H Operators V58 Implementation
 *
 * @version 58
 * @date 2026-01-30
 */

#include "tpch_operators_v58.h"
#include "tpch_constants.h"
#include <algorithm>
#include <future>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v58 {

const char* V58_VERSION = "V58-DeepOptimization";
const char* V58_DATE = "2026-01-30";
const char* V58_FEATURES[] = {
    "Q3-DirectArrayAggregator",
    "Q9-PrecomputedBitmap",
    "Q2-ParallelScan",
    nullptr
};

// ============================================================================
// Q3 V58: DirectArrayAggregator 优化
// ============================================================================

void run_q3_v58(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;
    constexpr size_t FINAL_TOP_N = 10;

    // ========================================================================
    // Phase 1: 构建 BUILDING custkey bitmap
    // ========================================================================

    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }

    std::vector<uint8_t> is_building(static_cast<size_t>(max_custkey) + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            is_building[cust.c_custkey[i]] = 1;
        }
    }

    // ========================================================================
    // Phase 2: 构建紧凑 Hash Table + Bloom Filter + 直接聚合数组
    // ========================================================================

    // 找到最大 orderkey (用于 DirectArrayAggregator)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    // 预估有效订单数
    size_t estimated_valid = ord.count / 5;

    // 使用 Q3OrderHashTable (存储 orderdate + shippriority)
    Q3OrderHashTable order_table;
    order_table.init(estimated_valid);

    // Bloom Filter
    size_t bloom_bits = estimated_valid * 2;
    bloom_bits = ((bloom_bits + 63) / 64) * 64;
    std::vector<uint64_t> bloom(bloom_bits / 64, 0);
    uint32_t bloom_mask = static_cast<uint32_t>(bloom_bits - 1);

    // 构建
    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (custkey <= max_custkey && is_building[custkey] && orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];

#ifdef __aarch64__
            uint32_t hash = __crc32w(0, static_cast<uint32_t>(orderkey));
#else
            uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif
            order_table.insert(orderkey, orderdate,
                               static_cast<int8_t>(ord.o_shippriority[i]));
            bloom[(hash & bloom_mask) >> 6] |= (1ULL << ((hash & bloom_mask) & 63));
        }
    }

    // ========================================================================
    // Phase 3: 线程局部 DirectArrayAggregator (核心优化!)
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 使用 DirectArrayAggregator 替代 unordered_map
    std::vector<DirectArrayAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        agg.init(max_orderkey);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    uint32_t table_mask = order_table.mask();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            // 8 路展开 + 预取
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&li.l_shipdate[i + 64], 0, 3);
                __builtin_prefetch(&li.l_orderkey[i + 64], 0, 3);

                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;

                    // 快速日期过滤
                    if (li.l_shipdate[idx] <= DATE_THRESHOLD) continue;

                    int32_t orderkey = li.l_orderkey[idx];

#ifdef __aarch64__
                    uint32_t hash = __crc32w(0, static_cast<uint32_t>(orderkey));
#else
                    uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif

                    // Bloom Filter 快速拒绝
                    if (!((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1)) {
                        continue;
                    }

                    // Hash table 查找确认
                    const auto* entry = order_table.find(orderkey);
                    if (entry) {
                        // O(1) DirectArray 累加 (无 hash 开销!)
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[idx]) *
                                      (10000 - li.l_discount[idx]) / 10000;
                        local_agg.add(orderkey, rev);
                    }
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

#ifdef __aarch64__
                uint32_t hash = __crc32w(0, static_cast<uint32_t>(orderkey));
#else
                uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif

                if (!((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1)) {
                    continue;
                }

                const auto* entry = order_table.find(orderkey);
                if (entry) {
                    int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                    local_agg.add(orderkey, rev);
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 4: 合并线程局部 DirectArray (高效合并)
    // ========================================================================

    DirectArrayAggregator<int64_t> global_agg;
    global_agg.init(max_orderkey);

    for (auto& local : thread_aggs) {
        global_agg.merge(local);
    }

    // ========================================================================
    // Phase 5: 提取 Top-N
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int8_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(global_agg.count_nonzero());

    global_agg.for_each_nonzero([&](int32_t orderkey, int64_t revenue) {
        if (revenue <= 0) return;

        const auto* entry = order_table.find(orderkey);
        if (entry) {
            results.push_back({orderkey, revenue, entry->orderdate, entry->shippriority});
        }
    });

    // 部分排序获取 Top-N
    size_t top_n = std::min(FINAL_TOP_N, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > FINAL_TOP_N) results.resize(FINAL_TOP_N);

    // 防止优化
    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q9 V58: PrecomputedBitmap 优化
// ============================================================================

void run_q9_v58(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();

    // ========================================================================
    // Phase 1: 预计算 "green" parts 位图 (消除热路径字符串操作)
    // ========================================================================

    PrecomputedBitmap green_parts;
    green_parts.build_from_string_contains(
        part.p_partkey.data(),
        part.p_name.data(),
        part.count,
        "green"
    );

    // ========================================================================
    // Phase 2: 构建直接数组查找表
    // ========================================================================

    // supp → nation (直接数组)
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_suppkey[i] > max_suppkey) max_suppkey = supp.s_suppkey[i];
    }

    std::vector<int32_t> supp_to_nation(static_cast<size_t>(max_suppkey) + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        supp_to_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
    }

    // order → year (直接数组)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    std::vector<int16_t> order_to_year(static_cast<size_t>(max_orderkey) + 1, -1);
    for (size_t i = 0; i < ord.count; ++i) {
        int16_t year = static_cast<int16_t>(1970 + ord.o_orderdate[i] / 365);
        order_to_year[ord.o_orderkey[i]] = year;
    }

    // partsupp cost (复合键 hash map - 难以用直接数组)
    std::unordered_map<int64_t, int64_t> ps_cost_map;
    ps_cost_map.reserve(partsupp.count);
    for (size_t i = 0; i < partsupp.count; ++i) {
        int64_t key = (static_cast<int64_t>(partsupp.ps_partkey[i]) << 32) |
                      static_cast<uint32_t>(partsupp.ps_suppkey[i]);
        ps_cost_map[key] = partsupp.ps_supplycost[i];
    }

    // nation names
    std::vector<std::string> nation_names(25);
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_nationkey[i] >= 0 && nat.n_nationkey[i] < 25) {
            nation_names[nat.n_nationkey[i]] = nat.n_name[i];
        }
    }

    // ========================================================================
    // Phase 3: 并行扫描 lineitem (使用 DirectArrayAggregator)
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 聚合键: (nationkey << 16) | year
    std::vector<DirectArrayAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        agg.init(25 * 65536);  // 25 nations * 65536 years
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&li.l_partkey[i + 64], 0, 3);

                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;
                    int32_t partkey = li.l_partkey[idx];

                    // O(1) 位图测试 (替代 string::find)
                    if (!green_parts.test(partkey)) continue;

                    int32_t suppkey = li.l_suppkey[idx];
                    int32_t orderkey = li.l_orderkey[idx];

                    // O(1) 直接数组查找
                    if (suppkey < 0 || suppkey > max_suppkey) continue;
                    int32_t nationkey = supp_to_nation[suppkey];
                    if (nationkey < 0) continue;

                    if (orderkey < 0 || orderkey > max_orderkey) continue;
                    int16_t year = order_to_year[orderkey];
                    if (year < 0) continue;

                    // ps_cost 查找 (hash map - 复合键)
                    int64_t ps_key = (static_cast<int64_t>(partkey) << 32) |
                                     static_cast<uint32_t>(suppkey);
                    auto cost_it = ps_cost_map.find(ps_key);
                    if (cost_it == ps_cost_map.end()) continue;

                    // 计算 profit
                    __int128 disc_price = (__int128)li.l_extendedprice[idx] *
                                          (10000 - li.l_discount[idx]) / 10000;
                    __int128 cost = (__int128)cost_it->second * li.l_quantity[idx] / 10000;
                    int64_t amount = static_cast<int64_t>(disc_price - cost);

                    // O(1) 直接数组聚合
                    int32_t agg_key = (nationkey << 16) | (year & 0xFFFF);
                    local_agg.add(agg_key, amount);
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t partkey = li.l_partkey[i];
                if (!green_parts.test(partkey)) continue;

                int32_t suppkey = li.l_suppkey[i];
                int32_t orderkey = li.l_orderkey[i];

                if (suppkey < 0 || suppkey > max_suppkey) continue;
                int32_t nationkey = supp_to_nation[suppkey];
                if (nationkey < 0) continue;

                if (orderkey < 0 || orderkey > max_orderkey) continue;
                int16_t year = order_to_year[orderkey];
                if (year < 0) continue;

                int64_t ps_key = (static_cast<int64_t>(partkey) << 32) |
                                 static_cast<uint32_t>(suppkey);
                auto cost_it = ps_cost_map.find(ps_key);
                if (cost_it == ps_cost_map.end()) continue;

                __int128 disc_price = (__int128)li.l_extendedprice[i] *
                                      (10000 - li.l_discount[i]) / 10000;
                __int128 cost = (__int128)cost_it->second * li.l_quantity[i] / 10000;
                int64_t amount = static_cast<int64_t>(disc_price - cost);

                int32_t agg_key = (nationkey << 16) | (year & 0xFFFF);
                local_agg.add(agg_key, amount);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 4: 合并结果
    // ========================================================================

    DirectArrayAggregator<int64_t> global_agg;
    global_agg.init(25 * 65536);
    for (auto& local : thread_aggs) {
        global_agg.merge(local);
    }

    // 防止优化
    volatile int64_t sink = 0;
    global_agg.for_each_nonzero([&sink](int32_t, int64_t val) { sink += val; });
    (void)sink;
}

// ============================================================================
// Q2 V58: ParallelScan + SIMDSuffix 优化
// ============================================================================

void run_q2_v58(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // ========================================================================
    // Phase 1: 预计算 EUROPE suppliers
    // ========================================================================

    std::unordered_set<int32_t> europe_nations;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::EUROPE) {
            int32_t regionkey = reg.r_regionkey[i];
            for (size_t j = 0; j < nat.count; ++j) {
                if (nat.n_regionkey[j] == regionkey) {
                    europe_nations.insert(nat.n_nationkey[j]);
                }
            }
            break;
        }
    }

    // supp → nationkey/acctbal (直接数组)
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_suppkey[i] > max_suppkey) max_suppkey = supp.s_suppkey[i];
    }

    std::vector<int32_t> supp_nationkey(static_cast<size_t>(max_suppkey) + 1, -1);
    std::vector<int64_t> supp_acctbal(static_cast<size_t>(max_suppkey) + 1, 0);

    for (size_t i = 0; i < supp.count; ++i) {
        if (europe_nations.count(supp.s_nationkey[i])) {
            int32_t suppkey = supp.s_suppkey[i];
            supp_nationkey[suppkey] = supp.s_nationkey[i];
            supp_acctbal[suppkey] = supp.s_acctbal[i];
        }
    }

    // ========================================================================
    // Phase 2: 预计算 "BRASS" 后缀位图 (使用 PrecomputedBitmap)
    // ========================================================================

    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_partkey[i] > max_partkey) max_partkey = part.p_partkey[i];
    }

    // 构建 size=15 且 type 以 BRASS 结尾的 parts 位图
    PrecomputedBitmap brass_parts;
    brass_parts.build(part.p_partkey.data(), part.count, [&](size_t i) {
        if (part.p_size[i] != 15) return false;
        const auto& ptype = part.p_type[i];
        if (ptype.size() < 5) return false;
        return ptype.compare(ptype.size() - 5, 5, "BRASS") == 0;
    });

    // ========================================================================
    // Phase 3: 预计算 MIN(ps_supplycost) per partkey (DirectArrayDecorrelation)
    // ========================================================================

    std::vector<int64_t> min_cost(static_cast<size_t>(max_partkey) + 1, INT64_MAX);
    std::vector<bool> has_cost(static_cast<size_t>(max_partkey) + 1, false);

    for (size_t i = 0; i < partsupp.count; ++i) {
        int32_t suppkey = partsupp.ps_suppkey[i];

        // 只考虑 EUROPE suppliers
        if (suppkey < 0 || suppkey > max_suppkey || supp_nationkey[suppkey] < 0) continue;

        int32_t partkey = partsupp.ps_partkey[i];
        if (partkey < 0 || partkey > max_partkey) continue;

        int64_t cost = partsupp.ps_supplycost[i];
        if (!has_cost[partkey] || cost < min_cost[partkey]) {
            min_cost[partkey] = cost;
            has_cost[partkey] = true;
        }
    }

    // ========================================================================
    // Phase 4: 并行扫描 partsupp (核心优化!)
    // ========================================================================

    struct Q2Result {
        int64_t s_acctbal;
        int32_t suppkey;
        int32_t partkey;
        int32_t nationkey;
    };

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;
    size_t chunk_size = (partsupp.count + num_threads - 1) / num_threads;

    std::vector<std::vector<Q2Result>> thread_results(num_threads);
    for (auto& r : thread_results) r.reserve(100);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, partsupp.count);
        if (start >= partsupp.count) break;

        threads.emplace_back([&, t, start, end]() {
            auto& local_results = thread_results[t];

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&partsupp.ps_partkey[i + 64], 0, 3);

                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;
                    int32_t partkey = partsupp.ps_partkey[idx];
                    int32_t suppkey = partsupp.ps_suppkey[idx];
                    int64_t supplycost = partsupp.ps_supplycost[idx];

                    // O(1) 位图测试
                    if (!brass_parts.test(partkey)) continue;

                    // O(1) 直接数组测试
                    if (suppkey < 0 || suppkey > max_suppkey ||
                        supp_nationkey[suppkey] < 0) continue;

                    // O(1) min_cost 比较
                    if (!has_cost[partkey] || supplycost != min_cost[partkey]) continue;

                    local_results.push_back({
                        supp_acctbal[suppkey],
                        suppkey,
                        partkey,
                        supp_nationkey[suppkey]
                    });
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t partkey = partsupp.ps_partkey[i];
                int32_t suppkey = partsupp.ps_suppkey[i];
                int64_t supplycost = partsupp.ps_supplycost[i];

                if (!brass_parts.test(partkey)) continue;
                if (suppkey < 0 || suppkey > max_suppkey ||
                    supp_nationkey[suppkey] < 0) continue;
                if (!has_cost[partkey] || supplycost != min_cost[partkey]) continue;

                local_results.push_back({
                    supp_acctbal[suppkey],
                    suppkey,
                    partkey,
                    supp_nationkey[suppkey]
                });
            }
        });
    }

    for (auto& th : threads) th.join();

    // ========================================================================
    // Phase 5: 合并 + 排序
    // ========================================================================

    std::vector<Q2Result> results;
    size_t total = 0;
    for (const auto& r : thread_results) total += r.size();
    results.reserve(total);

    for (auto& r : thread_results) {
        results.insert(results.end(), r.begin(), r.end());
    }

    // 排序 (s_acctbal DESC, nationkey, suppkey, partkey)
    std::sort(results.begin(), results.end(), [](const Q2Result& a, const Q2Result& b) {
        if (a.s_acctbal != b.s_acctbal) return a.s_acctbal > b.s_acctbal;
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        if (a.suppkey != b.suppkey) return a.suppkey < b.suppkey;
        return a.partkey < b.partkey;
    });

    // 取前 100
    if (results.size() > 100) results.resize(100);

    // 防止优化
    volatile size_t sink = results.size();
    (void)sink;
}

} // namespace ops_v58
} // namespace tpch
} // namespace thunderduck
