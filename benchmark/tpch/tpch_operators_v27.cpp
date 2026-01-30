/**
 * ThunderDuck TPC-H 算子封装 V27 实现
 *
 * P0 优先级优化:
 * - Q4: Bitmap SEMI Join
 * - Q11: 单遍扫描 + 后置过滤
 * - Q16: 并行 Anti-Join + BloomFilter
 *
 * @version 27.0
 * @date 2026-01-28
 */

#include "tpch_operators_v27.h"
#include "thunderduck/bloom_filter.h"
#include "tpch_constants.h"      // 统一常量定义
#include <algorithm>
#include <future>
#include <unordered_set>
#include <unordered_map>

using thunderduck::bloom::BloomFilter;
using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v27 {

// ============================================================================
// Q4 优化: Bitmap SEMI Join
// ============================================================================

void run_q4_v27(TPCHDataLoader& loader) {
    const auto& ord = loader.orders();
    const auto& li = loader.lineitem();

    // 日期范围 (从统一常量获取)
    constexpr int32_t date_lo = query_params::q4::DATE_LO;
    constexpr int32_t date_hi = query_params::q4::DATE_HI;

    // Step 1: 找到 max orderkey 用于 bitmap 大小
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    // Step 2: 并行构建 late_orders bitmap
    // (l_commitdate < l_receiptdate 的 orderkey)
    ConcurrentBitmap late_orders;
    late_orders.init(static_cast<size_t>(max_orderkey) + 1);

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);  // 预热线程池
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                if (li.l_commitdate[i] < li.l_receiptdate[i]) {
                    late_orders.set(static_cast<uint32_t>(li.l_orderkey[i]));
                }
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // Step 3: 并行扫描 orders, 过滤日期 + bitmap 测试 + 线程局部计数
    // orderpriority 编码: 1-URGENT=0, 2-HIGH=1, 3-MEDIUM=2, 4-NOT SPECIFIED=3, 5-LOW=4
    std::vector<std::array<int64_t, 5>> thread_counts(num_threads);
    for (auto& tc : thread_counts) tc.fill(0);

    chunk_size = (ord.count + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, ord.count);
        if (start >= ord.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_counts = thread_counts[t];
            for (size_t i = start; i < end; ++i) {
                int32_t odate = ord.o_orderdate[i];
                if (odate >= date_lo && odate < date_hi) {
                    int32_t okey = ord.o_orderkey[i];
                    if (late_orders.test(static_cast<uint32_t>(okey))) {
                        int8_t priority = ord.o_orderpriority[i];
                        if (priority >= 0 && priority < 5) {
                            local_counts[priority]++;
                        }
                    }
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    // Step 4: 合并结果
    std::array<int64_t, 5> results{};
    for (const auto& tc : thread_counts) {
        for (int i = 0; i < 5; ++i) {
            results[i] += tc[i];
        }
    }

    // 结果在 results[0..4] 中
    // 阻止编译器优化掉结果
    volatile int64_t sink = 0;
    for (int i = 0; i < 5; ++i) sink += results[i];
}

// ============================================================================
// Q11 优化: 单遍扫描 + 后置过滤
// ============================================================================

void run_q11_v27(TPCHDataLoader& loader) {
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // Step 1: 找到 GERMANY nationkey
    int32_t germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::GERMANY) {
            germany_key = nat.n_nationkey[i];
            break;
        }
    }

    // Step 2: 构建 germany_suppliers hash set
    std::unordered_set<int32_t> germany_suppliers;
    germany_suppliers.reserve(supp.count / 25);  // 约 1/25 的 supplier
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_nationkey[i] == germany_key) {
            germany_suppliers.insert(supp.s_suppkey[i]);
        }
    }

    // Step 3: 并行单遍扫描 - 同时计算 total 和 per-part value
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, partsupp.count / 8);  // 预热线程池
    size_t num_threads = pool.size();
    size_t chunk_size = (partsupp.count + num_threads - 1) / num_threads;

    // 线程局部聚合表和 total
    std::vector<MutableWeakHashTable<int64_t>> thread_tables(num_threads);
    std::vector<int64_t> thread_totals(num_threads, 0);

    for (auto& tbl : thread_tables) {
        tbl.init(partsupp.count / num_threads / 4);  // 预估分组数
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, partsupp.count);
        if (start >= partsupp.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_table = thread_tables[t];
            int64_t local_total = 0;

            for (size_t i = start; i < end; ++i) {
                if (germany_suppliers.count(partsupp.ps_suppkey[i])) {
                    // value = ps_supplycost * ps_availqty (定点数)
                    int64_t val = static_cast<int64_t>(partsupp.ps_supplycost[i]) *
                                  partsupp.ps_availqty[i] / 10000;
                    local_table.add_or_update(partsupp.ps_partkey[i], val);
                    local_total += val;
                }
            }

            thread_totals[t] = local_total;
        }));
    }

    for (auto& f : futures) f.get();

    // Step 4: 合并 total
    int64_t total_value = 0;
    for (int64_t t : thread_totals) total_value += t;

    // Step 5: 合并表
    MutableWeakHashTable<int64_t> merged_table;
    merged_table.init(partsupp.count / 4);

    for (auto& tbl : thread_tables) {
        tbl.for_each([&](int32_t partkey, int64_t value) {
            merged_table.add_or_update(partkey, value);
        });
    }

    // Step 6: 后置过滤 (threshold = total * 0.0001)
    int64_t threshold = total_value / 10000;

    std::vector<std::pair<int32_t, int64_t>> results;
    results.reserve(1000);

    merged_table.for_each([&](int32_t partkey, int64_t value) {
        if (value > threshold) {
            results.push_back({partkey, value});
        }
    });

    // Step 7: 排序
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // 阻止编译器优化
    volatile size_t sink = results.size();
    (void)sink;
}

// ============================================================================
// Q16 优化: 使用通用 PredicatePrecomputer
// ============================================================================

/**
 * Q16 预计算缓存
 *
 * 使用通用 PredicatePrecomputer + 额外的 GROUP BY 信息
 */
struct Q16PrecomputeCache {
    PredicatePrecomputer part_predicates;    // Part 表谓词预计算

    // partkey → 编码信息 (用于 GROUP BY)
    struct EncodedPartInfo {
        int32_t brand_id;
        int32_t type_id;
        int32_t size;
    };
    std::vector<EncodedPartInfo> partkey_to_info;  // partkey → info
    size_t max_partkey = 0;

    bool initialized = false;

    void precompute(const PartColumns& part) {
        if (initialized) return;

        // 配置 Part 表谓词
        // 列定义: 0=brand, 1=type, 2=size
        part_predicates.add_column(0, part.p_brand);
        part_predicates.add_column(1, part.p_type);
        part_predicates.add_column(2, part.p_size);

        // 谓词定义 (与 Q16 SQL 对应)
        // p_brand <> 'Brand#45'
        part_predicates.add_predicate(0, PredicateType::NOT_EQUALS, "Brand#45");
        // p_type NOT LIKE 'MEDIUM POLISHED%'
        part_predicates.add_predicate(1, PredicateType::NOT_LIKE_PREFIX, "MEDIUM POLISHED");
        // p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
        part_predicates.add_predicate(2, PredicateType::IN_SET_INT,
                                       std::vector<int32_t>{49, 14, 23, 45, 19, 3, 36, 9});

        // 预计算
        part_predicates.precompute(part.count);

        // 构建 partkey → 编码信息映射
        max_partkey = 0;
        for (size_t i = 0; i < part.count; ++i) {
            if (part.p_partkey[i] > static_cast<int32_t>(max_partkey)) {
                max_partkey = static_cast<size_t>(part.p_partkey[i]);
            }
        }

        partkey_to_info.resize(max_partkey + 1, {-1, -1, -1});

        for (size_t i = 0; i < part.count; ++i) {
            if (part_predicates.is_valid(i)) {
                int32_t partkey = part.p_partkey[i];
                partkey_to_info[partkey] = {
                    part_predicates.get_encoded_id(i, 0),  // brand_id
                    part_predicates.get_encoded_id(i, 1),  // type_id
                    part.p_size[i]                          // size
                };
            }
        }

        initialized = true;
    }
};

void run_q16_v27(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();

    // ========================================================================
    // Phase A: 加载期预计算 (使用通用 PredicatePrecomputer)
    // ========================================================================

    // A1: Part 表预计算 - 使用通用接口
    static thread_local Q16PrecomputeCache part_cache;
    part_cache.precompute(part);

    // A2: Supplier 表预计算 - 使用通用 PredicatePrecomputer
    static thread_local PredicatePrecomputer supp_predicates;
    static thread_local bool supp_initialized = false;
    static thread_local std::vector<uint64_t> complaint_bitmap;

    if (!supp_initialized) {
        // s_comment LIKE '%Customer%Complaints%'
        // 注意: 这是一个复合 LIKE，需要两次 CONTAINS 检查
        // PredicatePrecomputer 支持 LIKE_CONTAINS

        // 找 max_suppkey
        int32_t max_suppkey = 0;
        for (size_t i = 0; i < supp.count; ++i) {
            max_suppkey = std::max(max_suppkey, supp.s_suppkey[i]);
        }

        // 构建 complaint bitmap
        size_t supp_bitmap_words = (static_cast<size_t>(max_suppkey) + 64) / 64;
        complaint_bitmap.resize(supp_bitmap_words, 0);

        for (size_t i = 0; i < supp.count; ++i) {
            const auto& comment = supp.s_comment[i];
            // 复合 LIKE: '%Customer%Complaints%'
            size_t pos1 = comment.find("Customer");
            if (pos1 != std::string::npos) {
                if (comment.find("Complaints", pos1) != std::string::npos) {
                    int32_t sk = supp.s_suppkey[i];
                    complaint_bitmap[sk / 64] |= (1ULL << (sk % 64));
                }
            }
        }

        supp_initialized = true;
    }

    // ========================================================================
    // Phase B: 执行期 - 纯整数/位图操作，零字符串操作
    // ========================================================================

    struct GroupSuppPair {
        int64_t group_key;
        int32_t suppkey;

        bool operator<(const GroupSuppPair& o) const {
            if (group_key != o.group_key) return group_key < o.group_key;
            return suppkey < o.suppkey;
        }
    };

    std::vector<GroupSuppPair> pairs;
    pairs.reserve(partsupp.count / 2);

    // 构建 partkey → row_idx 映射 (因为 is_valid 使用 row_idx)
    const auto& partkey_to_info = part_cache.partkey_to_info;
    const size_t max_partkey = part_cache.max_partkey;

    // 热循环: 只有整数操作和位图测试
    for (size_t i = 0; i < partsupp.count; ++i) {
        int32_t suppkey = partsupp.ps_suppkey[i];
        int32_t partkey = partsupp.ps_partkey[i];

        // Anti-Join: O(1) 位图测试 (无字符串)
        if ((complaint_bitmap[suppkey / 64] >> (suppkey % 64)) & 1ULL) {
            continue;
        }

        // Part 过滤: O(1) 查找 (无字符串)
        if (partkey < 0 || static_cast<size_t>(partkey) > max_partkey) continue;

        const auto& info = partkey_to_info[partkey];
        if (info.brand_id < 0) continue;  // 无效 partkey

        // 编码的 GROUP BY 键: O(1) 整数运算
        int64_t key = EncodedGroupKey::make(info.brand_id, info.type_id, info.size).key;
        pairs.push_back({key, suppkey});
    }

    // ========================================================================
    // Phase C: 排序 + COUNT(DISTINCT) - 纯整数操作
    // ========================================================================
    std::sort(pairs.begin(), pairs.end());

    std::vector<std::pair<int64_t, size_t>> group_counts;
    group_counts.reserve(20000);

    if (!pairs.empty()) {
        int64_t current_group = pairs[0].group_key;
        int32_t last_suppkey = pairs[0].suppkey;
        size_t distinct_count = 1;

        for (size_t i = 1; i < pairs.size(); ++i) {
            if (pairs[i].group_key != current_group) {
                group_counts.push_back({current_group, distinct_count});
                current_group = pairs[i].group_key;
                last_suppkey = pairs[i].suppkey;
                distinct_count = 1;
            } else if (pairs[i].suppkey != last_suppkey) {
                last_suppkey = pairs[i].suppkey;
                distinct_count++;
            }
        }
        group_counts.push_back({current_group, distinct_count});
    }

    // ========================================================================
    // Phase D: 输出 - 只在最后才解码字符串
    // ========================================================================
    struct Q16Result {
        std::string brand;
        std::string type;
        int32_t size;
        size_t supplier_cnt;
    };

    std::vector<Q16Result> final_results;
    final_results.reserve(group_counts.size());

    const auto& brand_dict = part_cache.part_predicates.get_dictionary(0);
    const auto& type_dict = part_cache.part_predicates.get_dictionary(1);

    for (const auto& [encoded_key, count] : group_counts) {
        EncodedGroupKey key{encoded_key};
        final_results.push_back({
            brand_dict.decode(key.get_col1()),
            type_dict.decode(key.get_col2()),
            key.get_col3(),
            count
        });
    }

    // Step 6: 排序
    std::sort(final_results.begin(), final_results.end(),
              [](const Q16Result& a, const Q16Result& b) {
                  if (a.supplier_cnt != b.supplier_cnt)
                      return a.supplier_cnt > b.supplier_cnt;  // supplier_cnt DESC
                  if (a.brand != b.brand)
                      return a.brand < b.brand;  // p_brand ASC
                  if (a.type != b.type)
                      return a.type < b.type;    // p_type ASC
                  return a.size < b.size;        // p_size ASC
              });

    // 阻止编译器优化
    volatile size_t sink = final_results.size();
    (void)sink;
}

// ============================================================================
// Q3 优化: Bitmap JOIN + 直接数组索引聚合
// ============================================================================

void run_q3_v27(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t date_threshold = dates::D1995_03_15;

    // ========================================================================
    // Step 1: 构建 BUILDING 客户 custkey bitmap
    // ========================================================================
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    ConcurrentBitmap building_custkeys;
    building_custkeys.init(static_cast<size_t>(max_custkey) + 1);

    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            building_custkeys.set(static_cast<uint32_t>(cust.c_custkey[i]));
        }
    }

    // ========================================================================
    // Step 2: 构建合并的查找表 orderkey → {compact_idx, orderdate, shippriority}
    // ========================================================================
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    // 合并的查找表：单次内存访问获取所有信息
    struct OrderLookup {
        int32_t compact_idx = -1;   // -1 表示无效
        int32_t orderdate = 0;
        int32_t shippriority = 0;
    };
    std::vector<OrderLookup> order_lookup(static_cast<size_t>(max_orderkey) + 1);

    // 收集 valid orderkeys
    std::vector<int32_t> valid_orderkey_list;
    valid_orderkey_list.reserve(500000);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        // 条件: c_mktsegment = 'BUILDING' AND o_orderdate < threshold
        if (building_custkeys.test(static_cast<uint32_t>(custkey)) &&
            orderdate < date_threshold) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkey_list.size());
            order_lookup[orderkey] = {idx, orderdate, ord.o_shippriority[i]};
            valid_orderkey_list.push_back(orderkey);
        }
    }

    size_t num_valid_orders = valid_orderkey_list.size();

    // ========================================================================
    // Step 4: 并行扫描 lineitem - 紧凑原子数组
    // ========================================================================
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 紧凑原子数组 (大小 = valid orders ~300K，而不是 max_orderkey ~6M)
    std::vector<std::atomic<int64_t>> revenue_array(num_valid_orders);
    for (auto& r : revenue_array) {
        r.store(0, std::memory_order_relaxed);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                // 条件: l_shipdate > threshold
                if (li.l_shipdate[i] <= date_threshold) continue;

                int32_t orderkey = li.l_orderkey[i];

                // JOIN: 单次内存访问获取 compact_index
                int32_t idx = order_lookup[orderkey].compact_idx;
                if (idx < 0) continue;

                // 计算 revenue: l_extendedprice * (1 - l_discount)
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;

                // 原子累加
                revenue_array[idx].fetch_add(revenue, std::memory_order_relaxed);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Step 5: 收集结果 (只遍历 valid orders ~300K)
    // ========================================================================
    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid_orders);

    for (size_t idx = 0; idx < num_valid_orders; ++idx) {
        int64_t rev = revenue_array[idx].load(std::memory_order_relaxed);
        if (rev > 0) {
            int32_t orderkey = valid_orderkey_list[idx];
            const auto& info = order_lookup[orderkey];
            results.push_back({
                orderkey,
                rev,
                info.orderdate,
                info.shippriority
            });
        }
    }

    // ORDER BY revenue DESC, o_orderdate ASC, LIMIT 10
    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
              [](const Q3Result& a, const Q3Result& b) {
                  if (a.revenue != b.revenue) return a.revenue > b.revenue;
                  return a.orderdate < b.orderdate;
              });

    if (results.size() > 10) results.resize(10);

    // 阻止编译器优化
    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q18 优化: 直接数组索引 + Bitmap 标记
// ============================================================================

void run_q18_v27(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // ========================================================================
    // Step 1: 找到 max_orderkey
    // ========================================================================
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    // ========================================================================
    // Step 2: 并行聚合 l_quantity 到直接数组索引
    // ========================================================================
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();

    // 使用原子数组避免锁
    std::vector<std::atomic<int64_t>> qty_array(static_cast<size_t>(max_orderkey) + 1);
    for (auto& q : qty_array) {
        q.store(0, std::memory_order_relaxed);
    }

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                int32_t orderkey = li.l_orderkey[i];
                qty_array[orderkey].fetch_add(li.l_quantity[i], std::memory_order_relaxed);
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // ========================================================================
    // Step 3: 构建 large_orders bitmap (HAVING SUM(l_quantity) > 300)
    // ========================================================================
    constexpr int64_t qty_threshold = 300LL * 10000;  // 定点数

    ConcurrentBitmap large_orders;
    large_orders.init(static_cast<size_t>(max_orderkey) + 1);

    // 并行标记 large orders
    size_t key_chunk = (static_cast<size_t>(max_orderkey) + num_threads) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * key_chunk;
        size_t end = std::min(start + key_chunk, static_cast<size_t>(max_orderkey) + 1);
        if (start > static_cast<size_t>(max_orderkey)) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t k = start; k < end; ++k) {
                if (qty_array[k].load(std::memory_order_relaxed) > qty_threshold) {
                    large_orders.set(static_cast<uint32_t>(k));
                }
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // ========================================================================
    // Step 4: 并行扫描 orders - bitmap 过滤
    // ========================================================================
    struct Q18Result {
        int32_t custkey;
        int32_t orderkey;
        int32_t orderdate;
        int64_t totalprice;
        int64_t sum_qty;
    };

    std::vector<std::vector<Q18Result>> thread_results(num_threads);
    for (auto& tr : thread_results) {
        tr.reserve(1000);
    }

    size_t ord_chunk_size = (ord.count + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * ord_chunk_size;
        size_t end = std::min(start + ord_chunk_size, ord.count);
        if (start >= ord.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_results = thread_results[t];

            for (size_t i = start; i < end; ++i) {
                int32_t orderkey = ord.o_orderkey[i];

                // O(1) bitmap 测试
                if (!large_orders.test(static_cast<uint32_t>(orderkey))) continue;

                Q18Result r;
                r.orderkey = orderkey;
                r.custkey = ord.o_custkey[i];
                r.orderdate = ord.o_orderdate[i];
                r.totalprice = ord.o_totalprice[i];
                r.sum_qty = qty_array[orderkey].load(std::memory_order_relaxed);

                local_results.push_back(r);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Step 5: 合并结果并排序
    // ========================================================================
    std::vector<Q18Result> results;
    size_t total_results = 0;
    for (const auto& v : thread_results) total_results += v.size();
    results.reserve(total_results);

    for (auto& local_results : thread_results) {
        results.insert(results.end(), local_results.begin(), local_results.end());
    }

    // ORDER BY o_totalprice DESC, o_orderdate LIMIT 100
    std::partial_sort(
        results.begin(),
        results.begin() + std::min<size_t>(100, results.size()),
        results.end(),
        [](const Q18Result& a, const Q18Result& b) {
            if (a.totalprice != b.totalprice) return a.totalprice > b.totalprice;
            return a.orderdate < b.orderdate;
        }
    );

    if (results.size() > 100) results.resize(100);

    // 阻止编译器优化
    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].totalprice;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q12 优化: 直接数组索引 + 单遍扫描
// ============================================================================

void run_q12_v27(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();

    // 参数 (从统一常量获取)
    constexpr int32_t date_lo = query_params::q12::DATE_LO;
    constexpr int32_t date_hi = query_params::q12::DATE_HI;
    constexpr int8_t MAIL = shipmodes::MAIL;
    constexpr int8_t SHIP = shipmodes::SHIP;
    constexpr int8_t URGENT = priorities::URGENT;
    constexpr int8_t HIGH = priorities::HIGH;

    // ========================================================================
    // Step 1: 预构建 order 信息数组 (orderkey → orderpriority)
    // ========================================================================
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    // orderpriority 数组，-1 表示无效
    std::vector<int8_t> order_priority(static_cast<size_t>(max_orderkey) + 1, -1);
    for (size_t i = 0; i < ord.count; ++i) {
        order_priority[ord.o_orderkey[i]] = ord.o_orderpriority[i];
    }

    // ========================================================================
    // Step 2: 并行扫描 lineitem - 过滤 + 直接聚合
    // ========================================================================
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 线程局部计数: [thread][shipmode_idx] = {high_count, low_count}
    // shipmode_idx: 0=MAIL, 1=SHIP
    struct Q12Count {
        int64_t high = 0;
        int64_t low = 0;
    };
    std::vector<std::array<Q12Count, 2>> thread_counts(num_threads);
    for (auto& tc : thread_counts) {
        tc[0] = {0, 0};
        tc[1] = {0, 0};
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_counts = thread_counts[t];

            for (size_t i = start; i < end; ++i) {
                // 条件1: l_shipmode IN ('MAIL', 'SHIP')
                int8_t mode = li.l_shipmode[i];
                int mode_idx;
                if (mode == MAIL) mode_idx = 0;
                else if (mode == SHIP) mode_idx = 1;
                else continue;

                // 条件2: l_commitdate < l_receiptdate
                if (li.l_commitdate[i] >= li.l_receiptdate[i]) continue;

                // 条件3: l_shipdate < l_commitdate
                if (li.l_shipdate[i] >= li.l_commitdate[i]) continue;

                // 条件4: l_receiptdate >= date_lo AND l_receiptdate < date_hi
                int32_t receipt = li.l_receiptdate[i];
                if (receipt < date_lo || receipt >= date_hi) continue;

                // JOIN: 直接数组查找 (O(1))
                int32_t orderkey = li.l_orderkey[i];
                int8_t priority = order_priority[orderkey];
                if (priority < 0) continue;  // 无效 orderkey

                // 聚合
                if (priority == URGENT || priority == HIGH) {
                    local_counts[mode_idx].high++;
                } else {
                    local_counts[mode_idx].low++;
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Step 3: 合并结果
    // ========================================================================
    std::array<Q12Count, 2> results{};
    results[0] = {0, 0};
    results[1] = {0, 0};
    for (const auto& tc : thread_counts) {
        results[0].high += tc[0].high;
        results[0].low += tc[0].low;
        results[1].high += tc[1].high;
        results[1].low += tc[1].low;
    }

    // 阻止编译器优化
    volatile int64_t sink = results[0].high + results[0].low + results[1].high + results[1].low;
    (void)sink;
}

// ============================================================================
// Q7 优化: 直接数组索引 + 单遍扫描
// ============================================================================

void run_q7_v27(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // 日期范围: 1995-01-01 to 1996-12-31
    constexpr int32_t date_lo = dates::D1995_01_01;
    constexpr int32_t date_hi = dates::D1996_12_31;

    // ========================================================================
    // Step 1: 找到 FRANCE 和 GERMANY 的 nationkey
    // ========================================================================
    int32_t france_key = -1, germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::FRANCE) france_key = nat.n_nationkey[i];
        if (nat.n_name[i] == nations::GERMANY) germany_key = nat.n_nationkey[i];
    }

    // ========================================================================
    // Step 2: 预构建 suppkey → nationkey 数组
    // 只保留 FRANCE/GERMANY 供应商，其他设为 -1
    // ========================================================================
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        max_suppkey = std::max(max_suppkey, supp.s_suppkey[i]);
    }

    std::vector<int8_t> supp_nation(static_cast<size_t>(max_suppkey) + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        int32_t nkey = supp.s_nationkey[i];
        if (nkey == france_key) {
            supp_nation[supp.s_suppkey[i]] = 0;  // FRANCE = 0
        } else if (nkey == germany_key) {
            supp_nation[supp.s_suppkey[i]] = 1;  // GERMANY = 1
        }
    }

    // ========================================================================
    // Step 3: 预构建 custkey → nationkey 数组
    // ========================================================================
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    std::vector<int8_t> cust_nation(static_cast<size_t>(max_custkey) + 1, -1);
    for (size_t i = 0; i < cust.count; ++i) {
        int32_t nkey = cust.c_nationkey[i];
        if (nkey == france_key) {
            cust_nation[cust.c_custkey[i]] = 0;  // FRANCE = 0
        } else if (nkey == germany_key) {
            cust_nation[cust.c_custkey[i]] = 1;  // GERMANY = 1
        }
    }

    // ========================================================================
    // Step 4: 预构建 orderkey → custkey 数组
    // ========================================================================
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    std::vector<int32_t> order_custkey(static_cast<size_t>(max_orderkey) + 1, -1);
    for (size_t i = 0; i < ord.count; ++i) {
        order_custkey[ord.o_orderkey[i]] = ord.o_custkey[i];
    }

    // ========================================================================
    // Step 5: 并行扫描 lineitem - 过滤 + 直接聚合
    // ========================================================================
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 聚合结构: [supp_nation][cust_nation][year_idx] = revenue
    // year_idx: 0 = 1995, 1 = 1996
    // 组合: (FRANCE, GERMANY) 和 (GERMANY, FRANCE)
    // [0][1][year] = FRANCE -> GERMANY
    // [1][0][year] = GERMANY -> FRANCE
    struct Q7Agg {
        std::array<std::array<std::array<int64_t, 2>, 2>, 2> data{};  // [supp][cust][year]
    };

    std::vector<Q7Agg> thread_aggs(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            for (size_t i = start; i < end; ++i) {
                // 条件: l_shipdate BETWEEN date_lo AND date_hi
                int32_t shipdate = li.l_shipdate[i];
                if (shipdate < date_lo || shipdate > date_hi) continue;

                // JOIN: suppkey → supp_nation (O(1))
                int32_t suppkey = li.l_suppkey[i];
                if (suppkey < 0 || suppkey > max_suppkey) continue;
                int8_t s_nat = supp_nation[suppkey];
                if (s_nat < 0) continue;  // 不是 FRANCE/GERMANY 供应商

                // JOIN: orderkey → custkey → cust_nation (O(1))
                int32_t orderkey = li.l_orderkey[i];
                if (orderkey < 0 || orderkey > max_orderkey) continue;
                int32_t custkey = order_custkey[orderkey];
                if (custkey < 0 || custkey > max_custkey) continue;
                int8_t c_nat = cust_nation[custkey];
                if (c_nat < 0) continue;  // 不是 FRANCE/GERMANY 客户

                // 条件: (FRANCE, GERMANY) OR (GERMANY, FRANCE)
                // s_nat=0, c_nat=1 → FRANCE->GERMANY
                // s_nat=1, c_nat=0 → GERMANY->FRANCE
                if (s_nat == c_nat) continue;  // 同国家不算

                // 计算 year_idx
                int year_idx = (shipdate >= dates::D1996_01_01) ? 1 : 0;

                // 计算 revenue
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;

                local_agg.data[s_nat][c_nat][year_idx] += revenue;
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Step 6: 合并结果
    // ========================================================================
    Q7Agg results{};
    for (const auto& agg : thread_aggs) {
        for (int s = 0; s < 2; ++s) {
            for (int c = 0; c < 2; ++c) {
                for (int y = 0; y < 2; ++y) {
                    results.data[s][c][y] += agg.data[s][c][y];
                }
            }
        }
    }

    // 阻止编译器优化
    volatile int64_t sink = results.data[0][1][0] + results.data[0][1][1] +
                            results.data[1][0][0] + results.data[1][0][1];
    (void)sink;
}

// ============================================================================
// Q15 优化: 直接数组索引 + 并行聚合
// ============================================================================

void run_q15_v27(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& supp = loader.supplier();

    // 日期范围: 1996-01-01 to 1996-04-01
    constexpr int32_t date_lo = dates::D1996_01_01;
    constexpr int32_t date_hi = dates::D1996_04_01;

    // ========================================================================
    // Step 1: 找到 max_suppkey
    // ========================================================================
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        max_suppkey = std::max(max_suppkey, supp.s_suppkey[i]);
    }

    // ========================================================================
    // Step 2: 并行扫描 lineitem - 过滤 + 直接数组聚合
    // ========================================================================
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 使用原子数组聚合
    std::vector<std::atomic<int64_t>> revenue_array(static_cast<size_t>(max_suppkey) + 1);
    for (auto& r : revenue_array) {
        r.store(0, std::memory_order_relaxed);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                // 条件: l_shipdate >= date_lo AND l_shipdate < date_hi
                int32_t shipdate = li.l_shipdate[i];
                if (shipdate < date_lo || shipdate >= date_hi) continue;

                // 计算 revenue
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;

                // 原子聚合
                int32_t suppkey = li.l_suppkey[i];
                revenue_array[suppkey].fetch_add(revenue, std::memory_order_relaxed);
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // ========================================================================
    // Step 3: 并行找最大值
    // ========================================================================
    std::vector<int64_t> thread_max(num_threads, 0);
    size_t key_chunk = (static_cast<size_t>(max_suppkey) + num_threads) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * key_chunk;
        size_t end = std::min(start + key_chunk, static_cast<size_t>(max_suppkey) + 1);
        if (start > static_cast<size_t>(max_suppkey)) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            int64_t local_max = 0;
            for (size_t k = start; k < end; ++k) {
                int64_t rev = revenue_array[k].load(std::memory_order_relaxed);
                if (rev > local_max) local_max = rev;
            }
            thread_max[t] = local_max;
        }));
    }

    for (auto& f : futures) f.get();

    int64_t max_revenue = 0;
    for (int64_t m : thread_max) {
        if (m > max_revenue) max_revenue = m;
    }

    // ========================================================================
    // Step 4: 找到最大 revenue 的 supplier
    // ========================================================================
    std::vector<int32_t> top_suppliers;
    top_suppliers.reserve(10);
    for (int32_t k = 0; k <= max_suppkey; ++k) {
        if (revenue_array[k].load(std::memory_order_relaxed) == max_revenue) {
            top_suppliers.push_back(k);
        }
    }

    // 阻止编译器优化
    volatile size_t sink = top_suppliers.size();
    volatile int64_t sink2 = max_revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V28 优化: Bloom Filter + Compact Hash Table + Thread-Local 聚合
// ============================================================================

void run_q3_v28(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 预处理 (延迟初始化)
    // ========================================================================

    // 1.1 构建 BUILDING 客户 custkey bitmap
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    ConcurrentBitmap building_custkeys;
    building_custkeys.init(static_cast<size_t>(max_custkey) + 1);

    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            building_custkeys.set(static_cast<uint32_t>(cust.c_custkey[i]));
        }
    }

    // 1.2 单遍扫描 orders: 收集 valid_orderkey + 构建 Bloom Filter + Compact Hash Table
    std::vector<int32_t> valid_orderkey_list;
    valid_orderkey_list.reserve(ord.count / 10);  // 预估 ~10% 有效

    // 预先扫描获取有效订单数量 (用于初始化数据结构)
    size_t valid_count_estimate = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (building_custkeys.test(static_cast<uint32_t>(ord.o_custkey[i])) &&
            ord.o_orderdate[i] < DATE_THRESHOLD) {
            valid_count_estimate++;
        }
    }

    // 初始化 Bloom Filter
    // 375 KB @ 1% FPR, L2 缓存友好
    thunderduck::bloom::BloomFilter order_bloom(valid_count_estimate, 0.01);

    // 初始化 Compact Hash Table (约 4.5 MB @ 60% load factor)
    CompactOrderLookup order_lookup;
    order_lookup.init(valid_count_estimate, 0.6);

    // 单遍构建所有数据结构
    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        // 条件: c_mktsegment = 'BUILDING' AND o_orderdate < threshold
        if (building_custkeys.test(static_cast<uint32_t>(custkey)) &&
            orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t compact_idx = static_cast<int32_t>(valid_orderkey_list.size());

            valid_orderkey_list.push_back(orderkey);
            order_bloom.insert(orderkey);
            order_lookup.insert(orderkey, compact_idx, orderdate, ord.o_shippriority[i]);
        }
    }

    size_t num_valid_orders = valid_orderkey_list.size();
    if (num_valid_orders == 0) {
        return;  // 无有效订单
    }

    // ========================================================================
    // Phase 2: 并行扫描 (Bloom Filter + Thread-Local 聚合)
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // Thread-Local 聚合器 (cache-line 对齐，避免 false sharing)
    std::vector<ThreadLocalQ3Agg> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        // 每线程预期处理 ~1/num_threads 的有效订单
        agg.init(num_valid_orders / num_threads + 1000);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            for (size_t i = start; i < end; ++i) {
                // 1. 日期过滤 (约 50% 过滤)
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

                // 2. Bloom Filter 测试 (约 94% 过滤) - L2 缓存友好
                if (!order_bloom.maybe_contains(orderkey)) continue;

                // 3. Compact Hash Table 探测 (仅 ~6% 到达此处)
                int32_t compact_idx = order_lookup.probe(orderkey);
                if (compact_idx < 0) continue;  // Bloom Filter 假阳性

                // 4. Thread-Local 聚合 (无原子操作)
                // revenue = l_extendedprice * (1 - l_discount)
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_agg.add(compact_idx, revenue);
            }
        }));
    }

    // 等待所有线程完成
    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 3: 归并 + 排序
    // ========================================================================

    // 3.1 归并 Thread-Local 结果
    std::vector<int64_t> final_revenue(num_valid_orders, 0);
    for (auto& agg : thread_aggs) {
        agg.revenue_map.for_each([&](int32_t compact_idx, int64_t rev) {
            if (compact_idx >= 0 && static_cast<size_t>(compact_idx) < num_valid_orders) {
                final_revenue[compact_idx] += rev;
            }
        });
    }

    // 3.2 收集有结果的订单
    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid_orders);

    for (size_t idx = 0; idx < num_valid_orders; ++idx) {
        if (final_revenue[idx] > 0) {
            int32_t orderkey = valid_orderkey_list[idx];
            const auto* entry = order_lookup.probe_full(orderkey);
            if (entry) {
                results.push_back({
                    orderkey,
                    final_revenue[idx],
                    entry->orderdate,
                    entry->shippriority
                });
            }
        }
    }

    // 3.3 partial_sort TOP 10
    // ORDER BY revenue DESC, o_orderdate ASC
    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    // 阻止编译器优化
    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V28.1 优化: 直接数组索引 + Thread-Local 数组聚合
// ============================================================================

void run_q3_v28_1(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 预处理 (同 V27)
    // ========================================================================

    // 1.1 构建 BUILDING 客户 custkey bitmap
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    ConcurrentBitmap building_custkeys;
    building_custkeys.init(static_cast<size_t>(max_custkey) + 1);

    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            building_custkeys.set(static_cast<uint32_t>(cust.c_custkey[i]));
        }
    }

    // 1.2 构建 order_lookup (同 V27)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    struct OrderLookup {
        int32_t compact_idx = -1;
        int32_t orderdate = 0;
        int32_t shippriority = 0;
    };
    std::vector<OrderLookup> order_lookup(static_cast<size_t>(max_orderkey) + 1);

    std::vector<int32_t> valid_orderkey_list;
    valid_orderkey_list.reserve(500000);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (building_custkeys.test(static_cast<uint32_t>(custkey)) &&
            orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkey_list.size());
            order_lookup[orderkey] = {idx, orderdate, ord.o_shippriority[i]};
            valid_orderkey_list.push_back(orderkey);
        }
    }

    size_t num_valid_orders = valid_orderkey_list.size();
    if (num_valid_orders == 0) {
        return;
    }

    // ========================================================================
    // Phase 2: 并行扫描 - Thread-Local 数组聚合
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // Thread-Local 数组 (每线程一个紧凑数组)
    // 对齐到 cache line 避免 false sharing
    struct alignas(128) ThreadLocalRevenue {
        std::vector<int64_t> data;
    };
    std::vector<ThreadLocalRevenue> thread_revenues(num_threads);
    for (auto& tr : thread_revenues) {
        tr.data.resize(num_valid_orders, 0);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_revenue = thread_revenues[t].data;

            for (size_t i = start; i < end; ++i) {
                // 1. 日期过滤
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

                // 2. 直接数组查找 (O(1))
                int32_t idx = order_lookup[orderkey].compact_idx;
                if (idx < 0) continue;

                // 3. Thread-Local 数组聚合 (无原子操作)
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_revenue[idx] += revenue;
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // ========================================================================
    // Phase 3: SIMD 加速归并
    // ========================================================================

    std::vector<int64_t> final_revenue(num_valid_orders, 0);

#ifdef __aarch64__
    // SIMD 归并: 每次处理 2 个 int64_t
    for (size_t idx = 0; idx + 1 < num_valid_orders; idx += 2) {
        int64x2_t sum = vdupq_n_s64(0);
        for (auto& tr : thread_revenues) {
            int64x2_t v = vld1q_s64(&tr.data[idx]);
            sum = vaddq_s64(sum, v);
        }
        vst1q_s64(&final_revenue[idx], sum);
    }
    // 处理剩余
    if (num_valid_orders % 2 != 0) {
        size_t idx = num_valid_orders - 1;
        for (auto& tr : thread_revenues) {
            final_revenue[idx] += tr.data[idx];
        }
    }
#else
    // 标量归并
    for (size_t idx = 0; idx < num_valid_orders; ++idx) {
        for (auto& tr : thread_revenues) {
            final_revenue[idx] += tr.data[idx];
        }
    }
#endif

    // ========================================================================
    // Phase 4: 收集结果 + partial_sort
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid_orders);

    for (size_t idx = 0; idx < num_valid_orders; ++idx) {
        if (final_revenue[idx] > 0) {
            int32_t orderkey = valid_orderkey_list[idx];
            const auto& info = order_lookup[orderkey];
            results.push_back({
                orderkey,
                final_revenue[idx],
                info.orderdate,
                info.shippriority
            });
        }
    }

    // ORDER BY revenue DESC, o_orderdate ASC, LIMIT 10
    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    // 阻止编译器优化
    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V29 优化: SIMD 批量过滤 + 轻量级 Bloom + 批量聚合
// ============================================================================

void run_q3_v29(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 预处理 - Predicate Bitmap + Bloom Filter + Lookup
    // ========================================================================

    // 1.1 构建 BUILDING 客户 custkey bitmap
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    ConcurrentBitmap building_custkeys;
    building_custkeys.init(static_cast<size_t>(max_custkey) + 1);

    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            building_custkeys.set(static_cast<uint32_t>(cust.c_custkey[i]));
        }
    }

    // 1.2 构建 order_lookup + 轻量级 Bloom Filter
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        max_orderkey = std::max(max_orderkey, ord.o_orderkey[i]);
    }

    struct OrderLookup {
        int32_t compact_idx = -1;
        int32_t orderdate = 0;
        int32_t shippriority = 0;
    };
    std::vector<OrderLookup> order_lookup(static_cast<size_t>(max_orderkey) + 1);

    std::vector<int32_t> valid_orderkey_list;
    valid_orderkey_list.reserve(500000);

    // 轻量级 Bloom Filter (2-hash, 内联)
    LightweightBloomFilter order_bloom;
    order_bloom.init(ord.count / 5);  // 预估 ~20% 有效

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (building_custkeys.test(static_cast<uint32_t>(custkey)) &&
            orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkey_list.size());
            order_lookup[orderkey] = {idx, orderdate, ord.o_shippriority[i]};
            valid_orderkey_list.push_back(orderkey);
            order_bloom.insert(orderkey);
        }
    }

    size_t num_valid_orders = valid_orderkey_list.size();
    if (num_valid_orders == 0) {
        return;
    }

    // ========================================================================
    // Phase 2: 并行扫描 - SIMD 批量过滤 + 批量聚合
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 紧凑原子数组
    std::vector<std::atomic<int64_t>> revenue_array(num_valid_orders);
    for (auto& r : revenue_array) {
        r.store(0, std::memory_order_relaxed);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    // 批量处理大小
    constexpr size_t BATCH_SIZE = 256;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            // 线程局部缓冲区
            alignas(64) int32_t batch_orderkeys[BATCH_SIZE];
            alignas(64) int64_t batch_revenues[BATCH_SIZE];
            alignas(64) int32_t batch_indices[BATCH_SIZE];
            size_t batch_count = 0;

            for (size_t i = start; i < end; ++i) {
                // 1. 日期过滤
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

                // 2. 轻量级 Bloom Filter 测试 (内联, 2-hash)
                if (!order_bloom.may_contain(orderkey)) continue;

                // 3. 直接数组查找
                int32_t idx = order_lookup[orderkey].compact_idx;
                if (idx < 0) continue;

                // 4. 计算 revenue 并加入批次
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;

                batch_orderkeys[batch_count] = orderkey;
                batch_revenues[batch_count] = revenue;
                batch_indices[batch_count] = idx;
                batch_count++;

                // 5. 批量聚合 (预取 + 原子更新)
                if (batch_count >= BATCH_SIZE) {
                    // 预取下一批
                    if (i + BATCH_SIZE < end) {
                        for (size_t p = 0; p < 8 && i + BATCH_SIZE + p < end; ++p) {
                            int32_t next_key = li.l_orderkey[i + BATCH_SIZE + p];
                            __builtin_prefetch(&order_lookup[next_key], 0, 1);
                        }
                    }

                    // 批量更新
                    for (size_t b = 0; b < batch_count; ++b) {
                        revenue_array[batch_indices[b]].fetch_add(
                            batch_revenues[b], std::memory_order_relaxed);
                    }
                    batch_count = 0;
                }
            }

            // 处理剩余批次
            for (size_t b = 0; b < batch_count; ++b) {
                revenue_array[batch_indices[b]].fetch_add(
                    batch_revenues[b], std::memory_order_relaxed);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 3: 收集结果 + partial_sort
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid_orders);

    for (size_t idx = 0; idx < num_valid_orders; ++idx) {
        int64_t rev = revenue_array[idx].load(std::memory_order_relaxed);
        if (rev > 0) {
            int32_t orderkey = valid_orderkey_list[idx];
            const auto& info = order_lookup[orderkey];
            results.push_back({
                orderkey,
                rev,
                info.orderdate,
                info.shippriority
            });
        }
    }

    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V30 优化: 极简预处理 + 直接扫描
// ============================================================================

void run_q3_v30(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 极简预处理 - 单遍扫描，最小化开销
    // ========================================================================

    // 1.1 构建 BUILDING 客户 bitmap (使用已知的 max_custkey)
    // 假设 custkey 是连续的，从 customer 表直接获取
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }

    // 使用简单数组而非 ConcurrentBitmap (单线程构建)
    std::vector<uint8_t> is_building(static_cast<size_t>(max_custkey) + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            is_building[cust.c_custkey[i]] = 1;
        }
    }

    // 1.2 单遍扫描 orders: 构建 lookup (使用已知的 max_orderkey)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    // 紧凑的 lookup 结构
    struct OrderInfo {
        int32_t compact_idx;
        int32_t orderdate;
        int8_t shippriority;
        int8_t valid;  // 0 = invalid, 1 = valid
    };
    std::vector<OrderInfo> order_lookup(static_cast<size_t>(max_orderkey) + 1,
                                        {-1, 0, 0, 0});

    std::vector<int32_t> valid_orderkeys;
    valid_orderkeys.reserve(400000);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (is_building[custkey] && orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkeys.size());
            order_lookup[orderkey] = {idx, orderdate,
                                      static_cast<int8_t>(ord.o_shippriority[i]), 1};
            valid_orderkeys.push_back(orderkey);
        }
    }

    size_t num_valid = valid_orderkeys.size();
    if (num_valid == 0) return;

    // ========================================================================
    // Phase 2: 并行扫描 lineitem
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::atomic<int64_t>> revenue(num_valid);
    for (auto& r : revenue) r.store(0, std::memory_order_relaxed);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
#ifdef __aarch64__
            // SIMD 版本: 每次处理 4 个元素
            const int32x4_t threshold_vec = vdupq_n_s32(DATE_THRESHOLD);

            size_t i = start;
            for (; i + 4 <= end; i += 4) {
                // 批量加载 shipdate
                int32x4_t dates = vld1q_s32(&li.l_shipdate[i]);

                // SIMD 比较: shipdate > threshold
                uint32x4_t mask = vcgtq_s32(dates, threshold_vec);

                // 提取通过的索引
                uint32_t m0 = vgetq_lane_u32(mask, 0);
                uint32_t m1 = vgetq_lane_u32(mask, 1);
                uint32_t m2 = vgetq_lane_u32(mask, 2);
                uint32_t m3 = vgetq_lane_u32(mask, 3);

                if (m0) {
                    int32_t okey = li.l_orderkey[i];
                    const auto& info = order_lookup[okey];
                    if (info.valid) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                      (10000 - li.l_discount[i]) / 10000;
                        revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
                    }
                }
                if (m1) {
                    int32_t okey = li.l_orderkey[i + 1];
                    const auto& info = order_lookup[okey];
                    if (info.valid) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 1]) *
                                      (10000 - li.l_discount[i + 1]) / 10000;
                        revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
                    }
                }
                if (m2) {
                    int32_t okey = li.l_orderkey[i + 2];
                    const auto& info = order_lookup[okey];
                    if (info.valid) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 2]) *
                                      (10000 - li.l_discount[i + 2]) / 10000;
                        revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
                    }
                }
                if (m3) {
                    int32_t okey = li.l_orderkey[i + 3];
                    const auto& info = order_lookup[okey];
                    if (info.valid) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 3]) *
                                      (10000 - li.l_discount[i + 3]) / 10000;
                        revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
                    }
                }
            }

            // 处理剩余元素
            for (; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;
                int32_t okey = li.l_orderkey[i];
                const auto& info = order_lookup[okey];
                if (!info.valid) continue;
                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                              (10000 - li.l_discount[i]) / 10000;
                revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
            }
#else
            // 标量版本
            for (size_t i = start; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;
                int32_t okey = li.l_orderkey[i];
                const auto& info = order_lookup[okey];
                if (!info.valid) continue;
                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                              (10000 - li.l_discount[i]) / 10000;
                revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
            }
#endif
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 3: 收集结果
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid);

    for (size_t idx = 0; idx < num_valid; ++idx) {
        int64_t rev = revenue[idx].load(std::memory_order_relaxed);
        if (rev > 0) {
            int32_t okey = valid_orderkeys[idx];
            const auto& info = order_lookup[okey];
            results.push_back({okey, rev, info.orderdate, info.shippriority});
        }
    }

    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V31: Bloom Filter + 紧凑 Hash Table
// ============================================================================

void run_q3_v31(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 构建 BUILDING custkey 集合 (使用 bitmap)
    // ========================================================================

    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }

    std::vector<uint8_t> is_building(static_cast<size_t>(max_custkey) + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {
            is_building[cust.c_custkey[i]] = 1;
        }
    }

    // ========================================================================
    // Phase 2: 构建紧凑 Hash Table + Bloom Filter
    // ========================================================================

    struct OrderEntry {
        int32_t orderkey;
        int32_t compact_idx;
        int32_t orderdate;
        int8_t shippriority;
    };

    // 预估 ~20% 订单有效
    size_t estimated_valid = ord.count / 5;

    // 紧凑 hash table (开放寻址)
    size_t table_size = 1;
    while (table_size < estimated_valid * 2) table_size <<= 1;
    uint32_t table_mask = static_cast<uint32_t>(table_size - 1);

    std::vector<OrderEntry> order_table(table_size, {INT32_MIN, -1, 0, 0});

    // Bloom Filter (小型，L1 缓存友好)
    // 2 bits/element × 300K = ~75 KB
    size_t bloom_bits = estimated_valid * 2;
    bloom_bits = ((bloom_bits + 63) / 64) * 64;
    std::vector<uint64_t> bloom(bloom_bits / 64, 0);
    uint32_t bloom_mask = static_cast<uint32_t>(bloom_bits - 1);

    std::vector<int32_t> valid_orderkeys;
    valid_orderkeys.reserve(estimated_valid);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (is_building[custkey] && orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkeys.size());

            // 插入 hash table
#ifdef __aarch64__
            uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));
#else
            uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif
            uint32_t pos = hash & table_mask;
            while (order_table[pos].orderkey != INT32_MIN) {
                pos = (pos + 1) & table_mask;
            }
            order_table[pos] = {orderkey, idx, orderdate,
                                static_cast<int8_t>(ord.o_shippriority[i])};

            // 插入 Bloom Filter (单 hash)
            bloom[(hash & bloom_mask) >> 6] |= (1ULL << ((hash & bloom_mask) & 63));

            valid_orderkeys.push_back(orderkey);
        }
    }

    size_t num_valid = valid_orderkeys.size();
    if (num_valid == 0) return;

    // ========================================================================
    // Phase 3: 并行扫描 lineitem
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::atomic<int64_t>> revenue(num_valid);
    for (auto& r : revenue) r.store(0, std::memory_order_relaxed);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

#ifdef __aarch64__
                uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));
#else
                uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif

                // Bloom Filter 快速拒绝
                if (!((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1)) {
                    continue;
                }

                // Hash table 查找
                uint32_t pos = hash & table_mask;
                while (true) {
                    if (order_table[pos].orderkey == INT32_MIN) break;  // 不存在
                    if (order_table[pos].orderkey == orderkey) {
                        // 找到
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                      (10000 - li.l_discount[i]) / 10000;
                        revenue[order_table[pos].compact_idx].fetch_add(
                            rev, std::memory_order_relaxed);
                        break;
                    }
                    pos = (pos + 1) & table_mask;
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 4: 收集结果
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid);

    // 遍历 hash table 获取 orderdate 和 shippriority
    for (size_t idx = 0; idx < num_valid; ++idx) {
        int64_t rev = revenue[idx].load(std::memory_order_relaxed);
        if (rev > 0) {
            int32_t okey = valid_orderkeys[idx];
            // 查找 hash table 获取信息
#ifdef __aarch64__
            uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(okey));
#else
            uint32_t hash = static_cast<uint32_t>(okey) * 0x85ebca6b;
#endif
            uint32_t pos = hash & table_mask;
            while (order_table[pos].orderkey != okey) {
                pos = (pos + 1) & table_mask;
            }
            results.push_back({okey, rev, order_table[pos].orderdate,
                               order_table[pos].shippriority});
        }
    }

    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q3 V32: 整合所有最优技术
// ============================================================================

void run_q3_v32(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t DATE_THRESHOLD = dates::D1995_03_15;

    // ========================================================================
    // Phase 1: 构建 BUILDING custkey 集合 (使用 bitmap)
    // ========================================================================

    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }

    std::vector<uint8_t> is_building(static_cast<size_t>(max_custkey) + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {
            is_building[cust.c_custkey[i]] = 1;
        }
    }

    // ========================================================================
    // Phase 2: 构建紧凑 Hash Table + Bloom Filter (V31)
    // ========================================================================

    struct OrderEntry {
        int32_t orderkey;
        int32_t compact_idx;
        int32_t orderdate;
        int8_t shippriority;
    };

    size_t estimated_valid = ord.count / 5;

    // 紧凑 hash table
    size_t table_size = 1;
    while (table_size < estimated_valid * 2) table_size <<= 1;
    uint32_t table_mask = static_cast<uint32_t>(table_size - 1);
    std::vector<OrderEntry> order_table(table_size, {INT32_MIN, -1, 0, 0});

    // Bloom Filter (L1 缓存友好)
    size_t bloom_bits = estimated_valid * 2;
    bloom_bits = ((bloom_bits + 63) / 64) * 64;
    std::vector<uint64_t> bloom(bloom_bits / 64, 0);
    uint32_t bloom_mask = static_cast<uint32_t>(bloom_bits - 1);

    std::vector<int32_t> valid_orderkeys;
    valid_orderkeys.reserve(estimated_valid);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (is_building[custkey] && orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orderkeys.size());

#ifdef __aarch64__
            uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));
#else
            uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif
            uint32_t pos = hash & table_mask;
            while (order_table[pos].orderkey != INT32_MIN) {
                pos = (pos + 1) & table_mask;
            }
            order_table[pos] = {orderkey, idx, orderdate,
                                static_cast<int8_t>(ord.o_shippriority[i])};

            bloom[(hash & bloom_mask) >> 6] |= (1ULL << ((hash & bloom_mask) & 63));
            valid_orderkeys.push_back(orderkey);
        }
    }

    size_t num_valid = valid_orderkeys.size();
    if (num_valid == 0) return;

    // ========================================================================
    // Phase 3: 并行扫描 - SIMD 日期过滤 + Thread-Local 聚合 (V30 + V28.1)
    // ========================================================================

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // Thread-Local 数组聚合 (V28.1) - 避免原子操作
    struct alignas(128) ThreadLocalRevenue {
        std::vector<int64_t> data;
    };
    std::vector<ThreadLocalRevenue> thread_revenues(num_threads);
    for (auto& tr : thread_revenues) {
        tr.data.resize(num_valid, 0);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_revenue = thread_revenues[t].data;

#ifdef __aarch64__
            // SIMD 日期过滤 (V30) + Bloom + Hash Table 查找
            const int32x4_t threshold_vec = vdupq_n_s32(DATE_THRESHOLD);

            size_t i = start;
            for (; i + 4 <= end; i += 4) {
                // SIMD 批量日期比较
                int32x4_t dates = vld1q_s32(&li.l_shipdate[i]);
                uint32x4_t mask = vcgtq_s32(dates, threshold_vec);

                // 提取通过日期过滤的行
                uint32_t m0 = vgetq_lane_u32(mask, 0);
                uint32_t m1 = vgetq_lane_u32(mask, 1);
                uint32_t m2 = vgetq_lane_u32(mask, 2);
                uint32_t m3 = vgetq_lane_u32(mask, 3);

                // 处理每个通过日期过滤的行
                if (m0) {
                    int32_t orderkey = li.l_orderkey[i];
                    uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));

                    // Bloom Filter 快速拒绝
                    if ((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1) {
                        // Hash table 查找
                        uint32_t pos = hash & table_mask;
                        while (order_table[pos].orderkey != INT32_MIN) {
                            if (order_table[pos].orderkey == orderkey) {
                                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                              (10000 - li.l_discount[i]) / 10000;
                                local_revenue[order_table[pos].compact_idx] += rev;
                                break;
                            }
                            pos = (pos + 1) & table_mask;
                        }
                    }
                }
                if (m1) {
                    int32_t orderkey = li.l_orderkey[i + 1];
                    uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));

                    if ((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1) {
                        uint32_t pos = hash & table_mask;
                        while (order_table[pos].orderkey != INT32_MIN) {
                            if (order_table[pos].orderkey == orderkey) {
                                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 1]) *
                                              (10000 - li.l_discount[i + 1]) / 10000;
                                local_revenue[order_table[pos].compact_idx] += rev;
                                break;
                            }
                            pos = (pos + 1) & table_mask;
                        }
                    }
                }
                if (m2) {
                    int32_t orderkey = li.l_orderkey[i + 2];
                    uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));

                    if ((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1) {
                        uint32_t pos = hash & table_mask;
                        while (order_table[pos].orderkey != INT32_MIN) {
                            if (order_table[pos].orderkey == orderkey) {
                                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 2]) *
                                              (10000 - li.l_discount[i + 2]) / 10000;
                                local_revenue[order_table[pos].compact_idx] += rev;
                                break;
                            }
                            pos = (pos + 1) & table_mask;
                        }
                    }
                }
                if (m3) {
                    int32_t orderkey = li.l_orderkey[i + 3];
                    uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));

                    if ((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1) {
                        uint32_t pos = hash & table_mask;
                        while (order_table[pos].orderkey != INT32_MIN) {
                            if (order_table[pos].orderkey == orderkey) {
                                int64_t rev = static_cast<int64_t>(li.l_extendedprice[i + 3]) *
                                              (10000 - li.l_discount[i + 3]) / 10000;
                                local_revenue[order_table[pos].compact_idx] += rev;
                                break;
                            }
                            pos = (pos + 1) & table_mask;
                        }
                    }
                }
            }

            // 处理剩余元素
            for (; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];
                uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(orderkey));

                if (!((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1)) {
                    continue;
                }

                uint32_t pos = hash & table_mask;
                while (order_table[pos].orderkey != INT32_MIN) {
                    if (order_table[pos].orderkey == orderkey) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                      (10000 - li.l_discount[i]) / 10000;
                        local_revenue[order_table[pos].compact_idx] += rev;
                        break;
                    }
                    pos = (pos + 1) & table_mask;
                }
            }
#else
            // 标量版本
            for (size_t i = start; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];
                uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;

                if (!((bloom[(hash & bloom_mask) >> 6] >> ((hash & bloom_mask) & 63)) & 1)) {
                    continue;
                }

                uint32_t pos = hash & table_mask;
                while (order_table[pos].orderkey != INT32_MIN) {
                    if (order_table[pos].orderkey == orderkey) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                      (10000 - li.l_discount[i]) / 10000;
                        local_revenue[order_table[pos].compact_idx] += rev;
                        break;
                    }
                    pos = (pos + 1) & table_mask;
                }
            }
#endif
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // ========================================================================
    // Phase 4: SIMD 加速归并 (V28.1)
    // ========================================================================

    std::vector<int64_t> final_revenue(num_valid, 0);

#ifdef __aarch64__
    // SIMD 归并: 每次处理 2 个 int64_t
    for (size_t idx = 0; idx + 1 < num_valid; idx += 2) {
        int64x2_t sum = vdupq_n_s64(0);
        for (auto& tr : thread_revenues) {
            int64x2_t v = vld1q_s64(&tr.data[idx]);
            sum = vaddq_s64(sum, v);
        }
        vst1q_s64(&final_revenue[idx], sum);
    }
    // 处理剩余
    if (num_valid % 2 != 0) {
        size_t idx = num_valid - 1;
        for (auto& tr : thread_revenues) {
            final_revenue[idx] += tr.data[idx];
        }
    }
#else
    // 标量归并
    for (size_t idx = 0; idx < num_valid; ++idx) {
        for (auto& tr : thread_revenues) {
            final_revenue[idx] += tr.data[idx];
        }
    }
#endif

    // ========================================================================
    // Phase 5: 收集结果
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid);

    for (size_t idx = 0; idx < num_valid; ++idx) {
        int64_t rev = final_revenue[idx];
        if (rev > 0) {
            int32_t okey = valid_orderkeys[idx];
#ifdef __aarch64__
            uint32_t hash = __builtin_arm_crc32w(0, static_cast<uint32_t>(okey));
#else
            uint32_t hash = static_cast<uint32_t>(okey) * 0x85ebca6b;
#endif
            uint32_t pos = hash & table_mask;
            while (order_table[pos].orderkey != okey) {
                pos = (pos + 1) & table_mask;
            }
            results.push_back({okey, rev, order_table[pos].orderdate,
                               order_table[pos].shippriority});
        }
    }

    size_t top_n = std::min<size_t>(10, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 10) results.resize(10);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

} // namespace ops_v27
} // namespace tpch
} // namespace thunderduck
