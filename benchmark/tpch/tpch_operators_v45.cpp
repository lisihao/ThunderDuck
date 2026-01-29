/**
 * ThunderDuck TPC-H V45 优化算子实现
 *
 * 直接数组优化 - 消除 Hash 探测开销
 *
 * @version 45.0
 * @date 2026-01-29
 */

#include "tpch_operators_v45.h"
#include "tpch_operators_v25.h"  // ThreadPool
#include "tpch_operators_v32.h"  // CompactHashTable

#include <algorithm>
#include <vector>
#include <thread>
#include <future>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v45 {

// ============================================================================
// Q14 V45: 直接数组优化
//
// 优化对比 V25:
// - V25: WeakHashTable (part_promo) + KeyHashCache + find_with_hash()
// - V45: 直接数组 part_is_promo[partkey] → O(1) 无探测
//
// 内存分析:
// - Part 表 ~200K 行, max_partkey ~200K
// - 直接数组: 200KB (uint8_t) → 完全装入 L2 cache (4MB on M4)
// ============================================================================

void run_q14_v45(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    constexpr int32_t date_lo = 9374;  // 1995-09-01
    constexpr int32_t date_hi = 9404;  // 1995-10-01

    // Phase 1: 构建直接数组 part_is_promo[partkey]
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        max_partkey = std::max(max_partkey, part.p_partkey[i]);
    }

    // 0 = unknown, 1 = promo, 2 = not promo
    std::vector<uint8_t> part_is_promo(max_partkey + 1, 0);

    for (size_t i = 0; i < part.count; ++i) {
        int32_t pkey = part.p_partkey[i];
        // PROMO% 检查
        bool is_promo = (part.p_type[i].size() >= 5 &&
                         part.p_type[i].compare(0, 5, "PROMO") == 0);
        part_is_promo[pkey] = is_promo ? 1 : 2;
    }

    // Phase 2: 并行扫描 lineitem
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 线程局部聚合
    std::vector<__int128> thread_promo(num_threads, 0);
    std::vector<__int128> thread_total(num_threads, 0);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    const int32_t* l_shipdate = li.l_shipdate.data();
    const int32_t* l_partkey = li.l_partkey.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();
    const uint8_t* promo_arr = part_is_promo.data();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([=, &thread_promo, &thread_total]() {
            __int128 local_promo = 0;
            __int128 local_total = 0;

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(l_shipdate + i + 64, 0, 3);
                __builtin_prefetch(l_partkey + i + 64, 0, 3);

                #define PROCESS_ONE(idx) do { \
                    int32_t shipdate = l_shipdate[i + idx]; \
                    if (shipdate >= date_lo && shipdate < date_hi) { \
                        int32_t pkey = l_partkey[i + idx]; \
                        if (pkey >= 0 && pkey <= max_partkey && promo_arr[pkey] != 0) { \
                            __int128 val = (__int128)l_extendedprice[i + idx] * \
                                           (10000 - l_discount[i + idx]) / 10000; \
                            local_total += val; \
                            if (promo_arr[pkey] == 1) local_promo += val; \
                        } \
                    } \
                } while(0)

                PROCESS_ONE(0); PROCESS_ONE(1);
                PROCESS_ONE(2); PROCESS_ONE(3);
                PROCESS_ONE(4); PROCESS_ONE(5);
                PROCESS_ONE(6); PROCESS_ONE(7);

                #undef PROCESS_ONE
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t shipdate = l_shipdate[i];
                if (shipdate < date_lo || shipdate >= date_hi) continue;

                int32_t pkey = l_partkey[i];
                if (pkey < 0 || pkey > max_partkey) continue;

                uint8_t promo_flag = promo_arr[pkey];
                if (promo_flag == 0) continue;  // 未知 part

                __int128 val = (__int128)l_extendedprice[i] *
                               (10000 - l_discount[i]) / 10000;
                local_total += val;
                if (promo_flag == 1) local_promo += val;
            }

            thread_promo[t] = local_promo;
            thread_total[t] = local_total;
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    __int128 sum_promo = 0, sum_total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        sum_promo += thread_promo[t];
        sum_total += thread_total[t];
    }

    volatile double result = 100.0 * static_cast<double>(sum_promo) /
                             static_cast<double>(sum_total);
    (void)result;
}

// ============================================================================
// Q11 V45: 位图 + 直接数组聚合
//
// 优化对比 V27:
// - V27: unordered_set (germany_suppliers) + MutableWeakHashTable (聚合)
// - V45: 位图 (germany_bitmap) + 直接数组 (partkey_value)
//
// 内存分析:
// - Supplier ~10K, max_suppkey ~10K → 位图 1.25KB
// - Part ~200K, max_partkey ~200K → 直接数组 1.6MB
// ============================================================================

void run_q11_v45(TPCHDataLoader& loader) {
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // Step 1: 找到 GERMANY nationkey
    int32_t germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == "GERMANY") {
            germany_key = nat.n_nationkey[i];
            break;
        }
    }

    // Step 2: 构建 germany_suppliers 位图
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        max_suppkey = std::max(max_suppkey, supp.s_suppkey[i]);
    }

    // 位图: germany_bitmap[suppkey/8] 的第 (suppkey%8) 位
    std::vector<uint8_t> germany_bitmap((max_suppkey + 8) / 8, 0);
    size_t germany_count = 0;

    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_nationkey[i] == germany_key) {
            int32_t sk = supp.s_suppkey[i];
            germany_bitmap[sk >> 3] |= (1u << (sk & 7));
            germany_count++;
        }
    }

    // Step 3: 找到最大 partkey
    int32_t max_partkey = 0;
    for (size_t i = 0; i < partsupp.count; ++i) {
        max_partkey = std::max(max_partkey, partsupp.ps_partkey[i]);
    }

    // Step 4: 并行扫描 - 直接数组聚合
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, partsupp.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (partsupp.count + num_threads - 1) / num_threads;

    // 线程局部: 直接数组 partkey_value + total
    struct ThreadLocalData {
        std::vector<int64_t> partkey_value;
        int64_t total = 0;
    };
    std::vector<ThreadLocalData> thread_data(num_threads);
    for (auto& td : thread_data) {
        td.partkey_value.resize(max_partkey + 1, 0);
    }

    const uint8_t* bitmap = germany_bitmap.data();
    const int32_t* ps_suppkey = partsupp.ps_suppkey.data();
    const int32_t* ps_partkey = partsupp.ps_partkey.data();
    const int64_t* ps_supplycost = partsupp.ps_supplycost.data();
    const int32_t* ps_availqty = partsupp.ps_availqty.data();

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, partsupp.count);
        if (start >= partsupp.count) break;

        futures.push_back(pool.submit([=, &thread_data]() {
            auto& local = thread_data[t];
            int64_t* pv = local.partkey_value.data();
            int64_t local_total = 0;

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(ps_suppkey + i + 64, 0, 3);
                __builtin_prefetch(ps_partkey + i + 64, 0, 3);

                #define CHECK_ADD(idx) do { \
                    int32_t sk = ps_suppkey[i + idx]; \
                    if (sk >= 0 && sk <= max_suppkey && \
                        (bitmap[sk >> 3] & (1u << (sk & 7)))) { \
                        int64_t val = ps_supplycost[i + idx] * \
                                      ps_availqty[i + idx] / 10000; \
                        pv[ps_partkey[i + idx]] += val; \
                        local_total += val; \
                    } \
                } while(0)

                CHECK_ADD(0); CHECK_ADD(1);
                CHECK_ADD(2); CHECK_ADD(3);
                CHECK_ADD(4); CHECK_ADD(5);
                CHECK_ADD(6); CHECK_ADD(7);

                #undef CHECK_ADD
            }

            for (; i < end; ++i) {
                int32_t sk = ps_suppkey[i];
                if (sk < 0 || sk > max_suppkey) continue;
                if (!(bitmap[sk >> 3] & (1u << (sk & 7)))) continue;

                int64_t val = ps_supplycost[i] * ps_availqty[i] / 10000;
                pv[ps_partkey[i]] += val;
                local_total += val;
            }

            local.total = local_total;
        }));
    }

    for (auto& f : futures) f.get();

    // Step 5: 合并 total
    int64_t total_value = 0;
    for (const auto& td : thread_data) {
        total_value += td.total;
    }

    // Step 6: 合并 partkey_value 数组
    std::vector<int64_t> merged_value(max_partkey + 1, 0);
    for (const auto& td : thread_data) {
        for (int32_t pk = 0; pk <= max_partkey; ++pk) {
            merged_value[pk] += td.partkey_value[pk];
        }
    }

    // Step 7: 后置过滤 (threshold = total * 0.0001)
    int64_t threshold = total_value / 10000;

    std::vector<std::pair<int32_t, int64_t>> results;
    results.reserve(1000);

    for (int32_t pk = 0; pk <= max_partkey; ++pk) {
        if (merged_value[pk] > threshold) {
            results.emplace_back(pk, merged_value[pk]);
        }
    }

    // Step 8: 排序
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].second;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q5 V45: 直接数组维度表
//
// 优化对比 V32:
// - V32: CompactHashTable (supp_to_nation, cust_to_nation, order_to_cust)
// - V45: 直接数组 (supp_nation, cust_nation) + CompactHashTable (order_to_cust)
//
// 内存分析:
// - Supplier ~10K → supp_nation[suppkey] 40KB
// - Customer ~150K → cust_nation[custkey] 600KB
// - Order orderkey 范围 ~6M → 保持 Hash (直接数组 24MB 太大)
// ============================================================================

void run_q5_v45(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    constexpr int32_t date_lo = 8766;   // 1994-01-01
    constexpr int32_t date_hi = 9131;   // 1995-01-01

    // Phase 1: 找到 ASIA region 的 nations
    int32_t asia_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == "ASIA") {
            asia_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    // 收集 ASIA nations (最多 25 个 nation)
    std::vector<uint8_t> is_asia_nation(25, 0);
    for (size_t j = 0; j < nat.count; ++j) {
        if (nat.n_regionkey[j] == asia_regionkey) {
            is_asia_nation[nat.n_nationkey[j]] = 1;
        }
    }

    // Phase 2: 构建 supp_nation 直接数组
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        max_suppkey = std::max(max_suppkey, supp.s_suppkey[i]);
    }

    // -1 = not ASIA, 0-24 = nationkey
    std::vector<int8_t> supp_nation(max_suppkey + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        int32_t nkey = supp.s_nationkey[i];
        if (nkey >= 0 && nkey < 25 && is_asia_nation[nkey]) {
            supp_nation[supp.s_suppkey[i]] = static_cast<int8_t>(nkey);
        }
    }

    // Phase 3: 构建 cust_nation 直接数组
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        max_custkey = std::max(max_custkey, cust.c_custkey[i]);
    }

    std::vector<int8_t> cust_nation(max_custkey + 1, -1);
    for (size_t i = 0; i < cust.count; ++i) {
        int32_t nkey = cust.c_nationkey[i];
        if (nkey >= 0 && nkey < 25 && is_asia_nation[nkey]) {
            cust_nation[cust.c_custkey[i]] = static_cast<int8_t>(nkey);
        }
    }

    // Phase 4: 构建 order_to_cust (保持 CompactHashTable，orderkey 范围太大)
    ops_v32::CompactHashTable<int32_t> order_to_cust;
    order_to_cust.init(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t ck = ord.o_custkey[i];
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            if (ck >= 0 && ck <= max_custkey && cust_nation[ck] >= 0) {
                order_to_cust.insert(ord.o_orderkey[i], ck);
            }
        }
    }

    // Phase 5: 并行扫描 lineitem
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 线程局部 nation 收入
    std::vector<std::array<int64_t, 25>> thread_revenues(num_threads);
    for (auto& arr : thread_revenues) arr.fill(0);

    const int8_t* sn_arr = supp_nation.data();
    const int8_t* cn_arr = cust_nation.data();

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([=, &thread_revenues, &order_to_cust]() {
            auto& local_rev = thread_revenues[t];

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&li.l_suppkey[i + 64], 0, 3);
                __builtin_prefetch(&li.l_orderkey[i + 64], 0, 3);

                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;

                    // 直接数组查找 supplier nation
                    int32_t sk = li.l_suppkey[idx];
                    if (sk < 0 || sk > max_suppkey) continue;
                    int8_t s_nat = sn_arr[sk];
                    if (s_nat < 0) continue;

                    // Hash 查找 orderkey → custkey
                    const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[idx]);
                    if (!cust_ptr) continue;
                    int32_t ck = *cust_ptr;

                    // 直接数组查找 customer nation
                    if (ck < 0 || ck > max_custkey) continue;
                    int8_t c_nat = cn_arr[ck];
                    if (c_nat < 0) continue;

                    // 检查 nation 匹配
                    if (s_nat != c_nat) continue;

                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[idx]) *
                                      (10000 - li.l_discount[idx]) / 10000;
                    local_rev[s_nat] += revenue;
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t sk = li.l_suppkey[i];
                if (sk < 0 || sk > max_suppkey) continue;
                int8_t s_nat = sn_arr[sk];
                if (s_nat < 0) continue;

                const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[i]);
                if (!cust_ptr) continue;
                int32_t ck = *cust_ptr;

                if (ck < 0 || ck > max_custkey) continue;
                int8_t c_nat = cn_arr[ck];
                if (c_nat < 0 || s_nat != c_nat) continue;

                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_rev[s_nat] += revenue;
            }
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    std::array<int64_t, 25> nation_revenue = {};
    for (const auto& local : thread_revenues) {
        for (size_t n = 0; n < 25; ++n) {
            nation_revenue[n] += local[n];
        }
    }

    volatile int64_t sink = 0;
    for (int64_t r : nation_revenue) sink += r;
    (void)sink;
}

}  // namespace ops_v45
}  // namespace tpch
}  // namespace thunderduck
