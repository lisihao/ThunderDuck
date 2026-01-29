/**
 * ThunderDuck TPC-H 算子实现 V32
 *
 * 自适应策略 + 批量哈希 SIMD 优化:
 * - SF < 3.5: 使用 V27 直接数组 (小数据，内存充足)
 * - SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化 (大数据，节省内存)
 * - Q3 保持 V31 (Bloom Filter 收益明显)
 *
 * @version 32.1
 * @date 2026-01-28
 */

#include "tpch_operators_v32.h"
#include "tpch_operators_v27.h"
#include "tpch_operators_v25.h"  // ThreadPool

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <future>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v32 {

// ============================================================================
// 日期常量 (epoch days from 1970-01-01)
// ============================================================================

namespace dates {
    constexpr int32_t DATE_1994_01_01 = 8766;
    constexpr int32_t DATE_1995_01_01 = 9131;
    constexpr int32_t DATE_1996_01_01 = 9496;
    constexpr int32_t DATE_1996_12_31 = 9861;
}

// ============================================================================
// Q5 自适应版本
// ============================================================================

void run_q5_adaptive(TPCHDataLoader& loader) {
    // 自适应策略: 根据数据量选择实现
    if (!adaptive::use_compact_structure(loader.orders().count)) {
        // SF < 3.5: 使用 V25 优化 (直接数组 + 线程池)
        run_q5_v25(loader);
    } else {
        // SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
        run_q5_v32_batch(loader);
    }
}

void run_q5_v32_batch(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    constexpr int32_t date_lo = dates::DATE_1994_01_01;
    constexpr int32_t date_hi = dates::DATE_1995_01_01;

    // Phase 1: 找到 ASIA region 的 nations
    int32_t asia_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == "ASIA") {
            asia_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    std::unordered_set<int32_t> asia_nation_set;
    for (size_t j = 0; j < nat.count; ++j) {
        if (nat.n_regionkey[j] == asia_regionkey) {
            asia_nation_set.insert(nat.n_nationkey[j]);
        }
    }

    // Phase 2: 构建紧凑 Hash Table (无 Bloom Filter，减少开销)
    CompactHashTable<int32_t> supp_to_nation;
    supp_to_nation.init(supp.count / 5);
    for (size_t i = 0; i < supp.count; ++i) {
        if (asia_nation_set.count(supp.s_nationkey[i])) {
            supp_to_nation.insert(supp.s_suppkey[i], supp.s_nationkey[i]);
        }
    }

    CompactHashTable<int32_t> cust_to_nation;
    cust_to_nation.init(cust.count / 5);
    for (size_t i = 0; i < cust.count; ++i) {
        if (asia_nation_set.count(cust.c_nationkey[i])) {
            cust_to_nation.insert(cust.c_custkey[i], cust.c_nationkey[i]);
        }
    }

    // Phase 3: 构建 orderkey → custkey (仅 ASIA 客户 + 日期范围)
    CompactHashTable<int32_t> order_to_cust;
    order_to_cust.init(ord.count / 4);
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            if (cust_to_nation.find(ord.o_custkey[i])) {
                order_to_cust.insert(ord.o_orderkey[i], ord.o_custkey[i]);
            }
        }
    }

    // Phase 4: 并行扫描 lineitem - 批量查找优化
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<ThreadLocalAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) agg.init(25);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            // 批量处理
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                alignas(32) int32_t orderkeys[8];
                alignas(32) int32_t suppkeys[8];
                for (int j = 0; j < 8; ++j) {
                    orderkeys[j] = li.l_orderkey[i + j];
                    suppkeys[j] = li.l_suppkey[i + j];
                }

                // 批量查找 orderkey → custkey
                const int32_t* cust_results[8];
                order_to_cust.batch_find(orderkeys, cust_results);

                // 批量查找 suppkey → nationkey
                const int32_t* supp_results[8];
                supp_to_nation.batch_find(suppkeys, supp_results);

                // 处理每个结果
                for (int j = 0; j < 8; ++j) {
                    if (!cust_results[j] || !supp_results[j]) continue;

                    int32_t custkey = *cust_results[j];
                    int32_t supp_nat = *supp_results[j];

                    const int32_t* cust_nat_ptr = cust_to_nation.find(custkey);
                    if (!cust_nat_ptr) continue;
                    int32_t cust_nat = *cust_nat_ptr;

                    if (cust_nat != supp_nat) continue;

                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i + j]) *
                                      (10000 - li.l_discount[i + j]) / 10000;
                    local_agg.add(cust_nat, revenue);
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[i]);
                if (!cust_ptr) continue;

                const int32_t* supp_nat_ptr = supp_to_nation.find(li.l_suppkey[i]);
                if (!supp_nat_ptr) continue;

                const int32_t* cust_nat_ptr = cust_to_nation.find(*cust_ptr);
                if (!cust_nat_ptr) continue;

                if (*cust_nat_ptr != *supp_nat_ptr) continue;

                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_agg.add(*cust_nat_ptr, revenue);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    CompactHashTable<int64_t> nation_revenue;
    nation_revenue.init(25);
    for (auto& agg : thread_aggs) {
        agg.table.for_each([&](int32_t nat, int64_t rev) {
            nation_revenue.add_or_update(nat, rev);
        });
    }

    volatile int64_t sink = 0;
    nation_revenue.for_each([&sink](int32_t, int64_t rev) { sink += rev; });
    (void)sink;
}

// ============================================================================
// Q7 自适应版本
// ============================================================================

void run_q7_adaptive(TPCHDataLoader& loader) {
    if (!adaptive::use_compact_structure(loader.orders().count)) {
        // SF < 3.5: 使用 V27 直接数组索引
        run_q7_v27(loader);
    } else {
        // SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
        run_q7_v32_batch(loader);
    }
}

void run_q7_v32_batch(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    constexpr int32_t date_lo = dates::DATE_1995_01_01;
    constexpr int32_t date_hi = dates::DATE_1996_12_31;

    // Phase 1: 找到 FRANCE 和 GERMANY
    int32_t france_key = -1, germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == "FRANCE") france_key = nat.n_nationkey[i];
        if (nat.n_name[i] == "GERMANY") germany_key = nat.n_nationkey[i];
    }

    // Phase 2: 构建紧凑 Hash Table (无 Bloom Filter)
    CompactHashTable<int8_t> supp_nation;
    supp_nation.init(supp.count / 12);
    for (size_t i = 0; i < supp.count; ++i) {
        int32_t nkey = supp.s_nationkey[i];
        if (nkey == france_key) supp_nation.insert(supp.s_suppkey[i], 0);
        else if (nkey == germany_key) supp_nation.insert(supp.s_suppkey[i], 1);
    }

    CompactHashTable<int8_t> cust_nation;
    cust_nation.init(cust.count / 12);
    for (size_t i = 0; i < cust.count; ++i) {
        int32_t nkey = cust.c_nationkey[i];
        if (nkey == france_key) cust_nation.insert(cust.c_custkey[i], 0);
        else if (nkey == germany_key) cust_nation.insert(cust.c_custkey[i], 1);
    }

    CompactHashTable<int32_t> order_to_cust;
    order_to_cust.init(ord.count / 6);
    for (size_t i = 0; i < ord.count; ++i) {
        if (cust_nation.find(ord.o_custkey[i])) {
            order_to_cust.insert(ord.o_orderkey[i], ord.o_custkey[i]);
        }
    }

    // Phase 3: 并行扫描 lineitem - 批量优化
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    struct Q7Agg {
        std::array<std::array<std::array<int64_t, 2>, 2>, 2> data{};
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

            // 批量处理
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                // 批量日期过滤
#ifdef __aarch64__
                int32x4_t dates_lo = vld1q_s32(&li.l_shipdate[i]);
                int32x4_t dates_hi = vld1q_s32(&li.l_shipdate[i + 4]);
                int32x4_t lo_vec = vdupq_n_s32(date_lo);
                int32x4_t hi_vec = vdupq_n_s32(date_hi);

                uint32x4_t mask_lo1 = vcgeq_s32(dates_lo, lo_vec);
                uint32x4_t mask_hi1 = vcleq_s32(dates_lo, hi_vec);
                uint32x4_t mask1 = vandq_u32(mask_lo1, mask_hi1);

                uint32x4_t mask_lo2 = vcgeq_s32(dates_hi, lo_vec);
                uint32x4_t mask_hi2 = vcleq_s32(dates_hi, hi_vec);
                uint32x4_t mask2 = vandq_u32(mask_lo2, mask_hi2);

                // 提取掩码到数组用于循环
                alignas(16) uint32_t mask_arr1[4], mask_arr2[4];
                vst1q_u32(mask_arr1, mask1);
                vst1q_u32(mask_arr2, mask2);

                // 处理前 4 个
                for (int j = 0; j < 4; ++j) {
                    if (mask_arr1[j] == 0) continue;
                    size_t idx = i + j;

                    const int8_t* s_nat = supp_nation.find(li.l_suppkey[idx]);
                    if (!s_nat) continue;

                    const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[idx]);
                    if (!cust_ptr) continue;

                    const int8_t* c_nat = cust_nation.find(*cust_ptr);
                    if (!c_nat || *s_nat == *c_nat) continue;

                    int year_idx = (li.l_shipdate[idx] >= dates::DATE_1996_01_01) ? 1 : 0;
                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[idx]) *
                                      (10000 - li.l_discount[idx]) / 10000;
                    local_agg.data[*s_nat][*c_nat][year_idx] += revenue;
                }

                // 处理后 4 个
                for (int j = 0; j < 4; ++j) {
                    if (mask_arr2[j] == 0) continue;
                    size_t idx = i + 4 + j;

                    const int8_t* s_nat = supp_nation.find(li.l_suppkey[idx]);
                    if (!s_nat) continue;

                    const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[idx]);
                    if (!cust_ptr) continue;

                    const int8_t* c_nat = cust_nation.find(*cust_ptr);
                    if (!c_nat || *s_nat == *c_nat) continue;

                    int year_idx = (li.l_shipdate[idx] >= dates::DATE_1996_01_01) ? 1 : 0;
                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[idx]) *
                                      (10000 - li.l_discount[idx]) / 10000;
                    local_agg.data[*s_nat][*c_nat][year_idx] += revenue;
                }
#else
                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;
                    int32_t shipdate = li.l_shipdate[idx];
                    if (shipdate < date_lo || shipdate > date_hi) continue;

                    const int8_t* s_nat = supp_nation.find(li.l_suppkey[idx]);
                    if (!s_nat) continue;

                    const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[idx]);
                    if (!cust_ptr) continue;

                    const int8_t* c_nat = cust_nation.find(*cust_ptr);
                    if (!c_nat || *s_nat == *c_nat) continue;

                    int year_idx = (shipdate >= dates::DATE_1996_01_01) ? 1 : 0;
                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[idx]) *
                                      (10000 - li.l_discount[idx]) / 10000;
                    local_agg.data[*s_nat][*c_nat][year_idx] += revenue;
                }
#endif
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t shipdate = li.l_shipdate[i];
                if (shipdate < date_lo || shipdate > date_hi) continue;

                const int8_t* s_nat = supp_nation.find(li.l_suppkey[i]);
                if (!s_nat) continue;

                const int32_t* cust_ptr = order_to_cust.find(li.l_orderkey[i]);
                if (!cust_ptr) continue;

                const int8_t* c_nat = cust_nation.find(*cust_ptr);
                if (!c_nat || *s_nat == *c_nat) continue;

                int year_idx = (shipdate >= dates::DATE_1996_01_01) ? 1 : 0;
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_agg.data[*s_nat][*c_nat][year_idx] += revenue;
            }
        }));
    }

    for (auto& f : futures) f.get();

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

    volatile int64_t sink = results.data[0][1][0] + results.data[0][1][1] +
                            results.data[1][0][0] + results.data[1][0][1];
    (void)sink;
}

// ============================================================================
// Q9 自适应版本
// ============================================================================

void run_q9_adaptive(TPCHDataLoader& loader) {
    if (!adaptive::use_compact_structure(loader.orders().count)) {
        // SF < 3.5: 使用 V25 优化
        run_q9_v25(loader);
    } else {
        // SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
        run_q9_v32_batch(loader);
    }
}

void run_q9_v32_batch(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();

    // Phase 1: 过滤 green parts
    std::unordered_set<int32_t> green_parts_set;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_name[i].find("green") != std::string::npos) {
            green_parts_set.insert(part.p_partkey[i]);
        }
    }

    // Phase 2: 构建查找表
    CompactHashTable<int32_t> supp_to_nation;
    supp_to_nation.init(supp.count);
    for (size_t i = 0; i < supp.count; ++i) {
        supp_to_nation.insert(supp.s_suppkey[i], supp.s_nationkey[i]);
    }

    std::unordered_map<int64_t, int64_t> ps_cost_map;
    ps_cost_map.reserve(partsupp.count);
    for (size_t i = 0; i < partsupp.count; ++i) {
        int64_t key = (static_cast<int64_t>(partsupp.ps_partkey[i]) << 32) |
                      static_cast<uint32_t>(partsupp.ps_suppkey[i]);
        ps_cost_map[key] = partsupp.ps_supplycost[i];
    }

    CompactHashTable<int16_t> order_to_year;
    order_to_year.init(ord.count);
    for (size_t i = 0; i < ord.count; ++i) {
        int16_t year = static_cast<int16_t>(1970 + ord.o_orderdate[i] / 365);
        order_to_year.insert(ord.o_orderkey[i], year);
    }

    std::vector<std::string> nation_names(25);
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_nationkey[i] >= 0 && nat.n_nationkey[i] < 25) {
            nation_names[nat.n_nationkey[i]] = nat.n_name[i];
        }
    }

    // Phase 3: 并行扫描
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<ThreadLocalAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) agg.init(200);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            for (size_t i = start; i < end; ++i) {
                int32_t partkey = li.l_partkey[i];
                if (green_parts_set.find(partkey) == green_parts_set.end()) continue;

                int32_t suppkey = li.l_suppkey[i];
                int32_t orderkey = li.l_orderkey[i];

                const int32_t* nat_ptr = supp_to_nation.find(suppkey);
                if (!nat_ptr) continue;
                int32_t nationkey = *nat_ptr;

                const int16_t* year_ptr = order_to_year.find(orderkey);
                if (!year_ptr) continue;
                int16_t year = *year_ptr;

                int64_t ps_key = (static_cast<int64_t>(partkey) << 32) | static_cast<uint32_t>(suppkey);
                auto cost_it = ps_cost_map.find(ps_key);
                if (cost_it == ps_cost_map.end()) continue;

                __int128 disc_price = (__int128)li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;
                __int128 cost = (__int128)cost_it->second * li.l_quantity[i] / 10000;
                int64_t amount = static_cast<int64_t>(disc_price - cost);

                int32_t agg_key = (nationkey << 16) | (year & 0xFFFF);
                local_agg.add(agg_key, amount);
            }
        }));
    }

    for (auto& f : futures) f.get();

    CompactHashTable<int64_t> final_results;
    final_results.init(200);
    for (auto& agg : thread_aggs) {
        agg.table.for_each([&](int32_t key, int64_t val) {
            final_results.add_or_update(key, val);
        });
    }

    volatile int64_t sink = 0;
    final_results.for_each([&sink](int32_t, int64_t val) { sink += val; });
    (void)sink;
}

// ============================================================================
// Q18 自适应版本
// ============================================================================

void run_q18_adaptive(TPCHDataLoader& loader) {
    if (!adaptive::use_compact_structure(loader.orders().count)) {
        // SF < 3.5: 使用 V27 直接数组
        run_q18_v27(loader);
    } else {
        // SF >= 3.5: 使用 V32 Thread-local 紧凑 Hash
        run_q18_v32_batch(loader);
    }
}

void run_q18_v32_batch(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // Phase 1: Thread-local 聚合
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    size_t estimated_orders = ord.count;

    std::vector<ThreadLocalAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        agg.init(estimated_orders / num_threads + 10000);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            // 8 路展开批量处理
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&li.l_orderkey[i + 64], 0, 3);
                __builtin_prefetch(&li.l_quantity[i + 64], 0, 3);

                local_agg.add(li.l_orderkey[i], li.l_quantity[i]);
                local_agg.add(li.l_orderkey[i+1], li.l_quantity[i+1]);
                local_agg.add(li.l_orderkey[i+2], li.l_quantity[i+2]);
                local_agg.add(li.l_orderkey[i+3], li.l_quantity[i+3]);
                local_agg.add(li.l_orderkey[i+4], li.l_quantity[i+4]);
                local_agg.add(li.l_orderkey[i+5], li.l_quantity[i+5]);
                local_agg.add(li.l_orderkey[i+6], li.l_quantity[i+6]);
                local_agg.add(li.l_orderkey[i+7], li.l_quantity[i+7]);
            }

            for (; i < end; ++i) {
                local_agg.add(li.l_orderkey[i], li.l_quantity[i]);
            }
        }));
    }

    for (auto& f : futures) f.get();
    futures.clear();

    // Phase 2: 合并
    CompactHashTable<int64_t> order_qty;
    order_qty.init(estimated_orders);
    for (auto& agg : thread_aggs) {
        agg.table.for_each([&](int32_t orderkey, int64_t qty) {
            order_qty.add_or_update(orderkey, qty);
        });
    }

    // Phase 3: 过滤 sum > 300 * 10000
    constexpr int64_t qty_threshold = 300 * 10000;
    std::unordered_set<int32_t> large_orders_set;

    order_qty.for_each([&](int32_t orderkey, int64_t qty) {
        if (qty > qty_threshold) {
            large_orders_set.insert(orderkey);
        }
    });

    // Phase 4: 获取结果
    struct Q18Result {
        int32_t custkey;
        int32_t orderkey;
        int32_t orderdate;
        int64_t totalprice;
        int64_t sum_qty;
    };

    std::vector<Q18Result> results;
    results.reserve(large_orders_set.size());

    for (size_t j = 0; j < ord.count; ++j) {
        int32_t okey = ord.o_orderkey[j];
        if (large_orders_set.find(okey) == large_orders_set.end()) continue;

        const int64_t* qty_ptr = order_qty.find(okey);
        if (!qty_ptr) continue;

        Q18Result r;
        r.orderkey = okey;
        r.custkey = ord.o_custkey[j];
        r.orderdate = ord.o_orderdate[j];
        r.totalprice = ord.o_totalprice[j];
        r.sum_qty = *qty_ptr;
        results.push_back(r);
    }

    // Phase 5: 排序
    std::partial_sort(results.begin(),
                      results.begin() + std::min<size_t>(100, results.size()),
                      results.end(),
                      [](const Q18Result& a, const Q18Result& b) {
                          if (a.totalprice != b.totalprice)
                              return a.totalprice > b.totalprice;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 100) results.resize(100);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].totalprice;
    (void)sink;
    (void)sink2;
}

// ============================================================================
// Q19 V32: PredicatePrecomputer + 直接数组 + 并行扫描
// ============================================================================

/**
 * Q19 优化策略:
 * 1. PredicatePrecomputer: 预计算每个 part 匹配的条件组 (0=无, 1/2/3=条件组)
 * 2. 直接数组索引: part_category[partkey] 替代 unordered_map
 * 3. 并行扫描: ThreadPool 分块处理 lineitem
 * 4. 消除字符串比较: 全部在预处理阶段完成
 */

void run_q19_v32(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    // Phase 1: 找到最大 partkey 并分配直接数组
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        max_partkey = std::max(max_partkey, part.p_partkey[i]);
    }

    // part_category[partkey] = 条件组 (0=无匹配, 1/2/3=条件组1/2/3)
    // 同时存储 size 用于 quantity 范围检查的关联
    struct PartMatch {
        uint8_t category;  // 0=无, 1=Brand#12, 2=Brand#23, 3=Brand#34
        int8_t size_valid; // size 是否在范围内 (已预检查)
    };
    std::vector<PartMatch> part_match(max_partkey + 1, {0, 0});

    // Phase 2: PredicatePrecomputer - 预计算条件匹配
    for (size_t i = 0; i < part.count; ++i) {
        int32_t pkey = part.p_partkey[i];
        const std::string& brand = part.p_brand[i];
        const std::string& container = part.p_container[i];
        int32_t psize = part.p_size[i];

        // 条件 1: Brand#12, SM*, size 1-5
        if (brand == "Brand#12" &&
            (container == "SM CASE" || container == "SM BOX" ||
             container == "SM PACK" || container == "SM PKG") &&
            psize >= 1 && psize <= 5) {
            part_match[pkey] = {1, 1};
            continue;
        }

        // 条件 2: Brand#23, MED*, size 1-10
        if (brand == "Brand#23" &&
            (container == "MED BAG" || container == "MED BOX" ||
             container == "MED PKG" || container == "MED PACK") &&
            psize >= 1 && psize <= 10) {
            part_match[pkey] = {2, 1};
            continue;
        }

        // 条件 3: Brand#34, LG*, size 1-15
        if (brand == "Brand#34" &&
            (container == "LG CASE" || container == "LG BOX" ||
             container == "LG PACK" || container == "LG PKG") &&
            psize >= 1 && psize <= 15) {
            part_match[pkey] = {3, 1};
            continue;
        }
    }

    // Phase 3: 并行扫描 lineitem
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 定义 quantity 范围常量 (已乘 10000)
    // 条件 1: qty 1-11 → 10000-110000
    // 条件 2: qty 10-20 → 100000-200000
    // 条件 3: qty 20-30 → 200000-300000
    constexpr int64_t QTY_LO_1 = 10000, QTY_HI_1 = 110000;
    constexpr int64_t QTY_LO_2 = 100000, QTY_HI_2 = 200000;
    constexpr int64_t QTY_LO_3 = 200000, QTY_HI_3 = 300000;

    std::vector<__int128> thread_revenues(num_threads, 0);
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            __int128 local_revenue = 0;

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&li.l_partkey[i + 64], 0, 3);
                __builtin_prefetch(&li.l_shipinstruct[i + 64], 0, 3);

                for (int j = 0; j < 8; ++j) {
                    size_t idx = i + j;

                    // 快速过滤: shipinstruct 和 shipmode
                    if (li.l_shipinstruct[idx] != 0) continue;  // DELIVER IN PERSON
                    int8_t mode = li.l_shipmode[idx];
                    if (mode != 0 && mode != 1) continue;  // AIR or REG AIR

                    // 直接数组查找
                    int32_t pkey = li.l_partkey[idx];
                    if (pkey < 0 || pkey > max_partkey) continue;

                    const auto& pm = part_match[pkey];
                    if (pm.category == 0) continue;  // 无匹配条件

                    // quantity 范围检查 (根据 category)
                    int64_t qty = li.l_quantity[idx];
                    bool qty_match = false;

                    switch (pm.category) {
                        case 1: qty_match = (qty >= QTY_LO_1 && qty <= QTY_HI_1); break;
                        case 2: qty_match = (qty >= QTY_LO_2 && qty <= QTY_HI_2); break;
                        case 3: qty_match = (qty >= QTY_LO_3 && qty <= QTY_HI_3); break;
                    }

                    if (qty_match) {
                        local_revenue += (__int128)li.l_extendedprice[idx] *
                                        (10000 - li.l_discount[idx]) / 10000;
                    }
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                if (li.l_shipinstruct[i] != 0) continue;
                int8_t mode = li.l_shipmode[i];
                if (mode != 0 && mode != 1) continue;

                int32_t pkey = li.l_partkey[i];
                if (pkey < 0 || pkey > max_partkey) continue;

                const auto& pm = part_match[pkey];
                if (pm.category == 0) continue;

                int64_t qty = li.l_quantity[i];
                bool qty_match = false;

                switch (pm.category) {
                    case 1: qty_match = (qty >= QTY_LO_1 && qty <= QTY_HI_1); break;
                    case 2: qty_match = (qty >= QTY_LO_2 && qty <= QTY_HI_2); break;
                    case 3: qty_match = (qty >= QTY_LO_3 && qty <= QTY_HI_3); break;
                }

                if (qty_match) {
                    local_revenue += (__int128)li.l_extendedprice[i] *
                                    (10000 - li.l_discount[i]) / 10000;
                }
            }

            thread_revenues[t] = local_revenue;
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    __int128 total_revenue = 0;
    for (const auto& rev : thread_revenues) {
        total_revenue += rev;
    }

    volatile double result = static_cast<double>(total_revenue) / 10000.0;
    (void)result;
}

} // namespace ops_v32
} // namespace tpch
} // namespace thunderduck
