/**
 * ThunderDuck TPC-H 算子封装 V26 - 实现
 */

#include "tpch_operators_v26.h"
#include <algorithm>
#include <thread>
#include <numeric>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v26 {

// ============================================================================
// VectorizedGroupBySum 实现
// ============================================================================

void VectorizedGroupBySum::init(size_t estimated_groups) {
    table_.init(estimated_groups);
}

void VectorizedGroupBySum::batch_hash(const int32_t* keys, size_t n, uint32_t* out_hashes) {
#ifdef __aarch64__
    // SIMD批量hash: 4个元素一组
    const uint32_t golden = 2654435769u;
    uint32x4_t golden_vec = vdupq_n_u32(golden);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t k = vld1q_s32(&keys[i]);
        uint32x4_t uk = vreinterpretq_u32_s32(k);
        uint32x4_t h = vmulq_u32(uk, golden_vec);
        vst1q_u32(&out_hashes[i], h);
    }

    // 处理剩余元素
    for (; i < n; ++i) {
        out_hashes[i] = weak_hash_i32(keys[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out_hashes[i] = weak_hash_i32(keys[i]);
    }
#endif
}

void VectorizedGroupBySum::aggregate(const int32_t* keys, const int64_t* values, size_t n) {
    // 8路展开 + prefetch
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __builtin_prefetch(&keys[i + 64], 0, 3);
        __builtin_prefetch(&values[i + 64], 0, 3);

        table_.add_or_update(keys[i + 0], values[i + 0]);
        table_.add_or_update(keys[i + 1], values[i + 1]);
        table_.add_or_update(keys[i + 2], values[i + 2]);
        table_.add_or_update(keys[i + 3], values[i + 3]);
        table_.add_or_update(keys[i + 4], values[i + 4]);
        table_.add_or_update(keys[i + 5], values[i + 5]);
        table_.add_or_update(keys[i + 6], values[i + 6]);
        table_.add_or_update(keys[i + 7], values[i + 7]);
    }

    for (; i < n; ++i) {
        table_.add_or_update(keys[i], values[i]);
    }
}

// ============================================================================
// Fused Filter-Join-Aggregate 实现
// ============================================================================

void fused_q3_v26(
    const int32_t* l_orderkey,
    const int32_t* l_shipdate,
    const int64_t* l_extendedprice,
    const int64_t* l_discount,
    size_t lineitem_count,
    const WeakHashTable<Q3AggResultV26>& valid_orders,
    MutableWeakHashTable<Q3AggResultV26>& results,
    int32_t date_threshold
) {
    // 预计算hash缓存
    KeyHashCache orderkey_hash;
    orderkey_hash.build(l_orderkey, lineitem_count,
                        static_cast<uint32_t>(valid_orders.table_size()));

    const uint32_t* hash_cache = orderkey_hash.data();

    // 单遍扫描: Filter + Join + Aggregate
    size_t i = 0;

#ifdef __aarch64__
    // SIMD过滤 + 标量聚合
    int32x4_t threshold_vec = vdupq_n_s32(date_threshold);

    for (; i + 4 <= lineitem_count; i += 4) {
        // 加载shipdate
        int32x4_t dates = vld1q_s32(&l_shipdate[i]);

        // 比较: shipdate > threshold
        uint32x4_t mask = vcgtq_s32(dates, threshold_vec);

        // 提取mask并处理匹配的行
        uint32_t mask_bits[4];
        vst1q_u32(mask_bits, mask);

        for (int j = 0; j < 4; ++j) {
            if (mask_bits[j] == 0) continue;

            size_t idx = i + j;
            int32_t orderkey = l_orderkey[idx];

            // Hash Join查找
            int32_t entry = valid_orders.find_with_hash(orderkey, hash_cache[idx]);
            if (entry < 0) continue;

            // 计算revenue
            __int128 rev = (__int128)l_extendedprice[idx] *
                           (10000 - l_discount[idx]) / 10000;

            // 聚合 (需要获取orderdate和shippriority)
            const auto& order_info = valid_orders.get_value(entry);

            // 更新结果
            Q3AggResultV26* existing = results.find(orderkey);
            if (existing) {
                existing->revenue += static_cast<int64_t>(rev);
            } else {
                Q3AggResultV26 new_result;
                new_result.revenue = static_cast<int64_t>(rev);
                new_result.orderdate = order_info.orderdate;
                new_result.shippriority = order_info.shippriority;
                results.add_or_update(orderkey, new_result);
            }
        }
    }
#endif

    // 处理剩余元素
    for (; i < lineitem_count; ++i) {
        if (l_shipdate[i] <= date_threshold) continue;

        int32_t orderkey = l_orderkey[i];
        int32_t entry = valid_orders.find_with_hash(orderkey, hash_cache[i]);
        if (entry < 0) continue;

        __int128 rev = (__int128)l_extendedprice[i] *
                       (10000 - l_discount[i]) / 10000;

        const auto& order_info = valid_orders.get_value(entry);

        Q3AggResultV26* existing = results.find(orderkey);
        if (existing) {
            existing->revenue += static_cast<int64_t>(rev);
        } else {
            Q3AggResultV26 new_result;
            new_result.revenue = static_cast<int64_t>(rev);
            new_result.orderdate = order_info.orderdate;
            new_result.shippriority = order_info.shippriority;
            results.add_or_update(orderkey, new_result);
        }
    }
}

// ============================================================================
// Q3 优化版: SIMD预过滤 + Bloom Filter + 批量聚合
// ============================================================================

void fused_q3_optimized_v26(
    const int32_t* l_orderkey,
    const int32_t* l_shipdate,
    const int64_t* l_extendedprice,
    const int64_t* l_discount,
    size_t lineitem_count,
    const WeakHashTable<Q3AggResultV26>& valid_orders,
    const BloomFilter& valid_orders_bloom,
    MutableWeakHashTable<int64_t>& revenue_agg,
    int32_t date_threshold
) {
    // 预计算hash缓存
    KeyHashCache orderkey_hash;
    orderkey_hash.build(l_orderkey, lineitem_count,
                        static_cast<uint32_t>(valid_orders.table_size()));
    const uint32_t* hash_cache = orderkey_hash.data();

    // 分批处理 (每批4096行，适合L1缓存)
    constexpr size_t BATCH_SIZE = 4096;

    // 临时缓冲区 (存储通过shipdate过滤的索引)
    alignas(64) uint32_t date_pass_indices[BATCH_SIZE];
    // 通过Bloom Filter的索引
    alignas(64) uint32_t bloom_pass_indices[BATCH_SIZE];

#ifdef __aarch64__
    int32x4_t threshold_vec = vdupq_n_s32(date_threshold);
#endif

    for (size_t batch_start = 0; batch_start < lineitem_count; batch_start += BATCH_SIZE) {
        size_t batch_end = std::min(batch_start + BATCH_SIZE, lineitem_count);
        size_t batch_size = batch_end - batch_start;

        // ===== Step 1: SIMD批量过滤shipdate =====
        size_t date_pass_count = 0;

#ifdef __aarch64__
        size_t i = 0;
        for (; i + 4 <= batch_size; i += 4) {
            size_t idx = batch_start + i;
            int32x4_t dates = vld1q_s32(&l_shipdate[idx]);

            // shipdate > threshold
            uint32x4_t mask = vcgtq_s32(dates, threshold_vec);

            // 提取通过的索引
            uint32_t mask_bits[4];
            vst1q_u32(mask_bits, mask);

            for (int j = 0; j < 4; ++j) {
                if (mask_bits[j]) {
                    date_pass_indices[date_pass_count++] = static_cast<uint32_t>(idx + j);
                }
            }
        }

        // 处理剩余
        for (; i < batch_size; ++i) {
            size_t idx = batch_start + i;
            if (l_shipdate[idx] > date_threshold) {
                date_pass_indices[date_pass_count++] = static_cast<uint32_t>(idx);
            }
        }
#else
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = batch_start + i;
            if (l_shipdate[idx] > date_threshold) {
                date_pass_indices[date_pass_count++] = static_cast<uint32_t>(idx);
            }
        }
#endif

        if (date_pass_count == 0) continue;

        // ===== Step 2: Bloom Filter预过滤orderkey =====
        size_t bloom_pass_count = 0;
        for (size_t i = 0; i < date_pass_count; ++i) {
            uint32_t idx = date_pass_indices[i];
            if (valid_orders_bloom.may_contain(l_orderkey[idx])) {
                bloom_pass_indices[bloom_pass_count++] = idx;
            }
        }

        if (bloom_pass_count == 0) continue;

        // ===== Step 3: Hash查找 + 聚合 =====
        // 只对通过Bloom Filter的行做精确hash查找
        for (size_t i = 0; i < bloom_pass_count; ++i) {
            uint32_t idx = bloom_pass_indices[i];
            int32_t orderkey = l_orderkey[idx];

            // 精确hash查找
            if (valid_orders.find_with_hash(orderkey, hash_cache[idx]) < 0) {
                continue;  // Bloom Filter假阳性
            }

            // 计算revenue并聚合
            __int128 rev = (__int128)l_extendedprice[idx] *
                           (10000 - l_discount[idx]) / 10000;
            revenue_agg.add_or_update(orderkey, static_cast<int64_t>(rev));
        }
    }
}

void fused_q18_groupby_v26(
    const int32_t* l_orderkey,
    const int64_t* l_quantity,
    size_t count,
    MutableWeakHashTable<int64_t>& order_qty
) {
    // 使用MutableWeakHashTable进行高效聚合
    // 8路展开 + prefetch
    size_t i = 0;

    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(&l_orderkey[i + 64], 0, 3);
        __builtin_prefetch(&l_quantity[i + 64], 0, 3);

        order_qty.add_or_update(l_orderkey[i + 0], l_quantity[i + 0]);
        order_qty.add_or_update(l_orderkey[i + 1], l_quantity[i + 1]);
        order_qty.add_or_update(l_orderkey[i + 2], l_quantity[i + 2]);
        order_qty.add_or_update(l_orderkey[i + 3], l_quantity[i + 3]);
        order_qty.add_or_update(l_orderkey[i + 4], l_quantity[i + 4]);
        order_qty.add_or_update(l_orderkey[i + 5], l_quantity[i + 5]);
        order_qty.add_or_update(l_orderkey[i + 6], l_quantity[i + 6]);
        order_qty.add_or_update(l_orderkey[i + 7], l_quantity[i + 7]);
    }

    for (; i < count; ++i) {
        order_qty.add_or_update(l_orderkey[i], l_quantity[i]);
    }
}

// ============================================================================
// Q3 V26: 多线程并行 + 分区聚合
// ============================================================================

void run_q3_v26(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t date_threshold = 9204;  // 1995-03-15
    const size_t num_threads = ThreadPool::instance().size();

    // ===== Step 1: 构建BUILDING客户 hash set =====
    WeakHashTable<uint32_t> building_cust;
    building_cust.init(cust.count / 5);

    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {
            building_cust.insert(cust.c_custkey[i], static_cast<uint32_t>(i));
        }
    }

    // ===== Step 2: 并行过滤orders =====
    struct OrderInfo { int32_t orderdate; int32_t shippriority; };

    // 先并行收集有效的orders
    struct ValidOrder { int32_t orderkey; int32_t orderdate; int32_t shippriority; };
    std::vector<std::vector<ValidOrder>> thread_orders(num_threads);

    KeyHashCache ord_custkey_hash;
    ord_custkey_hash.build(ord.o_custkey.data(), ord.count,
                           static_cast<uint32_t>(building_cust.table_size()));
    const uint32_t* och = ord_custkey_hash.data();

    const size_t ord_chunk_size = (ord.count + num_threads - 1) / num_threads;

    std::vector<std::future<void>> ord_futures;
    ord_futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        ord_futures.push_back(ThreadPool::instance().submit([&, t]() {
            size_t start = t * ord_chunk_size;
            size_t end = std::min(start + ord_chunk_size, ord.count);
            auto& local_orders = thread_orders[t];
            local_orders.reserve((end - start) / 4);

            for (size_t i = start; i < end; ++i) {
                if (ord.o_orderdate[i] >= date_threshold) continue;
                if (building_cust.find_with_hash(ord.o_custkey[i], och[i]) < 0) continue;
                local_orders.push_back({ord.o_orderkey[i], ord.o_orderdate[i], ord.o_shippriority[i]});
            }
        }));
    }
    for (auto& f : ord_futures) f.get();

    // 合并到 valid_orders
    size_t total_valid = 0;
    for (const auto& v : thread_orders) total_valid += v.size();

    WeakHashTable<OrderInfo> valid_orders;
    valid_orders.init(total_valid + 100);

    for (const auto& local_orders : thread_orders) {
        for (const auto& vo : local_orders) {
            valid_orders.insert(vo.orderkey, {vo.orderdate, vo.shippriority});
        }
    }

    // ===== Step 3: 多线程并行扫描 lineitem =====
    // 每个线程有独立的局部聚合表，最后合并
    std::vector<MutableWeakHashTable<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        agg.init(valid_orders.entry_count() / num_threads + 1000);
    }

    const size_t chunk_size = (li.count + num_threads - 1) / num_threads;
    const uint32_t valid_orders_mask = static_cast<uint32_t>(valid_orders.table_size() - 1);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        futures.push_back(ThreadPool::instance().submit([&, t]() {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, li.count);
            auto& local_agg = thread_aggs[t];

            // 线程局部处理，延迟hash计算
            for (size_t i = start; i < end; ++i) {
                if (li.l_shipdate[i] <= date_threshold) continue;

                int32_t orderkey = li.l_orderkey[i];
                // 延迟hash: 只对通过日期过滤的行计算
                uint32_t hash = weak_hash_i32(orderkey) & valid_orders_mask;

                if (valid_orders.find_with_hash(orderkey, hash) < 0) continue;

                __int128 rev = (__int128)li.l_extendedprice[i] *
                               (10000 - li.l_discount[i]) / 10000;
                local_agg.add_or_update(orderkey, static_cast<int64_t>(rev));
            }
        }));
    }

    // 等待所有线程完成
    for (auto& f : futures) f.get();

    // ===== Step 4: 合并线程局部结果 =====
    MutableWeakHashTable<int64_t> revenue_agg;
    revenue_agg.init(valid_orders.entry_count());

    for (auto& local_agg : thread_aggs) {
        local_agg.for_each([&](int32_t key, int64_t value) {
            revenue_agg.add_or_update(key, value);
        });
    }

    // ===== Step 5: 构建最终结果并排序 =====
    struct Q3Final {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Final> final_results;
    final_results.reserve(revenue_agg.size());

    revenue_agg.for_each([&](int32_t orderkey, int64_t revenue) {
        int32_t entry = valid_orders.find(orderkey);
        if (entry >= 0) {
            const auto& info = valid_orders.get_value(entry);
            final_results.push_back({orderkey, revenue, info.orderdate, info.shippriority});
        }
    });

    std::partial_sort(
        final_results.begin(),
        final_results.begin() + std::min<size_t>(10, final_results.size()),
        final_results.end(),
        [](const Q3Final& a, const Q3Final& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        }
    );

    if (final_results.size() > 10) final_results.resize(10);

    asm volatile("" : : "r"(final_results.data()) : "memory");
}

// ============================================================================
// Q18 V26: MutableWeakHashTable + VectorizedGroupBySum
// ============================================================================

void run_q18_v26(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const size_t num_threads = ThreadPool::instance().size();

    // ===== Step 1: 并行GROUP BY SUM =====
    // orderkey -> sum(quantity)，使用线程局部聚合
    std::vector<MutableWeakHashTable<int64_t>> thread_qtys(num_threads);
    for (auto& tbl : thread_qtys) {
        tbl.init(ord.count / num_threads + 1000);
    }

    const size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        futures.push_back(ThreadPool::instance().submit([&, t]() {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, li.count);
            auto& local_qty = thread_qtys[t];

            for (size_t i = start; i < end; ++i) {
                local_qty.add_or_update(li.l_orderkey[i], li.l_quantity[i]);
            }
        }));
    }
    for (auto& f : futures) f.get();

    // 合并线程局部结果
    MutableWeakHashTable<int64_t> order_qty;
    order_qty.init(ord.count);
    for (auto& local_qty : thread_qtys) {
        local_qty.for_each([&](int32_t key, int64_t value) {
            order_qty.add_or_update(key, value);
        });
    }

    // ===== Step 2: HAVING SUM(l_quantity) > 300 =====
    constexpr int64_t qty_threshold = 300LL * 10000;

    WeakHashTable<int8_t> large_orders;
    size_t estimated_large = order_qty.size() / 100;
    if (estimated_large < 100) estimated_large = 100;
    large_orders.init(estimated_large);

    order_qty.for_each([&](int32_t orderkey, int64_t qty) {
        if (qty > qty_threshold) {
            large_orders.insert(orderkey, 1);
        }
    });

    // ===== Step 3: JOIN orders (parallel) =====
    WeakHashTable<uint32_t> cust_table;
    cust_table.init(cust.count);
    for (size_t i = 0; i < cust.count; ++i) {
        cust_table.insert(cust.c_custkey[i], static_cast<uint32_t>(i));
    }

    struct Q18Result {
        int32_t custkey;
        int32_t orderkey;
        int32_t orderdate;
        int64_t totalprice;
        int64_t sum_qty;
    };

    // 并行扫描orders
    std::vector<std::vector<Q18Result>> thread_results(num_threads);
    const size_t ord_chunk_size = (ord.count + num_threads - 1) / num_threads;
    const uint32_t large_orders_mask = static_cast<uint32_t>(large_orders.table_size() - 1);

    futures.clear();
    for (size_t t = 0; t < num_threads; ++t) {
        futures.push_back(ThreadPool::instance().submit([&, t]() {
            size_t start = t * ord_chunk_size;
            size_t end = std::min(start + ord_chunk_size, ord.count);
            auto& local_results = thread_results[t];
            local_results.reserve(large_orders.entry_count() / num_threads + 10);

            for (size_t i = start; i < end; ++i) {
                int32_t orderkey = ord.o_orderkey[i];
                uint32_t hash = weak_hash_i32(orderkey) & large_orders_mask;

                if (large_orders.find_with_hash(orderkey, hash) < 0) continue;

                Q18Result r;
                r.orderkey = orderkey;
                r.custkey = ord.o_custkey[i];
                r.orderdate = ord.o_orderdate[i];
                r.totalprice = ord.o_totalprice[i];

                const int64_t* qty_ptr = order_qty.find(r.orderkey);
                r.sum_qty = qty_ptr ? *qty_ptr : 0;

                local_results.push_back(r);
            }
        }));
    }
    for (auto& f : futures) f.get();

    // 合并结果
    std::vector<Q18Result> results;
    size_t total_results = 0;
    for (const auto& v : thread_results) total_results += v.size();
    results.reserve(total_results);

    for (auto& local_results : thread_results) {
        results.insert(results.end(), local_results.begin(), local_results.end());
    }

    // ===== Step 4: ORDER BY o_totalprice DESC, o_orderdate LIMIT 100 =====
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

    asm volatile("" : : "r"(results.data()) : "memory");
}

} // namespace ops_v26
} // namespace tpch
} // namespace thunderduck
