/**
 * ThunderDuck TPC-H V49 - Top-N Aware Partial Aggregation
 *
 * @version 49.0
 * @date 2026-01-29
 */

#include "tpch_operators_v49.h"
#include <algorithm>
#include <future>
#include <queue>
#include <unordered_map>

namespace thunderduck {
namespace tpch {
namespace ops_v49 {

// ============================================================================
// Q3 V49: Top-N Aware Partial Aggregation
// ============================================================================

void run_q3_v49(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t DATE_THRESHOLD = 9204;  // 1995-03-15
    constexpr size_t FINAL_TOP_N = 10;
    constexpr size_t LOCAL_TOP_K = 32;  // 每线程保留 top-K (> FINAL_TOP_N)

    // ========================================================================
    // Phase 1: 构建 BUILDING custkey bitmap (同 V31)
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
    // Phase 2: 构建紧凑 Hash Table + Bloom Filter (同 V31)
    // ========================================================================

    struct OrderEntry {
        int32_t orderkey;
        int32_t orderdate;
        int8_t shippriority;
    };

    // 预估 ~20% 订单有效
    size_t estimated_valid = ord.count / 5;

    // 紧凑 hash table (开放寻址)
    size_t table_size = 1;
    while (table_size < estimated_valid * 2) table_size <<= 1;
    uint32_t table_mask = static_cast<uint32_t>(table_size - 1);

    std::vector<OrderEntry> order_table(table_size, {INT32_MIN, 0, 0});

    // Bloom Filter
    size_t bloom_bits = estimated_valid * 2;
    bloom_bits = ((bloom_bits + 63) / 64) * 64;
    std::vector<uint64_t> bloom(bloom_bits / 64, 0);
    uint32_t bloom_mask = static_cast<uint32_t>(bloom_bits - 1);

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
            uint32_t pos = hash & table_mask;
            while (order_table[pos].orderkey != INT32_MIN) {
                pos = (pos + 1) & table_mask;
            }
            order_table[pos] = {orderkey, orderdate,
                                static_cast<int8_t>(ord.o_shippriority[i])};

            bloom[(hash & bloom_mask) >> 6] |= (1ULL << ((hash & bloom_mask) & 63));
        }
    }

    // ========================================================================
    // Phase 3: 线程局部聚合 (无原子操作)
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int8_t shippriority;

        // min-heap 比较器: revenue 小的在顶部
        bool operator<(const Q3Result& other) const {
            if (revenue != other.revenue) return revenue > other.revenue;
            return orderdate > other.orderdate;
        }
    };

    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 每线程的局部 Top-K 结果
    std::vector<std::vector<Q3Result>> local_tops(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            // 线程局部聚合 map
            std::unordered_map<int32_t, int64_t> local_revenue;
            local_revenue.reserve(estimated_valid / num_threads);

            for (size_t i = start; i < end; ++i) {
                if (li.l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t orderkey = li.l_orderkey[i];

#ifdef __aarch64__
                uint32_t hash = __crc32w(0, static_cast<uint32_t>(orderkey));
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
                    if (order_table[pos].orderkey == INT32_MIN) break;
                    if (order_table[pos].orderkey == orderkey) {
                        int64_t rev = static_cast<int64_t>(li.l_extendedprice[i]) *
                                      (10000 - li.l_discount[i]) / 10000;
                        local_revenue[orderkey] += rev;
                        break;
                    }
                    pos = (pos + 1) & table_mask;
                }
            }

            // 线程局部 Top-K 提取
            std::priority_queue<Q3Result> local_heap;

            for (auto& [orderkey, revenue] : local_revenue) {
                if (revenue <= 0) continue;

                // 查找 orderdate 和 shippriority
#ifdef __aarch64__
                uint32_t hash = __crc32w(0, static_cast<uint32_t>(orderkey));
#else
                uint32_t hash = static_cast<uint32_t>(orderkey) * 0x85ebca6b;
#endif
                uint32_t pos = hash & table_mask;
                while (order_table[pos].orderkey != orderkey) {
                    pos = (pos + 1) & table_mask;
                }

                Q3Result result{orderkey, revenue, order_table[pos].orderdate,
                                order_table[pos].shippriority};

                if (local_heap.size() < LOCAL_TOP_K) {
                    local_heap.push(result);
                } else if (result.revenue > local_heap.top().revenue ||
                           (result.revenue == local_heap.top().revenue &&
                            result.orderdate < local_heap.top().orderdate)) {
                    local_heap.pop();
                    local_heap.push(result);
                }
            }

            // 提取到 vector
            local_tops[t].reserve(local_heap.size());
            while (!local_heap.empty()) {
                local_tops[t].push_back(local_heap.top());
                local_heap.pop();
            }
        }));
    }

    for (auto& f : futures) f.get();

    // ========================================================================
    // Phase 4: 全局合并 (增量 heap)
    // ========================================================================

    // 合并相同 orderkey 的 revenue
    std::unordered_map<int32_t, Q3Result> merged;
    merged.reserve(num_threads * LOCAL_TOP_K);

    for (size_t t = 0; t < num_threads; ++t) {
        for (const auto& r : local_tops[t]) {
            auto it = merged.find(r.orderkey);
            if (it == merged.end()) {
                merged[r.orderkey] = r;
            } else {
                it->second.revenue += r.revenue;
            }
        }
    }

    // 最终 Top-N
    std::vector<Q3Result> results;
    results.reserve(merged.size());
    for (auto& [_, r] : merged) {
        results.push_back(r);
    }

    size_t top_n = std::min(FINAL_TOP_N, results.size());
    std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                      [](const Q3Result& a, const Q3Result& b) {
                          if (a.revenue != b.revenue) return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > FINAL_TOP_N) results.resize(FINAL_TOP_N);

    volatile size_t sink = results.size();
    volatile int64_t sink2 = results.empty() ? 0 : results[0].revenue;
    (void)sink;
    (void)sink2;
}

} // namespace ops_v49
} // namespace tpch
} // namespace thunderduck
