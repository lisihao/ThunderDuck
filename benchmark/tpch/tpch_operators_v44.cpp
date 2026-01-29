/**
 * ThunderDuck TPC-H V44 优化算子实现
 *
 * Q3 紧凑位图 + 直接数组访问 (消除 Hash Table 探测开销)
 *
 * @version 44.0
 * @date 2026-01-29
 */

#include "tpch_operators_v44.h"
#include <algorithm>
#include <thread>
#include <atomic>
#include <vector>
#include <cstring>

#ifdef __aarch64__
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v44 {

// ============================================================================
// Q3 优化实现 V44 - 紧凑位图早期拒绝 + 直接数组访问
//
// 优化点对比 V31:
// - V31: Bloom Filter (75KB) + Hash Table (需探测)
// - V44: 紧凑位图 (750KB) + 直接数组 (无探测)
// ============================================================================

void run_q3_v44(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    constexpr int32_t DATE_THRESHOLD = 9204;  // 1995-03-15

    // ========================================================================
    // Phase 1: 构建 BUILDING 客户位图
    // ========================================================================

    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }

    std::vector<uint8_t> is_building(max_custkey + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {
            is_building[cust.c_custkey[i]] = 1;
        }
    }

    // ========================================================================
    // Phase 2: 构建紧凑位图 + 直接数组
    // ========================================================================

    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    // 紧凑位图: valid_order_bitmap[orderkey/8] 的第 (orderkey%8) 位
    // 大小: ~750 KB (6M/8 bytes) - 比 Bloom Filter 略大但更精确
    std::vector<uint8_t> valid_order_bitmap((max_orderkey + 8) / 8, 0);

    // 直接数组: order_info[orderkey]
    struct OrderInfo {
        int32_t compact_idx;
        int32_t orderdate;
        int8_t shippriority;
    };
    std::vector<OrderInfo> order_info(max_orderkey + 1, {-1, 0, 0});

    struct ValidOrder {
        int32_t orderkey;
        int32_t orderdate;
        int8_t shippriority;
    };
    std::vector<ValidOrder> valid_orders;
    valid_orders.reserve(ord.count / 5);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t custkey = ord.o_custkey[i];
        int32_t orderdate = ord.o_orderdate[i];

        if (custkey <= max_custkey && is_building[custkey] && orderdate < DATE_THRESHOLD) {
            int32_t orderkey = ord.o_orderkey[i];
            int32_t idx = static_cast<int32_t>(valid_orders.size());

            // 设置位图
            valid_order_bitmap[orderkey >> 3] |= (1u << (orderkey & 7));

            // 直接数组
            order_info[orderkey] = {idx, orderdate, static_cast<int8_t>(ord.o_shippriority[i])};

            valid_orders.push_back({orderkey, orderdate, static_cast<int8_t>(ord.o_shippriority[i])});
        }
    }

    size_t num_valid = valid_orders.size();
    if (num_valid == 0) {
        volatile size_t sink = 0;
        (void)sink;
        return;
    }

    // ========================================================================
    // Phase 3: 并行扫描 lineitem
    // ========================================================================

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 8) num_threads = 8;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<std::atomic<int64_t>> revenue(num_valid);
    for (auto& r : revenue) r.store(0, std::memory_order_relaxed);

    const int32_t* l_orderkey = li.l_orderkey.data();
    const int32_t* l_shipdate = li.l_shipdate.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();
    const uint8_t* bitmap = valid_order_bitmap.data();
    const OrderInfo* info_arr = order_info.data();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([=, &revenue]() {
            for (size_t i = start; i < end; ++i) {
                // 过滤 1: shipdate
                if (l_shipdate[i] <= DATE_THRESHOLD) continue;

                int32_t okey = l_orderkey[i];
                if (okey <= 0 || okey > max_orderkey) continue;

                // 过滤 2: 紧凑位图快速拒绝 (缓存友好)
                if (!(bitmap[okey >> 3] & (1u << (okey & 7)))) continue;

                // 直接数组访问 (无 hash, 无探测)
                const auto& info = info_arr[okey];

                int64_t rev = l_extendedprice[i] * (10000 - l_discount[i]) / 10000;
                revenue[info.compact_idx].fetch_add(rev, std::memory_order_relaxed);
            }
        });
    }

    for (auto& th : threads) th.join();

    // ========================================================================
    // Phase 4: Top-10 排序
    // ========================================================================

    struct Q3Result {
        int32_t orderkey;
        int64_t revenue;
        int32_t orderdate;
        int32_t shippriority;
    };

    std::vector<Q3Result> results;
    results.reserve(num_valid / 10);

    for (size_t i = 0; i < num_valid; ++i) {
        int64_t rev = revenue[i].load(std::memory_order_relaxed);
        if (rev > 0) {
            results.push_back({
                valid_orders[i].orderkey,
                rev,
                valid_orders[i].orderdate,
                valid_orders[i].shippriority
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

}  // namespace ops_v44
}  // namespace tpch
}  // namespace thunderduck
