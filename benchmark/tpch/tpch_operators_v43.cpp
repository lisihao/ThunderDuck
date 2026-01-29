/**
 * ThunderDuck TPC-H V43 优化算子实现
 *
 * Q17 位图过滤 + 两阶段聚合
 *
 * @version 43.0
 * @date 2026-01-29
 */

#include "tpch_operators_v43.h"
#include <algorithm>
#include <thread>
#include <vector>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v43 {

// ============================================================================
// Q17 优化实现 V43 - 位图过滤 + 两阶段聚合
// ============================================================================

void run_q17_v43(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& li = loader.lineitem();

    // ========== 参数 ==========
    const std::string target_brand = "Brand#23";
    const std::string target_container = "MED BOX";
    constexpr double quantity_factor = 0.2;

    // ========================================================================
    // Phase 1: 构建目标 parts 位图 + partkey→index 映射
    // ========================================================================

    // 找最大 partkey
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_partkey[i] > max_partkey) max_partkey = part.p_partkey[i];
    }

    // 位图: is_target_part[partkey] = 1 表示目标 part
    std::vector<uint8_t> is_target_part(max_partkey + 1, 0);
    size_t target_count = 0;

    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_brand[i] == target_brand && part.p_container[i] == target_container) {
            is_target_part[part.p_partkey[i]] = 1;
            target_count++;
        }
    }

    if (target_count == 0) {
        volatile double sink = 0.0;
        (void)sink;
        return;
    }

    // ========================================================================
    // Phase 2: 第一遍扫描 - 计算每个目标 partkey 的 SUM(qty) 和 COUNT
    // ========================================================================

    // 直接数组: qty_sum[partkey], qty_count[partkey]
    std::vector<int64_t> qty_sum(max_partkey + 1, 0);
    std::vector<int32_t> qty_count(max_partkey + 1, 0);

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 8) num_threads = 8;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 线程局部聚合
    struct ThreadLocalQtyState {
        std::vector<int64_t> sum;
        std::vector<int32_t> count;
    };
    std::vector<ThreadLocalQtyState> thread_qty_states(num_threads);
    for (auto& s : thread_qty_states) {
        s.sum.resize(max_partkey + 1, 0);
        s.count.resize(max_partkey + 1, 0);
    }

    // 指针
    const int32_t* l_partkey = li.l_partkey.data();
    const int64_t* l_quantity = li.l_quantity.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const uint8_t* is_target = is_target_part.data();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 第一遍: 计算 AVG
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([=, &thread_qty_states]() {
            auto& local = thread_qty_states[t];

            for (size_t i = start; i < end; ++i) {
                int32_t pk = l_partkey[i];
                if (pk > 0 && pk <= max_partkey && is_target[pk]) {
                    local.sum[pk] += l_quantity[i];
                    local.count[pk]++;
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // 合并 qty 统计
    for (size_t t = 0; t < num_threads; ++t) {
        for (int32_t pk = 1; pk <= max_partkey; ++pk) {
            if (is_target_part[pk]) {
                qty_sum[pk] += thread_qty_states[t].sum[pk];
                qty_count[pk] += thread_qty_states[t].count[pk];
            }
        }
    }

    // 计算阈值: threshold[pk] = 0.2 * AVG(qty) = 0.2 * sum / count
    // 为避免浮点数，使用: qty * count < sum * 0.2
    // 即: qty * count * 5 < sum (乘以 5 消除 0.2)
    std::vector<int64_t> threshold(max_partkey + 1, 0);
    for (int32_t pk = 1; pk <= max_partkey; ++pk) {
        if (is_target_part[pk] && qty_count[pk] > 0) {
            // threshold = 0.2 * avg = 0.2 * sum / count
            // 用整数: l_qty < threshold 等价于 l_qty * count * 5 < sum
            threshold[pk] = static_cast<int64_t>(
                quantity_factor * static_cast<double>(qty_sum[pk]) / qty_count[pk]
            );
        }
    }

    // ========================================================================
    // Phase 3: 第二遍扫描 - 过滤并累加 extendedprice
    // ========================================================================

    // 线程局部累加
    std::vector<int64_t> thread_price_sum(num_threads, 0);

    threads.clear();
    const int64_t* threshold_ptr = threshold.data();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([=, &thread_price_sum]() {
            int64_t local_sum = 0;

            for (size_t i = start; i < end; ++i) {
                int32_t pk = l_partkey[i];
                if (pk > 0 && pk <= max_partkey && is_target[pk]) {
                    // l_quantity < threshold
                    if (l_quantity[i] < threshold_ptr[pk]) {
                        local_sum += l_extendedprice[i];
                    }
                }
            }

            thread_price_sum[t] = local_sum;
        });
    }

    for (auto& th : threads) th.join();

    // 合并最终结果
    int64_t total_price = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        total_price += thread_price_sum[t];
    }

    // avg_yearly = total / 7.0
    double avg_yearly = static_cast<double>(total_price) / 7.0 / 10000.0;  // l_extendedprice 是 x10000

    // 防止优化器消除
    volatile double sink = avg_yearly;
    (void)sink;
}

}  // namespace ops_v43
}  // namespace tpch
}  // namespace thunderduck
