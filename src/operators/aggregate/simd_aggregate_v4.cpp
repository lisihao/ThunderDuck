/**
 * ThunderDuck - SIMD Aggregation Implementation v4.0
 *
 * V9 清理版本 - 只保留有效优化:
 * - P3: 多线程分组聚合 (2.87-3x 加速)
 * - P0: 展开循环 + 预取 (分组聚合 ~18% 提升)
 *
 * 已移除 (测试显示无效或回退):
 * - P1: 256B预取 (实测 -9%~-13% 回退)
 * - P2: 缓存分块 (实测 ~0% 收益)
 *
 * 对于 SUM/MIN/MAX 等单算子，V7/V8 的 v2 实现已是最优，
 * v4 版本委托给 v2 保持兼容性。
 */

#include "thunderduck/aggregate.h"
#include <limits>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thread>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 常量定义
// ============================================================================

// 多线程参数
constexpr size_t MIN_ELEMENTS_PER_THREAD = 100000;  // 最小每线程元素数
constexpr size_t MAX_THREADS = 4;  // M4 性能核数量

// ============================================================================
// SUM v4 - 委托给 v2 (P1 256B预取已移除，v2 更优)
// ============================================================================

int64_t sum_i32_v4(const int32_t* input, size_t count) {
    return sum_i32_v2(input, count);
}

int64_t sum_i32_v4_blocked(const int32_t* input, size_t count) {
    // P2 缓存分块无明显收益，直接委托 v2
    return sum_i32_v2(input, count);
}

// ============================================================================
// MIN/MAX v4 - 委托给 v2 (P1 256B预取已移除，v2 更优)
// ============================================================================

void minmax_i32_v4(const int32_t* input, size_t count,
                   int32_t* out_min, int32_t* out_max) {
    minmax_i32(input, count, out_min, out_max);
}

// ============================================================================
// 融合统计量 v4 - 委托给基础版本
// ============================================================================

AggregateStats aggregate_all_i32_v4(const int32_t* input, size_t count) {
    return aggregate_all_i32(input, count);
}

// ============================================================================
// P0: 分组聚合优化 - 展开循环 + 预取
// ============================================================================
// 测试显示 ~18% 提升，保留此优化

void group_sum_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    // 初始化输出
    std::memset(out_sums, 0, num_groups * sizeof(int64_t));

    size_t i = 0;

#ifdef __aarch64__
    // 4路展开 + 预取
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(&groups[i + 32], 0, 2);
        __builtin_prefetch(&values[i + 32], 0, 2);

        uint32_t g0 = groups[i];
        uint32_t g1 = groups[i + 1];
        uint32_t g2 = groups[i + 2];
        uint32_t g3 = groups[i + 3];

        if (g0 < num_groups) out_sums[g0] += values[i];
        if (g1 < num_groups) out_sums[g1] += values[i + 1];
        if (g2 < num_groups) out_sums[g2] += values[i + 2];
        if (g3 < num_groups) out_sums[g3] += values[i + 3];
    }
#endif

    // 处理剩余
    for (; i < count; ++i) {
        uint32_t gid = groups[i];
        if (gid < num_groups) {
            out_sums[gid] += values[i];
        }
    }
}

// ============================================================================
// P3: 多线程分组聚合 - 核心优化 (2.87-3x 加速)
// ============================================================================

void group_sum_i32_v4_parallel(const int32_t* values, const uint32_t* groups,
                               size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    // 初始化输出
    std::memset(out_sums, 0, num_groups * sizeof(int64_t));

    // 计算线程数
    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    // 小数据量：单线程
    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD) {
        group_sum_i32_v4(values, groups, count, num_groups, out_sums);
        return;
    }

    // 每线程的局部累加结果
    std::vector<std::vector<int64_t>> local_sums(num_threads);
    for (auto& ls : local_sums) {
        ls.resize(num_groups, 0);
    }

    // 计算每线程的数据范围
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    // 启动工作线程
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int64_t* local = local_sums[t].data();

            // 局部累加
            for (size_t i = start; i < end; ++i) {
                uint32_t gid = groups[i];
                if (gid < num_groups) {
                    local[gid] += values[i];
                }
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t g = 0; g < num_groups; ++g) {
            out_sums[g] += local_sums[t][g];
        }
    }
}

// ============================================================================
// P3: 多线程 COUNT
// ============================================================================

void group_count_v4(const uint32_t* groups, size_t count,
                    size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    std::memset(out_counts, 0, num_groups * sizeof(size_t));

    size_t i = 0;

#ifdef __aarch64__
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(&groups[i + 32], 0, 2);

        uint32_t g0 = groups[i];
        uint32_t g1 = groups[i + 1];
        uint32_t g2 = groups[i + 2];
        uint32_t g3 = groups[i + 3];

        if (g0 < num_groups) out_counts[g0]++;
        if (g1 < num_groups) out_counts[g1]++;
        if (g2 < num_groups) out_counts[g2]++;
        if (g3 < num_groups) out_counts[g3]++;
    }
#endif

    for (; i < count; ++i) {
        uint32_t gid = groups[i];
        if (gid < num_groups) {
            out_counts[gid]++;
        }
    }
}

void group_count_v4_parallel(const uint32_t* groups, size_t count,
                             size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    std::memset(out_counts, 0, num_groups * sizeof(size_t));

    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD) {
        group_count_v4(groups, count, num_groups, out_counts);
        return;
    }

    std::vector<std::vector<size_t>> local_counts(num_threads);
    for (auto& lc : local_counts) {
        lc.resize(num_groups, 0);
    }

    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            size_t* local = local_counts[t].data();
            for (size_t i = start; i < end; ++i) {
                uint32_t gid = groups[i];
                if (gid < num_groups) {
                    local[gid]++;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t g = 0; g < num_groups; ++g) {
            out_counts[g] += local_counts[t][g];
        }
    }
}

// ============================================================================
// P0: 分组 MIN/MAX - 展开循环 + 预取
// ============================================================================

void group_min_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int32_t* out_mins) {
    if (count == 0 || !values || !groups || !out_mins) return;

    std::fill(out_mins, out_mins + num_groups, std::numeric_limits<int32_t>::max());

    size_t i = 0;

#ifdef __aarch64__
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(&groups[i + 32], 0, 2);
        __builtin_prefetch(&values[i + 32], 0, 2);

        uint32_t g0 = groups[i], g1 = groups[i + 1], g2 = groups[i + 2], g3 = groups[i + 3];
        int32_t v0 = values[i], v1 = values[i + 1], v2 = values[i + 2], v3 = values[i + 3];

        if (g0 < num_groups && v0 < out_mins[g0]) out_mins[g0] = v0;
        if (g1 < num_groups && v1 < out_mins[g1]) out_mins[g1] = v1;
        if (g2 < num_groups && v2 < out_mins[g2]) out_mins[g2] = v2;
        if (g3 < num_groups && v3 < out_mins[g3]) out_mins[g3] = v3;
    }
#endif

    for (; i < count; ++i) {
        uint32_t gid = groups[i];
        if (gid < num_groups && values[i] < out_mins[gid]) {
            out_mins[gid] = values[i];
        }
    }
}

void group_max_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int32_t* out_maxs) {
    if (count == 0 || !values || !groups || !out_maxs) return;

    std::fill(out_maxs, out_maxs + num_groups, std::numeric_limits<int32_t>::min());

    size_t i = 0;

#ifdef __aarch64__
    for (; i + 4 <= count; i += 4) {
        __builtin_prefetch(&groups[i + 32], 0, 2);
        __builtin_prefetch(&values[i + 32], 0, 2);

        uint32_t g0 = groups[i], g1 = groups[i + 1], g2 = groups[i + 2], g3 = groups[i + 3];
        int32_t v0 = values[i], v1 = values[i + 1], v2 = values[i + 2], v3 = values[i + 3];

        if (g0 < num_groups && v0 > out_maxs[g0]) out_maxs[g0] = v0;
        if (g1 < num_groups && v1 > out_maxs[g1]) out_maxs[g1] = v1;
        if (g2 < num_groups && v2 > out_maxs[g2]) out_maxs[g2] = v2;
        if (g3 < num_groups && v3 > out_maxs[g3]) out_maxs[g3] = v3;
    }
#endif

    for (; i < count; ++i) {
        uint32_t gid = groups[i];
        if (gid < num_groups && values[i] > out_maxs[gid]) {
            out_maxs[gid] = values[i];
        }
    }
}

} // namespace aggregate
} // namespace thunderduck
