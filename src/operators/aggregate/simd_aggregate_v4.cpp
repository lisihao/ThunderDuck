/**
 * ThunderDuck - SIMD Aggregation Implementation v4.0
 *
 * V9 性能优化版本：
 * - P0: 向量化哈希分组 (分区+顺序累加)
 * - P1: 预取距离优化 (64B → 256B)
 * - P2: 缓存分块 (L2 友好)
 * - P3: 多线程分组聚合
 */

#include "thunderduck/aggregate.h"
#include <limits>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thread>
#include <atomic>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 常量定义
// ============================================================================

// M4 缓存参数
constexpr size_t CACHE_LINE_SIZE = 128;      // M4 L1 cache line
constexpr size_t L2_CACHE_SIZE = 12 * 1024 * 1024;  // ~12MB L2
constexpr size_t CACHE_BLOCK_SIZE = L2_CACHE_SIZE / 4;  // 3MB 块

// 预取参数
constexpr size_t PREFETCH_DISTANCE = 256;    // 256B = 2 cache lines (v4优化)
constexpr size_t PREFETCH_DISTANCE_GROUP = 512;  // 分组聚合用更大预取

// 多线程参数
constexpr size_t MIN_ELEMENTS_PER_THREAD = 100000;  // 最小每线程元素数
constexpr size_t MAX_THREADS = 4;  // M4 性能核数量

// 分组聚合阈值
constexpr size_t GROUP_PARTITION_THRESHOLD = 10000;  // 分区优化阈值

// ============================================================================
// P1: 预取距离优化 - SUM v4
// ============================================================================

int64_t sum_i32_v4(const int32_t* input, size_t count) {
    if (count == 0 || !input) return 0;

#ifdef __aarch64__
    // 4个 int64 累加器
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int64x2_t sum2 = vdupq_n_s64(0);
    int64x2_t sum3 = vdupq_n_s64(0);
    size_t i = 0;

    // P1: 更激进的预取 - 256B (vs v2的64B)
    // 主循环：每次处理 32 个元素 (vs v2的16个)
    for (; i + 32 <= count; i += 32) {
        // 预取 256B ahead (64 elements = 256 bytes)
        __builtin_prefetch(input + i + 64, 0, 3);  // L1
        __builtin_prefetch(input + i + 128, 0, 2); // L2

        // 加载 32 个 int32
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);
        int32x4_t d4 = vld1q_s32(input + i + 16);
        int32x4_t d5 = vld1q_s32(input + i + 20);
        int32x4_t d6 = vld1q_s32(input + i + 24);
        int32x4_t d7 = vld1q_s32(input + i + 28);

        // 扩展到 64 位并累加 (pairwise add long)
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d0));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d1));
        sum2 = vaddq_s64(sum2, vpaddlq_s32(d2));
        sum3 = vaddq_s64(sum3, vpaddlq_s32(d3));
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d4));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d5));
        sum2 = vaddq_s64(sum2, vpaddlq_s32(d6));
        sum3 = vaddq_s64(sum3, vpaddlq_s32(d7));
    }

    // 处理剩余 16 元素块
    for (; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        sum0 = vaddq_s64(sum0, vpaddlq_s32(d0));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d1));
        sum2 = vaddq_s64(sum2, vpaddlq_s32(d2));
        sum3 = vaddq_s64(sum3, vpaddlq_s32(d3));
    }

    // 合并累加器
    int64x2_t sum_01 = vaddq_s64(sum0, sum1);
    int64x2_t sum_23 = vaddq_s64(sum2, sum3);
    int64x2_t sum_all = vaddq_s64(sum_01, sum_23);

    int64_t result = vaddvq_s64(sum_all);

    // 标量处理剩余
    for (; i < count; ++i) {
        result += input[i];
    }

    return result;
#else
    int64_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        result += input[i];
    }
    return result;
#endif
}

// ============================================================================
// P2: 缓存分块 SUM
// ============================================================================

int64_t sum_i32_v4_blocked(const int32_t* input, size_t count) {
    if (count == 0 || !input) return 0;

    // 计算每块元素数 (3MB / 4B = 768K 元素)
    constexpr size_t ELEMENTS_PER_BLOCK = CACHE_BLOCK_SIZE / sizeof(int32_t);

    int64_t total = 0;
    size_t processed = 0;

    while (processed < count) {
        size_t block_size = std::min(ELEMENTS_PER_BLOCK, count - processed);
        total += sum_i32_v4(input + processed, block_size);
        processed += block_size;
    }

    return total;
}

// ============================================================================
// P1 + P2: MIN/MAX v4 (预取优化 + 合并)
// ============================================================================

void minmax_i32_v4(const int32_t* input, size_t count,
                   int32_t* out_min, int32_t* out_max) {
    if (count == 0) {
        *out_min = std::numeric_limits<int32_t>::max();
        *out_max = std::numeric_limits<int32_t>::min();
        return;
    }

#ifdef __aarch64__
    int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
    int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
    size_t i = 0;

    // 主循环：每次处理 32 个元素
    for (; i + 32 <= count; i += 32) {
        // P1: 激进预取
        __builtin_prefetch(input + i + 64, 0, 3);
        __builtin_prefetch(input + i + 128, 0, 2);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);
        int32x4_t d4 = vld1q_s32(input + i + 16);
        int32x4_t d5 = vld1q_s32(input + i + 20);
        int32x4_t d6 = vld1q_s32(input + i + 24);
        int32x4_t d7 = vld1q_s32(input + i + 28);

        // 分层合并 min/max
        int32x4_t min_01 = vminq_s32(d0, d1);
        int32x4_t min_23 = vminq_s32(d2, d3);
        int32x4_t min_45 = vminq_s32(d4, d5);
        int32x4_t min_67 = vminq_s32(d6, d7);

        int32x4_t max_01 = vmaxq_s32(d0, d1);
        int32x4_t max_23 = vmaxq_s32(d2, d3);
        int32x4_t max_45 = vmaxq_s32(d4, d5);
        int32x4_t max_67 = vmaxq_s32(d6, d7);

        int32x4_t min_0123 = vminq_s32(min_01, min_23);
        int32x4_t min_4567 = vminq_s32(min_45, min_67);
        int32x4_t max_0123 = vmaxq_s32(max_01, max_23);
        int32x4_t max_4567 = vmaxq_s32(max_45, max_67);

        int32x4_t min_batch = vminq_s32(min_0123, min_4567);
        int32x4_t max_batch = vmaxq_s32(max_0123, max_4567);

        min_vec = vminq_s32(min_vec, min_batch);
        max_vec = vmaxq_s32(max_vec, max_batch);
    }

    // 处理剩余 4 元素块
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        min_vec = vminq_s32(min_vec, data);
        max_vec = vmaxq_s32(max_vec, data);
    }

    *out_min = vminvq_s32(min_vec);
    *out_max = vmaxvq_s32(max_vec);

    // 标量处理剩余
    for (; i < count; ++i) {
        if (input[i] < *out_min) *out_min = input[i];
        if (input[i] > *out_max) *out_max = input[i];
    }
#else
    *out_min = input[0];
    *out_max = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] < *out_min) *out_min = input[i];
        if (input[i] > *out_max) *out_max = input[i];
    }
#endif
}

// ============================================================================
// P1 + P2: 融合统计量 v4
// ============================================================================

AggregateStats aggregate_all_i32_v4(const int32_t* input, size_t count) {
    AggregateStats stats = {0, 0, 0, 0};
    if (count == 0) return stats;

    stats.count = static_cast<int64_t>(count);

#ifdef __aarch64__
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
    int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
    size_t i = 0;

    // 主循环：每次处理 16 个元素
    for (; i + 16 <= count; i += 16) {
        // P1: 激进预取
        __builtin_prefetch(input + i + 64, 0, 3);
        __builtin_prefetch(input + i + 128, 0, 2);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // SUM (扩展到 64 位)
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d0));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d1));
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d2));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d3));

        // MIN/MAX
        int32x4_t batch_min = vminq_s32(vminq_s32(d0, d1), vminq_s32(d2, d3));
        int32x4_t batch_max = vmaxq_s32(vmaxq_s32(d0, d1), vmaxq_s32(d2, d3));
        min_vec = vminq_s32(min_vec, batch_min);
        max_vec = vmaxq_s32(max_vec, batch_max);
    }

    stats.sum = vaddvq_s64(vaddq_s64(sum0, sum1));
    stats.min_val = vminvq_s32(min_vec);
    stats.max_val = vmaxvq_s32(max_vec);

    // 标量处理剩余
    for (; i < count; ++i) {
        stats.sum += input[i];
        if (input[i] < stats.min_val) stats.min_val = input[i];
        if (input[i] > stats.max_val) stats.max_val = input[i];
    }
#else
    stats.min_val = input[0];
    stats.max_val = input[0];
    for (size_t i = 0; i < count; ++i) {
        stats.sum += input[i];
        if (input[i] < stats.min_val) stats.min_val = input[i];
        if (input[i] > stats.max_val) stats.max_val = input[i];
    }
#endif

    return stats;
}

// ============================================================================
// P0: 分组聚合优化
// ============================================================================
// 注: 分区优化在测试中表现不佳 (开销 > 收益)
// V9 改用: 展开循环 + 预取 + 多线程并行

void group_sum_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    // 初始化输出
    std::memset(out_sums, 0, num_groups * sizeof(int64_t));

    // V9 优化: 使用展开循环 + 预取
    // 分区优化开销过大,直接使用优化的标量循环
    size_t i = 0;

#ifdef __aarch64__
    // 4路展开 + 预取
    for (; i + 4 <= count; i += 4) {
        // 预取
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
// P3: 多线程分组聚合
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

    // V9 优化: 展开循环 + 预取
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
// P0 + P3: 分组 MIN/MAX v4
// ============================================================================

void group_min_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int32_t* out_mins) {
    if (count == 0 || !values || !groups || !out_mins) return;

    std::fill(out_mins, out_mins + num_groups, std::numeric_limits<int32_t>::max());

    // V9 优化: 展开循环 + 预取
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

    // V9 优化: 展开循环 + 预取
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
