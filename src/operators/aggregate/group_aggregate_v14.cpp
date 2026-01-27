/**
 * ThunderDuck - GROUP BY Aggregation v14.0 Implementation
 *
 * V14 策略：基于基准测试结果，优化多线程并行实现
 *
 * 测试结论：
 * - 寄存器缓冲在低基数场景下反而更慢（O(n) 查找 + 频繁 flush）
 * - V4 Parallel 的多线程并行已是最优基础
 * - V14 专注于 SIMD 合并优化
 *
 * V14 = V4 Parallel + SIMD 合并 + 更好的线程调度
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

namespace {

constexpr size_t MIN_ELEMENTS_PER_THREAD = 100000;
constexpr size_t MAX_THREADS = 4;
constexpr size_t PREFETCH_DISTANCE = 32;

const char* V14_VERSION = "V14.0 - SIMD 合并 + 优化并行";

} // anonymous namespace

// ============================================================================
// V14 并行分组求和 (SIMD 合并优化)
// ============================================================================

void group_sum_i32_v14_parallel(const int32_t* values, const uint32_t* groups,
                                 size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    std::memset(out_sums, 0, num_groups * sizeof(int64_t));

    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD) {
        // 单线程：使用 V4 的 4 路展开 + 预取
        size_t i = 0;
#ifdef __aarch64__
        for (; i + 4 <= count; i += 4) {
            __builtin_prefetch(&groups[i + PREFETCH_DISTANCE], 0, 2);
            __builtin_prefetch(&values[i + PREFETCH_DISTANCE], 0, 2);

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
        for (; i < count; ++i) {
            uint32_t gid = groups[i];
            if (gid < num_groups) out_sums[gid] += values[i];
        }
        return;
    }

    // 每线程局部累加
    std::vector<std::vector<int64_t>> local_sums(num_threads);
    for (auto& ls : local_sums) {
        ls.resize(num_groups, 0);
    }

    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int64_t* local = local_sums[t].data();

            size_t i = start;
#ifdef __aarch64__
            for (; i + 4 <= end; i += 4) {
                __builtin_prefetch(&groups[i + PREFETCH_DISTANCE], 0, 2);
                __builtin_prefetch(&values[i + PREFETCH_DISTANCE], 0, 2);

                uint32_t g0 = groups[i];
                uint32_t g1 = groups[i + 1];
                uint32_t g2 = groups[i + 2];
                uint32_t g3 = groups[i + 3];

                if (g0 < num_groups) local[g0] += values[i];
                if (g1 < num_groups) local[g1] += values[i + 1];
                if (g2 < num_groups) local[g2] += values[i + 2];
                if (g3 < num_groups) local[g3] += values[i + 3];
            }
#endif
            for (; i < end; ++i) {
                uint32_t gid = groups[i];
                if (gid < num_groups) local[gid] += values[i];
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // SIMD 合并结果
#ifdef __aarch64__
    for (size_t g = 0; g + 2 <= num_groups; g += 2) {
        int64x2_t sum = vdupq_n_s64(0);
        for (size_t t = 0; t < num_threads; ++t) {
            int64x2_t local = vld1q_s64(&local_sums[t][g]);
            sum = vaddq_s64(sum, local);
        }
        vst1q_s64(&out_sums[g], sum);
    }
    for (size_t g = (num_groups / 2) * 2; g < num_groups; ++g) {
        for (size_t t = 0; t < num_threads; ++t) {
            out_sums[g] += local_sums[t][g];
        }
    }
#else
    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t g = 0; g < num_groups; ++g) {
            out_sums[g] += local_sums[t][g];
        }
    }
#endif
}

// ============================================================================
// V14 公开接口（委托给优化的并行实现）
// ============================================================================

void group_sum_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    group_sum_i32_v14_parallel(values, groups, count, num_groups, out_sums);
}

// ============================================================================
// V14 分组计数
// ============================================================================

void group_count_v14(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    std::memset(out_counts, 0, num_groups * sizeof(size_t));

    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD) {
        size_t i = 0;
#ifdef __aarch64__
        for (; i + 4 <= count; i += 4) {
            __builtin_prefetch(&groups[i + PREFETCH_DISTANCE], 0, 2);

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
            if (gid < num_groups) out_counts[gid]++;
        }
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
                if (gid < num_groups) local[gid]++;
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
// V14 分组 MIN/MAX
// ============================================================================

void group_min_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins) {
    // 委托给 V4 实现（已经是最优）
    group_min_i32_v4(values, groups, count, num_groups, out_mins);
}

void group_max_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs) {
    // 委托给 V4 实现（已经是最优）
    group_max_i32_v4(values, groups, count, num_groups, out_maxs);
}

const char* get_group_aggregate_v14_version() {
    return V14_VERSION;
}

} // namespace aggregate
} // namespace thunderduck
