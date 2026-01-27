/**
 * ThunderDuck - Parallel SIMD Aggregate Implementation
 *
 * 多线程并行 Aggregate 优化:
 * - 4 线程并行 (M4 性能核)
 * - 每线程独立 SIMD 累加
 * - 最终归约合并
 * - 可选 vDSP 加速
 *
 * 目标: 10M 数据 3.3x → 5-6x
 */

#include "thunderduck/aggregate.h"
#include "thunderduck/memory.h"
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

constexpr size_t MIN_ELEMENTS_PER_THREAD = 500000;  // 500K
constexpr size_t MAX_THREADS = 4;                   // M4 性能核
constexpr size_t VDSP_THRESHOLD = 1000000;          // 1M 元素使用 vDSP

} // anonymous namespace

// ============================================================================
// vDSP 加速的 SUM (单线程)
// ============================================================================

#ifdef __APPLE__
int64_t sum_i32_vdsp(const int32_t* input, size_t count) {
    if (count == 0) return 0;

    // vDSP 没有直接的 int32 sum，需要转换为 float
    // 对于大数据，转换开销可能不值得
    // 这里使用 vDSP_sve 来加速 float sum

    // 方案: 使用 SIMD 多线程代替
    return sum_i32_v2(input, count);
}
#endif

// ============================================================================
// 多线程并行 SUM
// ============================================================================

int64_t sum_i32_parallel(const int32_t* input, size_t count) {
    if (count == 0 || !input) return 0;

    // 计算线程数
    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    // 小数据量：单线程
    if (num_threads <= 1) {
        return sum_i32_v2(input, count);
    }

#ifdef __aarch64__
    // 每线程的局部结果
    std::vector<int64_t> local_sums(num_threads, 0);

    // 计算每线程的数据范围
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动工作线程
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int64x2_t sum_vec = vdupq_n_s64(0);
            size_t i = start;

            // 16 元素展开
            for (; i + 16 <= end; i += 16) {
                __builtin_prefetch(&input[i + 64], 0, 0);

                int32x4_t d0 = vld1q_s32(&input[i]);
                int32x4_t d1 = vld1q_s32(&input[i + 4]);
                int32x4_t d2 = vld1q_s32(&input[i + 8]);
                int32x4_t d3 = vld1q_s32(&input[i + 12]);

                // 扩展到 64-bit 并累加
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d0));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d1));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d2));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d3));
            }

            // 4 元素块
            for (; i + 4 <= end; i += 4) {
                int32x4_t d = vld1q_s32(&input[i]);
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d));
            }

            // 水平归约
            int64_t partial = vaddvq_s64(sum_vec);

            // 标量处理剩余
            for (; i < end; ++i) {
                partial += input[i];
            }

            local_sums[t] = partial;
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
    int64_t total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        total += local_sums[t];
    }

    return total;
#else
    return sum_i32_v2(input, count);
#endif
}

// ============================================================================
// 多线程并行 MIN/MAX
// ============================================================================

void minmax_i32_parallel(const int32_t* input, size_t count,
                          int32_t* out_min, int32_t* out_max) {
    if (count == 0 || !input || !out_min || !out_max) {
        if (out_min) *out_min = std::numeric_limits<int32_t>::max();
        if (out_max) *out_max = std::numeric_limits<int32_t>::min();
        return;
    }

    // 计算线程数
    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    // 小数据量：单线程
    if (num_threads <= 1) {
        minmax_i32(input, count, out_min, out_max);
        return;
    }

#ifdef __aarch64__
    // 每线程的局部结果
    std::vector<int32_t> local_mins(num_threads, std::numeric_limits<int32_t>::max());
    std::vector<int32_t> local_maxs(num_threads, std::numeric_limits<int32_t>::min());

    // 计算每线程的数据范围
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动工作线程
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
            int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
            size_t i = start;

            // 16 元素展开
            for (; i + 16 <= end; i += 16) {
                __builtin_prefetch(&input[i + 64], 0, 0);

                int32x4_t d0 = vld1q_s32(&input[i]);
                int32x4_t d1 = vld1q_s32(&input[i + 4]);
                int32x4_t d2 = vld1q_s32(&input[i + 8]);
                int32x4_t d3 = vld1q_s32(&input[i + 12]);

                // 合并 4 个向量
                int32x4_t min_01 = vminq_s32(d0, d1);
                int32x4_t min_23 = vminq_s32(d2, d3);
                int32x4_t min_batch = vminq_s32(min_01, min_23);

                int32x4_t max_01 = vmaxq_s32(d0, d1);
                int32x4_t max_23 = vmaxq_s32(d2, d3);
                int32x4_t max_batch = vmaxq_s32(max_01, max_23);

                min_vec = vminq_s32(min_vec, min_batch);
                max_vec = vmaxq_s32(max_vec, max_batch);
            }

            // 4 元素块
            for (; i + 4 <= end; i += 4) {
                int32x4_t d = vld1q_s32(&input[i]);
                min_vec = vminq_s32(min_vec, d);
                max_vec = vmaxq_s32(max_vec, d);
            }

            // 水平归约
            local_mins[t] = vminvq_s32(min_vec);
            local_maxs[t] = vmaxvq_s32(max_vec);

            // 标量处理剩余
            for (; i < end; ++i) {
                if (input[i] < local_mins[t]) local_mins[t] = input[i];
                if (input[i] > local_maxs[t]) local_maxs[t] = input[i];
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
    *out_min = local_mins[0];
    *out_max = local_maxs[0];
    for (size_t t = 1; t < num_threads; ++t) {
        if (local_mins[t] < *out_min) *out_min = local_mins[t];
        if (local_maxs[t] > *out_max) *out_max = local_maxs[t];
    }
#else
    minmax_i32(input, count, out_min, out_max);
#endif
}

// ============================================================================
// 多线程并行 COUNT (非零/满足条件)
// ============================================================================

size_t count_nonzero_i32_parallel(const int32_t* input, size_t count) {
    if (count == 0 || !input) return 0;

    // 计算线程数
    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    if (num_threads <= 1) {
        // 单线程 SIMD
        size_t result = 0;
#ifdef __aarch64__
        int32x4_t zero = vdupq_n_s32(0);
        size_t i = 0;

        for (; i + 16 <= count; i += 16) {
            int32x4_t d0 = vld1q_s32(&input[i]);
            int32x4_t d1 = vld1q_s32(&input[i + 4]);
            int32x4_t d2 = vld1q_s32(&input[i + 8]);
            int32x4_t d3 = vld1q_s32(&input[i + 12]);

            uint32x4_t nz0 = vmvnq_u32(vceqq_s32(d0, zero));
            uint32x4_t nz1 = vmvnq_u32(vceqq_s32(d1, zero));
            uint32x4_t nz2 = vmvnq_u32(vceqq_s32(d2, zero));
            uint32x4_t nz3 = vmvnq_u32(vceqq_s32(d3, zero));

            // 计数：每个 0xFFFFFFFF 代表 1
            result += vaddvq_u32(vshrq_n_u32(nz0, 31));
            result += vaddvq_u32(vshrq_n_u32(nz1, 31));
            result += vaddvq_u32(vshrq_n_u32(nz2, 31));
            result += vaddvq_u32(vshrq_n_u32(nz3, 31));
        }

        for (; i < count; ++i) {
            if (input[i] != 0) ++result;
        }
#else
        for (size_t i = 0; i < count; ++i) {
            if (input[i] != 0) ++result;
        }
#endif
        return result;
    }

#ifdef __aarch64__
    // 多线程版本
    std::vector<size_t> local_counts(num_threads, 0);
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int32x4_t zero = vdupq_n_s32(0);
            size_t local = 0;
            size_t i = start;

            for (; i + 16 <= end; i += 16) {
                __builtin_prefetch(&input[i + 64], 0, 0);

                int32x4_t d0 = vld1q_s32(&input[i]);
                int32x4_t d1 = vld1q_s32(&input[i + 4]);
                int32x4_t d2 = vld1q_s32(&input[i + 8]);
                int32x4_t d3 = vld1q_s32(&input[i + 12]);

                uint32x4_t nz0 = vmvnq_u32(vceqq_s32(d0, zero));
                uint32x4_t nz1 = vmvnq_u32(vceqq_s32(d1, zero));
                uint32x4_t nz2 = vmvnq_u32(vceqq_s32(d2, zero));
                uint32x4_t nz3 = vmvnq_u32(vceqq_s32(d3, zero));

                local += vaddvq_u32(vshrq_n_u32(nz0, 31));
                local += vaddvq_u32(vshrq_n_u32(nz1, 31));
                local += vaddvq_u32(vshrq_n_u32(nz2, 31));
                local += vaddvq_u32(vshrq_n_u32(nz3, 31));
            }

            for (; i < end; ++i) {
                if (input[i] != 0) ++local;
            }

            local_counts[t] = local;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    size_t total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        total += local_counts[t];
    }

    return total;
#else
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (input[i] != 0) ++result;
    }
    return result;
#endif
}

// ============================================================================
// 融合统计量并行版本
// ============================================================================

AggregateStats aggregate_all_i32_parallel(const int32_t* input, size_t count) {
    AggregateStats stats = {0, 0, 0, 0};

    if (count == 0 || !input) {
        stats.min_val = std::numeric_limits<int32_t>::max();
        stats.max_val = std::numeric_limits<int32_t>::min();
        return stats;
    }

    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    if (num_threads <= 1) {
        return aggregate_all_i32(input, count);
    }

#ifdef __aarch64__
    struct LocalStats {
        int64_t sum = 0;
        int32_t min = std::numeric_limits<int32_t>::max();
        int32_t max = std::numeric_limits<int32_t>::min();
    };

    std::vector<LocalStats> local_stats(num_threads);
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int64x2_t sum_vec = vdupq_n_s64(0);
            int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
            int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
            size_t i = start;

            for (; i + 16 <= end; i += 16) {
                __builtin_prefetch(&input[i + 64], 0, 0);

                int32x4_t d0 = vld1q_s32(&input[i]);
                int32x4_t d1 = vld1q_s32(&input[i + 4]);
                int32x4_t d2 = vld1q_s32(&input[i + 8]);
                int32x4_t d3 = vld1q_s32(&input[i + 12]);

                // SUM
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d0));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d1));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d2));
                sum_vec = vaddq_s64(sum_vec, vpaddlq_s32(d3));

                // MIN/MAX
                int32x4_t min_01 = vminq_s32(d0, d1);
                int32x4_t min_23 = vminq_s32(d2, d3);
                min_vec = vminq_s32(min_vec, vminq_s32(min_01, min_23));

                int32x4_t max_01 = vmaxq_s32(d0, d1);
                int32x4_t max_23 = vmaxq_s32(d2, d3);
                max_vec = vmaxq_s32(max_vec, vmaxq_s32(max_01, max_23));
            }

            // 水平归约
            local_stats[t].sum = vaddvq_s64(sum_vec);
            local_stats[t].min = vminvq_s32(min_vec);
            local_stats[t].max = vmaxvq_s32(max_vec);

            // 处理剩余
            for (; i < end; ++i) {
                local_stats[t].sum += input[i];
                if (input[i] < local_stats[t].min) local_stats[t].min = input[i];
                if (input[i] > local_stats[t].max) local_stats[t].max = input[i];
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 合并
    stats.sum = local_stats[0].sum;
    stats.min_val = local_stats[0].min;
    stats.max_val = local_stats[0].max;
    stats.count = count;

    for (size_t t = 1; t < num_threads; ++t) {
        stats.sum += local_stats[t].sum;
        if (local_stats[t].min < stats.min_val) stats.min_val = local_stats[t].min;
        if (local_stats[t].max > stats.max_val) stats.max_val = local_stats[t].max;
    }

    return stats;
#else
    return aggregate_all_i32(input, count);
#endif
}

} // namespace aggregate
} // namespace thunderduck
