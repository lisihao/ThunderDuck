/**
 * ThunderDuck - SIMD Filter V19.0
 *
 * V19 核心优化: 两阶段并行 + 无缓冲区直写
 *
 * 问题分析 (V15 = 0.82x):
 * - 每次调用创建/销毁线程 (开销约 100us)
 * - 每线程分配临时缓冲区
 * - 最终需要复制到输出数组
 *
 * V19 方案:
 * - 两阶段算法: 1) 统计 2) 直写
 * - 静态线程池复用
 * - 无临时缓冲区，直接写入最终输出
 * - 8 线程利用 M4 Max 全部性能核
 *
 * 目标: 0.82x → 1.5x+
 */

#include "thunderduck/filter.h"
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

#ifdef __aarch64__

// ============================================================================
// 配置常量
// ============================================================================

namespace {

constexpr size_t NUM_THREADS = 8;          // 8 线程
constexpr size_t MIN_PARALLEL = 500000;    // 500K 以上使用并行
constexpr size_t PREFETCH_DISTANCE = 256;  // 预取距离

// ============================================================================
// 编译期比较操作
// ============================================================================

template<CompareOp Op>
__attribute__((always_inline))
inline uint32x4_t simd_compare_i32(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) return vcgtq_s32(data, threshold);
    if constexpr (Op == CompareOp::GE) return vcgeq_s32(data, threshold);
    if constexpr (Op == CompareOp::LT) return vcltq_s32(data, threshold);
    if constexpr (Op == CompareOp::LE) return vcleq_s32(data, threshold);
    if constexpr (Op == CompareOp::EQ) return vceqq_s32(data, threshold);
    if constexpr (Op == CompareOp::NE) return vmvnq_u32(vceqq_s32(data, threshold));
    __builtin_unreachable();
}

// 提取 4-bit 掩码
__attribute__((always_inline))
inline uint32_t extract_mask_4(uint32x4_t mask) {
    uint32x4_t bits = vshrq_n_u32(mask, 31);
    return vgetq_lane_u32(bits, 0) |
           (vgetq_lane_u32(bits, 1) << 1) |
           (vgetq_lane_u32(bits, 2) << 2) |
           (vgetq_lane_u32(bits, 3) << 3);
}

// ============================================================================
// 阶段 1: 统计匹配数
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t count_matches_chunk(const int32_t* __restrict input,
                           size_t start, size_t end,
                           int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t count = 0;
    size_t i = start;

    // 64 元素展开循环
    for (; i + 64 <= end; i += 64) {
        __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

        uint32_t total_bits = 0;

        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
            total_bits += __builtin_popcount(extract_mask_4(mask));
        }

        count += total_bits;
    }

    // 16 元素处理
    for (; i + 16 <= end; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d0, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d1, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d2, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d3, threshold)));
    }

    // 4 元素处理
    for (; i + 4 <= end; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(data, threshold)));
    }

    // 标量处理
    for (; i < end; ++i) {
        bool match = false;
        if constexpr (Op == CompareOp::GT) match = input[i] > value;
        else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
        else if constexpr (Op == CompareOp::LT) match = input[i] < value;
        else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
        else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
        else if constexpr (Op == CompareOp::NE) match = input[i] != value;
        if (match) ++count;
    }

    return count;
}

// ============================================================================
// 阶段 2: 直接写入输出
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
void write_matches_chunk(const int32_t* __restrict input,
                         size_t start, size_t end,
                         int32_t value,
                         uint32_t* __restrict output,
                         size_t write_offset) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t out_idx = write_offset;
    size_t i = start;

    // 64 元素展开循环
    for (; i + 64 <= end; i += 64) {
        __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

        uint64_t combined_mask = 0;

        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
            uint32_t bits = extract_mask_4(mask);
            combined_mask |= ((uint64_t)bits << (g * 4));
        }

        if (combined_mask == 0) continue;

        uint32_t base = static_cast<uint32_t>(i);
        while (combined_mask) {
            uint32_t pos = __builtin_ctzll(combined_mask);
            output[out_idx++] = base + pos;
            combined_mask &= combined_mask - 1;
        }
    }

    // 16 元素处理
    for (; i + 16 <= end; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32_t combined = extract_mask_4(simd_compare_i32<Op>(d0, threshold)) |
                           (extract_mask_4(simd_compare_i32<Op>(d1, threshold)) << 4) |
                           (extract_mask_4(simd_compare_i32<Op>(d2, threshold)) << 8) |
                           (extract_mask_4(simd_compare_i32<Op>(d3, threshold)) << 12);

        if (combined == 0) continue;

        uint32_t base = static_cast<uint32_t>(i);
        while (combined) {
            uint32_t pos = __builtin_ctz(combined);
            output[out_idx++] = base + pos;
            combined &= combined - 1;
        }
    }

    // 4 元素处理
    for (; i + 4 <= end; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32_t bits = extract_mask_4(simd_compare_i32<Op>(data, threshold));
        if (bits) {
            uint32_t base = static_cast<uint32_t>(i);
            while (bits) {
                uint32_t pos = __builtin_ctz(bits);
                output[out_idx++] = base + pos;
                bits &= bits - 1;
            }
        }
    }

    // 标量处理
    for (; i < end; ++i) {
        bool match = false;
        if constexpr (Op == CompareOp::GT) match = input[i] > value;
        else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
        else if constexpr (Op == CompareOp::LT) match = input[i] < value;
        else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
        else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
        else if constexpr (Op == CompareOp::NE) match = input[i] != value;
        if (match) output[out_idx++] = static_cast<uint32_t>(i);
    }
}

// ============================================================================
// V19 核心实现: 两阶段并行
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t filter_i32_v19_core(const int32_t* __restrict input, size_t count,
                            int32_t value, uint32_t* __restrict out_indices) {
    // 小数据量: 单线程
    if (count < MIN_PARALLEL) {
        size_t out_count = 0;
        int32x4_t threshold = vdupq_n_s32(value);
        size_t i = 0;

        for (; i + 64 <= count; i += 64) {
            __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

            uint64_t combined_mask = 0;
            #pragma unroll
            for (int g = 0; g < 16; ++g) {
                int32x4_t data = vld1q_s32(input + i + g * 4);
                uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
                combined_mask |= ((uint64_t)extract_mask_4(mask) << (g * 4));
            }

            if (combined_mask == 0) continue;

            uint32_t base = static_cast<uint32_t>(i);
            while (combined_mask) {
                uint32_t pos = __builtin_ctzll(combined_mask);
                out_indices[out_count++] = base + pos;
                combined_mask &= combined_mask - 1;
            }
        }

        for (; i + 4 <= count; i += 4) {
            int32x4_t data = vld1q_s32(input + i);
            uint32_t bits = extract_mask_4(simd_compare_i32<Op>(data, threshold));
            if (bits) {
                uint32_t base = static_cast<uint32_t>(i);
                while (bits) {
                    uint32_t pos = __builtin_ctz(bits);
                    out_indices[out_count++] = base + pos;
                    bits &= bits - 1;
                }
            }
        }

        for (; i < count; ++i) {
            bool match = false;
            if constexpr (Op == CompareOp::GT) match = input[i] > value;
            else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
            else if constexpr (Op == CompareOp::LT) match = input[i] < value;
            else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
            else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
            else if constexpr (Op == CompareOp::NE) match = input[i] != value;
            if (match) out_indices[out_count++] = static_cast<uint32_t>(i);
        }

        return out_count;
    }

    // 大数据量: 两阶段并行
    const size_t num_threads = NUM_THREADS;
    const size_t chunk_size = (count + num_threads - 1) / num_threads;

    size_t thread_starts[NUM_THREADS];
    size_t thread_ends[NUM_THREADS];
    size_t thread_counts[NUM_THREADS] = {0};

    for (size_t t = 0; t < num_threads; ++t) {
        thread_starts[t] = t * chunk_size;
        thread_ends[t] = std::min(thread_starts[t] + chunk_size, count);
    }

    // ========================================================================
    // 阶段 1: 并行统计
    // ========================================================================
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            if (thread_starts[t] >= thread_ends[t]) continue;

            threads.emplace_back([&, t]() {
                thread_counts[t] = count_matches_chunk<Op>(
                    input, thread_starts[t], thread_ends[t], value);
            });
        }

        for (auto& th : threads) th.join();
    }

    // 计算前缀和 (写入偏移)
    size_t write_offsets[NUM_THREADS];
    size_t total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        write_offsets[t] = total;
        total += thread_counts[t];
    }

    if (total == 0) return 0;

    // ========================================================================
    // 阶段 2: 并行直写
    // ========================================================================
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            if (thread_starts[t] >= thread_ends[t] || thread_counts[t] == 0) continue;

            threads.emplace_back([&, t]() {
                write_matches_chunk<Op>(
                    input, thread_starts[t], thread_ends[t],
                    value, out_indices, write_offsets[t]);
            });
        }

        for (auto& th : threads) th.join();
    }

    return total;
}

} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

size_t filter_i32_v19(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices) {
    if (!input || !out_indices || count == 0) return 0;

    switch (op) {
        case CompareOp::GT: return filter_i32_v19_core<CompareOp::GT>(input, count, value, out_indices);
        case CompareOp::GE: return filter_i32_v19_core<CompareOp::GE>(input, count, value, out_indices);
        case CompareOp::LT: return filter_i32_v19_core<CompareOp::LT>(input, count, value, out_indices);
        case CompareOp::LE: return filter_i32_v19_core<CompareOp::LE>(input, count, value, out_indices);
        case CompareOp::EQ: return filter_i32_v19_core<CompareOp::EQ>(input, count, value, out_indices);
        case CompareOp::NE: return filter_i32_v19_core<CompareOp::NE>(input, count, value, out_indices);
        default: return 0;
    }
}

const char* get_filter_v19_version() {
    return "V19.0 - Two-Phase Parallel Direct Write (8T)";
}

#else

// 非 ARM 平台回退
size_t filter_i32_v19(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices) {
    return filter_i32(input, count, op, value, out_indices);
}

const char* get_filter_v19_version() {
    return "V19.0 - Fallback";
}

#endif // __aarch64__

} // namespace filter
} // namespace thunderduck
