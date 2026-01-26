/**
 * ThunderDuck - SIMD Filter Implementation v6.0
 *
 * V9.1 预取优化版本:
 * - 多级预取 (L1/L2 分离)
 * - 更激进预取距离 (256B → 512B)
 * - 基于 v3 优化架构
 */

#include "thunderduck/filter.h"
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

#ifdef __aarch64__

// ============================================================================
// V9.1 预取优化常量
// ============================================================================

// M4 预取参数优化
constexpr size_t PREFETCH_L1_DISTANCE = 256;   // L1: 2 cache lines (256B)
constexpr size_t PREFETCH_L2_DISTANCE = 512;   // L2: 4 cache lines (512B)
constexpr size_t BATCH_SIZE = 256;             // 每批处理 256 个元素
constexpr size_t INNER_ITERS = 16;             // 内层循环次数 (256 / 16)

// ============================================================================
// 编译期比较操作 - 模板特化
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

template<CompareOp Op>
__attribute__((always_inline))
inline bool scalar_compare_i32(int32_t data, int32_t value) {
    if constexpr (Op == CompareOp::GT) return data > value;
    if constexpr (Op == CompareOp::GE) return data >= value;
    if constexpr (Op == CompareOp::LT) return data < value;
    if constexpr (Op == CompareOp::LE) return data <= value;
    if constexpr (Op == CompareOp::EQ) return data == value;
    if constexpr (Op == CompareOp::NE) return data != value;
    __builtin_unreachable();
}

// ============================================================================
// v6.0 核心计数实现 - 多级预取优化
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t count_i32_v6_core(const int32_t* __restrict input,
                          size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t result = 0;
    size_t i = 0;

    // 计算预取元素偏移 (字节 / sizeof(int32_t))
    constexpr size_t L1_OFFSET = PREFETCH_L1_DISTANCE / sizeof(int32_t);  // 64 元素
    constexpr size_t L2_OFFSET = PREFETCH_L2_DISTANCE / sizeof(int32_t);  // 128 元素

    // ========================================================================
    // 阶段 1: 256 元素批次处理 + 多级预取
    // ========================================================================
    for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const int32_t* batch_ptr = input + i;

        // V9.1: 多级预取策略
        // L1 预取: 近距离,高优先级
        __builtin_prefetch(batch_ptr + BATCH_SIZE, 0, 3);           // 下一批开始
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 32, 0, 3);      // +128B
        // L2 预取: 远距离,中优先级
        __builtin_prefetch(batch_ptr + BATCH_SIZE + L1_OFFSET, 0, 2);  // +256B
        __builtin_prefetch(batch_ptr + BATCH_SIZE + L2_OFFSET, 0, 2);  // +512B

        #pragma unroll
        for (size_t j = 0; j < INNER_ITERS; ++j) {
            const int32_t* ptr = batch_ptr + j * 16;

            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            uint32x4_t m0 = simd_compare_i32<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare_i32<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare_i32<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare_i32<Op>(d3, threshold);

            // vsub 技巧: mask=0xFFFFFFFF时 acc-(-1)=acc+1
            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        // 批次归约
        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // ========================================================================
    // 阶段 2: 16 元素块处理
    // ========================================================================
    for (; i + 16 <= count; i += 16) {
        __builtin_prefetch(input + i + L1_OFFSET, 0, 3);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32x4_t m0 = simd_compare_i32<Op>(d0, threshold);
        uint32x4_t m1 = simd_compare_i32<Op>(d1, threshold);
        uint32x4_t m2 = simd_compare_i32<Op>(d2, threshold);
        uint32x4_t m3 = simd_compare_i32<Op>(d3, threshold);

        uint32x4_t sum01 = vaddq_u32(vsubq_u32(vdupq_n_u32(0), m0),
                                     vsubq_u32(vdupq_n_u32(0), m1));
        uint32x4_t sum23 = vaddq_u32(vsubq_u32(vdupq_n_u32(0), m2),
                                     vsubq_u32(vdupq_n_u32(0), m3));
        result += vaddvq_u32(vaddq_u32(sum01, sum23));
    }

    // ========================================================================
    // 阶段 3: 标量尾部处理
    // ========================================================================
    for (; i < count; ++i) {
        if (scalar_compare_i32<Op>(input[i], value)) {
            ++result;
        }
    }

    return result;
}

// ============================================================================
// v6.0 过滤实现 - 委托给 v3 (预取优化不适用于过滤)
// ============================================================================
// 注: 多级预取对 Filter 索引输出无明显收益
// Filter 主要瓶颈在索引提取,而非数据加载
// V9.1 仅保留 COUNT 的预取优化

#endif // __aarch64__

// ============================================================================
// v6.0 公共接口
// ============================================================================

size_t count_i32_v6(const int32_t* input, size_t count,
                    CompareOp op, int32_t value) {
#ifdef __aarch64__
    switch (op) {
        case CompareOp::GT: return count_i32_v6_core<CompareOp::GT>(input, count, value);
        case CompareOp::GE: return count_i32_v6_core<CompareOp::GE>(input, count, value);
        case CompareOp::LT: return count_i32_v6_core<CompareOp::LT>(input, count, value);
        case CompareOp::LE: return count_i32_v6_core<CompareOp::LE>(input, count, value);
        case CompareOp::EQ: return count_i32_v6_core<CompareOp::EQ>(input, count, value);
        case CompareOp::NE: return count_i32_v6_core<CompareOp::NE>(input, count, value);
    }
#endif
    // Fallback to v3
    return count_i32_v3(input, count, op, value);
}

size_t filter_i32_v6(const int32_t* input, size_t count,
                     CompareOp op, int32_t value,
                     uint32_t* out_indices) {
    // V9.1: 预取优化对 Filter 索引输出无收益
    // 委托给 v3 实现
    return filter_i32_v3(input, count, op, value, out_indices);
}

} // namespace filter
} // namespace thunderduck
