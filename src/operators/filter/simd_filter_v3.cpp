/**
 * ThunderDuck - SIMD Filter Implementation v3.0
 *
 * 优化特性：
 * 1. 模板特化消除 switch-case
 * 2. 4 独立累加器消除依赖链 (ILP)
 * 3. vsub 替代 vshr+vadd 减少指令
 * 4. 256 元素批次处理减少归约次数
 * 5. 自适应预取策略
 */

#include "thunderduck/filter.h"
#include "thunderduck/simd.h"
#include "thunderduck/memory.h"
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

#ifdef __aarch64__

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

// 标量比较 - 用于尾部处理
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
// v3.0 核心计数实现 - 模板特化
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t count_i32_v3_core(const int32_t* __restrict input,
                          size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t result = 0;
    size_t i = 0;

    // 常量定义
    constexpr size_t BATCH_SIZE = 256;      // 每批处理 256 个元素
    constexpr size_t INNER_ITERS = 16;      // 内层循环次数 (256 / 16)
    constexpr size_t PREFETCH_DISTANCE = 512; // 预取距离（字节）

    // ========================================================================
    // 阶段 1: 256 元素批次处理
    // ========================================================================
    for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
        // 4 个独立累加器 - 消除依赖链
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const int32_t* batch_ptr = input + i;

        // 预取下一批数据
        __builtin_prefetch(batch_ptr + BATCH_SIZE, 0, 0);
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 64, 0, 0);

        // 内层循环 - 编译器会尝试展开
        #pragma unroll
        for (size_t j = 0; j < INNER_ITERS; ++j) {
            const int32_t* ptr = batch_ptr + j * 16;

            // 加载 16 个元素 (4 个 NEON 向量)
            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            // 比较操作 (编译期确定，无分支)
            uint32x4_t m0 = simd_compare_i32<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare_i32<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare_i32<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare_i32<Op>(d3, threshold);

            // 累加计数 - 使用 vsub 技巧
            // 当 mask = 0xFFFFFFFF 时: acc - mask = acc - (-1) = acc + 1
            // 当 mask = 0x00000000 时: acc - mask = acc - 0 = acc
            // 这样做的好处：
            // 1. 只需要 1 条指令 (vsub) 替代 2 条 (vshr + vadd)
            // 2. 4 个累加器完全独立，充分利用 CPU 执行单元
            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        // 批次归约 - 合并 4 个累加器
        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // ========================================================================
    // 阶段 2: 16 元素迭代处理剩余
    // ========================================================================
    {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        for (; i + 16 <= count; i += 16) {
            int32x4_t d0 = vld1q_s32(input + i);
            int32x4_t d1 = vld1q_s32(input + i + 4);
            int32x4_t d2 = vld1q_s32(input + i + 8);
            int32x4_t d3 = vld1q_s32(input + i + 12);

            uint32x4_t m0 = simd_compare_i32<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare_i32<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare_i32<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare_i32<Op>(d3, threshold);

            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // ========================================================================
    // 阶段 3: 4 元素迭代处理剩余
    // ========================================================================
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i + 4 <= count; i += 4) {
            int32x4_t data = vld1q_s32(input + i);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
            acc = vsubq_u32(acc, mask);
        }
        result += vaddvq_u32(acc);
    }

    // ========================================================================
    // 阶段 4: 标量处理尾部
    // ========================================================================
    for (; i < count; ++i) {
        if (scalar_compare_i32<Op>(input[i], value)) {
            ++result;
        }
    }

    return result;
}

// ============================================================================
// v3.0 范围计数实现
// ============================================================================

size_t count_i32_range_v3(const int32_t* __restrict input, size_t count,
                           int32_t low, int32_t high) {
    int32x4_t low_vec = vdupq_n_s32(low);
    int32x4_t high_vec = vdupq_n_s32(high);
    size_t result = 0;
    size_t i = 0;

    constexpr size_t BATCH_SIZE = 256;
    constexpr size_t INNER_ITERS = 16;

    // 阶段 1: 256 元素批次
    for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const int32_t* batch_ptr = input + i;
        __builtin_prefetch(batch_ptr + BATCH_SIZE, 0, 0);
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 64, 0, 0);

        #pragma unroll
        for (size_t j = 0; j < INNER_ITERS; ++j) {
            const int32_t* ptr = batch_ptr + j * 16;

            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            // 范围检查: low <= x < high  =>  x >= low AND x < high
            uint32x4_t ge0 = vcgeq_s32(d0, low_vec);
            uint32x4_t ge1 = vcgeq_s32(d1, low_vec);
            uint32x4_t ge2 = vcgeq_s32(d2, low_vec);
            uint32x4_t ge3 = vcgeq_s32(d3, low_vec);

            uint32x4_t lt0 = vcltq_s32(d0, high_vec);
            uint32x4_t lt1 = vcltq_s32(d1, high_vec);
            uint32x4_t lt2 = vcltq_s32(d2, high_vec);
            uint32x4_t lt3 = vcltq_s32(d3, high_vec);

            // AND 组合
            uint32x4_t m0 = vandq_u32(ge0, lt0);
            uint32x4_t m1 = vandq_u32(ge1, lt1);
            uint32x4_t m2 = vandq_u32(ge2, lt2);
            uint32x4_t m3 = vandq_u32(ge3, lt3);

            // 使用 vsub 累加
            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // 阶段 2: 16 元素迭代
    {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        for (; i + 16 <= count; i += 16) {
            int32x4_t d0 = vld1q_s32(input + i);
            int32x4_t d1 = vld1q_s32(input + i + 4);
            int32x4_t d2 = vld1q_s32(input + i + 8);
            int32x4_t d3 = vld1q_s32(input + i + 12);

            uint32x4_t ge0 = vcgeq_s32(d0, low_vec);
            uint32x4_t ge1 = vcgeq_s32(d1, low_vec);
            uint32x4_t ge2 = vcgeq_s32(d2, low_vec);
            uint32x4_t ge3 = vcgeq_s32(d3, low_vec);

            uint32x4_t lt0 = vcltq_s32(d0, high_vec);
            uint32x4_t lt1 = vcltq_s32(d1, high_vec);
            uint32x4_t lt2 = vcltq_s32(d2, high_vec);
            uint32x4_t lt3 = vcltq_s32(d3, high_vec);

            uint32x4_t m0 = vandq_u32(ge0, lt0);
            uint32x4_t m1 = vandq_u32(ge1, lt1);
            uint32x4_t m2 = vandq_u32(ge2, lt2);
            uint32x4_t m3 = vandq_u32(ge3, lt3);

            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // 阶段 3: 标量尾部
    for (; i < count; ++i) {
        if (input[i] >= low && input[i] < high) {
            ++result;
        }
    }

    return result;
}

// ============================================================================
// v3.0 位图过滤实现
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t filter_to_bitmap_v3_core(const int32_t* __restrict input, size_t count,
                                 int32_t value, uint64_t* __restrict bitmap) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t match_count = 0;
    size_t i = 0;
    size_t bitmap_idx = 0;

    // 每 64 个元素生成一个 uint64_t 位图
    for (; i + 64 <= count; i += 64, ++bitmap_idx) {
        __builtin_prefetch(input + i + 128, 0, 0);

        uint64_t bits = 0;

        // 处理 16 组 4 元素
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);

            // 提取 4 位
            uint32_t m0 = vgetq_lane_u32(mask, 0) >> 31;
            uint32_t m1 = vgetq_lane_u32(mask, 1) >> 31;
            uint32_t m2 = vgetq_lane_u32(mask, 2) >> 31;
            uint32_t m3 = vgetq_lane_u32(mask, 3) >> 31;

            uint64_t group_bits = (uint64_t)m0 | ((uint64_t)m1 << 1) |
                                  ((uint64_t)m2 << 2) | ((uint64_t)m3 << 3);
            bits |= group_bits << (g * 4);
        }

        bitmap[bitmap_idx] = bits;
        match_count += __builtin_popcountll(bits);
    }

    // 处理剩余元素
    if (i < count) {
        uint64_t bits = 0;
        for (size_t j = 0; i + j < count; ++j) {
            if (scalar_compare_i32<Op>(input[i + j], value)) {
                bits |= (1ULL << j);
                ++match_count;
            }
        }
        bitmap[bitmap_idx] = bits;
    }

    return match_count;
}

#endif // __aarch64__

// ============================================================================
// 公开接口 - 运行时分发
// ============================================================================

size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
#ifdef __aarch64__
    switch (op) {
        case CompareOp::GT: return count_i32_v3_core<CompareOp::GT>(input, count, value);
        case CompareOp::GE: return count_i32_v3_core<CompareOp::GE>(input, count, value);
        case CompareOp::LT: return count_i32_v3_core<CompareOp::LT>(input, count, value);
        case CompareOp::LE: return count_i32_v3_core<CompareOp::LE>(input, count, value);
        case CompareOp::EQ: return count_i32_v3_core<CompareOp::EQ>(input, count, value);
        case CompareOp::NE: return count_i32_v3_core<CompareOp::NE>(input, count, value);
        default: return 0;
    }
#else
    // 非 ARM 平台回退到标量实现
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        bool match = false;
        switch (op) {
            case CompareOp::GT: match = input[i] > value; break;
            case CompareOp::GE: match = input[i] >= value; break;
            case CompareOp::LT: match = input[i] < value; break;
            case CompareOp::LE: match = input[i] <= value; break;
            case CompareOp::EQ: match = input[i] == value; break;
            case CompareOp::NE: match = input[i] != value; break;
        }
        if (match) ++result;
    }
    return result;
#endif
}

size_t filter_to_bitmap_v3(const int32_t* input, size_t count,
                            CompareOp op, int32_t value,
                            uint64_t* bitmap) {
#ifdef __aarch64__
    switch (op) {
        case CompareOp::GT: return filter_to_bitmap_v3_core<CompareOp::GT>(input, count, value, bitmap);
        case CompareOp::GE: return filter_to_bitmap_v3_core<CompareOp::GE>(input, count, value, bitmap);
        case CompareOp::LT: return filter_to_bitmap_v3_core<CompareOp::LT>(input, count, value, bitmap);
        case CompareOp::LE: return filter_to_bitmap_v3_core<CompareOp::LE>(input, count, value, bitmap);
        case CompareOp::EQ: return filter_to_bitmap_v3_core<CompareOp::EQ>(input, count, value, bitmap);
        case CompareOp::NE: return filter_to_bitmap_v3_core<CompareOp::NE>(input, count, value, bitmap);
        default: return 0;
    }
#else
    size_t match_count = 0;
    size_t bitmap_idx = 0;
    for (size_t i = 0; i < count; i += 64, ++bitmap_idx) {
        uint64_t bits = 0;
        for (size_t j = 0; j < 64 && i + j < count; ++j) {
            bool match = false;
            switch (op) {
                case CompareOp::GT: match = input[i + j] > value; break;
                case CompareOp::GE: match = input[i + j] >= value; break;
                case CompareOp::LT: match = input[i + j] < value; break;
                case CompareOp::LE: match = input[i + j] <= value; break;
                case CompareOp::EQ: match = input[i + j] == value; break;
                case CompareOp::NE: match = input[i + j] != value; break;
            }
            if (match) {
                bits |= (1ULL << j);
                ++match_count;
            }
        }
        bitmap[bitmap_idx] = bits;
    }
    return match_count;
#endif
}

size_t filter_i32_v3(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices) {
    // 分配位图
    size_t bitmap_size = (count + 63) / 64;
    uint64_t* bitmap = static_cast<uint64_t*>(
        aligned_alloc(bitmap_size * sizeof(uint64_t), 64));

    // 使用 v3 位图过滤
    size_t match_count = filter_to_bitmap_v3(input, count, op, value, bitmap);

    // 位图转索引
    if (out_indices && match_count > 0) {
        bitmap_to_indices(bitmap, count, out_indices);
    }

    aligned_free(bitmap);
    return match_count;
}

} // namespace filter
} // namespace thunderduck
