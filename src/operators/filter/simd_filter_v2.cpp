/**
 * ThunderDuck - SIMD Filter Implementation v2.0
 *
 * 优化版本：
 * - 16 元素/迭代的计数函数
 * - 位图过滤
 * - 批量索引提取
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

// ============================================================================
// 优化的计数函数 - 16 元素/迭代
// ============================================================================

size_t count_i32_v2(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
#ifdef __aarch64__
    int32x4_t threshold = vdupq_n_s32(value);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;

    // 主循环：每次处理 16 个元素
    for (; i + 16 <= count; i += 16) {
        // 预取下一批数据
        __builtin_prefetch(input + i + 64, 0, 0);

        // 加载 16 个元素 (4 个向量)
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 比较生成掩码
        uint32x4_t m0, m1, m2, m3;

        switch (op) {
            case CompareOp::GT:
                m0 = vcgtq_s32(d0, threshold);
                m1 = vcgtq_s32(d1, threshold);
                m2 = vcgtq_s32(d2, threshold);
                m3 = vcgtq_s32(d3, threshold);
                break;
            case CompareOp::GE:
                m0 = vcgeq_s32(d0, threshold);
                m1 = vcgeq_s32(d1, threshold);
                m2 = vcgeq_s32(d2, threshold);
                m3 = vcgeq_s32(d3, threshold);
                break;
            case CompareOp::LT:
                m0 = vcltq_s32(d0, threshold);
                m1 = vcltq_s32(d1, threshold);
                m2 = vcltq_s32(d2, threshold);
                m3 = vcltq_s32(d3, threshold);
                break;
            case CompareOp::LE:
                m0 = vcleq_s32(d0, threshold);
                m1 = vcleq_s32(d1, threshold);
                m2 = vcleq_s32(d2, threshold);
                m3 = vcleq_s32(d3, threshold);
                break;
            case CompareOp::EQ:
                m0 = vceqq_s32(d0, threshold);
                m1 = vceqq_s32(d1, threshold);
                m2 = vceqq_s32(d2, threshold);
                m3 = vceqq_s32(d3, threshold);
                break;
            case CompareOp::NE:
                m0 = vmvnq_u32(vceqq_s32(d0, threshold));
                m1 = vmvnq_u32(vceqq_s32(d1, threshold));
                m2 = vmvnq_u32(vceqq_s32(d2, threshold));
                m3 = vmvnq_u32(vceqq_s32(d3, threshold));
                break;
        }

        // 掩码右移 31 位得到 0 或 1
        uint32x4_t ones0 = vshrq_n_u32(m0, 31);
        uint32x4_t ones1 = vshrq_n_u32(m1, 31);
        uint32x4_t ones2 = vshrq_n_u32(m2, 31);
        uint32x4_t ones3 = vshrq_n_u32(m3, 31);

        // 累加
        count_vec = vaddq_u32(count_vec, ones0);
        count_vec = vaddq_u32(count_vec, ones1);
        count_vec = vaddq_u32(count_vec, ones2);
        count_vec = vaddq_u32(count_vec, ones3);
    }

    // 4 元素循环处理剩余
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32x4_t mask;

        switch (op) {
            case CompareOp::GT: mask = vcgtq_s32(data, threshold); break;
            case CompareOp::GE: mask = vcgeq_s32(data, threshold); break;
            case CompareOp::LT: mask = vcltq_s32(data, threshold); break;
            case CompareOp::LE: mask = vcleq_s32(data, threshold); break;
            case CompareOp::EQ: mask = vceqq_s32(data, threshold); break;
            case CompareOp::NE: mask = vmvnq_u32(vceqq_s32(data, threshold)); break;
        }

        count_vec = vaddq_u32(count_vec, vshrq_n_u32(mask, 31));
    }

    // 水平归约
    size_t result = vaddvq_u32(count_vec);

    // 标量处理剩余
    for (; i < count; ++i) {
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
#else
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

// ============================================================================
// 位图过滤 - 生成位图
// ============================================================================

size_t filter_to_bitmap_i32(const int32_t* input, size_t count,
                             CompareOp op, int32_t value,
                             uint64_t* bitmap) {
    size_t match_count = 0;

#ifdef __aarch64__
    int32x4_t threshold = vdupq_n_s32(value);
    size_t i = 0;
    size_t bitmap_idx = 0;

    // 每次处理 64 个元素生成一个 uint64_t
    for (; i + 64 <= count; i += 64, ++bitmap_idx) {
        __builtin_prefetch(input + i + 128, 0, 0);

        uint64_t bits = 0;

        // 处理 16 组 4 元素
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask;

            switch (op) {
                case CompareOp::GT: mask = vcgtq_s32(data, threshold); break;
                case CompareOp::GE: mask = vcgeq_s32(data, threshold); break;
                case CompareOp::LT: mask = vcltq_s32(data, threshold); break;
                case CompareOp::LE: mask = vcleq_s32(data, threshold); break;
                case CompareOp::EQ: mask = vceqq_s32(data, threshold); break;
                case CompareOp::NE: mask = vmvnq_u32(vceqq_s32(data, threshold)); break;
            }

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

#else
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
#endif

    return match_count;
}

// ============================================================================
// 位图转索引 - 快速提取
// ============================================================================

size_t bitmap_to_indices(const uint64_t* bitmap, size_t bit_count,
                         uint32_t* out_indices) {
    size_t out_count = 0;
    size_t num_words = (bit_count + 63) / 64;

    for (size_t i = 0; i < num_words; ++i) {
        uint64_t bits = bitmap[i];
        uint32_t base = static_cast<uint32_t>(i * 64);

        // 快速跳过空 word
        if (bits == 0) continue;

        // 使用 CTZ (Count Trailing Zeros) 快速提取索引
        while (bits) {
            uint32_t pos = __builtin_ctzll(bits);
            out_indices[out_count++] = base + pos;
            bits &= bits - 1;  // 清除最低位的 1
        }
    }

    return out_count;
}

// ============================================================================
// 优化的过滤函数 - 使用位图中间表示
// ============================================================================

size_t filter_i32_v2(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices) {
    // 分配位图
    size_t bitmap_size = (count + 63) / 64;
    uint64_t* bitmap = static_cast<uint64_t*>(
        aligned_alloc(bitmap_size * sizeof(uint64_t), 64));

    // 生成位图
    size_t match_count = filter_to_bitmap_i32(input, count, op, value, bitmap);

    // 位图转索引
    if (out_indices && match_count > 0) {
        bitmap_to_indices(bitmap, count, out_indices);
    }

    aligned_free(bitmap);
    return match_count;
}

// ============================================================================
// Range 过滤优化版本
// ============================================================================

size_t count_i32_range_v2(const int32_t* input, size_t count,
                           int32_t low, int32_t high) {
#ifdef __aarch64__
    int32x4_t low_vec = vdupq_n_s32(low);
    int32x4_t high_vec = vdupq_n_s32(high);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;

    // 主循环：每次处理 16 个元素
    for (; i + 16 <= count; i += 16) {
        __builtin_prefetch(input + i + 64, 0, 0);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // low <= x < high  =>  x >= low AND x < high
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

        // 累加
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m0, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m1, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m2, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m3, 31));
    }

    // 处理剩余
    size_t result = vaddvq_u32(count_vec);
    for (; i < count; ++i) {
        if (input[i] >= low && input[i] < high) {
            ++result;
        }
    }

    return result;
#else
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (input[i] >= low && input[i] < high) {
            ++result;
        }
    }
    return result;
#endif
}

} // namespace filter
} // namespace thunderduck
