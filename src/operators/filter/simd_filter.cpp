/**
 * ThunderDuck - SIMD Filter Implementation
 * 
 * ARM Neon 加速的过滤算子实现
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
// 查找表：用于快速从掩码提取索引
// ============================================================================

namespace {

// 4-bit 掩码到索引的查找表
// 对于 4 个元素的 SIMD 比较结果，掩码范围 0-15
// 每个条目存储：[count, idx0, idx1, idx2, idx3]
alignas(64) const uint8_t MASK_TO_INDICES_4[16][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},  // 0b0000 - 无匹配
    {1, 0, 0, 0, 0, 0, 0, 0},  // 0b0001 - 索引 0
    {1, 1, 0, 0, 0, 0, 0, 0},  // 0b0010 - 索引 1
    {2, 0, 1, 0, 0, 0, 0, 0},  // 0b0011 - 索引 0,1
    {1, 2, 0, 0, 0, 0, 0, 0},  // 0b0100 - 索引 2
    {2, 0, 2, 0, 0, 0, 0, 0},  // 0b0101 - 索引 0,2
    {2, 1, 2, 0, 0, 0, 0, 0},  // 0b0110 - 索引 1,2
    {3, 0, 1, 2, 0, 0, 0, 0},  // 0b0111 - 索引 0,1,2
    {1, 3, 0, 0, 0, 0, 0, 0},  // 0b1000 - 索引 3
    {2, 0, 3, 0, 0, 0, 0, 0},  // 0b1001 - 索引 0,3
    {2, 1, 3, 0, 0, 0, 0, 0},  // 0b1010 - 索引 1,3
    {3, 0, 1, 3, 0, 0, 0, 0},  // 0b1011 - 索引 0,1,3
    {2, 2, 3, 0, 0, 0, 0, 0},  // 0b1100 - 索引 2,3
    {3, 0, 2, 3, 0, 0, 0, 0},  // 0b1101 - 索引 0,2,3
    {3, 1, 2, 3, 0, 0, 0, 0},  // 0b1110 - 索引 1,2,3
    {4, 0, 1, 2, 3, 0, 0, 0},  // 0b1111 - 索引 0,1,2,3
};

#ifdef __aarch64__

// 从 uint32x4 掩码提取 4-bit movemask
inline uint32_t extract_mask_4(uint32x4_t mask) {
    // 每个 lane 是 0 或 0xFFFFFFFF
    // 右移 31 位得到 0 或 1
    uint32x4_t shifted = vshrq_n_u32(mask, 31);
    
    // 收集各位：lane0 + lane1*2 + lane2*4 + lane3*8
    uint32_t lane0 = vgetq_lane_u32(shifted, 0);
    uint32_t lane1 = vgetq_lane_u32(shifted, 1);
    uint32_t lane2 = vgetq_lane_u32(shifted, 2);
    uint32_t lane3 = vgetq_lane_u32(shifted, 3);
    
    return lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3);
}

// 根据比较操作生成掩码
template<CompareOp Op>
inline uint32x4_t compare_i32(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) {
        return vcgtq_s32(data, threshold);
    } else if constexpr (Op == CompareOp::GE) {
        return vcgeq_s32(data, threshold);
    } else if constexpr (Op == CompareOp::LT) {
        return vcltq_s32(data, threshold);
    } else if constexpr (Op == CompareOp::LE) {
        return vcleq_s32(data, threshold);
    } else if constexpr (Op == CompareOp::EQ) {
        return vceqq_s32(data, threshold);
    } else { // NE
        return vmvnq_u32(vceqq_s32(data, threshold));
    }
}

// 动态分发比较操作
inline uint32x4_t compare_i32_dynamic(int32x4_t data, int32x4_t threshold, CompareOp op) {
    switch (op) {
        case CompareOp::GT: return vcgtq_s32(data, threshold);
        case CompareOp::GE: return vcgeq_s32(data, threshold);
        case CompareOp::LT: return vcltq_s32(data, threshold);
        case CompareOp::LE: return vcleq_s32(data, threshold);
        case CompareOp::EQ: return vceqq_s32(data, threshold);
        case CompareOp::NE: return vmvnq_u32(vceqq_s32(data, threshold));
        default: return vdupq_n_u32(0);
    }
}

// float 比较
inline uint32x4_t compare_f32_dynamic(float32x4_t data, float32x4_t threshold, CompareOp op) {
    switch (op) {
        case CompareOp::GT: return vcgtq_f32(data, threshold);
        case CompareOp::GE: return vcgeq_f32(data, threshold);
        case CompareOp::LT: return vcltq_f32(data, threshold);
        case CompareOp::LE: return vcleq_f32(data, threshold);
        case CompareOp::EQ: return vceqq_f32(data, threshold);
        case CompareOp::NE: return vmvnq_u32(vceqq_f32(data, threshold));
        default: return vdupq_n_u32(0);
    }
}

#endif // __aarch64__

} // anonymous namespace

// ============================================================================
// int32 过滤实现
// ============================================================================

size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices) {
    size_t out_count = 0;

#ifdef __aarch64__
    // SIMD 主循环 - 每次处理 4 个元素
    int32x4_t threshold = vdupq_n_s32(value);
    size_t i = 0;
    
    // 处理 4 的倍数部分
    for (; i + 4 <= count; i += 4) {
        // 加载 4 个 int32
        int32x4_t data = vld1q_s32(input + i);
        
        // 比较生成掩码
        uint32x4_t mask = compare_i32_dynamic(data, threshold, op);
        
        // 提取掩码位
        uint32_t mask_bits = extract_mask_4(mask);
        
        if (mask_bits != 0) {
            // 使用查找表获取索引
            const uint8_t* lut = MASK_TO_INDICES_4[mask_bits];
            uint32_t match_count = lut[0];
            
            // 写入索引
            for (uint32_t j = 0; j < match_count; ++j) {
                out_indices[out_count + j] = static_cast<uint32_t>(i) + lut[1 + j];
            }
            out_count += match_count;
        }
    }
    
    // 处理剩余元素（标量）
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
        if (match) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
    
#else
    // 非 ARM 平台：标量实现
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
        if (match) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return out_count;
}

size_t filter_i32_values(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         int32_t* out_values) {
    size_t out_count = 0;

#ifdef __aarch64__
    int32x4_t threshold = vdupq_n_s32(value);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32x4_t mask = compare_i32_dynamic(data, threshold, op);
        uint32_t mask_bits = extract_mask_4(mask);
        
        if (mask_bits != 0) {
            const uint8_t* lut = MASK_TO_INDICES_4[mask_bits];
            uint32_t match_count = lut[0];
            
            // 写入匹配的值
            for (uint32_t j = 0; j < match_count; ++j) {
                out_values[out_count + j] = input[i + lut[1 + j]];
            }
            out_count += match_count;
        }
    }
    
    // 剩余元素
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
        if (match) {
            out_values[out_count++] = input[i];
        }
    }
#else
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
        if (match) {
            out_values[out_count++] = input[i];
        }
    }
#endif

    return out_count;
}

// ============================================================================
// float 过滤实现
// ============================================================================

size_t filter_f32(const float* input, size_t count,
                  CompareOp op, float value,
                  uint32_t* out_indices) {
    size_t out_count = 0;

#ifdef __aarch64__
    float32x4_t threshold = vdupq_n_f32(value);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        uint32x4_t mask = compare_f32_dynamic(data, threshold, op);
        uint32_t mask_bits = extract_mask_4(mask);
        
        if (mask_bits != 0) {
            const uint8_t* lut = MASK_TO_INDICES_4[mask_bits];
            uint32_t match_count = lut[0];
            
            for (uint32_t j = 0; j < match_count; ++j) {
                out_indices[out_count + j] = static_cast<uint32_t>(i) + lut[1 + j];
            }
            out_count += match_count;
        }
    }
    
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
        if (match) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#else
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
        if (match) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return out_count;
}

size_t filter_f32_values(const float* input, size_t count,
                         CompareOp op, float value,
                         float* out_values) {
    size_t out_count = 0;

#ifdef __aarch64__
    float32x4_t threshold = vdupq_n_f32(value);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        uint32x4_t mask = compare_f32_dynamic(data, threshold, op);
        uint32_t mask_bits = extract_mask_4(mask);
        
        if (mask_bits != 0) {
            const uint8_t* lut = MASK_TO_INDICES_4[mask_bits];
            uint32_t match_count = lut[0];
            
            for (uint32_t j = 0; j < match_count; ++j) {
                out_values[out_count + j] = input[i + lut[1 + j]];
            }
            out_count += match_count;
        }
    }
    
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
        if (match) {
            out_values[out_count++] = input[i];
        }
    }
#else
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
        if (match) {
            out_values[out_count++] = input[i];
        }
    }
#endif

    return out_count;
}

// ============================================================================
// 复合过滤实现
// ============================================================================

size_t filter_i32_and(const int32_t* input, size_t count,
                      CompareOp op1, int32_t value1,
                      CompareOp op2, int32_t value2,
                      uint32_t* out_indices) {
    size_t out_count = 0;

#ifdef __aarch64__
    int32x4_t threshold1 = vdupq_n_s32(value1);
    int32x4_t threshold2 = vdupq_n_s32(value2);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        
        // 两个条件的掩码
        uint32x4_t mask1 = compare_i32_dynamic(data, threshold1, op1);
        uint32x4_t mask2 = compare_i32_dynamic(data, threshold2, op2);
        
        // AND 组合
        uint32x4_t mask = vandq_u32(mask1, mask2);
        uint32_t mask_bits = extract_mask_4(mask);
        
        if (mask_bits != 0) {
            const uint8_t* lut = MASK_TO_INDICES_4[mask_bits];
            uint32_t match_count = lut[0];
            
            for (uint32_t j = 0; j < match_count; ++j) {
                out_indices[out_count + j] = static_cast<uint32_t>(i) + lut[1 + j];
            }
            out_count += match_count;
        }
    }
    
    // 剩余元素标量处理
    for (; i < count; ++i) {
        bool match1 = false, match2 = false;
        switch (op1) {
            case CompareOp::GT: match1 = input[i] > value1; break;
            case CompareOp::GE: match1 = input[i] >= value1; break;
            case CompareOp::LT: match1 = input[i] < value1; break;
            case CompareOp::LE: match1 = input[i] <= value1; break;
            case CompareOp::EQ: match1 = input[i] == value1; break;
            case CompareOp::NE: match1 = input[i] != value1; break;
        }
        switch (op2) {
            case CompareOp::GT: match2 = input[i] > value2; break;
            case CompareOp::GE: match2 = input[i] >= value2; break;
            case CompareOp::LT: match2 = input[i] < value2; break;
            case CompareOp::LE: match2 = input[i] <= value2; break;
            case CompareOp::EQ: match2 = input[i] == value2; break;
            case CompareOp::NE: match2 = input[i] != value2; break;
        }
        if (match1 && match2) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        bool match1 = false, match2 = false;
        switch (op1) {
            case CompareOp::GT: match1 = input[i] > value1; break;
            case CompareOp::GE: match1 = input[i] >= value1; break;
            case CompareOp::LT: match1 = input[i] < value1; break;
            case CompareOp::LE: match1 = input[i] <= value1; break;
            case CompareOp::EQ: match1 = input[i] == value1; break;
            case CompareOp::NE: match1 = input[i] != value1; break;
        }
        switch (op2) {
            case CompareOp::GT: match2 = input[i] > value2; break;
            case CompareOp::GE: match2 = input[i] >= value2; break;
            case CompareOp::LT: match2 = input[i] < value2; break;
            case CompareOp::LE: match2 = input[i] <= value2; break;
            case CompareOp::EQ: match2 = input[i] == value2; break;
            case CompareOp::NE: match2 = input[i] != value2; break;
        }
        if (match1 && match2) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return out_count;
}

size_t filter_i32_range(const int32_t* input, size_t count,
                        int32_t low, int32_t high,
                        uint32_t* out_indices) {
    // low <= x < high  等价于  x >= low AND x < high
    return filter_i32_and(input, count,
                          CompareOp::GE, low,
                          CompareOp::LT, high,
                          out_indices);
}

// ============================================================================
// 计数函数实现
// ============================================================================

size_t count_i32(const int32_t* input, size_t count,
                 CompareOp op, int32_t value) {
    size_t result = 0;

#ifdef __aarch64__
    int32x4_t threshold = vdupq_n_s32(value);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;
    
    // SIMD 循环，累加掩码位
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32x4_t mask = compare_i32_dynamic(data, threshold, op);
        // 掩码是 0 或 0xFFFFFFFF，右移 31 位得到 0 或 1
        uint32x4_t ones = vshrq_n_u32(mask, 31);
        count_vec = vaddq_u32(count_vec, ones);
    }
    
    // 水平归约
    result = vaddvq_u32(count_vec);
    
    // 处理剩余
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
#else
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
#endif

    return result;
}

size_t count_f32(const float* input, size_t count,
                 CompareOp op, float value) {
    size_t result = 0;

#ifdef __aarch64__
    float32x4_t threshold = vdupq_n_f32(value);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        uint32x4_t mask = compare_f32_dynamic(data, threshold, op);
        uint32x4_t ones = vshrq_n_u32(mask, 31);
        count_vec = vaddq_u32(count_vec, ones);
    }
    
    result = vaddvq_u32(count_vec);
    
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
#else
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
#endif

    return result;
}

} // namespace filter
} // namespace thunderduck
