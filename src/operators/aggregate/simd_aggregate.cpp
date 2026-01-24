/**
 * ThunderDuck - SIMD Aggregation Implementation
 * 
 * ARM Neon 加速的聚合算子实现
 */

#include "thunderduck/aggregate.h"
#include "thunderduck/simd.h"
#include "thunderduck/memory.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <cstring>
#include <algorithm>

namespace thunderduck {
namespace aggregate {

// ============================================================================
// SUM 实现
// ============================================================================

int64_t sum_i32(const int32_t* input, size_t count) {
    int64_t result = 0;

#ifdef __aarch64__
    // 使用 4 个 int64 累加器，避免溢出
    int64x2_t sum_lo = vdupq_n_s64(0);
    int64x2_t sum_hi = vdupq_n_s64(0);
    size_t i = 0;
    
    // 主循环：每次处理 4 个 int32
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        
        // 扩展到 int64 并累加
        int64x2_t data_lo = vmovl_s32(vget_low_s32(data));   // 低 2 个
        int64x2_t data_hi = vmovl_s32(vget_high_s32(data));  // 高 2 个
        
        sum_lo = vaddq_s64(sum_lo, data_lo);
        sum_hi = vaddq_s64(sum_hi, data_hi);
    }
    
    // 合并 4 个 int64
    int64x2_t sum_combined = vaddq_s64(sum_lo, sum_hi);
    result = vaddvq_s64(sum_combined);
    
    // 处理剩余元素
    for (; i < count; ++i) {
        result += input[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result += input[i];
    }
#endif

    return result;
}

int64_t sum_i64(const int64_t* input, size_t count) {
    int64_t result = 0;

#ifdef __aarch64__
    int64x2_t sum_vec = vdupq_n_s64(0);
    size_t i = 0;
    
    for (; i + 2 <= count; i += 2) {
        int64x2_t data = vld1q_s64(input + i);
        sum_vec = vaddq_s64(sum_vec, data);
    }
    
    result = vaddvq_s64(sum_vec);
    
    for (; i < count; ++i) {
        result += input[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result += input[i];
    }
#endif

    return result;
}

double sum_f32(const float* input, size_t count) {
    double result = 0.0;

#ifdef __aarch64__
    // 使用 float64 累加以保持精度
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    size_t i = 0;
    
    // 每次处理 2 个 float，扩展到 double
    for (; i + 2 <= count; i += 2) {
        float32x2_t data_f32 = vld1_f32(input + i);
        float64x2_t data_f64 = vcvt_f64_f32(data_f32);
        sum_vec = vaddq_f64(sum_vec, data_f64);
    }
    
    result = vaddvq_f64(sum_vec);
    
    for (; i < count; ++i) {
        result += static_cast<double>(input[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result += static_cast<double>(input[i]);
    }
#endif

    return result;
}

double sum_f64(const double* input, size_t count) {
    double result = 0.0;

#ifdef __aarch64__
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    size_t i = 0;
    
    for (; i + 2 <= count; i += 2) {
        float64x2_t data = vld1q_f64(input + i);
        sum_vec = vaddq_f64(sum_vec, data);
    }
    
    result = vaddvq_f64(sum_vec);
    
    for (; i < count; ++i) {
        result += input[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result += input[i];
    }
#endif

    return result;
}

// ============================================================================
// MIN 实现
// ============================================================================

int32_t min_i32(const int32_t* input, size_t count) {
    if (count == 0) return std::numeric_limits<int32_t>::max();

#ifdef __aarch64__
    int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        min_vec = vminq_s32(min_vec, data);
    }
    
    int32_t result = vminvq_s32(min_vec);
    
    for (; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    
    return result;
#else
    int32_t result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    return result;
#endif
}

int64_t min_i64(const int64_t* input, size_t count) {
    if (count == 0) return std::numeric_limits<int64_t>::max();

#ifdef __aarch64__
    int64_t min_val = input[0];
    size_t i = 1;
    
    // int64 没有直接的 vminvq，使用部分向量化
    for (; i + 2 <= count; i += 2) {
        int64x2_t data = vld1q_s64(input + i);
        int64_t a = vgetq_lane_s64(data, 0);
        int64_t b = vgetq_lane_s64(data, 1);
        if (a < min_val) min_val = a;
        if (b < min_val) min_val = b;
    }
    
    for (; i < count; ++i) {
        if (input[i] < min_val) min_val = input[i];
    }
    
    return min_val;
#else
    int64_t result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    return result;
#endif
}

float min_f32(const float* input, size_t count) {
    if (count == 0) return std::numeric_limits<float>::max();

#ifdef __aarch64__
    float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        min_vec = vminq_f32(min_vec, data);
    }
    
    float result = vminvq_f32(min_vec);
    
    for (; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    
    return result;
#else
    float result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    return result;
#endif
}

double min_f64(const double* input, size_t count) {
    if (count == 0) return std::numeric_limits<double>::max();

#ifdef __aarch64__
    float64x2_t min_vec = vdupq_n_f64(std::numeric_limits<double>::max());
    size_t i = 0;
    
    for (; i + 2 <= count; i += 2) {
        float64x2_t data = vld1q_f64(input + i);
        min_vec = vminq_f64(min_vec, data);
    }
    
    double result = vminvq_f64(min_vec);
    
    for (; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    
    return result;
#else
    double result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] < result) result = input[i];
    }
    return result;
#endif
}

// ============================================================================
// MAX 实现
// ============================================================================

int32_t max_i32(const int32_t* input, size_t count) {
    if (count == 0) return std::numeric_limits<int32_t>::min();

#ifdef __aarch64__
    int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        max_vec = vmaxq_s32(max_vec, data);
    }
    
    int32_t result = vmaxvq_s32(max_vec);
    
    for (; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    
    return result;
#else
    int32_t result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    return result;
#endif
}

int64_t max_i64(const int64_t* input, size_t count) {
    if (count == 0) return std::numeric_limits<int64_t>::min();

    int64_t max_val = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    return max_val;
}

float max_f32(const float* input, size_t count) {
    if (count == 0) return std::numeric_limits<float>::lowest();

#ifdef __aarch64__
    float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        max_vec = vmaxq_f32(max_vec, data);
    }
    
    float result = vmaxvq_f32(max_vec);
    
    for (; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    
    return result;
#else
    float result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    return result;
#endif
}

double max_f64(const double* input, size_t count) {
    if (count == 0) return std::numeric_limits<double>::lowest();

#ifdef __aarch64__
    float64x2_t max_vec = vdupq_n_f64(std::numeric_limits<double>::lowest());
    size_t i = 0;
    
    for (; i + 2 <= count; i += 2) {
        float64x2_t data = vld1q_f64(input + i);
        max_vec = vmaxq_f64(max_vec, data);
    }
    
    double result = vmaxvq_f64(max_vec);
    
    for (; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    
    return result;
#else
    double result = input[0];
    for (size_t i = 1; i < count; ++i) {
        if (input[i] > result) result = input[i];
    }
    return result;
#endif
}

// ============================================================================
// AVG 实现
// ============================================================================

double avg_i32(const int32_t* input, size_t count) {
    if (count == 0) return 0.0;
    return static_cast<double>(sum_i32(input, count)) / static_cast<double>(count);
}

double avg_i64(const int64_t* input, size_t count) {
    if (count == 0) return 0.0;
    return static_cast<double>(sum_i64(input, count)) / static_cast<double>(count);
}

double avg_f32(const float* input, size_t count) {
    if (count == 0) return 0.0;
    return sum_f32(input, count) / static_cast<double>(count);
}

double avg_f64(const double* input, size_t count) {
    if (count == 0) return 0.0;
    return sum_f64(input, count) / static_cast<double>(count);
}

// ============================================================================
// 带选择向量的聚合
// ============================================================================

int64_t sum_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count) {
    int64_t result = 0;
    
    // 选择向量聚合较难向量化，使用展开循环
    size_t i = 0;
    for (; i + 4 <= sel_count; i += 4) {
        result += input[sel[i]];
        result += input[sel[i + 1]];
        result += input[sel[i + 2]];
        result += input[sel[i + 3]];
    }
    for (; i < sel_count; ++i) {
        result += input[sel[i]];
    }
    
    return result;
}

double sum_f32_sel(const float* input, const uint32_t* sel, size_t sel_count) {
    double result = 0.0;
    
    size_t i = 0;
    for (; i + 4 <= sel_count; i += 4) {
        result += static_cast<double>(input[sel[i]]);
        result += static_cast<double>(input[sel[i + 1]]);
        result += static_cast<double>(input[sel[i + 2]]);
        result += static_cast<double>(input[sel[i + 3]]);
    }
    for (; i < sel_count; ++i) {
        result += static_cast<double>(input[sel[i]]);
    }
    
    return result;
}

int32_t min_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count) {
    if (sel_count == 0) return std::numeric_limits<int32_t>::max();
    
    int32_t result = input[sel[0]];
    for (size_t i = 1; i < sel_count; ++i) {
        int32_t val = input[sel[i]];
        if (val < result) result = val;
    }
    return result;
}

int32_t max_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count) {
    if (sel_count == 0) return std::numeric_limits<int32_t>::min();
    
    int32_t result = input[sel[0]];
    for (size_t i = 1; i < sel_count; ++i) {
        int32_t val = input[sel[i]];
        if (val > result) result = val;
    }
    return result;
}

// ============================================================================
// 分组聚合实现
// ============================================================================

void group_sum_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int64_t* out_sums) {
    // 初始化输出
    std::memset(out_sums, 0, num_groups * sizeof(int64_t));
    
    // 简单实现：直接累加到对应分组
    // TODO: 可以考虑分区处理以提高缓存局部性
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            out_sums[group_id] += values[i];
        }
    }
}

void group_sum_i64(const int64_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int64_t* out_sums) {
    std::memset(out_sums, 0, num_groups * sizeof(int64_t));
    
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            out_sums[group_id] += values[i];
        }
    }
}

void group_sum_f64(const double* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, double* out_sums) {
    std::memset(out_sums, 0, num_groups * sizeof(double));
    
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            out_sums[group_id] += values[i];
        }
    }
}

void group_count(const uint32_t* groups, size_t count, 
                 size_t num_groups, size_t* out_counts) {
    std::memset(out_counts, 0, num_groups * sizeof(size_t));
    
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            out_counts[group_id]++;
        }
    }
}

void group_min_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int32_t* out_mins) {
    // 初始化为最大值
    std::fill(out_mins, out_mins + num_groups, std::numeric_limits<int32_t>::max());
    
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            if (values[i] < out_mins[group_id]) {
                out_mins[group_id] = values[i];
            }
        }
    }
}

void group_max_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int32_t* out_maxs) {
    // 初始化为最小值
    std::fill(out_maxs, out_maxs + num_groups, std::numeric_limits<int32_t>::min());
    
    for (size_t i = 0; i < count; ++i) {
        uint32_t group_id = groups[i];
        if (group_id < num_groups) {
            if (values[i] > out_maxs[group_id]) {
                out_maxs[group_id] = values[i];
            }
        }
    }
}

// ============================================================================
// COUNT non-null 实现
// ============================================================================

size_t count_nonnull_i32(const int32_t* input, size_t count, int32_t null_value) {
    size_t result = 0;

#ifdef __aarch64__
    int32x4_t null_vec = vdupq_n_s32(null_value);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;
    
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        // 不等于 null 的掩码
        uint32x4_t mask = vmvnq_u32(vceqq_s32(data, null_vec));
        // 掩码右移 31 位得到 0 或 1
        uint32x4_t ones = vshrq_n_u32(mask, 31);
        count_vec = vaddq_u32(count_vec, ones);
    }
    
    result = vaddvq_u32(count_vec);
    
    for (; i < count; ++i) {
        if (input[i] != null_value) ++result;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        if (input[i] != null_value) ++result;
    }
#endif

    return result;
}

} // namespace aggregate
} // namespace thunderduck
