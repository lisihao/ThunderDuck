/**
 * ThunderDuck - SIMD Aggregation Implementation v2.0
 *
 * 优化版本：
 * - 合并的 minmax 函数
 * - 16 元素/迭代
 * - 预取优化
 */

#include "thunderduck/aggregate.h"
#include "thunderduck/simd.h"
#include "thunderduck/memory.h"
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 合并的 MIN/MAX 函数
// ============================================================================

void minmax_i32(const int32_t* input, size_t count,
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

    // 主循环：每次处理 16 个元素
    for (; i + 16 <= count; i += 16) {
        // 预取
        __builtin_prefetch(input + i + 64, 0, 0);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 先合并 4 个向量
        int32x4_t min_01 = vminq_s32(d0, d1);
        int32x4_t min_23 = vminq_s32(d2, d3);
        int32x4_t min_batch = vminq_s32(min_01, min_23);

        int32x4_t max_01 = vmaxq_s32(d0, d1);
        int32x4_t max_23 = vmaxq_s32(d2, d3);
        int32x4_t max_batch = vmaxq_s32(max_01, max_23);

        // 更新全局 min/max
        min_vec = vminq_s32(min_vec, min_batch);
        max_vec = vmaxq_s32(max_vec, max_batch);
    }

    // 处理剩余的 4 元素块
    for (; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        min_vec = vminq_s32(min_vec, data);
        max_vec = vmaxq_s32(max_vec, data);
    }

    // 水平归约
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

void minmax_i64(const int64_t* input, size_t count,
                int64_t* out_min, int64_t* out_max) {
    if (count == 0) {
        *out_min = std::numeric_limits<int64_t>::max();
        *out_max = std::numeric_limits<int64_t>::min();
        return;
    }

#ifdef __aarch64__
    int64x2_t min_vec = vdupq_n_s64(std::numeric_limits<int64_t>::max());
    int64x2_t max_vec = vdupq_n_s64(std::numeric_limits<int64_t>::min());
    size_t i = 0;

    // 处理每批 8 个元素
    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(input + i + 32, 0, 0);

        int64x2_t d0 = vld1q_s64(input + i);
        int64x2_t d1 = vld1q_s64(input + i + 2);
        int64x2_t d2 = vld1q_s64(input + i + 4);
        int64x2_t d3 = vld1q_s64(input + i + 6);

        // 使用比较和选择实现 min/max (ARM64 没有 vminq_s64)
        uint64x2_t cmp0 = vcltq_s64(d0, min_vec);
        uint64x2_t cmp1 = vcltq_s64(d1, min_vec);
        min_vec = vbslq_s64(cmp0, d0, min_vec);
        min_vec = vbslq_s64(cmp1, d1, min_vec);

        cmp0 = vcltq_s64(d2, min_vec);
        cmp1 = vcltq_s64(d3, min_vec);
        min_vec = vbslq_s64(cmp0, d2, min_vec);
        min_vec = vbslq_s64(cmp1, d3, min_vec);

        cmp0 = vcgtq_s64(d0, max_vec);
        cmp1 = vcgtq_s64(d1, max_vec);
        max_vec = vbslq_s64(cmp0, d0, max_vec);
        max_vec = vbslq_s64(cmp1, d1, max_vec);

        cmp0 = vcgtq_s64(d2, max_vec);
        cmp1 = vcgtq_s64(d3, max_vec);
        max_vec = vbslq_s64(cmp0, d2, max_vec);
        max_vec = vbslq_s64(cmp1, d3, max_vec);
    }

    // 水平归约
    int64_t min0 = vgetq_lane_s64(min_vec, 0);
    int64_t min1 = vgetq_lane_s64(min_vec, 1);
    int64_t max0 = vgetq_lane_s64(max_vec, 0);
    int64_t max1 = vgetq_lane_s64(max_vec, 1);

    *out_min = min0 < min1 ? min0 : min1;
    *out_max = max0 > max1 ? max0 : max1;

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

void minmax_f32(const float* input, size_t count,
                float* out_min, float* out_max) {
    if (count == 0) {
        *out_min = std::numeric_limits<float>::max();
        *out_max = std::numeric_limits<float>::lowest();
        return;
    }

#ifdef __aarch64__
    float32x4_t min_vec = vdupq_n_f32(std::numeric_limits<float>::max());
    float32x4_t max_vec = vdupq_n_f32(std::numeric_limits<float>::lowest());
    size_t i = 0;

    for (; i + 16 <= count; i += 16) {
        __builtin_prefetch(input + i + 64, 0, 0);

        float32x4_t d0 = vld1q_f32(input + i);
        float32x4_t d1 = vld1q_f32(input + i + 4);
        float32x4_t d2 = vld1q_f32(input + i + 8);
        float32x4_t d3 = vld1q_f32(input + i + 12);

        float32x4_t min_01 = vminq_f32(d0, d1);
        float32x4_t min_23 = vminq_f32(d2, d3);
        float32x4_t min_batch = vminq_f32(min_01, min_23);

        float32x4_t max_01 = vmaxq_f32(d0, d1);
        float32x4_t max_23 = vmaxq_f32(d2, d3);
        float32x4_t max_batch = vmaxq_f32(max_01, max_23);

        min_vec = vminq_f32(min_vec, min_batch);
        max_vec = vmaxq_f32(max_vec, max_batch);
    }

    for (; i + 4 <= count; i += 4) {
        float32x4_t data = vld1q_f32(input + i);
        min_vec = vminq_f32(min_vec, data);
        max_vec = vmaxq_f32(max_vec, data);
    }

    *out_min = vminvq_f32(min_vec);
    *out_max = vmaxvq_f32(max_vec);

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
// 优化的 SUM 函数 - 16 元素/迭代 + 预取
// ============================================================================

int64_t sum_i32_v2(const int32_t* input, size_t count) {
#ifdef __aarch64__
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int64x2_t sum2 = vdupq_n_s64(0);
    int64x2_t sum3 = vdupq_n_s64(0);
    size_t i = 0;

    // 主循环：每次处理 16 个元素
    for (; i + 16 <= count; i += 16) {
        __builtin_prefetch(input + i + 64, 0, 0);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 扩展到 64 位并累加
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

double sum_f32_v2(const float* input, size_t count) {
#ifdef __aarch64__
    float64x2_t sum0 = vdupq_n_f64(0);
    float64x2_t sum1 = vdupq_n_f64(0);
    size_t i = 0;

    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(input + i + 32, 0, 0);

        float32x4_t d0 = vld1q_f32(input + i);
        float32x4_t d1 = vld1q_f32(input + i + 4);

        // 扩展到 64 位
        float64x2_t d0_lo = vcvt_f64_f32(vget_low_f32(d0));
        float64x2_t d0_hi = vcvt_f64_f32(vget_high_f32(d0));
        float64x2_t d1_lo = vcvt_f64_f32(vget_low_f32(d1));
        float64x2_t d1_hi = vcvt_f64_f32(vget_high_f32(d1));

        sum0 = vaddq_f64(sum0, vaddq_f64(d0_lo, d0_hi));
        sum1 = vaddq_f64(sum1, vaddq_f64(d1_lo, d1_hi));
    }

    double result = vaddvq_f64(vaddq_f64(sum0, sum1));

    for (; i < count; ++i) {
        result += input[i];
    }

    return result;
#else
    double result = 0;
    for (size_t i = 0; i < count; ++i) {
        result += input[i];
    }
    return result;
#endif
}

// ============================================================================
// 优化的 AVG 函数 - Kahan 求和提高精度
// ============================================================================

double avg_f64_kahan(const double* input, size_t count) {
    if (count == 0) return 0.0;

    double sum = 0.0;
    double c = 0.0;  // 补偿项

    for (size_t i = 0; i < count; ++i) {
        double y = input[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum / static_cast<double>(count);
}

// ============================================================================
// 多聚合函数 - 一次遍历计算多个统计量
// ============================================================================

AggregateStats aggregate_all_i32(const int32_t* input, size_t count) {
    AggregateStats stats = {0, 0, 0, 0};
    if (count == 0) return stats;

    stats.count = static_cast<int64_t>(count);

#ifdef __aarch64__
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);
    int32x4_t min_vec = vdupq_n_s32(std::numeric_limits<int32_t>::max());
    int32x4_t max_vec = vdupq_n_s32(std::numeric_limits<int32_t>::min());
    size_t i = 0;

    for (; i + 8 <= count; i += 8) {
        __builtin_prefetch(input + i + 32, 0, 0);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);

        // SUM
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d0));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d1));

        // MIN/MAX
        int32x4_t batch_min = vminq_s32(d0, d1);
        int32x4_t batch_max = vmaxq_s32(d0, d1);
        min_vec = vminq_s32(min_vec, batch_min);
        max_vec = vmaxq_s32(max_vec, batch_max);
    }

    stats.sum = vaddvq_s64(vaddq_s64(sum0, sum1));
    stats.min_val = vminvq_s32(min_vec);
    stats.max_val = vmaxvq_s32(max_vec);

    for (; i < count; ++i) {
        stats.sum += input[i];
        if (input[i] < stats.min_val) stats.min_val = input[i];
        if (input[i] > stats.max_val) stats.max_val = input[i];
    }
#else
    stats.min_val = input[0];
    stats.max_val = input[0];
    stats.sum = 0;
    for (size_t i = 0; i < count; ++i) {
        stats.sum += input[i];
        if (input[i] < stats.min_val) stats.min_val = input[i];
        if (input[i] > stats.max_val) stats.max_val = input[i];
    }
#endif

    return stats;
}

} // namespace aggregate
} // namespace thunderduck
