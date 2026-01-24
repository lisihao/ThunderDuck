/**
 * ThunderDuck - SIMD Utilities
 * 
 * ARM Neon intrinsics 封装，提供统一的 SIMD 操作接口
 */

#ifndef THUNDERDUCK_SIMD_H
#define THUNDERDUCK_SIMD_H

#include <cstdint>
#include <cstddef>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace simd {

// ============================================================================
// 类型别名
// ============================================================================

#ifdef __aarch64__
using v128_i8  = int8x16_t;    // 16 x int8
using v128_i16 = int16x8_t;    // 8 x int16
using v128_i32 = int32x4_t;    // 4 x int32
using v128_i64 = int64x2_t;    // 2 x int64
using v128_u8  = uint8x16_t;   // 16 x uint8
using v128_u16 = uint16x8_t;   // 8 x uint16
using v128_u32 = uint32x4_t;   // 4 x uint32
using v128_u64 = uint64x2_t;   // 2 x uint64
using v128_f32 = float32x4_t;  // 4 x float32
using v128_f64 = float64x2_t;  // 2 x float64
#endif

// ============================================================================
// 加载操作
// ============================================================================

#ifdef __aarch64__

// 对齐加载（假设指针已对齐）
inline v128_i32 load_i32(const int32_t* ptr) {
    return vld1q_s32(ptr);
}

inline v128_i64 load_i64(const int64_t* ptr) {
    return vld1q_s64(ptr);
}

inline v128_f32 load_f32(const float* ptr) {
    return vld1q_f32(ptr);
}

inline v128_f64 load_f64(const double* ptr) {
    return vld1q_f64(ptr);
}

inline v128_u32 load_u32(const uint32_t* ptr) {
    return vld1q_u32(ptr);
}

// ============================================================================
// 存储操作
// ============================================================================

inline void store_i32(int32_t* ptr, v128_i32 v) {
    vst1q_s32(ptr, v);
}

inline void store_i64(int64_t* ptr, v128_i64 v) {
    vst1q_s64(ptr, v);
}

inline void store_f32(float* ptr, v128_f32 v) {
    vst1q_f32(ptr, v);
}

inline void store_f64(double* ptr, v128_f64 v) {
    vst1q_f64(ptr, v);
}

inline void store_u32(uint32_t* ptr, v128_u32 v) {
    vst1q_u32(ptr, v);
}

// ============================================================================
// 广播（Splat）
// ============================================================================

inline v128_i32 splat_i32(int32_t value) {
    return vdupq_n_s32(value);
}

inline v128_i64 splat_i64(int64_t value) {
    return vdupq_n_s64(value);
}

inline v128_f32 splat_f32(float value) {
    return vdupq_n_f32(value);
}

inline v128_f64 splat_f64(double value) {
    return vdupq_n_f64(value);
}

inline v128_u32 splat_u32(uint32_t value) {
    return vdupq_n_u32(value);
}

// ============================================================================
// 比较操作（返回掩码）
// ============================================================================

// Greater Than
inline v128_u32 cmp_gt_i32(v128_i32 a, v128_i32 b) {
    return vcgtq_s32(a, b);
}

inline v128_u64 cmp_gt_i64(v128_i64 a, v128_i64 b) {
    return vcgtq_s64(a, b);
}

inline v128_u32 cmp_gt_f32(v128_f32 a, v128_f32 b) {
    return vcgtq_f32(a, b);
}

inline v128_u64 cmp_gt_f64(v128_f64 a, v128_f64 b) {
    return vcgtq_f64(a, b);
}

// Greater or Equal
inline v128_u32 cmp_ge_i32(v128_i32 a, v128_i32 b) {
    return vcgeq_s32(a, b);
}

inline v128_u64 cmp_ge_i64(v128_i64 a, v128_i64 b) {
    return vcgeq_s64(a, b);
}

inline v128_u32 cmp_ge_f32(v128_f32 a, v128_f32 b) {
    return vcgeq_f32(a, b);
}

// Less Than
inline v128_u32 cmp_lt_i32(v128_i32 a, v128_i32 b) {
    return vcltq_s32(a, b);
}

inline v128_u64 cmp_lt_i64(v128_i64 a, v128_i64 b) {
    return vcltq_s64(a, b);
}

inline v128_u32 cmp_lt_f32(v128_f32 a, v128_f32 b) {
    return vcltq_f32(a, b);
}

// Less or Equal
inline v128_u32 cmp_le_i32(v128_i32 a, v128_i32 b) {
    return vcleq_s32(a, b);
}

inline v128_u64 cmp_le_i64(v128_i64 a, v128_i64 b) {
    return vcleq_s64(a, b);
}

// Equal
inline v128_u32 cmp_eq_i32(v128_i32 a, v128_i32 b) {
    return vceqq_s32(a, b);
}

inline v128_u64 cmp_eq_i64(v128_i64 a, v128_i64 b) {
    return vceqq_s64(a, b);
}

inline v128_u32 cmp_eq_f32(v128_f32 a, v128_f32 b) {
    return vceqq_f32(a, b);
}

// Not Equal (通过取反实现)
inline v128_u32 cmp_ne_i32(v128_i32 a, v128_i32 b) {
    return vmvnq_u32(vceqq_s32(a, b));
}

// ============================================================================
// 算术操作
// ============================================================================

// 加法
inline v128_i32 add_i32(v128_i32 a, v128_i32 b) {
    return vaddq_s32(a, b);
}

inline v128_i64 add_i64(v128_i64 a, v128_i64 b) {
    return vaddq_s64(a, b);
}

inline v128_f32 add_f32(v128_f32 a, v128_f32 b) {
    return vaddq_f32(a, b);
}

inline v128_f64 add_f64(v128_f64 a, v128_f64 b) {
    return vaddq_f64(a, b);
}

// 减法
inline v128_i32 sub_i32(v128_i32 a, v128_i32 b) {
    return vsubq_s32(a, b);
}

inline v128_f32 sub_f32(v128_f32 a, v128_f32 b) {
    return vsubq_f32(a, b);
}

// 乘法
inline v128_i32 mul_i32(v128_i32 a, v128_i32 b) {
    return vmulq_s32(a, b);
}

inline v128_f32 mul_f32(v128_f32 a, v128_f32 b) {
    return vmulq_f32(a, b);
}

inline v128_f64 mul_f64(v128_f64 a, v128_f64 b) {
    return vmulq_f64(a, b);
}

// 除法（仅浮点）
inline v128_f32 div_f32(v128_f32 a, v128_f32 b) {
    return vdivq_f32(a, b);
}

inline v128_f64 div_f64(v128_f64 a, v128_f64 b) {
    return vdivq_f64(a, b);
}

// ============================================================================
// Min/Max 操作
// ============================================================================

inline v128_i32 min_i32(v128_i32 a, v128_i32 b) {
    return vminq_s32(a, b);
}

inline v128_i32 max_i32(v128_i32 a, v128_i32 b) {
    return vmaxq_s32(a, b);
}

inline v128_f32 min_f32(v128_f32 a, v128_f32 b) {
    return vminq_f32(a, b);
}

inline v128_f32 max_f32(v128_f32 a, v128_f32 b) {
    return vmaxq_f32(a, b);
}

inline v128_f64 min_f64(v128_f64 a, v128_f64 b) {
    return vminq_f64(a, b);
}

inline v128_f64 max_f64(v128_f64 a, v128_f64 b) {
    return vmaxq_f64(a, b);
}

// ============================================================================
// 水平归约
// ============================================================================

// 水平求和
inline int32_t reduce_add_i32(v128_i32 v) {
    return vaddvq_s32(v);
}

inline int64_t reduce_add_i64(v128_i64 v) {
    return vaddvq_s64(v);
}

inline float reduce_add_f32(v128_f32 v) {
    return vaddvq_f32(v);
}

inline double reduce_add_f64(v128_f64 v) {
    return vaddvq_f64(v);
}

// 水平最小值
inline int32_t reduce_min_i32(v128_i32 v) {
    return vminvq_s32(v);
}

inline float reduce_min_f32(v128_f32 v) {
    return vminvq_f32(v);
}

// 水平最大值
inline int32_t reduce_max_i32(v128_i32 v) {
    return vmaxvq_s32(v);
}

inline float reduce_max_f32(v128_f32 v) {
    return vmaxvq_f32(v);
}

inline uint32_t reduce_max_u32(v128_u32 v) {
    return vmaxvq_u32(v);
}

// ============================================================================
// 位操作
// ============================================================================

inline v128_u32 and_u32(v128_u32 a, v128_u32 b) {
    return vandq_u32(a, b);
}

inline v128_u32 or_u32(v128_u32 a, v128_u32 b) {
    return vorrq_u32(a, b);
}

inline v128_u32 xor_u32(v128_u32 a, v128_u32 b) {
    return veorq_u32(a, b);
}

inline v128_u32 not_u32(v128_u32 a) {
    return vmvnq_u32(a);
}

// ============================================================================
// 掩码操作
// ============================================================================

// 提取掩码位（每个 32-bit 元素的最高位）
inline uint32_t movemask_u32(v128_u32 v) {
    // ARM 没有直接的 movemask，需要手动实现
    // 将每个 lane 的符号位提取出来
    static const int32_t shift_arr[4] = {0, 1, 2, 3};
    v128_i32 shift = vld1q_s32(shift_arr);
    
    // 右移 31 位得到符号位（0 或 1）
    v128_u32 signs = vshrq_n_u32(v, 31);
    // 左移不同位数
    v128_u32 shifted = vshlq_u32(signs, vreinterpretq_s32_u32(vld1q_u32((uint32_t*)shift_arr)));
    // 水平加得到最终掩码
    return vaddvq_u32(shifted);
}

// 检查是否全为零
inline bool all_zeros_u32(v128_u32 v) {
    return vmaxvq_u32(v) == 0;
}

// 检查是否全不为零
inline bool all_ones_u32(v128_u32 v) {
    return vminvq_u32(v) == 0xFFFFFFFF;
}

// ============================================================================
// 选择操作（基于掩码）
// ============================================================================

inline v128_i32 select_i32(v128_u32 mask, v128_i32 a, v128_i32 b) {
    return vbslq_s32(mask, a, b);
}

inline v128_f32 select_f32(v128_u32 mask, v128_f32 a, v128_f32 b) {
    return vbslq_f32(mask, a, b);
}

#endif // __aarch64__

// ============================================================================
// 非 SIMD 回退实现
// ============================================================================

#ifndef __aarch64__

// 标量回退 - 用于非 ARM 平台编译
inline int32_t reduce_add_i32_scalar(const int32_t* data, size_t count) {
    int32_t sum = 0;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum;
}

#endif

} // namespace simd
} // namespace thunderduck

#endif // THUNDERDUCK_SIMD_H
