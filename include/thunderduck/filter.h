/**
 * ThunderDuck - Filter Operator
 * 
 * SIMD 加速的过滤算子，支持各种比较操作
 */

#ifndef THUNDERDUCK_FILTER_H
#define THUNDERDUCK_FILTER_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace filter {

// ============================================================================
// 比较类型枚举
// ============================================================================

enum class CompareOp {
    EQ,   // ==
    NE,   // !=
    LT,   // <
    LE,   // <=
    GT,   // >
    GE    // >=
};

// ============================================================================
// 过滤结果结构
// ============================================================================

struct FilterResult {
    size_t count;           // 满足条件的元素数量
    size_t* indices;        // 满足条件的元素索引（可选）
};

// ============================================================================
// SIMD 过滤函数 - int32
// ============================================================================

/**
 * 过滤 int32 数组，返回满足条件的元素索引
 * 
 * @param input 输入数组（建议 128 字节对齐）
 * @param count 输入数量
 * @param op 比较操作
 * @param value 比较值
 * @param out_indices 输出索引数组（需预分配足够空间）
 * @return 满足条件的元素数量
 */
size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices);

/**
 * 过滤 int32 数组，直接输出满足条件的值
 */
size_t filter_i32_values(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         int32_t* out_values);

/**
 * 生成过滤位掩码
 * 
 * @param input 输入数组
 * @param count 输入数量
 * @param op 比较操作
 * @param value 比较值
 * @param out_mask 输出位掩码（1 bit per element）
 * @return 满足条件的元素数量
 */
size_t filter_i32_mask(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint64_t* out_mask);

// ============================================================================
// SIMD 过滤函数 - int64
// ============================================================================

size_t filter_i64(const int64_t* input, size_t count,
                  CompareOp op, int64_t value,
                  uint32_t* out_indices);

size_t filter_i64_values(const int64_t* input, size_t count,
                         CompareOp op, int64_t value,
                         int64_t* out_values);

// ============================================================================
// SIMD 过滤函数 - float
// ============================================================================

size_t filter_f32(const float* input, size_t count,
                  CompareOp op, float value,
                  uint32_t* out_indices);

size_t filter_f32_values(const float* input, size_t count,
                         CompareOp op, float value,
                         float* out_values);

// ============================================================================
// SIMD 过滤函数 - double
// ============================================================================

size_t filter_f64(const double* input, size_t count,
                  CompareOp op, double value,
                  uint32_t* out_indices);

size_t filter_f64_values(const double* input, size_t count,
                         CompareOp op, double value,
                         double* out_values);

// ============================================================================
// 复合过滤（AND/OR）
// ============================================================================

/**
 * AND 复合过滤：同时满足两个条件
 */
size_t filter_i32_and(const int32_t* input, size_t count,
                      CompareOp op1, int32_t value1,
                      CompareOp op2, int32_t value2,
                      uint32_t* out_indices);

/**
 * OR 复合过滤：满足任一条件
 */
size_t filter_i32_or(const int32_t* input, size_t count,
                     CompareOp op1, int32_t value1,
                     CompareOp op2, int32_t value2,
                     uint32_t* out_indices);

/**
 * 范围过滤：value1 <= x < value2
 */
size_t filter_i32_range(const int32_t* input, size_t count,
                        int32_t low, int32_t high,
                        uint32_t* out_indices);

// ============================================================================
// 统计函数
// ============================================================================

/**
 * 仅计数满足条件的元素（不输出索引）
 */
size_t count_i32(const int32_t* input, size_t count,
                 CompareOp op, int32_t value);

size_t count_f32(const float* input, size_t count,
                 CompareOp op, float value);

// ============================================================================
// v2.0 优化版本
// ============================================================================

/**
 * 优化版计数函数 - 16 元素/迭代 + 预取
 */
size_t count_i32_v2(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

/**
 * 优化版范围计数
 */
size_t count_i32_range_v2(const int32_t* input, size_t count,
                           int32_t low, int32_t high);

/**
 * 位图过滤 - 生成位图
 */
size_t filter_to_bitmap_i32(const int32_t* input, size_t count,
                             CompareOp op, int32_t value,
                             uint64_t* bitmap);

/**
 * 位图转索引
 */
size_t bitmap_to_indices(const uint64_t* bitmap, size_t bit_count,
                         uint32_t* out_indices);

/**
 * 优化版过滤 - 使用位图中间表示
 */
size_t filter_i32_v2(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// ============================================================================
// v3.0 优化版本 - 模板特化 + 独立累加器 + vsub 优化
// ============================================================================

/**
 * v3.0 优化版计数函数
 *
 * 优化特性：
 * - 模板特化消除循环内 switch-case
 * - 4 独立累加器消除依赖链 (ILP)
 * - vsub 替代 vshr+vadd 减少指令
 * - 256 元素批次处理减少归约次数
 * - 自适应预取策略
 *
 * @param input 输入数组（建议 128 字节对齐）
 * @param count 输入数量
 * @param op 比较操作
 * @param value 比较值
 * @return 满足条件的元素数量
 */
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

/**
 * v3.0 优化版范围计数
 */
size_t count_i32_range_v3(const int32_t* input, size_t count,
                           int32_t low, int32_t high);

/**
 * v3.0 优化版位图过滤
 */
size_t filter_to_bitmap_v3(const int32_t* input, size_t count,
                            CompareOp op, int32_t value,
                            uint64_t* bitmap);

/**
 * v3.0 优化版过滤
 */
size_t filter_i32_v3(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// ============================================================================
// v4.0 UMA 优化版本 - GPU 加速 + 零拷贝
// ============================================================================

/**
 * 过滤策略
 */
enum class FilterStrategy {
    AUTO,       // 自动选择
    CPU_SIMD,   // CPU SIMD (v3)
    GPU_ATOMIC, // GPU 原子版 (适合低选择率)
    GPU_SCAN,   // GPU 前缀和版 (适合高选择率)
};

/**
 * v4.0 过滤配置
 */
struct FilterConfigV4 {
    FilterStrategy strategy = FilterStrategy::AUTO;
    float selectivity_hint = -1.0f;  // 预估选择率 (0-1), -1 表示未知
};

/**
 * 检查 GPU 过滤是否可用
 */
bool is_filter_gpu_available();

/**
 * v4.0 UMA 优化版过滤
 *
 * 特性:
 * - 零拷贝数据传输 (UMA)
 * - 自动选择 CPU/GPU
 * - GPU 原子或前缀和策略
 *
 * @param input 输入数组 (建议页对齐 16KB)
 * @param count 输入数量
 * @param op 比较操作
 * @param value 比较值
 * @param out_indices 输出索引数组
 * @return 满足条件的元素数量
 */
size_t filter_i32_v4(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

size_t filter_i32_v4_config(const int32_t* input, size_t count,
                             CompareOp op, int32_t value,
                             uint32_t* out_indices,
                             const FilterConfigV4& config);

/**
 * v4.0 UMA 优化版计数
 */
size_t count_i32_v4(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

/**
 * v4.0 范围过滤
 */
size_t filter_i32_range_v4(const int32_t* input, size_t count,
                            int32_t low, int32_t high,
                            uint32_t* out_indices);

/**
 * v4.0 浮点过滤
 */
size_t filter_f32_v4(const float* input, size_t count,
                      CompareOp op, float value,
                      uint32_t* out_indices);

} // namespace filter
} // namespace thunderduck

#endif // THUNDERDUCK_FILTER_H
