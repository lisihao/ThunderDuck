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

// ============================================================================
// v5.0 优化版本 - 掩码压缩LUT + 缓存对齐 + 字符串SIMD
// ============================================================================

/**
 * v5.0 优化版计数
 *
 * 优化特性:
 * - 128 字节缓存行对齐优化 (M4 L1)
 * - 多级预取 (L1 + L2)
 * - 对齐边界处理
 */
size_t count_i32_v5(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

/**
 * v5.0 优化版位图生成
 */
size_t filter_to_bitmap_v5(const int32_t* input, size_t count,
                            CompareOp op, int32_t value,
                            uint64_t* bitmap);

/**
 * v5.0 优化版过滤
 *
 * 优化特性:
 * - 4-bit LUT 掩码压缩 (加速 mask→indices)
 * - 缓存对齐位图分配
 * - 高选择率优化路径
 */
size_t filter_i32_v5(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// ============================================================================
// v6.0 优化版本 - V9.1 多级预取优化
// ============================================================================

/**
 * v6.0 优化版计数 - 多级预取优化
 *
 * 优化特性:
 * - 多级预取 (L1: 256B, L2: 512B)
 * - 基于 v3 架构
 * - 更激进预取距离
 */
size_t count_i32_v6(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

/**
 * v6.0 优化版过滤
 */
size_t filter_i32_v6(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// ============================================================================
// 字符串过滤 - SIMD 优化
// ============================================================================

/**
 * 字符串数组过滤 - 相等比较
 *
 * @param strings 字符串指针数组
 * @param count 字符串数量
 * @param target 目标字符串
 * @param out_indices 输出索引数组
 * @return 匹配数量
 */
size_t filter_string_eq(const char* const* strings, size_t count,
                         const char* target, uint32_t* out_indices);

/**
 * 字符串数组过滤 - 前缀匹配 (LIKE 'prefix%')
 */
size_t filter_string_startswith(const char* const* strings, size_t count,
                                 const char* prefix, uint32_t* out_indices);

/**
 * 字符串数组过滤 - 包含子串 (LIKE '%substr%')
 */
size_t filter_string_contains(const char* const* strings, size_t count,
                               const char* substr, uint32_t* out_indices);

// ============================================================================
// 缓存对齐辅助函数
// ============================================================================

/**
 * 检查指针是否 128 字节对齐
 */
bool is_filter_cache_aligned(const void* ptr);

/**
 * 分配 128 字节对齐内存
 */
void* filter_cache_alloc(size_t size);

/**
 * 释放缓存对齐内存
 */
void filter_cache_free(void* ptr);

// ============================================================================
// 多线程并行版本 (10M+ 数据优化)
// ============================================================================

/**
 * 多线程并行过滤 - 4 线程并行
 *
 * 当数据量 >= 1M 时自动启用多线程
 * 预期: 10M 数据 2.6x → 4-5x vs DuckDB
 */
size_t filter_i32_parallel(const int32_t* input, size_t count,
                            CompareOp op, int32_t value,
                            uint32_t* out_indices);

// ============================================================================
// v15.0 优化版本 - 直接 SIMD 索引生成 (跳过位图中间层)
// ============================================================================

/**
 * v15.0 直接索引生成过滤
 *
 * 优化特性:
 * - 跳过位图中间层，直接生成索引
 * - 使用 4-bit 掩码 LUT 压缩存储
 * - 本地缓冲区批量写入减少内存延迟
 * - 预期: 相比 V3 filter 3x+ 加速
 */
size_t filter_i32_v15(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices);

/**
 * 获取 V15 版本信息
 */
const char* get_filter_v15_version();

} // namespace filter
} // namespace thunderduck

#endif // THUNDERDUCK_FILTER_H
