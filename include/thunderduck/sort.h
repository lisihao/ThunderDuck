/**
 * ThunderDuck - Sort Operator
 * 
 * SIMD 加速的排序算子，使用 Bitonic Sort 和分块合并
 */

#ifndef THUNDERDUCK_SORT_H
#define THUNDERDUCK_SORT_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace sort {

// ============================================================================
// 排序方向
// ============================================================================

enum class SortOrder {
    ASC,   // 升序
    DESC   // 降序
};

// ============================================================================
// 基础排序函数
// ============================================================================

/**
 * 原地排序 int32 数组
 */
void sort_i32(int32_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 原地排序 int64 数组
 */
void sort_i64(int64_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 原地排序 float 数组
 */
void sort_f32(float* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 原地排序 double 数组
 */
void sort_f64(double* data, size_t count, SortOrder order = SortOrder::ASC);

// ============================================================================
// 带索引排序（argsort）
// ============================================================================

/**
 * 返回排序后的索引（不修改原数组）
 * 
 * @param data 输入数组
 * @param count 元素数量
 * @param out_indices 输出索引数组（需预分配）
 * @param order 排序方向
 */
void argsort_i32(const int32_t* data, size_t count, 
                 uint32_t* out_indices, SortOrder order = SortOrder::ASC);

void argsort_i64(const int64_t* data, size_t count, 
                 uint32_t* out_indices, SortOrder order = SortOrder::ASC);

void argsort_f32(const float* data, size_t count, 
                 uint32_t* out_indices, SortOrder order = SortOrder::ASC);

void argsort_f64(const double* data, size_t count, 
                 uint32_t* out_indices, SortOrder order = SortOrder::ASC);

// ============================================================================
// 部分排序（Top-K）
// ============================================================================

/**
 * 获取最小的 K 个元素
 * 
 * @param data 输入数组
 * @param count 元素数量
 * @param k 要获取的元素数量
 * @param out_values 输出值数组（需预分配 k 个元素）
 * @param out_indices 输出索引数组（可选，需预分配 k 个元素）
 */
void topk_min_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices = nullptr);

void topk_min_f32(const float* data, size_t count, size_t k,
                  float* out_values, uint32_t* out_indices = nullptr);

/**
 * 获取最大的 K 个元素
 */
void topk_max_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices = nullptr);

void topk_max_f32(const float* data, size_t count, size_t k,
                  float* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// 键值对排序
// ============================================================================

/**
 * 按键排序，同时重排值
 * 
 * @param keys 键数组（会被排序）
 * @param values 值数组（会被重排）
 * @param count 元素数量
 * @param order 排序方向
 */
void sort_pairs_i32_i32(int32_t* keys, int32_t* values, size_t count,
                        SortOrder order = SortOrder::ASC);

void sort_pairs_i32_i64(int32_t* keys, int64_t* values, size_t count,
                        SortOrder order = SortOrder::ASC);

void sort_pairs_i64_i64(int64_t* keys, int64_t* values, size_t count,
                        SortOrder order = SortOrder::ASC);

// ============================================================================
// SIMD 内部排序（小数组）
// ============================================================================

/**
 * SIMD Bitonic Sort - 适用于小数组（<= 1024 元素）
 */
void bitonic_sort_i32(int32_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * SIMD 排序 4 个元素（单个向量）
 */
void sort_4_i32(int32_t* data, SortOrder order = SortOrder::ASC);

/**
 * SIMD 排序 8 个元素（两个向量）
 */
void sort_8_i32(int32_t* data, SortOrder order = SortOrder::ASC);

/**
 * SIMD 排序 16 个元素（四个向量）
 */
void sort_16_i32(int32_t* data, SortOrder order = SortOrder::ASC);

// ============================================================================
// 多路合并
// ============================================================================

/**
 * 合并两个已排序数组
 */
void merge_sorted_i32(const int32_t* a, size_t a_count,
                      const int32_t* b, size_t b_count,
                      int32_t* out, SortOrder order = SortOrder::ASC);

/**
 * 多路合并（K 个已排序数组）
 */
void merge_k_sorted_i32(const int32_t** arrays, const size_t* counts, size_t k,
                        int32_t* out, SortOrder order = SortOrder::ASC);

// ============================================================================
// v2.0 优化版本 - Radix Sort
// ============================================================================

/**
 * Radix Sort - O(n) 整数排序
 */
void radix_sort_i32(int32_t* data, size_t count, SortOrder order = SortOrder::ASC);
void radix_sort_u32(uint32_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 优化版 Radix Sort - 11-11-10 位分组（3趟）
 */
void radix_sort_i32_v2(int32_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 优化的主排序函数 - 自动选择最佳算法
 */
void sort_i32_v2(int32_t* data, size_t count, SortOrder order = SortOrder::ASC);

/**
 * 优化版 Top-K - 堆/快速选择混合
 */
void topk_max_i32_v2(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

void topk_min_i32_v2(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// v3.0 优化版本 - 自适应 TopK
// ============================================================================

/**
 * v3.0 优化版 Top-K Max
 *
 * 自适应策略选择：
 * - K ≤ 64: 纯堆方法 (L1 常驻)
 * - 64 < K ≤ 1024: SIMD 加速堆
 * - 1024 < K ≤ 4096: 分块处理
 * - K > 4096: 无复制 nth_element
 *
 * @param data 输入数组
 * @param count 元素数量
 * @param k 要获取的元素数量
 * @param out_values 输出值数组
 * @param out_indices 输出索引数组（可选）
 */
void topk_max_i32_v3(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

/**
 * v3.0 优化版 Top-K Min
 */
void topk_min_i32_v3(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// v4.0 优化版本 - 采样预过滤 (针对大 N 小 K 场景)
// ============================================================================

/**
 * v4.0 优化版 Top-K Max
 *
 * 核心优化: 采样预过滤 + SIMD 批量跳过
 *
 * 针对 T4 场景 (10M 行, K=10) 的专项优化:
 * - 采样估计第 K 大值作为阈值
 * - SIMD 批量预过滤，跳过 ~90% 的元素
 * - 只对候选元素进行最终选择
 *
 * 策略选择:
 * - N >= 1M 且 K <= 64: 采样预过滤 (专为大数据小K优化)
 * - K <= 64: 纯堆方法 (L1 常驻)
 * - 64 < K <= 1024: SIMD 加速堆
 * - K > 1024: 无复制 nth_element
 *
 * @param data 输入数组
 * @param count 元素数量
 * @param k 要获取的元素数量
 * @param out_values 输出值数组
 * @param out_indices 输出索引数组（可选）
 */
void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

/**
 * v4.0 优化版 Top-K Min
 */
void topk_min_i32_v4(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// v5.0 优化版本 - Count-Based TopK (低基数全面优化)
// ============================================================================

/**
 * v5.0 优化版 Top-K Max
 *
 * 核心优化: 自适应基数检测 + Count-Based TopK
 *
 * 算法:
 * 1. 快速基数估计 (采样 + 去重统计)
 * 2. 低基数 (< 10000): Count-Based 方法
 *    - 统计每个值的出现次数 O(n)
 *    - 在唯一值集合上找 TopK O(cardinality)
 * 3. 高基数: v4 采样预过滤
 *
 * 低基数优势:
 * - 10M 行，基数 100: 从 3ms 降到 ~0.5ms
 * - 核心: 在 100 个元素上操作，而非 10M
 *
 * @param data 输入数组
 * @param count 元素数量
 * @param k 要获取的元素数量
 * @param out_values 输出值数组
 * @param out_indices 输出索引数组（可选）
 */
void topk_max_i32_v5(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

/**
 * v5.0 优化版 Top-K Min
 */
void topk_min_i32_v5(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// v6.0 UMA 优化版本 - GPU 加速 + 零拷贝
// ============================================================================

/**
 * TopK 策略
 */
enum class TopKStrategy {
    AUTO,           // 自动选择
    CPU_HEAP,       // CPU 堆方法
    CPU_SAMPLE,     // CPU 采样预过滤 (v4)
    CPU_COUNT,      // CPU Count-Based (v5)
    GPU_BITONIC,    // GPU Bitonic Sort + 截断
    GPU_FILTER,     // GPU 并行过滤 + 小规模排序
};

/**
 * v6.0 TopK 配置
 */
struct TopKConfigV6 {
    TopKStrategy strategy = TopKStrategy::AUTO;
    float cardinality_hint = -1.0f;  // 预估基数比例 (0-1), -1 表示未知
};

/**
 * 检查 GPU TopK 是否可用
 */
bool is_topk_gpu_available();

/**
 * v6.0 UMA 优化版 Top-K Max
 *
 * 核心优化: UMA 零拷贝 + GPU 并行
 *
 * 策略选择:
 * - N < 100K: CPU 方法 (v5)
 * - N >= 100K, K <= 64: GPU 并行过滤
 * - N >= 100K, K > 64: GPU Bitonic Sort
 *
 * @param data 输入数组 (建议页对齐 16KB)
 * @param count 元素数量
 * @param k 要获取的元素数量
 * @param out_values 输出值数组
 * @param out_indices 输出索引数组（可选）
 */
void topk_max_i32_v6(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

void topk_max_i32_v6_config(const int32_t* data, size_t count, size_t k,
                             int32_t* out_values, uint32_t* out_indices,
                             const TopKConfigV6& config);

/**
 * v6.0 UMA 优化版 Top-K Min
 */
void topk_min_i32_v6(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices = nullptr);

// ============================================================================
// V13 GPU TopK - P3 优化
// ============================================================================

/**
 * 检查 V13 GPU TopK 是否可用
 */
bool is_topk_v13_gpu_available();

/**
 * V13 GPU TopK Max - 并行选择算法
 *
 * 算法: 分层选择 + Bitonic Sort
 * - Phase 1: 每个 threadgroup 找本地 TopK
 * - Phase 2: 合并所有 threadgroup 结果
 *
 * 目标: CPU 4x → GPU 5x+
 */
void topk_max_i32_v13(const int32_t* data, size_t count, size_t k,
                       int32_t* out_values, uint32_t* out_indices = nullptr);

void topk_min_i32_v13(const int32_t* data, size_t count, size_t k,
                       int32_t* out_values, uint32_t* out_indices = nullptr);

} // namespace sort
} // namespace thunderduck

#endif // THUNDERDUCK_SORT_H
