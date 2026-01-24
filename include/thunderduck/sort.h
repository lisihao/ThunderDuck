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

} // namespace sort
} // namespace thunderduck

#endif // THUNDERDUCK_SORT_H
