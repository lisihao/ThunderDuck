#ifndef THUNDERDUCK_OPERATORS_H
#define THUNDERDUCK_OPERATORS_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace operators {

// Filter 算子
namespace filter {
    // SIMD 批量过滤 - 返回满足条件的元素数量
    size_t simd_filter_gt_i32(const int32_t* input, int32_t* output, 
                               size_t count, int32_t threshold);
    size_t simd_filter_eq_i32(const int32_t* input, int32_t* output,
                               size_t count, int32_t value);
}

// Aggregation 算子
namespace aggregate {
    // SIMD 聚合
    int64_t simd_sum_i32(const int32_t* input, size_t count);
    int64_t simd_sum_i64(const int64_t* input, size_t count);
    double  simd_sum_f64(const double* input, size_t count);
    
    int32_t simd_min_i32(const int32_t* input, size_t count);
    int32_t simd_max_i32(const int32_t* input, size_t count);
}

// Join 算子
namespace join {
    // SIMD 哈希计算
    void simd_hash_i32(const int32_t* keys, uint32_t* hashes, size_t count);
    void simd_hash_i64(const int64_t* keys, uint64_t* hashes, size_t count);
}

// Sort 算子
namespace sort {
    // SIMD 排序
    void simd_sort_i32(int32_t* data, size_t count);
    void simd_sort_i64(int64_t* data, size_t count);
}

} // namespace operators
} // namespace thunderduck

#endif // THUNDERDUCK_OPERATORS_H
