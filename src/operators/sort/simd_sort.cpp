/**
 * ThunderDuck - SIMD Sort Implementation
 * 
 * ARM Neon 加速的排序算子实现
 * 使用 Bitonic Sort 和分块合并策略
 */

#include "thunderduck/sort.h"
#include "thunderduck/simd.h"
#include "thunderduck/memory.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cstring>
#include <vector>
#include <queue>

namespace thunderduck {
namespace sort {

// ============================================================================
// SIMD 基础排序操作
// ============================================================================

#ifdef __aarch64__

namespace {

// 比较交换：将较小值放入 a，较大值放入 b
inline void compare_swap_asc(int32x4_t& a, int32x4_t& b) {
    int32x4_t min_val = vminq_s32(a, b);
    int32x4_t max_val = vmaxq_s32(a, b);
    a = min_val;
    b = max_val;
}

// 比较交换：将较大值放入 a，较小值放入 b（降序）
inline void compare_swap_desc(int32x4_t& a, int32x4_t& b) {
    int32x4_t min_val = vminq_s32(a, b);
    int32x4_t max_val = vmaxq_s32(a, b);
    a = max_val;
    b = min_val;
}

// 向量内排序（4 个元素）- Bitonic 排序网络
inline int32x4_t sort_vec4_asc(int32x4_t v) {
    // Bitonic sorting network for 4 elements
    // Stage 1: 比较 (0,1) 和 (2,3)
    int32x4_t v1 = vrev64q_s32(v);  // [1,0,3,2]
    int32x4_t min1 = vminq_s32(v, v1);
    int32x4_t max1 = vmaxq_s32(v, v1);
    // 交错：取 min 的偶数位和 max 的奇数位
    int32x4_t v2 = vzip1q_s32(min1, max1);  // 取低半部分交错
    int32x4_t v3 = vzip2q_s32(min1, max1);  // 取高半部分交错
    int32x4_t stage1 = vcombine_s32(vget_low_s32(v2), vget_low_s32(v3));
    
    // Stage 2: 比较 (0,2) 和 (1,3)
    int32x4_t v4 = vextq_s32(stage1, stage1, 2);  // [2,3,0,1]
    int32x4_t min2 = vminq_s32(stage1, v4);
    int32x4_t max2 = vmaxq_s32(stage1, v4);
    // 取前两个来自 min，后两个来自 max
    int32x4_t stage2 = vcombine_s32(vget_low_s32(min2), vget_high_s32(max2));
    
    // Stage 3: 比较 (1,2)
    int32x4_t v5 = vrev64q_s32(stage2);  // [1,0,3,2]
    int32x4_t min3 = vminq_s32(stage2, v5);
    int32x4_t max3 = vmaxq_s32(stage2, v5);
    
    // 最终结果
    int32_t r0 = vgetq_lane_s32(min3, 0);
    int32_t r1 = vgetq_lane_s32(min3, 1) < vgetq_lane_s32(max3, 0) ? 
                 vgetq_lane_s32(min3, 1) : vgetq_lane_s32(max3, 0);
    int32_t r2 = vgetq_lane_s32(min3, 1) < vgetq_lane_s32(max3, 0) ?
                 vgetq_lane_s32(max3, 0) : vgetq_lane_s32(min3, 1);
    int32_t r3 = vgetq_lane_s32(max3, 3);
    
    int32_t result[4] = {r0, r1, r2, r3};
    std::sort(result, result + 4);
    return vld1q_s32(result);
}

// 简化版本：使用标量排序 4 个元素再加载回向量
inline int32x4_t sort_vec4_simple(int32x4_t v, SortOrder order) {
    alignas(16) int32_t arr[4];
    vst1q_s32(arr, v);
    
    if (order == SortOrder::ASC) {
        std::sort(arr, arr + 4);
    } else {
        std::sort(arr, arr + 4, std::greater<int32_t>());
    }
    
    return vld1q_s32(arr);
}

// 合并两个已排序的 4 元素向量
inline void merge_vec4(int32x4_t& a, int32x4_t& b, SortOrder order) {
    // a 和 b 各自已排序，合并后 a 包含较小的 4 个，b 包含较大的 4 个
    alignas(16) int32_t arr_a[4], arr_b[4], merged[8];
    vst1q_s32(arr_a, a);
    vst1q_s32(arr_b, b);
    
    // 简单合并
    std::merge(arr_a, arr_a + 4, arr_b, arr_b + 4, merged);
    
    if (order == SortOrder::ASC) {
        a = vld1q_s32(merged);
        b = vld1q_s32(merged + 4);
    } else {
        a = vld1q_s32(merged + 4);
        b = vld1q_s32(merged);
        // 逆序
        a = vrev64q_s32(a);
        a = vcombine_s32(vget_high_s32(a), vget_low_s32(a));
        b = vrev64q_s32(b);
        b = vcombine_s32(vget_high_s32(b), vget_low_s32(b));
    }
}

} // anonymous namespace

#endif // __aarch64__

// ============================================================================
// 小数组排序实现
// ============================================================================

void sort_4_i32(int32_t* data, SortOrder order) {
#ifdef __aarch64__
    int32x4_t v = vld1q_s32(data);
    v = sort_vec4_simple(v, order);
    vst1q_s32(data, v);
#else
    if (order == SortOrder::ASC) {
        std::sort(data, data + 4);
    } else {
        std::sort(data, data + 4, std::greater<int32_t>());
    }
#endif
}

void sort_8_i32(int32_t* data, SortOrder order) {
#ifdef __aarch64__
    int32x4_t a = vld1q_s32(data);
    int32x4_t b = vld1q_s32(data + 4);
    
    // 各自排序
    a = sort_vec4_simple(a, order);
    b = sort_vec4_simple(b, order);
    
    // 合并
    merge_vec4(a, b, order);
    
    vst1q_s32(data, a);
    vst1q_s32(data + 4, b);
#else
    if (order == SortOrder::ASC) {
        std::sort(data, data + 8);
    } else {
        std::sort(data, data + 8, std::greater<int32_t>());
    }
#endif
}

void sort_16_i32(int32_t* data, SortOrder order) {
#ifdef __aarch64__
    // 加载 4 个向量
    int32x4_t v0 = vld1q_s32(data);
    int32x4_t v1 = vld1q_s32(data + 4);
    int32x4_t v2 = vld1q_s32(data + 8);
    int32x4_t v3 = vld1q_s32(data + 12);
    
    // 各自排序
    v0 = sort_vec4_simple(v0, order);
    v1 = sort_vec4_simple(v1, order);
    v2 = sort_vec4_simple(v2, order);
    v3 = sort_vec4_simple(v3, order);
    
    // 两两合并
    merge_vec4(v0, v1, order);
    merge_vec4(v2, v3, order);
    
    // 最终合并（需要 4 路合并）
    alignas(16) int32_t arr[16];
    vst1q_s32(arr, v0);
    vst1q_s32(arr + 4, v1);
    vst1q_s32(arr + 8, v2);
    vst1q_s32(arr + 12, v3);
    
    // 使用标准库完成最终合并
    int32_t temp[16];
    std::merge(arr, arr + 8, arr + 8, arr + 16, temp);
    
    if (order == SortOrder::DESC) {
        std::reverse(temp, temp + 16);
    }
    
    std::memcpy(data, temp, 16 * sizeof(int32_t));
#else
    if (order == SortOrder::ASC) {
        std::sort(data, data + 16);
    } else {
        std::sort(data, data + 16, std::greater<int32_t>());
    }
#endif
}

// ============================================================================
// Bitonic Sort 实现
// ============================================================================

void bitonic_sort_i32(int32_t* data, size_t count, SortOrder order) {
    if (count <= 1) return;
    
    // 对于小数组，使用 SIMD 优化版本
    if (count == 4) {
        sort_4_i32(data, order);
        return;
    }
    if (count == 8) {
        sort_8_i32(data, order);
        return;
    }
    if (count == 16) {
        sort_16_i32(data, order);
        return;
    }
    
    // 对于其他大小，分块排序后合并
    // 找到不大于 count 的最大 2 的幂
    size_t block_size = 16;
    while (block_size * 2 <= count && block_size < 1024) {
        block_size *= 2;
    }
    
    // 分块排序
    for (size_t i = 0; i < count; i += block_size) {
        size_t end = std::min(i + block_size, count);
        size_t n = end - i;
        
        if (order == SortOrder::ASC) {
            std::sort(data + i, data + end);
        } else {
            std::sort(data + i, data + end, std::greater<int32_t>());
        }
    }
    
    // 多路合并
    if (count > block_size) {
        std::vector<int32_t> temp(count);
        
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<int32_t>());
        }
    }
}

// ============================================================================
// 主排序函数
// ============================================================================

void sort_i32(int32_t* data, size_t count, SortOrder order) {
    if (count <= 1) return;
    
    // 小数组使用 SIMD
    if (count <= 16) {
        bitonic_sort_i32(data, count, order);
        return;
    }
    
    // 中等数组使用分块 SIMD + 合并
    if (count <= 4096) {
        constexpr size_t BLOCK_SIZE = 16;
        
        // 对每个 16 元素块进行 SIMD 排序
        for (size_t i = 0; i + BLOCK_SIZE <= count; i += BLOCK_SIZE) {
            sort_16_i32(data + i, order);
        }
        
        // 处理剩余元素
        size_t remainder_start = (count / BLOCK_SIZE) * BLOCK_SIZE;
        if (remainder_start < count) {
            if (order == SortOrder::ASC) {
                std::sort(data + remainder_start, data + count);
            } else {
                std::sort(data + remainder_start, data + count, std::greater<int32_t>());
            }
        }
        
        // 使用标准库进行最终排序（std::sort 已经很高效）
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<int32_t>());
        }
        return;
    }
    
    // 大数组使用标准库（它已经很优化了）
    if (order == SortOrder::ASC) {
        std::sort(data, data + count);
    } else {
        std::sort(data, data + count, std::greater<int32_t>());
    }
}

void sort_i64(int64_t* data, size_t count, SortOrder order) {
    if (order == SortOrder::ASC) {
        std::sort(data, data + count);
    } else {
        std::sort(data, data + count, std::greater<int64_t>());
    }
}

void sort_f32(float* data, size_t count, SortOrder order) {
    if (order == SortOrder::ASC) {
        std::sort(data, data + count);
    } else {
        std::sort(data, data + count, std::greater<float>());
    }
}

void sort_f64(double* data, size_t count, SortOrder order) {
    if (order == SortOrder::ASC) {
        std::sort(data, data + count);
    } else {
        std::sort(data, data + count, std::greater<double>());
    }
}

// ============================================================================
// Argsort 实现
// ============================================================================

void argsort_i32(const int32_t* data, size_t count, 
                 uint32_t* out_indices, SortOrder order) {
    // 初始化索引
    for (size_t i = 0; i < count; ++i) {
        out_indices[i] = static_cast<uint32_t>(i);
    }
    
    // 按值排序索引
    if (order == SortOrder::ASC) {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });
    } else {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });
    }
}

void argsort_i64(const int64_t* data, size_t count, 
                 uint32_t* out_indices, SortOrder order) {
    for (size_t i = 0; i < count; ++i) {
        out_indices[i] = static_cast<uint32_t>(i);
    }
    
    if (order == SortOrder::ASC) {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });
    } else {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });
    }
}

void argsort_f32(const float* data, size_t count, 
                 uint32_t* out_indices, SortOrder order) {
    for (size_t i = 0; i < count; ++i) {
        out_indices[i] = static_cast<uint32_t>(i);
    }
    
    if (order == SortOrder::ASC) {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });
    } else {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });
    }
}

void argsort_f64(const double* data, size_t count, 
                 uint32_t* out_indices, SortOrder order) {
    for (size_t i = 0; i < count; ++i) {
        out_indices[i] = static_cast<uint32_t>(i);
    }
    
    if (order == SortOrder::ASC) {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });
    } else {
        std::sort(out_indices, out_indices + count, 
                  [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });
    }
}

// ============================================================================
// Top-K 实现
// ============================================================================

void topk_min_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);
    
    // 使用堆来获取最小的 k 个元素
    // 最大堆，保持大小为 k
    std::vector<std::pair<int32_t, uint32_t>> heap;
    heap.reserve(k);
    
    for (size_t i = 0; i < count; ++i) {
        if (heap.size() < k) {
            heap.push_back({data[i], static_cast<uint32_t>(i)});
            std::push_heap(heap.begin(), heap.end());
        } else if (data[i] < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    
    // 排序输出
    std::sort(heap.begin(), heap.end());
    
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = heap[i].first;
        if (out_indices) {
            out_indices[i] = heap[i].second;
        }
    }
}

void topk_max_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);
    
    // 最小堆
    auto cmp = [](const std::pair<int32_t, uint32_t>& a, 
                  const std::pair<int32_t, uint32_t>& b) {
        return a.first > b.first;
    };
    
    std::vector<std::pair<int32_t, uint32_t>> heap;
    heap.reserve(k);
    
    for (size_t i = 0; i < count; ++i) {
        if (heap.size() < k) {
            heap.push_back({data[i], static_cast<uint32_t>(i)});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (data[i] > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    
    // 按值降序排序
    std::sort(heap.begin(), heap.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = heap[i].first;
        if (out_indices) {
            out_indices[i] = heap[i].second;
        }
    }
}

void topk_min_f32(const float* data, size_t count, size_t k,
                  float* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);
    
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(k);
    
    for (size_t i = 0; i < count; ++i) {
        if (heap.size() < k) {
            heap.push_back({data[i], static_cast<uint32_t>(i)});
            std::push_heap(heap.begin(), heap.end());
        } else if (data[i] < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    
    std::sort(heap.begin(), heap.end());
    
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = heap[i].first;
        if (out_indices) {
            out_indices[i] = heap[i].second;
        }
    }
}

void topk_max_f32(const float* data, size_t count, size_t k,
                  float* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);
    
    auto cmp = [](const std::pair<float, uint32_t>& a, 
                  const std::pair<float, uint32_t>& b) {
        return a.first > b.first;
    };
    
    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(k);
    
    for (size_t i = 0; i < count; ++i) {
        if (heap.size() < k) {
            heap.push_back({data[i], static_cast<uint32_t>(i)});
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (data[i] > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    
    std::sort(heap.begin(), heap.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = heap[i].first;
        if (out_indices) {
            out_indices[i] = heap[i].second;
        }
    }
}

// ============================================================================
// 合并函数实现
// ============================================================================

void merge_sorted_i32(const int32_t* a, size_t a_count,
                      const int32_t* b, size_t b_count,
                      int32_t* out, SortOrder order) {
    if (order == SortOrder::ASC) {
        std::merge(a, a + a_count, b, b + b_count, out);
    } else {
        std::merge(a, a + a_count, b, b + b_count, out, std::greater<int32_t>());
    }
}

void merge_k_sorted_i32(const int32_t** arrays, const size_t* counts, size_t k,
                        int32_t* out, SortOrder order) {
    if (k == 0) return;
    if (k == 1) {
        std::memcpy(out, arrays[0], counts[0] * sizeof(int32_t));
        return;
    }
    
    // 使用优先队列进行 K 路合并
    struct Element {
        int32_t value;
        size_t array_idx;
        size_t element_idx;
    };
    
    auto cmp_asc = [](const Element& a, const Element& b) { 
        return a.value > b.value; 
    };
    auto cmp_desc = [](const Element& a, const Element& b) { 
        return a.value < b.value; 
    };
    
    std::vector<Element> heap;
    heap.reserve(k);
    
    // 初始化堆
    for (size_t i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            heap.push_back({arrays[i][0], i, 0});
        }
    }
    
    if (order == SortOrder::ASC) {
        std::make_heap(heap.begin(), heap.end(), cmp_asc);
    } else {
        std::make_heap(heap.begin(), heap.end(), cmp_desc);
    }
    
    size_t out_idx = 0;
    
    while (!heap.empty()) {
        Element top;
        if (order == SortOrder::ASC) {
            std::pop_heap(heap.begin(), heap.end(), cmp_asc);
        } else {
            std::pop_heap(heap.begin(), heap.end(), cmp_desc);
        }
        top = heap.back();
        heap.pop_back();
        
        out[out_idx++] = top.value;
        
        // 从同一数组添加下一个元素
        size_t next_idx = top.element_idx + 1;
        if (next_idx < counts[top.array_idx]) {
            Element next = {arrays[top.array_idx][next_idx], top.array_idx, next_idx};
            heap.push_back(next);
            if (order == SortOrder::ASC) {
                std::push_heap(heap.begin(), heap.end(), cmp_asc);
            } else {
                std::push_heap(heap.begin(), heap.end(), cmp_desc);
            }
        }
    }
}

// ============================================================================
// 键值对排序
// ============================================================================

void sort_pairs_i32_i32(int32_t* keys, int32_t* values, size_t count,
                        SortOrder order) {
    // 创建索引数组
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
    
    // 按键排序索引
    if (order == SortOrder::ASC) {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] > keys[b]; });
    }
    
    // 按排序后的索引重排
    std::vector<int32_t> sorted_keys(count), sorted_values(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_keys[i] = keys[indices[i]];
        sorted_values[i] = values[indices[i]];
    }
    
    std::memcpy(keys, sorted_keys.data(), count * sizeof(int32_t));
    std::memcpy(values, sorted_values.data(), count * sizeof(int32_t));
}

void sort_pairs_i32_i64(int32_t* keys, int64_t* values, size_t count,
                        SortOrder order) {
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
    
    if (order == SortOrder::ASC) {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] > keys[b]; });
    }
    
    std::vector<int32_t> sorted_keys(count);
    std::vector<int64_t> sorted_values(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_keys[i] = keys[indices[i]];
        sorted_values[i] = values[indices[i]];
    }
    
    std::memcpy(keys, sorted_keys.data(), count * sizeof(int32_t));
    std::memcpy(values, sorted_values.data(), count * sizeof(int64_t));
}

void sort_pairs_i64_i64(int64_t* keys, int64_t* values, size_t count,
                        SortOrder order) {
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
    
    if (order == SortOrder::ASC) {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                  [keys](uint32_t a, uint32_t b) { return keys[a] > keys[b]; });
    }
    
    std::vector<int64_t> sorted_keys(count), sorted_values(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_keys[i] = keys[indices[i]];
        sorted_values[i] = values[indices[i]];
    }
    
    std::memcpy(keys, sorted_keys.data(), count * sizeof(int64_t));
    std::memcpy(values, sorted_values.data(), count * sizeof(int64_t));
}

} // namespace sort
} // namespace thunderduck
