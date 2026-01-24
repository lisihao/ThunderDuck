/**
 * ThunderDuck - Radix Sort Implementation
 *
 * LSD Radix Sort - O(n) 时间复杂度
 * 针对整数排序的最优算法
 */

#include "thunderduck/sort.h"
#include "thunderduck/memory.h"
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace sort {

// ============================================================================
// Radix Sort 配置
// ============================================================================

constexpr int RADIX_BITS = 8;
constexpr int RADIX = 1 << RADIX_BITS;  // 256
constexpr int PASSES = sizeof(int32_t) * 8 / RADIX_BITS;  // 4 passes for int32

// ============================================================================
// LSD Radix Sort for int32
// ============================================================================

void radix_sort_i32(int32_t* data, size_t count, SortOrder order) {
    if (count <= 1) return;

    // 小数组使用 std::sort
    if (count < 256) {
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<int32_t>());
        }
        return;
    }

    // 分配缓冲区
    std::vector<int32_t> buffer(count);
    int32_t* src = data;
    int32_t* dst = buffer.data();

    // 4 趟基数排序 (每趟处理 8 位)
    for (int pass = 0; pass < PASSES; ++pass) {
        int shift = pass * RADIX_BITS;

        // 计数数组 (256 个桶)
        alignas(64) size_t counts[RADIX] = {0};

        // 第一遍：计数
#ifdef __aarch64__
        // 使用 SIMD 加速计数
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            uint32_t k0 = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            uint32_t k1 = (static_cast<uint32_t>(src[i+1]) >> shift) & (RADIX - 1);
            uint32_t k2 = (static_cast<uint32_t>(src[i+2]) >> shift) & (RADIX - 1);
            uint32_t k3 = (static_cast<uint32_t>(src[i+3]) >> shift) & (RADIX - 1);
            counts[k0]++;
            counts[k1]++;
            counts[k2]++;
            counts[k3]++;
        }
        for (; i < count; ++i) {
            uint32_t key = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            counts[key]++;
        }
#else
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            counts[key]++;
        }
#endif

        // 计算前缀和 (确定每个桶的起始位置)
        size_t offsets[RADIX];
        size_t offset = 0;
        for (int i = 0; i < RADIX; ++i) {
            offsets[i] = offset;
            offset += counts[i];
        }

        // 第二遍：分配到目标位置
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            dst[offsets[key]++] = src[i];
        }

        // 交换 src 和 dst
        std::swap(src, dst);
    }

    // 处理符号位 (最高位是符号位)
    // 符号位为 1 的是负数，应该在前面
    if (src != data) {
        // 找到正负数的分界点
        size_t neg_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (src[i] < 0) ++neg_count;
        }

        // 复制负数到前面，正数到后面
        size_t neg_idx = 0, pos_idx = neg_count;
        for (size_t i = 0; i < count; ++i) {
            if (src[i] < 0) {
                data[neg_idx++] = src[i];
            } else {
                data[pos_idx++] = src[i];
            }
        }
    } else {
        // src == data，直接处理符号位
        size_t neg_count = 0;
        for (size_t i = 0; i < count; ++i) {
            if (data[i] < 0) ++neg_count;
        }

        if (neg_count > 0 && neg_count < count) {
            std::vector<int32_t> temp(count);
            size_t neg_idx = 0, pos_idx = neg_count;
            for (size_t i = 0; i < count; ++i) {
                if (data[i] < 0) {
                    temp[neg_idx++] = data[i];
                } else {
                    temp[pos_idx++] = data[i];
                }
            }
            std::memcpy(data, temp.data(), count * sizeof(int32_t));
        }
    }

    // 降序排列
    if (order == SortOrder::DESC) {
        std::reverse(data, data + count);
    }
}

// ============================================================================
// 优化版 Radix Sort - 使用 11-11-10 分组减少趟数
// ============================================================================

void radix_sort_i32_v2(int32_t* data, size_t count, SortOrder order) {
    if (count <= 1) return;

    if (count < 256) {
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<int32_t>());
        }
        return;
    }

    // 将有符号整数转换为可正确排序的无符号整数
    // 通过翻转符号位实现
    for (size_t i = 0; i < count; ++i) {
        uint32_t u = static_cast<uint32_t>(data[i]);
        data[i] = static_cast<int32_t>(u ^ 0x80000000);  // 翻转符号位
    }

    std::vector<uint32_t> buffer(count);
    uint32_t* src = reinterpret_cast<uint32_t*>(data);
    uint32_t* dst = buffer.data();

    // 3 趟基数排序: 11-11-10 位
    constexpr int BITS[3] = {11, 11, 10};
    constexpr int RADIXES[3] = {1 << 11, 1 << 11, 1 << 10};
    int shift = 0;

    for (int pass = 0; pass < 3; ++pass) {
        int radix = RADIXES[pass];
        int bits = BITS[pass];

        std::vector<size_t> counts(radix, 0);

        // 计数
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (src[i] >> shift) & (radix - 1);
            counts[key]++;
        }

        // 前缀和
        std::vector<size_t> offsets(radix);
        size_t offset = 0;
        for (int i = 0; i < radix; ++i) {
            offsets[i] = offset;
            offset += counts[i];
        }

        // 分配
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (src[i] >> shift) & (radix - 1);
            dst[offsets[key]++] = src[i];
        }

        std::swap(src, dst);
        shift += bits;
    }

    // 确保结果在 data 中
    if (src != reinterpret_cast<uint32_t*>(data)) {
        std::memcpy(data, src, count * sizeof(uint32_t));
    }

    // 转换回有符号整数
    for (size_t i = 0; i < count; ++i) {
        uint32_t u = static_cast<uint32_t>(data[i]);
        data[i] = static_cast<int32_t>(u ^ 0x80000000);
    }

    if (order == SortOrder::DESC) {
        std::reverse(data, data + count);
    }
}

// ============================================================================
// Radix Sort for uint32
// ============================================================================

void radix_sort_u32(uint32_t* data, size_t count, SortOrder order) {
    if (count <= 1) return;

    if (count < 256) {
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<uint32_t>());
        }
        return;
    }

    std::vector<uint32_t> buffer(count);
    uint32_t* src = data;
    uint32_t* dst = buffer.data();

    for (int pass = 0; pass < 4; ++pass) {
        int shift = pass * 8;
        alignas(64) size_t counts[256] = {0};

        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (src[i] >> shift) & 0xFF;
            counts[key]++;
        }

        size_t offsets[256];
        size_t offset = 0;
        for (int i = 0; i < 256; ++i) {
            offsets[i] = offset;
            offset += counts[i];
        }

        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (src[i] >> shift) & 0xFF;
            dst[offsets[key]++] = src[i];
        }

        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, count * sizeof(uint32_t));
    }

    if (order == SortOrder::DESC) {
        std::reverse(data, data + count);
    }
}

// ============================================================================
// 新的主排序函数 - 使用 Radix Sort
// ============================================================================

void sort_i32_v2(int32_t* data, size_t count, SortOrder order) {
    if (count <= 64) {
        // 小数组使用 std::sort
        if (order == SortOrder::ASC) {
            std::sort(data, data + count);
        } else {
            std::sort(data, data + count, std::greater<int32_t>());
        }
        return;
    }

    // 大数组使用 Radix Sort
    radix_sort_i32_v2(data, count, order);
}

// ============================================================================
// 优化的 Top-K - 使用部分快速选择
// ============================================================================

void topk_max_i32_v2(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);

    // 对于小 K，使用堆更高效
    if (k <= 100) {
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

        std::sort(heap.begin(), heap.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < k; ++i) {
            out_values[i] = heap[i].first;
            if (out_indices) out_indices[i] = heap[i].second;
        }
        return;
    }

    // 对于较大 K，使用 nth_element
    std::vector<std::pair<int32_t, uint32_t>> pairs(count);
    for (size_t i = 0; i < count; ++i) {
        pairs[i] = {data[i], static_cast<uint32_t>(i)};
    }

    std::nth_element(pairs.begin(), pairs.begin() + k, pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    std::sort(pairs.begin(), pairs.begin() + k,
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = pairs[i].first;
        if (out_indices) out_indices[i] = pairs[i].second;
    }
}

} // namespace sort
} // namespace thunderduck
