/**
 * ThunderDuck - TopK v4.0 Implementation
 *
 * 核心优化: 采样预过滤 + SIMD 批量跳过
 *
 * 针对 T4 场景 (10M 行, K=10) 的专项优化:
 * - 问题: 纯堆方法需遍历全部 10M 元素，每次比较
 * - 方案: 采样估计阈值 → SIMD 批量过滤 → 只处理候选
 *
 * 性能目标: T4 从 0.41x 提升到 2x+
 */

#include "thunderduck/sort.h"
#include "thunderduck/memory.h"
#include <algorithm>
#include <vector>
#include <cstring>
#include <random>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace sort {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

// v4.0 策略阈值
constexpr size_t SMALL_N_THRESHOLD = 500000;    // N < 500K 时使用 partial_sort (v4.1 优化)
constexpr size_t LARGE_N_THRESHOLD = 1000000;   // N >= 1M 时启用采样预过滤
constexpr size_t SAMPLE_SIZE = 8192;            // 采样数量
constexpr size_t K_SMALL_THRESHOLD = 64;        // K <= 64 视为小 K
constexpr size_t K_MEDIUM_THRESHOLD = 1024;     // K <= 1024 视为中等 K

// SIMD 批处理
constexpr size_t SIMD_BATCH = 64;               // 每批 64 个元素 (16 个向量)

// 预过滤安全系数 (采样估计阈值时，取第 K * SAFETY_FACTOR 大的值)
// 使用 > 1.0 的值让阈值更低，收集更多候选，确保不漏掉真正的 TopK
constexpr double SAFETY_FACTOR = 3.0;           // 保守估计：收集约 3K 个候选

} // anonymous namespace

// ============================================================================
// 辅助函数: 采样估计阈值
// ============================================================================

namespace {

/**
 * 从数据中采样，估计第 K 大的值作为预过滤阈值
 *
 * @param data 输入数据
 * @param count 数据量
 * @param k 目标 K 值
 * @return 预估的第 K 大值 (保守估计，可能略小于实际值)
 */
int32_t estimate_threshold_max(const int32_t* data, size_t count, size_t k) {
    // 确定采样数量
    size_t sample_size = std::min(SAMPLE_SIZE, count);

    // 计算采样步长 (均匀采样)
    size_t step = count / sample_size;
    if (step == 0) step = 1;

    // 采样
    std::vector<int32_t> samples;
    samples.reserve(sample_size);

    for (size_t i = 0; i < count && samples.size() < sample_size; i += step) {
        samples.push_back(data[i]);
    }

    // 计算在采样中对应的 K 值 (比例映射)
    size_t sample_k = static_cast<size_t>(
        static_cast<double>(k) * samples.size() / count * SAFETY_FACTOR
    );
    sample_k = std::max(sample_k, size_t(1));
    sample_k = std::min(sample_k, samples.size());

    // 部分排序找到第 sample_k 大的值
    std::nth_element(samples.begin(), samples.begin() + sample_k - 1, samples.end(),
                     std::greater<int32_t>());

    return samples[sample_k - 1];
}

int32_t estimate_threshold_min(const int32_t* data, size_t count, size_t k) {
    size_t sample_size = std::min(SAMPLE_SIZE, count);
    size_t step = count / sample_size;
    if (step == 0) step = 1;

    std::vector<int32_t> samples;
    samples.reserve(sample_size);

    for (size_t i = 0; i < count && samples.size() < sample_size; i += step) {
        samples.push_back(data[i]);
    }

    size_t sample_k = static_cast<size_t>(
        static_cast<double>(k) * samples.size() / count * SAFETY_FACTOR
    );
    sample_k = std::max(sample_k, size_t(1));
    sample_k = std::min(sample_k, samples.size());

    std::nth_element(samples.begin(), samples.begin() + sample_k - 1, samples.end());

    return samples[sample_k - 1];
}

} // anonymous namespace

// ============================================================================
// 核心算法: SIMD 批量预过滤 + 候选收集
// ============================================================================

#ifdef __aarch64__

namespace {

/**
 * SIMD 批量预过滤: 只收集大于阈值的候选元素
 *
 * 优势:
 * - 一次比较 16 个元素 (4 个 int32x4_t)
 * - 如果批次内全部 <= 阈值，整批跳过
 * - 对于随机数据，预期跳过 ~90% 的元素
 */
void collect_candidates_max_simd(const int32_t* data, size_t count,
                                  int32_t threshold,
                                  std::vector<std::pair<int32_t, uint32_t>>& candidates) {
    candidates.clear();
    candidates.reserve(count / 10);  // 预估 10% 通过过滤

    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    size_t i = 0;
    for (; i + SIMD_BATCH <= count; i += SIMD_BATCH) {
        // 软件预取
        __builtin_prefetch(data + i + 256, 0, 0);

        // 快速检查: 批次内是否有任何元素 > threshold
        bool has_candidate = false;

        for (size_t j = 0; j < SIMD_BATCH; j += 16) {
            int32x4_t d0 = vld1q_s32(data + i + j);
            int32x4_t d1 = vld1q_s32(data + i + j + 4);
            int32x4_t d2 = vld1q_s32(data + i + j + 8);
            int32x4_t d3 = vld1q_s32(data + i + j + 12);

            uint32x4_t m0 = vcgeq_s32(d0, thresh_vec);
            uint32x4_t m1 = vcgeq_s32(d1, thresh_vec);
            uint32x4_t m2 = vcgeq_s32(d2, thresh_vec);
            uint32x4_t m3 = vcgeq_s32(d3, thresh_vec);

            uint32x4_t any01 = vorrq_u32(m0, m1);
            uint32x4_t any23 = vorrq_u32(m2, m3);
            uint32x4_t any = vorrq_u32(any01, any23);

            if (vmaxvq_u32(any)) {
                has_candidate = true;
                break;
            }
        }

        // 如果批次有候选，逐个检查并收集
        if (has_candidate) {
            for (size_t j = 0; j < SIMD_BATCH; ++j) {
                if (data[i + j] >= threshold) {
                    candidates.push_back({data[i + j], static_cast<uint32_t>(i + j)});
                }
            }
        }
        // 否则整个批次跳过
    }

    // 处理剩余元素
    for (; i < count; ++i) {
        if (data[i] > threshold) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }
}

void collect_candidates_min_simd(const int32_t* data, size_t count,
                                  int32_t threshold,
                                  std::vector<std::pair<int32_t, uint32_t>>& candidates) {
    candidates.clear();
    candidates.reserve(count / 10);

    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    size_t i = 0;
    for (; i + SIMD_BATCH <= count; i += SIMD_BATCH) {
        __builtin_prefetch(data + i + 256, 0, 0);

        bool has_candidate = false;

        for (size_t j = 0; j < SIMD_BATCH; j += 16) {
            int32x4_t d0 = vld1q_s32(data + i + j);
            int32x4_t d1 = vld1q_s32(data + i + j + 4);
            int32x4_t d2 = vld1q_s32(data + i + j + 8);
            int32x4_t d3 = vld1q_s32(data + i + j + 12);

            uint32x4_t m0 = vcleq_s32(d0, thresh_vec);
            uint32x4_t m1 = vcleq_s32(d1, thresh_vec);
            uint32x4_t m2 = vcleq_s32(d2, thresh_vec);
            uint32x4_t m3 = vcleq_s32(d3, thresh_vec);

            uint32x4_t any01 = vorrq_u32(m0, m1);
            uint32x4_t any23 = vorrq_u32(m2, m3);
            uint32x4_t any = vorrq_u32(any01, any23);

            if (vmaxvq_u32(any)) {
                has_candidate = true;
                break;
            }
        }

        if (has_candidate) {
            for (size_t j = 0; j < SIMD_BATCH; ++j) {
                if (data[i + j] <= threshold) {
                    candidates.push_back({data[i + j], static_cast<uint32_t>(i + j)});
                }
            }
        }
    }

    for (; i < count; ++i) {
        if (data[i] < threshold) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }
}

} // anonymous namespace

#endif // __aarch64__

// ============================================================================
// 策略: 采样预过滤 TopK (针对大 N 小 K)
// ============================================================================

namespace {

/**
 * 采样预过滤 TopK (Max)
 *
 * 算法流程:
 * 1. 采样估计阈值 (第 K 大的值)
 * 2. SIMD 批量过滤，只收集 > 阈值的候选
 * 3. 如果候选不足 K 个，降低阈值重新收集
 * 4. 从候选中用 nth_element 选择最终 TopK
 *
 * 时间复杂度: O(n) 采样 + O(n) 过滤 + O(c) 选择
 * 其中 c 是候选数量，通常 c << n
 */
void topk_sampled_prefilter_max(const int32_t* data, size_t count, size_t k,
                                  int32_t* out_values, uint32_t* out_indices) {
    // 1. 采样估计阈值
    int32_t threshold = estimate_threshold_max(data, count, k);

    // 2. SIMD 预过滤收集候选
    std::vector<std::pair<int32_t, uint32_t>> candidates;

#ifdef __aarch64__
    collect_candidates_max_simd(data, count, threshold, candidates);
#else
    candidates.reserve(count / 10);
    for (size_t i = 0; i < count; ++i) {
        if (data[i] >= threshold) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }
#endif

    // 3. 检查候选数量是否合理
    // 如果候选过多 (> 5% of total)，说明数据基数低，采样无效
    // 回退到堆方法
    if (candidates.size() > count / 20) {
        // 使用堆方法处理
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

        for (size_t i = 0; i < k && i < heap.size(); ++i) {
            out_values[i] = heap[i].first;
            if (out_indices) out_indices[i] = heap[i].second;
        }
        return;
    }

    // 4. 如果候选不足 K 个，需要降低阈值重新收集
    if (candidates.size() < k) {
        // 降低阈值到最小值，收集所有元素
        candidates.clear();
        candidates.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }

    // 5. 从候选中选择 TopK
    if (candidates.size() > k) {
        std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates.resize(k);
    }

    // 6. 排序输出
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < k && i < candidates.size(); ++i) {
        out_values[i] = candidates[i].first;
        if (out_indices) out_indices[i] = candidates[i].second;
    }
}

void topk_sampled_prefilter_min(const int32_t* data, size_t count, size_t k,
                                  int32_t* out_values, uint32_t* out_indices) {
    int32_t threshold = estimate_threshold_min(data, count, k);

    std::vector<std::pair<int32_t, uint32_t>> candidates;

#ifdef __aarch64__
    collect_candidates_min_simd(data, count, threshold, candidates);
#else
    candidates.reserve(count / 10);
    for (size_t i = 0; i < count; ++i) {
        if (data[i] <= threshold) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }
#endif

    // 检查候选数量是否合理
    if (candidates.size() > count / 20) {
        // 使用堆方法处理
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

        std::sort(heap.begin(), heap.end());

        for (size_t i = 0; i < k && i < heap.size(); ++i) {
            out_values[i] = heap[i].first;
            if (out_indices) out_indices[i] = heap[i].second;
        }
        return;
    }

    if (candidates.size() < k) {
        candidates.clear();
        candidates.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            candidates.push_back({data[i], static_cast<uint32_t>(i)});
        }
    }

    if (candidates.size() > k) {
        std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        candidates.resize(k);
    }

    std::sort(candidates.begin(), candidates.end());

    for (size_t i = 0; i < k && i < candidates.size(); ++i) {
        out_values[i] = candidates[i].first;
        if (out_indices) out_indices[i] = candidates[i].second;
    }
}

} // anonymous namespace

// ============================================================================
// v4.1 优化: partial_sort 回退 (针对小数据量)
// ============================================================================

namespace {

/**
 * 使用 partial_sort 实现 TopK (小数据量最优)
 *
 * 优势:
 * - 比堆方法更少的内存分配
 * - partial_sort 对小 K 高度优化
 * - 无堆维护开销
 *
 * 适用: N < 500K
 */
void topk_partial_sort_max(const int32_t* data, size_t count, size_t k,
                            int32_t* out_values, uint32_t* out_indices) {
    // 创建索引数组
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    // partial_sort: 只排前 K 个
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    // 输出结果
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}

void topk_partial_sort_min(const int32_t* data, size_t count, size_t k,
                            int32_t* out_values, uint32_t* out_indices) {
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}

} // anonymous namespace

// ============================================================================
// v3.0 原有策略 (用于中小数据集)
// ============================================================================

namespace {

// 小 K 纯堆方法
void topk_heap_small_max(const int32_t* data, size_t count, size_t k,
                          int32_t* out_values, uint32_t* out_indices) {
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
}

void topk_heap_small_min(const int32_t* data, size_t count, size_t k,
                          int32_t* out_values, uint32_t* out_indices) {
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

    std::sort(heap.begin(), heap.end());

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = heap[i].first;
        if (out_indices) out_indices[i] = heap[i].second;
    }
}

#ifdef __aarch64__

// SIMD 加速堆方法 (中等 K)
void topk_simd_heap_max(const int32_t* data, size_t count, size_t k,
                         int32_t* out_values, uint32_t* out_indices) {
    auto cmp = [](const std::pair<int32_t, uint32_t>& a,
                  const std::pair<int32_t, uint32_t>& b) {
        return a.first > b.first;
    };

    std::vector<std::pair<int32_t, uint32_t>> heap;
    heap.reserve(k);

    size_t i = 0;
    for (; i < k && i < count; ++i) {
        heap.push_back({data[i], static_cast<uint32_t>(i)});
    }
    std::make_heap(heap.begin(), heap.end(), cmp);

    if (i >= count) {
        std::sort(heap.begin(), heap.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t j = 0; j < heap.size(); ++j) {
            out_values[j] = heap[j].first;
            if (out_indices) out_indices[j] = heap[j].second;
        }
        return;
    }

    int32_t threshold = heap.front().first;
    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    for (; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(data + i);
        int32x4_t d1 = vld1q_s32(data + i + 4);
        int32x4_t d2 = vld1q_s32(data + i + 8);
        int32x4_t d3 = vld1q_s32(data + i + 12);

        uint32x4_t m0 = vcgtq_s32(d0, thresh_vec);
        uint32x4_t m1 = vcgtq_s32(d1, thresh_vec);
        uint32x4_t m2 = vcgtq_s32(d2, thresh_vec);
        uint32x4_t m3 = vcgtq_s32(d3, thresh_vec);

        uint32x4_t any01 = vorrq_u32(m0, m1);
        uint32x4_t any23 = vorrq_u32(m2, m3);
        uint32x4_t any = vorrq_u32(any01, any23);

        if (vmaxvq_u32(any)) {
            for (size_t j = 0; j < 16; ++j) {
                if (data[i + j] >= threshold) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {data[i + j], static_cast<uint32_t>(i + j)};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                    threshold = heap.front().first;
                }
            }
            thresh_vec = vdupq_n_s32(threshold);
        }
    }

    for (; i < count; ++i) {
        if (data[i] > threshold) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end(), cmp);
            threshold = heap.front().first;
        }
    }

    std::sort(heap.begin(), heap.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t j = 0; j < k; ++j) {
        out_values[j] = heap[j].first;
        if (out_indices) out_indices[j] = heap[j].second;
    }
}

void topk_simd_heap_min(const int32_t* data, size_t count, size_t k,
                         int32_t* out_values, uint32_t* out_indices) {
    std::vector<std::pair<int32_t, uint32_t>> heap;
    heap.reserve(k);

    size_t i = 0;
    for (; i < k && i < count; ++i) {
        heap.push_back({data[i], static_cast<uint32_t>(i)});
    }
    std::make_heap(heap.begin(), heap.end());

    if (i >= count) {
        std::sort(heap.begin(), heap.end());
        for (size_t j = 0; j < heap.size(); ++j) {
            out_values[j] = heap[j].first;
            if (out_indices) out_indices[j] = heap[j].second;
        }
        return;
    }

    int32_t threshold = heap.front().first;
    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    for (; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(data + i);
        int32x4_t d1 = vld1q_s32(data + i + 4);
        int32x4_t d2 = vld1q_s32(data + i + 8);
        int32x4_t d3 = vld1q_s32(data + i + 12);

        uint32x4_t m0 = vcltq_s32(d0, thresh_vec);
        uint32x4_t m1 = vcltq_s32(d1, thresh_vec);
        uint32x4_t m2 = vcltq_s32(d2, thresh_vec);
        uint32x4_t m3 = vcltq_s32(d3, thresh_vec);

        uint32x4_t any01 = vorrq_u32(m0, m1);
        uint32x4_t any23 = vorrq_u32(m2, m3);
        uint32x4_t any = vorrq_u32(any01, any23);

        if (vmaxvq_u32(any)) {
            for (size_t j = 0; j < 16; ++j) {
                if (data[i + j] <= threshold) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {data[i + j], static_cast<uint32_t>(i + j)};
                    std::push_heap(heap.begin(), heap.end());
                    threshold = heap.front().first;
                }
            }
            thresh_vec = vdupq_n_s32(threshold);
        }
    }

    for (; i < count; ++i) {
        if (data[i] < threshold) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end());
            threshold = heap.front().first;
        }
    }

    std::sort(heap.begin(), heap.end());

    for (size_t j = 0; j < k; ++j) {
        out_values[j] = heap[j].first;
        if (out_indices) out_indices[j] = heap[j].second;
    }
}

#endif // __aarch64__

// 大 K: nth_element
void topk_nth_element_max(const int32_t* data, size_t count, size_t k,
                           int32_t* out_values, uint32_t* out_indices) {
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    std::sort(indices.begin(), indices.begin() + k,
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}

void topk_nth_element_min(const int32_t* data, size_t count, size_t k,
                           int32_t* out_values, uint32_t* out_indices) {
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });

    std::sort(indices.begin(), indices.begin() + k,
        [data](uint32_t a, uint32_t b) { return data[a] < data[b]; });

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}

} // anonymous namespace

// ============================================================================
// 公开接口 - v4.0 自适应策略
// ============================================================================

void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);

    // v4.1 优化: 小数据量直接 partial_sort
    // 对于 N < 500K 且 K 小的场景，partial_sort 比堆方法更快
    if (count < SMALL_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
        topk_partial_sort_max(data, count, k, out_values, out_indices);
        return;
    }

    // 核心优化: 大数据量 + 小 K → 采样预过滤
    if (count >= LARGE_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
        // T4 场景: 10M 行, K=10
        topk_sampled_prefilter_max(data, count, k, out_values, out_indices);
        return;
    }

    // 中等数据量 (500K-1M) 使用堆方法
    if (k <= K_SMALL_THRESHOLD) {
        topk_heap_small_max(data, count, k, out_values, out_indices);
    }
#ifdef __aarch64__
    else if (k <= K_MEDIUM_THRESHOLD) {
        topk_simd_heap_max(data, count, k, out_values, out_indices);
    }
#endif
    else {
        topk_nth_element_max(data, count, k, out_values, out_indices);
    }
}

void topk_min_i32_v4(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);

    // v4.1 优化: 小数据量直接 partial_sort
    if (count < SMALL_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
        topk_partial_sort_min(data, count, k, out_values, out_indices);
        return;
    }

    // 大数据量 + 小 K → 采样预过滤
    if (count >= LARGE_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
        topk_sampled_prefilter_min(data, count, k, out_values, out_indices);
        return;
    }

    // 中等数据量 (500K-1M) 使用堆方法
    if (k <= K_SMALL_THRESHOLD) {
        topk_heap_small_min(data, count, k, out_values, out_indices);
    }
#ifdef __aarch64__
    else if (k <= K_MEDIUM_THRESHOLD) {
        topk_simd_heap_min(data, count, k, out_values, out_indices);
    }
#endif
    else {
        topk_nth_element_min(data, count, k, out_values, out_indices);
    }
}

} // namespace sort
} // namespace thunderduck
