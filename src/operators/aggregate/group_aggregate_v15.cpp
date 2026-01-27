/**
 * ThunderDuck - GROUP BY Aggregation v15.0 Implementation
 *
 * V15 优化策略：
 * - 8 线程并行（M4 有 10 核）
 * - 移除边界检查（假设输入合法）
 * - 8 路循环展开
 * - 缓存行对齐避免伪共享
 * - 更激进的预取
 */

#include "thunderduck/aggregate.h"
#include <limits>
#include <cstring>
#include <algorithm>
#include <vector>
#include <thread>
#include <memory>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace aggregate {

namespace {

constexpr size_t MIN_ELEMENTS_PER_THREAD_V15 = 100000;
constexpr size_t MAX_THREADS_V15 = 8;  // 8 线程
constexpr size_t PREFETCH_DISTANCE_V15 = 64;
constexpr size_t CACHE_LINE = 128;  // M4 L1 cache line

const char* V15_VERSION = "V15.0 - 8线程 + 8路展开 + 缓存优化";

// 缓存行对齐的局部累加器
struct alignas(CACHE_LINE) AlignedLocalSum {
    int64_t* data;
    size_t size;

    AlignedLocalSum() : data(nullptr), size(0) {}

    void allocate(size_t num_groups) {
        size = num_groups;
        // 分配对齐内存
        data = static_cast<int64_t*>(aligned_alloc(CACHE_LINE,
            ((num_groups * sizeof(int64_t) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE));
        std::memset(data, 0, num_groups * sizeof(int64_t));
    }

    ~AlignedLocalSum() {
        if (data) free(data);
    }

    AlignedLocalSum(const AlignedLocalSum&) = delete;
    AlignedLocalSum& operator=(const AlignedLocalSum&) = delete;
    AlignedLocalSum(AlignedLocalSum&& o) noexcept : data(o.data), size(o.size) { o.data = nullptr; }
};

} // anonymous namespace

// ============================================================================
// V15 并行分组求和 (8线程 + 8路展开)
// ============================================================================

void group_sum_i32_v15_parallel(const int32_t* __restrict values,
                                 const uint32_t* __restrict groups,
                                 size_t count, size_t num_groups,
                                 int64_t* __restrict out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    std::memset(out_sums, 0, num_groups * sizeof(int64_t));

    // 动态选择线程数
    size_t num_threads = std::min(MAX_THREADS_V15,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD_V15));

    // 小数据量单线程处理
    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD_V15) {
        size_t i = 0;
#ifdef __aarch64__
        // 8 路展开
        for (; i + 8 <= count; i += 8) {
            __builtin_prefetch(&groups[i + PREFETCH_DISTANCE_V15], 0, 3);
            __builtin_prefetch(&values[i + PREFETCH_DISTANCE_V15], 0, 3);

            // 无边界检查版本（假设输入合法）
            out_sums[groups[i]]     += values[i];
            out_sums[groups[i + 1]] += values[i + 1];
            out_sums[groups[i + 2]] += values[i + 2];
            out_sums[groups[i + 3]] += values[i + 3];
            out_sums[groups[i + 4]] += values[i + 4];
            out_sums[groups[i + 5]] += values[i + 5];
            out_sums[groups[i + 6]] += values[i + 6];
            out_sums[groups[i + 7]] += values[i + 7];
        }
#endif
        for (; i < count; ++i) {
            out_sums[groups[i]] += values[i];
        }
        return;
    }

    // 分配对齐的局部累加器
    std::vector<AlignedLocalSum> local_sums(num_threads);
    for (auto& ls : local_sums) {
        ls.allocate(num_groups);
    }

    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            int64_t* local = local_sums[t].data;

            size_t i = start;
#ifdef __aarch64__
            // 8 路展开 + 双预取
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(&groups[i + PREFETCH_DISTANCE_V15], 0, 3);
                __builtin_prefetch(&values[i + PREFETCH_DISTANCE_V15], 0, 3);
                __builtin_prefetch(&groups[i + PREFETCH_DISTANCE_V15 + 32], 0, 2);
                __builtin_prefetch(&values[i + PREFETCH_DISTANCE_V15 + 32], 0, 2);

                // 无边界检查
                local[groups[i]]     += values[i];
                local[groups[i + 1]] += values[i + 1];
                local[groups[i + 2]] += values[i + 2];
                local[groups[i + 3]] += values[i + 3];
                local[groups[i + 4]] += values[i + 4];
                local[groups[i + 5]] += values[i + 5];
                local[groups[i + 6]] += values[i + 6];
                local[groups[i + 7]] += values[i + 7];
            }
#endif
            for (; i < end; ++i) {
                local[groups[i]] += values[i];
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // SIMD 合并结果 (4 路并行)
#ifdef __aarch64__
    for (size_t g = 0; g + 4 <= num_groups; g += 4) {
        int64x2_t sum0 = vdupq_n_s64(0);
        int64x2_t sum1 = vdupq_n_s64(0);

        for (size_t t = 0; t < num_threads; ++t) {
            int64x2_t v0 = vld1q_s64(&local_sums[t].data[g]);
            int64x2_t v1 = vld1q_s64(&local_sums[t].data[g + 2]);
            sum0 = vaddq_s64(sum0, v0);
            sum1 = vaddq_s64(sum1, v1);
        }
        vst1q_s64(&out_sums[g], sum0);
        vst1q_s64(&out_sums[g + 2], sum1);
    }

    // 处理剩余
    for (size_t g = (num_groups / 4) * 4; g < num_groups; ++g) {
        for (size_t t = 0; t < num_threads; ++t) {
            out_sums[g] += local_sums[t].data[g];
        }
    }
#else
    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t g = 0; g < num_groups; ++g) {
            out_sums[g] += local_sums[t].data[g];
        }
    }
#endif
}

// ============================================================================
// V15 公开接口
// ============================================================================

void group_sum_i32_v15(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    group_sum_i32_v15_parallel(values, groups, count, num_groups, out_sums);
}

const char* get_group_aggregate_v15_version() {
    return V15_VERSION;
}

} // namespace aggregate
} // namespace thunderduck
