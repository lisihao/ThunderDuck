/**
 * ThunderDuck - SIMD Filter V19.1
 *
 * V19.1 核心优化: 静态线程池 + 两阶段并行
 *
 * V19 问题:
 * - 每次调用创建/销毁 8 个线程 (~100us 开销)
 *
 * V19.1 方案:
 * - 静态线程池，启动时创建，程序结束时销毁
 * - 任务提交到线程池，无创建开销
 * - 保持两阶段直写算法
 *
 * 目标: 1.80x → 2.0x+
 */

#include "thunderduck/filter.h"
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

#ifdef __aarch64__

// ============================================================================
// 静态线程池
// ============================================================================

namespace {

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop_(false) {
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    template<typename F>
    void submit(F&& f) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    void wait_all(std::atomic<size_t>& counter, size_t target) {
        while (counter.load(std::memory_order_acquire) < target) {
            std::this_thread::yield();
        }
    }

    size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

// 全局线程池 (懒加载)
ThreadPool& get_thread_pool() {
    static ThreadPool pool(8);  // 8 线程
    return pool;
}

// ============================================================================
// 配置常量
// ============================================================================

constexpr size_t NUM_THREADS = 8;
constexpr size_t MIN_PARALLEL = 500000;    // 500K 以上使用并行
constexpr size_t PREFETCH_DISTANCE = 256;

// ============================================================================
// 编译期比较操作 (与 V19 相同)
// ============================================================================

template<CompareOp Op>
__attribute__((always_inline))
inline uint32x4_t simd_compare_i32(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) return vcgtq_s32(data, threshold);
    if constexpr (Op == CompareOp::GE) return vcgeq_s32(data, threshold);
    if constexpr (Op == CompareOp::LT) return vcltq_s32(data, threshold);
    if constexpr (Op == CompareOp::LE) return vcleq_s32(data, threshold);
    if constexpr (Op == CompareOp::EQ) return vceqq_s32(data, threshold);
    if constexpr (Op == CompareOp::NE) return vmvnq_u32(vceqq_s32(data, threshold));
    __builtin_unreachable();
}

__attribute__((always_inline))
inline uint32_t extract_mask_4(uint32x4_t mask) {
    uint32x4_t bits = vshrq_n_u32(mask, 31);
    return vgetq_lane_u32(bits, 0) |
           (vgetq_lane_u32(bits, 1) << 1) |
           (vgetq_lane_u32(bits, 2) << 2) |
           (vgetq_lane_u32(bits, 3) << 3);
}

// ============================================================================
// 阶段 1: 统计匹配数 (与 V19 相同)
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t count_matches_chunk(const int32_t* __restrict input,
                           size_t start, size_t end,
                           int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t count = 0;
    size_t i = start;

    for (; i + 64 <= end; i += 64) {
        __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

        uint32_t total_bits = 0;
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
            total_bits += __builtin_popcount(extract_mask_4(mask));
        }
        count += total_bits;
    }

    for (; i + 16 <= end; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d0, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d1, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d2, threshold)));
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(d3, threshold)));
    }

    for (; i + 4 <= end; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        count += __builtin_popcount(extract_mask_4(simd_compare_i32<Op>(data, threshold)));
    }

    for (; i < end; ++i) {
        bool match = false;
        if constexpr (Op == CompareOp::GT) match = input[i] > value;
        else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
        else if constexpr (Op == CompareOp::LT) match = input[i] < value;
        else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
        else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
        else if constexpr (Op == CompareOp::NE) match = input[i] != value;
        if (match) ++count;
    }

    return count;
}

// ============================================================================
// 阶段 2: 直接写入输出 (与 V19 相同)
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
void write_matches_chunk(const int32_t* __restrict input,
                         size_t start, size_t end,
                         int32_t value,
                         uint32_t* __restrict output,
                         size_t write_offset) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t out_idx = write_offset;
    size_t i = start;

    for (; i + 64 <= end; i += 64) {
        __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

        uint64_t combined_mask = 0;
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            int32x4_t data = vld1q_s32(input + i + g * 4);
            uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
            uint32_t bits = extract_mask_4(mask);
            combined_mask |= ((uint64_t)bits << (g * 4));
        }

        if (combined_mask == 0) continue;

        uint32_t base = static_cast<uint32_t>(i);
        while (combined_mask) {
            uint32_t pos = __builtin_ctzll(combined_mask);
            output[out_idx++] = base + pos;
            combined_mask &= combined_mask - 1;
        }
    }

    for (; i + 16 <= end; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32_t combined = extract_mask_4(simd_compare_i32<Op>(d0, threshold)) |
                           (extract_mask_4(simd_compare_i32<Op>(d1, threshold)) << 4) |
                           (extract_mask_4(simd_compare_i32<Op>(d2, threshold)) << 8) |
                           (extract_mask_4(simd_compare_i32<Op>(d3, threshold)) << 12);

        if (combined == 0) continue;

        uint32_t base = static_cast<uint32_t>(i);
        while (combined) {
            uint32_t pos = __builtin_ctz(combined);
            output[out_idx++] = base + pos;
            combined &= combined - 1;
        }
    }

    for (; i + 4 <= end; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32_t bits = extract_mask_4(simd_compare_i32<Op>(data, threshold));
        if (bits) {
            uint32_t base = static_cast<uint32_t>(i);
            while (bits) {
                uint32_t pos = __builtin_ctz(bits);
                output[out_idx++] = base + pos;
                bits &= bits - 1;
            }
        }
    }

    for (; i < end; ++i) {
        bool match = false;
        if constexpr (Op == CompareOp::GT) match = input[i] > value;
        else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
        else if constexpr (Op == CompareOp::LT) match = input[i] < value;
        else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
        else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
        else if constexpr (Op == CompareOp::NE) match = input[i] != value;
        if (match) output[out_idx++] = static_cast<uint32_t>(i);
    }
}

// ============================================================================
// V19.1 核心实现: 线程池 + 两阶段并行
// ============================================================================

template<CompareOp Op>
__attribute__((noinline))
size_t filter_i32_v19_1_core(const int32_t* __restrict input, size_t count,
                              int32_t value, uint32_t* __restrict out_indices) {
    // 小数据量: 单线程
    if (count < MIN_PARALLEL) {
        size_t out_count = 0;
        int32x4_t threshold = vdupq_n_s32(value);
        size_t i = 0;

        for (; i + 64 <= count; i += 64) {
            __builtin_prefetch(input + i + PREFETCH_DISTANCE, 0, 0);

            uint64_t combined_mask = 0;
            #pragma unroll
            for (int g = 0; g < 16; ++g) {
                int32x4_t data = vld1q_s32(input + i + g * 4);
                uint32x4_t mask = simd_compare_i32<Op>(data, threshold);
                combined_mask |= ((uint64_t)extract_mask_4(mask) << (g * 4));
            }

            if (combined_mask == 0) continue;

            uint32_t base = static_cast<uint32_t>(i);
            while (combined_mask) {
                uint32_t pos = __builtin_ctzll(combined_mask);
                out_indices[out_count++] = base + pos;
                combined_mask &= combined_mask - 1;
            }
        }

        for (; i + 4 <= count; i += 4) {
            int32x4_t data = vld1q_s32(input + i);
            uint32_t bits = extract_mask_4(simd_compare_i32<Op>(data, threshold));
            if (bits) {
                uint32_t base = static_cast<uint32_t>(i);
                while (bits) {
                    uint32_t pos = __builtin_ctz(bits);
                    out_indices[out_count++] = base + pos;
                    bits &= bits - 1;
                }
            }
        }

        for (; i < count; ++i) {
            bool match = false;
            if constexpr (Op == CompareOp::GT) match = input[i] > value;
            else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
            else if constexpr (Op == CompareOp::LT) match = input[i] < value;
            else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
            else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
            else if constexpr (Op == CompareOp::NE) match = input[i] != value;
            if (match) out_indices[out_count++] = static_cast<uint32_t>(i);
        }

        return out_count;
    }

    // 大数据量: 线程池 + 两阶段并行
    auto& pool = get_thread_pool();
    const size_t num_threads = pool.size();
    const size_t chunk_size = (count + num_threads - 1) / num_threads;

    std::vector<size_t> thread_starts(num_threads);
    std::vector<size_t> thread_ends(num_threads);
    std::vector<size_t> thread_counts(num_threads, 0);
    std::atomic<size_t> completed{0};

    for (size_t t = 0; t < num_threads; ++t) {
        thread_starts[t] = t * chunk_size;
        thread_ends[t] = std::min(thread_starts[t] + chunk_size, count);
    }

    // ========================================================================
    // 阶段 1: 线程池并行统计
    // ========================================================================
    for (size_t t = 0; t < num_threads; ++t) {
        if (thread_starts[t] >= thread_ends[t]) {
            completed.fetch_add(1, std::memory_order_release);
            continue;
        }

        pool.submit([&, t]() {
            thread_counts[t] = count_matches_chunk<Op>(
                input, thread_starts[t], thread_ends[t], value);
            completed.fetch_add(1, std::memory_order_release);
        });
    }

    pool.wait_all(completed, num_threads);

    // 计算前缀和
    std::vector<size_t> write_offsets(num_threads);
    size_t total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        write_offsets[t] = total;
        total += thread_counts[t];
    }

    if (total == 0) return 0;

    // ========================================================================
    // 阶段 2: 线程池并行直写
    // ========================================================================
    completed.store(0, std::memory_order_release);

    for (size_t t = 0; t < num_threads; ++t) {
        if (thread_starts[t] >= thread_ends[t] || thread_counts[t] == 0) {
            completed.fetch_add(1, std::memory_order_release);
            continue;
        }

        pool.submit([&, t]() {
            write_matches_chunk<Op>(
                input, thread_starts[t], thread_ends[t],
                value, out_indices, write_offsets[t]);
            completed.fetch_add(1, std::memory_order_release);
        });
    }

    pool.wait_all(completed, num_threads);

    return total;
}

} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

size_t filter_i32_v19_1(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         uint32_t* out_indices) {
    if (!input || !out_indices || count == 0) return 0;

    switch (op) {
        case CompareOp::GT: return filter_i32_v19_1_core<CompareOp::GT>(input, count, value, out_indices);
        case CompareOp::GE: return filter_i32_v19_1_core<CompareOp::GE>(input, count, value, out_indices);
        case CompareOp::LT: return filter_i32_v19_1_core<CompareOp::LT>(input, count, value, out_indices);
        case CompareOp::LE: return filter_i32_v19_1_core<CompareOp::LE>(input, count, value, out_indices);
        case CompareOp::EQ: return filter_i32_v19_1_core<CompareOp::EQ>(input, count, value, out_indices);
        case CompareOp::NE: return filter_i32_v19_1_core<CompareOp::NE>(input, count, value, out_indices);
        default: return 0;
    }
}

const char* get_filter_v19_1_version() {
    return "V19.1 - Thread Pool + Two-Phase Direct Write (8T)";
}

#else

// 非 ARM 平台回退
size_t filter_i32_v19_1(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         uint32_t* out_indices) {
    return filter_i32(input, count, op, value, out_indices);
}

const char* get_filter_v19_1_version() {
    return "V19.1 - Fallback";
}

#endif // __aarch64__

} // namespace filter
} // namespace thunderduck
