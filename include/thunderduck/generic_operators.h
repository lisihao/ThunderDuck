/**
 * ThunderDuck Generic Operators
 *
 * 通用高性能算子库 - V51
 *
 * 包含:
 * 1. ParallelRadixSort - 两级并行基数排序 (Q21)
 * 2. PartitionedAggregation - 分区聚合 (Q3/Q5)
 * 3. FusedFilterAggregate - Filter-Aggregate 融合 (Q6)
 *
 * 设计原则:
 * - 通用化: 支持多种数据类型和聚合函数
 * - Cache 友好: L1/L2 大小的局部处理
 * - 无分支: SIMD + mask 操作
 * - 可注册: 集成到系统表和优化器
 *
 * @version 1.0
 * @date 2026-01-29
 */

#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>
#include <atomic>
#include <type_traits>
#include <unordered_map>
#include <arm_neon.h>

// C++17 兼容性
#if __cplusplus >= 201703L
    #define TD_IF_CONSTEXPR if constexpr
#else
    #define TD_IF_CONSTEXPR if
#endif

namespace thunderduck {
namespace operators {

// ============================================================================
// 常量定义
// ============================================================================

// Cache 大小 (Apple M4)
constexpr size_t L1_CACHE_SIZE = 128 * 1024;      // 128 KB
constexpr size_t L2_CACHE_SIZE = 4 * 1024 * 1024; // 4 MB per cluster
constexpr size_t CACHE_LINE_SIZE = 128;           // M4 cache line

// 默认线程数
constexpr size_t DEFAULT_THREADS = 8;

// ============================================================================
// 1. ParallelRadixSort - 两级并行基数排序
// ============================================================================

/**
 * 两级并行基数排序
 *
 * Phase 1: per-thread local radix (L1-sized buckets)
 * Phase 2: global merge (prefix-sum + memcpy)
 *
 * 特点:
 * - 每个线程私有 bucket
 * - bucket 尺寸 <= L1/L2
 * - 禁止跨线程写同一 cache line
 * - ARM 顺序写优化
 */
template<typename Key, typename Value = void>
class ParallelRadixSort {
public:
    // 配置
    struct Config {
        size_t num_threads = DEFAULT_THREADS;
        size_t radix_bits = 8;           // 每 pass 的位数
        size_t max_passes = 4;           // 最大 pass 数
        bool ascending = true;
    };

    // 结果统计
    struct Stats {
        size_t total_elements;
        size_t num_passes;
        double histogram_time_ms;
        double scatter_time_ms;
        double merge_time_ms;
        double total_time_ms;
    };

    explicit ParallelRadixSort(const Config& config = Config{})
        : config_(config) {
        num_buckets_ = 1ULL << config_.radix_bits;
        init_thread_local_storage();
    }

    /**
     * 排序 key-value 对
     */
    template<typename KeyIter, typename ValueIter>
    Stats sort(KeyIter keys_begin, KeyIter keys_end,
               ValueIter values_begin) {
        Stats stats{};
        stats.total_elements = std::distance(keys_begin, keys_end);

        if (stats.total_elements == 0) return stats;

        auto start = std::chrono::high_resolution_clock::now();

        // 分配临时缓冲区
        std::vector<Key> temp_keys(stats.total_elements);
        std::vector<Value> temp_values(stats.total_elements);

        Key* src_keys = &(*keys_begin);
        Key* dst_keys = temp_keys.data();
        Value* src_values = &(*values_begin);
        Value* dst_values = temp_values.data();

        // 确定需要的 pass 数
        Key max_key = *std::max_element(keys_begin, keys_end);
        stats.num_passes = compute_passes(max_key);

        // 执行多 pass 基数排序
        for (size_t pass = 0; pass < stats.num_passes; ++pass) {
            size_t shift = pass * config_.radix_bits;

            // Phase 1: 并行 histogram
            auto hist_start = std::chrono::high_resolution_clock::now();
            parallel_histogram(src_keys, stats.total_elements, shift);
            auto hist_end = std::chrono::high_resolution_clock::now();
            stats.histogram_time_ms += std::chrono::duration<double, std::milli>(
                hist_end - hist_start).count();

            // Phase 2: prefix sum (单线程)
            compute_global_offsets();

            // Phase 3: 并行 scatter (无分支)
            auto scatter_start = std::chrono::high_resolution_clock::now();
            parallel_scatter(src_keys, src_values, dst_keys, dst_values,
                           stats.total_elements, shift);
            auto scatter_end = std::chrono::high_resolution_clock::now();
            stats.scatter_time_ms += std::chrono::duration<double, std::milli>(
                scatter_end - scatter_start).count();

            // 交换 src/dst
            std::swap(src_keys, dst_keys);
            std::swap(src_values, dst_values);
        }

        // 如果最终结果在临时缓冲区，拷贝回去
        if (stats.num_passes % 2 == 1) {
            std::copy(temp_keys.begin(), temp_keys.end(), keys_begin);
            std::copy(temp_values.begin(), temp_values.end(), values_begin);
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(
            end - start).count();

        return stats;
    }

    /**
     * 仅排序 key (无 value)
     */
    template<typename KeyIter>
    Stats sort_keys_only(KeyIter keys_begin, KeyIter keys_end) {
        // 创建虚拟 value
        std::vector<uint8_t> dummy(std::distance(keys_begin, keys_end));
        return sort(keys_begin, keys_end, dummy.begin());
    }

private:
    Config config_;
    size_t num_buckets_;

    // 线程局部存储 (避免 false sharing)
    struct alignas(CACHE_LINE_SIZE) ThreadLocalData {
        std::vector<uint32_t> histogram;  // 局部 histogram
        std::vector<size_t> offsets;      // 局部 offset
        uint8_t padding[CACHE_LINE_SIZE]; // 填充避免 false sharing
    };
    std::vector<ThreadLocalData> thread_data_;

    // 全局 offset
    std::vector<size_t> global_offsets_;

    void init_thread_local_storage() {
        thread_data_.resize(config_.num_threads);
        for (auto& td : thread_data_) {
            td.histogram.resize(num_buckets_, 0);
            td.offsets.resize(num_buckets_, 0);
        }
        global_offsets_.resize(num_buckets_ + 1, 0);
    }

    size_t compute_passes(Key max_key) const {
        if (max_key == 0) return 1;

        size_t bits_needed = 0;
        while (max_key > 0) {
            bits_needed++;
            max_key >>= 1;
        }

        size_t passes = (bits_needed + config_.radix_bits - 1) / config_.radix_bits;
        return std::min(passes, config_.max_passes);
    }

    void parallel_histogram(const Key* keys, size_t n, size_t shift) {
        size_t mask = num_buckets_ - 1;
        size_t chunk_size = (n + config_.num_threads - 1) / config_.num_threads;

        // 清零所有 histogram
        for (auto& td : thread_data_) {
            std::fill(td.histogram.begin(), td.histogram.end(), 0);
        }

        // 并行计算 histogram
        std::vector<std::thread> threads;
        for (size_t t = 0; t < config_.num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);

            threads.emplace_back([this, keys, start, end, shift, mask, t]() {
                auto& hist = thread_data_[t].histogram;
                for (size_t i = start; i < end; ++i) {
                    size_t bucket = (keys[i] >> shift) & mask;
                    hist[bucket]++;
                }
            });
        }

        for (auto& th : threads) th.join();
    }

    void compute_global_offsets() {
        // 合并所有线程的 histogram
        std::vector<size_t> total_hist(num_buckets_, 0);
        for (const auto& td : thread_data_) {
            for (size_t i = 0; i < num_buckets_; ++i) {
                total_hist[i] += td.histogram[i];
            }
        }

        // 计算全局 prefix sum
        global_offsets_[0] = 0;
        for (size_t i = 0; i < num_buckets_; ++i) {
            global_offsets_[i + 1] = global_offsets_[i] + total_hist[i];
        }

        // 计算每个线程的局部 offset
        std::vector<size_t> running_offset(num_buckets_, 0);
        for (size_t t = 0; t < config_.num_threads; ++t) {
            auto& td = thread_data_[t];
            for (size_t i = 0; i < num_buckets_; ++i) {
                td.offsets[i] = global_offsets_[i] + running_offset[i];
                running_offset[i] += td.histogram[i];
            }
        }
    }

    void parallel_scatter(const Key* src_keys, const Value* src_values,
                         Key* dst_keys, Value* dst_values,
                         size_t n, size_t shift) {
        size_t mask = num_buckets_ - 1;
        size_t chunk_size = (n + config_.num_threads - 1) / config_.num_threads;

        std::vector<std::thread> threads;
        for (size_t t = 0; t < config_.num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);

            threads.emplace_back([this, src_keys, src_values, dst_keys, dst_values,
                                 start, end, shift, mask, t]() {
                auto& offsets = thread_data_[t].offsets;

                // 无分支 scatter: pointer bump
                for (size_t i = start; i < end; ++i) {
                    size_t bucket = (src_keys[i] >> shift) & mask;
                    size_t pos = offsets[bucket]++;

                    // 顺序写 (ARM 友好)
                    dst_keys[pos] = src_keys[i];
                    if (!std::is_same<Value, void>::value) {
                        dst_values[pos] = src_values[i];
                    }
                }
            });
        }

        for (auto& th : threads) th.join();
    }
};

// ============================================================================
// 2. PartitionedAggregation - 分区聚合
// ============================================================================

/**
 * 分区聚合算子
 *
 * 架构:
 * Scan → Partition(key % P) → Local Aggregate → Merge
 *
 * 特点:
 * - Partition 数 = 2-4x 线程数
 * - 每个 partition 的 HT 是短命的 (用完即 free)
 * - 支持 heavy hitter 单独分区
 * - Cache 友好
 */
template<typename Key, typename Value, typename AggState>
class PartitionedAggregation {
public:
    // 聚合函数接口
    using InitFunc = std::function<AggState()>;
    using UpdateFunc = std::function<void(AggState&, const Value&)>;
    using MergeFunc = std::function<void(AggState&, const AggState&)>;
    using FinalizeFunc = std::function<Value(const AggState&)>;

    struct Config {
        size_t num_threads = DEFAULT_THREADS;
        size_t num_partitions = 0;        // 0 = auto (2x threads)
        size_t max_ht_size = L2_CACHE_SIZE / 4;  // 每个 partition HT 上限
        bool detect_heavy_hitters = true;
        size_t heavy_hitter_threshold = 1000;  // Top-K heavy hitter
    };

    struct Stats {
        size_t total_rows;
        size_t unique_keys;
        size_t num_partitions;
        size_t heavy_hitters_count;
        double partition_time_ms;
        double aggregate_time_ms;
        double merge_time_ms;
        double total_time_ms;
    };

    PartitionedAggregation(
        InitFunc init,
        UpdateFunc update,
        MergeFunc merge,
        FinalizeFunc finalize = nullptr,
        const Config& config = Config{}
    ) : init_(init), update_(update), merge_(merge), finalize_(finalize),
        config_(config) {
        if (config_.num_partitions == 0) {
            config_.num_partitions = config_.num_threads * 2;
        }
    }

    /**
     * 执行分区聚合
     */
    template<typename KeyIter, typename ValueIter, typename OutputFunc>
    Stats aggregate(KeyIter keys_begin, KeyIter keys_end,
                   ValueIter values_begin,
                   OutputFunc output) {
        Stats stats{};
        stats.total_rows = std::distance(keys_begin, keys_end);
        stats.num_partitions = config_.num_partitions;

        if (stats.total_rows == 0) return stats;

        auto start = std::chrono::high_resolution_clock::now();

        // Phase 1: 分区
        auto part_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::pair<Key, Value>>> partitions(config_.num_partitions);

        // 预估每个 partition 大小
        size_t est_size = (stats.total_rows / config_.num_partitions) * 1.2;
        for (auto& p : partitions) {
            p.reserve(est_size);
        }

        // 分区: key % num_partitions
        auto kit = keys_begin;
        auto vit = values_begin;
        for (; kit != keys_end; ++kit, ++vit) {
            size_t pid = static_cast<size_t>(*kit) % config_.num_partitions;
            partitions[pid].emplace_back(*kit, *vit);
        }

        auto part_end = std::chrono::high_resolution_clock::now();
        stats.partition_time_ms = std::chrono::duration<double, std::milli>(
            part_end - part_start).count();

        // Phase 2: 并行局部聚合
        auto agg_start = std::chrono::high_resolution_clock::now();

        using LocalHT = std::unordered_map<Key, AggState>;
        std::vector<LocalHT> local_results(config_.num_partitions);

        std::vector<std::thread> threads;
        size_t partitions_per_thread = (config_.num_partitions + config_.num_threads - 1)
                                       / config_.num_threads;

        for (size_t t = 0; t < config_.num_threads; ++t) {
            size_t p_start = t * partitions_per_thread;
            size_t p_end = std::min(p_start + partitions_per_thread, config_.num_partitions);

            threads.emplace_back([this, &partitions, &local_results, p_start, p_end]() {
                for (size_t p = p_start; p < p_end; ++p) {
                    auto& ht = local_results[p];
                    for (const auto& kv : partitions[p]) {
                        const Key& key = kv.first;
                        const Value& value = kv.second;
                        auto it = ht.find(key);
                        if (it == ht.end()) {
                            ht[key] = init_();
                            update_(ht[key], value);
                        } else {
                            update_(it->second, value);
                        }
                    }
                    // 释放 partition 内存
                    partitions[p].clear();
                    partitions[p].shrink_to_fit();
                }
            });
        }

        for (auto& th : threads) th.join();

        auto agg_end = std::chrono::high_resolution_clock::now();
        stats.aggregate_time_ms = std::chrono::duration<double, std::milli>(
            agg_end - agg_start).count();

        // Phase 3: 合并结果
        auto merge_start = std::chrono::high_resolution_clock::now();

        // 由于 key 已经按 partition 分开，不需要跨 partition 合并
        // 直接输出每个 partition 的结果
        for (size_t p = 0; p < config_.num_partitions; ++p) {
            for (const auto& kv : local_results[p]) {
                const Key& key = kv.first;
                const AggState& state = kv.second;
                if (finalize_) {
                    output(key, finalize_(state));
                } else {
                    // 直接输出 state (假设 AggState 可转换为 Value)
                    output(key, static_cast<Value>(state));
                }
                stats.unique_keys++;
            }
        }

        auto merge_end = std::chrono::high_resolution_clock::now();
        stats.merge_time_ms = std::chrono::duration<double, std::milli>(
            merge_end - merge_start).count();

        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(
            end - start).count();

        return stats;
    }

private:
    InitFunc init_;
    UpdateFunc update_;
    MergeFunc merge_;
    FinalizeFunc finalize_;
    Config config_;
};

// ============================================================================
// 3. FusedFilterAggregate - Filter-Aggregate 融合
// ============================================================================

/**
 * Filter-Aggregate 融合算子
 *
 * 模式:
 * for each tuple:
 *   mask = predicate(x)
 *   agg += mask * value
 *
 * 特点:
 * - 无分支: 使用 mask 乘法
 * - SIMD: 批量处理 4/8 条
 * - 无中间物化
 */
class FusedFilterAggregate {
public:
    // 谓词类型
    enum class PredicateOp {
        EQ, NE, LT, LE, GT, GE, BETWEEN, IN
    };

    // 聚合类型
    enum class AggregateOp {
        SUM, COUNT, MIN, MAX, AVG
    };

    struct Config {
        size_t num_threads;
        size_t simd_batch_size;

        Config() : num_threads(DEFAULT_THREADS), simd_batch_size(4) {}
    };

    struct Stats {
        size_t total_rows;
        size_t matched_rows;
        double filter_time_ms;
        double aggregate_time_ms;
        double total_time_ms;
    };

    FusedFilterAggregate() : config_() {}

    explicit FusedFilterAggregate(const Config& config)
        : config_(config) {}

    /**
     * 融合 Filter + SUM
     *
     * SQL: SELECT SUM(value_col * multiplier)
     *      FROM table
     *      WHERE filter_col >= low AND filter_col < high
     *            AND cond2_col BETWEEN low2 AND high2
     *            AND cond3_col < threshold
     */
    template<typename T>
    Stats fused_filter_sum(
        const T* filter_col,           // 过滤列 1
        const T* cond2_col,            // 过滤列 2 (可选, nullptr 跳过)
        const T* cond3_col,            // 过滤列 3 (可选, nullptr 跳过)
        const T* value_col,            // 值列
        const T* multiplier_col,       // 乘数列 (可选, nullptr = 1)
        size_t n,
        T filter_low, T filter_high,   // 条件 1: [low, high)
        T cond2_low, T cond2_high,     // 条件 2: [low, high]
        T cond3_threshold,             // 条件 3: < threshold
        T& result                      // 输出
    ) {
        Stats stats{};
        stats.total_rows = n;

        auto start = std::chrono::high_resolution_clock::now();

        // 多线程并行
        size_t chunk_size = (n + config_.num_threads - 1) / config_.num_threads;
        std::vector<T> partial_sums(config_.num_threads, T{0});
        std::vector<size_t> partial_counts(config_.num_threads, 0);

        std::vector<std::thread> threads;
        for (size_t t = 0; t < config_.num_threads; ++t) {
            size_t start_idx = t * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, n);

            threads.emplace_back([&, t, start_idx, end_idx]() {
                T local_sum = T{0};
                size_t local_count = 0;

                // SIMD 处理 (4 条一批)
                size_t i = start_idx;

                // 对齐到 4 的倍数
                for (; i < end_idx && (i % 4) != 0; ++i) {
                    // 标量处理
                    bool mask = (filter_col[i] >= filter_low && filter_col[i] < filter_high);
                    if (cond2_col) {
                        mask = mask && (cond2_col[i] >= cond2_low && cond2_col[i] <= cond2_high);
                    }
                    if (cond3_col) {
                        mask = mask && (cond3_col[i] < cond3_threshold);
                    }

                    if (mask) {
                        T val = value_col[i];
                        if (multiplier_col) val *= multiplier_col[i];
                        local_sum += val;
                        local_count++;
                    }
                }

                // SIMD 批量处理 (仅 float 类型启用)
                if (std::is_same<T, float>::value) {
                    const float* f_filter = reinterpret_cast<const float*>(filter_col);
                    const float* f_value = reinterpret_cast<const float*>(value_col);
                    const float* f_cond2 = reinterpret_cast<const float*>(cond2_col);
                    const float* f_cond3 = reinterpret_cast<const float*>(cond3_col);
                    const float* f_mult = reinterpret_cast<const float*>(multiplier_col);
                    float f_sum = 0.0f;

                    for (; i + 4 <= end_idx; i += 4) {
                        float32x4_t filter_vec = vld1q_f32(&f_filter[i]);
                        float32x4_t value_vec = vld1q_f32(&f_value[i]);

                        // 条件 1: filter_col >= low && filter_col < high
                        uint32x4_t mask1 = vcgeq_f32(filter_vec, vdupq_n_f32(static_cast<float>(filter_low)));
                        uint32x4_t mask2 = vcltq_f32(filter_vec, vdupq_n_f32(static_cast<float>(filter_high)));
                        uint32x4_t mask = vandq_u32(mask1, mask2);

                        // 条件 2 (如果有)
                        if (f_cond2) {
                            float32x4_t cond2_vec = vld1q_f32(&f_cond2[i]);
                            uint32x4_t m2_low = vcgeq_f32(cond2_vec, vdupq_n_f32(static_cast<float>(cond2_low)));
                            uint32x4_t m2_high = vcleq_f32(cond2_vec, vdupq_n_f32(static_cast<float>(cond2_high)));
                            mask = vandq_u32(mask, vandq_u32(m2_low, m2_high));
                        }

                        // 条件 3 (如果有)
                        if (f_cond3) {
                            float32x4_t cond3_vec = vld1q_f32(&f_cond3[i]);
                            uint32x4_t m3 = vcltq_f32(cond3_vec, vdupq_n_f32(static_cast<float>(cond3_threshold)));
                            mask = vandq_u32(mask, m3);
                        }

                        // 乘数 (如果有)
                        if (f_mult) {
                            float32x4_t mult_vec = vld1q_f32(&f_mult[i]);
                            value_vec = vmulq_f32(value_vec, mult_vec);
                        }

                        // 无分支累加: mask * value
                        float32x4_t mask_f = vcvtq_f32_u32(vandq_u32(mask, vdupq_n_u32(1)));
                        float32x4_t masked_val = vmulq_f32(value_vec, mask_f);

                        // 水平累加
                        f_sum += vaddvq_f32(masked_val);

                        // 统计匹配行数
                        uint32_t mask_arr[4];
                        vst1q_u32(mask_arr, mask);
                        for (int j = 0; j < 4; ++j) {
                            if (mask_arr[j]) local_count++;
                        }
                    }
                    local_sum += static_cast<T>(f_sum);  // 累加而非覆盖
                } else if (std::is_same<T, int32_t>::value) {
                    const int32_t* i_filter = reinterpret_cast<const int32_t*>(filter_col);
                    const int32_t* i_value = reinterpret_cast<const int32_t*>(value_col);
                    const int32_t* i_cond2 = reinterpret_cast<const int32_t*>(cond2_col);
                    const int32_t* i_cond3 = reinterpret_cast<const int32_t*>(cond3_col);
                    const int32_t* i_mult = reinterpret_cast<const int32_t*>(multiplier_col);
                    int32_t i_sum = 0;

                    for (; i + 4 <= end_idx; i += 4) {
                        int32x4_t filter_vec = vld1q_s32(&i_filter[i]);
                        int32x4_t value_vec = vld1q_s32(&i_value[i]);

                        // 条件 1
                        uint32x4_t mask1 = vcgeq_s32(filter_vec, vdupq_n_s32(static_cast<int32_t>(filter_low)));
                        uint32x4_t mask2 = vcltq_s32(filter_vec, vdupq_n_s32(static_cast<int32_t>(filter_high)));
                        uint32x4_t mask = vandq_u32(mask1, mask2);

                        // 条件 2
                        if (i_cond2) {
                            int32x4_t cond2_vec = vld1q_s32(&i_cond2[i]);
                            uint32x4_t m2_low = vcgeq_s32(cond2_vec, vdupq_n_s32(static_cast<int32_t>(cond2_low)));
                            uint32x4_t m2_high = vcleq_s32(cond2_vec, vdupq_n_s32(static_cast<int32_t>(cond2_high)));
                            mask = vandq_u32(mask, vandq_u32(m2_low, m2_high));
                        }

                        // 条件 3
                        if (i_cond3) {
                            int32x4_t cond3_vec = vld1q_s32(&i_cond3[i]);
                            uint32x4_t m3 = vcltq_s32(cond3_vec, vdupq_n_s32(static_cast<int32_t>(cond3_threshold)));
                            mask = vandq_u32(mask, m3);
                        }

                        // 乘法
                        if (i_mult) {
                            int32x4_t mult_vec = vld1q_s32(&i_mult[i]);
                            value_vec = vmulq_s32(value_vec, mult_vec);
                        }

                        // 无分支累加
                        int32x4_t mask_i = vreinterpretq_s32_u32(vshrq_n_u32(mask, 31));
                        int32x4_t masked_val = vmulq_s32(value_vec, mask_i);
                        i_sum += vaddvq_s32(masked_val);

                        // 统计
                        uint32_t mask_arr[4];
                        vst1q_u32(mask_arr, mask);
                        for (int j = 0; j < 4; ++j) {
                            if (mask_arr[j]) local_count++;
                        }
                    }
                    local_sum += static_cast<T>(i_sum);  // 累加而非覆盖
                }

                // 处理剩余元素
                for (; i < end_idx; ++i) {
                    bool mask = (filter_col[i] >= filter_low && filter_col[i] < filter_high);
                    if (cond2_col) {
                        mask = mask && (cond2_col[i] >= cond2_low && cond2_col[i] <= cond2_high);
                    }
                    if (cond3_col) {
                        mask = mask && (cond3_col[i] < cond3_threshold);
                    }

                    if (mask) {
                        T val = value_col[i];
                        if (multiplier_col) val *= multiplier_col[i];
                        local_sum += val;
                        local_count++;
                    }
                }

                partial_sums[t] = local_sum;
                partial_counts[t] = local_count;
            });
        }

        for (auto& th : threads) th.join();

        // 合并结果
        result = T{0};
        for (size_t t = 0; t < config_.num_threads; ++t) {
            result += partial_sums[t];
            stats.matched_rows += partial_counts[t];
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(
            end - start).count();

        return stats;
    }

    /**
     * 融合 Filter + COUNT
     */
    template<typename T>
    Stats fused_filter_count(
        const T* filter_col,
        size_t n,
        T filter_low, T filter_high,
        size_t& result
    ) {
        Stats stats{};
        stats.total_rows = n;

        auto start = std::chrono::high_resolution_clock::now();

        size_t chunk_size = (n + config_.num_threads - 1) / config_.num_threads;
        std::vector<size_t> partial_counts(config_.num_threads, 0);

        std::vector<std::thread> threads;
        for (size_t t = 0; t < config_.num_threads; ++t) {
            size_t start_idx = t * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, n);

            threads.emplace_back([&, t, start_idx, end_idx]() {
                size_t local_count = 0;

                // SIMD 批量处理
                if (std::is_same<T, int32_t>::value) {
                    size_t i = start_idx;
                    for (; i + 4 <= end_idx; i += 4) {
                        int32x4_t filter_vec = vld1q_s32(&filter_col[i]);
                        uint32x4_t mask1 = vcgeq_s32(filter_vec, vdupq_n_s32(filter_low));
                        uint32x4_t mask2 = vcltq_s32(filter_vec, vdupq_n_s32(filter_high));
                        uint32x4_t mask = vandq_u32(mask1, mask2);

                        // popcount
                        local_count += (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) +
                                      (mask[2] ? 1 : 0) + (mask[3] ? 1 : 0);
                    }

                    for (; i < end_idx; ++i) {
                        if (filter_col[i] >= filter_low && filter_col[i] < filter_high) {
                            local_count++;
                        }
                    }
                } else {
                    for (size_t i = start_idx; i < end_idx; ++i) {
                        if (filter_col[i] >= filter_low && filter_col[i] < filter_high) {
                            local_count++;
                        }
                    }
                }

                partial_counts[t] = local_count;
            });
        }

        for (auto& th : threads) th.join();

        result = 0;
        for (size_t t = 0; t < config_.num_threads; ++t) {
            result += partial_counts[t];
        }
        stats.matched_rows = result;

        auto end = std::chrono::high_resolution_clock::now();
        stats.total_time_ms = std::chrono::duration<double, std::milli>(
            end - start).count();

        return stats;
    }

private:
    Config config_;
};

// ============================================================================
// 算子注册信息
// ============================================================================

struct OperatorInfo {
    const char* name;
    const char* version;
    const char* description;
    float startup_cost_ms;
    float per_row_cost_us;
};

inline std::vector<OperatorInfo> get_generic_operator_infos() {
    return {
        {"ParallelRadixSort", "V51", "两级并行基数排序，L1-sized buckets", 0.5f, 0.005f},
        {"PartitionedAggregation", "V51", "分区聚合，cache 友好", 0.3f, 0.008f},
        {"FusedFilterAggregate", "V51", "Filter-Aggregate 融合，无分支 SIMD", 0.1f, 0.002f}
    };
}

} // namespace operators
} // namespace thunderduck
