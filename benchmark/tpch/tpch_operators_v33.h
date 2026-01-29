/**
 * ThunderDuck TPC-H V33 通用算子
 *
 * V33 架构: 通用化 + 无硬编码 + 保持 V32 性能
 *
 * Layer 2: 通用算子
 * - DateRangeFilter: 日期范围过滤
 * - StringSetMatcher: 字符串集合匹配
 * - AdaptiveHashJoin: 自适应 Hash Join
 * - ThreadLocalAggregator: 线程本地聚合器
 * - PredicatePrecomputer: 条件预计算器
 *
 * Layer 3: 执行引擎
 * - TaskScheduler: 任务调度器
 * - BatchProcessor: 批量处理器
 *
 * Layer 4: 自动调优
 * - DataStatistics: 数据统计
 * - HardwareProfile: 硬件配置
 * - AutoTuner: 自动调优器
 *
 * @version 33.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V33_H
#define TPCH_OPERATORS_V33_H

#include "tpch_config_v33.h"
#include "tpch_data_loader.h"
#include "tpch_operators_v32.h"  // 复用 V32 的 CompactHashTable, BatchHasher 等
#include "tpch_operators_v25.h"  // 复用 ThreadPool

#include <cstdint>
#include <vector>
#include <atomic>
#include <thread>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#ifdef __aarch64__
#include <arm_acle.h>
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v33 {

using ::thunderduck::tpch::TPCHDataLoader;
using ops_v32::CompactHashTable;
using ops_v32::BatchHasher;
using ops_v32::SingleHashBloomFilter;
using ops_v25::ThreadPool;

// ============================================================================
// Layer 2: 通用算子接口
// ============================================================================

// ============================================================================
// 日期范围过滤器
// ============================================================================

/**
 * 日期范围过滤器
 *
 * 支持从 QueryConfig 配置，返回匹配的行索引
 */
class DateRangeFilter {
public:
    DateRangeFilter() = default;
    DateRangeFilter(const int32_t* dates, size_t count)
        : dates_(dates), count_(count) {}

    /**
     * 从配置加载日期范围参数
     */
    void configure(const QueryConfig& cfg, const std::string& param_name) {
        range_ = cfg.get_date_range(param_name);
    }

    /**
     * 设置日期范围
     */
    void set_range(DateRange range) {
        range_ = range;
    }

    /**
     * 执行过滤，返回匹配的行索引
     */
    std::vector<uint32_t> execute();

    /**
     * 执行过滤并应用到结果位图
     * @param bitmap 输出位图 (预分配)
     */
    void execute_bitmap(std::vector<uint64_t>& bitmap);

    /**
     * 检查单个日期
     */
    __attribute__((always_inline))
    bool test(size_t idx) const {
        return range_.contains(dates_[idx]);
    }

private:
    const int32_t* dates_ = nullptr;
    size_t count_ = 0;
    DateRange range_;
};

// ============================================================================
// 字符串集合匹配器
// ============================================================================

/**
 * 字符串集合匹配器
 *
 * 预计算字符串匹配结果，支持多种谓词类型
 */
class StringSetMatcher {
public:
    StringSetMatcher() = default;

    /**
     * 从配置加载字符串参数
     */
    void configure_equals(const QueryConfig& cfg, const std::string& param_name);
    void configure_set(const QueryConfig& cfg, const std::string& param_name);
    void configure_predicate(const QueryConfig& cfg, const std::string& param_name);

    /**
     * 预计算匹配结果
     * @param strings 输入字符串数组
     */
    void precompute(const std::vector<std::string>& strings);

    /**
     * 检查是否匹配
     */
    __attribute__((always_inline))
    bool test(size_t idx) const {
        return (bitmap_[idx >> 6] >> (idx & 63)) & 1;
    }

    /**
     * 获取匹配的索引列表
     */
    std::vector<uint32_t> get_matching_indices() const;

    /**
     * 获取匹配数量
     */
    size_t match_count() const { return match_count_; }

private:
    StringPredicate predicate_;
    std::vector<uint64_t> bitmap_;
    size_t count_ = 0;
    size_t match_count_ = 0;
};

// ============================================================================
// 自适应 Hash Join
// ============================================================================

/**
 * Join 策略
 */
enum class JoinStrategy {
    DIRECT_ARRAY,    // 直接数组索引 (小基数, 稠密键)
    COMPACT_HASH,    // 紧凑 Hash Table (中等基数)
    BLOOM_HASH       // Bloom Filter + Hash (高选择性过滤)
};

/**
 * 自适应 Hash Join
 *
 * 根据数据特征自动选择最优 Join 策略
 */
class AdaptiveHashJoin {
public:
    AdaptiveHashJoin() = default;

    /**
     * 从配置加载参数 (可选)
     */
    void configure(const QueryConfig& cfg);

    /**
     * Build 阶段 - 构建 Hash 表
     * @param keys Build 侧键
     * @param values Build 侧值
     * @param count 元素数量
     */
    void build(const int32_t* keys, const int32_t* values, size_t count);

    /**
     * Build 阶段 (仅键，值为索引)
     */
    void build_keys_only(const int32_t* keys, size_t count);

    /**
     * Probe 阶段
     * @param probe_keys Probe 侧键
     * @param probe_count Probe 侧数量
     * @param callback 匹配回调 (probe_idx, value)
     */
    template<typename Func>
    void probe(const int32_t* probe_keys, size_t probe_count, Func&& callback);

    /**
     * 查找单个键
     */
    const int32_t* find(int32_t key) const;

    /**
     * 批量查找
     */
    void batch_find(const int32_t* keys, const int32_t** results) const;

    /**
     * 获取选中的策略
     */
    JoinStrategy selected_strategy() const { return strategy_; }

private:
    JoinStrategy strategy_ = JoinStrategy::COMPACT_HASH;

    // 直接数组策略
    std::vector<int32_t> direct_array_;
    int32_t array_offset_ = 0;

    // 紧凑 Hash 策略
    CompactHashTable<int32_t> hash_table_;

    // Bloom Filter + Hash 策略
    SingleHashBloomFilter bloom_;
    bool use_bloom_ = false;

    void select_strategy(size_t build_count, int32_t min_key, int32_t max_key);
};

// ============================================================================
// 线程本地聚合器
// ============================================================================

/**
 * 线程本地聚合器
 *
 * 支持并行聚合，消除 atomic 竞争
 */
template<typename Value>
class GenericThreadLocalAggregator {
public:
    GenericThreadLocalAggregator() = default;

    /**
     * 初始化
     * @param thread_count 线程数 (0 = 自动)
     * @param estimated_groups 预估分组数
     */
    void init(size_t thread_count, size_t estimated_groups);

    /**
     * 从配置初始化
     */
    void init(const ExecutionConfig& cfg, size_t estimated_groups);

    /**
     * 获取线程本地表
     */
    CompactHashTable<Value>& get_thread_table(size_t thread_id);

    /**
     * 添加到线程本地表
     */
    void add(size_t thread_id, int32_t key, Value delta);

    /**
     * 合并所有线程结果
     */
    void merge(CompactHashTable<Value>& result);

    /**
     * 遍历合并后的结果
     */
    template<typename Func>
    void for_each_merged(Func&& callback);

    /**
     * 清空
     */
    void clear();

    size_t thread_count() const { return thread_tables_.size(); }

private:
    struct alignas(128) ThreadTable {
        CompactHashTable<Value> table;
    };
    std::vector<ThreadTable> thread_tables_;
    size_t estimated_groups_ = 0;
};

// ============================================================================
// 条件预计算器
// ============================================================================

/**
 * 条件组定义
 */
struct ConditionGroup {
    int id = 0;  // 条件组 ID (1-based, 0 表示无匹配)
    std::vector<StringPredicate> string_predicates;
    std::vector<std::pair<std::string, NumericRange<int32_t>>> int32_ranges;
    std::vector<std::pair<std::string, NumericRange<int64_t>>> int64_ranges;
};

/**
 * 条件预计算器
 *
 * 预计算每个元素匹配的条件组，用于复杂 OR 条件优化 (如 Q19)
 */
class PredicatePrecomputer {
public:
    PredicatePrecomputer() = default;

    /**
     * 从 QueryConfig 加载条件组
     * @param cfg 配置
     * @param num_groups 条件组数量
     * @param group_loader 加载条件组的回调
     */
    void configure(const QueryConfig& cfg, int num_groups,
                   std::function<ConditionGroup(const QueryConfig&, int)> group_loader);

    /**
     * 添加条件组
     */
    void add_condition_group(ConditionGroup group);

    /**
     * 预计算: 返回每个元素匹配的条件组 ID
     * @return 条件组 ID 数组 (0=无匹配, 1-n=条件组ID)
     */
    std::vector<uint8_t> precompute(
        const std::vector<std::string>* string_cols,  // 可选
        size_t string_col_count,
        const std::vector<int32_t>* int32_cols,       // 可选
        size_t int32_col_count,
        const std::vector<int64_t>* int64_cols,       // 可选
        size_t int64_col_count,
        size_t row_count
    );

    size_t group_count() const { return groups_.size(); }

private:
    std::vector<ConditionGroup> groups_;
};

// ============================================================================
// Layer 3: 执行引擎
// ============================================================================

/**
 * 任务调度器 (封装 ThreadPool)
 */
class TaskScheduler {
public:
    static TaskScheduler& instance();

    /**
     * 从配置初始化
     */
    void configure(const ExecutionConfig& cfg);

    /**
     * 并行 for 循环
     */
    template<typename Func>
    void parallel_for(size_t count, Func&& func);

    /**
     * 并行 for 循环 (带分块)
     */
    template<typename Func>
    void parallel_for_chunked(size_t count, size_t chunk_size, Func&& func);

    /**
     * 并行归约
     */
    template<typename T, typename Map, typename Reduce>
    T parallel_reduce(size_t count, T init, Map&& map, Reduce&& reduce);

    size_t thread_count() const { return thread_count_; }

private:
    TaskScheduler() = default;
    size_t thread_count_ = 8;
};

/**
 * 批量处理器
 *
 * 支持预取优化的批量处理
 */
class BatchProcessor {
public:
    BatchProcessor(size_t batch_size = 8, size_t prefetch_distance = 64)
        : batch_size_(batch_size), prefetch_dist_(prefetch_distance) {}

    /**
     * 从配置初始化
     */
    void configure(const ExecutionConfig& cfg) {
        batch_size_ = cfg.batch_size;
        prefetch_dist_ = cfg.prefetch_distance;
    }

    /**
     * 批量处理
     */
    template<typename Func>
    void process(size_t count, Func&& func);

    /**
     * 批量处理 (带预取回调)
     */
    template<typename Func, typename Prefetch>
    void process_with_prefetch(size_t count, Func&& func, Prefetch&& prefetch);

    size_t batch_size() const { return batch_size_; }

private:
    size_t batch_size_ = 8;
    size_t prefetch_dist_ = 64;
};

// ============================================================================
// Layer 4: 自动调优
// ============================================================================

/**
 * 数据统计
 */
struct DataStatistics {
    size_t count = 0;
    size_t distinct_count = 0;
    int64_t min_value = 0;
    int64_t max_value = 0;

    double density() const {
        if (max_value == min_value) return 1.0;
        return static_cast<double>(distinct_count) / (max_value - min_value + 1);
    }

    /**
     * 从数据计算统计信息
     */
    template<typename T>
    static DataStatistics compute(const T* data, size_t count);
};

/**
 * 硬件配置
 */
struct HardwareProfile {
    size_t num_cores = 8;
    size_t cache_line_size = 128;  // Apple Silicon
    size_t l1_cache_size = 192 * 1024;
    size_t l2_cache_size = 32 * 1024 * 1024;

    static HardwareProfile detect();
    static HardwareProfile apple_m4() {
        return {10, 128, 192 * 1024, 32 * 1024 * 1024};
    }
};

/**
 * 自动调优器
 */
class AutoTuner {
public:
    static AutoTuner& instance();

    /**
     * 推荐线程数
     */
    size_t recommend_thread_count(size_t data_size) const;

    /**
     * 推荐批量大小
     */
    size_t recommend_batch_size(size_t data_size) const;

    /**
     * 推荐 Join 策略
     */
    JoinStrategy recommend_join_strategy(size_t build_count, int64_t key_range) const;

    /**
     * 推荐 Hash 表容量
     */
    size_t recommend_hash_capacity(size_t expected_count, double target_load = 0.5) const;

    /**
     * 自动配置执行参数
     */
    void auto_configure(ExecutionConfig& cfg, size_t data_size);

private:
    AutoTuner();
    HardwareProfile hw_;
};

// ============================================================================
// V33 查询入口
// ============================================================================

/**
 * Q5 V33 版本
 * @param loader 数据加载器
 * @param config 查询配置 (默认使用工厂配置)
 */
void run_q5_v33(TPCHDataLoader& loader, const QueryConfig& config);
void run_q5_v33(TPCHDataLoader& loader);  // 使用默认配置

/**
 * Q7 V33 版本
 */
void run_q7_v33(TPCHDataLoader& loader, const QueryConfig& config);
void run_q7_v33(TPCHDataLoader& loader);

/**
 * Q9 V33 版本
 */
void run_q9_v33(TPCHDataLoader& loader, const QueryConfig& config);
void run_q9_v33(TPCHDataLoader& loader);

/**
 * Q18 V33 版本
 */
void run_q18_v33(TPCHDataLoader& loader, const QueryConfig& config);
void run_q18_v33(TPCHDataLoader& loader);

/**
 * Q19 V33 版本
 */
void run_q19_v33(TPCHDataLoader& loader, const QueryConfig& config);
void run_q19_v33(TPCHDataLoader& loader);

// ============================================================================
// 性能回归检测
// ============================================================================

namespace perf {
    // V32 基线 (几何平均加速比)
    constexpr double V32_GEOMETRIC_MEAN = 2.22;

    // 允许的性能波动
    constexpr double ALLOWED_VARIANCE = 0.05;  // 5%

    /**
     * 检查是否有性能回退
     */
    inline bool check_no_regression(double v33_mean) {
        return v33_mean >= V32_GEOMETRIC_MEAN * (1.0 - ALLOWED_VARIANCE);
    }
}

// ============================================================================
// Template 实现
// ============================================================================

template<typename Func>
void AdaptiveHashJoin::probe(const int32_t* probe_keys, size_t probe_count, Func&& callback) {
    switch (strategy_) {
        case JoinStrategy::DIRECT_ARRAY: {
            for (size_t i = 0; i < probe_count; ++i) {
                int32_t key = probe_keys[i];
                size_t idx = static_cast<size_t>(key - array_offset_);
                if (idx < direct_array_.size() && direct_array_[idx] != INT32_MIN) {
                    callback(i, direct_array_[idx]);
                }
            }
            break;
        }
        case JoinStrategy::COMPACT_HASH:
        case JoinStrategy::BLOOM_HASH: {
            if (use_bloom_) {
                // Bloom Filter 预过滤
                for (size_t i = 0; i < probe_count; ++i) {
                    int32_t key = probe_keys[i];
                    if (!bloom_.may_contain(key)) continue;
                    const int32_t* val = hash_table_.find(key);
                    if (val) callback(i, *val);
                }
            } else {
                for (size_t i = 0; i < probe_count; ++i) {
                    const int32_t* val = hash_table_.find(probe_keys[i]);
                    if (val) callback(i, *val);
                }
            }
            break;
        }
    }
}

template<typename Value>
void GenericThreadLocalAggregator<Value>::init(size_t thread_count, size_t estimated_groups) {
    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 8;
        thread_count = std::min<size_t>(thread_count, 8);
    }
    estimated_groups_ = estimated_groups;
    thread_tables_.resize(thread_count);
    for (auto& tt : thread_tables_) {
        tt.table.init(estimated_groups);
    }
}

template<typename Value>
void GenericThreadLocalAggregator<Value>::init(const ExecutionConfig& cfg, size_t estimated_groups) {
    init(cfg.get_thread_count(), estimated_groups);
}

template<typename Value>
CompactHashTable<Value>& GenericThreadLocalAggregator<Value>::get_thread_table(size_t thread_id) {
    return thread_tables_[thread_id].table;
}

template<typename Value>
void GenericThreadLocalAggregator<Value>::add(size_t thread_id, int32_t key, Value delta) {
    thread_tables_[thread_id].table.add_or_update(key, delta);
}

template<typename Value>
void GenericThreadLocalAggregator<Value>::merge(CompactHashTable<Value>& result) {
    for (auto& tt : thread_tables_) {
        tt.table.for_each([&](int32_t key, Value val) {
            result.add_or_update(key, val);
        });
    }
}

template<typename Value>
template<typename Func>
void GenericThreadLocalAggregator<Value>::for_each_merged(Func&& callback) {
    CompactHashTable<Value> merged;
    merged.init(estimated_groups_ * 2);
    merge(merged);
    merged.for_each(std::forward<Func>(callback));
}

template<typename Value>
void GenericThreadLocalAggregator<Value>::clear() {
    for (auto& tt : thread_tables_) {
        tt.table = CompactHashTable<Value>();
    }
}

template<typename Func>
void TaskScheduler::parallel_for(size_t count, Func&& func) {
    auto& pool = ThreadPool::instance();
    pool.prewarm(thread_count_, count / thread_count_);

    size_t chunk_size = (count + thread_count_ - 1) / thread_count_;
    std::vector<std::future<void>> futures;
    futures.reserve(thread_count_);

    for (size_t t = 0; t < thread_count_; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= count) break;

        futures.push_back(pool.submit([=, &func]() {
            for (size_t i = start; i < end; ++i) {
                func(i);
            }
        }));
    }

    for (auto& f : futures) f.get();
}

template<typename Func>
void TaskScheduler::parallel_for_chunked(size_t count, size_t chunk_size, Func&& func) {
    auto& pool = ThreadPool::instance();
    pool.prewarm(thread_count_, count / chunk_size);

    size_t per_thread = (count + thread_count_ - 1) / thread_count_;
    std::vector<std::future<void>> futures;
    futures.reserve(thread_count_);

    for (size_t t = 0; t < thread_count_; ++t) {
        size_t start = t * per_thread;
        size_t end = std::min(start + per_thread, count);
        if (start >= count) break;

        futures.push_back(pool.submit([=, &func]() {
            func(t, start, end);
        }));
    }

    for (auto& f : futures) f.get();
}

template<typename T, typename Map, typename Reduce>
T TaskScheduler::parallel_reduce(size_t count, T init, Map&& map, Reduce&& reduce) {
    auto& pool = ThreadPool::instance();
    return pool.parallel_reduce(count, init, std::forward<Map>(map), std::forward<Reduce>(reduce));
}

template<typename Func>
void BatchProcessor::process(size_t count, Func&& func) {
    size_t i = 0;
    for (; i + batch_size_ <= count; i += batch_size_) {
        for (size_t j = 0; j < batch_size_; ++j) {
            func(i + j);
        }
    }
    for (; i < count; ++i) {
        func(i);
    }
}

template<typename Func, typename Prefetch>
void BatchProcessor::process_with_prefetch(size_t count, Func&& func, Prefetch&& prefetch) {
    size_t i = 0;
    for (; i + batch_size_ <= count; i += batch_size_) {
        // 预取下一批
        if (i + prefetch_dist_ < count) {
            prefetch(i + prefetch_dist_);
        }
        for (size_t j = 0; j < batch_size_; ++j) {
            func(i + j);
        }
    }
    for (; i < count; ++i) {
        func(i);
    }
}

template<typename T>
DataStatistics DataStatistics::compute(const T* data, size_t count) {
    DataStatistics stats;
    stats.count = count;

    if (count == 0) return stats;

    std::unordered_set<T> unique;
    T min_val = data[0], max_val = data[0];

    for (size_t i = 0; i < count; ++i) {
        unique.insert(data[i]);
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    stats.distinct_count = unique.size();
    stats.min_value = static_cast<int64_t>(min_val);
    stats.max_value = static_cast<int64_t>(max_val);

    return stats;
}

} // namespace ops_v33
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V33_H
