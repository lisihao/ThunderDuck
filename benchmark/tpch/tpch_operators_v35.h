/**
 * ThunderDuck TPC-H V35 通用算子
 *
 * V35 架构: 通用化算子，多查询复用
 *
 * 新增组件:
 * - DirectArrayIndexBuilder: 自适应直接数组/Hash 索引
 * - SIMDStringProcessor: SIMD 字符串处理
 * - SemiAntiJoin: EXISTS/NOT EXISTS 支持
 * - ConditionalAggregator: CASE WHEN 条件聚合
 * - PipelineFusion: Filter→JOIN→Aggregate 管道融合
 *
 * @version 35.0
 * @date 2026-01-29
 */

#ifndef TPCH_OPERATORS_V35_H
#define TPCH_OPERATORS_V35_H

#include "tpch_config_v33.h"
#include "tpch_data_loader.h"
#include "tpch_operators_v33.h"
#include "tpch_operators_v32.h"
#include "tpch_operators_v25.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <future>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v35 {

using ::thunderduck::tpch::TPCHDataLoader;
using ops_v33::QueryConfig;
using ops_v33::ExecutionConfig;
using ops_v33::DateRange;
using ops_v33::AdaptiveHashJoin;
using ops_v33::GenericThreadLocalAggregator;
using ops_v32::CompactHashTable;
using ops_v32::SingleHashBloomFilter;
using ops_v32::BatchHasher;
using ops_v25::ThreadPool;

// ============================================================================
// 1. DirectArrayIndexBuilder - 自适应直接数组/Hash 索引
// ============================================================================

/**
 * 直接数组索引构建器
 *
 * 自动检测数据特征，选择最优索引策略:
 * - 直接数组: 键范围小且稠密 (density > 0.1, range < 10M)
 * - 紧凑 Hash: 键范围大或稀疏
 */
template<typename Value>
class DirectArrayIndexBuilder {
public:
    static constexpr Value INVALID_VALUE = Value{};

    DirectArrayIndexBuilder() = default;

    /**
     * 从数据构建索引
     */
    void build(const int32_t* keys, const Value* values, size_t count,
               Value default_value = Value{}) {
        if (count == 0) return;

        // 计算键范围
        int32_t min_key = keys[0], max_key = keys[0];
        for (size_t i = 1; i < count; ++i) {
            if (keys[i] < min_key) min_key = keys[i];
            if (keys[i] > max_key) max_key = keys[i];
        }

        default_value_ = default_value;
        auto_select_strategy(min_key, max_key, count);

        if (use_direct_array_) {
            min_key_ = min_key;
            size_t range = static_cast<size_t>(max_key - min_key + 1);
            direct_array_.resize(range, default_value);
            valid_.resize(range, false);

            for (size_t i = 0; i < count; ++i) {
                size_t idx = static_cast<size_t>(keys[i] - min_key);
                direct_array_[idx] = values[i];
                valid_[idx] = true;
            }
        } else {
            hash_table_.init(count);
            for (size_t i = 0; i < count; ++i) {
                hash_table_.insert(keys[i], values[i]);
            }
        }
    }

    /**
     * 仅构建存在性索引
     */
    void build_existence(const int32_t* keys, size_t count) {
        if (count == 0) return;

        int32_t min_key = keys[0], max_key = keys[0];
        for (size_t i = 1; i < count; ++i) {
            if (keys[i] < min_key) min_key = keys[i];
            if (keys[i] > max_key) max_key = keys[i];
        }

        auto_select_strategy(min_key, max_key, count);

        if (use_direct_array_) {
            min_key_ = min_key;
            size_t range = static_cast<size_t>(max_key - min_key + 1);
            valid_.resize(range, false);

            for (size_t i = 0; i < count; ++i) {
                size_t idx = static_cast<size_t>(keys[i] - min_key);
                valid_[idx] = true;
            }
        } else {
            existence_set_.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                existence_set_.insert(keys[i]);
            }
        }
    }

    /**
     * 查找
     */
    __attribute__((always_inline))
    const Value* find(int32_t key) const {
        if (use_direct_array_) {
            size_t idx = static_cast<size_t>(key - min_key_);
            if (idx < valid_.size() && valid_[idx]) {
                return &direct_array_[idx];
            }
            return nullptr;
        }
        return hash_table_.find(key);
    }

    /**
     * 获取值 (带默认值)
     */
    __attribute__((always_inline))
    Value get(int32_t key) const {
        if (use_direct_array_) {
            size_t idx = static_cast<size_t>(key - min_key_);
            if (idx < valid_.size() && valid_[idx]) {
                return direct_array_[idx];
            }
            return default_value_;
        }
        const Value* v = hash_table_.find(key);
        return v ? *v : default_value_;
    }

    /**
     * 检查存在性
     */
    __attribute__((always_inline))
    bool contains(int32_t key) const {
        if (use_direct_array_) {
            size_t idx = static_cast<size_t>(key - min_key_);
            return idx < valid_.size() && valid_[idx];
        }
        if (!existence_set_.empty()) {
            return existence_set_.count(key) > 0;
        }
        return hash_table_.find(key) != nullptr;
    }

    bool is_direct_array() const { return use_direct_array_; }

    size_t memory_usage() const {
        if (use_direct_array_) {
            return direct_array_.size() * sizeof(Value) + valid_.size() / 8;
        }
        return hash_table_.capacity() * (sizeof(int32_t) + sizeof(Value));
    }

private:
    bool use_direct_array_ = false;
    int32_t min_key_ = 0;
    Value default_value_{};

    std::vector<Value> direct_array_;
    std::vector<bool> valid_;
    CompactHashTable<Value> hash_table_;
    std::unordered_set<int32_t> existence_set_;

    void auto_select_strategy(int32_t min_key, int32_t max_key, size_t count) {
        int64_t range = static_cast<int64_t>(max_key) - min_key + 1;
        double density = static_cast<double>(count) / range;

        // 直接数组条件: 范围 < 10M 且 (密度 > 0.1 或 范围 < 100K)
        use_direct_array_ = (range < 10'000'000) && (density > 0.1 || range < 100'000);
    }
};

// ============================================================================
// 2. SIMDStringProcessor - SIMD 字符串处理
// ============================================================================

/**
 * SIMD 字符串处理器
 */
class SIMDStringProcessor {
public:
    // ========== SUBSTRING 操作 ==========

    /**
     * 批量提取固定长度前缀并转换为整数
     * 用途: Q22 电话号码前2位 → 国家码
     */
    static void prefix_to_int_batch(
        const std::vector<std::string>& strings,
        size_t prefix_len,
        std::vector<int16_t>& results) {

        results.resize(strings.size());

        for (size_t i = 0; i < strings.size(); ++i) {
            if (strings[i].size() >= prefix_len) {
                int16_t val = 0;
                for (size_t j = 0; j < prefix_len; ++j) {
                    char c = strings[i][j];
                    if (c >= '0' && c <= '9') {
                        val = val * 10 + (c - '0');
                    } else {
                        val = -1;
                        break;
                    }
                }
                results[i] = val;
            } else {
                results[i] = -1;
            }
        }
    }

    // ========== 模式匹配 ==========

    /**
     * 批量 LIKE 模式匹配 (%pattern%)
     * 用途: Q9 p_name LIKE '%green%'
     */
    static void like_contains_batch(
        const std::vector<std::string>& strings,
        const std::string& pattern,
        std::vector<bool>& results) {

        results.resize(strings.size());
        const char* pat = pattern.c_str();
        size_t pat_len = pattern.size();

        for (size_t i = 0; i < strings.size(); ++i) {
            const char* str = strings[i].c_str();
            size_t str_len = strings[i].size();

            if (str_len < pat_len) {
                results[i] = false;
            } else {
                results[i] = (memmem(str, str_len, pat, pat_len) != nullptr);
            }
        }
    }

    /**
     * 批量双模式匹配 (%pattern1%pattern2%)
     * 用途: Q13 '%special%requests%'
     */
    static void like_two_patterns_batch(
        const std::vector<std::string>& strings,
        const std::string& pattern1,
        const std::string& pattern2,
        std::vector<bool>& results) {

        results.resize(strings.size());
        const char* pat1 = pattern1.c_str();
        size_t pat1_len = pattern1.size();
        const char* pat2 = pattern2.c_str();
        size_t pat2_len = pattern2.size();

        for (size_t i = 0; i < strings.size(); ++i) {
            const char* str = strings[i].c_str();
            size_t str_len = strings[i].size();

            const void* pos1 = memmem(str, str_len, pat1, pat1_len);
            if (pos1 == nullptr) {
                results[i] = false;
            } else {
                size_t offset = static_cast<const char*>(pos1) - str + pat1_len;
                if (offset >= str_len) {
                    results[i] = false;
                } else {
                    results[i] = (memmem(str + offset, str_len - offset, pat2, pat2_len) != nullptr);
                }
            }
        }
    }

    // ========== 前缀匹配 ==========

    /**
     * 批量前缀匹配
     * 用途: Q14 p_type LIKE 'PROMO%'
     */
    static void starts_with_batch(
        const std::vector<std::string>& strings,
        const std::string& prefix,
        std::vector<bool>& results) {

        results.resize(strings.size());
        size_t prefix_len = prefix.size();

        for (size_t i = 0; i < strings.size(); ++i) {
            if (strings[i].size() >= prefix_len) {
                results[i] = (memcmp(strings[i].c_str(), prefix.c_str(), prefix_len) == 0);
            } else {
                results[i] = false;
            }
        }
    }

    // ========== 集合成员检测 ==========

    /**
     * 构建字符串集合的快速查找结构
     */
    static void build_string_set(
        const std::vector<std::string>& target_set,
        std::unordered_set<std::string>& hash_set) {

        hash_set.clear();
        hash_set.reserve(target_set.size());
        for (const auto& s : target_set) {
            hash_set.insert(s);
        }
    }

    /**
     * 批量字符串集合检测
     * 用途: Q7 n_name IN ('FRANCE', 'GERMANY')
     */
    static void in_set_batch(
        const std::vector<std::string>& strings,
        const std::unordered_set<std::string>& target_set,
        std::vector<bool>& results) {

        results.resize(strings.size());
        for (size_t i = 0; i < strings.size(); ++i) {
            results[i] = (target_set.count(strings[i]) > 0);
        }
    }

    /**
     * 预计算字符串到索引的映射
     * 返回每个字符串在 target_set 中的索引 (-1 表示不在集合中)
     */
    static void string_to_index_batch(
        const std::vector<std::string>& strings,
        const std::vector<std::string>& target_list,
        std::vector<int8_t>& results) {

        std::unordered_map<std::string, int8_t> str_to_idx;
        for (size_t i = 0; i < target_list.size(); ++i) {
            str_to_idx[target_list[i]] = static_cast<int8_t>(i);
        }

        results.resize(strings.size());
        for (size_t i = 0; i < strings.size(); ++i) {
            auto it = str_to_idx.find(strings[i]);
            results[i] = (it != str_to_idx.end()) ? it->second : -1;
        }
    }
};

// ============================================================================
// 3. SemiAntiJoin - EXISTS/NOT EXISTS 支持
// ============================================================================

/**
 * SEMI/ANTI JOIN 算子
 */
class SemiAntiJoin {
public:
    SemiAntiJoin() = default;

    /**
     * 构建右侧
     */
    void build(const int32_t* keys, size_t count) {
        auto_select_strategy(count);

        if (use_bloom_) {
            bloom_.init(count, 2);
            for (size_t i = 0; i < count; ++i) {
                bloom_.insert(keys[i]);
            }
        }

        key_set_.clear();
        key_set_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            key_set_.insert(keys[i]);
        }
    }

    /**
     * 带过滤的构建
     */
    void build_filtered(const int32_t* keys, size_t count,
                        const std::function<bool(size_t)>& filter) {
        std::vector<int32_t> filtered_keys;
        filtered_keys.reserve(count / 2);

        for (size_t i = 0; i < count; ++i) {
            if (filter(i)) {
                filtered_keys.push_back(keys[i]);
            }
        }

        build(filtered_keys.data(), filtered_keys.size());
    }

    /**
     * SEMI JOIN: 返回存在匹配的左侧索引
     */
    std::vector<uint32_t> semi_join(const int32_t* probe_keys, size_t probe_count) {
        std::vector<uint32_t> result;
        result.reserve(probe_count / 4);

        if (use_bloom_) {
            for (size_t i = 0; i < probe_count; ++i) {
                if (bloom_.may_contain(probe_keys[i])) {
                    if (key_set_.count(probe_keys[i]) > 0) {
                        result.push_back(static_cast<uint32_t>(i));
                    }
                }
            }
        } else {
            for (size_t i = 0; i < probe_count; ++i) {
                if (key_set_.count(probe_keys[i]) > 0) {
                    result.push_back(static_cast<uint32_t>(i));
                }
            }
        }

        return result;
    }

    /**
     * ANTI JOIN: 返回不存在匹配的左侧索引
     */
    std::vector<uint32_t> anti_join(const int32_t* probe_keys, size_t probe_count) {
        std::vector<uint32_t> result;
        result.reserve(probe_count / 2);

        if (use_bloom_) {
            for (size_t i = 0; i < probe_count; ++i) {
                if (!bloom_.may_contain(probe_keys[i])) {
                    // Bloom Filter 说不存在，一定不存在
                    result.push_back(static_cast<uint32_t>(i));
                } else if (key_set_.count(probe_keys[i]) == 0) {
                    // Bloom Filter 误判，精确检查不存在
                    result.push_back(static_cast<uint32_t>(i));
                }
            }
        } else {
            for (size_t i = 0; i < probe_count; ++i) {
                if (key_set_.count(probe_keys[i]) == 0) {
                    result.push_back(static_cast<uint32_t>(i));
                }
            }
        }

        return result;
    }

    /**
     * 检查单个键是否存在
     */
    __attribute__((always_inline))
    bool exists(int32_t key) const {
        if (use_bloom_ && !bloom_.may_contain(key)) {
            return false;
        }
        return key_set_.count(key) > 0;
    }

    /**
     * 检查单个键是否不存在
     */
    __attribute__((always_inline))
    bool not_exists(int32_t key) const {
        return !exists(key);
    }

    size_t build_size() const { return key_set_.size(); }

private:
    SingleHashBloomFilter bloom_;
    bool use_bloom_ = false;
    std::unordered_set<int32_t> key_set_;

    void auto_select_strategy(size_t build_count) {
        // 大于 10K 时使用 Bloom Filter 预过滤
        use_bloom_ = (build_count > 10'000);
    }
};

// ============================================================================
// 4. ConditionalAggregator - CASE WHEN 条件聚合
// ============================================================================

/**
 * 条件聚合结果
 */
template<typename Value>
struct ConditionalAggResult {
    Value value{};
    size_t count = 0;
};

/**
 * 条件聚合器
 * 支持 CASE WHEN condition THEN value ELSE default END 的聚合
 */
template<typename Key, typename Value>
class ConditionalAggregator {
public:
    ConditionalAggregator() = default;

    /**
     * 初始化分支数量
     */
    void init(size_t num_branches, size_t estimated_keys) {
        num_branches_ = num_branches;
        branch_aggs_.resize(num_branches);
        for (auto& agg : branch_aggs_) {
            agg.init(8, estimated_keys);
        }
    }

    /**
     * 添加到指定分支
     * @param branch_idx 分支索引 (0 到 num_branches-1)
     * @param key 聚合键
     * @param value 聚合值
     */
    void add(size_t thread_id, size_t branch_idx, Key key, Value value) {
        if (branch_idx < num_branches_) {
            branch_aggs_[branch_idx].add(thread_id, key, value);
        }
    }

    /**
     * 合并线程结果并遍历
     */
    template<typename Func>
    void for_each_branch(size_t branch_idx, Func&& callback) {
        if (branch_idx < num_branches_) {
            branch_aggs_[branch_idx].for_each_merged(std::forward<Func>(callback));
        }
    }

    /**
     * 获取所有分支的合并结果
     * callback(branch_idx, key, value)
     */
    template<typename Func>
    void for_each_all(Func&& callback) {
        for (size_t b = 0; b < num_branches_; ++b) {
            branch_aggs_[b].for_each_merged([&](Key key, Value val) {
                callback(b, key, val);
            });
        }
    }

    size_t num_branches() const { return num_branches_; }

private:
    size_t num_branches_ = 0;
    std::vector<GenericThreadLocalAggregator<Value>> branch_aggs_;
};

// ============================================================================
// 5. PipelineFusion - Filter→JOIN→Aggregate 管道融合
// ============================================================================

/**
 * JOIN 上下文
 */
struct JoinContext {
    std::vector<int32_t> values;

    int32_t get(size_t idx) const {
        return idx < values.size() ? values[idx] : -1;
    }

    void set(size_t idx, int32_t val) {
        if (idx >= values.size()) values.resize(idx + 1, -1);
        values[idx] = val;
    }
};

/**
 * 管道融合框架
 */
template<typename AggValue>
class PipelineFusion {
public:
    PipelineFusion() = default;

    /**
     * 添加 Hash Join 阶段 (仅检查存在性)
     */
    void add_existence_join(const std::string& name, const int32_t* build_keys, size_t build_count) {
        JoinStage stage;
        stage.name = name;
        stage.existence_only = true;
        stage.existence_index.build_existence(build_keys, build_count);
        join_stages_.push_back(std::move(stage));
    }

    /**
     * 添加 Hash Join 阶段 (带值)
     */
    void add_value_join(const std::string& name,
                        const int32_t* build_keys,
                        const int32_t* build_values,
                        size_t build_count) {
        JoinStage stage;
        stage.name = name;
        stage.existence_only = false;
        stage.value_index.build(build_keys, build_values, build_count, -1);
        join_stages_.push_back(std::move(stage));
    }

    /**
     * 执行融合管道
     * @param probe_keys 第一阶段的探测键
     * @param probe_count 探测数量
     * @param key_extractor 提取聚合键的函数 (row_idx, join_context) -> key
     * @param value_extractor 提取聚合值的函数 (row_idx, join_context) -> value
     * @param thread_count 线程数
     */
    template<typename KeyExtractor, typename ValueExtractor>
    void execute(const int32_t* probe_keys,
                 size_t probe_count,
                 KeyExtractor&& key_extractor,
                 ValueExtractor&& value_extractor,
                 size_t thread_count = 0) {

        if (thread_count == 0) {
            thread_count = std::min<size_t>(8, std::thread::hardware_concurrency());
        }

        aggregator_.init(thread_count, 1000);

        auto& pool = ThreadPool::instance();
        pool.prewarm(thread_count, probe_count / thread_count);

        size_t chunk_size = (probe_count + thread_count - 1) / thread_count;
        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < thread_count; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, probe_count);
            if (start >= probe_count) break;

            futures.push_back(pool.submit([this, t, start, end, &probe_keys, &key_extractor, &value_extractor]() {
                JoinContext ctx;
                ctx.values.resize(join_stages_.size(), -1);

                for (size_t i = start; i < end; ++i) {
                    int32_t probe_key = probe_keys[i];
                    bool all_match = true;

                    // 执行所有 JOIN 阶段
                    for (size_t s = 0; s < join_stages_.size(); ++s) {
                        const auto& stage = join_stages_[s];

                        int32_t current_key = (s == 0) ? probe_key : ctx.values[s - 1];
                        if (current_key < 0) {
                            all_match = false;
                            break;
                        }

                        if (stage.existence_only) {
                            if (!stage.existence_index.contains(current_key)) {
                                all_match = false;
                                break;
                            }
                            ctx.values[s] = current_key;  // 保持 key
                        } else {
                            int32_t val = stage.value_index.get(current_key);
                            if (val < 0) {
                                all_match = false;
                                break;
                            }
                            ctx.values[s] = val;
                        }
                    }

                    if (all_match) {
                        int32_t agg_key = key_extractor(i, ctx);
                        AggValue agg_val = value_extractor(i, ctx);
                        aggregator_.add(t, agg_key, agg_val);
                    }
                }
            }));
        }

        for (auto& f : futures) f.get();
    }

    /**
     * 遍历聚合结果
     */
    template<typename Func>
    void for_each_result(Func&& callback) {
        aggregator_.for_each_merged(std::forward<Func>(callback));
    }

private:
    struct JoinStage {
        std::string name;
        bool existence_only = false;
        DirectArrayIndexBuilder<int32_t> value_index;
        DirectArrayIndexBuilder<bool> existence_index;
    };

    std::vector<JoinStage> join_stages_;
    GenericThreadLocalAggregator<AggValue> aggregator_;
};

// ============================================================================
// V35 查询入口声明
// ============================================================================

// 使用 V35 通用算子优化的查询
void run_q3_v35(TPCHDataLoader& loader);
void run_q8_v35(TPCHDataLoader& loader);
void run_q14_v35(TPCHDataLoader& loader);
void run_q22_v35(TPCHDataLoader& loader);
void run_q21_v35(TPCHDataLoader& loader);

// 可选: 其他查询的 V35 版本
void run_q5_v35(TPCHDataLoader& loader);
void run_q7_v35(TPCHDataLoader& loader);
void run_q9_v35(TPCHDataLoader& loader);
void run_q13_v35(TPCHDataLoader& loader);

} // namespace ops_v35
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V35_H
