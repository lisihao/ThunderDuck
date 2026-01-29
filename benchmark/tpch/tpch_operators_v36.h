/**
 * ThunderDuck TPC-H V36 优化算子
 *
 * 核心优化: 相关子查询解关联 (Correlated Subquery Decorrelation)
 *
 * 适用查询:
 * - Q17: l_quantity < 0.2 * AVG(l_quantity) WHERE l_partkey = p_partkey
 * - Q20: ps_availqty > 0.5 * SUM(l_quantity) WHERE l_partkey = ps_partkey
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <optional>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v36 {

// ============================================================================
// 聚合类型和比较运算符
// ============================================================================

enum class AggregateType {
    AVG,
    SUM,
    COUNT,
    MIN,
    MAX
};

enum class CompareOp {
    LT,     // <
    LE,     // <=
    GT,     // >
    GE,     // >=
    EQ,     // =
    NE      // <>
};

// ============================================================================
// 紧凑 Hash Table (开放寻址)
// ============================================================================

template<typename Key, typename Value>
class CompactHashTable {
public:
    static constexpr size_t EMPTY_KEY = static_cast<size_t>(-1);

    CompactHashTable() = default;

    void reserve(size_t capacity) {
        size_t new_capacity = next_power_of_2(capacity * 2);
        keys_.resize(new_capacity, static_cast<Key>(EMPTY_KEY));
        values_.resize(new_capacity);
        mask_ = new_capacity - 1;
    }

    Value& operator[](Key key) {
        if (keys_.empty()) reserve(1024);

        size_t idx = hash(key) & mask_;
        while (keys_[idx] != static_cast<Key>(EMPTY_KEY) && keys_[idx] != key) {
            idx = (idx + 1) & mask_;
        }
        if (keys_[idx] == static_cast<Key>(EMPTY_KEY)) {
            keys_[idx] = key;
            size_++;
            // 扩容检查
            if (size_ > keys_.size() * 3 / 4) {
                rehash();
                return (*this)[key];
            }
        }
        return values_[idx];
    }

    const Value* find(Key key) const {
        if (keys_.empty()) return nullptr;

        size_t idx = hash(key) & mask_;
        while (keys_[idx] != static_cast<Key>(EMPTY_KEY)) {
            if (keys_[idx] == key) {
                return &values_[idx];
            }
            idx = (idx + 1) & mask_;
        }
        return nullptr;
    }

    bool contains(Key key) const {
        return find(key) != nullptr;
    }

    size_t size() const { return size_; }

    // 迭代器支持
    template<typename Func>
    void for_each(Func&& func) const {
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] != static_cast<Key>(EMPTY_KEY)) {
                func(keys_[i], values_[i]);
            }
        }
    }

private:
    std::vector<Key> keys_;
    std::vector<Value> values_;
    size_t mask_ = 0;
    size_t size_ = 0;

    static size_t hash(Key key) {
        size_t h = static_cast<size_t>(key);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    static size_t next_power_of_2(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    void rehash() {
        auto old_keys = std::move(keys_);
        auto old_values = std::move(values_);

        size_t new_capacity = old_keys.size() * 2;
        keys_.resize(new_capacity, static_cast<Key>(EMPTY_KEY));
        values_.resize(new_capacity);
        mask_ = new_capacity - 1;
        size_ = 0;

        for (size_t i = 0; i < old_keys.size(); ++i) {
            if (old_keys[i] != static_cast<Key>(EMPTY_KEY)) {
                (*this)[old_keys[i]] = old_values[i];
            }
        }
    }
};

// ============================================================================
// 预计算聚合结果
// ============================================================================

template<typename KeyType, typename ValueType>
class PrecomputedAggregates {
public:
    // AVG 需要的状态
    struct AvgState {
        ValueType sum = 0;
        int64_t count = 0;

        ValueType avg() const {
            return count > 0 ? sum / count : 0;
        }
    };

    // 构建预计算结果
    void build(const KeyType* keys,
               const ValueType* values,
               size_t count,
               AggregateType agg_type,
               double scale_factor = 1.0) {
        agg_type_ = agg_type;
        scale_factor_ = scale_factor;

        switch (agg_type) {
            case AggregateType::AVG:
                build_avg(keys, values, count);
                break;
            case AggregateType::SUM:
                build_sum(keys, values, count);
                break;
            case AggregateType::COUNT:
                build_count(keys, count);
                break;
            case AggregateType::MIN:
                build_min(keys, values, count);
                break;
            case AggregateType::MAX:
                build_max(keys, values, count);
                break;
        }
    }

    // 查询单个 key 的结果
    std::optional<ValueType> get(KeyType key) const {
        const auto* val = table_.find(key);
        if (val) {
            return *val;
        }
        return std::nullopt;
    }

    // 批量查询
    void batch_get(const KeyType* keys,
                   size_t count,
                   ValueType* results,
                   ValueType default_value = ValueType{}) const {
        for (size_t i = 0; i < count; ++i) {
            const auto* val = table_.find(keys[i]);
            results[i] = val ? *val : default_value;
        }
    }

    const CompactHashTable<KeyType, ValueType>& get_table() const {
        return table_;
    }

    size_t size() const { return table_.size(); }

private:
    CompactHashTable<KeyType, ValueType> table_;
    CompactHashTable<KeyType, AvgState> avg_states_;
    AggregateType agg_type_;
    double scale_factor_;

    void build_avg(const KeyType* keys, const ValueType* values, size_t count) {
        avg_states_.reserve(count / 10);

        // 累加
        for (size_t i = 0; i < count; ++i) {
            auto& state = avg_states_[keys[i]];
            state.sum += values[i];
            state.count++;
        }

        // 计算最终 AVG 并应用 scale_factor
        table_.reserve(avg_states_.size());
        avg_states_.for_each([this](KeyType key, const AvgState& state) {
            if (state.count > 0) {
                ValueType avg = state.sum / state.count;
                table_[key] = static_cast<ValueType>(avg * scale_factor_);
            }
        });
    }

    void build_sum(const KeyType* keys, const ValueType* values, size_t count) {
        table_.reserve(count / 10);

        for (size_t i = 0; i < count; ++i) {
            table_[keys[i]] += values[i];
        }

        if (scale_factor_ != 1.0) {
            // 需要重新遍历应用 scale_factor
            CompactHashTable<KeyType, ValueType> scaled;
            scaled.reserve(table_.size());
            table_.for_each([this, &scaled](KeyType key, ValueType val) {
                scaled[key] = static_cast<ValueType>(val * scale_factor_);
            });
            table_ = std::move(scaled);
        }
    }

    void build_count(const KeyType* keys, size_t count) {
        CompactHashTable<KeyType, int64_t> count_table;
        count_table.reserve(count / 10);

        for (size_t i = 0; i < count; ++i) {
            count_table[keys[i]]++;
        }

        table_.reserve(count_table.size());
        count_table.for_each([this](KeyType key, int64_t cnt) {
            table_[key] = static_cast<ValueType>(cnt * scale_factor_);
        });
    }

    void build_min(const KeyType* keys, const ValueType* values, size_t count) {
        table_.reserve(count / 10);

        for (size_t i = 0; i < count; ++i) {
            const auto* existing = table_.find(keys[i]);
            if (!existing || values[i] < *existing) {
                table_[keys[i]] = values[i];
            }
        }

        if (scale_factor_ != 1.0) {
            CompactHashTable<KeyType, ValueType> scaled;
            scaled.reserve(table_.size());
            table_.for_each([this, &scaled](KeyType key, ValueType val) {
                scaled[key] = static_cast<ValueType>(val * scale_factor_);
            });
            table_ = std::move(scaled);
        }
    }

    void build_max(const KeyType* keys, const ValueType* values, size_t count) {
        table_.reserve(count / 10);

        for (size_t i = 0; i < count; ++i) {
            const auto* existing = table_.find(keys[i]);
            if (!existing || values[i] > *existing) {
                table_[keys[i]] = values[i];
            }
        }

        if (scale_factor_ != 1.0) {
            CompactHashTable<KeyType, ValueType> scaled;
            scaled.reserve(table_.size());
            table_.for_each([this, &scaled](KeyType key, ValueType val) {
                scaled[key] = static_cast<ValueType>(val * scale_factor_);
            });
            table_ = std::move(scaled);
        }
    }
};

// ============================================================================
// 相关子查询转换器
// ============================================================================

template<typename KeyType, typename ValueType>
class CorrelatedSubqueryTransformer {
public:
    struct Config {
        AggregateType agg_type;
        CompareOp compare_op;
        double scale_factor;
        bool allow_null_mismatch = false;
    };

    explicit CorrelatedSubqueryTransformer(const Config& config)
        : config_(config) {}

    // Step 1: 预计算子查询结果
    void precompute(const KeyType* keys,
                    const ValueType* values,
                    size_t count) {
        precomputed_.build(keys, values, count, config_.agg_type, config_.scale_factor);
    }

    // Step 2: 应用到外层查询，返回满足条件的行索引
    std::vector<uint32_t> apply(const KeyType* outer_keys,
                                const ValueType* outer_values,
                                size_t outer_count) {
        std::vector<uint32_t> result;
        result.reserve(outer_count / 10);

        for (size_t i = 0; i < outer_count; ++i) {
            auto threshold = precomputed_.get(outer_keys[i]);
            if (threshold.has_value()) {
                if (compare(outer_values[i], threshold.value())) {
                    result.push_back(static_cast<uint32_t>(i));
                }
            } else if (config_.allow_null_mismatch) {
                // 子查询返回 NULL 时的处理
            }
        }

        return result;
    }

    // 获取指定 key 的预计算结果
    std::optional<ValueType> get_threshold(KeyType key) const {
        return precomputed_.get(key);
    }

    const PrecomputedAggregates<KeyType, ValueType>& get_precomputed() const {
        return precomputed_;
    }

private:
    Config config_;
    PrecomputedAggregates<KeyType, ValueType> precomputed_;

    bool compare(ValueType outer_val, ValueType threshold) const {
        switch (config_.compare_op) {
            case CompareOp::LT: return outer_val < threshold;
            case CompareOp::LE: return outer_val <= threshold;
            case CompareOp::GT: return outer_val > threshold;
            case CompareOp::GE: return outer_val >= threshold;
            case CompareOp::EQ: return outer_val == threshold;
            case CompareOp::NE: return outer_val != threshold;
        }
        return false;
    }
};

// ============================================================================
// Q17 优化器 - 并行版本
// ============================================================================

class Q17Optimizer {
public:
    struct Result {
        int64_t sum_extendedprice;
        double avg_yearly;
    };

    // 并行执行 Q17
    static Result execute(
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_brand,
        const std::vector<std::string>& p_container,
        size_t part_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int64_t* l_quantity,
        const int64_t* l_extendedprice,
        size_t lineitem_count,
        // 参数
        const std::string& target_brand,
        const std::string& target_container,
        double quantity_factor,
        size_t num_threads = 8
    );

private:
    // AVG 计算的状态
    struct QuantityState {
        int64_t sum = 0;
        int64_t count = 0;

        int64_t avg() const {
            return count > 0 ? sum / count : 0;
        }
    };

    // 待评估的行
    struct PendingRow {
        int64_t quantity;
        int64_t extendedprice;
    };

    // 线程局部状态
    struct alignas(128) ThreadLocalState {
        CompactHashTable<int32_t, QuantityState> qty_states;
        CompactHashTable<int32_t, std::vector<PendingRow>> pending_rows;
    };
};

// ============================================================================
// Q20 优化器 (复合键版本)
// ============================================================================

class Q20Optimizer {
public:
    struct Result {
        std::vector<std::pair<std::string, std::string>> suppliers;  // (s_name, s_address)
    };

    // 执行 Q20
    static Result execute(
        // Supplier 表
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        const std::vector<std::string>& s_address,
        size_t supplier_count,
        // Nation 表
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_name,
        size_t part_count,
        // PartSupp 表
        const int32_t* ps_partkey,
        const int32_t* ps_suppkey,
        const int32_t* ps_availqty,
        size_t partsupp_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int32_t* l_suppkey,
        const int64_t* l_quantity,
        const int32_t* l_shipdate,
        size_t lineitem_count,
        // 参数
        const std::string& part_prefix,      // "forest"
        const std::string& target_nation,    // "CANADA"
        int32_t date_lo,                     // 1994-01-01
        int32_t date_hi,                     // 1995-01-01
        double quantity_factor               // 0.5
    );

private:
    // 复合键: (partkey, suppkey)
    struct CompositeKey {
        int32_t partkey;
        int32_t suppkey;

        bool operator==(const CompositeKey& other) const {
            return partkey == other.partkey && suppkey == other.suppkey;
        }
    };

    struct CompositeKeyHash {
        size_t operator()(const CompositeKey& key) const {
            return std::hash<int64_t>()(
                (static_cast<int64_t>(key.partkey) << 32) |
                static_cast<uint32_t>(key.suppkey)
            );
        }
    };
};

// ============================================================================
// V36 查询入口
// ============================================================================

void run_q17_v36(TPCHDataLoader& loader);
void run_q20_v36(TPCHDataLoader& loader);

} // namespace ops_v36
} // namespace tpch
} // namespace thunderduck
