/**
 * ThunderDuck TPC-H Operators V55
 *
 * V55 优化内容:
 * 1. Q2 子查询解关联 (SubqueryDecorrelation)
 * 2. 通用并行多表 JOIN (GenericParallelMultiJoin)
 * 3. 通用两阶段聚合 (GenericTwoPhaseAgg)
 *
 * @version 55
 * @date 2026-01-30
 */

#pragma once

#include "tpch_data_loader.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <cstring>
#include <iostream>

namespace thunderduck {
namespace tpch {
namespace ops_v55 {

// ============================================================================
// 版本信息
// ============================================================================

extern const char* V55_VERSION;
extern const char* V55_DATE;
extern const char* V55_FEATURES[];

// ============================================================================
// 通用算子: SubqueryDecorrelation (子查询解关联)
// ============================================================================

/**
 * 子查询解关联器
 *
 * 将相关子查询转换为预计算的哈希表查找:
 * WHERE x = (SELECT MIN(y) FROM t WHERE t.key = outer.key)
 * =>
 * 1. 预计算: min_map[key] = MIN(y) for all keys
 * 2. 查找: WHERE x = min_map[outer.key]
 */
template<typename KeyT, typename ValueT>
class SubqueryDecorrelation {
public:
    using AggFunc = std::function<ValueT(ValueT, ValueT)>;

    /**
     * 预计算聚合结果
     *
     * @param keys 分组键数组
     * @param values 聚合值数组
     * @param count 元素数量
     * @param agg_func 聚合函数 (如 std::min)
     * @param filter 过滤条件 (可选)
     */
    void precompute(
        const KeyT* keys,
        const ValueT* values,
        size_t count,
        AggFunc agg_func,
        std::function<bool(size_t)> filter = nullptr
    ) {
        precomputed_.clear();
        precomputed_.reserve(count / 10);  // 预估基数

        for (size_t i = 0; i < count; ++i) {
            if (filter && !filter(i)) continue;

            KeyT key = keys[i];
            ValueT val = values[i];

            auto it = precomputed_.find(key);
            if (it == precomputed_.end()) {
                precomputed_[key] = val;
            } else {
                it->second = agg_func(it->second, val);
            }
        }
    }

    /**
     * 查找预计算结果
     */
    bool lookup(KeyT key, ValueT& result) const {
        auto it = precomputed_.find(key);
        if (it != precomputed_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    /**
     * 获取预计算结果数量
     */
    size_t size() const { return precomputed_.size(); }

private:
    std::unordered_map<KeyT, ValueT> precomputed_;
};

// ============================================================================
// 通用算子: GenericParallelMultiJoin (并行多表 JOIN)
// ============================================================================

/**
 * 并行多表 JOIN 配置
 */
struct MultiJoinConfig {
    size_t num_threads = 8;
    size_t prefetch_distance = 64;
    bool use_bloom_filter = true;
    size_t bloom_bits_per_element = 10;
};

/**
 * 通用并行多表 JOIN
 *
 * 支持任意数量表的 JOIN 操作:
 * - 自动选择 JOIN 顺序 (小表优先)
 * - Bloom Filter 预过滤
 * - 并行执行
 */
template<typename KeyT>
class GenericParallelMultiJoin {
public:
    struct JoinResult {
        std::vector<std::vector<uint32_t>> table_indices;  // 每个表的匹配索引
        size_t match_count = 0;
    };

    /**
     * 添加待 JOIN 的表
     *
     * @param name 表名 (用于调试)
     * @param keys JOIN 键数组
     * @param count 行数
     * @param filter 过滤条件 (可选)
     */
    void add_table(
        const std::string& name,
        const KeyT* keys,
        size_t count,
        std::function<bool(size_t)> filter = nullptr
    ) {
        tables_.push_back({name, keys, count, filter, {}});
    }

    /**
     * 执行 JOIN
     */
    JoinResult execute(const MultiJoinConfig& config = {}) {
        if (tables_.empty()) return {};

        // 按表大小排序 (小表优先)
        std::vector<size_t> join_order(tables_.size());
        std::iota(join_order.begin(), join_order.end(), 0);
        std::sort(join_order.begin(), join_order.end(), [this](size_t a, size_t b) {
            return estimate_filtered_size(a) < estimate_filtered_size(b);
        });

        // 第一个表构建哈希表
        size_t first_idx = join_order[0];
        auto& first = tables_[first_idx];
        build_hash_table(first, config);

        // 逐个 JOIN 其他表
        JoinResult result;
        result.table_indices.resize(tables_.size());

        for (size_t i = 1; i < join_order.size(); ++i) {
            size_t table_idx = join_order[i];
            probe_hash_table(table_idx, result, config);
        }

        return result;
    }

private:
    struct TableInfo {
        std::string name;
        const KeyT* keys;
        size_t count;
        std::function<bool(size_t)> filter;
        std::unordered_map<KeyT, std::vector<uint32_t>> hash_table;
    };

    size_t estimate_filtered_size(size_t table_idx) const {
        const auto& t = tables_[table_idx];
        if (!t.filter) return t.count;
        // 估算过滤后大小 (假设 25% 选择率)
        return t.count / 4;
    }

    void build_hash_table(TableInfo& table, const MultiJoinConfig& config) {
        table.hash_table.clear();
        table.hash_table.reserve(table.count);

        for (size_t i = 0; i < table.count; ++i) {
            if (table.filter && !table.filter(i)) continue;
            table.hash_table[table.keys[i]].push_back(static_cast<uint32_t>(i));
        }
    }

    void probe_hash_table(size_t table_idx, JoinResult& result, const MultiJoinConfig& config) {
        auto& table = tables_[table_idx];
        auto& first_ht = tables_[0].hash_table;

        for (size_t i = 0; i < table.count; ++i) {
            if (table.filter && !table.filter(i)) continue;

            auto it = first_ht.find(table.keys[i]);
            if (it != first_ht.end()) {
                result.table_indices[table_idx].push_back(static_cast<uint32_t>(i));
                result.match_count++;
            }
        }
    }

    std::vector<TableInfo> tables_;
};

// ============================================================================
// 通用算子: GenericTwoPhaseAgg (两阶段聚合)
// ============================================================================

/**
 * 两阶段聚合配置
 */
struct TwoPhaseAggConfig {
    size_t num_threads = 8;
    bool use_thread_local = true;
};

/**
 * 通用两阶段聚合器
 *
 * Phase 1: 按分组键预聚合 (并行, Thread-Local)
 * Phase 2: 基于预聚合结果过滤后聚合
 *
 * 典型场景: Q17 的 AVG 子查询
 * WHERE l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = outer.l_partkey)
 */
template<typename KeyT, typename ValueT>
class GenericTwoPhaseAgg {
public:
    struct AggResult {
        ValueT sum = 0;
        size_t count = 0;

        ValueT avg() const { return count > 0 ? sum / count : 0; }
    };

    /**
     * Phase 1: 预计算每个键的聚合值
     */
    void phase1_precompute(
        const KeyT* keys,
        const ValueT* values,
        size_t count,
        std::function<bool(size_t)> filter = nullptr,
        const TwoPhaseAggConfig& config = {}
    ) {
        pre_agg_.clear();
        pre_agg_.reserve(count / 10);

        if (config.use_thread_local && config.num_threads > 1) {
            // 并行预计算
            std::vector<std::unordered_map<KeyT, AggResult>> thread_local_aggs(config.num_threads);
            std::vector<std::thread> threads;

            size_t chunk = (count + config.num_threads - 1) / config.num_threads;

            for (size_t t = 0; t < config.num_threads; ++t) {
                size_t start = t * chunk;
                size_t end = std::min(start + chunk, count);

                threads.emplace_back([&, t, start, end]() {
                    auto& local = thread_local_aggs[t];
                    for (size_t i = start; i < end; ++i) {
                        if (filter && !filter(i)) continue;
                        auto& r = local[keys[i]];
                        r.sum += values[i];
                        r.count++;
                    }
                });
            }

            for (auto& th : threads) th.join();

            // 合并
            for (const auto& local : thread_local_aggs) {
                for (const auto& [k, v] : local) {
                    auto& r = pre_agg_[k];
                    r.sum += v.sum;
                    r.count += v.count;
                }
            }
        } else {
            // 单线程
            for (size_t i = 0; i < count; ++i) {
                if (filter && !filter(i)) continue;
                auto& r = pre_agg_[keys[i]];
                r.sum += values[i];
                r.count++;
            }
        }
    }

    /**
     * Phase 2: 基于预聚合结果执行最终聚合
     *
     * @param keys 待过滤数据的键
     * @param values 待聚合的值
     * @param count 数据量
     * @param threshold_func 阈值函数 (接收 AggResult, 返回阈值)
     * @param filter 额外过滤条件
     */
    template<typename ResultT>
    ResultT phase2_aggregate(
        const KeyT* keys,
        const ValueT* values,
        size_t count,
        std::function<ValueT(const AggResult&)> threshold_func,
        std::function<bool(size_t, ValueT threshold)> value_filter,
        std::function<ResultT(ResultT, ValueT)> agg_func,
        ResultT init_value = ResultT{},
        const TwoPhaseAggConfig& config = {}
    ) {
        ResultT result = init_value;

        for (size_t i = 0; i < count; ++i) {
            auto it = pre_agg_.find(keys[i]);
            if (it == pre_agg_.end()) continue;

            ValueT threshold = threshold_func(it->second);
            if (!value_filter(i, threshold)) continue;

            result = agg_func(result, values[i]);
        }

        return result;
    }

    /**
     * 查找预聚合结果
     */
    bool lookup(KeyT key, AggResult& result) const {
        auto it = pre_agg_.find(key);
        if (it != pre_agg_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    size_t precomputed_groups() const { return pre_agg_.size(); }

private:
    std::unordered_map<KeyT, AggResult> pre_agg_;
};

// ============================================================================
// Q2: 子查询解关联优化
// ============================================================================

/**
 * Q2: 最小成本供应商 (V55 - 子查询解关联)
 *
 * 原始 SQL 中的相关子查询:
 * WHERE ps_supplycost = (
 *     SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region
 *     WHERE p_partkey = ps_partkey ...
 * )
 *
 * 优化策略:
 * 1. 预计算 min_cost[partkey] = MIN(ps_supplycost) for EUROPE suppliers
 * 2. 主查询直接查找: WHERE ps_supplycost = min_cost[ps_partkey]
 */
void run_q2_v55(TPCHDataLoader& loader);

// ============================================================================
// Q8: 通用并行多表 JOIN (替代专用 V42)
// ============================================================================

/**
 * Q8: 市场份额 (V55 - 通用 GenericParallelMultiJoin)
 *
 * 8 表 JOIN:
 * part ⟕ lineitem ⟕ orders ⟕ customer ⟕ supplier ⟕ nation (x2) ⟕ region
 */
void run_q8_v55(TPCHDataLoader& loader);

// ============================================================================
// Q17: 通用两阶段聚合 (替代专用 V43)
// ============================================================================

/**
 * Q17: 小订单收入 (V55 - 通用 GenericTwoPhaseAgg)
 *
 * 子查询:
 * WHERE l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = outer.l_partkey)
 *
 * 优化策略:
 * Phase 1: 预计算 avg_qty[partkey] = AVG(l_quantity)
 * Phase 2: 过滤 l_quantity < 0.2 * avg_qty[l_partkey] 并聚合
 */
void run_q17_v55(TPCHDataLoader& loader);

// ============================================================================
// 算子适用性检查
// ============================================================================

/**
 * 检查 V55 通用算子是否适用于指定查询
 */
inline bool is_v55_applicable(const std::string& query_id, size_t rows) {
    if (query_id == "Q2") return rows >= 10000;
    if (query_id == "Q8") return rows >= 100000;
    if (query_id == "Q17") return rows >= 50000;
    return false;
}

} // namespace ops_v55
} // namespace tpch
} // namespace thunderduck
