/**
 * ThunderDuck TPC-H V48 - 通用 Group-then-Filter 算子
 *
 * 核心模式 (非 JOIN/EXISTS):
 *   groupkey → entity状态集合 → EXACT-K predicate
 *
 * 通用算子:
 * - CountingSortGrouper: 计数排序分组
 * - GenerationDeduplicator: Generation Counter 去重
 * - ExactKPredicate: EXACT-K 谓词评估
 *
 * @deprecated 专用类已迁移至通用别名:
 *   - Q21GenericOptimizer → ExistsSubqueryOptimizer (通用别名)
 *   - Q21CorrectOptimizer → CorrectExistsOptimizer (通用别名)
 *
 * @version 48.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_constants.h"   // 统一常量
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace thunderduck {
namespace tpch {
namespace ops_v48 {

// ============================================================================
// 通用配置结构
// ============================================================================

/**
 * Q21Config - Q21 查询配置 (完全参数化)
 */
struct Q21Config {
    // 目标国家
    std::string target_nation = constants::nations::SAUDI_ARABIA;

    // 订单状态过滤 (默认 'F' = 0)
    int8_t order_status_filter = 0;

    // EXACT-K predicate 参数
    uint16_t min_supplier_count = 2;  // supplier_count > 1 means >= 2
    uint16_t exact_late_count = 1;    // late_count == 1

    // 结果限制
    size_t limit = 100;
};

// ============================================================================
// 通用算子: CountingSortGrouper
// ============================================================================

/**
 * CountingSortGrouper - 计数排序分组器
 *
 * 通用模式: 将数据按 groupkey 分组，支持自定义过滤
 *
 * 复杂度: O(n + k) where n=数据量, k=groupkey范围
 */
template<typename T>
class CountingSortGrouper {
public:
    using FilterFunc = std::function<bool(size_t idx)>;
    using ExtractFunc = std::function<T(size_t idx)>;

    struct Config {
        int32_t max_key;
        size_t data_count;
        FilterFunc filter;      // 过滤函数
        ExtractFunc extract;    // 数据提取函数
    };

    /**
     * 执行分组
     * @return (counts, sorted_data) - counts[k] 是 key=k 的起始偏移
     */
    static std::pair<std::vector<uint32_t>, std::vector<T>> execute(
        const int32_t* keys,
        const Config& config
    ) {
        std::vector<uint32_t> counts(config.max_key + 2, 0);

        // 第一遍: 计数
        for (size_t i = 0; i < config.data_count; ++i) {
            int32_t k = keys[i];
            if (k > 0 && k <= config.max_key && config.filter(i)) {
                counts[k + 1]++;
            }
        }

        // 前缀和
        for (int32_t k = 1; k <= config.max_key + 1; ++k) {
            counts[k] += counts[k - 1];
        }

        // 第二遍: 分配数据
        std::vector<T> sorted_data(counts[config.max_key + 1]);
        std::vector<uint32_t> offsets = counts;

        for (size_t i = 0; i < config.data_count; ++i) {
            int32_t k = keys[i];
            if (k > 0 && k <= config.max_key && config.filter(i)) {
                sorted_data[offsets[k]++] = config.extract(i);
            }
        }

        return {counts, sorted_data};
    }
};

// ============================================================================
// 通用算子: GenerationDeduplicator
// ============================================================================

/**
 * GenerationDeduplicator - Generation Counter 去重器
 *
 * 通用模式: 跨多个组高效去重，无需清理状态
 *
 * 原理: 每个 entity 存储 seen_gen，当 seen_gen != current_gen 时表示未见
 */
template<size_t N = 2>  // N = 状态维度数
class GenerationDeduplicator {
public:
    struct State {
        uint32_t gen[N] = {0};
    };

    GenerationDeduplicator(size_t max_entity) : states_(max_entity + 1), current_gen_(1) {}

    void next_group() { ++current_gen_; }

    // 检查并标记第 dim 维度
    bool check_and_mark(int32_t entity, size_t dim = 0) {
        if (entity <= 0 || entity >= static_cast<int32_t>(states_.size())) return false;
        auto& state = states_[entity];
        if (state.gen[dim] == current_gen_) return false;  // 已见
        state.gen[dim] = current_gen_;
        return true;  // 首次见
    }

private:
    std::vector<State> states_;
    uint32_t current_gen_;
};

// ============================================================================
// Q21 通用优化器
// ============================================================================

class Q21GenericOptimizer {
public:
    struct Result {
        std::vector<std::pair<std::string, int64_t>> suppliers;
    };

    /**
     * 执行 Q21 - 完全参数化的通用实现
     */
    static Result execute(
        // Supplier 表
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        size_t supplier_count,
        // Lineitem 表
        const int32_t* l_orderkey,
        const int32_t* l_suppkey,
        const int32_t* l_commitdate,
        const int32_t* l_receiptdate,
        size_t lineitem_count,
        // Orders 表
        const int32_t* o_orderkey,
        const int8_t* o_orderstatus,
        size_t orders_count,
        // Nation 表
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // 配置
        const Q21Config& config
    );
};

// 保持向后兼容的旧接口
class Q21CorrectOptimizer {
public:
    struct Result {
        std::vector<std::pair<std::string, int64_t>> suppliers;
    };

    static Result execute(
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        size_t supplier_count,
        const int32_t* l_orderkey,
        const int32_t* l_suppkey,
        const int32_t* l_commitdate,
        const int32_t* l_receiptdate,
        size_t lineitem_count,
        const int32_t* o_orderkey,
        const int8_t* o_orderstatus,
        size_t orders_count,
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        const std::string& target_nation,
        size_t limit = 100
    );
};

/**
 * V48 Q21 查询入口
 */
void run_q21_v48(TPCHDataLoader& loader);
void run_q21_v48(TPCHDataLoader& loader, const Q21Config& config);

// ============================================================================
// 通用别名 (推荐使用)
// ============================================================================

/**
 * ExistsSubqueryOptimizer - EXISTS 子查询优化器通用别名
 * @note 推荐使用此别名以提高代码可读性
 */
using ExistsSubqueryOptimizer = Q21GenericOptimizer;

/**
 * CorrectExistsOptimizer - 正确实现的 EXISTS 优化器通用别名
 * @note 推荐使用此别名以提高代码可读性
 */
using CorrectExistsOptimizer = Q21CorrectOptimizer;

} // namespace ops_v48
} // namespace tpch
} // namespace thunderduck
