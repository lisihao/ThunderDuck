/**
 * ThunderDuck TPC-H Query Optimizer
 *
 * 动态算子选择系统，根据查询特征和数据量选择最优算子版本
 * 集成系统表，支持基于历史性能数据的智能选择
 *
 * @version 2.0
 * @date 2026-01-29
 */

#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace thunderduck {

// 前向声明系统表
namespace catalog {
    class SystemCatalog;
    SystemCatalog& catalog();
}

namespace tpch {

// 前向声明
class TPCHDataLoader;

// ============================================================================
// 查询执行函数类型
// ============================================================================

using QueryExecutor = std::function<void(TPCHDataLoader&)>;

// ============================================================================
// 选择策略
// ============================================================================

enum class SelectionStrategy {
    STATIC,         // 静态配置 (基于预设加速比)
    ADAPTIVE,       // 自适应 (基于历史数据)
    HYBRID          // 混合 (历史数据 + 静态配置)
};

// ============================================================================
// 查询算子配置
// ============================================================================

struct QueryOperatorConfig {
    std::string query_id;                    // 查询 ID (Q1-Q22)
    std::string description;                 // 查询描述

    // 数据特征
    size_t estimated_rows = 0;               // 估计处理行数
    size_t join_count = 0;                   // JOIN 数量
    bool has_subquery = false;               // 是否有子查询
    bool has_aggregation = true;             // 是否有聚合
    bool has_top_n = false;                  // 是否有 TOP-N

    // 算子适用性检查上下文
    struct ApplicabilityContext {
        size_t row_count = 0;                // 实际行数
        size_t max_key_range = 0;            // 最大 key 范围 (用于 DirectArray 适用性)
        size_t distinct_keys = 0;            // 不同 key 数量 (基数)
        double selectivity = 1.0;            // 选择率
        size_t l2_cache_bytes = 4 * 1024 * 1024;  // L2 缓存大小
        bool has_native_double = false;      // 是否有原生 double 列
    };

    // 适用性检查函数类型
    using ApplicabilityChecker = std::function<bool(const ApplicabilityContext&)>;

    // 版本候选 (按优先级排序)
    struct VersionCandidate {
        std::string version;                 // 版本号 (V25, V32, V37...)
        double speedup;                      // 相对 DuckDB 的加速比
        size_t min_rows;                     // 最小适用行数
        size_t max_rows;                     // 最大适用行数 (0 = 无限)
        QueryExecutor executor;              // 执行函数
        ApplicabilityChecker is_applicable;  // 适用性检查 (可选)
    };
    std::vector<VersionCandidate> candidates;

    // 默认回退
    QueryExecutor fallback;                  // 回退执行函数
};

// ============================================================================
// 选择结果
// ============================================================================

struct SelectionResult {
    std::string selected_version;
    QueryExecutor executor;
    double predicted_time_ms;
    double confidence;
    std::string selection_reason;
    SelectionStrategy strategy_used;
};

// ============================================================================
// TPC-H 查询优化器
// ============================================================================

class TPCHQueryOptimizer {
public:
    // 单例
    static TPCHQueryOptimizer& instance() {
        static TPCHQueryOptimizer opt;
        return opt;
    }

    // ========================================================================
    // 配置
    // ========================================================================

    void set_strategy(SelectionStrategy strategy) {
        strategy_ = strategy;
    }

    SelectionStrategy get_strategy() const {
        return strategy_;
    }

    void enable_logging(bool enable) {
        logging_enabled_ = enable;
    }

    // ========================================================================
    // 注册
    // ========================================================================

    void register_query(const QueryOperatorConfig& config) {
        configs_[config.query_id] = config;
    }

    const QueryOperatorConfig* get_config(const std::string& query_id) const {
        auto it = configs_.find(query_id);
        return it != configs_.end() ? &it->second : nullptr;
    }

    // ========================================================================
    // 动态选择最优版本
    // ========================================================================

    /**
     * 根据数据量选择最优执行器 (静态策略)
     */
    QueryExecutor select_best_executor(
        const std::string& query_id,
        size_t actual_rows
    ) const {
        QueryOperatorConfig::ApplicabilityContext ctx;
        ctx.row_count = actual_rows;
        return select_best_executor(query_id, ctx);
    }

    /**
     * 根据完整上下文选择最优执行器
     */
    QueryExecutor select_best_executor(
        const std::string& query_id,
        const QueryOperatorConfig::ApplicabilityContext& ctx
    ) const {
        auto result = select_with_details(query_id, ctx);
        return result.executor;
    }

    /**
     * 选择最优版本并返回详细信息 (简化版)
     */
    SelectionResult select_with_details(
        const std::string& query_id,
        size_t actual_rows
    ) const {
        QueryOperatorConfig::ApplicabilityContext ctx;
        ctx.row_count = actual_rows;
        return select_with_details(query_id, ctx);
    }

    /**
     * 选择最优版本并返回详细信息 (完整上下文)
     */
    SelectionResult select_with_details(
        const std::string& query_id,
        const QueryOperatorConfig::ApplicabilityContext& ctx
    ) const {
        SelectionResult result;
        result.strategy_used = strategy_;

        auto it = configs_.find(query_id);
        if (it == configs_.end()) {
            result.selection_reason = "Query not registered";
            return result;
        }

        const auto& config = it->second;

        // 根据策略选择
        switch (strategy_) {
            case SelectionStrategy::ADAPTIVE:
                return select_adaptive(query_id, ctx, config);

            case SelectionStrategy::HYBRID:
                return select_hybrid(query_id, ctx, config);

            case SelectionStrategy::STATIC:
            default:
                return select_static(query_id, ctx, config);
        }
    }

    /**
     * 选择最优版本 (简化版，使用估计行数)
     */
    QueryExecutor select_best(const std::string& query_id) const {
        auto it = configs_.find(query_id);
        if (it == configs_.end()) {
            return nullptr;
        }

        const auto& config = it->second;
        return select_best_executor(query_id, config.estimated_rows);
    }

    /**
     * 获取选中版本的描述
     */
    std::string get_selected_version(
        const std::string& query_id,
        size_t actual_rows
    ) const {
        auto result = select_with_details(query_id, actual_rows);
        return result.selected_version;
    }

    /**
     * 获取所有已注册查询 ID
     */
    std::vector<std::string> get_all_query_ids() const {
        std::vector<std::string> ids;
        ids.reserve(configs_.size());
        for (const auto& kv : configs_) {
            ids.push_back(kv.first);
        }
        std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) {
            int na = std::stoi(a.substr(1));
            int nb = std::stoi(b.substr(1));
            return na < nb;
        });
        return ids;
    }

    /**
     * 打印所有注册的查询配置
     */
    void print_registry() const {
        std::cout << "\n========== TPC-H Query Optimizer Registry ==========\n";
        std::cout << "Strategy: " << strategy_name() << "\n\n";

        auto ids = get_all_query_ids();
        for (const auto& id : ids) {
            const auto& config = configs_.at(id);
            auto selected = get_selected_version(id, config.estimated_rows);

            std::cout << id << ": " << config.description << "\n";
            std::cout << "  Selected: " << selected << "\n";
            std::cout << "  Candidates:\n";
            for (const auto& c : config.candidates) {
                std::cout << "    " << (c.version == selected ? "* " : "  ")
                          << c.version << " (" << c.speedup << "x)"
                          << " [" << c.min_rows << "-"
                          << (c.max_rows > 0 ? std::to_string(c.max_rows) : "inf") << "]\n";
            }
        }
        std::cout << "====================================================\n";
    }

    bool is_initialized() const {
        return !configs_.empty();
    }

    // ========================================================================
    // 性能记录 (集成系统表)
    // ========================================================================

    /**
     * 记录执行性能
     */
    void record_execution(
        const std::string& query_id,
        const std::string& version,
        double execution_time_ms,
        size_t rows_processed
    );

    /**
     * 获取版本的历史性能统计
     */
    struct VersionPerformance {
        std::string version;
        double avg_time_ms;
        double median_time_ms;
        double stddev_ms;
        size_t sample_count;
        double confidence;
    };

    VersionPerformance get_version_performance(
        const std::string& query_id,
        const std::string& version
    ) const;

    /**
     * 获取查询的所有版本性能
     */
    std::vector<VersionPerformance> get_all_version_performance(
        const std::string& query_id
    ) const;

private:
    TPCHQueryOptimizer() : strategy_(SelectionStrategy::STATIC), logging_enabled_(false) {}

    // 静态选择 (基于预设加速比 + 适用性检查)
    SelectionResult select_static(
        const std::string& query_id,
        const QueryOperatorConfig::ApplicabilityContext& ctx,
        const QueryOperatorConfig& config
    ) const {
        SelectionResult result;
        result.strategy_used = SelectionStrategy::STATIC;

        const QueryOperatorConfig::VersionCandidate* best = nullptr;
        double best_speedup = 0.0;

        for (const auto& candidate : config.candidates) {
            // 检查行数范围
            if (ctx.row_count < candidate.min_rows) continue;
            if (candidate.max_rows > 0 && ctx.row_count > candidate.max_rows) continue;

            // 检查适用性函数 (如果提供)
            if (candidate.is_applicable && !candidate.is_applicable(ctx)) {
                continue;  // 算子不适用于当前上下文
            }

            if (candidate.speedup > best_speedup) {
                best_speedup = candidate.speedup;
                best = &candidate;
            }
        }

        if (best && best->executor) {
            result.selected_version = best->version;
            result.executor = best->executor;
            result.confidence = 1.0;  // 静态配置置信度为 1
            result.selection_reason = "Static config (speedup=" +
                std::to_string(best_speedup).substr(0, 4) + "x)";
        } else {
            result.selected_version = "Fallback";
            result.executor = config.fallback;
            result.confidence = 0.5;
            result.selection_reason = "No matching candidate, using fallback";
        }

        if (logging_enabled_) {
            log_selection(query_id, result);
        }

        return result;
    }

    // 自适应选择 (基于历史数据)
    SelectionResult select_adaptive(
        const std::string& query_id,
        const QueryOperatorConfig::ApplicabilityContext& ctx,
        const QueryOperatorConfig& config
    ) const;

    // 混合选择 (历史数据 + 静态配置)
    SelectionResult select_hybrid(
        const std::string& query_id,
        const QueryOperatorConfig::ApplicabilityContext& ctx,
        const QueryOperatorConfig& config
    ) const;

    void log_selection(const std::string& query_id, const SelectionResult& result) const {
        std::cout << "[Optimizer] " << query_id << " -> " << result.selected_version
                  << " (" << result.selection_reason << ")\n";
    }

    const char* strategy_name() const {
        switch (strategy_) {
            case SelectionStrategy::STATIC: return "STATIC";
            case SelectionStrategy::ADAPTIVE: return "ADAPTIVE";
            case SelectionStrategy::HYBRID: return "HYBRID";
            default: return "UNKNOWN";
        }
    }

    std::unordered_map<std::string, QueryOperatorConfig> configs_;
    SelectionStrategy strategy_;
    bool logging_enabled_;
};

// ============================================================================
// 注册所有 TPC-H 查询配置
// ============================================================================

void register_tpch_query_configs();

// ============================================================================
// 初始化算子元数据到系统表
// ============================================================================

void register_operator_metadata();

} // namespace tpch
} // namespace thunderduck
