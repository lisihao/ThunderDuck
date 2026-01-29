/**
 * ThunderDuck TPC-H Query Optimizer Implementation
 *
 * 注册所有 TPC-H 查询的算子版本配置
 * 基于 V50 版本演进报告中的性能数据
 */

// 系统表必须在优化器之前 include
#include "../../include/thunderduck/system_catalog.h"
#include "tpch_query_optimizer.h"
#include "tpch_data_loader.h"
#include "tpch_operators_v24.h"
#include "tpch_operators_v25.h"
#include "tpch_operators_v27.h"
#include "tpch_operators_v32.h"
#include "tpch_operators_v33.h"
#include "tpch_operators_v34.h"
#include "tpch_operators_v35.h"
#include "tpch_operators_v36.h"
#include "tpch_operators_v37.h"
#include "tpch_operators_v40.h"
#include "tpch_operators_v42.h"
#include "tpch_operators_v43.h"
#include "tpch_operators_v46.h"
#include "tpch_operators_v47.h"
#include "tpch_operators_v48.h"
#include "tpch_operators_v49.h"
#include "tpch_operators_v50.h"

namespace thunderduck {
namespace tpch {

// 前向声明基础查询实现 (仅声明已存在的)
namespace queries {
    void run_q1(TPCHDataLoader& loader);
    void run_q2(TPCHDataLoader& loader);
    void run_q3(TPCHDataLoader& loader);
    void run_q4(TPCHDataLoader& loader);
    void run_q5(TPCHDataLoader& loader);
    void run_q6(TPCHDataLoader& loader);
    void run_q7(TPCHDataLoader& loader);
    // Q8 无基础实现，使用优化版本
    void run_q9(TPCHDataLoader& loader);
    void run_q10(TPCHDataLoader& loader);
    void run_q11(TPCHDataLoader& loader);
    void run_q12(TPCHDataLoader& loader);
    // Q13 无基础实现，使用优化版本
    void run_q14(TPCHDataLoader& loader);
    void run_q15(TPCHDataLoader& loader);
    void run_q16(TPCHDataLoader& loader);
    // Q17 无基础实现，使用优化版本
    void run_q18(TPCHDataLoader& loader);
    void run_q19(TPCHDataLoader& loader);
    // Q20, Q21, Q22 无基础实现，使用优化版本
}

void register_tpch_query_configs() {
    auto& opt = TPCHQueryOptimizer::instance();

    // ========================================================================
    // Q1: 定价汇总报告 - 单表聚合 (9.15x)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q1";
        config.description = "Pricing Summary Report (single table aggregation)";
        config.estimated_rows = 6000000;
        config.join_count = 0;
        config.has_aggregation = true;

        // 基础版本已经是最优
        config.candidates = {
            {"Base", 9.15, 0, 0, queries::run_q1}
        };
        config.fallback = queries::run_q1;

        opt.register_query(config);
    }

    // ========================================================================
    // Q2: 最低成本供应商 - 4 表 JOIN (基础)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q2";
        config.description = "Minimum Cost Supplier (4-way join)";
        config.estimated_rows = 800000;
        config.join_count = 4;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"Base", 1.0, 0, 0, queries::run_q2}
        };
        config.fallback = queries::run_q2;

        opt.register_query(config);
    }

    // ========================================================================
    // Q3: 运输优先级 - 3 表 JOIN + Top-N (1.29x V49)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q3";
        config.description = "Shipping Priority (3-way join with Top-N)";
        config.estimated_rows = 7500000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_top_n = true;

        config.candidates = {
            // V49 最优: Top-N Aware 聚合 (+9% vs V31)
            {"V49", 1.29, 1000000, 0, [](TPCHDataLoader& l) { ops_v49::run_q3_v49(l); }},
            // V31 次优: GPU SEMI + V19.2 JOIN
            {"V31", 1.14, 100000, 0, [](TPCHDataLoader& l) { ops_v27::run_q3_v31(l); }},
            // V27 小数据
            {"V27", 0.82, 10000, 100000, [](TPCHDataLoader& l) { ops_v27::run_q3_v27(l); }},
            // V25 基础
            {"V25", 0.53, 0, 10000, [](TPCHDataLoader& l) { ops_v25::run_q3_v25(l); }}
        };
        config.fallback = queries::run_q3;

        opt.register_query(config);
    }

    // ========================================================================
    // Q4: 订单优先级 - 2 表 SEMI JOIN (1.2x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q4";
        config.description = "Order Priority Checking (SEMI join)";
        config.estimated_rows = 3000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V27", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q4_v27(l); }}
        };
        config.fallback = queries::run_q4;

        opt.register_query(config);
    }

    // ========================================================================
    // Q5: 本地供应商收入 - 6 表 JOIN (1.27x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q5";
        config.description = "Local Supplier Volume (6-way join)";
        config.estimated_rows = 8000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            // V32 最优: 紧凑 Hash + Bloom Filter
            {"V32", 1.27, 100000, 0, [](TPCHDataLoader& l) { ops_v32::run_q5_v32(l); }},
            // V25 基础
            {"V25", 0.8, 0, 100000, [](TPCHDataLoader& l) { ops_v25::run_q5_v25(l); }}
        };
        config.fallback = queries::run_q5;

        opt.register_query(config);
    }

    // ========================================================================
    // Q6: 预测收入变化 - 单表聚合 (目标 3.0x+)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q6";
        config.description = "Forecasting Revenue Change (single table)";
        config.estimated_rows = 6000000;
        config.join_count = 0;
        config.has_aggregation = true;

        config.candidates = {
            // V47 最优: SIMD 无分支过滤
            {"V47", 3.0, 0, 0, [](TPCHDataLoader& l) { ops_v47::run_q6_v47(l); }},
            // V25 基础
            {"V25", 2.0, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q6_v25(l); }}
        };
        config.fallback = queries::run_q6;

        opt.register_query(config);
    }

    // ========================================================================
    // Q7: 国家间运输量 - 6 表 JOIN (2.63x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q7";
        config.description = "Volume Shipping (6-way join with date filter)";
        config.estimated_rows = 6000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            {"V32", 2.63, 0, 0, [](TPCHDataLoader& l) { ops_v32::run_q7_v32(l); }},
            {"V25", 1.5, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q7_v25(l); }}
        };
        config.fallback = queries::run_q7;

        opt.register_query(config);
    }

    // ========================================================================
    // Q8: 市场份额 - 8 表 JOIN (1.85x V42)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q8";
        config.description = "National Market Share (8-way join)";
        config.estimated_rows = 8000000;
        config.join_count = 7;
        config.has_aggregation = true;

        config.candidates = {
            {"V42", 1.85, 0, 0, [](TPCHDataLoader& l) { ops_v42::run_q8_v42(l); }},
            {"V34", 1.05, 0, 0, [](TPCHDataLoader& l) { ops_v34::run_q8_v34(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v34::run_q8_v34(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q9: 产品利润 - 6 表 JOIN (1.30x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q9";
        config.description = "Product Type Profit Measure (6-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            {"V32", 1.30, 0, 0, [](TPCHDataLoader& l) { ops_v32::run_q9_v32(l); }},
            {"V25", 0.9, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q9_v25(l); }}
        };
        config.fallback = queries::run_q9;

        opt.register_query(config);
    }

    // ========================================================================
    // Q10: 退货客户 - 4 表 JOIN (1.7x V25)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q10";
        config.description = "Returned Item Reporting (4-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_top_n = true;

        config.candidates = {
            {"V25", 1.7, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q10_v25(l); }}
        };
        config.fallback = queries::run_q10;

        opt.register_query(config);
    }

    // ========================================================================
    // Q11: 重要库存 - 3 表 JOIN (4.14x V46)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q11";
        config.description = "Important Stock Identification (3-way join)";
        config.estimated_rows = 800000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V46", 4.14, 0, 0, [](TPCHDataLoader& l) { ops_v46::run_q11_v46(l); }},
            {"V27", 2.0, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q11_v27(l); }}
        };
        config.fallback = queries::run_q11;

        opt.register_query(config);
    }

    // ========================================================================
    // Q12: 运输方式统计 - 2 表 JOIN (0.8x V27, 待优化)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q12";
        config.description = "Shipping Modes and Order Priority (2-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V27", 0.8, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q12_v27(l); }},
            {"V25", 0.6, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q12_v25(l); }}
        };
        config.fallback = queries::run_q12;

        opt.register_query(config);
    }

    // ========================================================================
    // Q13: 客户订单分布 - 2 表 LEFT OUTER JOIN (1.96x V34)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q13";
        config.description = "Customer Distribution (LEFT OUTER join)";
        config.estimated_rows = 1500000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V34", 1.96, 0, 0, [](TPCHDataLoader& l) { ops_v34::run_q13_v34(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v34::run_q13_v34(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q14: 促销效果 - 2 表 JOIN (2.91x V46)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q14";
        config.description = "Promotion Effect (2-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V46", 2.91, 0, 0, [](TPCHDataLoader& l) { ops_v46::run_q14_v46(l); }},
            {"V25", 1.8, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q14_v25(l); }}
        };
        config.fallback = queries::run_q14;

        opt.register_query(config);
    }

    // ========================================================================
    // Q15: 顶级供应商 - 2 表 JOIN (1.3x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q15";
        config.description = "Top Supplier Query (2-way join with subquery)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V27", 1.3, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q15_v27(l); }}
        };
        config.fallback = queries::run_q15;

        opt.register_query(config);
    }

    // ========================================================================
    // Q16: 零件供应商关系 - 3 表 JOIN (1.2x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q16";
        config.description = "Parts/Supplier Relationship (3-way join)";
        config.estimated_rows = 800000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V27", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q16_v27(l); }}
        };
        config.fallback = queries::run_q16;

        opt.register_query(config);
    }

    // ========================================================================
    // Q17: 小订单收入 - 2 表 JOIN (4.30x V43)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q17";
        config.description = "Small-Quantity Order Revenue (2-way join with AVG)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V43", 4.30, 0, 0, [](TPCHDataLoader& l) { ops_v43::run_q17_v43(l); }},
            {"V36", 1.16, 0, 0, [](TPCHDataLoader& l) { ops_v36::run_q17_v36(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v36::run_q17_v36(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q18: 大订单客户 - 3 表 JOIN (4.27x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q18";
        config.description = "Large Volume Customer (3-way join with HAVING)";
        config.estimated_rows = 7500000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_top_n = true;
        config.has_subquery = true;

        config.candidates = {
            {"V32", 4.27, 0, 0, [](TPCHDataLoader& l) { ops_v32::run_q18_v32(l); }},
            {"V25", 2.0, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q18_v25(l); }}
        };
        config.fallback = queries::run_q18;

        opt.register_query(config);
    }

    // ========================================================================
    // Q19: 折扣收入 - 2 表 JOIN (2.0x V33)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q19";
        config.description = "Discounted Revenue (2-way join with OR conditions)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V33", 2.0, 0, 0, [](TPCHDataLoader& l) { ops_v33::run_q19_v33(l); }}
        };
        config.fallback = queries::run_q19;

        opt.register_query(config);
    }

    // ========================================================================
    // Q20: 潜在零件促销 - 4 表 JOIN (1.29x V40)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q20";
        config.description = "Potential Part Promotion (4-way join with subquery)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V40", 1.29, 0, 0, [](TPCHDataLoader& l) { ops_v40::run_q20_v40(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v40::run_q20_v40(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q21: 延迟供应商 - 4 表 JOIN (1.0x V48)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q21";
        config.description = "Suppliers Who Kept Orders Waiting (4-way join with EXISTS)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V50", 1.5, 0, 0, [](TPCHDataLoader& l) { ops_v50::run_q21_v50(l); }},
            {"V48", 1.0, 0, 0, [](TPCHDataLoader& l) { ops_v48::run_q21_v48(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v48::run_q21_v48(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q22: 全球销售机会 - ANTI JOIN (9.08x V37)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q22";
        config.description = "Global Sales Opportunity (ANTI join)";
        config.estimated_rows = 1500000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V37", 9.08, 0, 0, [](TPCHDataLoader& l) { ops_v37::run_q22_v37(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v37::run_q22_v37(l); };

        opt.register_query(config);
    }
}

// ============================================================================
// 系统表集成实现
// ============================================================================

void TPCHQueryOptimizer::record_execution(
    const std::string& query_id,
    const std::string& version,
    double execution_time_ms,
    size_t rows_processed
) {
    // 使用轻量级 API 记录性能数据
    catalog::catalog().record_metric(query_id, version, execution_time_ms, rows_processed);

    // 记录版本选择
    catalog::catalog().record_selection(query_id, version, rows_processed);
}

TPCHQueryOptimizer::VersionPerformance TPCHQueryOptimizer::get_version_performance(
    const std::string& query_id,
    const std::string& version
) const {
    VersionPerformance perf{};
    perf.version = version;

    auto stats = catalog::catalog().get_version_stats(query_id, version);
    perf.avg_time_ms = stats.avg_time_ms;
    perf.median_time_ms = stats.median_time_ms;
    perf.stddev_ms = stats.stddev_ms;
    perf.sample_count = stats.sample_count;
    perf.confidence = catalog::catalog().get_version_confidence(query_id, version);

    return perf;
}

std::vector<TPCHQueryOptimizer::VersionPerformance> TPCHQueryOptimizer::get_all_version_performance(
    const std::string& query_id
) const {
    std::vector<VersionPerformance> result;

    auto it = configs_.find(query_id);
    if (it == configs_.end()) {
        return result;
    }

    for (const auto& candidate : it->second.candidates) {
        result.push_back(get_version_performance(query_id, candidate.version));
    }

    return result;
}

SelectionResult TPCHQueryOptimizer::select_adaptive(
    const std::string& query_id,
    size_t actual_rows,
    const QueryOperatorConfig& config
) const {
    SelectionResult result;
    result.strategy_used = SelectionStrategy::ADAPTIVE;

    // 收集所有候选版本
    std::vector<std::string> candidates;
    for (const auto& c : config.candidates) {
        if (actual_rows >= c.min_rows &&
            (c.max_rows == 0 || actual_rows <= c.max_rows)) {
            candidates.push_back(c.version);
        }
    }

    if (candidates.empty()) {
        result.selected_version = "Fallback";
        result.executor = config.fallback;
        result.confidence = 0.5;
        result.selection_reason = "No candidates for row count";
        return result;
    }

    // 使用系统表选择最优版本
    auto [best_version, predicted_time] =
        catalog::catalog().select_best_version(query_id, actual_rows, candidates);

    if (best_version.empty()) {
        // 没有历史数据，回退到静态选择
        return select_static(query_id, actual_rows, config);
    }

    // 查找对应的执行器
    for (const auto& c : config.candidates) {
        if (c.version == best_version) {
            result.selected_version = best_version;
            result.executor = c.executor;
            result.predicted_time_ms = predicted_time;
            result.confidence = catalog::catalog().get_version_confidence(query_id, best_version);
            result.selection_reason = "Adaptive (predicted=" +
                std::to_string(static_cast<int>(predicted_time)) + "ms, conf=" +
                std::to_string(static_cast<int>(result.confidence * 100)) + "%)";

            if (logging_enabled_) {
                log_selection(query_id, result);
            }

            return result;
        }
    }

    // 未找到，回退
    result.selected_version = "Fallback";
    result.executor = config.fallback;
    result.confidence = 0.5;
    result.selection_reason = "Version not found in candidates";
    return result;
}

SelectionResult TPCHQueryOptimizer::select_hybrid(
    const std::string& query_id,
    size_t actual_rows,
    const QueryOperatorConfig& config
) const {
    SelectionResult result;
    result.strategy_used = SelectionStrategy::HYBRID;

    // 首先尝试自适应选择
    auto adaptive_result = select_adaptive(query_id, actual_rows, config);

    // 如果置信度足够高，使用自适应结果
    constexpr double CONFIDENCE_THRESHOLD = 0.7;

    if (adaptive_result.confidence >= CONFIDENCE_THRESHOLD) {
        adaptive_result.strategy_used = SelectionStrategy::HYBRID;
        adaptive_result.selection_reason = "Hybrid/Adaptive (" +
            adaptive_result.selection_reason.substr(adaptive_result.selection_reason.find('(') + 1);
        return adaptive_result;
    }

    // 否则，混合静态配置
    auto static_result = select_static(query_id, actual_rows, config);

    // 如果有历史数据但置信度低，加权混合
    if (adaptive_result.predicted_time_ms > 0 && !adaptive_result.selected_version.empty()) {
        // 比较两种选择
        double static_speedup = 0.0;
        for (const auto& c : config.candidates) {
            if (c.version == static_result.selected_version) {
                static_speedup = c.speedup;
                break;
            }
        }

        // 如果自适应选择的版本预测性能更好，即使置信度低也优先使用
        // (预测时间与静态加速比反相关)
        if (adaptive_result.predicted_time_ms < 1000.0 / static_speedup) {
            adaptive_result.strategy_used = SelectionStrategy::HYBRID;
            adaptive_result.selection_reason = "Hybrid/Low-conf adaptive";
            return adaptive_result;
        }
    }

    static_result.strategy_used = SelectionStrategy::HYBRID;
    static_result.selection_reason = "Hybrid/Static (low historical conf)";

    if (logging_enabled_) {
        log_selection(query_id, static_result);
    }

    return static_result;
}

// ============================================================================
// 算子元数据注册
// ============================================================================

void register_operator_metadata() {
    auto& cat = catalog::catalog();

    // 注册各版本算子的元数据 (使用轻量级 API)
    // 参数: version, startup_ms, per_row_us, min_rows, max_rows

    // Filter 算子
    cat.register_operator("V19", 0.1f, 0.005f, 0, 0);         // 8线程并行 SIMD Filter
    cat.register_operator("V19.1", 0.05f, 0.003f, 0, 0);      // 无分支 SIMD Filter

    // Join 算子
    cat.register_operator("V19.2", 0.5f, 0.02f, 10000, 0);    // 激进预取 + SIMD Hash Join
    cat.register_operator("GPU", 2.0f, 0.001f, 100000, 0);    // Metal GPU SEMI Join

    // Aggregate 算子
    cat.register_operator("V15", 0.2f, 0.01f, 0, 0);          // 8线程 + 8路展开聚合
    cat.register_operator("V46", 0.05f, 0.001f, 0, 0);        // 低基数直接数组聚合

    // Top-N 算子
    cat.register_operator("V49", 0.1f, 0.008f, 0, 0);         // Top-N Aware Partial Aggregation

    // Bitmap 算子
    cat.register_operator("V37", 0.3f, 0.002f, 0, 0);         // Bitmap Anti-Join

    // TPC-H 查询版本
    cat.register_operator("V25", 0.2f, 0.015f, 0, 0);
    cat.register_operator("V27", 0.2f, 0.012f, 0, 0);
    cat.register_operator("V31", 0.3f, 0.01f, 100000, 0);
    cat.register_operator("V32", 0.3f, 0.008f, 100000, 0);
    cat.register_operator("V33", 0.2f, 0.01f, 0, 0);
    cat.register_operator("V34", 0.3f, 0.009f, 0, 0);
    cat.register_operator("V36", 0.2f, 0.008f, 0, 0);
    cat.register_operator("V40", 0.3f, 0.007f, 0, 0);
    cat.register_operator("V42", 0.3f, 0.006f, 0, 0);
    cat.register_operator("V43", 0.2f, 0.005f, 0, 0);
    cat.register_operator("V47", 0.1f, 0.003f, 0, 0);
    cat.register_operator("V48", 0.3f, 0.008f, 0, 0);
    cat.register_operator("V50", 0.2f, 0.006f, 0, 0);
    cat.register_operator("Base", 0.1f, 0.02f, 0, 0);
}

} // namespace tpch
} // namespace thunderduck
