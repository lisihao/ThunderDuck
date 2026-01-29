/**
 * ThunderDuck TPC-H 查询实现
 *
 * 使用 ThunderDuck V22 最优算子实现 TPC-H 查询
 *
 * V22 算子配置:
 * - Filter: V19 (8线程并行)
 * - INNER JOIN: V19.2 (激进预取 + SIMD)
 * - SEMI JOIN: GPU Metal
 * - GROUP BY: V15 (8线程 + 8路展开)
 * - SUM: V21 (8路展开)
 *
 * @version 2.2 (V22)
 * @date 2026-01-28
 */

#ifndef TPCH_QUERIES_H
#define TPCH_QUERIES_H

#include "tpch_data_loader.h"
#include "tpch_operators.h"
#include "tpch_executor.h"

namespace thunderduck {
namespace tpch {
namespace queries {

// ============================================================================
// Category A: 完全可优化的查询 (预期 1.5-3x 加速)
// ============================================================================

/**
 * Q1: 定价汇总报告
 *
 * SELECT l_returnflag, l_linestatus, SUM(...), AVG(...), COUNT(*)
 * FROM lineitem WHERE l_shipdate <= threshold
 * GROUP BY l_returnflag, l_linestatus
 *
 * 优化: SIMD Filter + V15 GROUP BY + 多路聚合
 */
void run_q1(TPCHDataLoader& loader);

/**
 * Q3: 运输优先级
 *
 * SELECT l_orderkey, SUM(revenue), o_orderdate, o_shippriority
 * FROM customer, orders, lineitem
 * WHERE ... GROUP BY ...
 *
 * 优化: V14 Hash Join + GROUP BY
 */
void run_q3(TPCHDataLoader& loader);

/**
 * Q5: 本地供应商收入
 *
 * SELECT n_name, SUM(revenue) FROM ... WHERE ... GROUP BY n_name
 *
 * 优化: 多表 V14 Join + GROUP BY
 */
void run_q5(TPCHDataLoader& loader);

/**
 * Q6: 预测收入变化 - 最佳优化场景
 *
 * SELECT SUM(l_extendedprice * l_discount)
 * FROM lineitem
 * WHERE l_shipdate >= '1994-01-01'
 *   AND l_shipdate < '1995-01-01'
 *   AND l_discount BETWEEN 0.05 AND 0.07
 *   AND l_quantity < 24
 *
 * 优化: SIMD 单遍过滤聚合
 */
void run_q6(TPCHDataLoader& loader);

/**
 * Q7: 体量运输
 *
 * 优化: 多表 Join + GROUP BY
 */
void run_q7(TPCHDataLoader& loader);

/**
 * Q9: 产品类型利润
 *
 * 优化: 复杂 Join + GROUP BY
 */
void run_q9(TPCHDataLoader& loader);

/**
 * Q10: 退货报告
 *
 * 优化: Join + GROUP BY + Filter
 */
void run_q10(TPCHDataLoader& loader);

/**
 * Q12: 运输模式与订单优先级
 *
 * 优化: Join + CASE + GROUP BY
 */
void run_q12(TPCHDataLoader& loader);

/**
 * Q14: 促销效果
 *
 * 优化: Join + 条件聚合
 */
void run_q14(TPCHDataLoader& loader);

/**
 * Q18: 大批量客户
 *
 * 优化: GROUP BY + HAVING + Join
 */
void run_q18(TPCHDataLoader& loader);

// ============================================================================
// Category B: 部分可优化的查询 (预期 1.0-1.5x 加速)
// ============================================================================

/**
 * Q2: 最小成本供应商 - 包含子查询
 */
void run_q2(TPCHDataLoader& loader);

/**
 * Q4: 订单优先级检查 - 包含 EXISTS 子查询
 */
void run_q4(TPCHDataLoader& loader);

/**
 * Q11: 重要库存识别 - 包含 HAVING 子查询
 */
void run_q11(TPCHDataLoader& loader);

/**
 * Q15: 顶级供应商 - 包含 CTE
 */
void run_q15(TPCHDataLoader& loader);

/**
 * Q16: 零件供应商关系 - 包含 NOT IN 子查询
 */
void run_q16(TPCHDataLoader& loader);

/**
 * Q19: 折扣收入 - 复杂 OR 条件
 */
void run_q19(TPCHDataLoader& loader);

// ============================================================================
// Category C: DuckDB 回退的查询 (使用 DuckDB 原生执行)
// ============================================================================

// Q8, Q13, Q17, Q20, Q21, Q22 - 使用 DuckDB 执行
// 这些查询包含复杂子查询、EXISTS/NOT EXISTS 等，
// 暂时使用 DuckDB 原生执行

// ============================================================================
// 查询注册
// ============================================================================

/**
 * 注册所有查询实现
 * 在 tpch_executor.cpp 中调用
 */
void register_all_queries();

/**
 * 获取查询实现
 */
QueryImplFunc get_query_impl(const std::string& query_id);

/**
 * 检查是否有优化实现
 */
bool has_optimized_impl(const std::string& query_id);

} // namespace queries
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_QUERIES_H
