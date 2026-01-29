/**
 * ThunderDuck TPC-H 查询执行器
 *
 * 执行 TPC-H 查询并测量性能
 *
 * @version 1.0
 * @date 2026-01-28
 */

#ifndef TPCH_EXECUTOR_H
#define TPCH_EXECUTOR_H

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <map>

#include "tpch_data_loader.h"

namespace thunderduck {
namespace tpch {

// ============================================================================
// 查询结果结构
// ============================================================================

/**
 * 单个查询的执行结果
 */
struct QueryResult {
    std::string query_id;           // 查询 ID (Q1-Q22)
    double duckdb_ms = 0.0;         // DuckDB 执行时间 (ms)
    double thunderduck_ms = 0.0;    // ThunderDuck 执行时间 (ms)
    double speedup = 0.0;           // 加速比 (duckdb/thunderduck)
    bool correct = false;           // 结果是否正确
    std::string category;           // 查询类别 (A/B/C)
    std::string error_msg;          // 错误消息 (如果有)
    size_t result_rows = 0;         // 结果行数
};

/**
 * 完整的基准测试结果
 */
struct BenchmarkResult {
    int scale_factor = 1;
    std::string date;
    std::vector<QueryResult> queries;

    // 汇总统计
    double total_duckdb_ms = 0.0;
    double total_thunderduck_ms = 0.0;
    double geometric_mean_speedup = 0.0;
    int faster_count = 0;
    int slower_count = 0;
    int same_count = 0;
};

// ============================================================================
// TPC-H 查询分类
// ============================================================================

/**
 * 查询分类
 */
enum class QueryCategory {
    A,  // 完全可优化 - 预期加速 1.5-3x
    B,  // 部分可优化 - 预期加速 1.0-1.5x
    C   // DuckDB 回退 - 预期 ~1.0x
};

/**
 * 获取查询分类
 */
QueryCategory get_query_category(const std::string& query_id);

/**
 * 获取分类描述
 */
const char* get_category_description(QueryCategory cat);

// ============================================================================
// 查询执行器
// ============================================================================

/**
 * TPC-H 查询执行器
 *
 * 功能:
 * - 执行 DuckDB 基线查询
 * - 执行 ThunderDuck 优化查询
 * - 测量和比较性能
 * - 验证结果正确性
 */
class TPCHExecutor {
public:
    /**
     * 构造函数
     * @param loader 数据加载器 (已加载数据)
     * @param iterations 测量迭代次数 (默认 10)
     * @param warmup 预热次数 (默认 2)
     */
    TPCHExecutor(TPCHDataLoader& loader, size_t iterations = 10, size_t warmup = 2);

    /**
     * 运行单个查询
     * @param query_id 查询 ID (Q1-Q22)
     * @param verbose 是否输出详细信息
     * @return 查询结果
     */
    QueryResult run_query(const std::string& query_id, bool verbose = true);

    /**
     * 运行所有查询
     * @param verbose 是否输出详细信息
     * @return 完整基准测试结果
     */
    BenchmarkResult run_all_queries(bool verbose = true);

    /**
     * 运行指定类别的查询
     */
    BenchmarkResult run_category(QueryCategory category, bool verbose = true);

    /**
     * 设置迭代次数
     */
    void set_iterations(size_t iters) { iterations_ = iters; }
    void set_warmup(size_t warmup) { warmup_ = warmup; }

private:
    TPCHDataLoader& loader_;
    size_t iterations_;
    size_t warmup_;

    // IQR 中位数测量
    template<typename Func>
    double measure_median_iqr(Func&& func);

    // DuckDB 基线执行
    double run_duckdb_baseline(const std::string& query_id);

    // ThunderDuck 优化执行
    double run_thunderduck_optimized(const std::string& query_id);

    // 结果验证
    bool verify_results(const std::string& query_id);
};

// ============================================================================
// TPC-H SQL 查询定义
// ============================================================================

/**
 * 获取 TPC-H SQL 查询
 */
const std::string& get_tpch_sql(const std::string& query_id);

/**
 * 获取所有查询 ID 列表
 */
std::vector<std::string> get_all_query_ids();

/**
 * 获取指定类别的查询 ID 列表
 */
std::vector<std::string> get_category_query_ids(QueryCategory category);

// ============================================================================
// 查询实现接口
// ============================================================================

/**
 * 查询实现函数类型
 */
using QueryImplFunc = std::function<void(TPCHDataLoader&)>;

/**
 * 获取查询的 ThunderDuck 实现
 * @param query_id 查询 ID
 * @return 实现函数 (如果没有优化实现则返回空)
 */
QueryImplFunc get_thunderduck_impl(const std::string& query_id);

/**
 * 检查查询是否有 ThunderDuck 实现
 */
bool has_thunderduck_impl(const std::string& query_id);

} // namespace tpch
} // namespace thunderduck

#endif // TPCH_EXECUTOR_H
