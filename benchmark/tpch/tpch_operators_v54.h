/**
 * ThunderDuck TPC-H Operators V54
 *
 * 使用通用 NativeDoubleSIMDFilter 算子:
 * - 可复用的谓词系统
 * - 自动并行 SIMD 处理
 * - 零类型转换开销
 *
 * 适用场景:
 * - 单表扫描 + 多谓词过滤 + 聚合
 * - 目前仅 Q6 适用 (其他查询有 JOIN，不适用)
 *
 * @version 54
 * @date 2026-01-30
 */

#pragma once

#include "tpch_data_loader.h"
#include "../../include/thunderduck/generic_operators_v54.h"
#include <iostream>
#include <iomanip>

namespace thunderduck {
namespace tpch {
namespace ops_v54 {

// 引入通用算子
using namespace operators::v54;

// ============================================================================
// 版本信息
// ============================================================================

extern const char* V54_VERSION;
extern const char* V54_DATE;
extern const char* V54_FEATURES[];

// ============================================================================
// Q6: 使用通用 NativeDoubleSIMDFilter 算子
// ============================================================================

/**
 * Q6: 预测收入变化 (V54 - 通用算子版本)
 *
 * SELECT SUM(l_extendedprice * l_discount) AS revenue
 * FROM lineitem
 * WHERE l_shipdate >= DATE '1994-01-01'
 *   AND l_shipdate < DATE '1995-01-01'
 *   AND l_discount BETWEEN 0.05 AND 0.07
 *   AND l_quantity < 24
 *
 * 使用通用算子:
 * - NativeDoubleSIMDFilter 配置谓词
 * - 自动 8 线程并行
 * - SIMD 向量化过滤
 */
inline void run_q6_v54(TPCHDataLoader& loader) {
    const auto& lineitem = loader.lineitem();
    size_t n = lineitem.l_orderkey.size();

    // 检查是否有原生 double 列
    if (lineitem.l_discount_d.empty() || lineitem.l_quantity_d.empty() ||
        lineitem.l_extendedprice_d.empty()) {
        std::cerr << "V54 Error: Native double columns not available\n";
        return;
    }

    // 使用通用算子
    NativeDoubleSIMDFilter filter;

    // 配置谓词 (声明式)
    filter.add_int32_predicate(0, Predicate::ge(constants::dates::D1994_01_01));
    filter.add_int32_predicate(0, Predicate::lt(constants::dates::D1995_01_01));
    filter.add_double_predicate(0, Predicate::between(0.05, 0.07));  // discount BETWEEN
    filter.add_double_predicate(1, Predicate::lt(24.0));  // quantity < 24

    // 执行过滤 + 聚合
    auto result = filter.execute_with_aggregate(
        n,
        {lineitem.l_shipdate.data()},                      // int32 列
        {lineitem.l_discount_d.data(),                     // double 列 (谓词)
         lineitem.l_quantity_d.data()},
        AggregateSpec::sum_product(0, 1),                  // SUM(discount * extprice)
        {lineitem.l_discount_d.data(),                     // 聚合列
         lineitem.l_extendedprice_d.data()}
    );

    // 输出结果
    std::cout << "\n=== Q6 V54 Results (Generic NativeDoubleSIMDFilter) ===\n";
    std::cout << "Revenue: " << std::fixed << std::setprecision(2) << result.value << "\n";
    std::cout << "Matched: " << result.count << " / " << n
              << " (" << std::setprecision(2) << (100.0 * result.count / n) << "%)\n";
    std::cout << "Time: " << std::setprecision(2) << result.stats.total_time_ms
              << " ms (generic operator)\n";
}

// ============================================================================
// 算子适用性检查 (供优化器调用)
// ============================================================================

/**
 * 检查 V54 通用算子是否适用于指定查询
 *
 * @param query_id 查询 ID (Q1-Q22)
 * @param rows 数据行数
 * @param has_native_double 是否有原生 double 列
 * @return 是否适用
 */
inline bool is_v54_applicable(const std::string& query_id, size_t rows, bool has_native_double) {
    // V54 NativeDoubleSIMDFilter 仅适用于单表扫描查询
    // 目前仅 Q6 满足条件
    if (query_id == "Q6") {
        return NativeDoubleSIMDFilter::is_applicable(rows, 0, has_native_double);
    }
    return false;
}

/**
 * 估算 V54 执行时间 (毫秒)
 */
inline double estimate_v54_time_ms(const std::string& query_id, size_t rows) {
    if (query_id == "Q6") {
        // Q6: 4 个谓词
        return NativeDoubleSIMDFilter::estimate_time_ms(rows, 4);
    }
    return std::numeric_limits<double>::max();  // 不适用
}

} // namespace ops_v54
} // namespace tpch
} // namespace thunderduck
