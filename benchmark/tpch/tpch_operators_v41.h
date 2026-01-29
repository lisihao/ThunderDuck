/**
 * ThunderDuck TPC-H V41 优化算子
 *
 * 核心优化: Q21 单遍预计算 + 直接数组访问
 * - 消除排序开销
 * - O(1) 订单状态查找
 * - DynamicBitmapFilter 消除硬编码
 *
 * @version 41.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_generic_operators.h"
#include <vector>
#include <string>

namespace thunderduck {
namespace tpch {
namespace ops_v41 {

// ============================================================================
// Q21 优化器 V4 - 单遍预计算方案
// ============================================================================

/**
 * Q21 优化器 V4 - 单遍预计算 + 直接数组访问
 *
 * Q21 SQL 语义:
 * - 找到目标国家的供应商
 * - 他们有订单项延迟 (receiptdate > commitdate)
 * - 该订单是 'F' 状态
 * - EXISTS: 订单有其他供应商
 * - NOT EXISTS: 没有其他供应商也延迟
 *
 * 优化策略:
 * 1. 预计算每个订单的状态 (supplier_count, late_suppkeys)
 * 2. 使用直接数组访问 O(1) 替代二分查找 O(log n)
 * 3. 消除排序开销
 */
class Q21OptimizerV4 {
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
 * V41 Q21 查询入口
 */
void run_q21_v41(TPCHDataLoader& loader);

} // namespace ops_v41
} // namespace tpch
} // namespace thunderduck
