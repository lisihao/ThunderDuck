/**
 * ThunderDuck TPC-H V50 - Q21 查询重写优化
 *
 * 核心思想: 将 EXISTS/NOT EXISTS 重写为预聚合 + JOIN
 *
 * 原始 Q21:
 *   EXISTS (l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
 *   NOT EXISTS (l3... AND l3.l_receiptdate > l3.l_commitdate)
 *
 * 重写为:
 *   预聚合: 每个 orderkey 的 (total_suppliers, late_suppliers)
 *   条件: total_suppliers > 1 AND late_suppliers == 1
 *
 * @deprecated 专用类命名 (如 Q21RewriteOptimizer) 已废弃，请使用通用别名:
 *   - Q21RewriteOptimizer → ExistsRewriteOptimizer
 *
 * @version 50.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_operators_v25.h"  // ThreadPool
#include "tpch_constants.h"      // 统一常量
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace thunderduck {
namespace tpch {
namespace ops_v50 {

using ops_v25::ThreadPool;

// ============================================================================
// Q21 重写优化器 - 预聚合 + JOIN 方案
// ============================================================================

/**
 * Q21RewriteOptimizer
 *
 * 算法:
 * 1. 预聚合: 扫描 lineitem，计算每个 orderkey 的供应商统计
 *    - total_suppliers: 不同供应商数量
 *    - late_suppliers: 迟到的不同供应商数量
 *
 * 2. 过滤: 只保留满足条件的 orderkey
 *    - total_suppliers > 1 (EXISTS: 有其他供应商)
 *    - late_suppliers == 1 (NOT EXISTS: 只有一个迟到)
 *
 * 3. JOIN: supplier × lineitem × orders × nation
 *    - 使用位图快速过滤
 *
 * 4. 聚合: GROUP BY s_name, COUNT(*)
 */
class Q21RewriteOptimizer {
public:
    struct Config {
        std::string target_nation = constants::nations::SAUDI_ARABIA;
        int8_t order_status = 0;  // 'F'
        size_t limit = 100;
    };

    struct Result {
        std::vector<std::pair<std::string, int64_t>> suppliers;
    };

    static Result execute(
        // Supplier
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        size_t supplier_count,
        // Lineitem
        const int32_t* l_orderkey,
        const int32_t* l_suppkey,
        const int32_t* l_commitdate,
        const int32_t* l_receiptdate,
        size_t lineitem_count,
        // Orders
        const int32_t* o_orderkey,
        const int8_t* o_orderstatus,
        size_t orders_count,
        // Nation
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // Config
        const Config& config
    );
};

/**
 * V50 Q21 入口
 */
void run_q21_v50(TPCHDataLoader& loader);

// ============================================================================
// 通用别名 (推荐使用，取代查询专用命名)
// ============================================================================

using ExistsRewriteOptimizer = Q21RewriteOptimizer;

} // namespace ops_v50
} // namespace tpch
} // namespace thunderduck
