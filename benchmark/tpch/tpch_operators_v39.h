/**
 * ThunderDuck TPC-H V39 优化算子
 *
 * 核心优化: 纯排序方案，完全避免 Hash Map
 * - Q21: 排序 + 单遍聚合
 * - Q20: 排序 + 单遍聚合
 *
 * @deprecated 专用类命名 (如 Q21OptimizerV3) 已废弃，请使用通用别名:
 *   - Q21OptimizerV3 → PureSortAggregateOptimizer
 *   - Q20OptimizerV4 → PureSortAggregateOptimizerV2
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <algorithm>

namespace thunderduck {
namespace tpch {
namespace ops_v39 {

// ============================================================================
// Q21 优化器 V3 - 纯排序方案
// ============================================================================

class Q21OptimizerV3 {
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

// ============================================================================
// Q20 优化器 V4 - 纯排序方案
// ============================================================================

class Q20OptimizerV4 {
public:
    struct Result {
        std::vector<std::pair<std::string, std::string>> suppliers;
    };

    static Result execute(
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        const std::vector<std::string>& s_address,
        size_t supplier_count,
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        const int32_t* p_partkey,
        const std::vector<std::string>& p_name,
        size_t part_count,
        const int32_t* ps_partkey,
        const int32_t* ps_suppkey,
        const int32_t* ps_availqty,
        size_t partsupp_count,
        const int32_t* l_partkey,
        const int32_t* l_suppkey,
        const int64_t* l_quantity,
        const int32_t* l_shipdate,
        size_t lineitem_count,
        const std::string& part_prefix,
        const std::string& target_nation,
        int32_t date_lo,
        int32_t date_hi,
        double quantity_factor
    );
};

void run_q21_v39(TPCHDataLoader& loader);
void run_q20_v39(TPCHDataLoader& loader);

// ============================================================================
// 通用别名 (推荐使用，取代查询专用命名)
// ============================================================================

using PureSortAggregateOptimizer = Q21OptimizerV3;
using PureSortAggregateOptimizerV2 = Q20OptimizerV4;

} // namespace ops_v39
} // namespace tpch
} // namespace thunderduck
