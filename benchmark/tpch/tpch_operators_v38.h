/**
 * ThunderDuck TPC-H V38 优化算子
 *
 * 核心优化:
 * - Q21: 排序去重替代嵌套 Hash Set
 * - Q20: 紧凑编码 + 预过滤
 *
 * @deprecated 专用类命名 (如 Q21OptimizerV2) 已废弃，请使用通用别名:
 *   - Q21OptimizerV2 → SortDeduplicateJoinOptimizer
 *   - Q20OptimizerV3 → CompactEncodingOptimizer
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace thunderduck {
namespace tpch {
namespace ops_v38 {

// ============================================================================
// Q21 优化器 V2 - 排序去重方案
// ============================================================================

class Q21OptimizerV2 {
public:
    struct Result {
        std::vector<std::pair<std::string, int64_t>> suppliers;  // (s_name, numwait)
    };

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
        // 参数
        const std::string& target_nation,
        size_t limit = 100
    );
};

// ============================================================================
// Q20 优化器 V3 - 紧凑编码方案
// ============================================================================

class Q20OptimizerV3 {
public:
    struct Result {
        std::vector<std::pair<std::string, std::string>> suppliers;  // (s_name, s_address)
    };

    static Result execute(
        // Supplier 表
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        const std::vector<std::string>& s_address,
        size_t supplier_count,
        // Nation 表
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_name,
        size_t part_count,
        // PartSupp 表
        const int32_t* ps_partkey,
        const int32_t* ps_suppkey,
        const int32_t* ps_availqty,
        size_t partsupp_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int32_t* l_suppkey,
        const int64_t* l_quantity,
        const int32_t* l_shipdate,
        size_t lineitem_count,
        // 参数
        const std::string& part_prefix,
        const std::string& target_nation,
        int32_t date_lo,
        int32_t date_hi,
        double quantity_factor
    );

private:
    // 紧凑编码: partkey(18位) + suppkey(14位) = 32位
    static inline uint32_t encode_key(int32_t partkey, int32_t suppkey) {
        return (static_cast<uint32_t>(partkey) << 14) |
               (static_cast<uint32_t>(suppkey) & 0x3FFF);
    }
};

// ============================================================================
// V38 查询入口
// ============================================================================

void run_q21_v38(TPCHDataLoader& loader);
void run_q20_v38(TPCHDataLoader& loader);

// ============================================================================
// 通用别名 (推荐使用，取代查询专用命名)
// ============================================================================

using SortDeduplicateJoinOptimizer = Q21OptimizerV2;
using CompactEncodingOptimizer = Q20OptimizerV3;

} // namespace ops_v38
} // namespace tpch
} // namespace thunderduck
