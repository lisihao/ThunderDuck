/**
 * ThunderDuck TPC-H V40 优化算子
 *
 * 核心优化: 通用算子框架 + 消除硬编码
 * - Q20: 使用 DynamicBitmapFilter 消除 valid_suppkey(10001) 硬编码
 * - 使用 SortedGroupByAggregator 进行排序后聚合
 * - 使用 MergeJoinOperator 进行归并连接
 *
 * @deprecated 专用类命名 (如 Q20OptimizerV5) 已废弃，请使用通用别名:
 *   - Q20OptimizerV5 → DynamicBitmapFilterOptimizer
 *
 * @version 40.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_generic_operators.h"
#include <vector>
#include <string>

namespace thunderduck {
namespace tpch {
namespace ops_v40 {

// ============================================================================
// Q20 优化器 V5 - 通用算子框架版本
// ============================================================================

/**
 * Q20 优化器 V5 - 使用通用算子框架
 *
 * 相比 V4 的改进:
 * - 使用 DynamicBitmapFilter 替代固定大小 bitmap (消除 10001 硬编码)
 * - 使用 SortedGroupByAggregator 模板替代手写聚合循环
 * - 使用 MergeJoinOperator 模板替代手写双指针
 *
 * 执行流程:
 * 1. DynamicBitmapFilter 构建 forest% parts (自动检测范围)
 * 2. 收集 lineitem (partkey, suppkey, qty) + 排序
 * 3. SortedGroupByAggregator 单遍 SUM(qty)
 * 4. 收集 partsupp + 排序
 * 5. MergeJoin + 阈值比较 → matching_suppkeys
 * 6. DynamicBitmapFilter 构建 valid_suppkeys (无硬编码)
 * 7. 过滤 supplier by nation + IN valid_suppkeys
 */
class Q20OptimizerV5 {
public:
    struct Result {
        std::vector<std::pair<std::string, std::string>> suppliers;
    };

    /**
     * 执行 Q20 查询
     *
     * @param s_suppkey 供应商主键
     * @param s_nationkey 供应商国家键
     * @param s_name 供应商名称
     * @param s_address 供应商地址
     * @param supplier_count 供应商数量
     * @param n_nationkey 国家键
     * @param n_name 国家名称
     * @param nation_count 国家数量
     * @param p_partkey 零件键
     * @param p_name 零件名称
     * @param part_count 零件数量
     * @param ps_partkey partsupp 零件键
     * @param ps_suppkey partsupp 供应商键
     * @param ps_availqty partsupp 可用数量
     * @param partsupp_count partsupp 数量
     * @param l_partkey lineitem 零件键
     * @param l_suppkey lineitem 供应商键
     * @param l_quantity lineitem 数量
     * @param l_shipdate lineitem 发货日期
     * @param lineitem_count lineitem 数量
     * @param part_prefix 零件名称前缀 (如 "forest")
     * @param target_nation 目标国家名称 (如 "CANADA")
     * @param date_lo 日期范围下界 (epoch days)
     * @param date_hi 日期范围上界 (epoch days)
     * @param quantity_factor 数量因子 (如 0.5)
     */
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

/**
 * V40 Q20 查询入口
 */
void run_q20_v40(TPCHDataLoader& loader);

// ============================================================================
// 通用别名 (推荐使用，取代查询专用命名)
// ============================================================================

using DynamicBitmapFilterOptimizer = Q20OptimizerV5;

} // namespace ops_v40
} // namespace tpch
} // namespace thunderduck
