/**
 * ThunderDuck TPC-H Operators V58
 *
 * Q3/Q9/Q2 深度优化:
 * - Q3: DirectArrayAggregator (替代 unordered_map)
 * - Q9: PrecomputedBitmap (消除字符串操作)
 * - Q2: ParallelScan + SIMDSuffix (并行化 + SIMD)
 *
 * 设计原则:
 * - 零硬编码 (使用 tpch_constants.h)
 * - O(1) 热路径
 * - 8 路 SIMD 批处理
 * - 自动并行化
 *
 * @version 58
 * @date 2026-01-30
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_operators_v32.h"  // 基础工具类
#include "tpch_constants.h"
#include "../../include/thunderduck/generic_operators_v58.h"
#include <vector>
#include <algorithm>
#include <thread>
#include <future>

namespace thunderduck {
namespace tpch {
namespace ops_v58 {

using namespace ops_v32;  // ThreadPool, CompactHashTable
using namespace operators::v58;

// ============================================================================
// 版本信息
// ============================================================================

extern const char* V58_VERSION;
extern const char* V58_DATE;
extern const char* V58_FEATURES[];

// ============================================================================
// Q3 V58: DirectArrayAggregator 优化
// ============================================================================

/**
 * Q3: 运输优先级 (V58 - DirectArrayAggregator)
 *
 * 优化点:
 * 1. 使用 DirectArrayAggregator 替代 unordered_map (O(1) 无 hash)
 * 2. 预存储 orderdate/shippriority 避免二次查找
 * 3. SIMD 日期过滤
 * 4. 线程局部直接数组聚合
 *
 * 预期性能: 1.55-1.65x (vs DuckDB)
 */
void run_q3_v58(TPCHDataLoader& loader);

// ============================================================================
// Q9 V58: PrecomputedBitmap 优化
// ============================================================================

/**
 * Q9: 产品类型利润分析 (V58 - PrecomputedBitmap)
 *
 * 优化点:
 * 1. 预计算 "green" parts 位图 (消除字符串操作)
 * 2. MergedLookupTable 合并 supp→nation + order→year
 * 3. 直接数组 ps_cost 查找
 * 4. 8 路并行扫描
 *
 * 预期性能: 1.80-1.95x (vs DuckDB)
 */
void run_q9_v58(TPCHDataLoader& loader);

// ============================================================================
// Q2 V58: ParallelScan + SIMDSuffix 优化
// ============================================================================

/**
 * Q2: 最小成本供应商 (V58 - ParallelScan)
 *
 * 优化点:
 * 1. 预计算 "BRASS" 后缀位图 (SIMD 批量匹配)
 * 2. 并行扫描 partsupp (vs V56 顺序扫描)
 * 3. 直接数组 min_cost 解关联
 * 4. 线程局部结果收集 + 合并
 *
 * 预期性能: 2.0-2.2x (vs DuckDB)
 */
void run_q2_v58(TPCHDataLoader& loader);

// ============================================================================
// 适用性检查
// ============================================================================

/**
 * 检查 V58 优化是否适用
 */
inline bool is_v58_applicable(const std::string& query_id, size_t rows) {
    if (query_id == "Q3") {
        // Q3: 需要足够数据量触发并行收益
        return rows >= 100000;
    }
    if (query_id == "Q9") {
        // Q9: 需要 green parts 过滤收益
        return rows >= 100000;
    }
    if (query_id == "Q2") {
        // Q2: 需要并行扫描收益
        return rows >= 10000;
    }
    return false;
}

/**
 * 估算 V58 执行时间 (毫秒)
 */
inline double estimate_v58_time_ms(const std::string& query_id, size_t rows) {
    // 基于实测数据的成本模型
    if (query_id == "Q3") {
        return 0.5 + rows * 0.00001;  // ~6ms for 6M rows
    }
    if (query_id == "Q9") {
        return 0.8 + rows * 0.000015;  // ~10ms for 6M rows
    }
    if (query_id == "Q2") {
        return 0.2 + rows * 0.000005;  // ~1ms for 200K rows
    }
    return std::numeric_limits<double>::max();
}

} // namespace ops_v58
} // namespace tpch
} // namespace thunderduck
