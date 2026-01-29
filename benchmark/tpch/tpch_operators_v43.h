/**
 * ThunderDuck TPC-H V43 优化算子
 *
 * 核心优化: Q17 位图过滤 + 两阶段聚合
 *
 * @version 43.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <string>

namespace thunderduck {
namespace tpch {
namespace ops_v43 {

/**
 * V43 Q17 查询入口 - 位图过滤 + 两阶段聚合
 *
 * 优化策略:
 * 1. 位图替代 unordered_set 快速过滤目标 parts
 * 2. 第一阶段: 计算每个 partkey 的 SUM(qty) 和 COUNT
 * 3. 第二阶段: 过滤 qty < 0.2*AVG 的行并累加 price
 * 4. 并行 + 线程局部聚合
 */
void run_q17_v43(TPCHDataLoader& loader);

}  // namespace ops_v43
}  // namespace tpch
}  // namespace thunderduck
