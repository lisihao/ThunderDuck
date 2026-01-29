/**
 * ThunderDuck TPC-H V44 优化算子
 *
 * 核心优化: Q3 直接数组访问 + 线程局部聚合
 *
 * @version 44.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <string>

namespace thunderduck {
namespace tpch {
namespace ops_v44 {

/**
 * V44 Q3 查询入口 - 直接数组访问 + 线程局部聚合
 *
 * 优化策略 (对比 V31):
 * 1. 直接数组访问 O(1) 替代 Hash Table O(1+探测)
 * 2. 线程局部聚合避免 atomic 操作
 * 3. 消除 Bloom Filter 计算开销
 * 4. 预计算 order_info[orderkey] 结构
 */
void run_q3_v44(TPCHDataLoader& loader);

}  // namespace ops_v44
}  // namespace tpch
}  // namespace thunderduck
