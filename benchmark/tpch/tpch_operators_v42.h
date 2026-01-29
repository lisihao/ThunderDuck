/**
 * ThunderDuck TPC-H V42 优化算子
 *
 * 核心优化: Q8 并行化 + 线程局部聚合
 *
 * @version 42.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <string>
#include <array>

namespace thunderduck {
namespace tpch {
namespace ops_v42 {

/**
 * V42 Q8 查询入口 - 并行化版本
 */
void run_q8_v42(TPCHDataLoader& loader);

}  // namespace ops_v42
}  // namespace tpch
}  // namespace thunderduck
