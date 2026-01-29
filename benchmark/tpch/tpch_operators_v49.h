/**
 * ThunderDuck TPC-H V49 - Top-N Aware Partial Aggregation
 *
 * 核心优化:
 * - 线程局部聚合 (消除原子操作竞争)
 * - 局部 Top-K heap (每线程维护 top-K)
 * - 全局增量合并 (无需分配大数组)
 *
 * @version 49.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_operators_v25.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <cstdint>

#ifdef __aarch64__
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v49 {

using ops_v25::ThreadPool;

/**
 * Q3 Top-N Aware Partial Aggregation
 *
 * 算法:
 * 1. 构建 BUILDING custkey bitmap
 * 2. 构建紧凑 Hash Table + Bloom Filter (同 V31)
 * 3. 线程局部聚合 (无原子操作)
 * 4. 每线程提取局部 Top-K
 * 5. 全局合并 (增量 heap)
 */
void run_q3_v49(TPCHDataLoader& loader);

} // namespace ops_v49
} // namespace tpch
} // namespace thunderduck
