/**
 * ThunderDuck TPC-H V45 优化算子
 *
 * 直接数组优化 - 消除 Hash 探测开销
 *
 * Q14: 直接数组 part_is_promo[partkey] (200KB L2 cache)
 * Q11: 位图 germany_supps[suppkey/8] + 直接数组聚合 (1.25KB)
 * Q5:  直接数组 supp_nation[suppkey], cust_nation[custkey]
 *
 * @version 45.0
 * @date 2026-01-29
 */

#ifndef TPCH_OPERATORS_V45_H
#define TPCH_OPERATORS_V45_H

#include "tpch_data_loader.h"

namespace thunderduck {
namespace tpch {
namespace ops_v45 {

/**
 * Q14 V45: 直接数组优化
 *
 * 优化点:
 * - part_is_promo[partkey] 直接数组 (200KB)
 * - 消除 WeakHashTable 探测开销
 * - 并行扫描 lineitem
 */
void run_q14_v45(TPCHDataLoader& loader);

/**
 * Q11 V45: 位图 + 直接数组聚合
 *
 * 优化点:
 * - germany_bitmap[suppkey/8] 位图 (~1.25KB)
 * - 直接数组聚合 partkey_value[partkey]
 * - 并行单遍扫描
 */
void run_q11_v45(TPCHDataLoader& loader);

/**
 * Q5 V45: 直接数组维度表
 *
 * 优化点:
 * - supp_nation[suppkey] 直接数组 (~40KB)
 * - cust_nation[custkey] 直接数组 (~600KB)
 * - 保持 order_to_cust 紧凑 Hash (orderkey 范围太大)
 */
void run_q5_v45(TPCHDataLoader& loader);

}  // namespace ops_v45
}  // namespace tpch
}  // namespace thunderduck

#endif  // TPCH_OPERATORS_V45_H
