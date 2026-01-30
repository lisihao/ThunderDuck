/**
 * ThunderDuck TPC-H Operators V51 Implementation
 *
 * 通用算子实现:
 * - ParallelRadixSort
 * - PartitionedAggregation
 * - FusedFilterAggregate
 *
 * @version 51
 * @date 2026-01-29
 */

#include "tpch_operators_v51.h"
#include "tpch_constants.h"

// 所有实现都在头文件中 (模板 + inline)
// 这个文件用于确保编译单元存在

namespace thunderduck {
namespace tpch {
namespace ops_v51 {

// 版本标识
const char* V51_VERSION = "V51-GenericOperators";
const char* V51_DATE = "2026-01-29";

// 算子信息
const char* V51_OPERATORS[] = {
    "ParallelRadixSort - 两级并行基数排序",
    "PartitionedAggregation - 分区聚合",
    "FusedFilterAggregate - Filter-Aggregate 融合"
};

} // namespace ops_v51
} // namespace tpch
} // namespace thunderduck
