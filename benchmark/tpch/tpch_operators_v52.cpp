/**
 * ThunderDuck TPC-H Operators V52 Implementation
 *
 * 编译单元
 *
 * @version 52
 * @date 2026-01-29
 */

#include "tpch_operators_v52.h"

namespace thunderduck {
namespace tpch {
namespace ops_v52 {

// 版本标识
const char* V52_VERSION = "V52-AdvancedOperators";
const char* V52_DATE = "2026-01-29";

// 算子信息
const char* V52_OPERATORS[] = {
    "DirectArrayJoin - O(1) 数组直接索引",
    "SIMDBranchlessFilter - 无分支 SIMD 多条件过滤",
    "BitmapPredicateIndex - 64-bit 位图谓词索引"
};

} // namespace ops_v52
} // namespace tpch
} // namespace thunderduck
