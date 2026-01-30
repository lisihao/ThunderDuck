/**
 * ThunderDuck TPC-H Operators V53 Implementation
 *
 * @version 53
 * @date 2026-01-29
 */

#include "tpch_operators_v53.h"

namespace thunderduck {
namespace tpch {
namespace ops_v53 {

// 版本标识
const char* V53_VERSION = "V53-QueryMemoryInfra";
const char* V53_DATE = "2026-01-29";

// 架构特点
const char* V53_FEATURES[] = {
    "QueryArena - Query-scoped bump allocator",
    "ChunkedDirectArray - 64K chunk O(1) lookup",
    "TypeLiftedColumn - Scan-time type conversion",
    "BitmapPredicateIndex - 64-bit predicate bitmap"
};

} // namespace ops_v53
} // namespace tpch
} // namespace thunderduck
