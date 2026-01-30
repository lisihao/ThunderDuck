/**
 * ThunderDuck TPC-H Operators V54 Implementation
 *
 * @version 54
 * @date 2026-01-30
 */

#include "tpch_operators_v54.h"

namespace thunderduck {
namespace tpch {
namespace ops_v54 {

const char* V54_VERSION = "V54-NativeDouble";
const char* V54_DATE = "2026-01-30";

const char* V54_FEATURES[] = {
    "Native double columns (zero conversion)",
    "SIMD float32x4 operations",
    "8-thread parallel execution",
    "BitmapPredicateIndex pre-filtering"
};

} // namespace ops_v54
} // namespace tpch
} // namespace thunderduck
