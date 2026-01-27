/**
 * ThunderDuck - Hash Join v14.0 Implementation
 *
 * V14 策略：基于基准测试结果，委托给已验证最优的实现
 *
 * 测试结论：
 * - 两阶段算法（计数→预分配→填充）的额外计数遍历开销 > 收益
 * - V3 的 Radix 16 分区 + 单遍填充已是最优
 * - V10 的 SEMI/ANTI 优化有效
 *
 * V14 = V3 (INNER) + V10 (SEMI/ANTI)
 */

#include "thunderduck/join.h"

namespace thunderduck {
namespace join {

namespace {
const char* V14_VERSION = "V14.0 - 委托最优实现 (V3+V10)";
}

size_t hash_join_i32_v14(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type,
                          JoinResult* result) {
    if (!result) return 0;
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // SEMI/ANTI JOIN: 使用 V10 优化实现
    if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
        return hash_join_i32_v10(build_keys, build_count, probe_keys, probe_count,
                                  join_type, result);
    }

    // INNER JOIN: 使用 V3 最优实现
    // V3 特性：Radix 16 分区 + 1.7x 负载因子 + 单遍填充
    return hash_join_i32_v3(build_keys, build_count, probe_keys, probe_count,
                             join_type, result);
}

size_t hash_join_i32_v14_config(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinType join_type,
                                 JoinResult* result,
                                 const JoinConfig& config) {
    (void)config;
    return hash_join_i32_v14(build_keys, build_count, probe_keys, probe_count,
                              join_type, result);
}

const char* get_v14_version_info() {
    return V14_VERSION;
}

} // namespace join
} // namespace thunderduck
