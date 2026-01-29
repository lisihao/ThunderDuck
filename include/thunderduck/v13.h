/**
 * ThunderDuck V13 - 极致优化版本
 *
 * 核心优化:
 * - P0: Hash Join 两阶段算法 (0.06x → 1.5x+)
 * - P1: GROUP BY GPU 无原子优化 (0.78x → 2.0x+)
 * - P3: TopK GPU 并行版本 (新功能)
 *
 * 集成 V12.5 所有最优实现 + P0/P1/P3 优化
 */

#ifndef THUNDERDUCK_V13_H
#define THUNDERDUCK_V13_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v13 {

// ============================================================================
// V13 配置常量
// ============================================================================

constexpr size_t FILTER_GPU_THRESHOLD = 5000000;      // 5M
constexpr size_t AGGREGATE_GPU_THRESHOLD = 5000000;   // 5M
constexpr size_t TOPK_GPU_THRESHOLD = 1000000;        // 1M
constexpr size_t JOIN_MATCH_RATE_THRESHOLD = 200000;  // 200K
constexpr size_t GROUP_GPU_THRESHOLD = 100000;        // 100K

// ============================================================================
// 执行统计
// ============================================================================

struct ExecutionStats {
    const char* operator_name;
    const char* version_used;
    const char* device_used;
    size_t data_count;
    double throughput_gbps;
    double elapsed_ms;
};

// ============================================================================
// Filter - 继承 V12.5 自适应策略
// ============================================================================

enum class CompareOp { EQ, NE, LT, LE, GT, GE };

size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats = nullptr);

// ============================================================================
// Aggregate - 继承 V12.5 自适应策略
// ============================================================================

struct AggregateResult {
    int64_t sum;
    int64_t count;
    int32_t min_val;
    int32_t max_val;
    double avg;
};

int64_t sum_i32(const int32_t* input, size_t count,
                ExecutionStats* stats = nullptr);

void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max,
                ExecutionStats* stats = nullptr);

// ============================================================================
// GROUP BY - P1 优化: GPU 无原子版本
// ============================================================================

/**
 * V13 分组求和
 *
 * 策略:
 * - count >= 100K 且 num_groups <= 1024: GPU V13 分区聚合
 * - 否则: CPU V8 4核并行
 */
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);

void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats = nullptr);

// ============================================================================
// TopK - P3 优化: GPU 并行版本
// ============================================================================

/**
 * V13 TopK Max
 *
 * 策略:
 * - count >= 1M 且 k <= 100: GPU V13 并行选择
 * - count < 1M: CPU V8 Count-Based
 * - count >= 1M: CPU V7 Sampling
 */
size_t topk_max_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

size_t topk_min_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

// ============================================================================
// Hash Join - P0 优化: 两阶段算法
// ============================================================================

enum class JoinType { INNER, LEFT, RIGHT, FULL, SEMI, ANTI };

struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

JoinResult* create_join_result(size_t initial_capacity);
void free_join_result(JoinResult* result);

/**
 * V13 Hash Join - 两阶段算法
 *
 * 优化:
 * - Phase 1: 计数遍历，统计总匹配数
 * - Phase 2: 预分配精确容量，一次填充
 * - 消除 grow_join_result() 动态扩容开销
 *
 * 目标: 0.06x → 1.5x+
 */
size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type, JoinResult* result,
                     ExecutionStats* stats = nullptr);

// ============================================================================
// 版本信息
// ============================================================================

const char* get_version_info();
const char* get_optimal_versions();
bool is_gpu_available();

} // namespace v13
} // namespace thunderduck

#endif // THUNDERDUCK_V13_H
