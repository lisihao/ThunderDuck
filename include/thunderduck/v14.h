/**
 * ThunderDuck V14 - 深度优化版本
 *
 * 核心优化:
 * - P0: Hash Join 两阶段算法 (4.28x → 8x+)
 *   - 32 分区 (5-bit radix)
 *   - 1.5x 负载因子
 *   - 两阶段: 计数 → 预分配 → 填充
 *
 * - P1: GROUP BY 寄存器缓冲 + 多路分流 (2.66x → 4x+)
 *   - 8 个寄存器缓存热分组
 *   - 4 路分流并行累加
 *   - SIMD 合并结果
 *
 * 集成 V13 所有最优实现 + V14 优化
 */

#ifndef THUNDERDUCK_V14_H
#define THUNDERDUCK_V14_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v14 {

// ============================================================================
// V14 配置常量
// ============================================================================

constexpr size_t FILTER_GPU_THRESHOLD = 5000000;      // 5M
constexpr size_t AGGREGATE_GPU_THRESHOLD = 5000000;   // 5M
constexpr size_t TOPK_GPU_THRESHOLD = 1000000;        // 1M
constexpr size_t JOIN_PARALLEL_THRESHOLD = 50000;     // 50K
constexpr size_t GROUP_PARALLEL_THRESHOLD = 100000;   // 100K
constexpr size_t GROUP_REGBUF_THRESHOLD = 64;         // 寄存器缓冲阈值

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
// Filter - 继承 V13 自适应策略
// ============================================================================

enum class CompareOp { EQ, NE, LT, LE, GT, GE };

size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats = nullptr);

// ============================================================================
// Aggregate - 继承 V13 自适应策略
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
// GROUP BY - V14 优化: 寄存器缓冲 + 多路分流
// ============================================================================

/**
 * V14 分组求和
 *
 * 策略选择:
 * - count < 100K: 单线程
 *   - num_groups <= 64: 寄存器缓冲累加 (8 个热缓存)
 *   - num_groups > 64: V4 直接累加
 * - count >= 100K: 4 核并行
 *   - 每线程局部累加 + SIMD 合并
 *
 * 目标: 2.66x → 4x+
 */
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);

void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats = nullptr);

void group_min_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int32_t* out_mins,
                   ExecutionStats* stats = nullptr);

void group_max_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int32_t* out_maxs,
                   ExecutionStats* stats = nullptr);

// ============================================================================
// TopK - 继承 V13 策略
// ============================================================================

size_t topk_max_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

size_t topk_min_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

// ============================================================================
// Hash Join - V14 优化: 两阶段预分配
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
 * V14 Hash Join - 两阶段预分配算法
 *
 * 核心优化:
 * - Phase 1: 并行计数 (无写入)
 * - Phase 2: 一次性精确分配
 * - Phase 3: 并行填充 (无动态扩容)
 *
 * 参数优化:
 * - 32 分区 (5-bit radix) vs V3 的 16 分区
 * - 1.5x 负载因子 vs V3 的 1.7x
 *
 * 目标: 4.28x → 8x+
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

// ============================================================================
// 底层接口 (高级用户)
// ============================================================================

namespace detail {

// Hash Join V14 直接调用
size_t hash_join_i32_v14(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result);

// GROUP BY V14 直接调用
void group_sum_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v14(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts);

void group_min_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs);

// 寄存器缓冲版本 (低基数专用)
void group_sum_i32_v14_parallel(const int32_t* values, const uint32_t* groups,
                                 size_t count, size_t num_groups, int64_t* out_sums);

} // namespace detail

} // namespace v14
} // namespace thunderduck

#endif // THUNDERDUCK_V14_H
