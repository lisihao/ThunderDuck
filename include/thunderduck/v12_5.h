/**
 * ThunderDuck V12.5 - 性能之选
 *
 * 集成各算子历史最优实现，零路由开销，极致性能
 *
 * | 算子          | 小数据(<5M)   | 大数据(>=5M)  | 最优版本 |
 * |---------------|---------------|---------------|----------|
 * | Filter        | GPU (7.54x)   | CPU V3 (3.02x)| 自适应   |
 * | Aggregate     | V9 (5.83x)    | V7 (3.01x)    | 自适应   |
 * | TopK          | V8 (13.36x)   | V7 (5.12x)    | 直调     |
 * | GROUP BY      | V8 (4.47x)    | V8 (2.32x)    | V8固定   |
 * | Hash Join     | 自适应        | 自适应        | 匹配率   |
 *
 * V12.5 相比 V12 改进:
 * - TopK: 消除路由开销，8.97x → 13.36x (+49%)
 * - Filter 10M: 使用 CPU V3，2.70x → 3.02x (+12%)
 * - GROUP BY: 直调 V8，4.11x → 4.47x (+9%)
 */

#ifndef THUNDERDUCK_V12_5_H
#define THUNDERDUCK_V12_5_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v125 {

// ============================================================================
// V12.5 配置常量
// ============================================================================

/** 自适应策略阈值 - 基于基准测试调优 */
constexpr size_t FILTER_GPU_THRESHOLD = 5000000;      // 5M: GPU/CPU 切换点
constexpr size_t AGGREGATE_GPU_THRESHOLD = 5000000;   // 5M: GPU/CPU 切换点
constexpr size_t TOPK_SAMPLING_THRESHOLD = 1000000;   // 1M: Count/Sampling 切换点
constexpr size_t JOIN_MATCH_RATE_THRESHOLD = 200000;  // 200K: 匹配率判断阈值

// ============================================================================
// 执行统计
// ============================================================================

struct ExecutionStats {
    const char* operator_name;    // 算子名称
    const char* version_used;     // 使用的版本
    const char* device_used;      // 使用的设备
    size_t data_count;            // 数据量
    double throughput_gbps;       // 吞吐量 (GB/s)
    double elapsed_ms;            // 执行时间 (ms)
};

// ============================================================================
// Filter - 自适应 CPU/GPU
// ============================================================================

enum class CompareOp { EQ, NE, LT, LE, GT, GE };

/**
 * V12.5 统一 Filter
 *
 * 策略:
 * - count < 5M: GPU Metal (7.54x on 1M)
 * - count >= 5M: CPU V3 SIMD (3.02x on 10M)
 */
size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats = nullptr);

/**
 * V12.5 统一 Count
 */
size_t count_i32(const int32_t* input, size_t count,
                 CompareOp op, int32_t value,
                 ExecutionStats* stats = nullptr);

// ============================================================================
// Aggregate - 自适应 CPU/GPU
// ============================================================================

struct AggregateResult {
    int64_t sum;
    int64_t count;
    int32_t min_val;
    int32_t max_val;
    double avg;
};

/**
 * V12.5 统一 SUM
 *
 * 策略:
 * - count < 5M: V9 CPU SIMD+ (5.83x on 1M)
 * - count >= 5M: V7 GPU Metal (3.01x on 10M)
 */
int64_t sum_i32(const int32_t* input, size_t count,
                ExecutionStats* stats = nullptr);

/**
 * V12.5 统一 MIN/MAX
 */
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max,
                ExecutionStats* stats = nullptr);

/**
 * V12.5 统一全统计
 */
AggregateResult aggregate_all_i32(const int32_t* input, size_t count,
                                   ExecutionStats* stats = nullptr);

// ============================================================================
// GROUP BY - V8 CPU 4核并行 (最优)
// ============================================================================

/**
 * V12.5 分组求和
 *
 * 策略: 始终使用 V8 CPU 4核并行 (4.47x on 1M, 2.32x on 10M)
 */
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);

void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats = nullptr);

void group_min_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_mins,
                   ExecutionStats* stats = nullptr);

void group_max_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_maxs,
                   ExecutionStats* stats = nullptr);

// ============================================================================
// TopK - 直调最优实现 (零路由开销)
// ============================================================================

/**
 * V12.5 TopK Max - 直调实现
 *
 * 策略:
 * - count < 1M: V8 Count-Based 直调 (13.36x)
 * - count >= 1M: V7 Sampling 直调 (5.12x)
 *
 * 相比 V12: 消除路由开销，8.97x → 13.36x (+49%)
 */
size_t topk_max_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

/**
 * V12.5 TopK Min - 直调实现
 */
size_t topk_min_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices = nullptr,
                    ExecutionStats* stats = nullptr);

// ============================================================================
// Hash Join - 匹配率自适应
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
 * V12.5 Hash Join - 匹配率自适应
 *
 * 策略:
 * - 低匹配率 (build < 200K, probe > build*3): V7 Adaptive
 * - 高匹配率: V11 SIMD (预取+展开)
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
bool is_npu_available();

} // namespace v125
} // namespace thunderduck

#endif // THUNDERDUCK_V12_5_H
