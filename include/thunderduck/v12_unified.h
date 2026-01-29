/**
 * ThunderDuck V12 - 统一最优版本
 *
 * 集成各算子历史最优实现，根据数据规模自动选择最佳策略:
 *
 * | 算子          | 小数据(1M) | 大数据(10M) |
 * |---------------|------------|-------------|
 * | Filter        | V9 (8.0x)  | V7 (3.3x)   |
 * | Aggregate     | V9 (5.1x)  | V7 (3.5x)   |
 * | TopK          | V8 (14.7x) | V7 (4.5x)   |
 * | GROUP BY      | V8 (3.7x)  | V8 (2.8x)   |
 * | Hash Join     | V11 (1.08x)| V11 (1.08x) |
 */

#ifndef THUNDERDUCK_V12_UNIFIED_H
#define THUNDERDUCK_V12_UNIFIED_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v12 {

// ============================================================================
// V12 统一配置
// ============================================================================

/**
 * 数据规模阈值
 */
constexpr size_t SMALL_DATA_THRESHOLD = 1000000;   // 1M
constexpr size_t LARGE_DATA_THRESHOLD = 10000000;  // 10M

/**
 * 执行设备
 */
enum class ExecutionDevice {
    CPU_SCALAR,     // CPU 标量
    CPU_SIMD,       // CPU SIMD (Neon)
    CPU_PARALLEL,   // CPU 多线程
    GPU_METAL,      // GPU Metal
    NPU_COREML,     // NPU Core ML
    AUTO            // 自动选择
};

/**
 * V12 配置
 */
struct V12Config {
    ExecutionDevice preferred_device = ExecutionDevice::AUTO;
    bool debug_log = false;           // 输出策略选择日志
    bool force_version = false;       // 强制使用指定版本
    int force_version_number = 0;     // 强制版本号 (3-11)
};

/**
 * 执行统计
 */
struct ExecutionStats {
    const char* operator_name;        // 算子名称
    const char* version_used;         // 使用的版本
    const char* device_used;          // 使用的设备
    size_t data_count;                // 数据量
    double throughput_gbps;           // 吞吐量 (GB/s)
    double elapsed_ms;                // 执行时间 (ms)
};

// ============================================================================
// V12 Filter - 统一接口
// ============================================================================

/**
 * 比较操作
 */
enum class CompareOp {
    EQ,   // ==
    NE,   // !=
    LT,   // <
    LE,   // <=
    GT,   // >
    GE    // >=
};

/**
 * V12 统一 Filter
 *
 * 策略选择:
 * - count < 1M: V9 CPU SIMD (8.0x)
 * - count >= 1M: V7 GPU Metal (3.3x)
 *
 * @param input 输入数组
 * @param count 元素数量
 * @param op 比较操作
 * @param value 比较值
 * @param out_indices 输出索引数组
 * @param stats 输出执行统计 (可选)
 * @return 满足条件的元素数量
 */
size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats = nullptr);

/**
 * V12 统一 Count
 */
size_t count_i32(const int32_t* input, size_t count,
                 CompareOp op, int32_t value,
                 ExecutionStats* stats = nullptr);

/**
 * V12 带配置版本
 */
size_t filter_i32_config(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         uint32_t* out_indices,
                         const V12Config& config,
                         ExecutionStats* stats = nullptr);

// ============================================================================
// V12 Aggregate - 统一接口
// ============================================================================

/**
 * 聚合统计结果
 */
struct AggregateResult {
    int64_t sum;
    int64_t count;
    int32_t min_val;
    int32_t max_val;
    double avg;
};

/**
 * V12 统一 SUM
 *
 * 策略选择:
 * - count < 1M: V9 CPU SIMD (5.1x)
 * - count >= 1M: V7 GPU Metal (3.5x)
 */
int64_t sum_i32(const int32_t* input, size_t count,
                ExecutionStats* stats = nullptr);

/**
 * V12 统一 MIN/MAX
 */
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max,
                ExecutionStats* stats = nullptr);

/**
 * V12 统一全统计
 */
AggregateResult aggregate_all_i32(const int32_t* input, size_t count,
                                   ExecutionStats* stats = nullptr);

/**
 * V12 带配置版本
 */
int64_t sum_i32_config(const int32_t* input, size_t count,
                       const V12Config& config,
                       ExecutionStats* stats = nullptr);

// ============================================================================
// V12 GROUP BY - 统一接口
// ============================================================================

/**
 * V12 统一分组求和
 *
 * 策略选择:
 * - 始终使用 V8 CPU 多线程 (2.8-3.7x)
 */
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);

/**
 * V12 统一分组计数
 */
void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats = nullptr);

/**
 * V12 统一分组 MIN/MAX
 */
void group_min_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_mins,
                   ExecutionStats* stats = nullptr);

void group_max_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_maxs,
                   ExecutionStats* stats = nullptr);

// ============================================================================
// V12 TopK - 统一接口
// ============================================================================

/**
 * V12 统一 TopK Max
 *
 * 策略选择:
 * - count < 1M: V8 计数排序 (14.7x)
 * - count >= 1M: V7 采样预过滤 (4.5x)
 */
void topk_max_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices = nullptr,
                  ExecutionStats* stats = nullptr);

/**
 * V12 统一 TopK Min
 */
void topk_min_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices = nullptr,
                  ExecutionStats* stats = nullptr);

/**
 * V12 带配置版本
 */
void topk_max_i32_config(const int32_t* data, size_t count, size_t k,
                         int32_t* out_values, uint32_t* out_indices,
                         const V12Config& config,
                         ExecutionStats* stats = nullptr);

// ============================================================================
// V12 Hash Join - 统一接口
// ============================================================================

/**
 * 连接类型
 */
enum class JoinType {
    INNER,
    LEFT,
    RIGHT,
    FULL,
    SEMI,
    ANTI
};

/**
 * 连接结果
 */
struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

/**
 * 创建/释放连接结果
 */
JoinResult* create_join_result(size_t initial_capacity);
void free_join_result(JoinResult* result);

/**
 * V12 统一 Hash Join
 *
 * 策略选择:
 * - 始终使用 V11 SIMD 加速 (1.08x)
 *   - 负载因子 0.33
 *   - 三级预取
 *   - 8 路循环展开
 */
size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type, JoinResult* result,
                     ExecutionStats* stats = nullptr);

/**
 * V12 带配置版本
 */
size_t hash_join_i32_config(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinType join_type, JoinResult* result,
                            const V12Config& config,
                            ExecutionStats* stats = nullptr);

// ============================================================================
// V12 版本信息
// ============================================================================

/**
 * 获取 V12 版本信息
 */
const char* get_version_info();

/**
 * 获取各算子最优版本列表
 */
const char* get_optimal_versions();

/**
 * 检查功能是否可用
 */
bool is_gpu_available();
bool is_npu_available();

} // namespace v12
} // namespace thunderduck

#endif // THUNDERDUCK_V12_UNIFIED_H
