/**
 * ThunderDuck - Aggregation Operator
 * 
 * SIMD 加速的聚合算子，支持 SUM/AVG/MIN/MAX/COUNT
 */

#ifndef THUNDERDUCK_AGGREGATE_H
#define THUNDERDUCK_AGGREGATE_H

#include <cstdint>
#include <cstddef>
#include <limits>

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 简单聚合（无分组）
// ============================================================================

// --- SUM ---

int64_t sum_i32(const int32_t* input, size_t count);
int64_t sum_i64(const int64_t* input, size_t count);
double  sum_f32(const float* input, size_t count);
double  sum_f64(const double* input, size_t count);

// --- MIN ---

int32_t min_i32(const int32_t* input, size_t count);
int64_t min_i64(const int64_t* input, size_t count);
float   min_f32(const float* input, size_t count);
double  min_f64(const double* input, size_t count);

// --- MAX ---

int32_t max_i32(const int32_t* input, size_t count);
int64_t max_i64(const int64_t* input, size_t count);
float   max_f32(const float* input, size_t count);
double  max_f64(const double* input, size_t count);

// --- AVG ---

double avg_i32(const int32_t* input, size_t count);
double avg_i64(const int64_t* input, size_t count);
double avg_f32(const float* input, size_t count);
double avg_f64(const double* input, size_t count);

// --- COUNT (non-null) ---
// 假设 null 用特定值表示（如 INT32_MIN）

size_t count_nonnull_i32(const int32_t* input, size_t count,
                         int32_t null_value = std::numeric_limits<int32_t>::min());

// ============================================================================
// v2.0 优化版本 - 合并函数 + 预取
// ============================================================================

/**
 * 合并的 MIN/MAX 函数 - 单次遍历同时计算
 */
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max);

void minmax_i64(const int64_t* input, size_t count,
                int64_t* out_min, int64_t* out_max);

void minmax_f32(const float* input, size_t count,
                float* out_min, float* out_max);

/**
 * 优化版 SUM - 16 元素/迭代 + 预取
 */
int64_t sum_i32_v2(const int32_t* input, size_t count);
double sum_f32_v2(const float* input, size_t count);

/**
 * Kahan 求和 - 高精度版本
 */
double avg_f64_kahan(const double* input, size_t count);

/**
 * 一次遍历计算所有统计量
 */
struct AggregateStats {
    int64_t sum;
    int64_t count;
    int32_t min_val;
    int32_t max_val;
};

AggregateStats aggregate_all_i32(const int32_t* input, size_t count);

// ============================================================================
// 带选择向量的聚合（用于过滤后的数据）
// ============================================================================

int64_t sum_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count);
double  sum_f32_sel(const float* input, const uint32_t* sel, size_t sel_count);

int32_t min_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count);
int32_t max_i32_sel(const int32_t* input, const uint32_t* sel, size_t sel_count);

// ============================================================================
// 分组聚合
// ============================================================================

/**
 * 分组聚合结果结构
 */
struct GroupAggResult {
    size_t num_groups;      // 分组数量
    int64_t* sums;          // 每组的 SUM（需预分配）
    size_t* counts;         // 每组的 COUNT（需预分配）
    int32_t* mins;          // 每组的 MIN（可选）
    int32_t* maxs;          // 每组的 MAX（可选）
};

/**
 * 分组求和
 * 
 * @param values 值数组
 * @param groups 分组 ID 数组（0 到 num_groups-1）
 * @param count 元素数量
 * @param num_groups 分组数量
 * @param out_sums 输出每组的和（需预分配 num_groups 个元素）
 */
void group_sum_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int64_t* out_sums);

void group_sum_i64(const int64_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int64_t* out_sums);

void group_sum_f64(const double* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, double* out_sums);

/**
 * 分组计数
 */
void group_count(const uint32_t* groups, size_t count, 
                 size_t num_groups, size_t* out_counts);

/**
 * 分组 MIN/MAX
 */
void group_min_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32(const int32_t* values, const uint32_t* groups, 
                   size_t count, size_t num_groups, int32_t* out_maxs);

// ============================================================================
// 哈希聚合（动态分组）
// ============================================================================

/**
 * 哈希聚合上下文
 */
class HashAggregator {
public:
    HashAggregator(size_t initial_capacity = 1024);
    ~HashAggregator();

    // 禁止拷贝
    HashAggregator(const HashAggregator&) = delete;
    HashAggregator& operator=(const HashAggregator&) = delete;

    /**
     * 添加数据
     */
    void add_i32(int32_t key, int32_t value);
    void add_i64(int64_t key, int64_t value);

    /**
     * 获取结果
     */
    size_t get_num_groups() const;
    void get_results(int64_t* keys, int64_t* sums, size_t* counts) const;

    /**
     * 重置
     */
    void reset();

private:
    struct Impl;
    Impl* impl_;
};

// ============================================================================
// v3.0 UMA 优化版本 - GPU 加速 + 零拷贝
// ============================================================================

/**
 * 聚合策略
 */
enum class AggregateStrategy {
    AUTO,       // 自动选择
    CPU_SIMD,   // CPU SIMD
    GPU,        // GPU 并行归约
};

/**
 * v3.0 聚合配置
 */
struct AggregateConfigV3 {
    AggregateStrategy strategy = AggregateStrategy::AUTO;
};

/**
 * 检查 GPU 聚合是否可用
 */
bool is_aggregate_gpu_available();

// --- UMA 优化版 SUM ---

int64_t sum_i32_v3(const int32_t* input, size_t count);
int64_t sum_i32_v3_config(const int32_t* input, size_t count,
                           const AggregateConfigV3& config);

double sum_f32_v3(const float* input, size_t count);

// --- UMA 优化版 MIN/MAX ---

int32_t min_i32_v3(const int32_t* input, size_t count);
int32_t max_i32_v3(const int32_t* input, size_t count);

void minmax_i32_v3(const int32_t* input, size_t count,
                    int32_t* out_min, int32_t* out_max);

// --- UMA 优化版带选择向量聚合 ---

int64_t sum_i32_sel_v3(const int32_t* input, const uint32_t* sel, size_t sel_count);

// --- UMA 优化版分组聚合 ---

void group_sum_i32_v3(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v3(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts);

// --- UMA 优化版全统计 ---

AggregateStats aggregate_all_i32_v3(const int32_t* input, size_t count);

// ============================================================================
// v4.0 优化版本 - P0~P3 性能优化
// ============================================================================

/**
 * v4.0 SUM - P1预取优化 + P2缓存分块
 * 32元素/迭代, 256B预取距离
 */
int64_t sum_i32_v4(const int32_t* input, size_t count);
int64_t sum_i32_v4_blocked(const int32_t* input, size_t count);

/**
 * v4.0 MIN/MAX - P1预取优化
 */
void minmax_i32_v4(const int32_t* input, size_t count,
                   int32_t* out_min, int32_t* out_max);

/**
 * v4.0 融合统计量 - P1+P2优化
 */
AggregateStats aggregate_all_i32_v4(const int32_t* input, size_t count);

/**
 * v4.0 分组聚合 - P0向量化哈希分组
 */
void group_sum_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v4(const uint32_t* groups, size_t count,
                    size_t num_groups, size_t* out_counts);

void group_min_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v4(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int32_t* out_maxs);

/**
 * v4.0 多线程分组聚合 - P3并行优化
 */
void group_sum_i32_v4_parallel(const int32_t* values, const uint32_t* groups,
                               size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v4_parallel(const uint32_t* groups, size_t count,
                             size_t num_groups, size_t* out_counts);

// ============================================================================
// v5.0 优化版本 - V9.2 GPU 两阶段分组聚合
// ============================================================================

/**
 * 检查 V9.2 GPU 分组聚合是否可用
 */
bool is_group_aggregate_v2_available();

// ============================================================================
// v6.0 优化版本 - V9.3 智能策略选择
// ============================================================================

/**
 * 分组聚合策略版本
 */
enum class GroupAggregateVersion {
    V4_SINGLE,    // CPU 单线程 (低开销，小数据)
    V4_PARALLEL,  // CPU 多线程 (最佳通用性能)
    V5_GPU,       // GPU 两阶段 (大数据高竞争)
    AUTO          // 自动选择
};

/**
 * 分组聚合策略配置
 */
struct GroupAggregateConfig {
    GroupAggregateVersion version = GroupAggregateVersion::AUTO;
    bool debug_log = false;  // 输出策略选择日志
};

/**
 * 获取推荐的分组聚合策略
 *
 * 策略选择规则 (基于 M4 基准测试):
 * - count < 100K: V4_SINGLE (线程启动开销 > 收益)
 * - 100K ≤ count < 50M: V4_PARALLEL (最佳通用性能)
 * - count ≥ 50M AND num_groups ≤ 32: V5_GPU (GPU 带宽优势)
 * - 其他: V4_PARALLEL (默认最优)
 */
GroupAggregateVersion select_group_aggregate_strategy(
    size_t count, size_t num_groups);

/**
 * 获取策略选择的原因 (调试用)
 */
const char* get_group_aggregate_strategy_reason();

/**
 * v5.0 GPU 两阶段分组求和
 *
 * 优化特性:
 * - Phase 1: Threadgroup 本地累加 (共享内存原子)
 * - Phase 2: 全局合并 (设备内存原子)
 * - 全局原子操作从 count 降到 num_groups * num_threadgroups
 *
 * 适用条件: count >= 100K, num_groups <= 1024
 * 否则回退到 v4 多线程实现
 */
void group_sum_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v5(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts);

void group_min_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs);

/**
 * v6.0 智能策略分组求和
 *
 * 自动选择最优实现:
 * - V4_SINGLE: CPU 单线程 (count < 100K)
 * - V4_PARALLEL: CPU 多线程 (100K - 50M，或默认)
 * - V5_GPU: GPU 两阶段 (count >= 50M, groups <= 32)
 */
void group_sum_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums);

void group_sum_i32_v6_config(const int32_t* values, const uint32_t* groups,
                              size_t count, size_t num_groups, int64_t* out_sums,
                              const GroupAggregateConfig& config);

void group_count_v6(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts);

void group_min_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs);

// ============================================================================
// 多线程并行版本 (10M+ 数据优化)
// ============================================================================

/**
 * 多线程并行 SUM - 4 线程并行
 *
 * 当数据量 >= 1M 时自动启用多线程
 * 预期: 10M 数据 3.3x → 5-6x vs DuckDB
 */
int64_t sum_i32_parallel(const int32_t* input, size_t count);

/**
 * 多线程并行 MIN/MAX
 */
void minmax_i32_parallel(const int32_t* input, size_t count,
                          int32_t* out_min, int32_t* out_max);

/**
 * 多线程并行融合统计量
 */
AggregateStats aggregate_all_i32_parallel(const int32_t* input, size_t count);

/**
 * 多线程并行非零计数
 */
size_t count_nonzero_i32_parallel(const int32_t* input, size_t count);

// ============================================================================
// V12.1 优化版本 - P0 GPU GROUP BY 优化
// ============================================================================

/**
 * 检查 V12.1 GPU 分组聚合是否可用
 */
bool is_group_aggregate_v3_available();

/**
 * V12.1 GPU 分组求和 - Warp-level reduction
 *
 * 核心优化:
 * - 低基数 (<=32分组): 纯寄存器累加 + SIMD shuffle
 * - 中基数 (<=1024分组): 8路展开 + 增加 threadgroup 数量
 * - 目标: 0.88x → 2.0x+
 */
void group_sum_i32_v12_1(const int32_t* values, const uint32_t* groups,
                          size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v12_1(const uint32_t* groups, size_t count,
                        size_t num_groups, size_t* out_counts);

void group_min_i32_v12_1(const int32_t* values, const uint32_t* groups,
                          size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v12_1(const int32_t* values, const uint32_t* groups,
                          size_t count, size_t num_groups, int32_t* out_maxs);

// ============================================================================
// V13 优化版本 - P1 GPU GROUP BY 无原子优化
// ============================================================================

/**
 * 检查 V13 GPU 分组聚合是否可用
 */
bool is_group_aggregate_v13_available();

/**
 * V13 GPU 分组求和 - 分区聚合策略
 *
 * 核心优化:
 * - Phase 1: 每个 threadgroup 独立累加 (无全局原子)
 * - Phase 2: 合并分区结果 (仅 num_groups 次操作)
 *
 * 目标: 0.78x → 2.0x+
 */
void group_sum_i32_v13(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v13(const uint32_t* groups, size_t count,
                      size_t num_groups, size_t* out_counts);

void group_min_i32_v13(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v13(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int32_t* out_maxs);

// ============================================================================
// V14 优化版本 - 寄存器缓冲 + 多路分流
// ============================================================================

/**
 * V14 分组聚合 - 深度优化版本
 *
 * 核心优化:
 * 1. 寄存器缓冲累加 (低基数 <= 64 分组)
 *    - 8 个寄存器缓存热分组
 *    - 减少内存写入
 * 2. 多路分流 (高基数)
 *    - 按 group_id % 4 分 4 路
 *    - 每路独立 SIMD 累加
 * 3. 并行 + SIMD 合并
 *    - 4 核并行局部累加
 *    - SIMD 合并最终结果
 *
 * 目标: 2.66x → 4x+
 */
void group_sum_i32_v14(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int64_t* out_sums);

void group_sum_i32_v14_parallel(const int32_t* values, const uint32_t* groups,
                                 size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v14(const uint32_t* groups, size_t count,
                      size_t num_groups, size_t* out_counts);

void group_min_i32_v14(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int32_t* out_mins);

void group_max_i32_v14(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int32_t* out_maxs);

const char* get_group_aggregate_v14_version();

} // namespace aggregate
} // namespace thunderduck

#endif // THUNDERDUCK_AGGREGATE_H
