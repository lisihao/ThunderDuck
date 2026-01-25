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

} // namespace aggregate
} // namespace thunderduck

#endif // THUNDERDUCK_AGGREGATE_H
