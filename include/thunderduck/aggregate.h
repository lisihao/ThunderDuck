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

} // namespace aggregate
} // namespace thunderduck

#endif // THUNDERDUCK_AGGREGATE_H
