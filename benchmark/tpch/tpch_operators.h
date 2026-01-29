/**
 * ThunderDuck TPC-H 算子封装
 *
 * 封装 ThunderDuck 算子供 TPC-H 查询使用
 *
 * @version 1.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_H
#define TPCH_OPERATORS_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <string>

// ThunderDuck 算子头文件
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops {

// ============================================================================
// Filter 算子封装
// ============================================================================

/**
 * 过滤结果 - 存储通过过滤的行索引
 */
struct FilteredIndices {
    std::vector<uint32_t> indices;
    size_t count = 0;

    void resize(size_t n) {
        indices.resize(n);
        count = 0;
    }
};

/**
 * 范围过滤 (low <= x < high)
 */
size_t filter_range_i32(const int32_t* data, size_t n,
                        int32_t low, int32_t high,
                        uint32_t* out_indices);

/**
 * 范围过滤 (low <= x <= high)
 */
size_t filter_range_inclusive_i64(const int64_t* data, size_t n,
                                  int64_t low, int64_t high,
                                  uint32_t* out_indices);

/**
 * 多条件过滤 (AND)
 * 使用位图实现高效的多条件组合
 */
class MultiFilter {
public:
    explicit MultiFilter(size_t n);

    // 添加条件
    void add_range_i32(const int32_t* data, int32_t low, int32_t high);
    void add_range_inclusive_i64(const int64_t* data, int64_t low, int64_t high);
    void add_lt_i64(const int64_t* data, int64_t value);
    void add_eq_i8(const int8_t* data, int8_t value);
    void add_ne_i8(const int8_t* data, int8_t value);

    // 获取结果
    size_t get_indices(uint32_t* out_indices) const;
    size_t count() const;

private:
    size_t n_;
    std::vector<uint64_t> bitmap_;  // 位图表示
    bool initialized_ = false;
};

// ============================================================================
// Aggregate 算子封装
// ============================================================================

/**
 * 简单 SUM (使用 V21 优化版本)
 */
inline int64_t sum_i64(const int64_t* data, size_t n) {
    return aggregate::sum_i64_v21(data, n);
}

/**
 * 带选择向量的 SUM
 */
int64_t sum_i64_sel(const int64_t* data, const uint32_t* sel, size_t sel_count);

/**
 * 定点数乘积 SUM
 * SUM(a * b) where a, b are fixed-point (x10000)
 * 结果也是 x10000
 */
int64_t sum_product_fixed(const int64_t* a, const int64_t* b, size_t n);

/**
 * 带选择向量的定点数乘积 SUM
 */
int64_t sum_product_fixed_sel(const int64_t* a, const int64_t* b,
                              const uint32_t* sel, size_t sel_count);

/**
 * SUM(a * (10000 - b)) / 10000
 * 用于计算 SUM(price * (1 - discount))
 */
int64_t sum_price_discount(const int64_t* price, const int64_t* discount, size_t n);

/**
 * 带选择向量的价格折扣计算
 */
int64_t sum_price_discount_sel(const int64_t* price, const int64_t* discount,
                               const uint32_t* sel, size_t sel_count);

/**
 * 分组聚合结果
 */
template<typename K, typename V>
struct GroupAggResult {
    std::unordered_map<K, V> sums;
    std::unordered_map<K, size_t> counts;
};

/**
 * 整数键分组 SUM (使用 V15 优化版本)
 */
void group_sum_i32_key(const int64_t* values, const int32_t* keys,
                       size_t n, std::unordered_map<int32_t, int64_t>& sums);

/**
 * 字符串键分组 SUM (使用哈希表)
 */
void group_sum_string_key(const int64_t* values, const std::string* keys,
                          size_t n, std::unordered_map<std::string, int64_t>& sums);

/**
 * 复合键分组 SUM (两个 int8_t 键)
 * 用于 Q1: GROUP BY l_returnflag, l_linestatus
 */
struct Q1AggResult {
    int64_t sum_qty = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    int64_t sum_charge = 0;
    int64_t sum_discount = 0;
    size_t count = 0;
};

void q1_group_aggregate(
    const int64_t* quantity,
    const int64_t* extendedprice,
    const int64_t* discount,
    const int64_t* tax,
    const int8_t* returnflag,
    const int8_t* linestatus,
    const int32_t* shipdate,
    int32_t date_threshold,
    size_t n,
    std::unordered_map<int16_t, Q1AggResult>& results  // key = (rf << 8) | ls
);

// ============================================================================
// Join 算子封装
// ============================================================================

/**
 * Hash Join 结果
 */
struct JoinPairs {
    std::vector<uint32_t> left_indices;
    std::vector<uint32_t> right_indices;
    size_t count = 0;
};

/**
 * Inner Join (使用 V14 优化版本)
 */
void inner_join_i32(const int32_t* build_keys, size_t build_count,
                    const int32_t* probe_keys, size_t probe_count,
                    JoinPairs& result);

/**
 * Semi Join (使用 GPU 版本如果可用)
 */
void semi_join_i32(const int32_t* build_keys, size_t build_count,
                   const int32_t* probe_keys, size_t probe_count,
                   std::vector<uint32_t>& probe_matches);

/**
 * Anti Join
 */
void anti_join_i32(const int32_t* build_keys, size_t build_count,
                   const int32_t* probe_keys, size_t probe_count,
                   std::vector<uint32_t>& probe_non_matches);

/**
 * 多表 Join (链式执行)
 * 用于 Q3, Q5, Q7 等多表连接场景
 */
class MultiJoin {
public:
    // 添加 Join 条件
    void add_join(const int32_t* left_keys, size_t left_count,
                  const int32_t* right_keys, size_t right_count);

    // 执行并获取结果
    void execute();

    // 获取结果映射
    const std::vector<std::vector<uint32_t>>& get_indices() const { return table_indices_; }

private:
    struct JoinSpec {
        const int32_t* left_keys;
        size_t left_count;
        const int32_t* right_keys;
        size_t right_count;
    };

    std::vector<JoinSpec> joins_;
    std::vector<std::vector<uint32_t>> table_indices_;
};

// ============================================================================
// 组合算子 - 常见模式的优化实现
// ============================================================================

/**
 * Filter + SUM 组合 (Q6 模式)
 *
 * 等价于:
 * SELECT SUM(a * b) FROM table WHERE date >= lo AND date < hi AND ...
 */
int64_t filter_sum_product(
    const int64_t* a,           // 乘数1 (e.g., extendedprice)
    const int64_t* b,           // 乘数2 (e.g., discount)
    const int32_t* date,        // 日期列
    int32_t date_lo,            // 日期下界
    int32_t date_hi,            // 日期上界
    const int64_t* filter_col,  // 额外过滤列 (e.g., quantity)
    int64_t filter_lo,          // 过滤下界
    int64_t filter_hi,          // 过滤上界
    const int64_t* discount_col, // 折扣列用于范围过滤
    int64_t discount_lo,
    int64_t discount_hi,
    size_t n
);

/**
 * Join + GROUP BY + SUM 组合 (Q3, Q5, Q7 模式)
 */
struct JoinGroupSumResult {
    std::unordered_map<int64_t, int64_t> sums;  // group_key -> sum
    std::unordered_map<int64_t, size_t> counts;
};

void join_group_sum(
    const int32_t* left_keys, size_t left_count,
    const int32_t* right_keys, size_t right_count,
    const int64_t* values,      // 要聚合的值 (在 right 表中)
    const int32_t* group_keys,  // 分组键 (在 join 结果中)
    JoinGroupSumResult& result
);

// ============================================================================
// SIMD 辅助函数
// ============================================================================

#ifdef __aarch64__

/**
 * ARM Neon 向量化范围检查
 * 返回满足 lo <= x < hi 的掩码
 */
inline uint32x4_t simd_range_check_i32(int32x4_t values, int32_t lo, int32_t hi) {
    int32x4_t v_lo = vdupq_n_s32(lo);
    int32x4_t v_hi = vdupq_n_s32(hi);
    uint32x4_t ge_lo = vcgeq_s32(values, v_lo);
    uint32x4_t lt_hi = vcltq_s32(values, v_hi);
    return vandq_u32(ge_lo, lt_hi);
}

/**
 * ARM Neon 向量化范围检查 (包含边界)
 */
inline uint32x4_t simd_range_check_inclusive_i32(int32x4_t values, int32_t lo, int32_t hi) {
    int32x4_t v_lo = vdupq_n_s32(lo);
    int32x4_t v_hi = vdupq_n_s32(hi);
    uint32x4_t ge_lo = vcgeq_s32(values, v_lo);
    uint32x4_t le_hi = vcleq_s32(values, v_hi);
    return vandq_u32(ge_lo, le_hi);
}

/**
 * 统计掩码中的 1 的数量
 */
inline int simd_popcount_u32x4(uint32x4_t mask) {
    // 转换为字节然后计算
    uint8x16_t bytes = vreinterpretq_u8_u32(mask);
    uint8x16_t counts = vcntq_u8(bytes);
    return vaddvq_u8(counts) / 8;  // 每个 1 占用 8 位
}

#endif // __aarch64__

} // namespace ops
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_H
