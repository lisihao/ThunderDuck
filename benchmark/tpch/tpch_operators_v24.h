/**
 * ThunderDuck TPC-H 算子封装 V24
 *
 * 优化内容:
 * - P0: 选择向量替换中间 vector
 * - P1: 数组替换 hash 表
 * - P2: Filter + Join 融合
 *
 * @version 24.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V24_H
#define TPCH_OPERATORS_V24_H

#include "tpch_operators.h"
#include "tpch_data_loader.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <thread>
#include <array>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v24 {

// 使用父命名空间的 TPCHDataLoader
using ::thunderduck::tpch::TPCHDataLoader;

// ============================================================================
// P0: 选择向量 - 避免数据复制
// ============================================================================

/**
 * 轻量级选择向量
 *
 * 核心思想: 只存储索引，不复制实际数据
 * 后续操作通过索引访问原始数据
 */
class SelectionVector {
public:
    SelectionVector() : data_(nullptr), count_(0), capacity_(0), owned_(false) {}

    // 外部缓冲区模式 (零分配)
    SelectionVector(uint32_t* buffer, size_t capacity)
        : data_(buffer), count_(0), capacity_(capacity), owned_(false) {}

    // 自管理模式
    explicit SelectionVector(size_t capacity)
        : storage_(capacity), count_(0), capacity_(capacity), owned_(true) {
        data_ = storage_.data();
    }

    // 移动语义
    SelectionVector(SelectionVector&& other) noexcept
        : storage_(std::move(other.storage_))
        , data_(other.data_)
        , count_(other.count_)
        , capacity_(other.capacity_)
        , owned_(other.owned_) {
        if (owned_) data_ = storage_.data();
        other.data_ = nullptr;
        other.count_ = 0;
    }

    SelectionVector& operator=(SelectionVector&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            data_ = other.data_;
            count_ = other.count_;
            capacity_ = other.capacity_;
            owned_ = other.owned_;
            if (owned_) data_ = storage_.data();
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // 禁止拷贝
    SelectionVector(const SelectionVector&) = delete;
    SelectionVector& operator=(const SelectionVector&) = delete;

    uint32_t* data() { return data_; }
    const uint32_t* data() const { return data_; }
    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }

    void set_count(size_t n) { count_ = n; }
    uint32_t operator[](size_t i) const { return data_[i]; }

    // 预分配 (仅自管理模式)
    void reserve(size_t n) {
        if (owned_ && n > capacity_) {
            storage_.resize(n);
            data_ = storage_.data();
            capacity_ = n;
        }
    }

private:
    std::vector<uint32_t> storage_;
    uint32_t* data_;
    size_t count_;
    size_t capacity_;
    bool owned_;
};

// ============================================================================
// P0: 基于选择向量的算子
// ============================================================================

/**
 * Filter 输出选择向量 (不复制数据)
 */
size_t filter_to_sel_i32_gt(
    const int32_t* data, size_t n,
    int32_t threshold,
    uint32_t* out_sel
);

size_t filter_to_sel_i32_lt(
    const int32_t* data, size_t n,
    int32_t threshold,
    uint32_t* out_sel
);

size_t filter_to_sel_i32_range(
    const int32_t* data, size_t n,
    int32_t lo, int32_t hi,  // lo <= x < hi
    uint32_t* out_sel
);

/**
 * 在已有选择向量上继续过滤 (级联过滤)
 * 输入: sel_in[0..sel_count)
 * 输出: sel_out[0..返回值)
 * 注: sel_out 可以和 sel_in 是同一个缓冲区 (原地过滤)
 */
size_t filter_sel_i32_lt(
    const int32_t* data,
    const uint32_t* sel_in, size_t sel_count,
    int32_t threshold,
    uint32_t* sel_out
);

size_t filter_sel_i32_gt(
    const int32_t* data,
    const uint32_t* sel_in, size_t sel_count,
    int32_t threshold,
    uint32_t* sel_out
);

size_t filter_sel_i8_eq(
    const int8_t* data,
    const uint32_t* sel_in, size_t sel_count,
    int8_t value,
    uint32_t* sel_out
);

// ============================================================================
// P1: 数组替换 hash 表
// ============================================================================

/**
 * 直接数组聚合 (适用于小基数分组)
 *
 * 预分配 num_groups 大小的数组，直接索引
 * 比 unordered_map 快 10-50x
 */
template<typename T>
struct DirectArrayAgg {
    std::vector<T> sums;
    std::vector<size_t> counts;
    size_t num_groups;

    explicit DirectArrayAgg(size_t n) : sums(n, 0), counts(n, 0), num_groups(n) {}

    void add(size_t group, T value) {
        sums[group] += value;
        counts[group]++;
    }

    void merge(const DirectArrayAgg& other) {
        for (size_t i = 0; i < num_groups; ++i) {
            sums[i] += other.sums[i];
            counts[i] += other.counts[i];
        }
    }
};

/**
 * 稀疏键聚合 (适用于 orderkey 等大范围键)
 *
 * 使用排序 + 线性扫描替代 hash 表
 * 对于已排序或部分排序的数据特别高效
 */
struct SparseKeyAgg {
    // 键值对数组 (按键排序)
    std::vector<std::pair<int32_t, int64_t>> entries;

    void reserve(size_t n) { entries.reserve(n); }

    // 添加 (假设键已排序)
    void add_sorted(int32_t key, int64_t value);

    // 批量添加并合并
    void merge_from(const std::vector<std::pair<int32_t, int64_t>>& other);

    // 获取结果
    size_t size() const { return entries.size(); }
};

// ============================================================================
// P1: Q3 优化聚合 (orderkey 分组)
// ============================================================================

struct Q3AggEntry {
    int32_t orderkey;
    int64_t revenue;
    int32_t orderdate;
    int32_t shippriority;
};

/**
 * Q3 聚合优化版
 *
 * 使用排序数组而非 hash 表
 * 适用于 orderkey 基数较大但结果集中的场景
 */
class Q3AggregatorV24 {
public:
    Q3AggregatorV24() : last_key_(-1), last_idx_(0) {}  // 默认构造函数
    explicit Q3AggregatorV24(size_t estimated_groups);

    void add(int32_t orderkey, int64_t revenue, int32_t orderdate, int32_t shippriority);

    // 合并多个聚合器的结果
    void merge(Q3AggregatorV24&& other);

    // 获取 Top K 结果 (按 revenue 降序, orderdate 升序)
    std::vector<Q3AggEntry> get_top_k(size_t k);

    size_t size() const { return entries_.size(); }

private:
    // 使用 vector<pair> 而非 unordered_map
    // 假设同一 orderkey 的多条记录会相邻出现 (JOIN 后的局部性)
    std::vector<Q3AggEntry> entries_;

    // 最后一个 key 的索引 (用于快速查找)
    int32_t last_key_ = -1;
    size_t last_idx_ = 0;
};

// ============================================================================
// P2: Filter + Join 融合
// ============================================================================

/**
 * Semi Join with Filter (融合算子)
 *
 * 等价于:
 *   SELECT probe_idx FROM probe
 *   WHERE probe_key IN (SELECT build_key FROM build)
 *     AND filter_col op threshold
 *
 * 比先 Join 再 Filter 少一次遍历
 */
size_t semi_join_with_filter_i32(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    const int32_t* filter_col,      // probe 表上的过滤列
    int32_t filter_threshold,       // 过滤阈值
    bool filter_less_than,          // true: < threshold, false: > threshold
    uint32_t* out_probe_indices
);

/**
 * Inner Join with Filter (融合算子)
 *
 * 在 Join 过程中同时检查过滤条件
 */
size_t inner_join_with_filter_i32(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    const int32_t* probe_filter_col,  // probe 表上的过滤列
    int32_t filter_threshold,
    bool filter_less_than,
    uint32_t* out_build_indices,
    uint32_t* out_probe_indices
);

/**
 * 级联 Semi Join + Filter (Q3/Q5 模式)
 *
 * 等价于:
 *   orders 表: o_custkey IN (building customers) AND o_orderdate < threshold
 *   返回: 满足条件的 orders 行索引
 */
size_t cascaded_semi_filter(
    // Build 侧 (customer)
    const int32_t* build_keys, size_t build_count,
    // Probe 侧 (orders)
    const int32_t* probe_keys, size_t probe_count,
    const int32_t* probe_filter_col,  // o_orderdate
    int32_t filter_threshold,         // date threshold
    bool filter_less_than,
    uint32_t* out_probe_indices
);

// ============================================================================
// V24 优化版查询实现
// ============================================================================

/**
 * Q3 V24 优化版
 *
 * 使用:
 * - 选择向量避免中间数据复制
 * - 数组聚合替代 hash 表
 * - 融合 Filter + Join
 */
void run_q3_v24(TPCHDataLoader& loader);

/**
 * Q5 V24 优化版
 */
void run_q5_v24(TPCHDataLoader& loader);

/**
 * Q6 V24 优化版 (基线已经很好，小幅优化)
 */
void run_q6_v24(TPCHDataLoader& loader);

/**
 * Q9 V24 优化版
 */
void run_q9_v24(TPCHDataLoader& loader);

} // namespace ops_v24
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V24_H
