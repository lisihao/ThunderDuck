/**
 * ThunderDuck TPC-H 算子封装 V26
 *
 * 核心优化:
 * - P0-1: MutableWeakHashTable (支持原地更新聚合)
 * - P0-2: VectorizedGroupBySum (SIMD批量hash + 分区聚合)
 * - P1: FusedFilterJoinAggregate (单遍扫描融合)
 *
 * @version 26.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V26_H
#define TPCH_OPERATORS_V26_H

#include "tpch_data_loader.h"
#include "tpch_operators_v25.h"
#include <cstdint>
#include <vector>
#include <array>
#include <functional>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v26 {

using ::thunderduck::tpch::TPCHDataLoader;
using namespace ops_v25;  // 继承V25的ThreadPool等

// ============================================================================
// P0-1: MutableWeakHashTable - 支持原地更新的Hash表
// ============================================================================

/**
 * 可变弱Hash表
 *
 * 与WeakHashTable的区别:
 * - 支持 add_or_update: 存在则累加，不存在则插入
 * - 使用开放寻址 + 线性探测 (更好的缓存局部性)
 * - 支持批量更新操作
 */
template<typename V>
class MutableWeakHashTable {
public:
    static constexpr int32_t EMPTY_KEY = INT32_MIN;

    MutableWeakHashTable() = default;

    /**
     * 初始化表
     * @param estimated_size 预估元素数量
     */
    void init(size_t estimated_size) {
        // 2的幂次，负载因子 < 0.5 (开放寻址需要更低负载)
        size_t min_size = estimated_size * 2 + 1;
        table_size_ = 1;
        while (table_size_ < min_size) table_size_ <<= 1;
        mask_ = table_size_ - 1;

        keys_.assign(table_size_, EMPTY_KEY);
        values_.assign(table_size_, V{});
    }

    /**
     * 添加或更新: 如果key存在则累加value，否则插入
     * @return 该key的新值
     */
    V add_or_update(int32_t key, V delta) {
        uint32_t idx = weak_hash_i32(key) & mask_;

        while (true) {
            int32_t existing = keys_[idx];
            if (existing == EMPTY_KEY) {
                // 插入新key
                keys_[idx] = key;
                values_[idx] = delta;
                count_++;
                return delta;
            }
            if (existing == key) {
                // 累加到现有值
                values_[idx] += delta;
                return values_[idx];
            }
            // 线性探测
            idx = (idx + 1) & mask_;
        }
    }

    /**
     * 批量添加或更新 (优化版本)
     */
    void batch_add(const int32_t* keys, const V* deltas, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            add_or_update(keys[i], deltas[i]);
        }
    }

    /**
     * 批量添加 (key数组 + 单一delta值)
     */
    void batch_add_uniform(const int32_t* keys, size_t n, V delta) {
        for (size_t i = 0; i < n; ++i) {
            add_or_update(keys[i], delta);
        }
    }

    /**
     * 查找key的值
     * @return 值指针 (如果不存在返回nullptr)
     */
    V* find(int32_t key) {
        uint32_t idx = weak_hash_i32(key) & mask_;
        size_t probes = 0;

        while (probes < table_size_) {
            if (keys_[idx] == EMPTY_KEY) return nullptr;
            if (keys_[idx] == key) return &values_[idx];
            idx = (idx + 1) & mask_;
            probes++;
        }
        return nullptr;
    }

    const V* find(int32_t key) const {
        return const_cast<MutableWeakHashTable*>(this)->find(key);
    }

    /**
     * 遍历所有非空条目
     */
    template<typename Func>
    void for_each(Func&& func) const {
        for (size_t i = 0; i < table_size_; ++i) {
            if (keys_[i] != EMPTY_KEY) {
                func(keys_[i], values_[i]);
            }
        }
    }

    size_t size() const { return count_; }
    size_t table_size() const { return table_size_; }

private:
    std::vector<int32_t> keys_;
    std::vector<V> values_;
    size_t table_size_ = 0;
    uint32_t mask_ = 0;
    size_t count_ = 0;
};

// ============================================================================
// P0-1.5: BloomFilter - 快速预过滤
// ============================================================================

/**
 * 简单Bloom Filter
 *
 * 用于Q3等查询的预过滤，快速拒绝不在hash表中的key
 * 假阳性率约 ~1% (8 bits/key)
 */
class BloomFilter {
public:
    void init(size_t expected_elements) {
        // 8 bits per element, ~1% false positive rate
        size_t num_bits = expected_elements * 8;
        // 对齐到64字节（缓存行）
        num_bits = ((num_bits + 511) / 512) * 512;
        bits_.resize(num_bits / 64, 0);
        mask_ = num_bits - 1;
        num_bits_ = num_bits;
    }

    void insert(int32_t key) {
        uint32_t h1 = weak_hash_i32(key);
        uint32_t h2 = weak_hash_i32(key ^ 0x9E3779B9);
        uint32_t h3 = h1 ^ (h2 >> 3);

        set_bit(h1 & mask_);
        set_bit(h2 & mask_);
        set_bit(h3 & mask_);
    }

    bool may_contain(int32_t key) const {
        uint32_t h1 = weak_hash_i32(key);
        uint32_t h2 = weak_hash_i32(key ^ 0x9E3779B9);
        uint32_t h3 = h1 ^ (h2 >> 3);

        return test_bit(h1 & mask_) &&
               test_bit(h2 & mask_) &&
               test_bit(h3 & mask_);
    }

    // 批量测试 (返回通过的索引数量)
    size_t batch_filter(const int32_t* keys, size_t n,
                        uint32_t* out_indices) const {
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (may_contain(keys[i])) {
                out_indices[count++] = static_cast<uint32_t>(i);
            }
        }
        return count;
    }

private:
    void set_bit(uint32_t pos) {
        bits_[pos / 64] |= (1ULL << (pos % 64));
    }

    bool test_bit(uint32_t pos) const {
        return (bits_[pos / 64] & (1ULL << (pos % 64))) != 0;
    }

    std::vector<uint64_t> bits_;
    uint32_t mask_ = 0;
    size_t num_bits_ = 0;
};

// ============================================================================
// P0-2: VectorizedGroupBySum - 向量化GROUP BY聚合
// ============================================================================

/**
 * 向量化GROUP BY SUM
 *
 * 优化策略:
 * - SIMD批量hash计算
 * - 分区聚合 (减少hash冲突)
 * - 8路展开循环
 */
class VectorizedGroupBySum {
public:
    /**
     * 初始化
     * @param estimated_groups 预估分组数
     */
    void init(size_t estimated_groups);

    /**
     * 批量hash计算 (SIMD)
     */
    static void batch_hash(const int32_t* keys, size_t n, uint32_t* out_hashes);

    /**
     * 执行GROUP BY SUM
     * @param keys 分组键
     * @param values 聚合值
     * @param n 元素数量
     */
    void aggregate(const int32_t* keys, const int64_t* values, size_t n);

    /**
     * 获取结果
     */
    const MutableWeakHashTable<int64_t>& results() const { return table_; }

    /**
     * 获取满足条件的key
     */
    template<typename Pred>
    void filter_keys(Pred&& pred, std::vector<int32_t>& out_keys) const {
        table_.for_each([&](int32_t key, int64_t value) {
            if (pred(key, value)) {
                out_keys.push_back(key);
            }
        });
    }

private:
    MutableWeakHashTable<int64_t> table_;
};

// ============================================================================
// P1: FusedFilterJoinAggregate - Filter-Join-Aggregate融合
// ============================================================================

/**
 * 融合Filter-Join-Aggregate
 *
 * 单遍扫描实现:
 * 1. 对每行应用过滤条件
 * 2. 如果通过，执行hash查找
 * 3. 如果匹配，直接累加到聚合结果
 *
 * 优点:
 * - 避免中间vector分配
 * - 更好的缓存局部性
 * - 减少内存带宽
 */

// Q3专用融合结构
struct Q3AggResultV26 {
    int64_t revenue = 0;
    int32_t orderdate = 0;
    int32_t shippriority = 0;

    // 支持 += 操作符用于聚合
    Q3AggResultV26& operator+=(const Q3AggResultV26& other) {
        revenue += other.revenue;
        // orderdate 和 shippriority 保持不变
        return *this;
    }
};

/**
 * Q3融合执行
 *
 * 原始SQL:
 * SELECT l_orderkey, SUM(revenue), o_orderdate, o_shippriority
 * FROM customer, orders, lineitem
 * WHERE c_mktsegment = 'BUILDING'
 *   AND c_custkey = o_custkey
 *   AND l_orderkey = o_orderkey
 *   AND o_orderdate < '1995-03-15'
 *   AND l_shipdate > '1995-03-15'
 * GROUP BY l_orderkey, o_orderdate, o_shippriority
 */
void fused_q3_v26(
    // lineitem columns
    const int32_t* l_orderkey,
    const int32_t* l_shipdate,
    const int64_t* l_extendedprice,
    const int64_t* l_discount,  // int64_t 匹配 LineitemColumns
    size_t lineitem_count,
    // valid orders (已过滤BUILDING客户)
    const WeakHashTable<Q3AggResultV26>& valid_orders,  // orderkey -> (0, orderdate, shippriority)
    // output
    MutableWeakHashTable<Q3AggResultV26>& results,
    int32_t date_threshold  // l_shipdate > threshold
);

/**
 * Q18融合执行
 *
 * 原始SQL:
 * SELECT l_orderkey, SUM(l_quantity)
 * FROM lineitem
 * GROUP BY l_orderkey
 * HAVING SUM(l_quantity) > 300
 */
void fused_q18_groupby_v26(
    const int32_t* l_orderkey,
    const int64_t* l_quantity,
    size_t count,
    MutableWeakHashTable<int64_t>& order_qty  // orderkey -> sum(quantity)
);

/**
 * Q3优化版: SIMD预过滤 + Bloom Filter + 批量聚合
 *
 * 优化策略:
 * 1. SIMD批量过滤shipdate (消除~46%行)
 * 2. Bloom Filter预过滤orderkey (消除~53%行)
 * 3. 只对候选行做hash查找
 * 4. 批量收集后统一聚合
 */
void fused_q3_optimized_v26(
    // lineitem columns
    const int32_t* l_orderkey,
    const int32_t* l_shipdate,
    const int64_t* l_extendedprice,
    const int64_t* l_discount,
    size_t lineitem_count,
    // valid orders (已过滤BUILDING客户)
    const WeakHashTable<Q3AggResultV26>& valid_orders,
    const BloomFilter& valid_orders_bloom,  // 预构建的Bloom Filter
    // output
    MutableWeakHashTable<int64_t>& revenue_agg,
    int32_t date_threshold  // l_shipdate > threshold
);

// ============================================================================
// V26 优化版查询
// ============================================================================

void run_q3_v26(TPCHDataLoader& loader);
void run_q18_v26(TPCHDataLoader& loader);

// 其他查询沿用V25
inline void run_q5_v26(TPCHDataLoader& loader) { run_q5_v25(loader); }
inline void run_q6_v26(TPCHDataLoader& loader) { run_q6_v25(loader); }
inline void run_q7_v26(TPCHDataLoader& loader) { run_q7_v25(loader); }
inline void run_q9_v26(TPCHDataLoader& loader) { run_q9_v25(loader); }
inline void run_q10_v26(TPCHDataLoader& loader) { run_q10_v25(loader); }
inline void run_q12_v26(TPCHDataLoader& loader) { run_q12_v25(loader); }
inline void run_q14_v26(TPCHDataLoader& loader) { run_q14_v25(loader); }

} // namespace ops_v26
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V26_H
