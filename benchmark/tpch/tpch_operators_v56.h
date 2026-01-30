/**
 * ThunderDuck TPC-H Operators V56
 *
 * V56 优化内容:
 * 1. Q5 预计算 order → cust_nation (消除热路径第 3 次 hash 查找)
 * 2. SubqueryDecorrelation: Direct Array 优化 (key 范围小时)
 * 3. GenericParallelMultiJoin: Bloom Filter 预过滤
 * 4. GenericTwoPhaseAgg: Direct Array 优化
 *
 * @version 56
 * @date 2026-01-30
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_operators_v32.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <cstring>
#include <limits>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v56 {

using namespace ops_v32;  // 继承 V32 工具类

// ============================================================================
// 版本信息
// ============================================================================

extern const char* V56_VERSION;
extern const char* V56_DATE;

// ============================================================================
// L2 缓存感知阈值计算 (消除硬编码)
// ============================================================================

/**
 * 运行时获取 L2 缓存大小 (字节)
 * Apple M4: 4MB per performance core
 */
inline size_t get_l2_cache_size() {
#ifdef __APPLE__
    size_t size = 0;
    size_t len = sizeof(size);
    // 尝试获取 L2 缓存大小
    if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0 && size > 0) {
        return size;
    }
    // 回退: Apple Silicon 默认 4MB
    return 4 * 1024 * 1024;
#else
    // 非 Apple 平台默认 256KB
    return 256 * 1024;
#endif
}

/**
 * 计算直接数组最大条目数 (基于 L2 缓存)
 *
 * @tparam EntrySize 每个条目的字节数
 * @param cache_utilization 缓存利用率 (默认 0.5, 留空间给其他数据)
 */
template<size_t EntrySize>
inline size_t compute_max_direct_entries(double cache_utilization = 0.5) {
    size_t l2_size = get_l2_cache_size();
    return static_cast<size_t>(l2_size * cache_utilization / EntrySize);
}

// ============================================================================
// 通用算子: DirectArrayDecorrelation (直接数组解关联)
// ============================================================================

/**
 * 直接数组解关联器 (适用于 key 范围适合 L2 缓存的情况)
 *
 * 阈值自动基于 L2 缓存大小计算:
 *   max_entries = L2_size * 0.5 / sizeof(ValueT)
 *
 * Apple M4 (4MB L2): ~250K int64 entries
 */
template<typename ValueT>
class DirectArrayDecorrelation {
public:
    // 基于 L2 缓存动态计算阈值，而非硬编码
    static size_t max_direct_size() {
        static size_t cached = compute_max_direct_entries<sizeof(ValueT) + sizeof(bool)>();
        return cached;
    }

    using AggFunc = std::function<ValueT(ValueT, ValueT)>;

    /**
     * 预计算聚合结果 (自动选择最优存储)
     *
     * @param keys 分组键数组
     * @param values 聚合值数组
     * @param count 元素数量
     * @param agg_func 聚合函数
     * @param filter 过滤条件 (可选)
     */
    void precompute(
        const int32_t* keys,
        const ValueT* values,
        size_t count,
        AggFunc agg_func,
        std::function<bool(size_t)> filter = nullptr
    ) {
        // 第一遍：找到 key 范围
        int32_t min_key = INT32_MAX, max_key = INT32_MIN;
        for (size_t i = 0; i < count; ++i) {
            if (filter && !filter(i)) continue;
            min_key = std::min(min_key, keys[i]);
            max_key = std::max(max_key, keys[i]);
        }

        key_offset_ = min_key;
        size_t range = static_cast<size_t>(max_key - min_key + 1);

        if (range <= max_direct_size() && range > 0) {
            // 使用直接数组
            use_direct_ = true;
            direct_values_.resize(range);
            direct_valid_.resize(range, false);

            for (size_t i = 0; i < count; ++i) {
                if (filter && !filter(i)) continue;

                size_t idx = static_cast<size_t>(keys[i] - key_offset_);
                if (!direct_valid_[idx]) {
                    direct_values_[idx] = values[i];
                    direct_valid_[idx] = true;
                } else {
                    direct_values_[idx] = agg_func(direct_values_[idx], values[i]);
                }
            }
        } else {
            // 回退到 hash map
            use_direct_ = false;
            hash_map_.clear();
            hash_map_.reserve(count / 10);

            for (size_t i = 0; i < count; ++i) {
                if (filter && !filter(i)) continue;

                auto it = hash_map_.find(keys[i]);
                if (it == hash_map_.end()) {
                    hash_map_[keys[i]] = values[i];
                } else {
                    it->second = agg_func(it->second, values[i]);
                }
            }
        }
    }

    /**
     * O(1) 查找
     */
    bool lookup(int32_t key, ValueT& result) const {
        if (use_direct_) {
            size_t idx = static_cast<size_t>(key - key_offset_);
            if (idx < direct_valid_.size() && direct_valid_[idx]) {
                result = direct_values_[idx];
                return true;
            }
            return false;
        } else {
            auto it = hash_map_.find(key);
            if (it != hash_map_.end()) {
                result = it->second;
                return true;
            }
            return false;
        }
    }

    /**
     * 批量 O(1) 查找 (8 路)
     */
    void batch_lookup(const int32_t* keys, ValueT* results, bool* found, size_t n = 8) const {
        if (use_direct_) {
            for (size_t i = 0; i < n; ++i) {
                size_t idx = static_cast<size_t>(keys[i] - key_offset_);
                if (idx < direct_valid_.size() && direct_valid_[idx]) {
                    results[i] = direct_values_[idx];
                    found[i] = true;
                } else {
                    found[i] = false;
                }
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                auto it = hash_map_.find(keys[i]);
                if (it != hash_map_.end()) {
                    results[i] = it->second;
                    found[i] = true;
                } else {
                    found[i] = false;
                }
            }
        }
    }

    bool uses_direct_array() const { return use_direct_; }
    size_t size() const { return use_direct_ ? direct_values_.size() : hash_map_.size(); }

private:
    bool use_direct_ = false;
    int32_t key_offset_ = 0;
    std::vector<ValueT> direct_values_;
    std::vector<bool> direct_valid_;
    std::unordered_map<int32_t, ValueT> hash_map_;
};

// ============================================================================
// 通用算子: BloomFilteredJoin (带 Bloom Filter 预过滤的 JOIN)
// ============================================================================

/**
 * 带 Bloom Filter 预过滤的 Hash Join
 *
 * 对于选择率低的 JOIN，先用 Bloom Filter 快速排除不匹配的行
 */
template<typename ValueT>
class BloomFilteredJoin {
public:
    static constexpr size_t BLOOM_BITS_PER_ELEMENT = 8;

    void build(const int32_t* keys, const ValueT* values, size_t count,
               std::function<bool(size_t)> filter = nullptr) {
        // 初始化 Bloom Filter
        bloom_.init(count, BLOOM_BITS_PER_ELEMENT);

        // 初始化 Hash Table
        hash_table_.init(count);

        for (size_t i = 0; i < count; ++i) {
            if (filter && !filter(i)) continue;
            bloom_.insert(keys[i]);
            hash_table_.insert(keys[i], values[i]);
        }
    }

    /**
     * 探测 (带 Bloom Filter 预过滤)
     */
    const ValueT* probe(int32_t key) const {
        // Bloom Filter 快速检查
        if (!bloom_.may_contain(key)) return nullptr;

        // Hash Table 精确查找
        return hash_table_.find(key);
    }

    /**
     * 批量探测 (8 路)
     */
    void batch_probe(const int32_t* keys, const ValueT** results) const {
        // Bloom Filter 批量测试
        uint8_t bloom_mask = bloom_.batch_test(keys);

        // 只查找可能存在的
        for (int i = 0; i < 8; ++i) {
            if (bloom_mask & (1 << i)) {
                results[i] = hash_table_.find(keys[i]);
            } else {
                results[i] = nullptr;
            }
        }
    }

    size_t size() const { return hash_table_.size(); }

private:
    SingleHashBloomFilter bloom_;
    CompactHashTable<ValueT> hash_table_;
};

// ============================================================================
// 通用算子: DirectArrayTwoPhaseAgg (直接数组两阶段聚合)
// ============================================================================

/**
 * 直接数组两阶段聚合器
 *
 * 当 key 范围小时使用直接数组存储预聚合结果
 */
template<typename ValueT>
class DirectArrayTwoPhaseAgg {
public:
    // 基于 L2 缓存动态计算阈值 (AggResult = sum + count)
    static size_t max_direct_size() {
        static size_t cached = compute_max_direct_entries<sizeof(ValueT) + sizeof(size_t) + sizeof(bool)>();
        return cached;
    }

    struct AggResult {
        ValueT sum = 0;
        size_t count = 0;

        ValueT avg() const { return count > 0 ? sum / static_cast<ValueT>(count) : 0; }
    };

    /**
     * Phase 1: 预计算每个键的聚合值
     */
    void phase1_precompute(
        const int32_t* keys,
        const ValueT* values,
        size_t count,
        std::function<bool(size_t)> filter = nullptr
    ) {
        // 找到 key 范围
        int32_t min_key = INT32_MAX, max_key = INT32_MIN;
        for (size_t i = 0; i < count; ++i) {
            if (filter && !filter(i)) continue;
            min_key = std::min(min_key, keys[i]);
            max_key = std::max(max_key, keys[i]);
        }

        key_offset_ = min_key;
        size_t range = static_cast<size_t>(max_key - min_key + 1);

        if (range <= max_direct_size() && range > 0) {
            use_direct_ = true;
            direct_agg_.resize(range);
            direct_valid_.resize(range, false);

            for (size_t i = 0; i < count; ++i) {
                if (filter && !filter(i)) continue;

                size_t idx = static_cast<size_t>(keys[i] - key_offset_);
                direct_agg_[idx].sum += values[i];
                direct_agg_[idx].count++;
                direct_valid_[idx] = true;
            }
        } else {
            use_direct_ = false;
            hash_agg_.clear();
            hash_agg_.reserve(count / 10);

            for (size_t i = 0; i < count; ++i) {
                if (filter && !filter(i)) continue;

                auto& agg = hash_agg_[keys[i]];
                agg.sum += values[i];
                agg.count++;
            }
        }
    }

    /**
     * 查找预聚合结果
     */
    bool lookup(int32_t key, AggResult& result) const {
        if (use_direct_) {
            size_t idx = static_cast<size_t>(key - key_offset_);
            if (idx < direct_valid_.size() && direct_valid_[idx]) {
                result = direct_agg_[idx];
                return true;
            }
            return false;
        } else {
            auto it = hash_agg_.find(key);
            if (it != hash_agg_.end()) {
                result = it->second;
                return true;
            }
            return false;
        }
    }

    bool uses_direct_array() const { return use_direct_; }

private:
    bool use_direct_ = false;
    int32_t key_offset_ = 0;
    std::vector<AggResult> direct_agg_;
    std::vector<bool> direct_valid_;
    std::unordered_map<int32_t, AggResult> hash_agg_;
};

// ============================================================================
// Q5 优化: 预计算 order → cust_nation (消除第 3 次 hash 查找)
// ============================================================================

/**
 * Q5: 本地供应商收入 (V56 - 消除热路径额外查找)
 *
 * 原始 V32 热路径:
 *   1. order_to_cust.find(orderkey) → custkey
 *   2. supp_to_nation.find(suppkey) → supp_nation
 *   3. cust_to_nation.find(custkey) → cust_nation  ← 额外查找!
 *
 * V56 优化:
 *   预计算 orderkey → cust_nation 直接映射
 *   热路径只需 2 次查找
 */
void run_q5_v56(TPCHDataLoader& loader);

// ============================================================================
// Q2 优化: Direct Array 解关联
// ============================================================================

/**
 * Q2: 最小成本供应商 (V56 - Direct Array 解关联)
 *
 * partkey 范围通常 <= 200K (SF=1)，使用直接数组 O(1) 查找
 */
void run_q2_v56(TPCHDataLoader& loader);

// ============================================================================
// Q8 优化: Bloom Filter + 预计算维度
// ============================================================================

/**
 * Q8: 市场份额 (V56 - Bloom Filter 预过滤 + 预计算维度)
 */
void run_q8_v56(TPCHDataLoader& loader);

// ============================================================================
// Q17 优化: Direct Array 两阶段聚合
// ============================================================================

/**
 * Q17: 小订单收入 (V56 - Direct Array 两阶段聚合)
 */
void run_q17_v56(TPCHDataLoader& loader);

// ============================================================================
// 适用性检查
// ============================================================================

inline bool is_v56_applicable(const std::string& query_id, size_t rows) {
    if (query_id == "Q2") return rows >= 5000;
    if (query_id == "Q5") return rows >= 50000;
    if (query_id == "Q8") return rows >= 100000;
    if (query_id == "Q17") return rows >= 30000;
    return false;
}

} // namespace ops_v56
} // namespace tpch
} // namespace thunderduck
