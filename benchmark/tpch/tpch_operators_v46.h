/**
 * ThunderDuck TPC-H V46 通用化算子
 *
 * 基于 V45 直接数组优化，消除所有硬编码
 *
 * 通用算子:
 * - DirectArrayFilter: 直接数组过滤器 (自动检测 key 范围)
 * - BitmapMembershipFilter: 位图成员过滤器
 * - DirectArrayAggregator: 直接数组聚合器
 *
 * @version 46.0
 * @date 2026-01-29
 */

#ifndef TPCH_OPERATORS_V46_H
#define TPCH_OPERATORS_V46_H

#include "tpch_data_loader.h"
#include "tpch_config_v33.h"  // DateRange, QueryConfig
#include <vector>
#include <functional>
#include <string>

namespace thunderduck {
namespace tpch {
namespace ops_v46 {

// ============================================================================
// 通用直接数组过滤器
// ============================================================================

/**
 * DirectArrayFilter - 自动检测 key 范围的直接数组
 *
 * 适用场景: key 范围较小 (< 1M)，需要 O(1) 查找
 *
 * 用法:
 *   DirectArrayFilter<uint8_t> filter;
 *   filter.build(keys, count, [&](size_t i) { return predicate(i); });
 *   if (filter.test(key)) { ... }
 */
template<typename ValueType = uint8_t>
class DirectArrayFilter {
public:
    // 自动检测 key 范围并构建
    template<typename Predicate>
    void build(const int32_t* keys, size_t count, Predicate pred) {
        // 检测 max_key
        max_key_ = 0;
        for (size_t i = 0; i < count; ++i) {
            if (keys[i] > max_key_) max_key_ = keys[i];
        }

        // 分配数组
        data_.assign(max_key_ + 1, default_value_);

        // 填充
        for (size_t i = 0; i < count; ++i) {
            if (pred(i)) {
                data_[keys[i]] = match_value_;
            }
        }
    }

    // 带值的构建
    template<typename ValueFunc>
    void build_with_value(const int32_t* keys, size_t count, ValueFunc value_func) {
        max_key_ = 0;
        for (size_t i = 0; i < count; ++i) {
            if (keys[i] > max_key_) max_key_ = keys[i];
        }

        data_.assign(max_key_ + 1, default_value_);

        for (size_t i = 0; i < count; ++i) {
            data_[keys[i]] = value_func(i);
        }
    }

    // O(1) 测试
    bool test(int32_t key) const {
        return key >= 0 && key <= max_key_ && data_[key] == match_value_;
    }

    // 获取值
    ValueType get(int32_t key) const {
        if (key < 0 || key > max_key_) return default_value_;
        return data_[key];
    }

    const ValueType* data() const { return data_.data(); }
    int32_t max_key() const { return max_key_; }

    void set_default(ValueType v) { default_value_ = v; }
    void set_match(ValueType v) { match_value_ = v; }

private:
    std::vector<ValueType> data_;
    int32_t max_key_ = 0;
    ValueType default_value_ = 0;
    ValueType match_value_ = 1;
};

// ============================================================================
// 通用位图成员过滤器
// ============================================================================

/**
 * BitmapMembershipFilter - 紧凑位图
 *
 * 内存: max_key / 8 字节
 * 适用: 大量 key 的成员测试
 */
class BitmapMembershipFilter {
public:
    template<typename Predicate>
    void build(const int32_t* keys, size_t count, Predicate pred) {
        // 检测 max_key
        max_key_ = 0;
        for (size_t i = 0; i < count; ++i) {
            if (keys[i] > max_key_) max_key_ = keys[i];
        }

        // 分配位图
        bitmap_.assign((max_key_ + 8) / 8, 0);
        member_count_ = 0;

        // 填充
        for (size_t i = 0; i < count; ++i) {
            if (pred(i)) {
                int32_t k = keys[i];
                bitmap_[k >> 3] |= (1u << (k & 7));
                member_count_++;
            }
        }
    }

    // O(1) 测试
    bool test(int32_t key) const {
        if (key < 0 || key > max_key_) return false;
        return (bitmap_[key >> 3] & (1u << (key & 7))) != 0;
    }

    const uint8_t* data() const { return bitmap_.data(); }
    int32_t max_key() const { return max_key_; }
    size_t member_count() const { return member_count_; }

private:
    std::vector<uint8_t> bitmap_;
    int32_t max_key_ = 0;
    size_t member_count_ = 0;
};

// ============================================================================
// 通用直接数组聚合器
// ============================================================================

/**
 * DirectArrayAggregator - 直接数组聚合
 *
 * 适用: GROUP BY key 范围较小的场景
 */
template<typename ValueType = int64_t>
class DirectArrayAggregator {
public:
    void init(int32_t max_key) {
        max_key_ = max_key;
        data_.assign(max_key + 1, 0);
    }

    // 自动检测 max_key
    void init_from_keys(const int32_t* keys, size_t count) {
        max_key_ = 0;
        for (size_t i = 0; i < count; ++i) {
            if (keys[i] > max_key_) max_key_ = keys[i];
        }
        data_.assign(max_key_ + 1, 0);
    }

    void add(int32_t key, ValueType value) {
        if (key >= 0 && key <= max_key_) {
            data_[key] += value;
        }
    }

    ValueType get(int32_t key) const {
        if (key < 0 || key > max_key_) return 0;
        return data_[key];
    }

    // 合并另一个聚合器
    void merge(const DirectArrayAggregator& other) {
        for (int32_t k = 0; k <= std::min(max_key_, other.max_key_); ++k) {
            data_[k] += other.data_[k];
        }
    }

    // 遍历非零元素
    template<typename Func>
    void for_each_nonzero(Func func) const {
        for (int32_t k = 0; k <= max_key_; ++k) {
            if (data_[k] != 0) {
                func(k, data_[k]);
            }
        }
    }

    ValueType* data() { return data_.data(); }
    const ValueType* data() const { return data_.data(); }
    int32_t max_key() const { return max_key_; }

private:
    std::vector<ValueType> data_;
    int32_t max_key_ = 0;
};

// ============================================================================
// Q14 通用化配置
// ============================================================================

struct Q14Config {
    ops_v33::DateRange date_range;  // 日期范围
    std::string type_prefix;        // 类型前缀 (如 "PROMO")

    Q14Config() : type_prefix("PROMO") {
        date_range.lo = 9374;       // 默认 1995-09-01
        date_range.hi = 9404;       // 默认 1995-10-01
    }
};

void run_q14_v46(TPCHDataLoader& loader, const Q14Config& config);

// ============================================================================
// Q11 通用化配置
// ============================================================================

struct Q11Config {
    std::string target_nation;      // 目标国家
    double threshold_factor;        // 阈值因子

    Q11Config() : target_nation("GERMANY"), threshold_factor(0.0001) {}
};

void run_q11_v46(TPCHDataLoader& loader, const Q11Config& config);

// ============================================================================
// Q5 通用化配置
// ============================================================================

struct Q5Config {
    ops_v33::DateRange date_range;  // 日期范围
    std::string target_region;      // 目标区域

    Q5Config() : target_region("ASIA") {
        date_range.lo = 8766;       // 默认 1994-01-01
        date_range.hi = 9131;       // 默认 1995-01-01
    }
};

void run_q5_v46(TPCHDataLoader& loader, const Q5Config& config);

// ============================================================================
// 默认参数版本 (兼容现有接口)
// ============================================================================

inline void run_q14_v46(TPCHDataLoader& loader) {
    run_q14_v46(loader, Q14Config{});
}

inline void run_q11_v46(TPCHDataLoader& loader) {
    run_q11_v46(loader, Q11Config{});
}

inline void run_q5_v46(TPCHDataLoader& loader) {
    run_q5_v46(loader, Q5Config{});
}

}  // namespace ops_v46
}  // namespace tpch
}  // namespace thunderduck

#endif  // TPCH_OPERATORS_V46_H
