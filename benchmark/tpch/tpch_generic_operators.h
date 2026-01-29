/**
 * ThunderDuck TPC-H V40 通用算子框架
 *
 * 提供可复用的通用算子模板:
 * - CompositeKeyEncoder: 复合键编码器
 * - SortedGroupByAggregator: 排序后聚合器
 * - MergeJoinOperator: 归并连接算子
 * - DynamicBitmapFilter: 动态位图过滤器
 *
 * @version 40.0
 * @date 2026-01-29
 */

#ifndef TPCH_GENERIC_OPERATORS_H
#define TPCH_GENERIC_OPERATORS_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <limits>
#include <cmath>

namespace thunderduck {
namespace tpch {
namespace generic {

// ============================================================================
// CompositeKeyEncoder - 复合键编码器
// ============================================================================

/**
 * 复合键编码器模板
 *
 * 将多个键合并为单一 64-bit 编码，用于高效比较和排序
 */
template<typename... KeyTypes>
class CompositeKeyEncoder;

/**
 * 双键特化 (int32_t, int32_t)
 *
 * 适用于 (partkey, suppkey) 等组合
 */
template<>
class CompositeKeyEncoder<int32_t, int32_t> {
public:
    using encoded_type = uint64_t;

    /**
     * 编码两个 int32_t 为 uint64_t
     * @param k1 第一个键 (高32位)
     * @param k2 第二个键 (低32位)
     * @return 编码后的 64-bit 值
     */
    __attribute__((always_inline))
    static constexpr encoded_type encode(int32_t k1, int32_t k2) noexcept {
        return (static_cast<uint64_t>(static_cast<uint32_t>(k1)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(k2));
    }

    /**
     * 解码 uint64_t 为两个 int32_t
     * @param e 编码值
     * @return (k1, k2) 对
     */
    __attribute__((always_inline))
    static constexpr std::pair<int32_t, int32_t> decode(encoded_type e) noexcept {
        return {static_cast<int32_t>(e >> 32),
                static_cast<int32_t>(e & 0xFFFFFFFF)};
    }

    /**
     * 获取第一个键
     */
    __attribute__((always_inline))
    static constexpr int32_t get_first(encoded_type e) noexcept {
        return static_cast<int32_t>(e >> 32);
    }

    /**
     * 获取第二个键
     */
    __attribute__((always_inline))
    static constexpr int32_t get_second(encoded_type e) noexcept {
        return static_cast<int32_t>(e & 0xFFFFFFFF);
    }
};

/**
 * 三键特化 (int32_t, int32_t, int16_t)
 *
 * 适用于 (orderkey, suppkey, flag) 等组合
 */
template<>
class CompositeKeyEncoder<int32_t, int32_t, int16_t> {
public:
    using encoded_type = uint64_t;

    __attribute__((always_inline))
    static constexpr encoded_type encode(int32_t k1, int32_t k2, int16_t k3) noexcept {
        // k1: 高 24 位, k2: 中 24 位, k3: 低 16 位
        return (static_cast<uint64_t>(k1 & 0xFFFFFF) << 40) |
               (static_cast<uint64_t>(k2 & 0xFFFFFF) << 16) |
               static_cast<uint64_t>(static_cast<uint16_t>(k3));
    }

    __attribute__((always_inline))
    static constexpr int32_t get_first(encoded_type e) noexcept {
        return static_cast<int32_t>((e >> 40) & 0xFFFFFF);
    }

    __attribute__((always_inline))
    static constexpr int32_t get_second(encoded_type e) noexcept {
        return static_cast<int32_t>((e >> 16) & 0xFFFFFF);
    }

    __attribute__((always_inline))
    static constexpr int16_t get_third(encoded_type e) noexcept {
        return static_cast<int16_t>(e & 0xFFFF);
    }
};

// ============================================================================
// SortedGroupByAggregator - 排序后聚合器
// ============================================================================

/**
 * 排序后单遍聚合器
 *
 * 对已排序数据进行 O(n) 聚合，无需 Hash 表
 *
 * @tparam KeyEncoder 键编码器类型
 * @tparam ValueType 值类型 (默认 int64_t)
 */
template<typename KeyEncoder, typename ValueType = int64_t>
class SortedGroupByAggregator {
public:
    using encoded_key_type = typename KeyEncoder::encoded_type;

    /**
     * 分组聚合结果
     */
    struct GroupResult {
        encoded_key_type key;
        int64_t sum;
        int64_t count;
    };

    /**
     * 对已排序数据进行单遍聚合
     *
     * @param sorted_keys 已排序的编码键数组
     * @param values 对应的值数组
     * @param count 元素数量
     * @return 分组聚合结果
     */
    std::vector<GroupResult> aggregate(
        const encoded_key_type* sorted_keys,
        const ValueType* values,
        size_t count
    ) {
        std::vector<GroupResult> results;
        if (count == 0) return results;

        results.reserve(count / 4);  // 预估分组数

        size_t i = 0;
        while (i < count) {
            encoded_key_type current_key = sorted_keys[i];
            int64_t sum = 0;
            int64_t group_count = 0;

            // 聚合相同键的所有值
            while (i < count && sorted_keys[i] == current_key) {
                sum += static_cast<int64_t>(values[i]);
                ++group_count;
                ++i;
            }

            results.push_back({current_key, sum, group_count});
        }

        return results;
    }

    /**
     * 对已排序的记录进行单遍聚合 (带索引提取)
     *
     * @tparam RecordType 记录类型
     * @tparam KeyExtractor 键提取函数 Record -> encoded_key_type
     * @tparam ValueExtractor 值提取函数 Record -> ValueType
     */
    template<typename RecordType, typename KeyExtractor, typename ValueExtractor>
    std::vector<GroupResult> aggregate_records(
        const std::vector<RecordType>& sorted_records,
        KeyExtractor key_fn,
        ValueExtractor value_fn
    ) {
        std::vector<GroupResult> results;
        if (sorted_records.empty()) return results;

        results.reserve(sorted_records.size() / 4);

        size_t i = 0;
        while (i < sorted_records.size()) {
            encoded_key_type current_key = key_fn(sorted_records[i]);
            int64_t sum = 0;
            int64_t group_count = 0;

            while (i < sorted_records.size() && key_fn(sorted_records[i]) == current_key) {
                sum += static_cast<int64_t>(value_fn(sorted_records[i]));
                ++group_count;
                ++i;
            }

            results.push_back({current_key, sum, group_count});
        }

        return results;
    }
};

// ============================================================================
// MergeJoinOperator - 归并连接算子
// ============================================================================

/**
 * 归并连接算子
 *
 * 对两个已排序数组进行归并连接，时间复杂度 O(n+m)
 *
 * @tparam LeftKey 左表键类型
 * @tparam RightKey 右表键类型
 */
template<typename LeftKey = int64_t, typename RightKey = int64_t>
class MergeJoinOperator {
public:
    /**
     * 连接对结果
     */
    struct JoinPair {
        size_t left_idx;
        size_t right_idx;
    };

    /**
     * 等值内连接
     *
     * @param left_keys 左表已排序键
     * @param left_count 左表元素数
     * @param right_keys 右表已排序键
     * @param right_count 右表元素数
     * @return 匹配的 (left_idx, right_idx) 对
     */
    std::vector<JoinPair> inner_join(
        const LeftKey* left_keys, size_t left_count,
        const RightKey* right_keys, size_t right_count
    ) {
        std::vector<JoinPair> results;
        results.reserve(std::min(left_count, right_count));

        size_t li = 0, ri = 0;

        while (li < left_count && ri < right_count) {
            if (left_keys[li] < static_cast<LeftKey>(right_keys[ri])) {
                ++li;
            } else if (left_keys[li] > static_cast<LeftKey>(right_keys[ri])) {
                ++ri;
            } else {
                // 处理多对多匹配
                size_t li_start = li;
                LeftKey match_key = left_keys[li];

                // 找到左边所有匹配
                while (li < left_count && left_keys[li] == match_key) {
                    size_t ri_temp = ri;
                    while (ri_temp < right_count &&
                           static_cast<LeftKey>(right_keys[ri_temp]) == match_key) {
                        results.push_back({li, ri_temp});
                        ++ri_temp;
                    }
                    ++li;
                }

                // 移动右边指针
                while (ri < right_count &&
                       static_cast<LeftKey>(right_keys[ri]) == match_key) {
                    ++ri;
                }
            }
        }

        return results;
    }

    /**
     * SEMI 连接 - 返回左表中存在于右表的索引
     *
     * @return 匹配的左表索引 (去重)
     */
    std::vector<size_t> semi_join(
        const LeftKey* left_keys, size_t left_count,
        const RightKey* right_keys, size_t right_count
    ) {
        std::vector<size_t> results;
        results.reserve(left_count);

        size_t li = 0, ri = 0;

        while (li < left_count && ri < right_count) {
            if (left_keys[li] < static_cast<LeftKey>(right_keys[ri])) {
                ++li;
            } else if (left_keys[li] > static_cast<LeftKey>(right_keys[ri])) {
                ++ri;
            } else {
                // 记录匹配的左表索引
                LeftKey match_key = left_keys[li];
                while (li < left_count && left_keys[li] == match_key) {
                    results.push_back(li);
                    ++li;
                }
                // 移动右边指针跳过相同键
                while (ri < right_count &&
                       static_cast<LeftKey>(right_keys[ri]) == match_key) {
                    ++ri;
                }
            }
        }

        return results;
    }

    /**
     * 带谓词的连接
     *
     * @tparam Predicate 谓词函数 (left_idx, right_idx) -> bool
     */
    template<typename Predicate>
    std::vector<JoinPair> join_with_predicate(
        const LeftKey* left_keys, size_t left_count,
        const RightKey* right_keys, size_t right_count,
        Predicate pred
    ) {
        std::vector<JoinPair> results;
        results.reserve(std::min(left_count, right_count));

        size_t li = 0, ri = 0;

        while (li < left_count && ri < right_count) {
            if (left_keys[li] < static_cast<LeftKey>(right_keys[ri])) {
                ++li;
            } else if (left_keys[li] > static_cast<LeftKey>(right_keys[ri])) {
                ++ri;
            } else {
                size_t li_start = li;
                LeftKey match_key = left_keys[li];

                while (li < left_count && left_keys[li] == match_key) {
                    size_t ri_temp = ri;
                    while (ri_temp < right_count &&
                           static_cast<LeftKey>(right_keys[ri_temp]) == match_key) {
                        if (pred(li, ri_temp)) {
                            results.push_back({li, ri_temp});
                        }
                        ++ri_temp;
                    }
                    ++li;
                }

                while (ri < right_count &&
                       static_cast<LeftKey>(right_keys[ri]) == match_key) {
                    ++ri;
                }
            }
        }

        return results;
    }
};

// ============================================================================
// DynamicBitmapFilter - 动态位图过滤器
// ============================================================================

/**
 * 动态位图过滤器
 *
 * 自动检测键范围，选择最优存储策略:
 * - 小范围: 使用 bitmap (O(1) 测试)
 * - 大范围: 使用 hash_set (O(1) 平均测试)
 *
 * 消除硬编码的范围假设
 */
class DynamicBitmapFilter {
public:
    static constexpr size_t DEFAULT_MAX_BITMAP_SIZE = 16 * 1024 * 1024;  // 16MB

    DynamicBitmapFilter() = default;

    /**
     * 从键数组构建过滤器
     *
     * @param keys 键数组
     * @param count 元素数量
     * @param max_bitmap_size 位图最大字节数
     */
    void build(const int32_t* keys, size_t count,
               size_t max_bitmap_size = DEFAULT_MAX_BITMAP_SIZE) {
        if (count == 0) {
            using_bitmap_ = true;
            bitmap_.clear();
            min_key_ = 0;
            max_key_ = 0;
            return;
        }

        // 扫描确定范围
        min_key_ = std::numeric_limits<int32_t>::max();
        max_key_ = std::numeric_limits<int32_t>::min();

        for (size_t i = 0; i < count; ++i) {
            min_key_ = std::min(min_key_, keys[i]);
            max_key_ = std::max(max_key_, keys[i]);
        }

        int64_t range = static_cast<int64_t>(max_key_) - min_key_ + 1;
        size_t required_bytes = (range + 7) / 8;

        // 选择策略
        if (required_bytes <= max_bitmap_size && range > 0) {
            using_bitmap_ = true;
            bitmap_.resize(range, false);
            hash_set_.clear();

            for (size_t i = 0; i < count; ++i) {
                bitmap_[keys[i] - min_key_] = true;
            }
        } else {
            using_bitmap_ = false;
            bitmap_.clear();
            hash_set_.clear();
            hash_set_.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                hash_set_.insert(keys[i]);
            }
        }
    }

    /**
     * 带过滤的构建
     *
     * @tparam Predicate 谓词函数 size_t idx -> bool
     * @param keys 键数组
     * @param count 元素数量
     * @param pred 过滤谓词
     * @param max_bitmap_size 位图最大字节数
     */
    template<typename Predicate>
    void build_filtered(const int32_t* keys, size_t count, Predicate pred,
                        size_t max_bitmap_size = DEFAULT_MAX_BITMAP_SIZE) {
        // 先收集满足条件的键
        std::vector<int32_t> filtered_keys;
        filtered_keys.reserve(count / 4);

        for (size_t i = 0; i < count; ++i) {
            if (pred(i)) {
                filtered_keys.push_back(keys[i]);
            }
        }

        // 使用收集的键构建
        build(filtered_keys.data(), filtered_keys.size(), max_bitmap_size);
    }

    /**
     * 测试键是否存在
     *
     * @param key 要测试的键
     * @return 是否存在
     */
    __attribute__((always_inline))
    bool test(int32_t key) const noexcept {
        if (using_bitmap_) {
            if (key < min_key_ || key > max_key_) return false;
            return bitmap_[key - min_key_];
        } else {
            return hash_set_.count(key) > 0;
        }
    }

    /**
     * 是否使用位图策略
     */
    bool is_using_bitmap() const noexcept { return using_bitmap_; }

    /**
     * 获取检测到的键范围
     */
    std::pair<int32_t, int32_t> get_key_range() const noexcept {
        return {min_key_, max_key_};
    }

    /**
     * 获取存储的键数量
     */
    size_t size() const noexcept {
        if (using_bitmap_) {
            size_t cnt = 0;
            for (bool b : bitmap_) if (b) ++cnt;
            return cnt;
        } else {
            return hash_set_.size();
        }
    }

    /**
     * 清空过滤器
     */
    void clear() {
        bitmap_.clear();
        hash_set_.clear();
        using_bitmap_ = true;
        min_key_ = 0;
        max_key_ = 0;
    }

private:
    bool using_bitmap_ = true;
    int32_t min_key_ = 0;
    int32_t max_key_ = 0;
    std::vector<bool> bitmap_;
    std::unordered_set<int32_t> hash_set_;
};

// ============================================================================
// SortedRecordBuffer - 排序记录缓冲区
// ============================================================================

/**
 * 排序记录缓冲区
 *
 * 用于收集、排序和处理记录流
 */
template<typename RecordType>
class SortedRecordBuffer {
public:
    /**
     * 预留空间
     */
    void reserve(size_t capacity) {
        records_.reserve(capacity);
    }

    /**
     * 添加记录
     */
    template<typename... Args>
    void emplace(Args&&... args) {
        records_.emplace_back(std::forward<Args>(args)...);
    }

    /**
     * 排序
     */
    template<typename Compare>
    void sort(Compare comp) {
        std::sort(records_.begin(), records_.end(), comp);
    }

    /**
     * 使用默认比较排序
     */
    void sort() {
        std::sort(records_.begin(), records_.end());
    }

    /**
     * 获取记录数组
     */
    const std::vector<RecordType>& records() const { return records_; }
    std::vector<RecordType>& records() { return records_; }

    /**
     * 记录数量
     */
    size_t size() const { return records_.size(); }

    /**
     * 是否为空
     */
    bool empty() const { return records_.empty(); }

    /**
     * 清空
     */
    void clear() { records_.clear(); }

private:
    std::vector<RecordType> records_;
};

} // namespace generic
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_GENERIC_OPERATORS_H
