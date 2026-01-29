/**
 * ThunderDuck TPC-H 算子封装 V27
 *
 * P0 优先级优化:
 * - Q4: Bitmap SEMI Join (替代 GPU SEMI Join)
 * - Q11: 单遍扫描 + 后置过滤 (替代两次扫描)
 * - Q16: 并行 Anti-Join + BloomFilter 预过滤
 *
 * @version 27.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V27_H
#define TPCH_OPERATORS_V27_H

#include "tpch_data_loader.h"
#include "tpch_operators_v26.h"
#include "tpch_queries.h"
#include "thunderduck/bloom_filter.h"
#include <cstdint>
#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>

#ifdef __aarch64__
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v27 {

using ::thunderduck::tpch::TPCHDataLoader;
using namespace ops_v26;  // 继承 V26 的所有优化
namespace queries = ::thunderduck::tpch::queries;

// ============================================================================
// Bitmap 工具类 - 用于 SEMI/ANTI Join
// ============================================================================

/**
 * 并发安全的 Bitmap
 *
 * 用于 EXISTS/NOT EXISTS 子查询优化:
 * - O(1) 设置和测试
 * - 原子操作保证线程安全
 * - 缓存友好的 64-bit 块
 */
class ConcurrentBitmap {
public:
    ConcurrentBitmap() = default;
    ~ConcurrentBitmap() = default;

    // 禁止拷贝
    ConcurrentBitmap(const ConcurrentBitmap&) = delete;
    ConcurrentBitmap& operator=(const ConcurrentBitmap&) = delete;

    // 允许移动
    ConcurrentBitmap(ConcurrentBitmap&&) = default;
    ConcurrentBitmap& operator=(ConcurrentBitmap&&) = default;

    /**
     * 初始化 bitmap
     * @param max_value 最大值 (bitmap 大小 = max_value / 64 + 1)
     */
    void init(size_t max_value) {
        num_words_ = (max_value + 63) / 64;
        bits_ = std::make_unique<std::atomic<uint64_t>[]>(num_words_);
        for (size_t i = 0; i < num_words_; ++i) {
            bits_[i].store(0, std::memory_order_relaxed);
        }
        max_value_ = max_value;
    }

    /**
     * 原子设置位 (线程安全)
     */
    void set(uint32_t value) {
        if (value > max_value_) return;
        size_t word_idx = value / 64;
        uint64_t bit_mask = 1ULL << (value % 64);
        bits_[word_idx].fetch_or(bit_mask, std::memory_order_relaxed);
    }

    /**
     * 测试位是否设置
     */
    bool test(uint32_t value) const {
        if (value > max_value_) return false;
        size_t word_idx = value / 64;
        uint64_t bit_mask = 1ULL << (value % 64);
        return (bits_[word_idx].load(std::memory_order_relaxed) & bit_mask) != 0;
    }

    /**
     * 批量测试 (返回匹配的索引)
     */
    size_t batch_test(const int32_t* keys, size_t n,
                      uint32_t* out_indices) const {
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (test(static_cast<uint32_t>(keys[i]))) {
                out_indices[count++] = static_cast<uint32_t>(i);
            }
        }
        return count;
    }

    /**
     * 批量反向测试 (返回不匹配的索引) - 用于 Anti-Join
     */
    size_t batch_anti_test(const int32_t* keys, size_t n,
                           uint32_t* out_indices) const {
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!test(static_cast<uint32_t>(keys[i]))) {
                out_indices[count++] = static_cast<uint32_t>(i);
            }
        }
        return count;
    }

    void clear() {
        for (size_t i = 0; i < num_words_; ++i) {
            bits_[i].store(0, std::memory_order_relaxed);
        }
    }

private:
    std::unique_ptr<std::atomic<uint64_t>[]> bits_;
    size_t num_words_ = 0;
    size_t max_value_ = 0;
};

// ============================================================================
// 通用工具类: StringDictionary - 字符串编码 (GROUP BY 优化)
// ============================================================================

/**
 * 通用字符串字典
 *
 * 将字符串映射为连续整数 ID，用于加速 GROUP BY:
 * - 编码: string -> int32_t, O(1) 平均
 * - 解码: int32_t -> string, O(1)
 * - 哈希成本从 O(字符串长度) 降至 O(1)
 *
 * 适用场景:
 * - 任何字符串作为 GROUP BY 键的查询
 * - COUNT(DISTINCT string_column)
 * - 多表 JOIN 的字符串键
 */
class StringDictionary {
public:
    StringDictionary() = default;

    /**
     * 预分配容量
     */
    void reserve(size_t capacity) {
        str_to_id_.reserve(capacity);
        id_to_str_.reserve(capacity);
    }

    /**
     * 编码字符串 (如果不存在则添加)
     * @return 字符串对应的整数 ID
     */
    int32_t encode(const std::string& str) {
        auto it = str_to_id_.find(str);
        if (it != str_to_id_.end()) {
            return it->second;
        }
        int32_t id = static_cast<int32_t>(id_to_str_.size());
        str_to_id_[str] = id;
        id_to_str_.push_back(str);
        return id;
    }

    /**
     * 编码字符串 (使用指针避免拷贝，字符串必须在字典生命周期内有效)
     */
    int32_t encode_ref(const std::string* str_ptr) {
        auto it = str_to_id_.find(*str_ptr);
        if (it != str_to_id_.end()) {
            return it->second;
        }
        int32_t id = static_cast<int32_t>(id_to_str_.size());
        str_to_id_[*str_ptr] = id;
        id_to_str_.push_back(*str_ptr);
        return id;
    }

    /**
     * 查询编码 (不添加新条目)
     * @return ID 或 -1 (如果不存在)
     */
    int32_t lookup(const std::string& str) const {
        auto it = str_to_id_.find(str);
        return (it != str_to_id_.end()) ? it->second : -1;
    }

    /**
     * 解码 ID 为字符串
     */
    const std::string& decode(int32_t id) const {
        return id_to_str_[id];
    }

    /**
     * 字典大小
     */
    size_t size() const { return id_to_str_.size(); }

    /**
     * 清空字典
     */
    void clear() {
        str_to_id_.clear();
        id_to_str_.clear();
    }

private:
    std::unordered_map<std::string, int32_t> str_to_id_;
    std::vector<std::string> id_to_str_;
};

// ============================================================================
// 通用工具类: BitmapDistinctCounter - COUNT(DISTINCT) 优化
// ============================================================================

/**
 * 基于 Bitmap 的 COUNT(DISTINCT) 聚合器
 *
 * 用于高效计算 COUNT(DISTINCT int_column):
 * - 插入: O(1) - bitmap set
 * - 计数: O(n/64) - popcount
 * - 空间: max_value / 8 bytes
 *
 * 适用场景:
 * - COUNT(DISTINCT suppkey/custkey/orderkey)
 * - 任何有界整数列的去重计数
 *
 * 限制:
 * - 仅适用于整数列
 * - 需要知道最大值范围
 */
class BitmapDistinctCounter {
public:
    BitmapDistinctCounter() = default;

    /**
     * 初始化 (设置最大可能值)
     */
    void init(size_t max_value) {
        num_words_ = (max_value + 63) / 64;
        bits_.resize(num_words_, 0);
        max_value_ = max_value;
    }

    /**
     * 添加值到去重集合
     */
    void add(int32_t value) {
        if (static_cast<size_t>(value) > max_value_) return;
        size_t word_idx = value / 64;
        uint64_t bit_mask = 1ULL << (value % 64);
        bits_[word_idx] |= bit_mask;
    }

    /**
     * 计算去重数量 (popcount)
     */
    size_t count() const {
        size_t total = 0;
        for (size_t i = 0; i < num_words_; ++i) {
            total += __builtin_popcountll(bits_[i]);
        }
        return total;
    }

    /**
     * 清空
     */
    void clear() {
        std::fill(bits_.begin(), bits_.end(), 0);
    }

    /**
     * 合并另一个 counter (用于并行聚合)
     */
    void merge(const BitmapDistinctCounter& other) {
        for (size_t i = 0; i < std::min(num_words_, other.num_words_); ++i) {
            bits_[i] |= other.bits_[i];
        }
    }

private:
    std::vector<uint64_t> bits_;
    size_t num_words_ = 0;
    size_t max_value_ = 0;
};

// ============================================================================
// 通用工具类: EncodedGroupKey - 编码的 GROUP BY 键
// ============================================================================

/**
 * 编码的 GROUP BY 键
 *
 * 将多列 GROUP BY 键编码为单个 int64_t:
 * - 哈希: O(1) - 直接用 int64 哈希
 * - 比较: O(1) - 整数比较
 *
 * 格式: (col1_id << 32) | (col2_id << 16) | col3_value
 * 支持最多 3 列，每列最大 65535 个不同值
 */
struct EncodedGroupKey {
    int64_t key;

    bool operator==(const EncodedGroupKey& o) const { return key == o.key; }

    static EncodedGroupKey make(int32_t col1, int32_t col2, int32_t col3) {
        return {(static_cast<int64_t>(col1) << 32) |
                (static_cast<int64_t>(col2 & 0xFFFF) << 16) |
                (col3 & 0xFFFF)};
    }

    int32_t get_col1() const { return static_cast<int32_t>(key >> 32); }
    int32_t get_col2() const { return static_cast<int32_t>((key >> 16) & 0xFFFF); }
    int32_t get_col3() const { return static_cast<int32_t>(key & 0xFFFF); }
};

struct EncodedGroupKeyHash {
    size_t operator()(const EncodedGroupKey& k) const {
        return std::hash<int64_t>()(k.key);
    }
};

// ============================================================================
// 通用谓词预计算器: PredicatePrecomputer
// ============================================================================

/**
 * 谓词类型枚举
 */
enum class PredicateType {
    EQUALS,             // col = 'value'
    NOT_EQUALS,         // col <> 'value'
    LIKE_PREFIX,        // col LIKE 'prefix%'
    NOT_LIKE_PREFIX,    // col NOT LIKE 'prefix%'
    LIKE_SUFFIX,        // col LIKE '%suffix'
    NOT_LIKE_SUFFIX,    // col NOT LIKE '%suffix'
    LIKE_CONTAINS,      // col LIKE '%substr%'
    NOT_LIKE_CONTAINS,  // col NOT LIKE '%substr%'
    IN_SET_INT,         // col IN (1, 2, 3, ...)
    NOT_IN_SET_INT,     // col NOT IN (1, 2, 3, ...)
    IN_SET_STR,         // col IN ('a', 'b', 'c', ...)
    NOT_IN_SET_STR,     // col NOT IN ('a', 'b', 'c', ...)
    RANGE_INT,          // col BETWEEN lo AND hi
    LESS_THAN_INT,      // col < value
    GREATER_THAN_INT,   // col > value
};

/**
 * 谓词定义 (配置阶段使用)
 */
struct PredicateDef {
    PredicateType type;
    std::string str_value;              // 用于字符串比较
    std::vector<int32_t> int_set;       // 用于 IN_SET_INT
    std::vector<std::string> str_set;   // 用于 IN_SET_STR
    int32_t int_lo = 0;                 // 用于范围
    int32_t int_hi = 0;                 // 用于范围
};

/**
 * 列定义 (用于预计算)
 */
struct ColumnDef {
    enum Type { STRING, INT32 };
    Type type;
    const std::vector<std::string>* str_data = nullptr;
    const std::vector<int32_t>* int_data = nullptr;
};

/**
 * 通用谓词预计算器
 *
 * 核心思想: 把字符串/复杂条件从"执行期"移到"加载期"
 * - 配置阶段: 定义谓词 (类型 + 参数)
 * - 预计算阶段: 遍历表，生成 row_valid bitmap
 * - 执行阶段: O(1) 位图测试，零字符串操作
 *
 * 使用示例:
 *   PredicatePrecomputer pp;
 *   pp.add_column(0, part.p_brand);
 *   pp.add_column(1, part.p_type);
 *   pp.add_column(2, part.p_size);
 *   pp.add_predicate(0, PredicateType::NOT_EQUALS, "Brand#45");
 *   pp.add_predicate(1, PredicateType::NOT_LIKE_PREFIX, "MEDIUM POLISHED");
 *   pp.add_predicate(2, PredicateType::IN_SET_INT, {49, 14, 23, 45, 19, 3, 36, 9});
 *   pp.precompute(row_count);
 *   // 执行期: pp.is_valid(row_idx) → O(1)
 */
class PredicatePrecomputer {
public:
    PredicatePrecomputer() = default;

    // ========== 配置阶段 ==========

    /**
     * 添加字符串列
     */
    void add_column(int col_id, const std::vector<std::string>& str_data) {
        if (col_id >= static_cast<int>(columns_.size())) {
            columns_.resize(col_id + 1);
            predicates_.resize(col_id + 1);
            dictionaries_.resize(col_id + 1);
        }
        columns_[col_id].type = ColumnDef::STRING;
        columns_[col_id].str_data = &str_data;
    }

    /**
     * 添加整数列
     */
    void add_column(int col_id, const std::vector<int32_t>& int_data) {
        if (col_id >= static_cast<int>(columns_.size())) {
            columns_.resize(col_id + 1);
            predicates_.resize(col_id + 1);
            dictionaries_.resize(col_id + 1);
        }
        columns_[col_id].type = ColumnDef::INT32;
        columns_[col_id].int_data = &int_data;
    }

    /**
     * 添加字符串谓词: EQUALS, NOT_EQUALS, LIKE_PREFIX, NOT_LIKE_PREFIX, etc.
     */
    void add_predicate(int col_id, PredicateType type, const std::string& value) {
        PredicateDef pred;
        pred.type = type;
        pred.str_value = value;
        predicates_[col_id].push_back(pred);
    }

    /**
     * 添加整数集合谓词: IN_SET_INT, NOT_IN_SET_INT
     */
    void add_predicate(int col_id, PredicateType type, std::vector<int32_t> values) {
        PredicateDef pred;
        pred.type = type;
        pred.int_set = std::move(values);
        predicates_[col_id].push_back(pred);
    }

    /**
     * 添加字符串集合谓词: IN_SET_STR, NOT_IN_SET_STR
     */
    void add_predicate(int col_id, PredicateType type, std::vector<std::string> values) {
        PredicateDef pred;
        pred.type = type;
        pred.str_set = std::move(values);
        predicates_[col_id].push_back(pred);
    }

    /**
     * 添加范围谓词: RANGE_INT
     */
    void add_predicate(int col_id, PredicateType type, int32_t lo, int32_t hi) {
        PredicateDef pred;
        pred.type = type;
        pred.int_lo = lo;
        pred.int_hi = hi;
        predicates_[col_id].push_back(pred);
    }

    /**
     * 添加单值整数谓词: LESS_THAN_INT, GREATER_THAN_INT
     */
    void add_predicate_int(int col_id, PredicateType type, int32_t value) {
        PredicateDef pred;
        pred.type = type;
        pred.int_lo = value;
        predicates_[col_id].push_back(pred);
    }

    // ========== 预计算阶段 ==========

    /**
     * 预计算所有行的有效性
     * @param row_count 表的行数
     */
    void precompute(size_t row_count) {
        row_count_ = row_count;

        // Step 1: 为字符串列构建字典
        for (size_t col_id = 0; col_id < columns_.size(); ++col_id) {
            if (columns_[col_id].type == ColumnDef::STRING && columns_[col_id].str_data) {
                auto& dict = dictionaries_[col_id];
                dict.reserve(1000);
                for (const auto& s : *columns_[col_id].str_data) {
                    dict.encode(s);
                }
            }
        }

        // Step 2: 预计算每列的谓词位图 (dict_id → valid)
        col_valid_bitmaps_.resize(columns_.size());
        for (size_t col_id = 0; col_id < columns_.size(); ++col_id) {
            precompute_column(col_id);
        }

        // Step 3: 合并为行级位图
        size_t num_words = (row_count + 63) / 64;
        row_valid_bitmap_.resize(num_words, ~0ULL);  // 初始全有效

        for (size_t row = 0; row < row_count; ++row) {
            bool row_valid = true;
            for (size_t col_id = 0; col_id < columns_.size() && row_valid; ++col_id) {
                if (predicates_[col_id].empty()) continue;
                row_valid = evaluate_row_column(row, col_id);
            }
            if (!row_valid) {
                size_t word_idx = row / 64;
                uint64_t bit_mask = 1ULL << (row % 64);
                row_valid_bitmap_[word_idx] &= ~bit_mask;
            }
        }
    }

    // ========== 执行阶段 ==========

    /**
     * O(1) 检查行是否有效
     */
    bool is_valid(size_t row_idx) const {
        if (row_idx >= row_count_) return false;
        size_t word_idx = row_idx / 64;
        uint64_t bit_mask = 1ULL << (row_idx % 64);
        return (row_valid_bitmap_[word_idx] & bit_mask) != 0;
    }

    /**
     * 获取有效行数
     */
    size_t count_valid() const {
        size_t total = 0;
        for (uint64_t word : row_valid_bitmap_) {
            total += __builtin_popcountll(word);
        }
        // 修正最后一个 word 的多余位
        size_t extra_bits = (64 - (row_count_ % 64)) % 64;
        if (extra_bits > 0 && !row_valid_bitmap_.empty()) {
            uint64_t last_word = row_valid_bitmap_.back();
            uint64_t extra_mask = (~0ULL) << (64 - extra_bits);
            total -= __builtin_popcountll(last_word & extra_mask);
        }
        return total;
    }

    /**
     * 获取字典 (用于 GROUP BY 编码)
     */
    const StringDictionary& get_dictionary(int col_id) const {
        return dictionaries_[col_id];
    }

    /**
     * 获取字符串列的编码 ID
     */
    int32_t get_encoded_id(size_t row_idx, int col_id) const {
        if (columns_[col_id].type != ColumnDef::STRING) return -1;
        const auto& str = (*columns_[col_id].str_data)[row_idx];
        return dictionaries_[col_id].lookup(str);
    }

    /**
     * 获取整数列的值
     */
    int32_t get_int_value(size_t row_idx, int col_id) const {
        if (columns_[col_id].type != ColumnDef::INT32) return 0;
        return (*columns_[col_id].int_data)[row_idx];
    }

private:
    std::vector<ColumnDef> columns_;
    std::vector<std::vector<PredicateDef>> predicates_;
    std::vector<StringDictionary> dictionaries_;

    // 每列的 dict_id → valid 位图 (用于字符串列)
    std::vector<std::vector<bool>> col_valid_bitmaps_;

    // 整数列的快速查找表 (用于 IN_SET_INT)
    std::vector<std::vector<bool>> int_in_set_tables_;

    // 行级有效性位图
    std::vector<uint64_t> row_valid_bitmap_;
    size_t row_count_ = 0;

    /**
     * 预计算单列的谓词位图
     */
    void precompute_column(size_t col_id) {
        if (predicates_[col_id].empty()) return;

        if (columns_[col_id].type == ColumnDef::STRING) {
            // 字符串列: 构建 dict_id → valid 位图
            auto& dict = dictionaries_[col_id];
            auto& valid = col_valid_bitmaps_[col_id];
            valid.resize(dict.size(), true);

            for (const auto& pred : predicates_[col_id]) {
                for (size_t id = 0; id < dict.size(); ++id) {
                    if (!valid[id]) continue;  // 已经无效
                    const std::string& str = dict.decode(static_cast<int32_t>(id));
                    valid[id] = evaluate_string_predicate(str, pred);
                }
            }
        } else if (columns_[col_id].type == ColumnDef::INT32) {
            // 整数列: 构建快速查找表 (用于 IN_SET)
            for (const auto& pred : predicates_[col_id]) {
                if (pred.type == PredicateType::IN_SET_INT ||
                    pred.type == PredicateType::NOT_IN_SET_INT) {
                    // 找最大值
                    int32_t max_val = 0;
                    for (int32_t v : pred.int_set) {
                        max_val = std::max(max_val, v);
                    }
                    if (int_in_set_tables_.size() <= col_id) {
                        int_in_set_tables_.resize(col_id + 1);
                    }
                    int_in_set_tables_[col_id].resize(max_val + 1, false);
                    for (int32_t v : pred.int_set) {
                        if (v >= 0) int_in_set_tables_[col_id][v] = true;
                    }
                }
            }
        }
    }

    /**
     * 评估字符串谓词
     */
    bool evaluate_string_predicate(const std::string& str, const PredicateDef& pred) const {
        switch (pred.type) {
            case PredicateType::EQUALS:
                return str == pred.str_value;
            case PredicateType::NOT_EQUALS:
                return str != pred.str_value;
            case PredicateType::LIKE_PREFIX:
                return str.size() >= pred.str_value.size() &&
                       str.compare(0, pred.str_value.size(), pred.str_value) == 0;
            case PredicateType::NOT_LIKE_PREFIX:
                return str.size() < pred.str_value.size() ||
                       str.compare(0, pred.str_value.size(), pred.str_value) != 0;
            case PredicateType::LIKE_SUFFIX:
                return str.size() >= pred.str_value.size() &&
                       str.compare(str.size() - pred.str_value.size(),
                                   pred.str_value.size(), pred.str_value) == 0;
            case PredicateType::NOT_LIKE_SUFFIX:
                return str.size() < pred.str_value.size() ||
                       str.compare(str.size() - pred.str_value.size(),
                                   pred.str_value.size(), pred.str_value) != 0;
            case PredicateType::LIKE_CONTAINS:
                return str.find(pred.str_value) != std::string::npos;
            case PredicateType::NOT_LIKE_CONTAINS:
                return str.find(pred.str_value) == std::string::npos;
            case PredicateType::IN_SET_STR:
                for (const auto& s : pred.str_set) {
                    if (str == s) return true;
                }
                return false;
            case PredicateType::NOT_IN_SET_STR:
                for (const auto& s : pred.str_set) {
                    if (str == s) return false;
                }
                return true;
            default:
                return true;
        }
    }

    /**
     * 评估单行单列
     */
    bool evaluate_row_column(size_t row, size_t col_id) const {
        if (columns_[col_id].type == ColumnDef::STRING) {
            // 使用预计算的 dict_id → valid 位图
            const auto& str = (*columns_[col_id].str_data)[row];
            int32_t dict_id = dictionaries_[col_id].lookup(str);
            if (dict_id < 0) return false;
            return col_valid_bitmaps_[col_id][dict_id];
        } else if (columns_[col_id].type == ColumnDef::INT32) {
            int32_t val = (*columns_[col_id].int_data)[row];
            for (const auto& pred : predicates_[col_id]) {
                if (!evaluate_int_predicate(val, pred, col_id)) return false;
            }
            return true;
        }
        return true;
    }

    /**
     * 评估整数谓词
     */
    bool evaluate_int_predicate(int32_t val, const PredicateDef& pred, size_t col_id) const {
        switch (pred.type) {
            case PredicateType::IN_SET_INT:
                if (col_id < int_in_set_tables_.size() &&
                    val >= 0 && static_cast<size_t>(val) < int_in_set_tables_[col_id].size()) {
                    return int_in_set_tables_[col_id][val];
                }
                return false;
            case PredicateType::NOT_IN_SET_INT:
                if (col_id < int_in_set_tables_.size() &&
                    val >= 0 && static_cast<size_t>(val) < int_in_set_tables_[col_id].size()) {
                    return !int_in_set_tables_[col_id][val];
                }
                return true;
            case PredicateType::RANGE_INT:
                return val >= pred.int_lo && val <= pred.int_hi;
            case PredicateType::LESS_THAN_INT:
                return val < pred.int_lo;
            case PredicateType::GREATER_THAN_INT:
                return val > pred.int_lo;
            default:
                return true;
        }
    }
};

// ============================================================================
// 辅助结构: 预计算的行信息 (用于 GROUP BY)
// ============================================================================

/**
 * 预计算的行编码信息
 *
 * 保存满足条件的行的编码信息，用于后续 GROUP BY
 */
struct PrecomputedRowInfo {
    int32_t row_idx;        // 原始行索引
    int32_t primary_key;    // 主键值 (如 partkey)
    int64_t group_key;      // 编码的 GROUP BY 键

    bool operator<(const PrecomputedRowInfo& o) const {
        return group_key < o.group_key;
    }
};

// ============================================================================
// Q4 优化: Bitmap SEMI Join
// ============================================================================

/**
 * Q4: 订单优先级检查
 *
 * 原始 SQL:
 * SELECT o_orderpriority, COUNT(*) AS order_count
 * FROM orders
 * WHERE o_orderdate >= '1993-07-01' AND o_orderdate < '1993-10-01'
 *   AND EXISTS (SELECT * FROM lineitem
 *               WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
 * GROUP BY o_orderpriority
 *
 * 优化策略:
 * 1. 并行扫描 lineitem, 构建 late_orders bitmap
 * 2. 并行扫描 orders, 过滤日期 + bitmap 测试
 * 3. 线程局部计数 + 合并
 */
void run_q4_v27(TPCHDataLoader& loader);

// ============================================================================
// Q11 优化: 单遍扫描 + 后置过滤
// ============================================================================

/**
 * Q11: 重要库存识别
 *
 * 原始 SQL:
 * SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
 * FROM partsupp, supplier, nation
 * WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
 * GROUP BY ps_partkey
 * HAVING SUM(...) > (SELECT SUM(...) * 0.0001 FROM same tables)
 *
 * 优化策略:
 * 1. 预构建 germany_suppliers hash set
 * 2. 单遍扫描: 同时累加 total 和 per-part value
 * 3. 后置过滤: value > total * 0.0001
 */
void run_q11_v27(TPCHDataLoader& loader);

// ============================================================================
// Q16 优化: 并行 Anti-Join + BloomFilter
// ============================================================================

/**
 * Q16: 零件供应商关系
 *
 * 原始 SQL:
 * SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
 * FROM partsupp, part
 * WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45'
 *   AND p_type NOT LIKE 'MEDIUM POLISHED%'
 *   AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
 *   AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier
 *                          WHERE s_comment LIKE '%Customer%Complaints%')
 * GROUP BY p_brand, p_type, p_size
 *
 * 优化策略:
 * 1. 构建 complaint_suppliers BloomFilter + Bitmap
 * 2. 并行扫描 partsupp: Bloom 预过滤 + Bitmap Anti-Join
 * 3. 并行聚合 COUNT(DISTINCT)
 */
void run_q16_v27(TPCHDataLoader& loader);

// ============================================================================
// V27 查询入口 - 其他查询沿用 V26
// ============================================================================

// ============================================================================
// Q3 优化: Bitmap JOIN + 直接数组索引聚合
// ============================================================================

/**
 * Q3: 运输优先级
 *
 * 原始 SQL:
 * SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
 *        o_orderdate, o_shippriority
 * FROM customer, orders, lineitem
 * WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey
 *   AND l_orderkey = o_orderkey
 *   AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
 * GROUP BY l_orderkey, o_orderdate, o_shippriority
 * ORDER BY revenue DESC, o_orderdate
 * LIMIT 10
 *
 * 优化策略:
 * 1. 构建 BUILDING 客户 custkey bitmap - O(1) 测试
 * 2. 构建 valid_orderkey bitmap + order 信息数组 (date, priority)
 * 3. 并行扫描 lineitem: bitmap 过滤 + 直接数组索引聚合
 * 4. 避免 hash table，使用 orderkey 直接索引
 */
void run_q3_v27(TPCHDataLoader& loader);

// ============================================================================
// Q3 V28 优化: Bloom Filter + Compact Hash Table + Thread-Local 聚合
// ============================================================================

/**
 * 紧凑 Order 查找表 (Robin Hood Hash Table)
 *
 * V28 优化: 替代 V27 的直接数组索引 (73 MB → 4.5 MB)
 * - Robin Hood hashing 减少最坏情况探测长度
 * - 只存储有效的 ~300K 条目
 * - 单次探测返回完整信息 (idx, date, priority)
 */
class CompactOrderLookup {
public:
    static constexpr int32_t EMPTY_KEY = INT32_MIN;
    static constexpr size_t CACHE_LINE_SIZE = 128;

    struct Entry {
        int32_t orderkey = EMPTY_KEY;
        int32_t compact_idx = -1;
        int32_t orderdate = 0;
        int32_t shippriority = 0;
    };

    CompactOrderLookup() = default;

    /**
     * 初始化 hash table
     * @param expected_count 预期元素数量
     * @param load_factor 负载因子 (默认 0.6)
     */
    void init(size_t expected_count, double load_factor = 0.6) {
        size_t min_capacity = static_cast<size_t>(expected_count / load_factor) + 1;
        // 向上取整到 2 的幂次
        capacity_ = 1;
        while (capacity_ < min_capacity) capacity_ <<= 1;
        mask_ = capacity_ - 1;
        entries_.resize(capacity_);
        count_ = 0;
    }

    /**
     * 插入条目
     */
    void insert(int32_t orderkey, int32_t compact_idx, int32_t orderdate, int32_t shippriority) {
        Entry entry{orderkey, compact_idx, orderdate, shippriority};
        uint32_t pos = hash_key(orderkey) & mask_;
        uint32_t dist = 0;

        while (true) {
            if (entries_[pos].orderkey == EMPTY_KEY) {
                entries_[pos] = entry;
                count_++;
                return;
            }
            // Robin Hood: 计算当前位置条目的探测距离
            uint32_t existing_dist = probe_distance(entries_[pos].orderkey, pos);
            if (dist > existing_dist) {
                // 交换: 新条目占据此位置，原条目继续寻找
                std::swap(entry, entries_[pos]);
                dist = existing_dist;
            }
            pos = (pos + 1) & mask_;
            dist++;
        }
    }

    /**
     * 探测并返回 compact_idx (-1 表示不存在)
     */
    int32_t probe(int32_t orderkey) const {
        uint32_t pos = hash_key(orderkey) & mask_;
        uint32_t dist = 0;

        while (dist < capacity_) {
            const auto& e = entries_[pos];
            if (e.orderkey == EMPTY_KEY) return -1;
            if (e.orderkey == orderkey) return e.compact_idx;
            // Robin Hood: 如果当前探测距离超过该位置条目的探测距离，则不存在
            uint32_t existing_dist = probe_distance(e.orderkey, pos);
            if (dist > existing_dist) return -1;
            pos = (pos + 1) & mask_;
            dist++;
        }
        return -1;
    }

    /**
     * 探测并返回完整条目 (通过指针)
     * @return 条目指针 (nullptr 表示不存在)
     */
    const Entry* probe_full(int32_t orderkey) const {
        uint32_t pos = hash_key(orderkey) & mask_;
        uint32_t dist = 0;

        while (dist < capacity_) {
            const auto& e = entries_[pos];
            if (e.orderkey == EMPTY_KEY) return nullptr;
            if (e.orderkey == orderkey) return &e;
            uint32_t existing_dist = probe_distance(e.orderkey, pos);
            if (dist > existing_dist) return nullptr;
            pos = (pos + 1) & mask_;
            dist++;
        }
        return nullptr;
    }

    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }

private:
    std::vector<Entry> entries_;
    size_t capacity_ = 0;
    uint32_t mask_ = 0;
    size_t count_ = 0;

    // CRC32 哈希 (ARM 硬件加速)
    static uint32_t hash_key(int32_t key) {
#ifdef __aarch64__
        return __builtin_arm_crc32w(0xFFFFFFFF, static_cast<uint32_t>(key));
#else
        // 备用: MurmurHash 风格
        uint32_t k = static_cast<uint32_t>(key);
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k;
#endif
    }

    uint32_t probe_distance(int32_t key, uint32_t pos) const {
        uint32_t home = hash_key(key) & mask_;
        return (pos - home + capacity_) & mask_;
    }
};

/**
 * 线程局部 Q3 聚合器
 *
 * V28 优化: 消除原子操作开销
 * - 每个线程独立的 hash table
 * - 使用 MutableWeakHashTable 存储 compact_idx → revenue
 * - 最后归并所有线程结果
 */
struct alignas(128) ThreadLocalQ3Agg {
    MutableWeakHashTable<int64_t> revenue_map;

    void init(size_t expected_count) {
        revenue_map.init(expected_count);
    }

    void add(int32_t compact_idx, int64_t revenue) {
        revenue_map.add_or_update(compact_idx, revenue);
    }

    void clear() {
        revenue_map = MutableWeakHashTable<int64_t>();
    }
};

/**
 * Q3 V28: Bloom Filter + Compact Hash Table + Thread-Local 聚合
 *
 * 优化策略:
 * 1. Bloom Filter 预过滤 - 94% 的无效 orderkey 在 L2 缓存中被过滤
 * 2. CompactOrderLookup - 4.5 MB vs 73 MB，更好的缓存利用
 * 3. ThreadLocalQ3Agg - 消除原子操作，最后 SIMD 归并
 *
 * 注意: 基准测试显示 V28 比 V27 慢，因为 Bloom Filter 和 hash table 开销
 *       超过了节省的缓存 miss。保留此版本用于大数据集 (SF>10)。
 */
void run_q3_v28(TPCHDataLoader& loader);

/**
 * Q3 V28.1: 直接数组索引 + Thread-Local 数组聚合
 *
 * 优化策略:
 * 1. 保留 V27 的直接数组索引 (O(1) 访问)
 * 2. Thread-Local 数组聚合 (消除原子操作)
 * 3. SIMD 加速归并
 *
 * 预期性能: V27 ~12 ms → V28.1 ~10 ms (1.2x 加速)
 */
void run_q3_v28_1(TPCHDataLoader& loader);

// ============================================================================
// Q3 V29 优化: SIMD 批量过滤 + 轻量级 Bloom + Predicate Bitmap
// ============================================================================

/**
 * 轻量级 Bloom Filter (2-hash, 内联)
 *
 * 优化点:
 * - 只用 2 次 CRC32 哈希 (vs 默认 7 次)
 * - 完全内联，避免函数调用
 * - 更小的位数组 (4 bits/element)
 * - 假阳性率 ~5%，但开销极低
 */
class LightweightBloomFilter {
public:
    LightweightBloomFilter() = default;

    void init(size_t expected_elements) {
        // 4 bits per element, ~5% false positive
        size_t num_bits = expected_elements * 4;
        num_bits = ((num_bits + 63) / 64) * 64;  // 对齐到 64 位
        bits_.resize(num_bits / 64, 0);
        mask_ = static_cast<uint32_t>(num_bits - 1);
    }

    // 内联插入
    inline void insert(int32_t key) {
#ifdef __aarch64__
        uint32_t h1 = __builtin_arm_crc32w(0, static_cast<uint32_t>(key));
        uint32_t h2 = __builtin_arm_crc32w(0x9E3779B9, static_cast<uint32_t>(key));
#else
        uint32_t h1 = static_cast<uint32_t>(key) * 0x85ebca6b;
        uint32_t h2 = static_cast<uint32_t>(key) * 0xc2b2ae35;
#endif
        bits_[(h1 & mask_) >> 6] |= (1ULL << ((h1 & mask_) & 63));
        bits_[(h2 & mask_) >> 6] |= (1ULL << ((h2 & mask_) & 63));
    }

    // 内联测试
    inline bool may_contain(int32_t key) const {
#ifdef __aarch64__
        uint32_t h1 = __builtin_arm_crc32w(0, static_cast<uint32_t>(key));
        uint32_t h2 = __builtin_arm_crc32w(0x9E3779B9, static_cast<uint32_t>(key));
#else
        uint32_t h1 = static_cast<uint32_t>(key) * 0x85ebca6b;
        uint32_t h2 = static_cast<uint32_t>(key) * 0xc2b2ae35;
#endif
        return ((bits_[(h1 & mask_) >> 6] >> ((h1 & mask_) & 63)) & 1) &&
               ((bits_[(h2 & mask_) >> 6] >> ((h2 & mask_) & 63)) & 1);
    }

private:
    std::vector<uint64_t> bits_;
    uint32_t mask_ = 0;
};

/**
 * Q3 V29: SIMD 批量过滤 + 轻量级 Bloom + 批量聚合
 *
 * 优化策略:
 * 1. Predicate Bitmap: 预计算 l_shipdate > threshold
 * 2. 轻量级 Bloom Filter: 2-hash, 内联, ~5% 假阳性
 * 3. SIMD 批量过滤: 每次处理 4 个 int32
 * 4. 批量聚合 + 预取: 减少随机访问延迟
 *
 * 目标: > 0.95x DuckDB
 */
void run_q3_v29(TPCHDataLoader& loader);

/**
 * Q3 V30: 极简预处理 + 直接扫描
 *
 * 优化策略:
 * 1. 单遍预处理: 合并 max_key 查找和 lookup 构建
 * 2. 紧凑 custkey bitmap: 使用预计算的 max_custkey
 * 3. 无 Bloom Filter: 直接数组索引足够快
 * 4. SIMD 加速日期比较
 *
 * 目标: 稳定 > 0.95x DuckDB
 */
void run_q3_v30(TPCHDataLoader& loader);

/**
 * Q3 V31: Bloom Filter + 紧凑 Hash Table
 *
 * 核心思想: 用 Bloom Filter 快速拒绝，用紧凑 hash table 代替大数组
 * - order_lookup: 73 MB → ~6 MB (只存储 300K 有效订单)
 * - Bloom Filter: ~40 KB (足够 L1 缓存)
 * - 减少内存初始化和随机访问
 */
void run_q3_v31(TPCHDataLoader& loader);

/**
 * Q3 V32: 整合所有最优技术
 *
 * 技术组合:
 * 1. 紧凑 Hash Table (V31) - 6 MB vs 73 MB
 * 2. 单 Hash Bloom Filter (V31) - L1 缓存友好
 * 3. SIMD 日期过滤 (V30) - 批量比较
 * 4. Thread-Local 数组聚合 - 消除原子操作
 * 5. SIMD 归并 - 快速合并线程结果
 */
void run_q3_v32(TPCHDataLoader& loader);

// Category A - 沿用 V26
inline void run_q1_v27(TPCHDataLoader& loader)  { queries::run_q1(loader); }
inline void run_q5_v27(TPCHDataLoader& loader)  { run_q5_v26(loader); }
inline void run_q6_v27(TPCHDataLoader& loader)  { run_q6_v26(loader); }
/**
 * Q7: 体量运输 (国家间贸易)
 *
 * 原始 SQL:
 * SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
 * FROM (supplier, lineitem, orders, customer, nation n1, nation n2)
 * WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
 *   AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
 *   AND c_nationkey = n2.n_nationkey
 *   AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
 *        OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
 *   AND l_shipdate BETWEEN '1995-01-01' AND '1996-12-31'
 * GROUP BY supp_nation, cust_nation, l_year
 *
 * 优化策略:
 * 1. 预构建 suppkey → nationkey 数组 (直接索引)
 * 2. 预构建 orderkey → {custkey, cust_nationkey} 数组
 * 3. 单遍并行扫描 lineitem - 过滤 + 直接聚合
 * 4. 避免 SEMI JOIN 和 INNER JOIN
 */
void run_q7_v27(TPCHDataLoader& loader);
inline void run_q9_v27(TPCHDataLoader& loader)  { run_q9_v26(loader); }
inline void run_q10_v27(TPCHDataLoader& loader) { run_q10_v26(loader); }
/**
 * Q12: 运输模式与订单优先级
 *
 * 原始 SQL:
 * SELECT l_shipmode,
 *        SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
 *        SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
 * FROM orders, lineitem
 * WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP')
 *   AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
 *   AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
 * GROUP BY l_shipmode
 *
 * 优化策略:
 * 1. 预构建 order 信息数组 (orderkey → orderpriority)
 * 2. 单遍并行扫描 lineitem - 过滤 + 直接聚合
 * 3. 避免 INNER JOIN 开销
 */
void run_q12_v27(TPCHDataLoader& loader);
inline void run_q14_v27(TPCHDataLoader& loader) { run_q14_v26(loader); }
/**
 * Q18: 大订单客户
 *
 * 原始 SQL:
 * SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity)
 * FROM customer, orders, lineitem
 * WHERE c_custkey = o_custkey AND o_orderkey = l_orderkey
 *   AND o_orderkey IN (SELECT l_orderkey FROM lineitem
 *                      GROUP BY l_orderkey HAVING SUM(l_quantity) > 300)
 * GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
 * ORDER BY o_totalprice DESC, o_orderdate
 * LIMIT 100
 *
 * 优化策略:
 * 1. 使用 orderkey 直接数组索引代替 hash table
 * 2. 并行聚合 l_quantity 到数组
 * 3. 使用 bitmap 标记 large orders
 */
void run_q18_v27(TPCHDataLoader& loader);

// Category B - Q4, Q11, Q16 使用 V27 优化，其他沿用基线
inline void run_q2_v27(TPCHDataLoader& loader)  { queries::run_q2(loader); }
// run_q4_v27 - 已声明
// run_q11_v27 - 已声明
/**
 * Q15: 最高收入供应商
 *
 * 原始 SQL (CTE):
 * CREATE VIEW revenue0 AS
 *   SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
 *   FROM lineitem WHERE l_shipdate >= '1996-01-01' AND l_shipdate < '1996-04-01'
 *   GROUP BY l_suppkey;
 *
 * SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
 * FROM supplier, revenue0
 * WHERE s_suppkey = supplier_no AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
 *
 * 优化策略:
 * 1. 使用 suppkey 直接数组索引 (suppkey 范围 ~10K)
 * 2. 并行扫描 lineitem + 原子聚合
 * 3. 并行找最大值
 */
void run_q15_v27(TPCHDataLoader& loader);
// run_q16_v27 - 已声明
inline void run_q19_v27(TPCHDataLoader& loader) { queries::run_q19(loader); }

} // namespace ops_v27
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V27_H
