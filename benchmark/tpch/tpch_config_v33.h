/**
 * ThunderDuck TPC-H V33 配置层
 *
 * 消除所有硬编码，通过参数化配置传递查询参数:
 * - DateRange: 日期范围配置
 * - NumericRange: 数值范围配置
 * - StringPredicate: 字符串谓词配置
 * - QueryConfig: 查询配置容器
 * - TPCHConfigFactory: TPC-H 默认配置工厂
 *
 * @version 33.0
 * @date 2026-01-28
 */

#ifndef TPCH_CONFIG_V33_H
#define TPCH_CONFIG_V33_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

namespace thunderduck {
namespace tpch {
namespace ops_v33 {

// ============================================================================
// 日期范围配置
// ============================================================================

/**
 * 日期范围配置
 *
 * 日期使用 epoch days (从 1970-01-01 开始的天数)
 */
struct DateRange {
    int32_t lo = 0;
    int32_t hi = 0;
    bool lo_inclusive = true;
    bool hi_inclusive = false;

    /**
     * 从字符串创建日期范围
     * @param lo_str 开始日期 "YYYY-MM-DD"
     * @param hi_str 结束日期 "YYYY-MM-DD"
     * @param lo_incl 开始日期是否包含
     * @param hi_incl 结束日期是否包含
     */
    static DateRange from_string(const char* lo_str, const char* hi_str,
                                  bool lo_incl = true, bool hi_incl = false);

    /**
     * 解析日期字符串为 epoch days
     * @param date_str 格式 "YYYY-MM-DD"
     * @return epoch days
     */
    static int32_t parse_date(const char* date_str);

    /**
     * 检查日期是否在范围内
     */
    __attribute__((always_inline))
    bool contains(int32_t date) const {
        bool lo_check = lo_inclusive ? (date >= lo) : (date > lo);
        bool hi_check = hi_inclusive ? (date <= hi) : (date < hi);
        return lo_check && hi_check;
    }
};

// ============================================================================
// 数值范围配置
// ============================================================================

/**
 * 通用数值范围配置
 */
template<typename T>
struct NumericRange {
    T lo = T{};
    T hi = T{};
    bool lo_inclusive = true;
    bool hi_inclusive = true;

    NumericRange() = default;
    NumericRange(T lo_val, T hi_val, bool lo_incl = true, bool hi_incl = true)
        : lo(lo_val), hi(hi_val), lo_inclusive(lo_incl), hi_inclusive(hi_incl) {}

    __attribute__((always_inline))
    bool contains(T val) const {
        bool lo_check = lo_inclusive ? (val >= lo) : (val > lo);
        bool hi_check = hi_inclusive ? (val <= hi) : (val < hi);
        return lo_check && hi_check;
    }
};

// ============================================================================
// 字符串谓词配置
// ============================================================================

/**
 * 字符串谓词类型
 */
enum class StringPredicateType {
    EQUALS,         // 精确匹配
    IN_SET,         // 在集合中
    LIKE_CONTAINS,  // 包含模式
    LIKE_PREFIX,    // 前缀匹配
    LIKE_SUFFIX     // 后缀匹配
};

/**
 * 字符串谓词配置
 */
struct StringPredicate {
    StringPredicateType type = StringPredicateType::EQUALS;
    std::string value;
    std::vector<std::string> value_set;
    std::unordered_set<std::string> value_set_hash;  // 用于快速查找

    StringPredicate() = default;

    /**
     * 创建精确匹配谓词
     */
    static StringPredicate equals(const std::string& val);

    /**
     * 创建集合包含谓词
     */
    static StringPredicate in_set(std::vector<std::string> values);

    /**
     * 创建 LIKE %pattern% 谓词
     */
    static StringPredicate like_contains(const std::string& pattern);

    /**
     * 创建 LIKE pattern% 谓词
     */
    static StringPredicate like_prefix(const std::string& pattern);

    /**
     * 评估谓词
     */
    bool evaluate(const std::string& input) const;
};

// ============================================================================
// 执行参数配置
// ============================================================================

/**
 * 执行参数配置
 */
struct ExecutionConfig {
    size_t thread_count = 0;       // 0 = 自动检测
    size_t batch_size = 8;         // 批量处理大小
    size_t prefetch_distance = 64; // 预取距离
    size_t result_limit = 0;       // 结果数量限制 (0 = 无限制)
    bool use_simd = true;          // 是否使用 SIMD
    bool use_parallel = true;      // 是否使用并行

    /**
     * 根据数据量自动配置
     */
    void auto_configure(size_t data_size);

    /**
     * 获取实际线程数
     */
    size_t get_thread_count() const;
};

// ============================================================================
// 查询配置容器
// ============================================================================

/**
 * 查询配置容器
 *
 * 统一管理查询的所有参数，消除硬编码
 */
class QueryConfig {
public:
    QueryConfig() = default;

    // ========== 日期参数 ==========

    void set_date_range(const std::string& name, DateRange range) {
        dates_[name] = range;
    }

    DateRange get_date_range(const std::string& name) const {
        auto it = dates_.find(name);
        if (it == dates_.end()) {
            throw std::runtime_error("Date range not found: " + name);
        }
        return it->second;
    }

    bool has_date_range(const std::string& name) const {
        return dates_.find(name) != dates_.end();
    }

    // ========== 数值参数 ==========

    void set_int64(const std::string& name, int64_t value) {
        int64_values_[name] = value;
    }

    int64_t get_int64(const std::string& name) const {
        auto it = int64_values_.find(name);
        if (it == int64_values_.end()) {
            throw std::runtime_error("Int64 value not found: " + name);
        }
        return it->second;
    }

    int64_t get_int64(const std::string& name, int64_t default_val) const {
        auto it = int64_values_.find(name);
        return (it != int64_values_.end()) ? it->second : default_val;
    }

    void set_int64_range(const std::string& name, NumericRange<int64_t> range) {
        int64_ranges_[name] = range;
    }

    NumericRange<int64_t> get_int64_range(const std::string& name) const {
        auto it = int64_ranges_.find(name);
        if (it == int64_ranges_.end()) {
            throw std::runtime_error("Int64 range not found: " + name);
        }
        return it->second;
    }

    void set_int32_range(const std::string& name, NumericRange<int32_t> range) {
        int32_ranges_[name] = range;
    }

    NumericRange<int32_t> get_int32_range(const std::string& name) const {
        auto it = int32_ranges_.find(name);
        if (it == int32_ranges_.end()) {
            throw std::runtime_error("Int32 range not found: " + name);
        }
        return it->second;
    }

    // ========== 字符串参数 ==========

    void set_string(const std::string& name, const std::string& value) {
        strings_[name] = value;
    }

    std::string get_string(const std::string& name) const {
        auto it = strings_.find(name);
        if (it == strings_.end()) {
            throw std::runtime_error("String not found: " + name);
        }
        return it->second;
    }

    std::string get_string(const std::string& name, const std::string& default_val) const {
        auto it = strings_.find(name);
        return (it != strings_.end()) ? it->second : default_val;
    }

    void set_string_set(const std::string& name, std::vector<std::string> values) {
        string_sets_[name] = std::move(values);
    }

    const std::vector<std::string>& get_string_set(const std::string& name) const {
        auto it = string_sets_.find(name);
        if (it == string_sets_.end()) {
            throw std::runtime_error("String set not found: " + name);
        }
        return it->second;
    }

    void set_string_predicate(const std::string& name, StringPredicate pred) {
        predicates_[name] = std::move(pred);
    }

    const StringPredicate& get_string_predicate(const std::string& name) const {
        auto it = predicates_.find(name);
        if (it == predicates_.end()) {
            throw std::runtime_error("String predicate not found: " + name);
        }
        return it->second;
    }

    // ========== 执行参数 ==========

    ExecutionConfig& execution() { return exec_; }
    const ExecutionConfig& execution() const { return exec_; }

private:
    std::unordered_map<std::string, DateRange> dates_;
    std::unordered_map<std::string, int64_t> int64_values_;
    std::unordered_map<std::string, NumericRange<int64_t>> int64_ranges_;
    std::unordered_map<std::string, NumericRange<int32_t>> int32_ranges_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, std::vector<std::string>> string_sets_;
    std::unordered_map<std::string, StringPredicate> predicates_;
    ExecutionConfig exec_;
};

// ============================================================================
// TPC-H 配置工厂
// ============================================================================

/**
 * TPC-H 默认配置工厂
 *
 * 为每个 TPC-H 查询提供默认参数配置
 */
namespace TPCHConfigFactory {

/**
 * Q5 默认配置
 * - region: "ASIA"
 * - order_date: 1994-01-01 to 1995-01-01
 */
QueryConfig q5_default();

/**
 * Q7 默认配置
 * - nations: ["FRANCE", "GERMANY"]
 * - ship_date: 1995-01-01 to 1996-12-31
 */
QueryConfig q7_default();

/**
 * Q9 默认配置
 * - product_pattern: "green" (LIKE %green%)
 */
QueryConfig q9_default();

/**
 * Q18 默认配置
 * - qty_threshold: 300 (定点数 300 * 10000)
 * - result_limit: 100
 */
QueryConfig q18_default();

/**
 * Q19 默认配置
 * - 三组条件参数 (brands, containers, sizes, quantities)
 */
QueryConfig q19_default();

/**
 * 通用查询配置创建器
 */
class ConfigBuilder {
public:
    ConfigBuilder& date_range(const std::string& name,
                               const char* lo, const char* hi,
                               bool lo_incl = true, bool hi_incl = false) {
        cfg_.set_date_range(name, DateRange::from_string(lo, hi, lo_incl, hi_incl));
        return *this;
    }

    ConfigBuilder& string_val(const std::string& name, const std::string& val) {
        cfg_.set_string(name, val);
        return *this;
    }

    ConfigBuilder& string_set(const std::string& name, std::vector<std::string> vals) {
        cfg_.set_string_set(name, std::move(vals));
        return *this;
    }

    ConfigBuilder& string_contains(const std::string& name, const std::string& pattern) {
        cfg_.set_string_predicate(name, StringPredicate::like_contains(pattern));
        return *this;
    }

    ConfigBuilder& int64_val(const std::string& name, int64_t val) {
        cfg_.set_int64(name, val);
        return *this;
    }

    ConfigBuilder& int64_range(const std::string& name, int64_t lo, int64_t hi,
                                bool lo_incl = true, bool hi_incl = true) {
        cfg_.set_int64_range(name, NumericRange<int64_t>(lo, hi, lo_incl, hi_incl));
        return *this;
    }

    ConfigBuilder& int32_range(const std::string& name, int32_t lo, int32_t hi,
                                bool lo_incl = true, bool hi_incl = true) {
        cfg_.set_int32_range(name, NumericRange<int32_t>(lo, hi, lo_incl, hi_incl));
        return *this;
    }

    ConfigBuilder& thread_count(size_t count) {
        cfg_.execution().thread_count = count;
        return *this;
    }

    ConfigBuilder& result_limit(size_t limit) {
        cfg_.execution().result_limit = limit;
        return *this;
    }

    QueryConfig build() { return std::move(cfg_); }

private:
    QueryConfig cfg_;
};

} // namespace TPCHConfigFactory

// ============================================================================
// 常用日期常量 (用于配置工厂)
// ============================================================================

namespace DateConstants {
    // 常用 TPC-H 日期 (epoch days)
    constexpr int32_t DATE_1993_01_01 = 8401;
    constexpr int32_t DATE_1993_07_01 = 8582;
    constexpr int32_t DATE_1993_10_01 = 8674;
    constexpr int32_t DATE_1994_01_01 = 8766;
    constexpr int32_t DATE_1995_01_01 = 9131;
    constexpr int32_t DATE_1995_03_15 = 9204;
    constexpr int32_t DATE_1995_09_01 = 9374;
    constexpr int32_t DATE_1995_10_01 = 9404;
    constexpr int32_t DATE_1996_01_01 = 9497;
    constexpr int32_t DATE_1996_04_01 = 9588;
    constexpr int32_t DATE_1996_12_31 = 9862;
    constexpr int32_t DATE_1998_09_02 = 10471;
    constexpr int32_t DATE_1998_12_01 = 10561;

    // 日期字符串到 epoch days 的转换函数
    int32_t string_to_epoch_days(const char* date_str);
}

} // namespace ops_v33
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_CONFIG_V33_H
