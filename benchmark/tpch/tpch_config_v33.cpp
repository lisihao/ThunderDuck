/**
 * ThunderDuck TPC-H V33 配置层实现
 *
 * @version 33.0
 * @date 2026-01-28
 */

#include "tpch_config_v33.h"
#include "tpch_constants.h"
#include <cstring>
#include <cstdlib>
#include <thread>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v33 {

// ============================================================================
// DateRange 实现
// ============================================================================

int32_t DateRange::parse_date(const char* date_str) {
    // 格式: "YYYY-MM-DD"
    int year = 0, month = 0, day = 0;

    // 简单解析 (假设格式正确)
    year = (date_str[0] - '0') * 1000 + (date_str[1] - '0') * 100 +
           (date_str[2] - '0') * 10 + (date_str[3] - '0');
    month = (date_str[5] - '0') * 10 + (date_str[6] - '0');
    day = (date_str[8] - '0') * 10 + (date_str[9] - '0');

    // 计算 epoch days (从 1970-01-01)
    // 使用简化公式 (足够准确用于 TPC-H 日期范围 1992-1998)
    int32_t days = 0;

    // 年份贡献
    for (int y = 1970; y < year; ++y) {
        bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
        days += leap ? 366 : 365;
    }

    // 月份贡献
    static const int days_in_month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    bool is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);

    for (int m = 1; m < month; ++m) {
        days += days_in_month[m];
        if (m == 2 && is_leap) days += 1;
    }

    // 天数贡献
    days += day - 1;

    return days;
}

DateRange DateRange::from_string(const char* lo_str, const char* hi_str,
                                  bool lo_incl, bool hi_incl) {
    DateRange range;
    range.lo = parse_date(lo_str);
    range.hi = parse_date(hi_str);
    range.lo_inclusive = lo_incl;
    range.hi_inclusive = hi_incl;
    return range;
}

// ============================================================================
// StringPredicate 实现
// ============================================================================

StringPredicate StringPredicate::equals(const std::string& val) {
    StringPredicate pred;
    pred.type = StringPredicateType::EQUALS;
    pred.value = val;
    return pred;
}

StringPredicate StringPredicate::in_set(std::vector<std::string> values) {
    StringPredicate pred;
    pred.type = StringPredicateType::IN_SET;
    pred.value_set = std::move(values);
    // 构建哈希集合用于快速查找
    for (const auto& v : pred.value_set) {
        pred.value_set_hash.insert(v);
    }
    return pred;
}

StringPredicate StringPredicate::like_contains(const std::string& pattern) {
    StringPredicate pred;
    pred.type = StringPredicateType::LIKE_CONTAINS;
    pred.value = pattern;
    return pred;
}

StringPredicate StringPredicate::like_prefix(const std::string& pattern) {
    StringPredicate pred;
    pred.type = StringPredicateType::LIKE_PREFIX;
    pred.value = pattern;
    return pred;
}

bool StringPredicate::evaluate(const std::string& input) const {
    switch (type) {
        case StringPredicateType::EQUALS:
            return input == value;

        case StringPredicateType::IN_SET:
            return value_set_hash.find(input) != value_set_hash.end();

        case StringPredicateType::LIKE_CONTAINS:
            return input.find(value) != std::string::npos;

        case StringPredicateType::LIKE_PREFIX:
            return input.size() >= value.size() &&
                   input.compare(0, value.size(), value) == 0;

        case StringPredicateType::LIKE_SUFFIX:
            return input.size() >= value.size() &&
                   input.compare(input.size() - value.size(), value.size(), value) == 0;

        default:
            return false;
    }
}

// ============================================================================
// ExecutionConfig 实现
// ============================================================================

void ExecutionConfig::auto_configure(size_t data_size) {
    // 自动配置线程数
    if (thread_count == 0) {
        size_t hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) hw_threads = 8;

        // 根据数据量决定线程数
        if (data_size < 10000) {
            thread_count = 1;  // 小数据单线程
        } else if (data_size < 100000) {
            thread_count = std::min<size_t>(4, hw_threads);
        } else {
            thread_count = std::min<size_t>(8, hw_threads);
        }
    }

    // 自动配置批量大小
    if (data_size > 1000000) {
        batch_size = 16;
        prefetch_distance = 128;
    }
}

size_t ExecutionConfig::get_thread_count() const {
    if (thread_count > 0) return thread_count;
    size_t hw = std::thread::hardware_concurrency();
    return hw > 0 ? std::min<size_t>(8, hw) : 8;
}

// ============================================================================
// TPCHConfigFactory 实现
// ============================================================================

namespace TPCHConfigFactory {

QueryConfig q5_default() {
    return ConfigBuilder()
        .string_val("region", regions::ASIA)
        .date_range("order_date", "1994-01-01", "1995-01-01", true, false)
        .thread_count(0)  // 自动
        .build();
}

QueryConfig q7_default() {
    return ConfigBuilder()
        .string_set("nations", {nations::FRANCE, nations::GERMANY})
        .date_range("ship_date", "1995-01-01", "1996-12-31", true, true)
        .thread_count(0)
        .build();
}

QueryConfig q9_default() {
    return ConfigBuilder()
        .string_contains("product_pattern", "green")
        .thread_count(0)
        .build();
}

QueryConfig q18_default() {
    // qty_threshold: 300 (定点数乘以 10000)
    return ConfigBuilder()
        .int64_val("qty_threshold", 300LL * 10000)
        .result_limit(100)
        .thread_count(0)
        .build();
}

QueryConfig q19_default() {
    QueryConfig cfg;

    // 条件组 1: Brand#12 + SM* + size 1-5 + qty 1-11
    cfg.set_string("brand_1", "Brand#12");
    cfg.set_string_set("container_1", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"});
    cfg.set_int32_range("size_1", NumericRange<int32_t>(1, 5));
    cfg.set_int64_range("qty_1", NumericRange<int64_t>(10000, 110000));  // 1-11 * 10000

    // 条件组 2: Brand#23 + MED* + size 1-10 + qty 10-20
    cfg.set_string("brand_2", "Brand#23");
    cfg.set_string_set("container_2", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"});
    cfg.set_int32_range("size_2", NumericRange<int32_t>(1, 10));
    cfg.set_int64_range("qty_2", NumericRange<int64_t>(100000, 200000));  // 10-20 * 10000

    // 条件组 3: Brand#34 + LG* + size 1-15 + qty 20-30
    cfg.set_string("brand_3", "Brand#34");
    cfg.set_string_set("container_3", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"});
    cfg.set_int32_range("size_3", NumericRange<int32_t>(1, 15));
    cfg.set_int64_range("qty_3", NumericRange<int64_t>(200000, 300000));  // 20-30 * 10000

    cfg.execution().thread_count = 0;  // 自动

    return cfg;
}

} // namespace TPCHConfigFactory

// ============================================================================
// DateConstants 实现
// ============================================================================

namespace DateConstants {

int32_t string_to_epoch_days(const char* date_str) {
    return DateRange::parse_date(date_str);
}

} // namespace DateConstants

} // namespace ops_v33
} // namespace tpch
} // namespace thunderduck
