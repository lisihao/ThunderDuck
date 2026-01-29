/**
 * ThunderDuck TPC-H V34 通用算子扩展
 *
 * V34 目标: 扩展 V33 通用架构，新增以下通用能力
 *
 * Layer 2 扩展:
 * - GenericAntiJoin: 通用 ANTI JOIN (支持 NOT EXISTS/NOT IN)
 * - GenericOuterJoin: 通用 OUTER JOIN (LEFT/RIGHT/FULL)
 * - ConditionalAggregator: 通用条件聚合 (CASE WHEN)
 * - StringFunctions: 通用字符串函数 (SUBSTRING/LIKE/etc.)
 *
 * 设计原则:
 * 1. 配置驱动 - 所有参数通过 QueryConfig 传递
 * 2. 零硬编码 - 无查询特定常量
 * 3. 可组合 - 算子可自由组合构建任意查询
 *
 * @version 34.0
 * @date 2026-01-29
 * @tag 继续攻坚
 */

#ifndef TPCH_OPERATORS_V34_H
#define TPCH_OPERATORS_V34_H

#include "tpch_config_v33.h"
#include "tpch_data_loader.h"
#include "tpch_operators_v33.h"
#include "tpch_operators_v32.h"

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <string_view>
#include <functional>
#include <variant>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v34 {

using ::thunderduck::tpch::TPCHDataLoader;
using ops_v33::QueryConfig;
using ops_v33::ExecutionConfig;
using ops_v33::DateRange;
using ops_v33::NumericRange;
using ops_v33::StringPredicate;
using ops_v33::StringPredicateType;
using ops_v33::AdaptiveHashJoin;
using ops_v33::GenericThreadLocalAggregator;
using ops_v33::AutoTuner;
using ops_v32::CompactHashTable;
using ops_v32::SingleHashBloomFilter;
using ops_v25::ThreadPool;

// ============================================================================
// 通用字符串函数库
// ============================================================================

/**
 * 字符串函数类型
 */
enum class StringFunctionType {
    SUBSTRING,      // SUBSTRING(str, start, len)
    LEFT,           // LEFT(str, len)
    RIGHT,          // RIGHT(str, len)
    UPPER,          // UPPER(str)
    LOWER,          // LOWER(str)
    TRIM,           // TRIM(str)
    LENGTH          // LENGTH(str)
};

/**
 * 通用字符串函数
 *
 * 支持配置化的字符串操作，可用于任意查询
 */
class StringFunctions {
public:
    /**
     * SUBSTRING 提取
     * @param str 输入字符串
     * @param start 起始位置 (1-based, SQL 标准)
     * @param length 提取长度
     * @return 子串
     */
    static std::string_view substring(const std::string& str, int start, int length);

    /**
     * 批量 SUBSTRING 提取
     * @param strings 输入字符串数组
     * @param start 起始位置
     * @param length 提取长度
     * @return 子串数组
     */
    static std::vector<std::string> substring_batch(
        const std::vector<std::string>& strings, int start, int length);

    /**
     * 将 SUBSTRING 结果转换为整数 (用于国家码等场景)
     * @param str 输入字符串
     * @param start 起始位置
     * @param length 提取长度
     * @return 整数值，无效返回 -1
     */
    static int32_t substring_to_int(const std::string& str, int start, int length);

    /**
     * 批量 SUBSTRING 转整数
     */
    static std::vector<int32_t> substring_to_int_batch(
        const std::vector<std::string>& strings, int start, int length);

    /**
     * LIKE 模式匹配
     * @param str 输入字符串
     * @param pattern 模式 (支持 % 和 _)
     * @return 是否匹配
     */
    static bool like(const std::string& str, const std::string& pattern);

    /**
     * NOT LIKE 模式匹配
     */
    static bool not_like(const std::string& str, const std::string& pattern);

    /**
     * 包含检查 (用于 %pattern% 优化)
     * @param str 输入字符串
     * @param patterns 需要包含的子串列表 (AND 关系)
     */
    static bool contains_all(const std::string& str,
                             const std::vector<std::string>& patterns);

    /**
     * 批量 LIKE 预计算
     * @param strings 输入字符串数组
     * @param pattern 模式
     * @return 匹配位图
     */
    static std::vector<uint64_t> like_batch_bitmap(
        const std::vector<std::string>& strings, const std::string& pattern);
};

// ============================================================================
// 通用集合匹配器
// ============================================================================

/**
 * 通用集合匹配器
 *
 * 支持 IN / NOT IN 操作，用于任意类型
 */
template<typename T>
class SetMatcher {
public:
    SetMatcher() = default;

    /**
     * 从配置加载集合
     */
    void configure(const std::vector<T>& values) {
        set_.clear();
        index_map_.clear();
        int32_t idx = 1;
        for (const auto& v : values) {
            set_.insert(v);
            index_map_[v] = idx++;
        }
    }

    /**
     * 检查值是否在集合中
     */
    __attribute__((always_inline))
    bool contains(const T& value) const {
        return set_.count(value) > 0;
    }

    /**
     * 检查值是否不在集合中
     */
    __attribute__((always_inline))
    bool not_contains(const T& value) const {
        return set_.count(value) == 0;
    }

    /**
     * 批量检查，返回匹配的索引
     */
    std::vector<uint32_t> filter_in(const T* values, size_t count) {
        std::vector<uint32_t> result;
        result.reserve(count / 4);
        for (size_t i = 0; i < count; ++i) {
            if (contains(values[i])) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
        return result;
    }

    /**
     * 批量检查，返回不匹配的索引
     */
    std::vector<uint32_t> filter_not_in(const T* values, size_t count) {
        std::vector<uint32_t> result;
        result.reserve(count / 2);
        for (size_t i = 0; i < count; ++i) {
            if (!contains(values[i])) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
        return result;
    }

    /**
     * 获取值的索引 (用于分组)
     * @return 索引 (0 = 不匹配, 1-N = 匹配的第几个值)
     */
    int32_t get_index(const T& value) const {
        auto it = index_map_.find(value);
        return it != index_map_.end() ? it->second : 0;
    }

    /**
     * 批量获取索引
     */
    std::vector<int32_t> get_index_batch(const T* values, size_t count) {
        std::vector<int32_t> result(count);
        for (size_t i = 0; i < count; ++i) {
            result[i] = get_index(values[i]);
        }
        return result;
    }

    size_t size() const { return set_.size(); }

private:
    std::unordered_set<T> set_;
    std::unordered_map<T, int32_t> index_map_;  // 值 → 索引
};

// 特化 string 版本
template<>
class SetMatcher<std::string> {
public:
    void configure(const std::vector<std::string>& values);
    bool contains(const std::string& value) const;
    bool not_contains(const std::string& value) const;
    std::vector<uint32_t> filter_in(const std::vector<std::string>& values);
    std::vector<uint32_t> filter_not_in(const std::vector<std::string>& values);
    int32_t get_index(const std::string& value) const;
    std::vector<int32_t> get_index_batch(const std::vector<std::string>& values);
    size_t size() const { return set_.size(); }

private:
    std::unordered_set<std::string> set_;
    std::unordered_map<std::string, int32_t> index_map_;
};

// ============================================================================
// 通用 ANTI JOIN 算子
// ============================================================================

/**
 * ANTI JOIN 类型
 */
enum class AntiJoinType {
    LEFT_ANTI,   // NOT EXISTS / NOT IN (返回左表中不在右表的行)
    RIGHT_ANTI   // 反向 (返回右表中不在左表的行)
};

/**
 * 通用 ANTI JOIN 算子
 *
 * 用于实现 NOT EXISTS / NOT IN 语义
 *
 * 示例:
 *   NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
 *   → GenericAntiJoin(customers.custkey, orders.custkey)
 */
class GenericAntiJoin {
public:
    GenericAntiJoin() = default;

    /**
     * 配置 ANTI JOIN 参数
     * @param cfg 查询配置
     * @param type_name 类型参数名 (从配置读取)
     */
    void configure(const QueryConfig& cfg, const std::string& type_name = "");

    /**
     * 设置 ANTI JOIN 类型
     */
    void set_type(AntiJoinType type) { type_ = type; }

    /**
     * Build 阶段 - 构建存在性哈希表
     * @param keys 键数组
     * @param count 数量
     */
    void build(const int32_t* keys, size_t count);

    /**
     * Build 阶段 (带过滤条件)
     * @param keys 键数组
     * @param count 数量
     * @param filter 过滤谓词 (返回 true 表示包含)
     */
    void build_filtered(const int32_t* keys, size_t count,
                        const std::function<bool(size_t)>& filter);

    /**
     * Probe 阶段 - 返回不存在的行索引
     * @param probe_keys 探测键数组
     * @param probe_count 数量
     * @return 不匹配的行索引
     */
    std::vector<uint32_t> probe(const int32_t* probe_keys, size_t probe_count);

    /**
     * Probe 阶段 (带额外过滤条件)
     */
    std::vector<uint32_t> probe_filtered(
        const int32_t* probe_keys, size_t probe_count,
        const std::function<bool(size_t)>& filter);

    /**
     * 检查键是否存在
     */
    __attribute__((always_inline))
    bool exists(int32_t key) const {
        if (use_bloom_ && !bloom_.may_contain(key)) return false;
        return exist_set_.count(key) > 0;
    }

    /**
     * 检查键是否不存在
     */
    __attribute__((always_inline))
    bool not_exists(int32_t key) const {
        return !exists(key);
    }

    size_t build_count() const { return exist_set_.size(); }

private:
    AntiJoinType type_ = AntiJoinType::LEFT_ANTI;
    std::unordered_set<int32_t> exist_set_;
    SingleHashBloomFilter bloom_;
    bool use_bloom_ = false;
};

// ============================================================================
// 通用 OUTER JOIN 算子
// ============================================================================

/**
 * OUTER JOIN 类型
 */
enum class OuterJoinType {
    LEFT_OUTER,   // 保留左表所有行
    RIGHT_OUTER,  // 保留右表所有行
    FULL_OUTER    // 保留两表所有行
};

/**
 * OUTER JOIN 聚合类型
 */
enum class OuterJoinAggType {
    COUNT,        // COUNT(right_key) - NULL 计为 0
    SUM,          // SUM(right_value)
    MIN,          // MIN(right_value)
    MAX,          // MAX(right_value)
    FIRST,        // 返回第一个匹配值
    COLLECT       // 收集所有匹配值
};

/**
 * 通用 OUTER JOIN 算子
 *
 * 支持 LEFT/RIGHT/FULL OUTER JOIN
 * 可配置聚合函数
 */
class GenericOuterJoin {
public:
    GenericOuterJoin() = default;

    /**
     * 配置 OUTER JOIN 参数
     */
    void configure(const QueryConfig& cfg,
                   const std::string& type_name = "",
                   const std::string& agg_name = "");

    /**
     * 设置 JOIN 类型和聚合类型
     */
    void set_type(OuterJoinType type) { type_ = type; }
    void set_agg_type(OuterJoinAggType agg) { agg_type_ = agg; }

    /**
     * Build 阶段 - 构建右表哈希
     * @param keys 右表键数组
     * @param count 数量
     */
    void build(const int32_t* keys, size_t count);

    /**
     * Build 阶段 (带过滤条件)
     */
    void build_filtered(const int32_t* keys, size_t count,
                        const std::function<bool(size_t)>& filter);

    /**
     * Build 阶段 (带值)
     */
    void build_with_values(const int32_t* keys, const int64_t* values, size_t count);

    /**
     * Probe 并聚合 - 返回每个左表键的聚合结果
     * @param probe_keys 左表键数组
     * @param probe_count 数量
     * @return 聚合结果 (COUNT 返回计数，SUM 返回和)
     */
    std::vector<int64_t> probe_aggregate(const int32_t* probe_keys, size_t probe_count);

    /**
     * Probe 并返回匹配计数
     */
    std::vector<int32_t> probe_count(const int32_t* probe_keys, size_t probe_count);

    /**
     * 获取单个键的聚合结果
     */
    __attribute__((always_inline))
    int64_t get_aggregate(int32_t key) const {
        auto it = agg_map_.find(key);
        return it != agg_map_.end() ? it->second : 0;
    }

    /**
     * 获取单个键的匹配计数
     */
    __attribute__((always_inline))
    int32_t get_count(int32_t key) const {
        auto it = count_map_.find(key);
        return it != count_map_.end() ? it->second : 0;
    }

private:
    OuterJoinType type_ = OuterJoinType::LEFT_OUTER;
    OuterJoinAggType agg_type_ = OuterJoinAggType::COUNT;

    std::unordered_map<int32_t, int32_t> count_map_;   // key → count
    std::unordered_map<int32_t, int64_t> agg_map_;     // key → aggregate
};

// ============================================================================
// 通用条件聚合框架
// ============================================================================

/**
 * 条件分支定义
 *
 * 对应 SQL: CASE WHEN condition THEN value ELSE default END
 */
struct CaseBranch {
    // 条件类型
    enum class ConditionType {
        INT_EQUALS,       // int_column = value
        INT_IN_SET,       // int_column IN (...)
        STRING_EQUALS,    // string_column = value
        STRING_IN_SET,    // string_column IN (...)
        STRING_LIKE,      // string_column LIKE pattern
        ALWAYS_TRUE       // ELSE 分支
    };

    ConditionType type = ConditionType::ALWAYS_TRUE;

    // 条件值
    int32_t int_value = 0;
    std::string string_value;
    std::vector<int32_t> int_set;
    std::vector<std::string> string_set;
    std::string like_pattern;

    // 分支 ID (用于结果聚合)
    int branch_id = 0;

    /**
     * 检查整数列是否满足条件
     */
    bool evaluate_int(int32_t value) const;

    /**
     * 检查字符串列是否满足条件
     */
    bool evaluate_string(const std::string& value) const;
};

/**
 * 通用条件聚合器
 *
 * 支持任意 CASE WHEN 表达式的聚合
 *
 * 示例:
 *   SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END)
 *   → ConditionalAggregator with CaseBranch(STRING_EQUALS, "BRAZIL")
 */
class GenericConditionalAggregator {
public:
    GenericConditionalAggregator() = default;

    /**
     * 配置分组键范围
     * @param min_key 最小分组键
     * @param max_key 最大分组键
     */
    void configure_groups(int32_t min_key, int32_t max_key);

    /**
     * 从配置加载
     */
    void configure(const QueryConfig& cfg,
                   const std::string& min_key_name,
                   const std::string& max_key_name);

    /**
     * 添加条件分支
     * @param branch 条件分支定义
     */
    void add_branch(CaseBranch branch);

    /**
     * 添加整数相等条件分支
     */
    void add_int_equals_branch(int branch_id, int32_t value);

    /**
     * 添加字符串相等条件分支
     */
    void add_string_equals_branch(int branch_id, const std::string& value);

    /**
     * 添加字符串集合条件分支
     */
    void add_string_in_set_branch(int branch_id, const std::vector<std::string>& values);

    /**
     * 添加 ELSE 分支
     */
    void add_else_branch(int branch_id);

    /**
     * 聚合 - 整数条件列
     * @param group_key 分组键
     * @param condition_value 条件列值
     * @param agg_value 聚合值
     */
    void aggregate_int(int32_t group_key, int32_t condition_value, int64_t agg_value);

    /**
     * 聚合 - 字符串条件列
     */
    void aggregate_string(int32_t group_key, const std::string& condition_value, int64_t agg_value);

    /**
     * 获取分支的聚合结果
     * @param group_key 分组键
     * @param branch_id 分支 ID
     */
    int64_t get_result(int32_t group_key, int branch_id) const;

    /**
     * 获取所有分组的结果
     * @return (group_key, branch_results) 对
     */
    std::vector<std::pair<int32_t, std::vector<int64_t>>> get_all_results() const;

    /**
     * 获取有效分组数
     */
    size_t group_count() const { return max_key_ - min_key_ + 1; }

    /**
     * 获取分支数
     */
    size_t branch_count() const { return branches_.size(); }

private:
    int32_t min_key_ = 0;
    int32_t max_key_ = 0;
    std::vector<CaseBranch> branches_;

    // 结果存储: [group_idx][branch_id] → aggregate
    std::vector<std::vector<int64_t>> results_;

    int evaluate_branches_int(int32_t value) const;
    int evaluate_branches_string(const std::string& value) const;
};

// ============================================================================
// 查询配置扩展
// ============================================================================

/**
 * V34 配置扩展
 *
 * 扩展 QueryConfig 支持新的参数类型
 */
namespace V34Config {

/**
 * 配置 ANTI JOIN
 */
void configure_anti_join(QueryConfig& cfg,
                         const std::string& name,
                         AntiJoinType type);

/**
 * 配置 OUTER JOIN
 */
void configure_outer_join(QueryConfig& cfg,
                          const std::string& name,
                          OuterJoinType type,
                          OuterJoinAggType agg_type);

/**
 * 配置条件聚合分支
 */
void configure_case_branch(QueryConfig& cfg,
                           const std::string& name,
                           const CaseBranch& branch);

/**
 * 从配置获取 ANTI JOIN 类型
 */
AntiJoinType get_anti_join_type(const QueryConfig& cfg, const std::string& name);

/**
 * 从配置获取 OUTER JOIN 类型
 */
OuterJoinType get_outer_join_type(const QueryConfig& cfg, const std::string& name);

}  // namespace V34Config

// ============================================================================
// V34 TPC-H 查询配置工厂
// ============================================================================

namespace V34ConfigFactory {

/**
 * Q22 配置
 * - phone_substring_start: 1
 * - phone_substring_length: 2
 * - country_codes: ["13", "31", "23", "29", "30", "18", "17"]
 * - anti_join_type: LEFT_ANTI
 */
QueryConfig q22_config();

/**
 * Q13 配置
 * - exclude_pattern: "%special%requests%"
 * - outer_join_type: LEFT_OUTER
 * - outer_join_agg: COUNT
 */
QueryConfig q13_config();

/**
 * Q8 配置
 * - target_nation: "BRAZIL"
 * - region: "AMERICA"
 * - part_type: "ECONOMY ANODIZED STEEL"
 * - date_range: 1995-01-01 to 1996-12-31
 * - case_branch: STRING_EQUALS("BRAZIL")
 */
QueryConfig q8_config();

}  // namespace V34ConfigFactory

// ============================================================================
// V34 查询入口
// ============================================================================

/**
 * Q22 V34 - 使用通用算子实现
 */
void run_q22_v34(TPCHDataLoader& loader, const QueryConfig& config);
void run_q22_v34(TPCHDataLoader& loader);

/**
 * Q13 V34 - 使用通用算子实现
 */
void run_q13_v34(TPCHDataLoader& loader, const QueryConfig& config);
void run_q13_v34(TPCHDataLoader& loader);

/**
 * Q8 V34 - 使用通用算子实现
 */
void run_q8_v34(TPCHDataLoader& loader, const QueryConfig& config);
void run_q8_v34(TPCHDataLoader& loader);

}  // namespace ops_v34
}  // namespace tpch
}  // namespace thunderduck

#endif  // TPCH_OPERATORS_V34_H
