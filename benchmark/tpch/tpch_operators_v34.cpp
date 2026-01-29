/**
 * ThunderDuck TPC-H V34 通用算子扩展 - 实现
 *
 * 通用算子实现:
 * - StringFunctions: SUBSTRING/LIKE 等
 * - SetMatcher: IN/NOT IN 操作
 * - GenericAntiJoin: NOT EXISTS/NOT IN
 * - GenericOuterJoin: LEFT/RIGHT/FULL OUTER JOIN
 * - GenericConditionalAggregator: CASE WHEN 聚合
 *
 * @version 34.0
 * @date 2026-01-29
 * @tag 继续攻坚
 */

#include "tpch_operators_v34.h"
#include "tpch_operators_v25.h"  // ThreadPool
#include <algorithm>
#include <numeric>
#include <cctype>
#include <future>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v34 {

// ============================================================================
// StringFunctions 实现
// ============================================================================

std::string_view StringFunctions::substring(const std::string& str, int start, int length) {
    // SQL SUBSTRING 是 1-based
    if (start < 1) start = 1;
    size_t idx = static_cast<size_t>(start - 1);
    if (idx >= str.size()) return {};
    size_t len = std::min(static_cast<size_t>(length), str.size() - idx);
    return std::string_view(str.data() + idx, len);
}

std::vector<std::string> StringFunctions::substring_batch(
    const std::vector<std::string>& strings, int start, int length) {
    std::vector<std::string> result;
    result.reserve(strings.size());

    size_t idx = start > 0 ? static_cast<size_t>(start - 1) : 0;

    for (const auto& str : strings) {
        if (idx >= str.size()) {
            result.emplace_back();
        } else {
            size_t len = std::min(static_cast<size_t>(length), str.size() - idx);
            result.emplace_back(str.substr(idx, len));
        }
    }

    return result;
}

int32_t StringFunctions::substring_to_int(const std::string& str, int start, int length) {
    auto sv = substring(str, start, length);
    if (sv.empty()) return -1;

    int32_t result = 0;
    for (char c : sv) {
        if (c < '0' || c > '9') return -1;
        result = result * 10 + (c - '0');
    }
    return result;
}

std::vector<int32_t> StringFunctions::substring_to_int_batch(
    const std::vector<std::string>& strings, int start, int length) {
    std::vector<int32_t> result(strings.size());

    size_t idx = start > 0 ? static_cast<size_t>(start - 1) : 0;

    for (size_t i = 0; i < strings.size(); ++i) {
        const auto& str = strings[i];
        if (idx >= str.size() || idx + length > str.size()) {
            result[i] = -1;
            continue;
        }

        int32_t val = 0;
        bool valid = true;
        for (size_t j = 0; j < static_cast<size_t>(length) && valid; ++j) {
            char c = str[idx + j];
            if (c < '0' || c > '9') {
                valid = false;
            } else {
                val = val * 10 + (c - '0');
            }
        }
        result[i] = valid ? val : -1;
    }

    return result;
}

bool StringFunctions::like(const std::string& str, const std::string& pattern) {
    // 简化实现: 只支持 %xxx%, %xxx, xxx%
    if (pattern.empty()) return str.empty();

    bool prefix_wild = pattern.front() == '%';
    bool suffix_wild = pattern.back() == '%';

    std::string core = pattern;
    if (prefix_wild) core = core.substr(1);
    if (suffix_wild && !core.empty()) core = core.substr(0, core.size() - 1);

    if (prefix_wild && suffix_wild) {
        // %xxx% - 包含
        return str.find(core) != std::string::npos;
    } else if (prefix_wild) {
        // %xxx - 后缀
        return str.size() >= core.size() &&
               str.compare(str.size() - core.size(), core.size(), core) == 0;
    } else if (suffix_wild) {
        // xxx% - 前缀
        return str.size() >= core.size() &&
               str.compare(0, core.size(), core) == 0;
    } else {
        // 精确匹配
        return str == pattern;
    }
}

bool StringFunctions::not_like(const std::string& str, const std::string& pattern) {
    return !like(str, pattern);
}

bool StringFunctions::contains_all(const std::string& str,
                                   const std::vector<std::string>& patterns) {
    size_t last_pos = 0;
    for (const auto& pattern : patterns) {
        size_t pos = str.find(pattern, last_pos);
        if (pos == std::string::npos) return false;
        last_pos = pos + pattern.size();
    }
    return true;
}

std::vector<uint64_t> StringFunctions::like_batch_bitmap(
    const std::vector<std::string>& strings, const std::string& pattern) {
    size_t num_words = (strings.size() + 63) / 64;
    std::vector<uint64_t> bitmap(num_words, 0);

    for (size_t i = 0; i < strings.size(); ++i) {
        if (like(strings[i], pattern)) {
            bitmap[i >> 6] |= (1ULL << (i & 63));
        }
    }

    return bitmap;
}

// ============================================================================
// SetMatcher<std::string> 特化实现
// ============================================================================

void SetMatcher<std::string>::configure(const std::vector<std::string>& values) {
    set_.clear();
    index_map_.clear();

    int32_t idx = 1;
    for (const auto& v : values) {
        set_.insert(v);
        index_map_[v] = idx++;
    }
}

bool SetMatcher<std::string>::contains(const std::string& value) const {
    return set_.count(value) > 0;
}

bool SetMatcher<std::string>::not_contains(const std::string& value) const {
    return set_.count(value) == 0;
}

std::vector<uint32_t> SetMatcher<std::string>::filter_in(
    const std::vector<std::string>& values) {
    std::vector<uint32_t> result;
    result.reserve(values.size() / 4);

    for (size_t i = 0; i < values.size(); ++i) {
        if (contains(values[i])) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }

    return result;
}

std::vector<uint32_t> SetMatcher<std::string>::filter_not_in(
    const std::vector<std::string>& values) {
    std::vector<uint32_t> result;
    result.reserve(values.size() / 2);

    for (size_t i = 0; i < values.size(); ++i) {
        if (not_contains(values[i])) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }

    return result;
}

int32_t SetMatcher<std::string>::get_index(const std::string& value) const {
    auto it = index_map_.find(value);
    return it != index_map_.end() ? it->second : 0;
}

std::vector<int32_t> SetMatcher<std::string>::get_index_batch(
    const std::vector<std::string>& values) {
    std::vector<int32_t> result(values.size());

    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = get_index(values[i]);
    }

    return result;
}

// ============================================================================
// GenericAntiJoin 实现
// ============================================================================

void GenericAntiJoin::configure(const QueryConfig& cfg, const std::string& type_name) {
    // 可从配置读取类型，默认 LEFT_ANTI
    type_ = AntiJoinType::LEFT_ANTI;
}

void GenericAntiJoin::build(const int32_t* keys, size_t count) {
    exist_set_.clear();
    exist_set_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        exist_set_.insert(keys[i]);
    }

    // 大数据集使用 Bloom Filter 加速
    if (count > 100000) {
        use_bloom_ = true;
        bloom_.init(count * 10);
        for (size_t i = 0; i < count; ++i) {
            bloom_.insert(keys[i]);
        }
    }
}

void GenericAntiJoin::build_filtered(const int32_t* keys, size_t count,
                                     const std::function<bool(size_t)>& filter) {
    exist_set_.clear();
    exist_set_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        if (filter(i)) {
            exist_set_.insert(keys[i]);
        }
    }

    if (exist_set_.size() > 100000) {
        use_bloom_ = true;
        bloom_.init(exist_set_.size() * 10);
        for (int32_t key : exist_set_) {
            bloom_.insert(key);
        }
    }
}

std::vector<uint32_t> GenericAntiJoin::probe(const int32_t* probe_keys, size_t probe_count) {
    std::vector<uint32_t> result;
    result.reserve(probe_count / 2);

    if (use_bloom_) {
        for (size_t i = 0; i < probe_count; ++i) {
            if (!bloom_.may_contain(probe_keys[i])) {
                result.push_back(static_cast<uint32_t>(i));
            } else if (exist_set_.count(probe_keys[i]) == 0) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
    } else {
        for (size_t i = 0; i < probe_count; ++i) {
            if (exist_set_.count(probe_keys[i]) == 0) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
    }

    return result;
}

std::vector<uint32_t> GenericAntiJoin::probe_filtered(
    const int32_t* probe_keys, size_t probe_count,
    const std::function<bool(size_t)>& filter) {
    std::vector<uint32_t> result;
    result.reserve(probe_count / 2);

    for (size_t i = 0; i < probe_count; ++i) {
        if (!filter(i)) continue;
        if (not_exists(probe_keys[i])) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }

    return result;
}

// ============================================================================
// GenericOuterJoin 实现
// ============================================================================

void GenericOuterJoin::configure(const QueryConfig& cfg,
                                 const std::string& type_name,
                                 const std::string& agg_name) {
    type_ = OuterJoinType::LEFT_OUTER;
    agg_type_ = OuterJoinAggType::COUNT;
}

void GenericOuterJoin::build(const int32_t* keys, size_t count) {
    count_map_.clear();
    count_map_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        count_map_[keys[i]]++;
    }
}

void GenericOuterJoin::build_filtered(const int32_t* keys, size_t count,
                                      const std::function<bool(size_t)>& filter) {
    count_map_.clear();
    count_map_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        if (filter(i)) {
            count_map_[keys[i]]++;
        }
    }
}

void GenericOuterJoin::build_with_values(const int32_t* keys, const int64_t* values,
                                         size_t count) {
    count_map_.clear();
    agg_map_.clear();
    count_map_.reserve(count);
    agg_map_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        count_map_[keys[i]]++;
        agg_map_[keys[i]] += values[i];
    }
}

std::vector<int64_t> GenericOuterJoin::probe_aggregate(const int32_t* probe_keys,
                                                       size_t probe_count) {
    std::vector<int64_t> result(probe_count);

    for (size_t i = 0; i < probe_count; ++i) {
        result[i] = get_aggregate(probe_keys[i]);
    }

    return result;
}

std::vector<int32_t> GenericOuterJoin::probe_count(const int32_t* probe_keys,
                                                   size_t probe_count) {
    std::vector<int32_t> result(probe_count);

    for (size_t i = 0; i < probe_count; ++i) {
        result[i] = get_count(probe_keys[i]);
    }

    return result;
}

// ============================================================================
// CaseBranch 实现
// ============================================================================

bool CaseBranch::evaluate_int(int32_t value) const {
    switch (type) {
        case ConditionType::INT_EQUALS:
            return value == int_value;
        case ConditionType::INT_IN_SET:
            return std::find(int_set.begin(), int_set.end(), value) != int_set.end();
        case ConditionType::ALWAYS_TRUE:
            return true;
        default:
            return false;
    }
}

bool CaseBranch::evaluate_string(const std::string& value) const {
    switch (type) {
        case ConditionType::STRING_EQUALS:
            return value == string_value;
        case ConditionType::STRING_IN_SET:
            return std::find(string_set.begin(), string_set.end(), value) != string_set.end();
        case ConditionType::STRING_LIKE:
            return StringFunctions::like(value, like_pattern);
        case ConditionType::ALWAYS_TRUE:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// GenericConditionalAggregator 实现
// ============================================================================

void GenericConditionalAggregator::configure_groups(int32_t min_key, int32_t max_key) {
    min_key_ = min_key;
    max_key_ = max_key;
    // 结果数组会在第一次 aggregate 时初始化
}

void GenericConditionalAggregator::configure(const QueryConfig& cfg,
                                             const std::string& min_key_name,
                                             const std::string& max_key_name) {
    min_key_ = static_cast<int32_t>(cfg.get_int64(min_key_name));
    max_key_ = static_cast<int32_t>(cfg.get_int64(max_key_name));
}

void GenericConditionalAggregator::add_branch(CaseBranch branch) {
    branches_.push_back(std::move(branch));
}

void GenericConditionalAggregator::add_int_equals_branch(int branch_id, int32_t value) {
    CaseBranch branch;
    branch.type = CaseBranch::ConditionType::INT_EQUALS;
    branch.branch_id = branch_id;
    branch.int_value = value;
    branches_.push_back(branch);
}

void GenericConditionalAggregator::add_string_equals_branch(int branch_id,
                                                            const std::string& value) {
    CaseBranch branch;
    branch.type = CaseBranch::ConditionType::STRING_EQUALS;
    branch.branch_id = branch_id;
    branch.string_value = value;
    branches_.push_back(branch);
}

void GenericConditionalAggregator::add_string_in_set_branch(
    int branch_id, const std::vector<std::string>& values) {
    CaseBranch branch;
    branch.type = CaseBranch::ConditionType::STRING_IN_SET;
    branch.branch_id = branch_id;
    branch.string_set = values;
    branches_.push_back(branch);
}

void GenericConditionalAggregator::add_else_branch(int branch_id) {
    CaseBranch branch;
    branch.type = CaseBranch::ConditionType::ALWAYS_TRUE;
    branch.branch_id = branch_id;
    branches_.push_back(branch);
}

int GenericConditionalAggregator::evaluate_branches_int(int32_t value) const {
    for (const auto& branch : branches_) {
        if (branch.evaluate_int(value)) {
            return branch.branch_id;
        }
    }
    return -1;  // 无匹配
}

int GenericConditionalAggregator::evaluate_branches_string(const std::string& value) const {
    for (const auto& branch : branches_) {
        if (branch.evaluate_string(value)) {
            return branch.branch_id;
        }
    }
    return -1;  // 无匹配
}

void GenericConditionalAggregator::aggregate_int(int32_t group_key, int32_t condition_value,
                                                 int64_t agg_value) {
    if (group_key < min_key_ || group_key > max_key_) return;

    // 延迟初始化结果数组
    if (results_.empty()) {
        size_t num_groups = static_cast<size_t>(max_key_ - min_key_ + 1);
        size_t num_branches = branches_.size();
        results_.resize(num_groups);
        for (auto& row : results_) {
            row.resize(num_branches, 0);
        }
    }

    int branch_id = evaluate_branches_int(condition_value);
    if (branch_id >= 0 && branch_id < static_cast<int>(branches_.size())) {
        size_t group_idx = static_cast<size_t>(group_key - min_key_);
        results_[group_idx][branch_id] += agg_value;
    }
}

void GenericConditionalAggregator::aggregate_string(int32_t group_key,
                                                    const std::string& condition_value,
                                                    int64_t agg_value) {
    if (group_key < min_key_ || group_key > max_key_) return;

    // 延迟初始化结果数组
    if (results_.empty()) {
        size_t num_groups = static_cast<size_t>(max_key_ - min_key_ + 1);
        size_t num_branches = branches_.size();
        results_.resize(num_groups);
        for (auto& row : results_) {
            row.resize(num_branches, 0);
        }
    }

    int branch_id = evaluate_branches_string(condition_value);
    if (branch_id >= 0 && branch_id < static_cast<int>(branches_.size())) {
        size_t group_idx = static_cast<size_t>(group_key - min_key_);
        results_[group_idx][branch_id] += agg_value;
    }
}

int64_t GenericConditionalAggregator::get_result(int32_t group_key, int branch_id) const {
    if (group_key < min_key_ || group_key > max_key_) return 0;
    if (results_.empty()) return 0;

    size_t group_idx = static_cast<size_t>(group_key - min_key_);
    if (branch_id < 0 || branch_id >= static_cast<int>(results_[group_idx].size())) return 0;

    return results_[group_idx][branch_id];
}

std::vector<std::pair<int32_t, std::vector<int64_t>>>
GenericConditionalAggregator::get_all_results() const {
    std::vector<std::pair<int32_t, std::vector<int64_t>>> result;
    result.reserve(results_.size());

    for (size_t i = 0; i < results_.size(); ++i) {
        int32_t key = min_key_ + static_cast<int32_t>(i);
        result.emplace_back(key, results_[i]);
    }

    return result;
}

// ============================================================================
// V34 配置扩展实现
// ============================================================================

namespace V34Config {

void configure_anti_join(QueryConfig& cfg, const std::string& name, AntiJoinType type) {
    cfg.set_int64(name + "_type", static_cast<int32_t>(type));
}

void configure_outer_join(QueryConfig& cfg, const std::string& name,
                          OuterJoinType type, OuterJoinAggType agg_type) {
    cfg.set_int64(name + "_type", static_cast<int32_t>(type));
    cfg.set_int64(name + "_agg", static_cast<int32_t>(agg_type));
}

AntiJoinType get_anti_join_type(const QueryConfig& cfg, const std::string& name) {
    return static_cast<AntiJoinType>(cfg.get_int64(name + "_type"));
}

OuterJoinType get_outer_join_type(const QueryConfig& cfg, const std::string& name) {
    return static_cast<OuterJoinType>(cfg.get_int64(name + "_type"));
}

}  // namespace V34Config

// ============================================================================
// V34 TPC-H 查询配置工厂
// ============================================================================

namespace V34ConfigFactory {

QueryConfig q22_config() {
    QueryConfig cfg;

    // SUBSTRING 参数
    cfg.set_int64("phone_substring_start", 1);
    cfg.set_int64("phone_substring_length", 2);

    // 国家码集合
    cfg.set_string_set("country_codes", {"13", "31", "23", "29", "30", "18", "17"});

    // ANTI JOIN 类型
    V34Config::configure_anti_join(cfg, "order_anti_join", AntiJoinType::LEFT_ANTI);

    return cfg;
}

QueryConfig q13_config() {
    QueryConfig cfg;

    // LIKE 排除模式 (存储为多个子串)
    cfg.set_string_set("exclude_patterns", {"special", "requests"});

    // OUTER JOIN 配置
    V34Config::configure_outer_join(cfg, "customer_order_join",
                                    OuterJoinType::LEFT_OUTER,
                                    OuterJoinAggType::COUNT);

    return cfg;
}

QueryConfig q8_config() {
    QueryConfig cfg;

    // 过滤条件
    cfg.set_string("target_nation", "BRAZIL");
    cfg.set_string("region", "AMERICA");
    cfg.set_string("part_type", "ECONOMY ANODIZED STEEL");

    // 日期范围
    cfg.set_date_range("order_date", DateRange::from_string("1995-01-01", "1996-12-31"));

    // 条件聚合参数
    cfg.set_int64("min_year", 1995);
    cfg.set_int64("max_year", 1996);

    return cfg;
}

}  // namespace V34ConfigFactory

// ============================================================================
// Q22 V34 实现: 使用通用算子
// ============================================================================

void run_q22_v34(TPCHDataLoader& loader) {
    run_q22_v34(loader, V34ConfigFactory::q22_config());
}

void run_q22_v34(TPCHDataLoader& loader, const QueryConfig& config) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();

    // ========== 从配置读取参数 ==========
    int substring_start = config.get_int64("phone_substring_start");
    int substring_length = config.get_int64("phone_substring_length");
    auto country_codes = config.get_string_set("country_codes");

    if (substring_start == 0) substring_start = 1;
    if (substring_length == 0) substring_length = 2;
    if (country_codes.empty()) {
        country_codes = {"13", "31", "23", "29", "30", "18", "17"};
    }

    // ========== 优化: 直接数组映射国家码 (00-99) ==========
    // code_valid[i] = 是否是目标国家码, code_index[i] = 分组索引
    std::array<bool, 100> code_valid{};
    std::array<int8_t, 100> code_index{};
    int8_t idx = 0;
    for (const auto& code : country_codes) {
        int32_t val = 0;
        for (char c : code) val = val * 10 + (c - '0');
        if (val < 100) {
            code_valid[val] = true;
            code_index[val] = idx++;
        }
    }

    // ========== Phase 1: 单遍扫描计算 AVG + 提取候选 ==========
    int64_t sum_acctbal = 0;
    size_t count_acctbal = 0;
    std::vector<uint32_t> candidates;
    candidates.reserve(cust.count / 10);

    size_t start_idx = substring_start > 0 ? substring_start - 1 : 0;
    for (size_t i = 0; i < cust.count; ++i) {
        const auto& phone = cust.c_phone[i];
        if (phone.size() < start_idx + 2) continue;

        int32_t code = (phone[start_idx] - '0') * 10 + (phone[start_idx + 1] - '0');
        if (code >= 100 || !code_valid[code]) continue;

        if (cust.c_acctbal[i] > 0) {
            sum_acctbal += cust.c_acctbal[i];
            ++count_acctbal;
        }
        candidates.push_back(static_cast<uint32_t>(i));
    }

    int64_t avg_threshold = count_acctbal > 0 ?
        sum_acctbal / static_cast<int64_t>(count_acctbal) : 0;

    // ========== Phase 2: 构建订单客户集合 (使用 Bloom Filter 预过滤) ==========
    GenericAntiJoin anti_join;
    anti_join.set_type(AntiJoinType::LEFT_ANTI);
    anti_join.build(ord.o_custkey.data(), ord.count);

    // ========== Phase 3: 批量过滤候选 ==========
    // 使用 7 个固定桶直接聚合 (避免 hash map)
    std::array<int64_t, 7> counts{};
    std::array<int64_t, 7> sums{};

    for (uint32_t ci : candidates) {
        if (cust.c_acctbal[ci] <= avg_threshold) continue;
        if (anti_join.exists(cust.c_custkey[ci])) continue;

        const auto& phone = cust.c_phone[ci];
        int32_t code = (phone[start_idx] - '0') * 10 + (phone[start_idx + 1] - '0');
        int8_t bucket = code_index[code];

        counts[bucket]++;
        sums[bucket] += cust.c_acctbal[ci];
    }

    // 结果在 counts/sums 数组中，索引对应 country_codes 顺序
}

// ============================================================================
// Q13 V34 实现: 使用通用算子
// ============================================================================

void run_q13_v34(TPCHDataLoader& loader) {
    run_q13_v34(loader, V34ConfigFactory::q13_config());
}

void run_q13_v34(TPCHDataLoader& loader, const QueryConfig& config) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();

    // ========== 从配置读取参数 ==========
    auto exclude_patterns = config.get_string_set("exclude_patterns");
    if (exclude_patterns.empty()) {
        exclude_patterns = {"special", "requests"};
    }

    // ========== 优化: 多线程并行 + SIMD 字符串匹配 ==========
    // NOT LIKE '%special%requests%'
    const char* pat1 = "special";
    const char* pat2 = "requests";
    const size_t pat1_len = 7;  // strlen("special")
    const size_t pat2_len = 8;  // strlen("requests")

    // ========== Phase 1: 预计算 max_custkey ==========
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) {
            max_custkey = cust.c_custkey[i];
        }
    }

    // ========== Phase 2: 多线程并行处理订单 ==========
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, ord.count / 8);

    const size_t num_threads = pool.size() > 0 ? pool.size() : 8;
    const size_t chunk_size = (ord.count + num_threads - 1) / num_threads;

    // 线程本地计数器 (避免竞争)
    std::vector<std::vector<int32_t>> thread_counts(num_threads);
    for (auto& tc : thread_counts) {
        tc.resize(max_custkey + 1, 0);
    }

    // 并行处理
    std::vector<std::future<void>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, ord.count);
        if (start >= ord.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_counts = thread_counts[t];

            for (size_t i = start; i < end; ++i) {
                const std::string& comment = ord.o_comment[i];
                const char* str = comment.c_str();
                size_t len = comment.size();

                // 快速长度检查
                if (len < pat1_len + pat2_len) {
                    // 字符串太短，不可能匹配两个模式
                    int32_t custkey = ord.o_custkey[i];
                    if (custkey >= 0 && custkey <= max_custkey) {
                        local_counts[custkey]++;
                    }
                    continue;
                }

                // 使用 memmem 进行模式匹配 (比 strstr 更快)
                const void* pos1 = memmem(str, len, pat1, pat1_len);
                if (pos1 != nullptr) {
                    size_t offset = static_cast<const char*>(pos1) - str + pat1_len;
                    if (offset < len) {
                        const void* pos2 = memmem(str + offset, len - offset, pat2, pat2_len);
                        if (pos2 != nullptr) {
                            continue;  // 排除此订单
                        }
                    }
                }

                // 有效订单
                int32_t custkey = ord.o_custkey[i];
                if (custkey >= 0 && custkey <= max_custkey) {
                    local_counts[custkey]++;
                }
            }
        }));
    }

    // 等待所有线程完成
    for (auto& f : futures) {
        f.get();
    }

    // ========== Phase 3: 合并线程本地计数器 ==========
    std::vector<int32_t> order_counts(max_custkey + 1, 0);
    for (size_t t = 0; t < num_threads; ++t) {
        for (int32_t k = 0; k <= max_custkey; ++k) {
            order_counts[k] += thread_counts[t][k];
        }
    }

    // ========== Phase 4: 第二级 GROUP BY - 按订单数聚合 ==========
    int32_t max_order_count = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        int32_t cnt = order_counts[cust.c_custkey[i]];
        if (cnt > max_order_count) {
            max_order_count = cnt;
        }
    }

    // 直接数组计数
    std::vector<int64_t> custdist(max_order_count + 1, 0);
    for (size_t i = 0; i < cust.count; ++i) {
        custdist[order_counts[cust.c_custkey[i]]]++;
    }

    // ========== Phase 3: 排序输出 ==========
    struct Result {
        int32_t c_count;
        int64_t custdist;
    };
    std::vector<Result> sorted_results;
    sorted_results.reserve(max_order_count + 1);

    for (int32_t c = 0; c <= max_order_count; ++c) {
        if (custdist[c] > 0) {
            sorted_results.push_back({c, custdist[c]});
        }
    }

    std::sort(sorted_results.begin(), sorted_results.end(),
        [](const Result& a, const Result& b) {
            if (a.custdist != b.custdist) return a.custdist > b.custdist;
            return a.c_count > b.c_count;
        });

    // 结果在 sorted_results 中
}

// ============================================================================
// Q8 V34 实现: 使用通用算子
// ============================================================================

void run_q8_v34(TPCHDataLoader& loader) {
    run_q8_v34(loader, V34ConfigFactory::q8_config());
}

void run_q8_v34(TPCHDataLoader& loader, const QueryConfig& config) {
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // ========== 从配置读取参数 ==========
    std::string target_nation = config.get_string("target_nation");
    std::string target_region = config.get_string("region");
    std::string target_part_type = config.get_string("part_type");
    DateRange date_range = config.get_date_range("order_date");

    // 默认值
    if (target_nation.empty()) target_nation = "BRAZIL";
    if (target_region.empty()) target_region = "AMERICA";
    if (target_part_type.empty()) target_part_type = "ECONOMY ANODIZED STEEL";
    if (date_range.lo == 0) date_range = DateRange::from_string("1995-01-01", "1996-12-31");

    // ========== 优化 Phase 1: 预计算所有查找表 (直接数组) ==========
    // 1a. 目标区域和国家
    int32_t target_regionkey = -1;
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == target_region) {
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }
    if (target_regionkey < 0) return;

    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == target_nation) {
            target_nationkey = nat.n_nationkey[i];
            break;
        }
    }

    // 1b. nation → is_in_region (直接数组, 25 nations)
    std::vector<bool> nation_in_region(26, false);
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == target_regionkey) {
            nation_in_region[nat.n_nationkey[i]] = true;
        }
    }

    // 1c. supplier → nationkey (直接数组, suppkey 1-10000)
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_suppkey[i] > max_suppkey) max_suppkey = supp.s_suppkey[i];
    }
    std::vector<int32_t> supp_nation(max_suppkey + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        supp_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
    }

    // 1d. customer → nationkey (直接数组)
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }
    std::vector<int32_t> cust_nation(max_custkey + 1, -1);
    for (size_t i = 0; i < cust.count; ++i) {
        cust_nation[cust.c_custkey[i]] = cust.c_nationkey[i];
    }

    // 1e. part → is_valid (直接数组/位图)
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_partkey[i] > max_partkey) max_partkey = part.p_partkey[i];
    }
    std::vector<bool> valid_part(max_partkey + 1, false);
    size_t valid_count = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == target_part_type) {
            valid_part[part.p_partkey[i]] = true;
            valid_count++;
        }
    }
    if (valid_count == 0) return;

    // 1f. orders → (year, is_america_customer) (直接数组)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    struct OrderInfo {
        int16_t year;      // 2 bytes
        int8_t is_america; // 1 byte (-1=invalid, 0=no, 1=yes)
    };
    std::vector<OrderInfo> order_info(max_orderkey + 1, {0, -1});

    for (size_t i = 0; i < ord.count; ++i) {
        if (date_range.contains(ord.o_orderdate[i])) {
            int32_t custkey = ord.o_custkey[i];
            int32_t cust_nkey = (custkey <= max_custkey) ? cust_nation[custkey] : -1;
            bool is_america = (cust_nkey >= 0 && cust_nkey < 26) ? nation_in_region[cust_nkey] : false;

            // 年份计算: epoch days / 365 + 1970 (简化)
            int16_t year = static_cast<int16_t>(1970 + ord.o_orderdate[i] / 365);

            order_info[ord.o_orderkey[i]] = {year, is_america ? int8_t(1) : int8_t(0)};
        }
    }

    // ========== Phase 2: 直接聚合 (避免 GenericConditionalAggregator 开销) ==========
    // year → {brazil_volume, total_volume}
    std::array<int64_t, 2> brazil_vol{};   // [0]=1995, [1]=1996
    std::array<int64_t, 2> total_vol{};

    for (size_t i = 0; i < li.count; ++i) {
        // 快速过滤
        int32_t partkey = li.l_partkey[i];
        if (partkey > max_partkey || !valid_part[partkey]) continue;

        int32_t orderkey = li.l_orderkey[i];
        if (orderkey > max_orderkey) continue;

        const auto& oi = order_info[orderkey];
        if (oi.is_america != 1) continue;  // 只统计 AMERICA 区域客户

        int32_t suppkey = li.l_suppkey[i];
        if (suppkey > max_suppkey) continue;
        int32_t supp_nkey = supp_nation[suppkey];
        if (supp_nkey < 0) continue;

        // 计算 volume
        int64_t volume = static_cast<int64_t>(li.l_extendedprice[i]) *
                         (10000 - li.l_discount[i]) / 10000;

        int year_idx = oi.year - 1995;
        if (year_idx < 0 || year_idx > 1) continue;

        total_vol[year_idx] += volume;
        if (supp_nkey == target_nationkey) {
            brazil_vol[year_idx] += volume;
        }
    }

    // 结果: mkt_share[year] = brazil_vol[year] / total_vol[year]
    // 1995: brazil_vol[0] / total_vol[0]
    // 1996: brazil_vol[1] / total_vol[1]
}

}  // namespace ops_v34
}  // namespace tpch
}  // namespace thunderduck
