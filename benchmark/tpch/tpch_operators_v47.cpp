/**
 * ThunderDuck TPC-H V47 通用算子实现
 *
 * 新通用算子:
 * - SIMDBranchlessFilter: SIMD 无分支多条件过滤 (Q6)
 * - SIMDPatternMatcher: SIMD 多模式字符串匹配 (Q13)
 * - ParallelRadixSort 在头文件中已有模板实现
 *
 * @version 47.0
 * @date 2026-01-29
 */

#include "tpch_operators_v47.h"
#include "tpch_operators_v25.h"  // ThreadPool

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <unordered_map>

namespace thunderduck {
namespace tpch {
namespace ops_v47 {

// ============================================================================
// SIMDBranchlessFilter 实现
// ============================================================================

bool SIMDBranchlessFilter::evaluate_row(size_t idx) const {
    for (const auto& cond : conditions_) {
        bool pass = false;

        switch (cond.type) {
            case CondType::RANGE_I32: {
                int32_t val = static_cast<const int32_t*>(cond.column)[idx];
                pass = (val >= static_cast<int32_t>(cond.lo) &&
                        val < static_cast<int32_t>(cond.hi));
                break;
            }
            case CondType::RANGE_I64: {
                int64_t val = static_cast<const int64_t*>(cond.column)[idx];
                pass = (val >= cond.lo && val <= cond.hi);
                break;
            }
            case CondType::LT_I64: {
                int64_t val = static_cast<const int64_t*>(cond.column)[idx];
                pass = (val < cond.hi);
                break;
            }
            case CondType::GT_I64: {
                int64_t val = static_cast<const int64_t*>(cond.column)[idx];
                pass = (val > cond.lo);
                break;
            }
            case CondType::EQ_I32: {
                int32_t val = static_cast<const int32_t*>(cond.column)[idx];
                pass = (val == static_cast<int32_t>(cond.lo));
                break;
            }
            case CondType::BETWEEN_I64: {
                int64_t val = static_cast<const int64_t*>(cond.column)[idx];
                pass = (val >= cond.lo && val <= cond.hi);
                break;
            }
        }

        if (!pass) return false;
    }
    return true;
}

#ifdef __aarch64__
uint32x4_t SIMDBranchlessFilter::evaluate_4rows_neon(size_t start) const {
    // 初始化全部为真
    uint32x4_t result = vdupq_n_u32(0xFFFFFFFF);

    for (const auto& cond : conditions_) {
        uint32x4_t mask;

        switch (cond.type) {
            case CondType::RANGE_I32: {
                const int32_t* col = static_cast<const int32_t*>(cond.column);
                int32x4_t vals = vld1q_s32(col + start);
                int32x4_t lo = vdupq_n_s32(static_cast<int32_t>(cond.lo));
                int32x4_t hi = vdupq_n_s32(static_cast<int32_t>(cond.hi));

                // val >= lo AND val < hi
                uint32x4_t ge_lo = vcgeq_s32(vals, lo);
                uint32x4_t lt_hi = vcltq_s32(vals, hi);
                mask = vandq_u32(ge_lo, lt_hi);
                break;
            }
            case CondType::RANGE_I64:
            case CondType::BETWEEN_I64: {
                // int64 需要标量处理 (NEON 64位比较不如32位高效)
                const int64_t* col = static_cast<const int64_t*>(cond.column);
                uint32_t m[4];
                for (int i = 0; i < 4; ++i) {
                    int64_t val = col[start + i];
                    m[i] = (val >= cond.lo && val <= cond.hi) ? 0xFFFFFFFF : 0;
                }
                mask = vld1q_u32(m);
                break;
            }
            case CondType::LT_I64: {
                const int64_t* col = static_cast<const int64_t*>(cond.column);
                uint32_t m[4];
                for (int i = 0; i < 4; ++i) {
                    m[i] = (col[start + i] < cond.hi) ? 0xFFFFFFFF : 0;
                }
                mask = vld1q_u32(m);
                break;
            }
            case CondType::GT_I64: {
                const int64_t* col = static_cast<const int64_t*>(cond.column);
                uint32_t m[4];
                for (int i = 0; i < 4; ++i) {
                    m[i] = (col[start + i] > cond.lo) ? 0xFFFFFFFF : 0;
                }
                mask = vld1q_u32(m);
                break;
            }
            case CondType::EQ_I32: {
                const int32_t* col = static_cast<const int32_t*>(cond.column);
                int32x4_t vals = vld1q_s32(col + start);
                int32x4_t target = vdupq_n_s32(static_cast<int32_t>(cond.lo));
                mask = vceqq_s32(vals, target);
                break;
            }
        }

        result = vandq_u32(result, mask);
    }

    return result;
}
#endif

std::vector<uint32_t> SIMDBranchlessFilter::execute(size_t count) {
    std::vector<uint32_t> result;
    result.reserve(count / 10);  // 预估 10% 通过率

    for (size_t i = 0; i < count; ++i) {
        if (evaluate_row(i)) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }

    return result;
}

// ============================================================================
// SIMDPatternMatcher 实现
// ============================================================================

void SIMDPatternMatcher::prepare_patterns() {
    prepared_.clear();
    prepared_.reserve(patterns_.size());

    for (const auto& p : patterns_) {
        PreparedPattern pp;
        pp.data = p.value.c_str();
        pp.len = p.value.length();
        pp.first_char = p.value.empty() ? '\0' : p.value[0];
        pp.sequential = p.sequential;
        prepared_.push_back(pp);
    }
}

#ifdef __aarch64__
const char* SIMDPatternMatcher::neon_memchr(const char* haystack, char needle, size_t len) const {
    if (len == 0) return nullptr;

    // 使用 NEON 搜索首字符
    int8x16_t needle_vec = vdupq_n_s8(needle);
    size_t i = 0;

    // 处理 16 字节对齐的块
    for (; i + 16 <= len; i += 16) {
        int8x16_t data = vld1q_s8(reinterpret_cast<const int8_t*>(haystack + i));
        uint8x16_t cmp = vceqq_s8(data, needle_vec);

        // 检查是否有匹配
        uint64_t matches[2];
        vst1q_u64(matches, vreinterpretq_u64_u8(cmp));

        if (matches[0] || matches[1]) {
            // 找到第一个匹配位置
            for (size_t j = 0; j < 16 && i + j < len; ++j) {
                if (haystack[i + j] == needle) {
                    return haystack + i + j;
                }
            }
        }
    }

    // 处理剩余字节
    for (; i < len; ++i) {
        if (haystack[i] == needle) {
            return haystack + i;
        }
    }

    return nullptr;
}
#endif

const char* SIMDPatternMatcher::simd_find(const char* haystack, size_t haystack_len,
                                          const char* needle, size_t needle_len) const {
    if (needle_len == 0) return haystack;
    if (needle_len > haystack_len) return nullptr;

    char first = needle[0];
    size_t search_len = haystack_len - needle_len + 1;

#ifdef __aarch64__
    // 使用 NEON 搜索首字符
    size_t pos = 0;
    while (pos < search_len) {
        const char* found = neon_memchr(haystack + pos, first, search_len - pos);
        if (!found) return nullptr;

        pos = found - haystack;

        // 验证完整模式
        if (std::memcmp(found, needle, needle_len) == 0) {
            return found;
        }
        pos++;
    }
    return nullptr;
#else
    // 标量回退
    for (size_t i = 0; i < search_len; ++i) {
        if (haystack[i] == first) {
            if (std::memcmp(haystack + i, needle, needle_len) == 0) {
                return haystack + i;
            }
        }
    }
    return nullptr;
#endif
}

bool SIMDPatternMatcher::match(const std::string& str) const {
    if (prepared_.empty()) return true;

    const char* haystack = str.c_str();
    size_t haystack_len = str.length();
    const char* search_start = haystack;
    size_t remaining = haystack_len;

    for (size_t i = 0; i < prepared_.size(); ++i) {
        const auto& p = prepared_[i];

        if (p.sequential && i > 0) {
            // 必须在上一个匹配之后搜索
            // search_start 已经被更新到上一个匹配之后
        }

        const char* found = simd_find(search_start, remaining, p.data, p.len);
        if (!found) {
            return false;
        }

        // 更新搜索起点
        search_start = found + p.len;
        remaining = haystack_len - (search_start - haystack);
    }

    return true;
}

void SIMDPatternMatcher::parallel_batch_match(
    const std::vector<std::string>& strings,
    std::vector<uint64_t>& bitmap,
    size_t thread_count) const
{
    size_t count = strings.size();
    size_t bitmap_size = (count + 63) / 64;
    bitmap.assign(bitmap_size, 0);

    if (count == 0) return;

    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(thread_count, count / thread_count);

    size_t chunk_size = (count + thread_count - 1) / thread_count;
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < thread_count; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start >= count) break;

        futures.push_back(pool.submit([this, &strings, &bitmap, start, end]() {
            for (size_t i = start; i < end; ++i) {
                if (match(strings[i])) {
                    // 原子设置位
                    size_t word_idx = i / 64;
                    uint64_t bit = 1ULL << (i % 64);
                    __atomic_or_fetch(&bitmap[word_idx], bit, __ATOMIC_RELAXED);
                }
            }
        }));
    }

    for (auto& f : futures) f.get();
}

std::vector<uint32_t> SIMDPatternMatcher::batch_match_indices(
    const std::vector<std::string>& strings,
    size_t thread_count) const
{
    std::vector<uint64_t> bitmap;
    parallel_batch_match(strings, bitmap, thread_count);

    std::vector<uint32_t> result;
    result.reserve(strings.size() / 10);

    for (size_t i = 0; i < strings.size(); ++i) {
        size_t word_idx = i / 64;
        uint64_t bit = 1ULL << (i % 64);
        if (bitmap[word_idx] & bit) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }

    return result;
}

// ============================================================================
// Q6 V47 实现 - SIMD 无分支过滤 + 直接聚合
// ============================================================================

void run_q6_v47(TPCHDataLoader& loader, const Q6Config& config) {
    const auto& li = loader.lineitem();

    // 配置 SIMD 无分支过滤器
    SIMDBranchlessFilter filter;
    filter.configure({
        // shipdate >= date_lo AND shipdate < date_hi
        {SIMDBranchlessFilter::CondType::RANGE_I32,
         li.l_shipdate.data(), config.date_lo, config.date_hi},
        // discount >= disc_lo AND discount <= disc_hi
        {SIMDBranchlessFilter::CondType::BETWEEN_I64,
         li.l_discount.data(), config.disc_lo, config.disc_hi},
        // quantity < qty_hi
        {SIMDBranchlessFilter::CondType::LT_I64,
         li.l_quantity.data(), 0, config.qty_hi}
    });

    // 获取数据指针
    const int64_t* extprice = li.l_extendedprice.data();
    const int64_t* discount = li.l_discount.data();

    // SIMD 无分支过滤 + 直接聚合
    __int128 revenue = filter.execute_and_aggregate(li.count, 8,
        [extprice, discount](size_t i) -> int64_t {
            return extprice[i] * discount[i];
        });

    // 转换为正确的定点数格式
    volatile double result = static_cast<double>(revenue) / 100000000.0;
    (void)result;
}

// ============================================================================
// Q13 V47 实现 - SIMD 字符串匹配 + 稀疏直接数组
// ============================================================================

void run_q13_v47(TPCHDataLoader& loader, const Q13Config& config) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();

    // Phase 1: SIMD 字符串匹配预过滤订单
    // 找出 NOT LIKE '%special%requests%' 的订单
    SIMDPatternMatcher matcher;
    matcher.configure({
        {config.pattern1, false},   // "special" (anywhere)
        {config.pattern2, true}     // "requests" (must follow "special")
    });

    // 获取匹配的订单索引 (这些是要排除的)
    std::vector<uint64_t> exclude_bitmap;
    matcher.parallel_batch_match(ord.o_comment, exclude_bitmap, 8);

    // Phase 2: 使用 SparseDirectArray 统计每个客户的订单数
    SparseDirectArray<int32_t> cust_counts;
    cust_counts.build_from_keys(cust.c_custkey.data(), cust.count);

    // Phase 3: 统计每个客户的有效订单数 (排除匹配的)
    for (size_t i = 0; i < ord.count; ++i) {
        // 检查是否在排除位图中
        size_t word_idx = i / 64;
        uint64_t bit = 1ULL << (i % 64);
        bool excluded = (word_idx < exclude_bitmap.size()) &&
                        (exclude_bitmap[word_idx] & bit);

        if (!excluded) {
            int32_t custkey = ord.o_custkey[i];
            cust_counts.add(custkey, 1);
        }
    }

    // Phase 4: 对于没有订单的客户，也需要统计
    // 所有客户都应该被计入，即使订单数为 0
    std::unordered_set<int32_t> cust_with_orders;
    for (size_t i = 0; i < ord.count; ++i) {
        size_t word_idx = i / 64;
        uint64_t bit = 1ULL << (i % 64);
        bool excluded = (word_idx < exclude_bitmap.size()) &&
                        (exclude_bitmap[word_idx] & bit);
        if (!excluded) {
            cust_with_orders.insert(ord.o_custkey[i]);
        }
    }

    // Phase 5: 统计 c_count 分布
    std::unordered_map<int32_t, int64_t> count_distribution;

    // 有订单的客户
    cust_counts.for_each_nonzero([&](int32_t custkey, int32_t order_count) {
        count_distribution[order_count]++;
    });

    // 没有订单的客户 (c_count = 0)
    int64_t cust_without_orders = static_cast<int64_t>(cust.count) -
                                   static_cast<int64_t>(cust_with_orders.size());
    if (cust_without_orders > 0) {
        count_distribution[0] += cust_without_orders;
    }

    // Phase 6: 排序结果 (按 custdist DESC, c_count DESC)
    std::vector<std::pair<int32_t, int64_t>> results(
        count_distribution.begin(), count_distribution.end());

    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first > b.first;
        });

    volatile size_t sink = results.size();
    (void)sink;
}

// ============================================================================
// Q21 V47 实现 - 并行基数排序
// ============================================================================

void run_q21_v47(TPCHDataLoader& loader, const Q21Config& config) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // Step 1: 找到目标国家的 nationkey
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == config.target_nation) {
            target_nationkey = nat.n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return;

    // Step 2: 找到目标国家的供应商
    std::unordered_set<int32_t> target_suppliers;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_nationkey[i] == target_nationkey) {
            target_suppliers.insert(supp.s_suppkey[i]);
        }
    }

    // Step 3: 找到状态为 'F' 的订单
    std::unordered_set<int32_t> f_orders;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderstatus[i] == 0) {  // 'F' = 0
            f_orders.insert(ord.o_orderkey[i]);
        }
    }

    // Step 4: 收集满足条件的 lineitem 记录
    // 条件: l_receiptdate > l_commitdate, orderkey in f_orders
    struct LineitemInfo {
        int32_t orderkey;
        int32_t suppkey;
        bool is_late;  // l_receiptdate > l_commitdate
    };

    std::vector<LineitemInfo> li_records;
    li_records.reserve(li.count / 4);

    for (size_t i = 0; i < li.count; ++i) {
        int32_t orderkey = li.l_orderkey[i];
        if (f_orders.find(orderkey) == f_orders.end()) continue;

        int32_t suppkey = li.l_suppkey[i];
        bool is_late = li.l_receiptdate[i] > li.l_commitdate[i];

        li_records.push_back({orderkey, suppkey, is_late});
    }

    // Step 5: 按 (orderkey, suppkey) 排序
    // 编码为 uint64_t: (orderkey << 32) | suppkey
    std::vector<uint64_t> encoded(li_records.size());
    for (size_t i = 0; i < li_records.size(); ++i) {
        encoded[i] = (static_cast<uint64_t>(static_cast<uint32_t>(li_records[i].orderkey)) << 32) |
                     static_cast<uint32_t>(li_records[i].suppkey);
    }

    // 使用并行基数排序获取排列
    ParallelRadixSort<uint64_t> sorter;
    sorter.configure({.radix_bits = 8, .thread_count = 8});
    auto perm = sorter.compute_permutation(encoded.data(), encoded.size());

    // Step 6: 按排列重新排序 li_records
    std::vector<LineitemInfo> sorted_records(li_records.size());
    for (size_t i = 0; i < li_records.size(); ++i) {
        sorted_records[i] = li_records[perm[i]];
    }

    // Step 7: 分析每个订单
    // 条件:
    // - 存在目标国家供应商的迟交记录
    // - 存在其他供应商的记录
    // - 不存在其他供应商的迟交记录

    std::unordered_map<int32_t, int64_t> supplier_counts;

    size_t i = 0;
    while (i < sorted_records.size()) {
        int32_t current_order = sorted_records[i].orderkey;

        // 收集当前订单的所有记录
        std::vector<std::pair<int32_t, bool>> order_records;  // (suppkey, is_late)
        while (i < sorted_records.size() && sorted_records[i].orderkey == current_order) {
            order_records.push_back({sorted_records[i].suppkey, sorted_records[i].is_late});
            i++;
        }

        // 去重并分析
        std::unordered_map<int32_t, bool> supp_late_map;  // suppkey -> has_late
        std::unordered_set<int32_t> supp_seen;

        for (const auto& [sk, late] : order_records) {
            if (supp_seen.find(sk) == supp_seen.end()) {
                supp_seen.insert(sk);
                supp_late_map[sk] = late;
            } else if (late) {
                supp_late_map[sk] = true;
            }
        }

        // 检查条件
        for (const auto& [sk, has_late] : supp_late_map) {
            // 跳过非目标国家供应商
            if (target_suppliers.find(sk) == target_suppliers.end()) continue;

            // 检查: 该供应商是迟交的
            if (!has_late) continue;

            // 检查: 存在其他供应商
            bool has_other_supplier = false;
            bool other_supplier_late = false;

            for (const auto& [other_sk, other_late] : supp_late_map) {
                if (other_sk != sk) {
                    has_other_supplier = true;
                    if (other_late) {
                        other_supplier_late = true;
                        break;
                    }
                }
            }

            // 条件: 存在其他供应商，但其他供应商没有迟交
            if (has_other_supplier && !other_supplier_late) {
                supplier_counts[sk]++;
            }
        }
    }

    // Step 8: 获取供应商名称并排序结果
    std::vector<std::pair<std::string, int64_t>> results;
    for (const auto& [suppkey, count] : supplier_counts) {
        // 找供应商名称
        for (size_t j = 0; j < supp.count; ++j) {
            if (supp.s_suppkey[j] == suppkey) {
                results.push_back({supp.s_name[j], count});
                break;
            }
        }
    }

    // 排序: count DESC, name ASC
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

    // 取前 100
    if (results.size() > 100) {
        results.resize(100);
    }

    volatile size_t sink = results.size();
    (void)sink;
}

}  // namespace ops_v47
}  // namespace tpch
}  // namespace thunderduck
