/**
 * ThunderDuck TPC-H V48 - 通用 Group-then-Filter 实现
 *
 * 核心算法: Group-then-Filter (非 JOIN/EXISTS)
 *
 * 通用算子:
 * - CountingSortGrouper: 计数排序分组 O(n+k)
 * - GenerationDeduplicator: 去重 O(1) per entity
 *
 * @version 48.0
 * @date 2026-01-29
 */

#include "tpch_operators_v48.h"
#include <algorithm>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v48 {

// ============================================================================
// Q21GenericOptimizer 实现 - 完全参数化
// ============================================================================

Q21GenericOptimizer::Result Q21GenericOptimizer::execute(
    const int32_t* s_suppkey,
    const int32_t* s_nationkey,
    const std::vector<std::string>& s_name,
    size_t supplier_count,
    const int32_t* l_orderkey,
    const int32_t* l_suppkey,
    const int32_t* l_commitdate,
    const int32_t* l_receiptdate,
    size_t lineitem_count,
    const int32_t* o_orderkey,
    const int8_t* o_orderstatus,
    size_t orders_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const Q21Config& config
) {
    Result result;

    // ========================================================================
    // Phase 1: 预计算位图 - 使用配置参数
    // ========================================================================

    // 1.1 找目标国家 nationkey (参数化)
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == config.target_nation) {  // 参数化
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // 1.2 目标国家供应商位图
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_suppkey[i] > max_suppkey) max_suppkey = s_suppkey[i];
    }

    std::vector<bool> is_target_supplier(max_suppkey + 1, false);
    std::vector<size_t> suppkey_to_idx(max_suppkey + 1, SIZE_MAX);

    for (size_t i = 0; i < supplier_count; ++i) {
        suppkey_to_idx[s_suppkey[i]] = i;
        if (s_nationkey[i] == target_nationkey) {
            is_target_supplier[s_suppkey[i]] = true;
        }
    }

    // 1.3 过滤订单位图 (参数化 status)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderkey[i] > max_orderkey) max_orderkey = o_orderkey[i];
    }

    std::vector<bool> is_filtered_order(max_orderkey + 1, false);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == config.order_status_filter) {  // 参数化
            is_filtered_order[o_orderkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 计数排序分组
    // ========================================================================

    struct CompactLineitem {
        int32_t suppkey;
        int8_t is_late;
    };

    std::vector<uint32_t> counts(max_orderkey + 2, 0);

    // 第一遍: 计数
    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok > 0 && ok <= max_orderkey && is_filtered_order[ok]) {
            counts[ok + 1]++;
        }
    }

    // 前缀和
    for (int32_t ok = 1; ok <= max_orderkey + 1; ++ok) {
        counts[ok] += counts[ok - 1];
    }

    // 第二遍: 存储数据
    const size_t total_li = counts[max_orderkey + 1];
    std::vector<CompactLineitem> sorted_data(total_li);
    std::vector<uint32_t> offsets = counts;

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok > 0 && ok <= max_orderkey && is_filtered_order[ok]) {
            sorted_data[offsets[ok]++] = {
                l_suppkey[i],
                static_cast<int8_t>(l_receiptdate[i] > l_commitdate[i] ? 1 : 0)
            };
        }
    }

    // ========================================================================
    // Phase 3: 收集过滤订单列表
    // ========================================================================

    std::vector<int32_t> filtered_orders;
    filtered_orders.reserve(orders_count / 2);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == config.order_status_filter) {  // 参数化
            filtered_orders.push_back(o_orderkey[i]);
        }
    }

    // ========================================================================
    // Phase 4: 聚合 - 使用 Generation Deduplicator
    // ========================================================================

    struct SuppState {
        uint32_t seen_gen = 0;
        uint32_t late_gen = 0;
    };
    std::vector<SuppState> supp_state(max_suppkey + 1);

    std::vector<int64_t> wait_counts(max_suppkey + 1, 0);
    uint32_t current_gen = 1;

    for (int32_t ok : filtered_orders) {
        if (ok <= 0 || ok > max_orderkey) continue;

        uint32_t start = counts[ok];
        uint32_t end = counts[ok + 1];
        if (start == end) continue;

        uint16_t supplier_count_val = 0;
        uint16_t late_count = 0;
        int32_t target_late_suppkey = -1;

        for (uint32_t idx = start; idx < end; ++idx) {
            const auto& li = sorted_data[idx];
            int32_t sk = li.suppkey;

            if (sk <= 0 || sk > max_suppkey) continue;

            auto& state = supp_state[sk];

            if (state.seen_gen != current_gen) {
                state.seen_gen = current_gen;
                supplier_count_val++;
            }

            if (li.is_late && state.late_gen != current_gen) {
                state.late_gen = current_gen;
                late_count++;
                if (is_target_supplier[sk]) {
                    target_late_suppkey = sk;
                }
            }
        }

        // EXACT-K predicate (参数化)
        if (supplier_count_val >= config.min_supplier_count &&  // 参数化
            late_count == config.exact_late_count &&            // 参数化
            target_late_suppkey > 0) {
            wait_counts[target_late_suppkey]++;
        }

        ++current_gen;
    }

    // ========================================================================
    // Phase 5: Top-K (参数化 limit)
    // ========================================================================

    std::vector<std::pair<int64_t, int32_t>> sorted_results;
    for (int32_t sk = 1; sk <= max_suppkey; ++sk) {
        if (wait_counts[sk] > 0) {
            sorted_results.emplace_back(wait_counts[sk], sk);
        }
    }

    std::sort(sorted_results.begin(), sorted_results.end(),
        [&](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            size_t ia = suppkey_to_idx[a.second];
            size_t ib = suppkey_to_idx[b.second];
            return s_name[ia] < s_name[ib];
        });

    size_t result_count = std::min(config.limit, sorted_results.size());  // 参数化
    result.suppliers.reserve(result_count);

    for (size_t j = 0; j < result_count; ++j) {
        int32_t sk = sorted_results[j].second;
        int64_t cnt = sorted_results[j].first;
        size_t sidx = suppkey_to_idx[sk];
        result.suppliers.emplace_back(s_name[sidx], cnt);
    }

    return result;
}

// ============================================================================
// 向后兼容接口
// ============================================================================

Q21CorrectOptimizer::Result Q21CorrectOptimizer::execute(
    const int32_t* s_suppkey,
    const int32_t* s_nationkey,
    const std::vector<std::string>& s_name,
    size_t supplier_count,
    const int32_t* l_orderkey,
    const int32_t* l_suppkey,
    const int32_t* l_commitdate,
    const int32_t* l_receiptdate,
    size_t lineitem_count,
    const int32_t* o_orderkey,
    const int8_t* o_orderstatus,
    size_t orders_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const std::string& target_nation,
    size_t limit
) {
    Q21Config config;
    config.target_nation = target_nation;
    config.limit = limit;

    auto generic_result = Q21GenericOptimizer::execute(
        s_suppkey, s_nationkey, s_name, supplier_count,
        l_orderkey, l_suppkey, l_commitdate, l_receiptdate, lineitem_count,
        o_orderkey, o_orderstatus, orders_count,
        n_nationkey, n_name, nation_count,
        config
    );

    Result result;
    result.suppliers = std::move(generic_result.suppliers);
    return result;
}

// ============================================================================
// 查询入口
// ============================================================================

void run_q21_v48(TPCHDataLoader& loader, const Q21Config& config) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    auto result = Q21GenericOptimizer::execute(
        supp.s_suppkey.data(),
        supp.s_nationkey.data(),
        supp.s_name,
        supp.count,
        li.l_orderkey.data(),
        li.l_suppkey.data(),
        li.l_commitdate.data(),
        li.l_receiptdate.data(),
        li.count,
        ord.o_orderkey.data(),
        ord.o_orderstatus.data(),
        ord.count,
        nat.n_nationkey.data(),
        nat.n_name,
        nat.count,
        config
    );

    volatile size_t count = result.suppliers.size();
    (void)count;
}

void run_q21_v48(TPCHDataLoader& loader) {
    run_q21_v48(loader, Q21Config{});  // 使用默认配置
}

} // namespace ops_v48
} // namespace tpch
} // namespace thunderduck
