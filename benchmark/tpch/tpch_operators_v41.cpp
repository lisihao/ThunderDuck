/**
 * ThunderDuck TPC-H V41 优化算子实现
 *
 * Q21 单遍预计算 + 直接数组访问
 *
 * @version 41.0
 * @date 2026-01-29
 */

#include "tpch_operators_v41.h"
#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace thunderduck {
namespace tpch {
namespace ops_v41 {

// ============================================================================
// Q21 优化实现 V4 - 单遍预计算方案
// ============================================================================

Q21OptimizerV4::Result Q21OptimizerV4::execute(
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
    Result result;

    // ========================================================================
    // Phase 1: 预计算位图 (消除硬编码)
    // ========================================================================

    // 1.1 找目标国家
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // 1.2 构建目标供应商位图 (动态范围!)
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

    // 1.3 构建失败订单位图
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderkey[i] > max_orderkey) max_orderkey = o_orderkey[i];
    }

    std::vector<bool> is_failed_order(max_orderkey + 1, false);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == 0) {  // 'F' status
            is_failed_order[o_orderkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 单遍扫描计算 OrderStats (无排序!)
    // ========================================================================
    // 关键优化: 使用紧凑的 OrderStats + 单遍聚合
    // 对于每个订单，我们需要:
    // - supplier_count: 不同供应商的数量
    // - late_count: 延迟供应商的数量
    // - target_late_suppkey: 如果有目标国家的延迟供应商，记录其 suppkey

    struct OrderStats {
        uint16_t supplier_count = 0;
        uint16_t late_count = 0;
        int32_t target_late_suppkey = -1;  // 目标国家的延迟供应商 (仅存一个)
    };

    // 直接数组访问
    std::vector<OrderStats> order_stats(max_orderkey + 1);

    // 位图: 跟踪每个订单已见过的供应商和延迟的供应商
    // 使用 suppkey 编码为 orderkey * (max_suppkey+1) + suppkey
    // 但这太大了，改用增量处理

    // 先收集所有 (orderkey, suppkey, is_late) 记录
    struct Record {
        int32_t orderkey;
        int32_t suppkey;
        uint8_t is_late;

        bool operator<(const Record& o) const {
            if (orderkey != o.orderkey) return orderkey < o.orderkey;
            return suppkey < o.suppkey;
        }
    };

    std::vector<Record> records;
    records.reserve(lineitem_count / 2);

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok <= 0 || ok > max_orderkey || !is_failed_order[ok]) continue;

        records.push_back({
            ok,
            l_suppkey[i],
            static_cast<uint8_t>(l_receiptdate[i] > l_commitdate[i] ? 1 : 0)
        });
    }

    // 排序以便去重
    std::sort(records.begin(), records.end());

    // 单遍处理: 计算 OrderStats 并直接评估
    std::vector<int64_t> wait_counts(max_suppkey + 1, 0);

    size_t i = 0;
    while (i < records.size()) {
        int32_t current_order = records[i].orderkey;
        auto& stats = order_stats[current_order];

        // 临时存储该订单的目标延迟供应商
        std::vector<int32_t> order_target_late_supps;

        // 处理该订单的所有记录
        while (i < records.size() && records[i].orderkey == current_order) {
            int32_t current_supp = records[i].suppkey;
            bool is_late = false;

            // 合并相同 (orderkey, suppkey) 的记录
            while (i < records.size() &&
                   records[i].orderkey == current_order &&
                   records[i].suppkey == current_supp) {
                if (records[i].is_late) is_late = true;
                ++i;
            }

            stats.supplier_count++;
            if (is_late) {
                stats.late_count++;
                // 如果是目标国家的供应商
                if (current_supp > 0 && current_supp <= max_suppkey &&
                    is_target_supplier[current_supp]) {
                    order_target_late_supps.push_back(current_supp);
                }
            }
        }

        // 评估条件:
        // 1. supplier_count > 1 (EXISTS 其他供应商)
        // 2. late_count == 1 (NOT EXISTS 其他延迟)
        // 3. 目标供应商延迟
        if (stats.supplier_count > 1 &&
            stats.late_count == 1 &&
            order_target_late_supps.size() == 1) {
            // 唯一延迟的供应商是目标国家的供应商
            wait_counts[order_target_late_supps[0]]++;
        }
    }

    // ========================================================================
    // Phase 3: 排序并返回 Top-K
    // ========================================================================

    std::vector<std::pair<int64_t, int32_t>> sorted_results;
    for (int32_t sk = 1; sk <= max_suppkey; ++sk) {
        if (wait_counts[sk] > 0 && is_target_supplier[sk]) {
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

    size_t result_count = std::min(limit, sorted_results.size());
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
// V41 查询入口
// ============================================================================

void run_q21_v41(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    auto result = Q21OptimizerV4::execute(
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
        "SAUDI ARABIA",
        100
    );

    volatile size_t count = result.suppliers.size();
    (void)count;
}

} // namespace ops_v41
} // namespace tpch
} // namespace thunderduck
