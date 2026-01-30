/**
 * ThunderDuck TPC-H V38 优化算子实现
 *
 * Q21: 排序去重替代嵌套 Hash Set
 * Q20: 紧凑编码 + Bitmap 预过滤
 */

#include "tpch_operators_v38.h"
#include "tpch_constants.h"      // 统一常量定义
#include <algorithm>
#include <cstring>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v38 {

// ============================================================================
// Q21 优化实现 V2 - 排序去重
// ============================================================================

Q21OptimizerV2::Result Q21OptimizerV2::execute(
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
    // Phase 1: 预计算过滤条件
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

    // 1.2 构建目标供应商 Bitmap (suppkey 范围 1-10000)
    std::vector<bool> is_target_supplier(10001, false);
    std::vector<size_t> suppkey_to_idx(10001, SIZE_MAX);
    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_nationkey[i] == target_nationkey) {
            int32_t sk = s_suppkey[i];
            if (sk > 0 && sk <= 10000) {
                is_target_supplier[sk] = true;
                suppkey_to_idx[sk] = i;
            }
        }
    }

    // 1.3 构建失败订单 Bitmap (orderkey 范围需要检查)
    // 找 orderkey 最大值
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderkey[i] > max_orderkey) max_orderkey = o_orderkey[i];
    }

    std::vector<bool> is_failed_order(max_orderkey + 1, false);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == 0) {  // F=0
            is_failed_order[o_orderkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 收集 (orderkey, suppkey, is_late) 三元组
    // ========================================================================
    struct LineitemInfo {
        int32_t orderkey;
        int32_t suppkey;
        bool is_late;

        bool operator<(const LineitemInfo& other) const {
            if (orderkey != other.orderkey) return orderkey < other.orderkey;
            return suppkey < other.suppkey;
        }
        bool operator==(const LineitemInfo& other) const {
            return orderkey == other.orderkey && suppkey == other.suppkey;
        }
    };

    std::vector<LineitemInfo> infos;
    infos.reserve(lineitem_count / 2);  // 估计一半是 status='F'

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok <= 0 || ok > max_orderkey || !is_failed_order[ok]) continue;

        infos.push_back({
            ok,
            l_suppkey[i],
            l_receiptdate[i] > l_commitdate[i]
        });
    }

    // ========================================================================
    // Phase 3: 排序并去重，合并 is_late 标志
    // ========================================================================
    std::sort(infos.begin(), infos.end());

    // 去重并合并 is_late (如果任一行是 late，则该 pair 是 late)
    std::vector<LineitemInfo> unique_pairs;
    unique_pairs.reserve(infos.size() / 4);

    for (size_t i = 0; i < infos.size(); ) {
        int32_t ok = infos[i].orderkey;
        int32_t sk = infos[i].suppkey;
        bool is_late = infos[i].is_late;

        // 合并相同 (orderkey, suppkey) 的所有行
        size_t j = i + 1;
        while (j < infos.size() && infos[j].orderkey == ok && infos[j].suppkey == sk) {
            is_late = is_late || infos[j].is_late;
            ++j;
        }

        unique_pairs.push_back({ok, sk, is_late});
        i = j;
    }

    // ========================================================================
    // Phase 4: 按 orderkey 分组，计算每个订单的统计信息
    // ========================================================================
    struct OrderStats {
        uint16_t total_suppliers;
        uint16_t late_suppliers;
        // 如果只有1个延迟供应商，记录它
        int32_t single_late_suppkey;
    };

    std::unordered_map<int32_t, OrderStats> order_stats;
    order_stats.reserve(unique_pairs.size() / 3);

    for (size_t i = 0; i < unique_pairs.size(); ) {
        int32_t ok = unique_pairs[i].orderkey;
        OrderStats stats{0, 0, -1};

        // 统计该订单的所有供应商
        size_t j = i;
        while (j < unique_pairs.size() && unique_pairs[j].orderkey == ok) {
            stats.total_suppliers++;
            if (unique_pairs[j].is_late) {
                if (stats.late_suppliers == 0) {
                    stats.single_late_suppkey = unique_pairs[j].suppkey;
                }
                stats.late_suppliers++;
            }
            ++j;
        }

        order_stats[ok] = stats;
        i = j;
    }

    // ========================================================================
    // Phase 5: 评估条件并聚合
    // ========================================================================
    // 条件:
    // - 该供应商是目标国家
    // - 该供应商在该订单上延迟 (is_late)
    // - EXISTS: total_suppliers > 1
    // - NOT EXISTS: late_suppliers == 1 AND single_late_suppkey == 该供应商

    std::unordered_map<int32_t, int64_t> supplier_waits;

    for (const auto& pair : unique_pairs) {
        // 检查是否是目标供应商
        if (pair.suppkey <= 0 || pair.suppkey > 10000) continue;
        if (!is_target_supplier[pair.suppkey]) continue;

        // 检查该供应商是否延迟
        if (!pair.is_late) continue;

        // 获取订单统计
        auto it = order_stats.find(pair.orderkey);
        if (it == order_stats.end()) continue;
        const auto& stats = it->second;

        // EXISTS: 订单有其他供应商
        if (stats.total_suppliers <= 1) continue;

        // NOT EXISTS: 没有其他延迟供应商
        // (只有当前供应商延迟，或没有人延迟)
        if (stats.late_suppliers > 1) continue;
        if (stats.late_suppliers == 1 && stats.single_late_suppkey != pair.suppkey) continue;

        // 满足所有条件
        supplier_waits[pair.suppkey]++;
    }

    // ========================================================================
    // Phase 6: 排序并返回 Top-K
    // ========================================================================
    std::vector<std::pair<int64_t, int32_t>> sorted_results;
    sorted_results.reserve(supplier_waits.size());

    for (const auto& [sk, count] : supplier_waits) {
        sorted_results.emplace_back(count, sk);
    }

    // ORDER BY numwait DESC, s_name ASC
    std::sort(sorted_results.begin(), sorted_results.end(),
        [&](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            size_t ia = suppkey_to_idx[a.second];
            size_t ib = suppkey_to_idx[b.second];
            return s_name[ia] < s_name[ib];
        });

    // 取 Top-K
    size_t result_count = std::min(limit, sorted_results.size());
    result.suppliers.reserve(result_count);

    for (size_t i = 0; i < result_count; ++i) {
        int32_t sk = sorted_results[i].second;
        int64_t cnt = sorted_results[i].first;
        size_t sidx = suppkey_to_idx[sk];
        result.suppliers.emplace_back(s_name[sidx], cnt);
    }

    return result;
}

// ============================================================================
// Q20 优化实现 V3 - 紧凑编码
// ============================================================================

Q20OptimizerV3::Result Q20OptimizerV3::execute(
    const int32_t* s_suppkey,
    const int32_t* s_nationkey,
    const std::vector<std::string>& s_name,
    const std::vector<std::string>& s_address,
    size_t supplier_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const int32_t* p_partkey,
    const std::vector<std::string>& p_name,
    size_t part_count,
    const int32_t* ps_partkey,
    const int32_t* ps_suppkey,
    const int32_t* ps_availqty,
    size_t partsupp_count,
    const int32_t* l_partkey,
    const int32_t* l_suppkey,
    const int64_t* l_quantity,
    const int32_t* l_shipdate,
    size_t lineitem_count,
    const std::string& part_prefix,
    const std::string& target_nation,
    int32_t date_lo,
    int32_t date_hi,
    double quantity_factor
) {
    Result result;

    // ========================================================================
    // Phase 1: 预计算过滤条件
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

    // 1.2 找 "forest%" 开头的 part - 使用 Bitmap
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part_count; ++i) {
        if (p_partkey[i] > max_partkey) max_partkey = p_partkey[i];
    }

    std::vector<bool> is_forest_part(max_partkey + 1, false);
    size_t forest_count = 0;
    for (size_t i = 0; i < part_count; ++i) {
        if (p_name[i].compare(0, part_prefix.size(), part_prefix) == 0) {
            is_forest_part[p_partkey[i]] = true;
            forest_count++;
        }
    }

    if (forest_count == 0) return result;

    // ========================================================================
    // Phase 2: 扫描 lineitem，计算 SUM(l_quantity) per (partkey, suppkey)
    // 使用紧凑编码: key = (partkey << 14) | suppkey
    // ========================================================================
    std::unordered_map<uint32_t, int64_t> qty_sum;
    qty_sum.reserve(partsupp_count / 10);

    for (size_t i = 0; i < lineitem_count; ++i) {
        // 日期过滤
        if (l_shipdate[i] < date_lo || l_shipdate[i] >= date_hi) continue;

        // Part 过滤
        int32_t pk = l_partkey[i];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        // 累加
        uint32_t key = encode_key(pk, l_suppkey[i]);
        qty_sum[key] += l_quantity[i];
    }

    // ========================================================================
    // Phase 3: 扫描 partsupp，找满足条件的 suppkey
    // ========================================================================
    std::unordered_set<int32_t> valid_suppkeys;

    for (size_t i = 0; i < partsupp_count; ++i) {
        int32_t pk = ps_partkey[i];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        uint32_t key = encode_key(pk, ps_suppkey[i]);
        auto it = qty_sum.find(key);
        if (it != qty_sum.end()) {
            int64_t threshold = static_cast<int64_t>(it->second * quantity_factor);
            if (ps_availqty[i] > threshold) {
                valid_suppkeys.insert(ps_suppkey[i]);
            }
        }
    }

    // ========================================================================
    // Phase 4: 过滤 supplier (国家 + IN valid_suppkeys)
    // ========================================================================
    std::vector<std::pair<std::string, std::string>> suppliers;

    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_nationkey[i] == target_nationkey && valid_suppkeys.count(s_suppkey[i])) {
            suppliers.emplace_back(s_name[i], s_address[i]);
        }
    }

    // 排序
    std::sort(suppliers.begin(), suppliers.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    result.suppliers = std::move(suppliers);
    return result;
}

// ============================================================================
// V38 查询入口
// ============================================================================

void run_q21_v38(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    auto result = Q21OptimizerV2::execute(
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
        nations::SAUDI_ARABIA,
        100
    );

    // 结果用于验证
    volatile size_t count = result.suppliers.size();
    (void)count;
}

void run_q20_v38(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& part = loader.part();
    const auto& ps = loader.partsupp();
    const auto& li = loader.lineitem();

    constexpr int32_t DATE_1994_01_01 = dates::D1994_01_01;
    constexpr int32_t DATE_1995_01_01 = dates::D1995_01_01;

    auto result = Q20OptimizerV3::execute(
        supp.s_suppkey.data(),
        supp.s_nationkey.data(),
        supp.s_name,
        supp.s_address,
        supp.count,
        nat.n_nationkey.data(),
        nat.n_name,
        nat.count,
        part.p_partkey.data(),
        part.p_name,
        part.count,
        ps.ps_partkey.data(),
        ps.ps_suppkey.data(),
        ps.ps_availqty.data(),
        ps.count,
        li.l_partkey.data(),
        li.l_suppkey.data(),
        li.l_quantity.data(),
        li.l_shipdate.data(),
        li.count,
        query_params::q20::COLOR_PREFIX,
        nations::CANADA,
        dates::D1994_01_01,
        dates::D1995_01_01,
        0.5
    );

    // 结果用于验证
    volatile size_t count = result.suppliers.size();
    (void)count;
}

} // namespace ops_v38
} // namespace tpch
} // namespace thunderduck
