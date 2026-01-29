/**
 * ThunderDuck TPC-H V39 优化算子实现
 *
 * 纯排序方案，完全避免 Hash Map 的开销
 */

#include "tpch_operators_v39.h"
#include <algorithm>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v39 {

// ============================================================================
// Q21 优化实现 V3 - 纯排序方案
// ============================================================================

Q21OptimizerV3::Result Q21OptimizerV3::execute(
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
    // Phase 1: 预计算 - 使用 Bitmap
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

    // 1.2 目标供应商 Bitmap (suppkey 1-10000)
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

    // 1.3 失败订单 Bitmap
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderkey[i] > max_orderkey) max_orderkey = o_orderkey[i];
    }

    std::vector<bool> is_failed_order(max_orderkey + 1, false);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == 0) {
            is_failed_order[o_orderkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 收集 (orderkey, suppkey, is_late) 并排序
    // ========================================================================
    struct Record {
        int32_t orderkey;
        int32_t suppkey;
        uint8_t is_late;  // 0 or 1

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

    // 排序
    std::sort(records.begin(), records.end());

    // ========================================================================
    // Phase 3: 单遍扫描 - 去重并计算每个订单的统计
    // ========================================================================
    // 存储去重后的 (orderkey, suppkey, is_late)
    // 以及每个 orderkey 的 (supplier_count, late_count, single_late_suppkey)

    struct OrderStats {
        int32_t orderkey;
        uint16_t supplier_count;
        uint16_t late_count;
        int32_t single_late_suppkey;  // 仅当 late_count == 1 时有效
    };

    std::vector<OrderStats> order_stats;
    order_stats.reserve(records.size() / 4);

    // 同时存储去重后的记录用于后续评估
    struct UniqueRecord {
        int32_t orderkey;
        int32_t suppkey;
        bool is_late;
    };
    std::vector<UniqueRecord> unique_records;
    unique_records.reserve(records.size() / 2);

    size_t i = 0;
    while (i < records.size()) {
        int32_t current_order = records[i].orderkey;
        OrderStats stats{current_order, 0, 0, -1};

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

            // 记录去重后的结果
            unique_records.push_back({current_order, current_supp, is_late});
            stats.supplier_count++;
            if (is_late) {
                if (stats.late_count == 0) {
                    stats.single_late_suppkey = current_supp;
                }
                stats.late_count++;
            }
        }

        order_stats.push_back(stats);
    }

    // ========================================================================
    // Phase 4: 构建 orderkey -> stats 的快速查找
    // ========================================================================
    // 由于 order_stats 已按 orderkey 排序，可以用二分查找
    auto find_stats = [&order_stats](int32_t orderkey) -> const OrderStats* {
        auto it = std::lower_bound(order_stats.begin(), order_stats.end(), orderkey,
            [](const OrderStats& s, int32_t k) { return s.orderkey < k; });
        if (it != order_stats.end() && it->orderkey == orderkey) {
            return &(*it);
        }
        return nullptr;
    };

    // ========================================================================
    // Phase 5: 评估条件并聚合 (使用数组计数)
    // ========================================================================
    std::vector<int64_t> wait_counts(10001, 0);  // suppkey -> count

    for (const auto& rec : unique_records) {
        // 检查是否是目标供应商
        if (rec.suppkey <= 0 || rec.suppkey > 10000) continue;
        if (!is_target_supplier[rec.suppkey]) continue;

        // 检查该供应商是否延迟
        if (!rec.is_late) continue;

        // 获取订单统计
        const OrderStats* stats = find_stats(rec.orderkey);
        if (!stats) continue;

        // EXISTS: 订单有多个供应商
        if (stats->supplier_count <= 1) continue;

        // NOT EXISTS: 没有其他延迟供应商
        if (stats->late_count > 1) continue;
        if (stats->late_count == 1 && stats->single_late_suppkey != rec.suppkey) continue;

        // 满足条件
        wait_counts[rec.suppkey]++;
    }

    // ========================================================================
    // Phase 6: 排序并返回 Top-K
    // ========================================================================
    std::vector<std::pair<int64_t, int32_t>> sorted_results;
    for (int32_t sk = 1; sk <= 10000; ++sk) {
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
// Q20 优化实现 V4 - 纯排序方案
// ============================================================================

Q20OptimizerV4::Result Q20OptimizerV4::execute(
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
    // Phase 1: 预计算
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

    // 1.2 forest% parts Bitmap
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part_count; ++i) {
        if (p_partkey[i] > max_partkey) max_partkey = p_partkey[i];
    }

    std::vector<bool> is_forest_part(max_partkey + 1, false);
    for (size_t i = 0; i < part_count; ++i) {
        if (p_name[i].compare(0, part_prefix.size(), part_prefix) == 0) {
            is_forest_part[p_partkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 收集 lineitem (partkey, suppkey, quantity) 并排序
    // ========================================================================
    struct LIRecord {
        int32_t partkey;
        int32_t suppkey;
        int64_t quantity;

        bool operator<(const LIRecord& o) const {
            if (partkey != o.partkey) return partkey < o.partkey;
            return suppkey < o.suppkey;
        }
    };

    std::vector<LIRecord> li_records;
    li_records.reserve(lineitem_count / 10);

    for (size_t i = 0; i < lineitem_count; ++i) {
        if (l_shipdate[i] < date_lo || l_shipdate[i] >= date_hi) continue;

        int32_t pk = l_partkey[i];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        li_records.push_back({pk, l_suppkey[i], l_quantity[i]});
    }

    std::sort(li_records.begin(), li_records.end());

    // ========================================================================
    // Phase 3: 聚合 SUM(quantity) per (partkey, suppkey)
    // ========================================================================
    struct PSSum {
        int32_t partkey;
        int32_t suppkey;
        int64_t sum_qty;
    };

    std::vector<PSSum> qty_sums;
    qty_sums.reserve(li_records.size() / 4);

    size_t i = 0;
    while (i < li_records.size()) {
        int32_t pk = li_records[i].partkey;
        int32_t sk = li_records[i].suppkey;
        int64_t sum = 0;

        while (i < li_records.size() &&
               li_records[i].partkey == pk &&
               li_records[i].suppkey == sk) {
            sum += li_records[i].quantity;
            ++i;
        }

        qty_sums.push_back({pk, sk, sum});
    }

    // ========================================================================
    // Phase 4: 收集并排序 partsupp
    // ========================================================================
    struct PSRecord {
        int32_t partkey;
        int32_t suppkey;
        int32_t availqty;

        bool operator<(const PSRecord& o) const {
            if (partkey != o.partkey) return partkey < o.partkey;
            return suppkey < o.suppkey;
        }
    };

    std::vector<PSRecord> ps_records;
    ps_records.reserve(partsupp_count / 100);

    for (size_t j = 0; j < partsupp_count; ++j) {
        int32_t pk = ps_partkey[j];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        ps_records.push_back({pk, ps_suppkey[j], ps_availqty[j]});
    }

    std::sort(ps_records.begin(), ps_records.end());

    // ========================================================================
    // Phase 5: 归并比较 - 找满足条件的 suppkey
    // ========================================================================
    std::vector<bool> valid_suppkey(10001, false);

    size_t qi = 0, pi = 0;
    while (qi < qty_sums.size() && pi < ps_records.size()) {
        const auto& q = qty_sums[qi];
        const auto& p = ps_records[pi];

        if (q.partkey < p.partkey || (q.partkey == p.partkey && q.suppkey < p.suppkey)) {
            ++qi;
        } else if (q.partkey > p.partkey || (q.partkey == p.partkey && q.suppkey > p.suppkey)) {
            ++pi;
        } else {
            // 匹配
            int64_t threshold = static_cast<int64_t>(q.sum_qty * quantity_factor);
            if (p.availqty > threshold) {
                if (p.suppkey > 0 && p.suppkey <= 10000) {
                    valid_suppkey[p.suppkey] = true;
                }
            }
            ++qi;
            ++pi;
        }
    }

    // ========================================================================
    // Phase 6: 过滤 supplier
    // ========================================================================
    std::vector<std::pair<std::string, std::string>> suppliers;

    for (size_t j = 0; j < supplier_count; ++j) {
        int32_t sk = s_suppkey[j];
        if (s_nationkey[j] == target_nationkey &&
            sk > 0 && sk <= 10000 && valid_suppkey[sk]) {
            suppliers.emplace_back(s_name[j], s_address[j]);
        }
    }

    std::sort(suppliers.begin(), suppliers.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    result.suppliers = std::move(suppliers);
    return result;
}

// ============================================================================
// V39 查询入口
// ============================================================================

void run_q21_v39(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    auto result = Q21OptimizerV3::execute(
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

void run_q20_v39(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& part = loader.part();
    const auto& ps = loader.partsupp();
    const auto& li = loader.lineitem();

    constexpr int32_t DATE_1994_01_01 = 8766;
    constexpr int32_t DATE_1995_01_01 = 9131;

    auto result = Q20OptimizerV4::execute(
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
        "forest",
        "CANADA",
        DATE_1994_01_01,
        DATE_1995_01_01,
        0.5
    );

    volatile size_t count = result.suppliers.size();
    (void)count;
}

} // namespace ops_v39
} // namespace tpch
} // namespace thunderduck
