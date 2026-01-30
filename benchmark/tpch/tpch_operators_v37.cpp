/**
 * ThunderDuck TPC-H V37 优化算子实现
 */

#include "tpch_operators_v37.h"
#include "tpch_constants.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <atomic>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v37 {

// ============================================================================
// Q22 优化实现 - Bitmap Anti-Join
// ============================================================================

Q22Optimizer::Result Q22Optimizer::execute(
    const int32_t* c_custkey,
    const int64_t* c_acctbal,
    const std::vector<std::string>& c_phone,
    size_t customer_count,
    const int32_t* o_custkey,
    size_t orders_count,
    const std::vector<std::string>& country_codes,
    int32_t substring_start,
    int32_t substring_length
) {
    Result result;

    // 初始化国家码
    for (size_t i = 0; i < country_codes.size() && i < 7; ++i) {
        result.codes[i] = country_codes[i];
    }

    // ========================================================================
    // Phase 1: 构建国家码快速查找表 (直接数组 0-99)
    // ========================================================================
    std::array<bool, 100> code_valid{};
    std::array<int8_t, 100> code_index{};

    int8_t idx = 0;
    for (const auto& code : country_codes) {
        if (code.size() >= 2) {
            int32_t val = (code[0] - '0') * 10 + (code[1] - '0');
            if (val < 100) {
                code_valid[val] = true;
                code_index[val] = idx++;
            }
        }
    }

    // ========================================================================
    // Phase 2: 构建 Bitmap Anti-Join (客户是否有订单)
    // ========================================================================
    // 找 custkey 范围
    int32_t min_ck = c_custkey[0], max_ck = c_custkey[0];
    for (size_t i = 1; i < customer_count; ++i) {
        if (c_custkey[i] < min_ck) min_ck = c_custkey[i];
        if (c_custkey[i] > max_ck) max_ck = c_custkey[i];
    }

    BitmapExistenceSet has_orders;
    has_orders.build(o_custkey, orders_count, min_ck, max_ck);

    // ========================================================================
    // Phase 3: 单遍扫描 - 计算 AVG + 过滤 + 聚合
    // ========================================================================
    // 先计算 AVG (只包含目标国家码 + c_acctbal > 0)
    int64_t sum_acctbal = 0;
    size_t count_acctbal = 0;

    size_t start_idx = substring_start > 0 ? substring_start - 1 : 0;

    // 第一遍: 计算 AVG
    for (size_t i = 0; i < customer_count; ++i) {
        const auto& phone = c_phone[i];
        if (phone.size() < start_idx + 2) continue;

        int32_t code = (phone[start_idx] - '0') * 10 + (phone[start_idx + 1] - '0');
        if (code >= 100 || !code_valid[code]) continue;

        if (c_acctbal[i] > 0) {
            sum_acctbal += c_acctbal[i];
            ++count_acctbal;
        }
    }

    int64_t avg_threshold = count_acctbal > 0 ?
        sum_acctbal / static_cast<int64_t>(count_acctbal) : 0;

    // 第二遍: 过滤 + 聚合
    for (size_t i = 0; i < customer_count; ++i) {
        const auto& phone = c_phone[i];
        if (phone.size() < start_idx + 2) continue;

        int32_t code = (phone[start_idx] - '0') * 10 + (phone[start_idx + 1] - '0');
        if (code >= 100 || !code_valid[code]) continue;

        // 条件: c_acctbal > avg AND NOT EXISTS (orders)
        if (c_acctbal[i] <= avg_threshold) continue;
        if (has_orders.exists(c_custkey[i])) continue;

        // 满足条件，聚合
        int8_t bucket = code_index[code];
        result.counts[bucket]++;
        result.sums[bucket] += c_acctbal[i];
    }

    return result;
}

// ============================================================================
// Q21 优化实现 - 预计算 OrderKeyState
// ============================================================================

Q21Optimizer::Result Q21Optimizer::execute(
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
    const int8_t* o_orderstatus,  // F=0, O=1, P=2
    size_t orders_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const std::string& target_nation,
    size_t limit
) {
    Result result;

    // ========================================================================
    // Phase 1: 找目标国家
    // ========================================================================
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // 构建目标国家的供应商集合
    std::unordered_set<int32_t> target_suppliers;
    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_nationkey[i] == target_nationkey) {
            target_suppliers.insert(s_suppkey[i]);
        }
    }

    // ========================================================================
    // Phase 2: 构建 orderstatus = 'F' 的订单集合
    // ========================================================================
    std::unordered_set<int32_t> failed_orders;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == 0) {  // F=0
            failed_orders.insert(o_orderkey[i]);
        }
    }

    // ========================================================================
    // Phase 3: 预计算 OrderKeyState
    // ========================================================================
    // 对于每个 orderkey，统计:
    // - 有多少不同的供应商
    // - 有多少供应商延迟交付 (receiptdate > commitdate)
    // 使用两层结构: orderkey -> (suppkey -> is_late)

    struct OrderInfo {
        std::unordered_set<int32_t> all_suppliers;       // 所有供应商
        std::unordered_set<int32_t> late_suppliers;      // 延迟的供应商
    };
    std::unordered_map<int32_t, OrderInfo> order_info;
    order_info.reserve(orders_count / 4);

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (!failed_orders.count(ok)) continue;  // 只处理 status='F' 的订单

        auto& info = order_info[ok];
        info.all_suppliers.insert(l_suppkey[i]);

        if (l_receiptdate[i] > l_commitdate[i]) {
            info.late_suppliers.insert(l_suppkey[i]);
        }
    }

    // ========================================================================
    // Phase 4: 评估条件并聚合
    // ========================================================================
    // 对于目标供应商的每个 lineitem:
    // - l_receiptdate > l_commitdate (该供应商延迟)
    // - EXISTS: 订单有其他供应商
    // - NOT EXISTS: 订单没有其他延迟供应商

    std::unordered_map<int32_t, int64_t> supplier_waits;

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t sk = l_suppkey[i];
        if (!target_suppliers.count(sk)) continue;  // 不是目标国家供应商

        int32_t ok = l_orderkey[i];
        if (!failed_orders.count(ok)) continue;  // 订单状态不是 'F'

        if (l_receiptdate[i] <= l_commitdate[i]) continue;  // 该供应商没有延迟

        auto it = order_info.find(ok);
        if (it == order_info.end()) continue;

        const auto& info = it->second;

        // EXISTS: 订单有其他供应商
        if (info.all_suppliers.size() <= 1) continue;

        // NOT EXISTS: 没有其他延迟供应商
        // (late_suppliers 只包含当前供应商，或者不包含其他供应商)
        bool has_other_late = false;
        for (int32_t late_sk : info.late_suppliers) {
            if (late_sk != sk) {
                has_other_late = true;
                break;
            }
        }
        if (has_other_late) continue;

        // 满足所有条件
        supplier_waits[sk]++;
    }

    // ========================================================================
    // Phase 5: 排序并返回 Top-K
    // ========================================================================
    std::vector<std::pair<int64_t, int32_t>> sorted_results;  // (count, suppkey)
    sorted_results.reserve(supplier_waits.size());

    for (const auto& [sk, count] : supplier_waits) {
        sorted_results.emplace_back(count, sk);
    }

    // ORDER BY numwait DESC, s_name ASC
    // 先按 count 降序，再按 s_name 升序
    // 需要查找 s_name
    std::unordered_map<int32_t, size_t> suppkey_to_idx;
    for (size_t i = 0; i < supplier_count; ++i) {
        suppkey_to_idx[s_suppkey[i]] = i;
    }

    std::sort(sorted_results.begin(), sorted_results.end(),
        [&](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;  // count DESC
            size_t ia = suppkey_to_idx[a.second];
            size_t ib = suppkey_to_idx[b.second];
            return s_name[ia] < s_name[ib];  // name ASC
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
// Q20 优化实现 V2
// ============================================================================

Q20OptimizerV2::Result Q20OptimizerV2::execute(
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
    // Phase 1: 找目标国家
    // ========================================================================
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // ========================================================================
    // Phase 2: 找 "forest%" 开头的 part (构建 Bitmap)
    // ========================================================================
    std::unordered_set<int32_t> forest_parts;
    forest_parts.reserve(part_count / 100);  // 估计 1%

    for (size_t i = 0; i < part_count; ++i) {
        if (p_name[i].compare(0, part_prefix.size(), part_prefix) == 0) {
            forest_parts.insert(p_partkey[i]);
        }
    }

    if (forest_parts.empty()) return result;

    // ========================================================================
    // Phase 3: 计算 SUM(l_quantity) GROUP BY (l_partkey, l_suppkey)
    // 只处理 forest_parts 中的 lineitem
    // ========================================================================
    // 使用复合键: (partkey << 20) | suppkey (假设 suppkey < 1M)
    std::unordered_map<int64_t, int64_t> qty_sum;
    qty_sum.reserve(partsupp_count / 10);

    for (size_t i = 0; i < lineitem_count; ++i) {
        if (l_shipdate[i] < date_lo || l_shipdate[i] >= date_hi) continue;
        if (!forest_parts.count(l_partkey[i])) continue;

        int64_t key = (static_cast<int64_t>(l_partkey[i]) << 20) |
                      static_cast<uint32_t>(l_suppkey[i] & 0xFFFFF);
        qty_sum[key] += l_quantity[i];
    }

    // ========================================================================
    // Phase 4: 找满足条件的 suppkey
    // ========================================================================
    std::unordered_set<int32_t> valid_suppkeys;

    for (size_t i = 0; i < partsupp_count; ++i) {
        if (!forest_parts.count(ps_partkey[i])) continue;

        int64_t key = (static_cast<int64_t>(ps_partkey[i]) << 20) |
                      static_cast<uint32_t>(ps_suppkey[i] & 0xFFFFF);

        auto it = qty_sum.find(key);
        if (it != qty_sum.end()) {
            int64_t threshold = static_cast<int64_t>(it->second * quantity_factor);
            if (ps_availqty[i] > threshold) {
                valid_suppkeys.insert(ps_suppkey[i]);
            }
        }
    }

    // ========================================================================
    // Phase 5: 过滤 supplier (国家 + IN valid_suppkeys)
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
// Q17 优化实现 V2 - 批量处理
// ============================================================================

Q17OptimizerV2::Result Q17OptimizerV2::execute(
    const int32_t* p_partkey,
    const std::vector<std::string>& p_brand,
    const std::vector<std::string>& p_container,
    size_t part_count,
    const int32_t* l_partkey,
    const int64_t* l_quantity,
    const int64_t* l_extendedprice,
    size_t lineitem_count,
    const std::string& target_brand,
    const std::string& target_container,
    double quantity_factor
) {
    // ========================================================================
    // Phase 1: 找目标 parts
    // ========================================================================
    std::unordered_set<int32_t> target_parts;
    target_parts.reserve(part_count / 1000);

    for (size_t i = 0; i < part_count; ++i) {
        if (p_brand[i] == target_brand && p_container[i] == target_container) {
            target_parts.insert(p_partkey[i]);
        }
    }

    if (target_parts.empty()) {
        return {0, 0.0};
    }

    // ========================================================================
    // Phase 2: 单遍扫描 - 累加 qty_sum/qty_count + 收集待评估行
    // ========================================================================
    struct PartStats {
        int64_t qty_sum = 0;
        int64_t qty_count = 0;
    };
    std::unordered_map<int32_t, PartStats> part_stats;
    part_stats.reserve(target_parts.size());

    // 批量收集待评估行
    struct PendingRow {
        int32_t partkey;
        int64_t quantity;
        int64_t extendedprice;
    };
    std::vector<PendingRow> pending_rows;
    pending_rows.reserve(lineitem_count / 100);  // 估计 1% 匹配

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t pk = l_partkey[i];
        if (!target_parts.count(pk)) continue;

        auto& stats = part_stats[pk];
        stats.qty_sum += l_quantity[i];
        stats.qty_count++;

        pending_rows.push_back({pk, l_quantity[i], l_extendedprice[i]});
    }

    // ========================================================================
    // Phase 3: 计算 AVG 阈值
    // ========================================================================
    std::unordered_map<int32_t, int64_t> thresholds;
    for (const auto& [pk, stats] : part_stats) {
        if (stats.qty_count > 0) {
            int64_t avg = stats.qty_sum / stats.qty_count;
            thresholds[pk] = static_cast<int64_t>(avg * quantity_factor);
        }
    }

    // ========================================================================
    // Phase 4: 批量过滤并聚合
    // ========================================================================
    int64_t total_price = 0;

    #ifdef __aarch64__
    // SIMD 版本 - 4 路展开
    size_t i = 0;
    for (; i + 4 <= pending_rows.size(); i += 4) {
        // 获取阈值
        int64_t t0 = thresholds[pending_rows[i].partkey];
        int64_t t1 = thresholds[pending_rows[i+1].partkey];
        int64_t t2 = thresholds[pending_rows[i+2].partkey];
        int64_t t3 = thresholds[pending_rows[i+3].partkey];

        // 比较 quantity < threshold
        if (pending_rows[i].quantity < t0) {
            total_price += pending_rows[i].extendedprice;
        }
        if (pending_rows[i+1].quantity < t1) {
            total_price += pending_rows[i+1].extendedprice;
        }
        if (pending_rows[i+2].quantity < t2) {
            total_price += pending_rows[i+2].extendedprice;
        }
        if (pending_rows[i+3].quantity < t3) {
            total_price += pending_rows[i+3].extendedprice;
        }
    }

    // 处理剩余
    for (; i < pending_rows.size(); ++i) {
        int64_t t = thresholds[pending_rows[i].partkey];
        if (pending_rows[i].quantity < t) {
            total_price += pending_rows[i].extendedprice;
        }
    }
    #else
    for (const auto& row : pending_rows) {
        int64_t t = thresholds[row.partkey];
        if (row.quantity < t) {
            total_price += row.extendedprice;
        }
    }
    #endif

    // ========================================================================
    // 返回结果
    // ========================================================================
    Result result;
    result.sum_extendedprice = total_price;
    result.avg_yearly = static_cast<double>(total_price) / 7.0 / 10000.0;
    return result;
}

// ============================================================================
// Q8 优化实现
// ============================================================================

Q8Optimizer::Result Q8Optimizer::execute(
    TPCHDataLoader& loader,
    const std::string& target_region,
    const std::string& target_nation,
    const std::string& target_type,
    int32_t date_lo,
    int32_t date_hi
) {
    Result result;

    const auto& reg = loader.region();
    const auto& nat = loader.nation();
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();
    const auto& li = loader.lineitem();
    const auto& supp = loader.supplier();
    const auto& part = loader.part();

    // ========================================================================
    // Phase 1: 预计算小表
    // ========================================================================
    // 1.1 找目标区域
    int32_t target_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == target_region) {
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }
    if (target_regionkey < 0) return result;

    // 1.2 找目标区域的国家 -> 客户
    std::unordered_set<int32_t> america_nations;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == target_regionkey) {
            america_nations.insert(nat.n_nationkey[i]);
        }
    }

    // 1.3 构建 custkey -> 是否在目标区域
    std::unordered_set<int32_t> america_customers;
    for (size_t i = 0; i < cust.count; ++i) {
        if (america_nations.count(cust.c_nationkey[i])) {
            america_customers.insert(cust.c_custkey[i]);
        }
    }

    // 1.4 找目标 part (p_type = 'ECONOMY ANODIZED STEEL')
    std::unordered_set<int32_t> target_parts;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == target_type) {
            target_parts.insert(part.p_partkey[i]);
        }
    }

    // 1.5 构建 suppkey -> nationkey -> nation_name
    std::unordered_map<int32_t, std::string> suppkey_to_nation;
    for (size_t i = 0; i < supp.count; ++i) {
        int32_t nk = supp.s_nationkey[i];
        for (size_t j = 0; j < nat.count; ++j) {
            if (nat.n_nationkey[j] == nk) {
                suppkey_to_nation[supp.s_suppkey[i]] = nat.n_name[j];
                break;
            }
        }
    }

    // ========================================================================
    // Phase 2: 构建 orderkey -> (year, custkey) 映射
    // ========================================================================
    struct OrderInfo {
        int32_t year;
        int32_t custkey;
    };
    std::unordered_map<int32_t, OrderInfo> order_info;
    order_info.reserve(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t odate = ord.o_orderdate[i];
        if (odate < date_lo || odate > date_hi) continue;

        if (!america_customers.count(ord.o_custkey[i])) continue;

        // 计算年份 (简化: epoch days / 365 + 1970)
        int32_t year = 1970 + odate / 365;

        order_info[ord.o_orderkey[i]] = {year, ord.o_custkey[i]};
    }

    // ========================================================================
    // Phase 3: 扫描 lineitem 并聚合
    // ========================================================================
    // year -> (brazil_volume, total_volume)
    std::unordered_map<int32_t, std::pair<int64_t, int64_t>> year_volumes;

    for (size_t i = 0; i < li.count; ++i) {
        // 检查 part
        if (!target_parts.count(li.l_partkey[i])) continue;

        // 检查 order
        auto oit = order_info.find(li.l_orderkey[i]);
        if (oit == order_info.end()) continue;

        int32_t year = oit->second.year;

        // 计算 volume
        int64_t volume = li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;

        // 检查 supplier nation
        auto sit = suppkey_to_nation.find(li.l_suppkey[i]);
        std::string nation_name = sit != suppkey_to_nation.end() ? sit->second : "";

        auto& yv = year_volumes[year];
        yv.second += volume;  // total
        if (nation_name == target_nation) {
            yv.first += volume;  // brazil
        }
    }

    // ========================================================================
    // Phase 4: 计算结果
    // ========================================================================
    for (const auto& [year, volumes] : year_volumes) {
        double mkt_share = volumes.second > 0 ?
            static_cast<double>(volumes.first) / volumes.second : 0.0;
        result.year_shares.emplace_back(year, mkt_share);
    }

    // 按年份排序
    std::sort(result.year_shares.begin(), result.year_shares.end());

    return result;
}

// ============================================================================
// V37 查询入口
// ============================================================================

void run_q22_v37(TPCHDataLoader& loader) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();

    auto result = Q22Optimizer::execute(
        cust.c_custkey.data(),
        cust.c_acctbal.data(),
        cust.c_phone,
        cust.count,
        ord.o_custkey.data(),
        ord.count,
        {"13", "31", "23", "29", "30", "18", "17"},
        1, 2
    );

    // 结果用于验证
    volatile int64_t total = 0;
    for (int i = 0; i < 7; ++i) {
        total += result.counts[i];
    }
    (void)total;
}

void run_q21_v37(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    auto result = Q21Optimizer::execute(
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
        ord.o_orderstatus.data(),  // 注意: .data()
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

void run_q20_v37(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& part = loader.part();
    const auto& ps = loader.partsupp();
    const auto& li = loader.lineitem();

    auto result = Q20OptimizerV2::execute(
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
        nations::CANADA,
        thunderduck::tpch::constants::dates::D1994_01_01,
        thunderduck::tpch::constants::dates::D1995_01_01,
        0.5
    );

    // 结果用于验证
    volatile size_t count = result.suppliers.size();
    (void)count;
}

void run_q17_v37(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& li = loader.lineitem();

    auto result = Q17OptimizerV2::execute(
        part.p_partkey.data(),
        part.p_brand,
        part.p_container,
        part.count,
        li.l_partkey.data(),
        li.l_quantity.data(),
        li.l_extendedprice.data(),
        li.count,
        "Brand#23",
        "MED BOX",
        0.2
    );

    // 结果用于验证
    volatile double avg_yearly = result.avg_yearly;
    (void)avg_yearly;
}

void run_q8_v37(TPCHDataLoader& loader) {
    auto result = Q8Optimizer::execute(
        loader,
        regions::AMERICA,
        nations::BRAZIL,
        "ECONOMY ANODIZED STEEL",
        thunderduck::tpch::constants::dates::D1995_01_01,
        thunderduck::tpch::constants::dates::D1996_12_31
    );

    // 结果用于验证
    volatile size_t count = result.year_shares.size();
    (void)count;
}

} // namespace ops_v37
} // namespace tpch
} // namespace thunderduck
