/**
 * ThunderDuck TPC-H V35 通用算子实现
 *
 * 实现查询: Q3, Q8, Q14, Q22, Q21
 * 使用通用化算子: DirectArrayIndexBuilder, SIMDStringProcessor, SemiAntiJoin, etc.
 *
 * @version 35.0
 * @date 2026-01-29
 */

#include "tpch_operators_v35.h"
#include "tpch_operators_v34.h"  // V34 for Q13
#include "tpch_constants.h"      // 统一常量定义
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v35 {

// ============================================================================
// Q3: 航运优先级 (使用 PipelineFusion + DirectArrayIndexBuilder)
// ============================================================================

void run_q3_v35(TPCHDataLoader& loader) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();
    const auto& li = loader.lineitem();

    // 日期常量
    const int32_t DATE_1995_03_15 = dates::D1995_03_15;

    // Phase 1: 过滤 customer (BUILDING) -> custkey 集合
    DirectArrayIndexBuilder<int8_t> building_customers;
    {
        std::vector<int32_t> valid_custkeys;
        std::vector<int8_t> flags;
        valid_custkeys.reserve(cust.count / 5);
        flags.reserve(cust.count / 5);

        for (size_t i = 0; i < cust.count; ++i) {
            if (cust.c_mktsegment[i] == "BUILDING") {
                valid_custkeys.push_back(cust.c_custkey[i]);
                flags.push_back(1);
            }
        }

        building_customers.build(valid_custkeys.data(), flags.data(), valid_custkeys.size(), int8_t(0));
    }

    // Phase 2: 过滤 orders (o_orderdate < DATE) 且 custkey 在 BUILDING 中
    // 构建 orderkey -> (orderdate, shippriority) 映射
    struct OrderInfo {
        int32_t orderdate;
        int32_t shippriority;
    };

    DirectArrayIndexBuilder<OrderInfo> valid_orders;
    {
        std::vector<int32_t> orderkeys;
        std::vector<OrderInfo> order_infos;
        orderkeys.reserve(ord.count / 2);
        order_infos.reserve(ord.count / 2);

        for (size_t i = 0; i < ord.count; ++i) {
            if (ord.o_orderdate[i] < DATE_1995_03_15 &&
                building_customers.contains(ord.o_custkey[i])) {
                orderkeys.push_back(ord.o_orderkey[i]);
                order_infos.push_back({ord.o_orderdate[i], ord.o_shippriority[i]});
            }
        }

        valid_orders.build(orderkeys.data(), order_infos.data(), orderkeys.size());
    }

    // Phase 3: 并行扫描 lineitem，融合 JOIN + Aggregate
    std::unordered_map<int32_t, int64_t> revenue_map;
    std::unordered_map<int32_t, OrderInfo> order_info_map;

    // 并行聚合
    const size_t num_threads = 8;
    auto& pool = ThreadPool::instance();
    pool.prewarm(num_threads, li.count / num_threads);

    struct ThreadLocal {
        std::unordered_map<int32_t, int64_t> revenue;
        std::unordered_map<int32_t, OrderInfo> info;
    };
    std::vector<ThreadLocal> thread_locals(num_threads);

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&thread_locals, &li, &valid_orders, t, start, end, DATE_1995_03_15]() {
            auto& local = thread_locals[t];

            for (size_t i = start; i < end; ++i) {
                // 过滤: l_shipdate > DATE_1995_03_15
                if (li.l_shipdate[i] <= DATE_1995_03_15) continue;

                int32_t orderkey = li.l_orderkey[i];
                const OrderInfo* info = valid_orders.find(orderkey);
                if (info == nullptr) continue;

                // 计算 revenue
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]);

                local.revenue[orderkey] += revenue;
                if (local.info.find(orderkey) == local.info.end()) {
                    local.info[orderkey] = *info;
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    // 合并线程结果
    for (const auto& local : thread_locals) {
        for (const auto& [k, v] : local.revenue) {
            revenue_map[k] += v;
        }
        for (const auto& [k, v] : local.info) {
            if (order_info_map.find(k) == order_info_map.end()) {
                order_info_map[k] = v;
            }
        }
    }

    // 结果在 revenue_map 和 order_info_map 中
    // 排序和输出由 executor 处理
}

// ============================================================================
// Q8: 国家市场份额 (使用 ConditionalAggregator + DirectArrayIndexBuilder)
// ============================================================================

void run_q8_v35(TPCHDataLoader& loader) {
    const auto& reg = loader.region();
    const auto& nat = loader.nation();
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();
    const auto& li = loader.lineitem();
    const auto& supp = loader.supplier();
    const auto& part = loader.part();

    // 日期范围
    const int32_t DATE_1995_01_01 = dates::D1995_01_01;
    const int32_t DATE_1996_12_31 = dates::D1996_12_31;

    // Phase 1: 找 AMERICA region 的国家
    int32_t america_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::AMERICA) {
            america_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    // 预计算 nation 信息
    DirectArrayIndexBuilder<int8_t> nation_in_america;
    int32_t brazil_nationkey = -1;

    {
        std::vector<int32_t> nation_keys;
        std::vector<int8_t> in_america;
        nation_keys.reserve(nat.count);
        in_america.reserve(nat.count);

        for (size_t i = 0; i < nat.count; ++i) {
            nation_keys.push_back(nat.n_nationkey[i]);
            in_america.push_back(nat.n_regionkey[i] == america_regionkey ? 1 : 0);
            if (nat.n_name[i] == nations::BRAZIL) {
                brazil_nationkey = nat.n_nationkey[i];
            }
        }

        nation_in_america.build(nation_keys.data(), in_america.data(), nation_keys.size(), int8_t(0));
    }

    // Phase 2: 预计算 supplier -> nationkey
    DirectArrayIndexBuilder<int32_t> supp_nation;
    {
        supp_nation.build(supp.s_suppkey.data(), supp.s_nationkey.data(), supp.count, -1);
    }

    // Phase 3: 预计算 customer -> nationkey, 并标记 AMERICA 客户
    DirectArrayIndexBuilder<int8_t> cust_in_america;
    {
        std::vector<int32_t> custkeys;
        std::vector<int8_t> in_america;
        custkeys.reserve(cust.count);
        in_america.reserve(cust.count);

        for (size_t i = 0; i < cust.count; ++i) {
            custkeys.push_back(cust.c_custkey[i]);
            int8_t is_america = nation_in_america.get(cust.c_nationkey[i]);
            in_america.push_back(is_america);
        }

        cust_in_america.build(custkeys.data(), in_america.data(), custkeys.size(), int8_t(0));
    }

    // Phase 4: 过滤 part (ECONOMY ANODIZED STEEL)
    DirectArrayIndexBuilder<int8_t> valid_parts;
    {
        std::vector<int32_t> valid_partkeys;
        std::vector<int8_t> flags;
        valid_partkeys.reserve(part.count / 100);
        flags.reserve(part.count / 100);

        for (size_t i = 0; i < part.count; ++i) {
            if (part.p_type[i] == "ECONOMY ANODIZED STEEL") {
                valid_partkeys.push_back(part.p_partkey[i]);
                flags.push_back(1);
            }
        }

        valid_parts.build(valid_partkeys.data(), flags.data(), valid_partkeys.size(), int8_t(0));
    }

    // Phase 5: 过滤 orders (日期范围) 并预计算信息
    struct OrderInfo {
        int16_t year;       // 1995 or 1996 -> 0 or 1
        int8_t is_america;  // 客户是否在 AMERICA
    };

    DirectArrayIndexBuilder<OrderInfo> order_info;
    {
        std::vector<int32_t> orderkeys;
        std::vector<OrderInfo> infos;
        orderkeys.reserve(ord.count / 4);
        infos.reserve(ord.count / 4);

        for (size_t i = 0; i < ord.count; ++i) {
            if (ord.o_orderdate[i] >= DATE_1995_01_01 &&
                ord.o_orderdate[i] <= DATE_1996_12_31) {
                int8_t is_america = cust_in_america.get(ord.o_custkey[i]);
                if (is_america) {
                    // 计算年份: 简化为 (date - DATE_1995_01_01) / 365
                    int32_t days_from_1995 = ord.o_orderdate[i] - DATE_1995_01_01;
                    int16_t year_idx = (days_from_1995 >= 365) ? 1 : 0;
                    orderkeys.push_back(ord.o_orderkey[i]);
                    infos.push_back({year_idx, is_america});
                }
            }
        }

        order_info.build(orderkeys.data(), infos.data(), orderkeys.size());
    }

    // Phase 6: 并行扫描 lineitem，条件聚合
    std::array<std::atomic<int64_t>, 2> brazil_vol{};
    std::array<std::atomic<int64_t>, 2> total_vol{};

    const size_t num_threads = 8;
    auto& pool = ThreadPool::instance();
    pool.prewarm(num_threads, li.count / num_threads);

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&brazil_vol, &total_vol, &li, &valid_parts, &order_info, &supp_nation, brazil_nationkey, start, end]() {
            std::array<int64_t, 2> local_brazil{};
            std::array<int64_t, 2> local_total{};

            for (size_t i = start; i < end; ++i) {
                // 检查 part
                if (!valid_parts.contains(li.l_partkey[i])) continue;

                // 检查 order
                const OrderInfo* oi = order_info.find(li.l_orderkey[i]);
                if (oi == nullptr) continue;

                // 获取 supplier nation
                int32_t supp_nationkey = supp_nation.get(li.l_suppkey[i]);
                if (supp_nationkey < 0) continue;

                // 计算 volume
                int64_t volume = static_cast<int64_t>(li.l_extendedprice[i]) *
                                 (10000 - li.l_discount[i]);

                int year_idx = oi->year;
                local_total[year_idx] += volume;

                if (supp_nationkey == brazil_nationkey) {
                    local_brazil[year_idx] += volume;
                }
            }

            // 原子累加
            brazil_vol[0] += local_brazil[0];
            brazil_vol[1] += local_brazil[1];
            total_vol[0] += local_total[0];
            total_vol[1] += local_total[1];
        }));
    }

    for (auto& f : futures) f.get();

    // 结果在 brazil_vol 和 total_vol 数组中
}

// ============================================================================
// Q14: 促销效果 (使用 SIMDStringProcessor + DirectArrayIndexBuilder)
// ============================================================================

void run_q14_v35(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& li = loader.lineitem();

    // 日期范围 (Q14: 1995年9月)
    const int32_t DATE_1995_09_01 = dates::D1995_09_01;
    const int32_t DATE_1995_10_01 = dates::D1995_10_01;

    // Phase 1: 使用 SIMDStringProcessor 批量检测 PROMO 前缀
    std::vector<bool> promo_results;
    SIMDStringProcessor::starts_with_batch(part.p_type, "PROMO", promo_results);

    // 构建 partkey -> is_promo 索引
    DirectArrayIndexBuilder<int8_t> is_promo;
    {
        std::vector<int32_t> partkeys;
        std::vector<int8_t> promo_flags;
        partkeys.reserve(part.count);
        promo_flags.reserve(part.count);

        for (size_t i = 0; i < part.count; ++i) {
            partkeys.push_back(part.p_partkey[i]);
            promo_flags.push_back(promo_results[i] ? 1 : 0);
        }

        is_promo.build(partkeys.data(), promo_flags.data(), partkeys.size(), int8_t(0));
    }

    // Phase 2: 并行扫描 lineitem，条件聚合
    std::atomic<int64_t> promo_revenue{0};
    std::atomic<int64_t> total_revenue{0};

    const size_t num_threads = 8;
    auto& pool = ThreadPool::instance();
    pool.prewarm(num_threads, li.count / num_threads);

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&promo_revenue, &total_revenue, &li, &is_promo, DATE_1995_09_01, DATE_1995_10_01, start, end]() {
            int64_t local_promo = 0;
            int64_t local_total = 0;

            for (size_t i = start; i < end; ++i) {
                // 过滤日期
                if (li.l_shipdate[i] < DATE_1995_09_01 ||
                    li.l_shipdate[i] > DATE_1995_10_01) continue;

                // 计算 revenue
                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]);

                local_total += revenue;

                // 检查是否是 PROMO
                if (is_promo.get(li.l_partkey[i]) == 1) {
                    local_promo += revenue;
                }
            }

            promo_revenue += local_promo;
            total_revenue += local_total;
        }));
    }

    for (auto& f : futures) f.get();

    // 结果: promo_revenue / total_revenue * 100
}

// ============================================================================
// Q22: 全球销售机会 (使用 SIMDStringProcessor + SemiAntiJoin)
// ============================================================================

void run_q22_v35(TPCHDataLoader& loader) {
    const auto& cust = loader.customer();
    const auto& ord = loader.orders();

    // 目标国家码
    const std::vector<int16_t> target_codes = {13, 31, 23, 29, 30, 18, 17};
    std::array<bool, 100> code_valid{};
    std::array<int8_t, 100> code_to_idx{};
    std::fill(code_to_idx.begin(), code_to_idx.end(), -1);

    for (size_t i = 0; i < target_codes.size(); ++i) {
        code_valid[target_codes[i]] = true;
        code_to_idx[target_codes[i]] = static_cast<int8_t>(i);
    }

    // Phase 1: 使用 SIMDStringProcessor 批量提取电话前缀
    std::vector<int16_t> phone_codes;
    SIMDStringProcessor::prefix_to_int_batch(cust.c_phone, 2, phone_codes);

    // Phase 2: 计算 AVG(c_acctbal)
    int64_t sum_positive = 0;
    int64_t count_positive = 0;

    for (size_t i = 0; i < cust.count; ++i) {
        int16_t code = phone_codes[i];
        if (code >= 0 && code < 100 && code_valid[code]) {
            if (cust.c_acctbal[i] > 0) {
                sum_positive += cust.c_acctbal[i];
                count_positive++;
            }
        }
    }

    int32_t avg_acctbal = (count_positive > 0)
        ? static_cast<int32_t>(sum_positive / count_positive)
        : 0;

    // Phase 3: 使用 SemiAntiJoin 构建订单客户集合
    SemiAntiJoin order_customers;
    order_customers.build(ord.o_custkey.data(), ord.count);

    // Phase 4: 扫描 customer，聚合结果
    std::array<int64_t, 7> counts{};
    std::array<int64_t, 7> sums{};

    for (size_t i = 0; i < cust.count; ++i) {
        int16_t code = phone_codes[i];
        if (code < 0 || code >= 100 || !code_valid[code]) continue;
        if (cust.c_acctbal[i] <= avg_acctbal) continue;

        // 使用 SemiAntiJoin 检查 NOT EXISTS
        if (order_customers.exists(cust.c_custkey[i])) continue;

        int8_t idx = code_to_idx[code];
        counts[idx]++;
        sums[idx] += cust.c_acctbal[i];
    }

    // 结果在 counts 和 sums 数组中
}

// ============================================================================
// Q21: 供应商等待 (使用 SemiAntiJoin + DirectArrayIndexBuilder)
// ============================================================================

void run_q21_v35(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    // 找 SAUDI ARABIA nation
    int32_t saudi_nationkey = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::SAUDI_ARABIA) {
            saudi_nationkey = nat.n_nationkey[i];
            break;
        }
    }

    // Phase 1: 过滤 supplier (SAUDI ARABIA)
    DirectArrayIndexBuilder<int8_t> saudi_suppliers;
    {
        std::vector<int32_t> valid_suppkeys;
        std::vector<int8_t> flags;
        valid_suppkeys.reserve(supp.count / 25);
        flags.reserve(supp.count / 25);

        for (size_t i = 0; i < supp.count; ++i) {
            if (supp.s_nationkey[i] == saudi_nationkey) {
                valid_suppkeys.push_back(supp.s_suppkey[i]);
                flags.push_back(1);
            }
        }

        saudi_suppliers.build(valid_suppkeys.data(), flags.data(), valid_suppkeys.size(), int8_t(0));
    }

    // Phase 2: 过滤 orders (o_orderstatus = 'F')
    DirectArrayIndexBuilder<int8_t> failed_orders;
    {
        std::vector<int32_t> failed_orderkeys;
        std::vector<int8_t> flags;
        failed_orderkeys.reserve(ord.count / 2);
        flags.reserve(ord.count / 2);

        for (size_t i = 0; i < ord.count; ++i) {
            if (ord.o_orderstatus[i] == 'F') {
                failed_orderkeys.push_back(ord.o_orderkey[i]);
                flags.push_back(1);
            }
        }

        failed_orders.build(failed_orderkeys.data(), flags.data(), failed_orderkeys.size(), int8_t(0));
    }

    // Phase 3: 构建 orderkey -> 迟交供应商集合 和 所有供应商集合
    std::unordered_map<int32_t, std::unordered_set<int32_t>> order_late_supps;
    std::unordered_map<int32_t, std::unordered_set<int32_t>> order_all_supps;

    for (size_t i = 0; i < li.count; ++i) {
        order_all_supps[li.l_orderkey[i]].insert(li.l_suppkey[i]);
        if (li.l_receiptdate[i] > li.l_commitdate[i]) {
            order_late_supps[li.l_orderkey[i]].insert(li.l_suppkey[i]);
        }
    }

    // Phase 4: 统计符合条件的供应商
    std::unordered_map<int32_t, int64_t> supplier_counts;

    for (size_t i = 0; i < li.count; ++i) {
        int32_t suppkey = li.l_suppkey[i];
        int32_t orderkey = li.l_orderkey[i];

        // 条件 1: SAUDI ARABIA supplier
        if (!saudi_suppliers.contains(suppkey)) continue;

        // 条件 2: Failed order
        if (!failed_orders.contains(orderkey)) continue;

        // 条件 3: 该 supplier 迟交
        if (li.l_receiptdate[i] <= li.l_commitdate[i]) continue;

        // 条件 4: 存在其他供应商
        const auto& all_supps = order_all_supps[orderkey];
        bool has_other_supplier = false;
        for (int32_t s : all_supps) {
            if (s != suppkey) {
                has_other_supplier = true;
                break;
            }
        }
        if (!has_other_supplier) continue;

        // 条件 5: 不存在其他迟交的供应商
        const auto& late_supps = order_late_supps[orderkey];
        bool has_other_late = false;
        for (int32_t s : late_supps) {
            if (s != suppkey) {
                has_other_late = true;
                break;
            }
        }
        if (has_other_late) continue;

        supplier_counts[suppkey]++;
    }

    // 结果在 supplier_counts 中
}

// ============================================================================
// 其他查询的 V35 版本 (暂时使用 V32/V34 版本)
// ============================================================================

void run_q5_v35(TPCHDataLoader& loader) {
    ops_v32::run_q5_v32(loader);
}

void run_q7_v35(TPCHDataLoader& loader) {
    ops_v32::run_q7_v32(loader);
}

void run_q9_v35(TPCHDataLoader& loader) {
    ops_v32::run_q9_v32(loader);
}

void run_q13_v35(TPCHDataLoader& loader) {
    ops_v34::run_q13_v34(loader);
}

} // namespace ops_v35
} // namespace tpch
} // namespace thunderduck
