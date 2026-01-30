/**
 * ThunderDuck TPC-H Operators V52
 *
 * 优化目标:
 * - Q5: DirectArrayJoin (1.07x → 2.0x+)
 * - Q6: SIMDBranchlessFilter (1.80x → 3.5x+)
 * - Q3: BitmapPredicateIndex (1.44x → 2.0x+)
 * - Q9: DirectArrayJoin + BitmapPredicateIndex (1.52x → 2.5x+)
 *
 * @version 52
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_constants.h"
#include "../../include/thunderduck/generic_operators_v52.h"
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v52 {

using namespace operators::v52;

// ============================================================================
// Q6: 使用 SIMDBranchlessFilter
// ============================================================================

/**
 * Q6: 预测收入变化
 *
 * SELECT SUM(l_extendedprice * l_discount) AS revenue
 * FROM lineitem
 * WHERE l_shipdate >= DATE '1994-01-01'
 *   AND l_shipdate < DATE '1995-01-01'
 *   AND l_discount BETWEEN 0.05 AND 0.07
 *   AND l_quantity < 24
 *
 * 优化策略:
 * - SIMDBranchlessFilter: 完全无分支 SIMD 4 条件过滤
 * - 8 线程并行
 */
inline void run_q6_v52(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& lineitem = loader.lineitem();
    size_t n = lineitem.l_orderkey.size();

    // 参数 (数据已经乘以 10000)
    constexpr int32_t DATE_LO = constants::dates::D1994_01_01;
    constexpr int32_t DATE_HI = constants::dates::D1995_01_01;
    constexpr double DISC_LO = 0.05;
    constexpr double DISC_HI = 0.07;
    constexpr double QTY_TH = 24.0;

    // 转换数据类型 (int64_t x10000 → double)
    std::vector<double> discount(n);
    std::vector<double> quantity(n);
    std::vector<double> extendedprice(n);

    for (size_t i = 0; i < n; ++i) {
        discount[i] = lineitem.l_discount[i] / 10000.0;
        quantity[i] = lineitem.l_quantity[i] / 10000.0;
        extendedprice[i] = lineitem.l_extendedprice[i] / 10000.0;
    }

    // 使用 SIMDBranchlessFilter
    SIMDBranchlessFilter::Config config;
    config.num_threads = 8;
    SIMDBranchlessFilter filter(config);

    double result = 0.0;
    auto stats = filter.filter_sum_q6(
        lineitem.l_shipdate.data(),
        discount.data(),
        quantity.data(),
        extendedprice.data(),
        n,
        DATE_LO, DATE_HI,
        DISC_LO, DISC_HI,
        QTY_TH,
        result
    );

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n=== Q6 V52 Results (SIMDBranchlessFilter) ===\n";
    std::cout << "Revenue: " << std::fixed << result << "\n";
    std::cout << "Matched: " << stats.matched_rows << " / " << stats.total_rows
              << " (" << (100.0 * stats.matched_rows / stats.total_rows) << "%)\n";
    std::cout << "Time: " << time_ms << " ms (filter: " << stats.filter_time_ms << " ms)\n";
}

// ============================================================================
// Q5: 使用 DirectArrayJoin
// ============================================================================

/**
 * Q5: 本地供应商收入
 *
 * 优化策略:
 * - DirectArrayJoin 替代哈希表 (suppkey, custkey 都是小范围整数)
 * - O(1) 数组索引 vs O(1) 哈希 (但缓存更友好)
 */
inline void run_q5_v52(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& customer = loader.customer();
    const auto& orders = loader.orders();
    const auto& lineitem = loader.lineitem();
    const auto& supplier = loader.supplier();
    const auto& nation = loader.nation();
    const auto& region = loader.region();

    // Step 1: 找到 ASIA 的 regionkey
    int32_t asia_regionkey = -1;
    for (size_t i = 0; i < region.r_regionkey.size(); ++i) {
        if (region.r_name[i] == constants::regions::ASIA) {
            asia_regionkey = region.r_regionkey[i];
            break;
        }
    }

    // Step 2: 构建 ASIA 国家映射
    std::unordered_map<int32_t, std::string> asia_nations;
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_regionkey[i] == asia_regionkey) {
            asia_nations[nation.n_nationkey[i]] = nation.n_name[i];
        }
    }

    // Step 3: DirectArrayJoin - suppkey → nationkey
    DirectArrayJoin<int8_t, 10001> supp_to_nation;
    std::vector<int32_t> supp_keys;
    std::vector<int8_t> supp_nations;
    for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
        if (asia_nations.count(supplier.s_nationkey[i])) {
            supp_keys.push_back(supplier.s_suppkey[i]);
            supp_nations.push_back(static_cast<int8_t>(supplier.s_nationkey[i]));
        }
    }
    supp_to_nation.build(supp_keys.begin(), supp_keys.end(), supp_nations.begin());

    // Step 4: DirectArrayJoin - custkey → nationkey
    DirectArrayJoin<int8_t, 150001> cust_to_nation;
    std::vector<int32_t> cust_keys;
    std::vector<int8_t> cust_nations;
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (asia_nations.count(customer.c_nationkey[i])) {
            cust_keys.push_back(customer.c_custkey[i]);
            cust_nations.push_back(static_cast<int8_t>(customer.c_nationkey[i]));
        }
    }
    cust_to_nation.build(cust_keys.begin(), cust_keys.end(), cust_nations.begin());

    // Step 5: 哈希表 - orderkey → custkey (避免栈溢出)
    constexpr int32_t DATE_1994_START = constants::dates::D1994_01_01;
    constexpr int32_t DATE_1995_START = constants::dates::D1995_01_01;

    std::unordered_map<int32_t, int32_t> order_to_cust;
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] >= DATE_1994_START &&
            orders.o_orderdate[i] < DATE_1995_START) {
            int8_t cust_nation = cust_to_nation.lookup(orders.o_custkey[i]);
            if (cust_nation != -1) {
                order_to_cust[orders.o_orderkey[i]] = orders.o_custkey[i];
            }
        }
    }

    // Step 6: 扫描 lineitem 并聚合
    std::array<double, 25> nation_revenue{};  // 25 nations max

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];
        auto it = order_to_cust.find(orderkey);
        if (it == order_to_cust.end()) continue;
        int32_t custkey = it->second;

        int32_t suppkey = lineitem.l_suppkey[i];
        int8_t supp_nation = supp_to_nation.lookup(suppkey);
        if (supp_nation == -1) continue;

        int8_t cust_nation = cust_to_nation.lookup(custkey);
        if (cust_nation != supp_nation) continue;

        // lineitem 数值列是 int64_t x10000
        double extprice = lineitem.l_extendedprice[i] / 10000.0;
        double disc = lineitem.l_discount[i] / 10000.0;
        double revenue = extprice * (1.0 - disc);
        nation_revenue[supp_nation] += revenue;
    }

    // Step 7: 收集结果并排序
    struct Result {
        std::string n_name;
        double revenue;
    };
    std::vector<Result> results;

    for (const auto& kv : asia_nations) {
        if (nation_revenue[kv.first] > 0) {
            results.push_back({kv.second, nation_revenue[kv.first]});
        }
    }

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.revenue > b.revenue; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n=== Q5 V52 Results (DirectArrayJoin) ===\n";
    for (const auto& r : results) {
        std::cout << r.n_name << " | " << std::fixed << r.revenue << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

// ============================================================================
// Q3: 使用 BitmapPredicateIndex
// ============================================================================

/**
 * Q3: 运输优先级
 *
 * 优化策略:
 * - BitmapPredicateIndex 预计算 BUILDING 客户
 * - BitmapPredicateIndex 预计算符合日期的订单
 * - 双重位图过滤减少 lineitem 扫描
 */
inline void run_q3_v52(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& customer = loader.customer();
    const auto& orders = loader.orders();
    const auto& lineitem = loader.lineitem();

    // Step 1: 构建 BUILDING 客户位图
    BitmapPredicateIndex<150001> building_customers;
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (customer.c_mktsegment[i] == "BUILDING") {
            building_customers.set(customer.c_custkey[i]);
        }
    }

    // Step 2: 构建符合条件的订单位图 + 订单信息
    constexpr int32_t DATE_THRESHOLD = constants::dates::D1995_03_15;

    // 使用哈希表替代 DirectArrayJoin (避免栈溢出)
    std::unordered_set<int32_t> valid_orders;
    std::unordered_map<int32_t, int64_t> order_info;  // orderkey → packed(orderdate, shippriority)

    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] < DATE_THRESHOLD &&
            building_customers.test(orders.o_custkey[i])) {
            valid_orders.insert(orders.o_orderkey[i]);
            int64_t info = (static_cast<int64_t>(orders.o_orderdate[i]) << 32) |
                           static_cast<uint32_t>(orders.o_shippriority[i]);
            order_info[orders.o_orderkey[i]] = info;
        }
    }

    // Step 3: 扫描 lineitem (使用哈希集合预过滤)
    std::unordered_map<int32_t, double> order_revenue;

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];

        // 哈希集合快速过滤
        if (valid_orders.find(orderkey) == valid_orders.end()) continue;

        // 日期条件: l_shipdate > 1995-03-15
        if (lineitem.l_shipdate[i] <= DATE_THRESHOLD) continue;

        // lineitem 数值列是 int64_t x10000，需要转换
        double extprice = lineitem.l_extendedprice[i] / 10000.0;
        double disc = lineitem.l_discount[i] / 10000.0;
        double revenue = extprice * (1.0 - disc);
        order_revenue[orderkey] += revenue;
    }

    // Step 4: 收集结果
    struct Result {
        int32_t orderkey;
        double revenue;
        int32_t orderdate;
        int32_t shippriority;
    };
    std::vector<Result> results;

    for (const auto& kv : order_revenue) {
        auto it = order_info.find(kv.first);
        if (it != order_info.end()) {
            int64_t info = it->second;
            results.push_back({
                kv.first,
                kv.second,
                static_cast<int32_t>(info >> 32),
                static_cast<int32_t>(info & 0xFFFFFFFF)
            });
        }
    }

    // 排序: revenue DESC, orderdate ASC
    std::partial_sort(results.begin(),
                      results.begin() + std::min(results.size(), size_t(10)),
                      results.end(),
                      [](const Result& a, const Result& b) {
                          if (std::abs(a.revenue - b.revenue) > 0.01)
                              return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n=== Q3 V52 Results (BitmapPredicateIndex) ===\n";
    size_t limit = std::min(results.size(), size_t(10));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << results[i].orderkey << " | " << std::fixed << results[i].revenue
                  << " | " << results[i].orderdate << " | " << results[i].shippriority << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

// ============================================================================
// Q9: 使用 DirectArrayJoin + BitmapPredicateIndex
// ============================================================================

/**
 * Q9: 产品类型利润
 *
 * 优化策略:
 * - BitmapPredicateIndex 预计算 LIKE '%green%' 的 partkey
 * - DirectArrayJoin 替代 supplier, partsupp 哈希表
 */
inline void run_q9_v52(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& part = loader.part();
    const auto& supplier = loader.supplier();
    const auto& lineitem = loader.lineitem();
    const auto& partsupp = loader.partsupp();
    const auto& orders = loader.orders();
    const auto& nation = loader.nation();

    // Step 1: 预计算 LIKE '%green%' 位图
    BitmapPredicateIndex<200001> green_parts;
    for (size_t i = 0; i < part.p_partkey.size(); ++i) {
        if (string_contains(part.p_name[i], "green")) {
            green_parts.set(part.p_partkey[i]);
        }
    }
    size_t green_count = green_parts.count();

    // Step 2: DirectArrayJoin - suppkey → nationkey
    DirectArrayJoin<int8_t, 10001> supp_to_nation;
    {
        std::vector<int32_t> keys;
        std::vector<int8_t> vals;
        for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
            keys.push_back(supplier.s_suppkey[i]);
            vals.push_back(static_cast<int8_t>(supplier.s_nationkey[i]));
        }
        supp_to_nation.build(keys.begin(), keys.end(), vals.begin());
    }

    // Step 3: 构建 nation 名称映射
    std::array<std::string, 25> nation_names;
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_nationkey[i] < 25) {
            nation_names[nation.n_nationkey[i]] = nation.n_name[i];
        }
    }

    // Step 4: 构建 partsupp 哈希表 (suppkey, partkey) → supplycost
    // 由于 partsupp 是多对多关系，使用哈希表
    std::unordered_map<uint64_t, double> ps_cost;
    for (size_t i = 0; i < partsupp.ps_partkey.size(); ++i) {
        if (green_parts.test(partsupp.ps_partkey[i])) {
            uint64_t key = (static_cast<uint64_t>(partsupp.ps_suppkey[i]) << 32) |
                           static_cast<uint32_t>(partsupp.ps_partkey[i]);
            ps_cost[key] = partsupp.ps_supplycost[i];
        }
    }

    // Step 5: 哈希表 - orderkey → year (避免栈溢出)
    std::unordered_map<int32_t, int16_t> order_to_year;
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        // 从 epoch days 提取年份
        int32_t year = 1970 + orders.o_orderdate[i] / 365;
        order_to_year[orders.o_orderkey[i]] = static_cast<int16_t>(year);
    }

    // Step 6: 扫描 lineitem 并聚合
    // Group by: nation, year → SUM(profit)
    std::unordered_map<uint32_t, double> profit_map;  // key = nation<<16|year

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t partkey = lineitem.l_partkey[i];

        // 位图预过滤
        if (!green_parts.test(partkey)) continue;

        int32_t suppkey = lineitem.l_suppkey[i];
        int32_t orderkey = lineitem.l_orderkey[i];

        // 查找 supplycost
        uint64_t ps_key = (static_cast<uint64_t>(suppkey) << 32) | static_cast<uint32_t>(partkey);
        auto it = ps_cost.find(ps_key);
        if (it == ps_cost.end()) continue;
        double supplycost = it->second;

        // 查找 nation
        int8_t nationkey = supp_to_nation.lookup(suppkey);
        if (nationkey == -1) continue;

        // 查找 year
        auto year_it = order_to_year.find(orderkey);
        if (year_it == order_to_year.end()) continue;
        int16_t year = year_it->second;

        // 计算 profit (lineitem 数值列是 int64_t x10000)
        double extprice = lineitem.l_extendedprice[i] / 10000.0;
        double disc = lineitem.l_discount[i] / 10000.0;
        double qty = lineitem.l_quantity[i] / 10000.0;
        double amount = extprice * (1.0 - disc) - supplycost * qty;

        uint32_t group_key = (static_cast<uint32_t>(nationkey) << 16) | static_cast<uint16_t>(year);
        profit_map[group_key] += amount;
    }

    // Step 7: 收集结果
    struct Result {
        std::string nation;
        int32_t year;
        double profit;
    };
    std::vector<Result> results;

    for (const auto& kv : profit_map) {
        int8_t nationkey = static_cast<int8_t>(kv.first >> 16);
        int16_t year = static_cast<int16_t>(kv.first & 0xFFFF);
        results.push_back({nation_names[nationkey], year, kv.second});
    }

    // 排序: nation ASC, year DESC
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) {
                  if (a.nation != b.nation) return a.nation < b.nation;
                  return a.year > b.year;
              });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\n=== Q9 V52 Results (DirectArrayJoin + BitmapPredicateIndex) ===\n";
    std::cout << "Green parts: " << green_count << " / " << part.p_partkey.size() << "\n";
    size_t limit = std::min(results.size(), size_t(20));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << results[i].nation << " | " << results[i].year
                  << " | " << std::fixed << results[i].profit << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

} // namespace ops_v52
} // namespace tpch
} // namespace thunderduck
