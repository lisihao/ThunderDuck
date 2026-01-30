/**
 * ThunderDuck TPC-H Operators V53
 *
 * 基于查询内存基础设施的优化版本:
 * - QueryArena: 查询级 bump allocator
 * - ChunkedDirectArray: 分块直接数组
 * - TypeLiftedColumn: Scan-time 类型提升
 *
 * 目标:
 * - Q3: 1.44x → 2.0x+ (ChunkedDirectArray + BitmapPredicateIndex)
 * - Q5: 1.07x → 2.0x+ (ChunkedDirectArray)
 * - Q6: 1.80x → 3.5x+ (TypeLifted + SIMDBranchlessFilter)
 * - Q9: 1.52x → 2.5x+ (ChunkedDirectArray + BitmapPredicateIndex)
 *
 * @version 53
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_constants.h"
#include "../../include/thunderduck/query_memory.h"
#include "../../include/thunderduck/generic_operators_v52.h"
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v53 {

using namespace memory;
using namespace operators::v52;

// ============================================================================
// Q6: TypeLifted + SIMDBranchlessFilter
// ============================================================================

/**
 * Q6: 预测收入变化 (V53)
 *
 * 优化策略:
 * - TypeLiftedColumn: scan-time 一次性转换 int64→double
 * - SIMDBranchlessFilter: 完全无分支 SIMD 过滤
 * - 8 线程并行
 */
inline void run_q6_v53(TPCHDataLoader& loader) {
    auto total_start = std::chrono::high_resolution_clock::now();

    const auto& lineitem = loader.lineitem();
    size_t n = lineitem.l_orderkey.size();

    // Step 1: 创建查询上下文 (lineitem 144MB)
    QueryContext ctx(192 * 1024 * 1024);  // 192MB

    // Step 2: Type lifting (一次性转换)
    auto lifted = ctx.lift_lineitem(lineitem);

    // Step 3: SIMDBranchlessFilter
    constexpr int32_t DATE_LO = constants::dates::D1994_01_01;
    constexpr int32_t DATE_HI = constants::dates::D1995_01_01;
    constexpr double DISC_LO = 0.05;
    constexpr double DISC_HI = 0.07;
    constexpr double QTY_TH = 24.0;

    SIMDBranchlessFilter::Config config;
    config.num_threads = 8;
    SIMDBranchlessFilter filter(config);

    double result = 0.0;
    auto stats = filter.filter_sum_q6(
        lineitem.l_shipdate.data(),
        lifted.l_discount,
        lifted.l_quantity,
        lifted.l_extendedprice,
        n,
        DATE_LO, DATE_HI,
        DISC_LO, DISC_HI,
        QTY_TH,
        result
    );

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n=== Q6 V53 Results (TypeLifted + SIMDBranchlessFilter) ===\n";
    std::cout << "Revenue: " << std::fixed << result << "\n";
    std::cout << "Matched: " << stats.matched_rows << " / " << stats.total_rows
              << " (" << (100.0 * stats.matched_rows / stats.total_rows) << "%)\n";
    std::cout << "Lift time: " << lifted.convert_time_ms << " ms\n";
    std::cout << "Filter time: " << stats.filter_time_ms << " ms\n";
    std::cout << "Total time: " << total_time_ms << " ms\n";
    std::cout << "Arena usage: " << (ctx.arena().used() / 1024.0 / 1024.0) << " MB\n";
}

// ============================================================================
// Q5: ChunkedDirectArray
// ============================================================================

/**
 * Q5: 本地供应商收入 (V53)
 *
 * 优化策略:
 * - ChunkedDirectArray: 分块 O(1) 索引
 * - QueryArena: 查询级内存
 * - TypeLiftedColumn: 数值列预转换
 */
inline void run_q5_v53(TPCHDataLoader& loader) {
    auto total_start = std::chrono::high_resolution_clock::now();

    const auto& customer = loader.customer();
    const auto& orders = loader.orders();
    const auto& lineitem = loader.lineitem();
    const auto& supplier = loader.supplier();
    const auto& nation = loader.nation();
    const auto& region = loader.region();

    // 创建查询上下文 (lineitem 144MB + ChunkedDirectArray + overhead)
    QueryContext ctx(256 * 1024 * 1024);  // 256MB

    // Step 1: Type lift lineitem 数值列
    auto lifted = ctx.lift_lineitem(lineitem);

    // Step 2: 找到 ASIA regionkey
    int32_t asia_regionkey = -1;
    for (size_t i = 0; i < region.r_regionkey.size(); ++i) {
        if (region.r_name[i] == constants::regions::ASIA) {
            asia_regionkey = region.r_regionkey[i];
            break;
        }
    }

    // Step 3: ASIA 国家映射
    std::unordered_map<int32_t, std::string> asia_nations;
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_regionkey[i] == asia_regionkey) {
            asia_nations[nation.n_nationkey[i]] = nation.n_name[i];
        }
    }

    // Step 4: ChunkedDirectArray - suppkey → nationkey (max 10000)
    ChunkedDirectArray<int8_t, 10001> supp_to_nation(ctx.arena());
    for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
        if (asia_nations.count(supplier.s_nationkey[i])) {
            supp_to_nation.set(supplier.s_suppkey[i],
                              static_cast<int8_t>(supplier.s_nationkey[i]));
        }
    }

    // Step 5: ChunkedDirectArray - custkey → nationkey (max 150000)
    ChunkedDirectArray<int8_t, 150001> cust_to_nation(ctx.arena());
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (asia_nations.count(customer.c_nationkey[i])) {
            cust_to_nation.set(customer.c_custkey[i],
                              static_cast<int8_t>(customer.c_nationkey[i]));
        }
    }

    // Step 6: ChunkedDirectArray - orderkey → custkey (max 6M)
    constexpr int32_t DATE_1994_START = constants::dates::D1994_01_01;
    constexpr int32_t DATE_1995_START = constants::dates::D1995_01_01;

    ChunkedDirectArray<int32_t, 6000001> order_to_cust(ctx.arena());
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] >= DATE_1994_START &&
            orders.o_orderdate[i] < DATE_1995_START) {
            int8_t cust_nation = cust_to_nation.lookup(orders.o_custkey[i]);
            if (cust_nation != -1) {
                order_to_cust.set(orders.o_orderkey[i], orders.o_custkey[i]);
            }
        }
    }

    // Step 7: 扫描 lineitem 并聚合
    std::array<double, 25> nation_revenue{};

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];
        int32_t custkey = order_to_cust.lookup(orderkey);
        if (custkey == -1) continue;

        int32_t suppkey = lineitem.l_suppkey[i];
        int8_t supp_nation = supp_to_nation.lookup(suppkey);
        if (supp_nation == -1) continue;

        int8_t cust_nation = cust_to_nation.lookup(custkey);
        if (cust_nation != supp_nation) continue;

        // 使用预转换的列
        double revenue = lifted.l_extendedprice[i] * (1.0 - lifted.l_discount[i]);
        nation_revenue[supp_nation] += revenue;
    }

    // Step 8: 收集结果
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

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n=== Q5 V53 Results (ChunkedDirectArray) ===\n";
    for (const auto& r : results) {
        std::cout << r.n_name << " | " << std::fixed << r.revenue << "\n";
    }
    std::cout << "Lift time: " << lifted.convert_time_ms << " ms\n";
    std::cout << "Total time: " << total_time_ms << " ms\n";
    std::cout << "Arena usage: " << (ctx.arena().used() / 1024.0 / 1024.0) << " MB\n";
}

// ============================================================================
// Q3: ChunkedDirectArray + BitmapPredicateIndex
// ============================================================================

/**
 * Q3: 运输优先级 (V53)
 *
 * 优化策略:
 * - BitmapPredicateIndex: BUILDING 客户位图
 * - ChunkedDirectArray: orderkey → info 映射
 * - TypeLiftedColumn: 数值列预转换
 */
inline void run_q3_v53(TPCHDataLoader& loader) {
    auto total_start = std::chrono::high_resolution_clock::now();

    const auto& customer = loader.customer();
    const auto& orders = loader.orders();
    const auto& lineitem = loader.lineitem();

    // 创建查询上下文 (lineitem 3 列 * 6M * 8 bytes = 144MB, 加上 ChunkedDirectArray)
    QueryContext ctx(256 * 1024 * 1024);  // 256MB

    // Step 1: Type lift lineitem
    auto lifted = ctx.lift_lineitem(lineitem);

    // Step 2: BUILDING 客户位图
    BitmapPredicateIndex<150001> building_customers;
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (customer.c_mktsegment[i] == "BUILDING") {
            building_customers.set(customer.c_custkey[i]);
        }
    }

    // Step 3: ChunkedDirectArray - orderkey → packed(orderdate, shippriority)
    constexpr int32_t DATE_THRESHOLD = constants::dates::D1995_03_15;

    // 使用 valid_orders 位图快速过滤
    BitmapPredicateIndex<6000001> valid_orders;

    // orderkey → info (64-bit packed: orderdate<<32 | shippriority)
    ChunkedDirectArray<int64_t, 6000001> order_info(ctx.arena());

    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] < DATE_THRESHOLD &&
            building_customers.test(orders.o_custkey[i])) {
            valid_orders.set(orders.o_orderkey[i]);

            int64_t info = (static_cast<int64_t>(orders.o_orderdate[i]) << 32) |
                           static_cast<uint32_t>(orders.o_shippriority[i]);
            order_info.set(orders.o_orderkey[i], info);
        }
    }

    // Step 4: 扫描 lineitem
    std::unordered_map<int32_t, double> order_revenue;

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];

        // 位图快速过滤
        if (!valid_orders.test(orderkey)) continue;

        // l_shipdate > 1995-03-15
        if (lineitem.l_shipdate[i] <= DATE_THRESHOLD) continue;

        double revenue = lifted.l_extendedprice[i] * (1.0 - lifted.l_discount[i]);
        order_revenue[orderkey] += revenue;
    }

    // Step 5: 收集结果
    struct Result {
        int32_t orderkey;
        double revenue;
        int32_t orderdate;
        int32_t shippriority;
    };
    std::vector<Result> results;

    for (const auto& kv : order_revenue) {
        int64_t info = order_info.lookup(kv.first);
        if (info != -1) {
            results.push_back({
                kv.first,
                kv.second,
                static_cast<int32_t>(info >> 32),
                static_cast<int32_t>(info & 0xFFFFFFFF)
            });
        }
    }

    // 排序
    std::partial_sort(results.begin(),
                      results.begin() + std::min(results.size(), size_t(10)),
                      results.end(),
                      [](const Result& a, const Result& b) {
                          if (std::abs(a.revenue - b.revenue) > 0.01)
                              return a.revenue > b.revenue;
                          return a.orderdate < b.orderdate;
                      });

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n=== Q3 V53 Results (ChunkedDirectArray + Bitmap) ===\n";
    size_t limit = std::min(results.size(), size_t(10));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << results[i].orderkey << " | " << std::fixed << results[i].revenue
                  << " | " << results[i].orderdate << " | " << results[i].shippriority << "\n";
    }
    std::cout << "Lift time: " << lifted.convert_time_ms << " ms\n";
    std::cout << "Total time: " << total_time_ms << " ms\n";
    std::cout << "Arena usage: " << (ctx.arena().used() / 1024.0 / 1024.0) << " MB\n";
}

// ============================================================================
// Q9: ChunkedDirectArray + BitmapPredicateIndex
// ============================================================================

/**
 * Q9: 产品类型利润 (V53)
 *
 * 优化策略:
 * - BitmapPredicateIndex: LIKE '%green%' 位图
 * - ChunkedDirectArray: suppkey→nation, orderkey→year
 * - TypeLiftedColumn: 数值列预转换
 */
inline void run_q9_v53(TPCHDataLoader& loader) {
    auto total_start = std::chrono::high_resolution_clock::now();

    const auto& part = loader.part();
    const auto& supplier = loader.supplier();
    const auto& lineitem = loader.lineitem();
    const auto& partsupp = loader.partsupp();
    const auto& orders = loader.orders();
    const auto& nation = loader.nation();

    // 创建查询上下文 (lineitem 144MB + ChunkedDirectArray + overhead)
    QueryContext ctx(256 * 1024 * 1024);  // 256MB

    // Step 1: Type lift lineitem
    auto lifted = ctx.lift_lineitem(lineitem);

    // Step 2: LIKE '%green%' 位图
    BitmapPredicateIndex<200001> green_parts;
    for (size_t i = 0; i < part.p_partkey.size(); ++i) {
        if (string_contains(part.p_name[i], "green")) {
            green_parts.set(part.p_partkey[i]);
        }
    }
    size_t green_count = green_parts.count();

    // Step 3: ChunkedDirectArray - suppkey → nationkey
    ChunkedDirectArray<int8_t, 10001> supp_to_nation(ctx.arena());
    for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
        supp_to_nation.set(supplier.s_suppkey[i],
                          static_cast<int8_t>(supplier.s_nationkey[i]));
    }

    // Step 4: nation 名称数组
    std::array<std::string, 25> nation_names;
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_nationkey[i] < 25) {
            nation_names[nation.n_nationkey[i]] = nation.n_name[i];
        }
    }

    // Step 5: partsupp 哈希表 (green parts only)
    std::unordered_map<uint64_t, double> ps_cost;
    for (size_t i = 0; i < partsupp.ps_partkey.size(); ++i) {
        if (green_parts.test(partsupp.ps_partkey[i])) {
            uint64_t key = (static_cast<uint64_t>(partsupp.ps_suppkey[i]) << 32) |
                           static_cast<uint32_t>(partsupp.ps_partkey[i]);
            ps_cost[key] = partsupp.ps_supplycost[i];
        }
    }

    // Step 6: ChunkedDirectArray - orderkey → year
    ChunkedDirectArray<int16_t, 6000001> order_to_year(ctx.arena());
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        // 从 epoch days 提取年份
        int32_t year = 1970 + orders.o_orderdate[i] / 365;
        order_to_year.set(orders.o_orderkey[i], static_cast<int16_t>(year));
    }

    // Step 7: 扫描 lineitem 并聚合
    std::unordered_map<uint32_t, double> profit_map;

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
        int16_t year = order_to_year.lookup(orderkey);
        if (year == 0) continue;

        // 计算 profit (使用预转换的列)
        double amount = lifted.l_extendedprice[i] * (1.0 - lifted.l_discount[i])
                       - supplycost * lifted.l_quantity[i];

        uint32_t group_key = (static_cast<uint32_t>(nationkey) << 16) | static_cast<uint16_t>(year);
        profit_map[group_key] += amount;
    }

    // Step 8: 收集结果
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

    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) {
                  if (a.nation != b.nation) return a.nation < b.nation;
                  return a.year > b.year;
              });

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n=== Q9 V53 Results (ChunkedDirectArray + Bitmap) ===\n";
    std::cout << "Green parts: " << green_count << " / " << part.p_partkey.size() << "\n";
    size_t limit = std::min(results.size(), size_t(20));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << results[i].nation << " | " << results[i].year
                  << " | " << std::fixed << results[i].profit << "\n";
    }
    std::cout << "Lift time: " << lifted.convert_time_ms << " ms\n";
    std::cout << "Total time: " << total_time_ms << " ms\n";
    std::cout << "Arena usage: " << (ctx.arena().used() / 1024.0 / 1024.0) << " MB\n";
}

} // namespace ops_v53
} // namespace tpch
} // namespace thunderduck
