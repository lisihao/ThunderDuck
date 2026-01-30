/**
 * ThunderDuck TPC-H Operators V51
 *
 * 使用通用算子优化 Q21, Q3/Q5, Q6
 *
 * V51 核心优化:
 * - Q21: ParallelRadixSort (两级基数排序)
 * - Q3/Q5: PartitionedAggregation (分区聚合)
 * - Q6: FusedFilterAggregate (Filter-Aggregate 融合)
 *
 * @version 51
 * @date 2026-01-29
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_constants.h"
#include "../../include/thunderduck/generic_operators.h"
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace thunderduck {
namespace tpch {
namespace ops_v51 {

using namespace operators;

// ============================================================================
// Q21: 延迟供应商 - 使用 ParallelRadixSort
// ============================================================================

/**
 * Q21: 无法按时交付的供应商
 *
 * 优化策略:
 * 1. 按 (l_orderkey, l_suppkey) 排序 LINEITEM
 * 2. 单遍扫描分析 EXISTS / NOT EXISTS
 * 3. 8 线程并行基数排序
 */
inline void run_q21_v51(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& lineitem = loader.lineitem();
    const auto& orders = loader.orders();
    const auto& supplier = loader.supplier();
    const auto& nation = loader.nation();

    // Step 1: 找到 SAUDI ARABIA 的 nationkey
    int32_t saudi_nationkey = -1;
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_name[i] == constants::nations::SAUDI_ARABIA) {
            saudi_nationkey = nation.n_nationkey[i];
            break;
        }
    }

    // Step 2: 找到 SAUDI ARABIA 的供应商
    std::unordered_set<int32_t> saudi_suppliers;
    for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
        if (supplier.s_nationkey[i] == saudi_nationkey) {
            saudi_suppliers.insert(supplier.s_suppkey[i]);
        }
    }

    // Step 3: 找到 o_orderstatus = 'F' 的订单
    std::unordered_set<int32_t> f_orders;
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderstatus[i] == 'F') {
            f_orders.insert(orders.o_orderkey[i]);
        }
    }

    // Step 4: 过滤 LINEITEM - l_receiptdate > l_commitdate 且在 F 订单中
    struct LineRecord {
        int32_t orderkey;
        int32_t suppkey;
        bool late;  // l_receiptdate > l_commitdate
    };
    std::vector<LineRecord> records;
    records.reserve(lineitem.l_orderkey.size() / 10);  // 估计 10% 符合条件

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        if (f_orders.count(lineitem.l_orderkey[i])) {
            records.push_back({
                lineitem.l_orderkey[i],
                lineitem.l_suppkey[i],
                lineitem.l_receiptdate[i] > lineitem.l_commitdate[i]
            });
        }
    }

    // Step 5: 使用 ParallelRadixSort 按 orderkey 排序
    // 创建排序键: orderkey * 100000 + suppkey (复合键)
    std::vector<uint64_t> sort_keys(records.size());
    std::vector<size_t> indices(records.size());

    for (size_t i = 0; i < records.size(); ++i) {
        sort_keys[i] = static_cast<uint64_t>(records[i].orderkey) * 100000 +
                       static_cast<uint64_t>(records[i].suppkey);
        indices[i] = i;
    }

    // 配置基数排序
    ParallelRadixSort<uint64_t, size_t>::Config sort_config;
    sort_config.num_threads = 8;
    sort_config.radix_bits = 8;

    ParallelRadixSort<uint64_t, size_t> sorter(sort_config);
    auto sort_stats = sorter.sort(sort_keys.begin(), sort_keys.end(), indices.begin());

    // 根据排序后的 indices 重排 records
    std::vector<LineRecord> sorted_records(records.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_records[i] = records[indices[i]];
    }

    // Step 6: 单遍扫描分析 EXISTS / NOT EXISTS
    // 对于每个供应商，检查:
    // - EXISTS: 同一订单中有其他供应商
    // - NOT EXISTS: 同一订单中没有其他延迟供应商
    std::unordered_map<int32_t, int32_t> supplier_wait_count;

    size_t i = 0;
    while (i < sorted_records.size()) {
        int32_t current_order = sorted_records[i].orderkey;

        // 收集同一订单的所有记录
        size_t order_start = i;
        while (i < sorted_records.size() && sorted_records[i].orderkey == current_order) {
            ++i;
        }
        size_t order_end = i;

        // 跳过只有一个供应商的订单 (不满足 EXISTS)
        if (order_end - order_start <= 1) continue;

        // 检查是否有其他延迟供应商
        std::unordered_set<int32_t> late_suppliers;
        for (size_t j = order_start; j < order_end; ++j) {
            if (sorted_records[j].late) {
                late_suppliers.insert(sorted_records[j].suppkey);
            }
        }

        // 对于每个延迟的沙特供应商
        for (size_t j = order_start; j < order_end; ++j) {
            int32_t suppkey = sorted_records[j].suppkey;
            if (!sorted_records[j].late) continue;
            if (!saudi_suppliers.count(suppkey)) continue;

            // NOT EXISTS: 没有其他延迟供应商
            bool has_other_late = false;
            for (int32_t other : late_suppliers) {
                if (other != suppkey) {
                    has_other_late = true;
                    break;
                }
            }

            if (!has_other_late) {
                supplier_wait_count[suppkey]++;
            }
        }
    }

    // Step 7: 获取供应商名称并排序
    struct Result {
        std::string s_name;
        int32_t numwait;
    };
    std::vector<Result> results;

    for (const auto& kv : supplier_wait_count) {
        // 查找供应商名称
        for (size_t s = 0; s < supplier.s_suppkey.size(); ++s) {
            if (supplier.s_suppkey[s] == kv.first) {
                results.push_back({supplier.s_name[s], kv.second});
                break;
            }
        }
    }

    // 按 numwait DESC, s_name ASC 排序
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        if (a.numwait != b.numwait) return a.numwait > b.numwait;
        return a.s_name < b.s_name;
    });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 输出 Top 10
    std::cout << "\n=== Q21 V51 Results (ParallelRadixSort) ===\n";
    std::cout << "Sort stats: " << sort_stats.num_passes << " passes, "
              << sort_stats.total_time_ms << " ms\n";
    size_t limit = std::min(results.size(), size_t(10));
    for (size_t r = 0; r < limit; ++r) {
        std::cout << results[r].s_name << " | " << results[r].numwait << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

// ============================================================================
// Q3: 运输优先级 - 使用 PartitionedAggregation
// ============================================================================

inline void run_q3_v51(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& customer = loader.customer();
    const auto& orders = loader.orders();
    const auto& lineitem = loader.lineitem();

    // Step 1: 过滤 BUILDING 客户
    std::unordered_set<int32_t> building_customers;
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (customer.c_mktsegment[i] == "BUILDING") {
            building_customers.insert(customer.c_custkey[i]);
        }
    }

    // Step 2: 过滤订单 (o_orderdate < 1995-03-15 且客户是 BUILDING)
    constexpr int32_t DATE_THRESHOLD = thunderduck::tpch::constants::dates::D1995_03_15;

    std::unordered_map<int32_t, std::pair<int32_t, int32_t>> order_info;  // orderkey -> (orderdate, shippriority)
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] < DATE_THRESHOLD &&
            building_customers.count(orders.o_custkey[i])) {
            order_info[orders.o_orderkey[i]] = {orders.o_orderdate[i], orders.o_shippriority[i]};
        }
    }

    // Step 3: 使用 PartitionedAggregation 聚合 LINEITEM
    // GROUP BY l_orderkey

    // 聚合状态
    struct AggState {
        double revenue;
        int32_t orderdate;
        int32_t shippriority;

        AggState() : revenue(0), orderdate(0), shippriority(0) {}
    };

    // 准备数据
    std::vector<int32_t> keys;
    std::vector<double> values;
    keys.reserve(lineitem.l_orderkey.size() / 5);
    values.reserve(lineitem.l_orderkey.size() / 5);

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];
        auto it = order_info.find(orderkey);
        if (it != order_info.end() && lineitem.l_shipdate[i] > DATE_THRESHOLD) {
            keys.push_back(orderkey);
            double rev = lineitem.l_extendedprice[i] * (1.0 - lineitem.l_discount[i]);
            values.push_back(rev);
        }
    }

    // 配置分区聚合
    PartitionedAggregation<int32_t, double, double>::Config agg_config;
    agg_config.num_threads = 8;
    agg_config.num_partitions = 16;

    PartitionedAggregation<int32_t, double, double> aggregator(
        []() -> double { return 0.0; },                    // init
        [](double& state, const double& val) { state += val; },  // update
        [](double& a, const double& b) { a += b; },        // merge
        nullptr,                                           // finalize
        agg_config
    );

    // 收集结果
    struct Result {
        int32_t orderkey;
        double revenue;
        int32_t orderdate;
        int32_t shippriority;
    };
    std::vector<Result> results;

    auto agg_stats = aggregator.aggregate(
        keys.begin(), keys.end(),
        values.begin(),
        [&results, &order_info](int32_t key, double revenue) {
            auto it = order_info.find(key);
            if (it != order_info.end()) {
                results.push_back({key, revenue, it->second.first, it->second.second});
            }
        }
    );

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

    // 输出
    std::cout << "\n=== Q3 V51 Results (PartitionedAggregation) ===\n";
    std::cout << "Agg stats: " << agg_stats.num_partitions << " partitions, "
              << agg_stats.unique_keys << " keys, " << agg_stats.total_time_ms << " ms\n";
    size_t limit = std::min(results.size(), size_t(10));
    for (size_t i = 0; i < limit; ++i) {
        std::cout << results[i].orderkey << " | " << results[i].revenue
                  << " | " << results[i].orderdate << " | " << results[i].shippriority << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

// ============================================================================
// Q6: 预测收入变化 - 使用 FusedFilterAggregate
// ============================================================================

inline void run_q6_v51(TPCHDataLoader& loader) {
    auto start = std::chrono::high_resolution_clock::now();

    const auto& lineitem = loader.lineitem();
    size_t n = lineitem.l_orderkey.size();

    // 日期范围: 1994-01-01 to 1995-01-01
    constexpr int32_t DATE_LOW = thunderduck::tpch::constants::dates::D1994_01_01;
    constexpr int32_t DATE_HIGH = thunderduck::tpch::constants::dates::D1995_01_01;

    // discount: 0.05 to 0.07
    constexpr double DISC_LOW = 0.05;
    constexpr double DISC_HIGH = 0.07;

    // quantity < 24
    constexpr double QTY_THRESHOLD = 24.0;

    // 使用 FusedFilterAggregate
    FusedFilterAggregate::Config config;
    config.num_threads = 8;
    FusedFilterAggregate ffa(config);

    // 准备数据 (转换为 float)
    std::vector<float> shipdate_f(n);
    std::vector<float> discount_f(n);
    std::vector<float> quantity_f(n);
    std::vector<float> price_f(n);

    for (size_t i = 0; i < n; ++i) {
        shipdate_f[i] = static_cast<float>(lineitem.l_shipdate[i]);
        discount_f[i] = static_cast<float>(lineitem.l_discount[i]);
        quantity_f[i] = static_cast<float>(lineitem.l_quantity[i]);
        price_f[i] = static_cast<float>(lineitem.l_extendedprice[i]);
    }

    // 执行融合 Filter + SUM
    // WHERE l_shipdate >= DATE_LOW AND l_shipdate < DATE_HIGH
    //   AND l_discount >= DISC_LOW AND l_discount <= DISC_HIGH
    //   AND l_quantity < QTY_THRESHOLD
    // SELECT SUM(l_extendedprice * l_discount)
    float result = 0;
    auto stats = ffa.fused_filter_sum(
        shipdate_f.data(),           // filter_col
        discount_f.data(),           // cond2_col
        quantity_f.data(),           // cond3_col
        price_f.data(),              // value_col
        discount_f.data(),           // multiplier_col (l_discount)
        n,
        static_cast<float>(DATE_LOW), static_cast<float>(DATE_HIGH),
        static_cast<float>(DISC_LOW), static_cast<float>(DISC_HIGH),
        static_cast<float>(QTY_THRESHOLD),
        result
    );

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 输出
    std::cout << "\n=== Q6 V51 Results (FusedFilterAggregate) ===\n";
    std::cout << "Revenue: " << std::fixed << result << "\n";
    std::cout << "Matched rows: " << stats.matched_rows << " / " << stats.total_rows << "\n";
    std::cout << "Time: " << time_ms << " ms (filter+agg fused: " << stats.total_time_ms << " ms)\n";
}

// ============================================================================
// Q5: 本地供应商收入 - 使用 PartitionedAggregation
// ============================================================================

inline void run_q5_v51(TPCHDataLoader& loader) {
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

    // Step 2: 找到 ASIA 国家
    std::unordered_map<int32_t, std::string> asia_nations;  // nationkey -> name
    for (size_t i = 0; i < nation.n_nationkey.size(); ++i) {
        if (nation.n_regionkey[i] == asia_regionkey) {
            asia_nations[nation.n_nationkey[i]] = nation.n_name[i];
        }
    }

    // Step 3: ASIA 供应商
    std::unordered_map<int32_t, int32_t> supplier_nation;  // suppkey -> nationkey
    for (size_t i = 0; i < supplier.s_suppkey.size(); ++i) {
        if (asia_nations.count(supplier.s_nationkey[i])) {
            supplier_nation[supplier.s_suppkey[i]] = supplier.s_nationkey[i];
        }
    }

    // Step 4: ASIA 客户
    std::unordered_map<int32_t, int32_t> customer_nation;  // custkey -> nationkey
    for (size_t i = 0; i < customer.c_custkey.size(); ++i) {
        if (asia_nations.count(customer.c_nationkey[i])) {
            customer_nation[customer.c_custkey[i]] = customer.c_nationkey[i];
        }
    }

    // Step 5: 过滤订单 (1994 年)
    constexpr int32_t DATE_1994_START = thunderduck::tpch::constants::dates::D1994_01_01;
    constexpr int32_t DATE_1995_START = thunderduck::tpch::constants::dates::D1995_01_01;

    std::unordered_map<int32_t, int32_t> order_customer;  // orderkey -> custkey
    for (size_t i = 0; i < orders.o_orderkey.size(); ++i) {
        if (orders.o_orderdate[i] >= DATE_1994_START &&
            orders.o_orderdate[i] < DATE_1995_START &&
            customer_nation.count(orders.o_custkey[i])) {
            order_customer[orders.o_orderkey[i]] = orders.o_custkey[i];
        }
    }

    // Step 6: 使用 PartitionedAggregation 按 nationkey 聚合
    std::vector<int32_t> keys;
    std::vector<double> values;
    keys.reserve(lineitem.l_orderkey.size() / 20);
    values.reserve(lineitem.l_orderkey.size() / 20);

    for (size_t i = 0; i < lineitem.l_orderkey.size(); ++i) {
        int32_t orderkey = lineitem.l_orderkey[i];
        auto oit = order_customer.find(orderkey);
        if (oit == order_customer.end()) continue;

        int32_t suppkey = lineitem.l_suppkey[i];
        auto sit = supplier_nation.find(suppkey);
        if (sit == supplier_nation.end()) continue;

        // c_nationkey = s_nationkey
        int32_t cust_nation = customer_nation[oit->second];
        if (cust_nation != sit->second) continue;

        keys.push_back(cust_nation);
        values.push_back(lineitem.l_extendedprice[i] * (1.0 - lineitem.l_discount[i]));
    }

    // 分区聚合
    PartitionedAggregation<int32_t, double, double>::Config agg_config;
    agg_config.num_threads = 8;
    agg_config.num_partitions = 16;

    PartitionedAggregation<int32_t, double, double> aggregator(
        []() -> double { return 0.0; },
        [](double& state, const double& val) { state += val; },
        [](double& a, const double& b) { a += b; },
        nullptr,
        agg_config
    );

    struct Result {
        std::string n_name;
        double revenue;
    };
    std::vector<Result> results;

    aggregator.aggregate(
        keys.begin(), keys.end(),
        values.begin(),
        [&results, &asia_nations](int32_t key, double revenue) {
            auto it = asia_nations.find(key);
            if (it != asia_nations.end()) {
                results.push_back({it->second, revenue});
            }
        }
    );

    // 按 revenue DESC 排序
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.revenue > b.revenue; });

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 输出
    std::cout << "\n=== Q5 V51 Results (PartitionedAggregation) ===\n";
    for (const auto& r : results) {
        std::cout << r.n_name << " | " << r.revenue << "\n";
    }
    std::cout << "Time: " << time_ms << " ms\n";
}

} // namespace ops_v51
} // namespace tpch
} // namespace thunderduck
