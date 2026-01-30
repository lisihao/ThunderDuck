/**
 * ThunderDuck TPC-H 查询实现
 *
 * V24 优化:
 * - P0: 选择向量替换中间 vector
 * - P1: 数组替换 hash 表
 * - P2: Filter + Join 融合
 */

#include "tpch_queries.h"
#include "tpch_query_optimizer.h"
#include "tpch_operators_v24.h"
#include "tpch_operators_v25.h"
#include "tpch_operators_v26.h"
#include "tpch_operators_v27.h"
#include "tpch_operators_v32.h"
#include "tpch_operators_v33.h"
#include "tpch_operators_v34.h"
#include "tpch_operators_v35.h"
#include "tpch_operators_v36.h"
#include "tpch_operators_v37.h"
#include "tpch_operators_v38.h"
#include "tpch_operators_v39.h"
#include "tpch_operators_v40.h"
#include "tpch_operators_v41.h"
#include "tpch_operators_v42.h"
#include "tpch_operators_v43.h"
#include "tpch_operators_v44.h"
#include "tpch_operators_v45.h"
#include "tpch_operators_v46.h"
#include "tpch_operators_v47.h"
#include "tpch_operators_v48.h"
#include "tpch_operators_v49.h"
#include "tpch_constants.h"      // 统一常量定义
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <thread>
#include <array>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace queries {

// ============================================================================
// 查询注册表
// ============================================================================

static std::map<std::string, QueryImplFunc> query_registry;

// V24 优化版本包装函数
static void run_q3_v24_wrapper(TPCHDataLoader& loader) {
    ops_v24::run_q3_v24(loader);
}

static void run_q5_v24_wrapper(TPCHDataLoader& loader) {
    ops_v24::run_q5_v24(loader);
}

static void run_q6_v24_wrapper(TPCHDataLoader& loader) {
    ops_v24::run_q6_v24(loader);
}

static void run_q9_v24_wrapper(TPCHDataLoader& loader) {
    ops_v24::run_q9_v24(loader);
}

// V25 优化版本包装函数 (线程池 + Hash 缓存)
static void run_q3_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q3_v25(loader);
}

static void run_q5_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q5_v25(loader);
}

static void run_q6_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q6_v25(loader);
}

static void run_q9_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q9_v25(loader);
}

static void run_q7_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q7_v25(loader);
}

static void run_q10_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q10_v25(loader);
}

static void run_q12_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q12_v25(loader);
}

static void run_q14_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q14_v25(loader);
}

static void run_q18_v25_wrapper(TPCHDataLoader& loader) {
    ops_v25::run_q18_v25(loader);
}

// V27 优化版本包装函数 (P0 优先级)
static void run_q3_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q3_v27(loader);
}

// V28 优化版本包装函数 (Q3 深度优化)
static void run_q3_v28_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q3_v28(loader);
}

// V31 优化版本包装函数 (Q3 最优)
static void run_q3_v31_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q3_v31(loader);
}

static void run_q4_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q4_v27(loader);
}

static void run_q5_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q5_v27(loader);
}

static void run_q7_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q7_v27(loader);
}

static void run_q9_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q9_v27(loader);
}

static void run_q10_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q10_v27(loader);
}

static void run_q11_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q11_v27(loader);
}

static void run_q12_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q12_v27(loader);
}

static void run_q14_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q14_v27(loader);
}

static void run_q15_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q15_v27(loader);
}

static void run_q16_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q16_v27(loader);
}

static void run_q18_v27_wrapper(TPCHDataLoader& loader) {
    ops_v27::run_q18_v27(loader);
}

// V32 优化版本包装函数 (紧凑 Hash Table + Bloom Filter)
static void run_q5_v32_wrapper(TPCHDataLoader& loader) {
    ops_v32::run_q5_v32(loader);
}

static void run_q7_v32_wrapper(TPCHDataLoader& loader) {
    ops_v32::run_q7_v32(loader);
}

static void run_q9_v32_wrapper(TPCHDataLoader& loader) {
    ops_v32::run_q9_v32(loader);
}

static void run_q18_v32_wrapper(TPCHDataLoader& loader) {
    ops_v32::run_q18_v32(loader);
}

static void run_q19_v32_wrapper(TPCHDataLoader& loader) {
    ops_v32::run_q19_v32(loader);
}

// V33 优化版本包装函数 (通用化架构 - 无硬编码)
static void run_q5_v33_wrapper(TPCHDataLoader& loader) {
    ops_v33::run_q5_v33(loader);
}

static void run_q7_v33_wrapper(TPCHDataLoader& loader) {
    ops_v33::run_q7_v33(loader);
}

static void run_q9_v33_wrapper(TPCHDataLoader& loader) {
    ops_v33::run_q9_v33(loader);
}

static void run_q18_v33_wrapper(TPCHDataLoader& loader) {
    ops_v33::run_q18_v33(loader);
}

static void run_q19_v33_wrapper(TPCHDataLoader& loader) {
    ops_v33::run_q19_v33(loader);
}

// V34 优化版本包装函数 (继续攻坚 - Q22/Q13/Q8)
static void run_q22_v34_wrapper(TPCHDataLoader& loader) {
    ops_v34::run_q22_v34(loader);
}

static void run_q13_v34_wrapper(TPCHDataLoader& loader) {
    ops_v34::run_q13_v34(loader);
}

static void run_q8_v34_wrapper(TPCHDataLoader& loader) {
    ops_v34::run_q8_v34(loader);
}

// V35 优化版本包装函数 (通用化算子)
static void run_q3_v35_wrapper(TPCHDataLoader& loader) {
    ops_v35::run_q3_v35(loader);
}

static void run_q8_v35_wrapper(TPCHDataLoader& loader) {
    ops_v35::run_q8_v35(loader);
}

static void run_q14_v35_wrapper(TPCHDataLoader& loader) {
    ops_v35::run_q14_v35(loader);
}

static void run_q22_v35_wrapper(TPCHDataLoader& loader) {
    ops_v35::run_q22_v35(loader);
}

static void run_q21_v35_wrapper(TPCHDataLoader& loader) {
    ops_v35::run_q21_v35(loader);
}

// V36 优化版本包装函数 (相关子查询解关联)
static void run_q17_v36_wrapper(TPCHDataLoader& loader) {
    ops_v36::run_q17_v36(loader);
}

static void run_q20_v36_wrapper(TPCHDataLoader& loader) {
    ops_v36::run_q20_v36(loader);
}

// V37 优化版本包装函数 (Bitmap Anti-Join, OrderKeyState 预计算)
static void run_q22_v37_wrapper(TPCHDataLoader& loader) {
    ops_v37::run_q22_v37(loader);
}

static void run_q21_v37_wrapper(TPCHDataLoader& loader) {
    ops_v37::run_q21_v37(loader);
}

static void run_q20_v37_wrapper(TPCHDataLoader& loader) {
    ops_v37::run_q20_v37(loader);
}

static void run_q17_v37_wrapper(TPCHDataLoader& loader) {
    ops_v37::run_q17_v37(loader);
}

static void run_q8_v37_wrapper(TPCHDataLoader& loader) {
    ops_v37::run_q8_v37(loader);
}

// V38 优化版本包装函数 (排序去重 + 紧凑编码)
static void run_q21_v38_wrapper(TPCHDataLoader& loader) {
    ops_v38::run_q21_v38(loader);
}

static void run_q20_v38_wrapper(TPCHDataLoader& loader) {
    ops_v38::run_q20_v38(loader);
}

// V39 优化版本包装函数 (纯排序方案)
static void run_q21_v39_wrapper(TPCHDataLoader& loader) {
    ops_v39::run_q21_v39(loader);
}

static void run_q20_v39_wrapper(TPCHDataLoader& loader) {
    ops_v39::run_q20_v39(loader);
}

// V40 优化版本包装函数 (通用算子框架)
static void run_q20_v40_wrapper(TPCHDataLoader& loader) {
    ops_v40::run_q20_v40(loader);
}

// V41 优化版本包装函数 (Q21 单遍预计算)
static void run_q21_v41_wrapper(TPCHDataLoader& loader) {
    ops_v41::run_q21_v41(loader);
}

// V42 优化版本包装函数 (Q8 并行化)
static void run_q8_v42_wrapper(TPCHDataLoader& loader) {
    ops_v42::run_q8_v42(loader);
}

// V43 优化版本包装函数 (Q17 位图过滤 + 两阶段聚合)
static void run_q17_v43_wrapper(TPCHDataLoader& loader) {
    ops_v43::run_q17_v43(loader);
}

// V44 优化版本包装函数 (Q3 直接数组访问 + 线程局部聚合)
static void run_q3_v44_wrapper(TPCHDataLoader& loader) {
    ops_v44::run_q3_v44(loader);
}

// V45 优化版本包装函数 (直接数组优化)
static void run_q14_v45_wrapper(TPCHDataLoader& loader) {
    ops_v45::run_q14_v45(loader);
}

static void run_q11_v45_wrapper(TPCHDataLoader& loader) {
    ops_v45::run_q11_v45(loader);
}

static void run_q5_v45_wrapper(TPCHDataLoader& loader) {
    ops_v45::run_q5_v45(loader);
}

// V46 通用化版本包装函数 (消除硬编码)
static void run_q14_v46_wrapper(TPCHDataLoader& loader) {
    ops_v46::run_q14_v46(loader);
}

static void run_q11_v46_wrapper(TPCHDataLoader& loader) {
    ops_v46::run_q11_v46(loader);
}

static void run_q5_v46_wrapper(TPCHDataLoader& loader) {
    ops_v46::run_q5_v46(loader);
}

// V47 通用算子框架版本包装函数
static void run_q6_v47_wrapper(TPCHDataLoader& loader) {
    ops_v47::run_q6_v47(loader);
}

static void run_q13_v47_wrapper(TPCHDataLoader& loader) {
    ops_v47::run_q13_v47(loader);
}

static void run_q21_v47_wrapper(TPCHDataLoader& loader) {
    ops_v47::run_q21_v47(loader);
}

// V48: Q21 正确实现 - Group-then-Filter (非 JOIN/EXISTS)
static void run_q21_v48_wrapper(TPCHDataLoader& loader) {
    ops_v48::run_q21_v48(loader);
}

// V49: Q3 Top-N Aware Partial Aggregation
static void run_q3_v49_wrapper(TPCHDataLoader& loader) {
    ops_v49::run_q3_v49(loader);
}

void register_all_queries() {
    // 初始化 TPC-H 查询优化器配置
    register_tpch_query_configs();
    auto& optimizer = TPCHQueryOptimizer::instance();

    // 动态注册所有查询 - 优化器自动选择最佳版本
    for (const auto& query_id : optimizer.get_all_query_ids()) {
        auto executor = optimizer.select_best(query_id);
        if (executor) {
            query_registry[query_id] = executor;
        }
    }

    // 打印注册信息 (调试)
    // optimizer.print_registry();
}

QueryImplFunc get_query_impl(const std::string& query_id) {
    auto it = query_registry.find(query_id);
    if (it != query_registry.end()) {
        return it->second;
    }
    return nullptr;
}

bool has_optimized_impl(const std::string& query_id) {
    return query_registry.count(query_id) > 0;
}

std::string get_selected_version(const std::string& query_id) {
    auto& optimizer = TPCHQueryOptimizer::instance();
    auto config = optimizer.get_config(query_id);
    if (config) {
        return optimizer.get_selected_version(query_id, config->estimated_rows);
    }
    return "Base";
}

// ============================================================================
// Q1: 定价汇总报告 - GROUP BY + 多聚合
// ============================================================================

void run_q1(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    size_t n = li.count;

    // 日期阈值: 1998-12-01 - 90 days = 1998-09-02 = epoch day 10471
    constexpr int32_t date_threshold = dates::Q1_THRESHOLD;

    // 结果存储: key = (returnflag << 8) | linestatus
    std::unordered_map<int16_t, ops::Q1AggResult> results;

    // 调用优化的分组聚合
    ops::q1_group_aggregate(
        li.l_quantity.data(),
        li.l_extendedprice.data(),
        li.l_discount.data(),
        li.l_tax.data(),
        li.l_returnflag.data(),
        li.l_linestatus.data(),
        li.l_shipdate.data(),
        date_threshold,
        n,
        results
    );

    // 结果已在 results 中
    // 实际应用中会输出或验证结果
}

// ============================================================================
// Q3: 运输优先级 - V22 优化: SEMI JOIN GPU + INNER JOIN V19.2
// ============================================================================

void run_q3(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // 日期阈值: 1995-03-15
    constexpr int32_t date_threshold = dates::Q3_DATE;

    // Step 1: 过滤 BUILDING 客户，提取 custkey 数组
    std::vector<int32_t> building_custkeys;
    building_custkeys.reserve(cust.count / 5);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_mktsegment_code[i] == 1) {  // BUILDING
            building_custkeys.push_back(cust.c_custkey[i]);
        }
    }

    // Step 2: SEMI JOIN GPU - orders.o_custkey ∈ building_custkeys
    // 同时过滤 o_orderdate < threshold
    std::vector<uint32_t> orders_semi_matches;
    ops::semi_join_i32(
        building_custkeys.data(), building_custkeys.size(),
        ord.o_custkey.data(), ord.count,
        orders_semi_matches
    );

    // 过滤日期并构建 valid_orderkeys 数组
    std::vector<int32_t> valid_orderkeys;
    std::vector<int32_t> valid_orderdates;
    std::vector<int32_t> valid_shippriorities;
    valid_orderkeys.reserve(orders_semi_matches.size());

    for (uint32_t idx : orders_semi_matches) {
        if (ord.o_orderdate[idx] < date_threshold) {
            valid_orderkeys.push_back(ord.o_orderkey[idx]);
            valid_orderdates.push_back(ord.o_orderdate[idx]);
            valid_shippriorities.push_back(ord.o_shippriority[idx]);
        }
    }

    // Step 3: 过滤 lineitem (l_shipdate > threshold) 并提取 orderkey
    std::vector<int32_t> li_orderkeys;
    std::vector<uint32_t> li_indices;
    li_orderkeys.reserve(li.count / 2);
    li_indices.reserve(li.count / 2);

    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_shipdate[i] > date_threshold) {
            li_orderkeys.push_back(li.l_orderkey[i]);
            li_indices.push_back(static_cast<uint32_t>(i));
        }
    }

    // Step 4: INNER JOIN V19.2 - valid_orderkeys × li_orderkeys
    ops::JoinPairs join_result;
    ops::inner_join_i32(
        valid_orderkeys.data(), valid_orderkeys.size(),
        li_orderkeys.data(), li_orderkeys.size(),
        join_result
    );

    // Step 5: 8 线程并行聚合
    struct Q3Result {
        int64_t revenue = 0;
        int32_t o_orderdate = 0;
        int32_t o_shippriority = 0;
    };

    constexpr size_t NUM_THREADS = 8;
    std::array<std::unordered_map<int32_t, Q3Result>, NUM_THREADS> thread_results;
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (join_result.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, join_result.count);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_results = thread_results[t];

            for (size_t j = start; j < end; ++j) {
                uint32_t ord_idx = join_result.left_indices[j];
                uint32_t li_local_idx = join_result.right_indices[j];
                uint32_t li_idx = li_indices[li_local_idx];

                int32_t orderkey = valid_orderkeys[ord_idx];
                auto& r = local_results[orderkey];

                __int128 val = (__int128)li.l_extendedprice[li_idx] *
                               (10000 - li.l_discount[li_idx]) / 10000;
                r.revenue += val;
                r.o_orderdate = valid_orderdates[ord_idx];
                r.o_shippriority = valid_shippriorities[ord_idx];
            }
        });
    }

    // 等待并合并结果
    for (auto& th : threads) { th.join(); }

    std::unordered_map<int32_t, Q3Result> results;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (auto& [key, val] : thread_results[t]) {
            auto& r = results[key];
            r.revenue += val.revenue;
            r.o_orderdate = val.o_orderdate;
            r.o_shippriority = val.o_shippriority;
        }
    }
}

// ============================================================================
// Q5: 本地供应商收入 - V22 优化: SEMI JOIN + INNER JOIN V19.2
// ============================================================================

void run_q5(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // 日期范围: 1994-01-01 to 1995-01-01
    constexpr int32_t date_lo = dates::DATE_1994_01_01;
    constexpr int32_t date_hi = dates::DATE_1995_01_01;

    // Step 1: 找到 ASIA region 的 nations (小表，直接遍历)
    std::vector<int32_t> asia_nation_keys;
    std::unordered_set<int32_t> asia_nation_set;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::ASIA) {
            int32_t asia_regionkey = reg.r_regionkey[i];
            for (size_t j = 0; j < nat.count; ++j) {
                if (nat.n_regionkey[j] == asia_regionkey) {
                    asia_nation_keys.push_back(nat.n_nationkey[j]);
                    asia_nation_set.insert(nat.n_nationkey[j]);
                }
            }
            break;
        }
    }

    // Step 2: 过滤 ASIA 供应商，建立 suppkey -> nationkey 映射
    std::vector<int32_t> asia_supp_keys;
    std::vector<int32_t> asia_supp_nations;
    asia_supp_keys.reserve(supp.count / 5);
    asia_supp_nations.reserve(supp.count / 5);
    for (size_t i = 0; i < supp.count; ++i) {
        if (asia_nation_set.count(supp.s_nationkey[i])) {
            asia_supp_keys.push_back(supp.s_suppkey[i]);
            asia_supp_nations.push_back(supp.s_nationkey[i]);
        }
    }

    // Step 3: 过滤 ASIA 客户，建立 custkey -> nationkey 映射
    std::vector<int32_t> asia_cust_keys;
    std::vector<int32_t> asia_cust_nations;
    asia_cust_keys.reserve(cust.count / 5);
    asia_cust_nations.reserve(cust.count / 5);
    for (size_t i = 0; i < cust.count; ++i) {
        if (asia_nation_set.count(cust.c_nationkey[i])) {
            asia_cust_keys.push_back(cust.c_custkey[i]);
            asia_cust_nations.push_back(cust.c_nationkey[i]);
        }
    }

    // Step 4: SEMI JOIN - orders.o_custkey ∈ asia_cust_keys，同时过滤日期
    std::vector<uint32_t> orders_semi_matches;
    ops::semi_join_i32(
        asia_cust_keys.data(), asia_cust_keys.size(),
        ord.o_custkey.data(), ord.count,
        orders_semi_matches
    );

    // 过滤日期并建立 orderkey -> custkey 映射
    std::vector<int32_t> valid_orderkeys;
    std::vector<int32_t> valid_order_custkeys;
    valid_orderkeys.reserve(orders_semi_matches.size());

    // 建立 custkey -> nationkey 查找表
    std::unordered_map<int32_t, int32_t> cust_to_nation;
    for (size_t i = 0; i < asia_cust_keys.size(); ++i) {
        cust_to_nation[asia_cust_keys[i]] = asia_cust_nations[i];
    }

    for (uint32_t idx : orders_semi_matches) {
        if (ord.o_orderdate[idx] >= date_lo && ord.o_orderdate[idx] < date_hi) {
            valid_orderkeys.push_back(ord.o_orderkey[idx]);
            valid_order_custkeys.push_back(ord.o_custkey[idx]);
        }
    }

    // Step 5: INNER JOIN V19.2 - lineitem.l_orderkey × valid_orderkeys
    ops::JoinPairs join_result;
    ops::inner_join_i32(
        valid_orderkeys.data(), valid_orderkeys.size(),
        li.l_orderkey.data(), li.count,
        join_result
    );

    // Step 6: 建立 suppkey -> nationkey 查找表
    std::unordered_map<int32_t, int32_t> supp_to_nation;
    for (size_t i = 0; i < asia_supp_keys.size(); ++i) {
        supp_to_nation[asia_supp_keys[i]] = asia_supp_nations[i];
    }

    // Step 7: 8 线程并行聚合 (检查 customer 和 supplier 在同一国家)
    constexpr size_t NUM_THREADS = 8;
    std::array<std::unordered_map<int32_t, int64_t>, NUM_THREADS> thread_revenues;
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (join_result.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, join_result.count);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_revenue = thread_revenues[t];

            for (size_t j = start; j < end; ++j) {
                uint32_t ord_local_idx = join_result.left_indices[j];
                uint32_t li_idx = join_result.right_indices[j];

                // 获取 supplier nationkey
                int32_t suppkey = li.l_suppkey[li_idx];
                auto supp_it = supp_to_nation.find(suppkey);
                if (supp_it == supp_to_nation.end()) continue;
                int32_t supp_nat = supp_it->second;

                // 获取 customer nationkey
                int32_t custkey = valid_order_custkeys[ord_local_idx];
                auto cust_it = cust_to_nation.find(custkey);
                if (cust_it == cust_to_nation.end()) continue;
                int32_t cust_nat = cust_it->second;

                // 检查同一国家
                if (cust_nat == supp_nat) {
                    __int128 revenue = (__int128)li.l_extendedprice[li_idx] *
                                       (10000 - li.l_discount[li_idx]) / 10000;
                    local_revenue[cust_nat] += revenue;
                }
            }
        });
    }

    // 等待并合并结果
    for (auto& th : threads) { th.join(); }

    std::unordered_map<int32_t, int64_t> nation_revenue;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (auto& [key, val] : thread_revenues[t]) {
            nation_revenue[key] += val;
        }
    }
}

// ============================================================================
// Q6: 预测收入变化 - SIMD 单遍过滤聚合 (最佳优化场景)
// ============================================================================

void run_q6(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    size_t n = li.count;

    // 参数
    constexpr int32_t date_lo = dates::Q6_DATE_LO;  // 1994-01-01
    constexpr int32_t date_hi = dates::Q6_DATE_HI;  // 1995-01-01
    constexpr int64_t disc_lo = 500;   // 0.05 * 10000
    constexpr int64_t disc_hi = 700;   // 0.07 * 10000
    constexpr int64_t qty_hi = 240000; // 24 * 10000

    // V20.1 优化: 8 线程并行 Filter + SUM
    constexpr size_t NUM_THREADS = 8;
    std::array<int64_t, NUM_THREADS> partial_sums = {};
    std::array<std::thread, NUM_THREADS> threads;

    const int32_t* shipdate = li.l_shipdate.data();
    const int64_t* discount = li.l_discount.data();
    const int64_t* quantity = li.l_quantity.data();
    const int64_t* extprice = li.l_extendedprice.data();

    size_t chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);

        threads[t] = std::thread([&, t, start, end]() {
            int64_t local_sum = 0;

            // 8 路展开
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                __builtin_prefetch(shipdate + i + 64, 0, 3);
                __builtin_prefetch(discount + i + 64, 0, 3);

                #define CHECK_AND_ADD(idx) \
                    if (shipdate[i+idx] >= date_lo && shipdate[i+idx] < date_hi && \
                        discount[i+idx] >= disc_lo && discount[i+idx] <= disc_hi && \
                        quantity[i+idx] < qty_hi) { \
                        local_sum += extprice[i+idx] * discount[i+idx]; \
                    }

                CHECK_AND_ADD(0); CHECK_AND_ADD(1);
                CHECK_AND_ADD(2); CHECK_AND_ADD(3);
                CHECK_AND_ADD(4); CHECK_AND_ADD(5);
                CHECK_AND_ADD(6); CHECK_AND_ADD(7);

                #undef CHECK_AND_ADD
            }

            // 处理剩余
            for (; i < end; ++i) {
                if (shipdate[i] >= date_lo && shipdate[i] < date_hi &&
                    discount[i] >= disc_lo && discount[i] <= disc_hi &&
                    quantity[i] < qty_hi) {
                    local_sum += extprice[i] * discount[i];
                }
            }

            partial_sums[t] = local_sum;
        });
    }

    // 等待所有线程完成并合并结果
    for (auto& th : threads) {
        th.join();
    }

    __int128 revenue = 0;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        revenue += partial_sums[t];
    }

    // 转换为正确的定点数格式
    volatile double result = static_cast<double>(revenue) / 100000000.0;
    (void)result;
}

// ============================================================================
// Q7: 体量运输 - V22 优化: SEMI JOIN + INNER JOIN V19.2
// ============================================================================

void run_q7(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // 日期范围: 1995-01-01 to 1996-12-31
    constexpr int32_t date_lo = dates::DATE_1995_01_01;
    constexpr int32_t date_hi = dates::DATE_1996_12_31;

    // 找到 FRANCE 和 GERMANY 的 nationkey
    int32_t france_key = -1, germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::FRANCE) france_key = nat.n_nationkey[i];
        if (nat.n_name[i] == nations::GERMANY) germany_key = nat.n_nationkey[i];
    }

    // Step 1: 过滤 FRANCE/GERMANY 供应商
    std::vector<int32_t> fg_supp_keys;
    std::vector<int32_t> fg_supp_nations;
    fg_supp_keys.reserve(supp.count / 12);
    fg_supp_nations.reserve(supp.count / 12);
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_nationkey[i] == france_key || supp.s_nationkey[i] == germany_key) {
            fg_supp_keys.push_back(supp.s_suppkey[i]);
            fg_supp_nations.push_back(supp.s_nationkey[i]);
        }
    }

    // Step 2: 过滤 FRANCE/GERMANY 客户
    std::vector<int32_t> fg_cust_keys;
    std::vector<int32_t> fg_cust_nations;
    fg_cust_keys.reserve(cust.count / 12);
    fg_cust_nations.reserve(cust.count / 12);
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_nationkey[i] == france_key || cust.c_nationkey[i] == germany_key) {
            fg_cust_keys.push_back(cust.c_custkey[i]);
            fg_cust_nations.push_back(cust.c_nationkey[i]);
        }
    }

    // Step 3: SEMI JOIN - orders.o_custkey ∈ fg_cust_keys
    std::vector<uint32_t> orders_semi_matches;
    ops::semi_join_i32(
        fg_cust_keys.data(), fg_cust_keys.size(),
        ord.o_custkey.data(), ord.count,
        orders_semi_matches
    );

    // 建立 orderkey -> custkey 映射
    std::vector<int32_t> valid_orderkeys;
    std::vector<int32_t> valid_order_custkeys;
    valid_orderkeys.reserve(orders_semi_matches.size());
    valid_order_custkeys.reserve(orders_semi_matches.size());

    for (uint32_t idx : orders_semi_matches) {
        valid_orderkeys.push_back(ord.o_orderkey[idx]);
        valid_order_custkeys.push_back(ord.o_custkey[idx]);
    }

    // Step 4: 过滤 lineitem (日期范围) 并提取
    std::vector<int32_t> li_orderkeys;
    std::vector<uint32_t> li_indices;
    li_orderkeys.reserve(li.count / 3);
    li_indices.reserve(li.count / 3);

    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_shipdate[i] >= date_lo && li.l_shipdate[i] <= date_hi) {
            li_orderkeys.push_back(li.l_orderkey[i]);
            li_indices.push_back(static_cast<uint32_t>(i));
        }
    }

    // Step 5: INNER JOIN V19.2 - valid_orderkeys × li_orderkeys
    ops::JoinPairs join_result;
    ops::inner_join_i32(
        valid_orderkeys.data(), valid_orderkeys.size(),
        li_orderkeys.data(), li_orderkeys.size(),
        join_result
    );

    // Step 6: 建立查找表
    std::unordered_map<int32_t, int32_t> supp_to_nation;
    for (size_t i = 0; i < fg_supp_keys.size(); ++i) {
        supp_to_nation[fg_supp_keys[i]] = fg_supp_nations[i];
    }

    std::unordered_map<int32_t, int32_t> cust_to_nation;
    for (size_t i = 0; i < fg_cust_keys.size(); ++i) {
        cust_to_nation[fg_cust_keys[i]] = fg_cust_nations[i];
    }

    // Step 7: 聚合结构
    struct Q7Key {
        int32_t supp_nation;
        int32_t cust_nation;
        int32_t year;

        bool operator==(const Q7Key& o) const {
            return supp_nation == o.supp_nation &&
                   cust_nation == o.cust_nation &&
                   year == o.year;
        }
    };

    struct Q7KeyHash {
        size_t operator()(const Q7Key& k) const {
            return std::hash<int64_t>()(
                (static_cast<int64_t>(k.supp_nation) << 32) |
                (static_cast<int64_t>(k.cust_nation) << 16) |
                k.year
            );
        }
    };

    // Step 8: 8 线程并行聚合
    constexpr size_t NUM_THREADS = 8;
    std::array<std::unordered_map<Q7Key, int64_t, Q7KeyHash>, NUM_THREADS> thread_results;
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (join_result.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, join_result.count);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_results = thread_results[t];

            for (size_t j = start; j < end; ++j) {
                uint32_t ord_local_idx = join_result.left_indices[j];
                uint32_t li_local_idx = join_result.right_indices[j];
                uint32_t li_idx = li_indices[li_local_idx];

                // 获取 supplier nationkey
                int32_t suppkey = li.l_suppkey[li_idx];
                auto supp_it = supp_to_nation.find(suppkey);
                if (supp_it == supp_to_nation.end()) continue;
                int32_t s_nat = supp_it->second;

                // 获取 customer nationkey
                int32_t custkey = valid_order_custkeys[ord_local_idx];
                auto cust_it = cust_to_nation.find(custkey);
                if (cust_it == cust_to_nation.end()) continue;
                int32_t c_nat = cust_it->second;

                // 检查 (FRANCE, GERMANY) 或 (GERMANY, FRANCE)
                if ((s_nat == france_key && c_nat == germany_key) ||
                    (s_nat == germany_key && c_nat == france_key)) {

                    int32_t year = 1970 + li.l_shipdate[li_idx] / 365;
                    Q7Key key{s_nat, c_nat, year};
                    __int128 volume = (__int128)li.l_extendedprice[li_idx] *
                                      (10000 - li.l_discount[li_idx]) / 10000;
                    local_results[key] += volume;
                }
            }
        });
    }

    // 等待并合并结果
    for (auto& th : threads) { th.join(); }

    std::unordered_map<Q7Key, int64_t, Q7KeyHash> results;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (auto& [key, val] : thread_results[t]) {
            results[key] += val;
        }
    }
}

// ============================================================================
// Q9: 产品类型利润 - V22 优化: SEMI JOIN + INNER JOIN V19.2
// ============================================================================

void run_q9(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();

    // Step 1: 过滤 part (p_name LIKE '%green%')
    std::vector<int32_t> green_partkeys;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_name[i].find("green") != std::string::npos) {
            green_partkeys.push_back(part.p_partkey[i]);
        }
    }

    // Step 2: SEMI JOIN - lineitem.l_partkey ∈ green_partkeys
    std::vector<uint32_t> li_semi_matches;
    ops::semi_join_i32(
        green_partkeys.data(), green_partkeys.size(),
        li.l_partkey.data(), li.count,
        li_semi_matches
    );

    // Step 3: 提取有效 lineitem 的 orderkey
    std::vector<int32_t> valid_li_orderkeys;
    valid_li_orderkeys.reserve(li_semi_matches.size());
    for (uint32_t idx : li_semi_matches) {
        valid_li_orderkeys.push_back(li.l_orderkey[idx]);
    }

    // Step 4: INNER JOIN V19.2 - orders.o_orderkey × valid_li_orderkeys
    ops::JoinPairs order_join;
    ops::inner_join_i32(
        ord.o_orderkey.data(), ord.count,
        valid_li_orderkeys.data(), valid_li_orderkeys.size(),
        order_join
    );

    // Step 5: 构建查找表
    // supplier -> nation
    std::unordered_map<int32_t, int32_t> supp_nation;
    for (size_t i = 0; i < supp.count; ++i) {
        supp_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
    }

    // partsupp (partkey, suppkey) -> supplycost
    std::unordered_map<int64_t, int64_t> ps_cost;
    for (size_t i = 0; i < partsupp.count; ++i) {
        int64_t key = (static_cast<int64_t>(partsupp.ps_partkey[i]) << 32) |
                      static_cast<uint32_t>(partsupp.ps_suppkey[i]);
        ps_cost[key] = partsupp.ps_supplycost[i];
    }

    // nation name
    std::unordered_map<int32_t, std::string> nation_name;
    for (size_t i = 0; i < nat.count; ++i) {
        nation_name[nat.n_nationkey[i]] = nat.n_name[i];
    }

    // Step 6: 聚合结构
    struct Q9Key {
        std::string nation;
        int32_t year;

        bool operator==(const Q9Key& o) const {
            return nation == o.nation && year == o.year;
        }
    };

    struct Q9KeyHash {
        size_t operator()(const Q9Key& k) const {
            return std::hash<std::string>()(k.nation) ^ (std::hash<int32_t>()(k.year) << 1);
        }
    };

    // Step 7: 8 线程并行聚合
    constexpr size_t NUM_THREADS = 8;
    std::array<std::unordered_map<Q9Key, int64_t, Q9KeyHash>, NUM_THREADS> thread_results;
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (order_join.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, order_join.count);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_results = thread_results[t];

            for (size_t j = start; j < end; ++j) {
                uint32_t ord_idx = order_join.left_indices[j];
                uint32_t li_local_idx = order_join.right_indices[j];
                uint32_t li_idx = li_semi_matches[li_local_idx];

                // 获取 supplier nation
                int32_t suppkey = li.l_suppkey[li_idx];
                auto supp_it = supp_nation.find(suppkey);
                if (supp_it == supp_nation.end()) continue;

                // 获取 supplycost
                int64_t ps_key = (static_cast<int64_t>(li.l_partkey[li_idx]) << 32) |
                                 static_cast<uint32_t>(suppkey);
                auto cost_it = ps_cost.find(ps_key);
                if (cost_it == ps_cost.end()) continue;

                // 计算 amount
                __int128 disc_price = (__int128)li.l_extendedprice[li_idx] *
                                      (10000 - li.l_discount[li_idx]) / 10000;
                __int128 cost = (__int128)cost_it->second * li.l_quantity[li_idx] / 10000;
                int64_t amount = static_cast<int64_t>(disc_price - cost);

                // 获取 year
                int32_t year = 1970 + ord.o_orderdate[ord_idx] / 365;

                auto nat_it = nation_name.find(supp_it->second);
                if (nat_it == nation_name.end()) continue;

                Q9Key key{nat_it->second, year};
                local_results[key] += amount;
            }
        });
    }

    // 等待并合并结果
    for (auto& th : threads) { th.join(); }

    std::unordered_map<Q9Key, int64_t, Q9KeyHash> results;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (auto& [key, val] : thread_results[t]) {
            results[key] += val;
        }
    }
}

// ============================================================================
// Q10: 退货报告 - Join + GROUP BY + Filter
// ============================================================================

void run_q10(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& nat = loader.nation();

    // 日期范围: 1993-10-01 to 1994-01-01
    constexpr int32_t date_lo = dates::DATE_1993_10_01;
    constexpr int32_t date_hi = dates::DATE_1994_01_01;

    // Step 1: 过滤 orders
    std::unordered_map<int32_t, int32_t> order_cust;  // orderkey -> custkey
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            order_cust[ord.o_orderkey[i]] = ord.o_custkey[i];
        }
    }

    // Step 2: 过滤 lineitem (l_returnflag = 'R' = 2)
    std::unordered_map<int32_t, int64_t> cust_revenue;

    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_returnflag[i] != 2) continue;  // R = 2

        auto it = order_cust.find(li.l_orderkey[i]);
        if (it == order_cust.end()) continue;

        __int128 revenue = (__int128)li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;
        cust_revenue[it->second] += revenue;
    }
}

// ============================================================================
// Q12: 运输模式与订单优先级 - V22 优化: INNER JOIN V19.2 + 8线程
// ============================================================================

void run_q12(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();

    // 日期范围: 1994-01-01 to 1995-01-01
    constexpr int32_t date_lo = dates::DATE_1994_01_01;
    constexpr int32_t date_hi = dates::DATE_1995_01_01;

    // shipmode: MAIL=5, SHIP=3
    constexpr int8_t MAIL = 5;
    constexpr int8_t SHIP = 3;

    // orderpriority: 1-URGENT=0, 2-HIGH=1
    constexpr int8_t URGENT = 0;
    constexpr int8_t HIGH = 1;

    // Step 1: 过滤 lineitem 并提取符合条件的行
    std::vector<int32_t> valid_li_orderkeys;
    std::vector<uint32_t> valid_li_indices;
    valid_li_orderkeys.reserve(li.count / 10);
    valid_li_indices.reserve(li.count / 10);

    for (size_t i = 0; i < li.count; ++i) {
        int8_t mode = li.l_shipmode[i];
        if (mode != MAIL && mode != SHIP) continue;
        if (li.l_commitdate[i] >= li.l_receiptdate[i]) continue;
        if (li.l_shipdate[i] >= li.l_commitdate[i]) continue;
        if (li.l_receiptdate[i] < date_lo || li.l_receiptdate[i] >= date_hi) continue;

        valid_li_orderkeys.push_back(li.l_orderkey[i]);
        valid_li_indices.push_back(static_cast<uint32_t>(i));
    }

    // Step 2: INNER JOIN V19.2 - orders.o_orderkey × valid_li_orderkeys
    ops::JoinPairs join_result;
    ops::inner_join_i32(
        ord.o_orderkey.data(), ord.count,
        valid_li_orderkeys.data(), valid_li_orderkeys.size(),
        join_result
    );

    // Step 3: 8 线程并行聚合
    struct Q12Result {
        int64_t high_count = 0;
        int64_t low_count = 0;
    };

    constexpr size_t NUM_THREADS = 8;
    // 每个线程维护 MAIL 和 SHIP 的结果
    alignas(128) std::array<std::array<Q12Result, 2>, NUM_THREADS> thread_results = {};
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (join_result.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, join_result.count);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_results = thread_results[t];

            for (size_t j = start; j < end; ++j) {
                uint32_t ord_idx = join_result.left_indices[j];
                uint32_t li_local_idx = join_result.right_indices[j];
                uint32_t li_idx = valid_li_indices[li_local_idx];

                int8_t mode = li.l_shipmode[li_idx];
                int8_t priority = ord.o_orderpriority[ord_idx];

                // mode_idx: MAIL=0, SHIP=1
                int mode_idx = (mode == MAIL) ? 0 : 1;

                if (priority == URGENT || priority == HIGH) {
                    local_results[mode_idx].high_count++;
                } else {
                    local_results[mode_idx].low_count++;
                }
            }
        });
    }

    // 等待并合并结果
    for (auto& th : threads) { th.join(); }

    std::unordered_map<int8_t, Q12Result> results;
    Q12Result mail_result = {}, ship_result = {};
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        mail_result.high_count += thread_results[t][0].high_count;
        mail_result.low_count += thread_results[t][0].low_count;
        ship_result.high_count += thread_results[t][1].high_count;
        ship_result.low_count += thread_results[t][1].low_count;
    }
    results[MAIL] = mail_result;
    results[SHIP] = ship_result;
}

// ============================================================================
// Q14: 促销效果 - Join + 条件聚合
// ============================================================================

void run_q14(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    // 日期范围: 1995-09-01 to 1995-10-01
    constexpr int32_t date_lo = dates::DATE_1995_09_01;
    constexpr int32_t date_hi = dates::DATE_1995_10_01;

    // 构建 part -> is_promo 映射
    std::unordered_map<int32_t, bool> part_promo;
    for (size_t i = 0; i < part.count; ++i) {
        part_promo[part.p_partkey[i]] = (part.p_type[i].find("PROMO") == 0);
    }

    __int128 sum_promo = 0;
    __int128 sum_total = 0;

    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_shipdate[i] < date_lo || li.l_shipdate[i] >= date_hi) continue;

        auto it = part_promo.find(li.l_partkey[i]);
        if (it == part_promo.end()) continue;

        __int128 val = (__int128)li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;
        sum_total += val;

        if (it->second) {
            sum_promo += val;
        }
    }

    volatile double result = 100.0 * static_cast<double>(sum_promo) / static_cast<double>(sum_total);
    (void)result;
}

// ============================================================================
// Q18: 大批量客户 - V22 优化: 单线程 8路展开 GROUP BY (避免合并开销)
// ============================================================================

void run_q18(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();

    // Step 1: 单线程 8路展开计算每个 orderkey 的 sum(l_quantity)
    // 注: 对于 GROUP BY 高基数场景，单线程避免了合并开销
    std::unordered_map<int32_t, int64_t> order_qty;
    order_qty.reserve(ord.count);

    size_t i = 0;
    for (; i + 8 <= li.count; i += 8) {
        __builtin_prefetch(&li.l_orderkey[i + 64], 0, 3);
        __builtin_prefetch(&li.l_quantity[i + 64], 0, 3);

        order_qty[li.l_orderkey[i]] += li.l_quantity[i];
        order_qty[li.l_orderkey[i+1]] += li.l_quantity[i+1];
        order_qty[li.l_orderkey[i+2]] += li.l_quantity[i+2];
        order_qty[li.l_orderkey[i+3]] += li.l_quantity[i+3];
        order_qty[li.l_orderkey[i+4]] += li.l_quantity[i+4];
        order_qty[li.l_orderkey[i+5]] += li.l_quantity[i+5];
        order_qty[li.l_orderkey[i+6]] += li.l_quantity[i+6];
        order_qty[li.l_orderkey[i+7]] += li.l_quantity[i+7];
    }
    for (; i < li.count; ++i) {
        order_qty[li.l_orderkey[i]] += li.l_quantity[i];
    }

    // Step 2: 过滤 sum > 300 * 10000
    constexpr int64_t qty_threshold = 300 * 10000;
    std::unordered_set<int32_t> large_orders;
    for (const auto& [key, qty] : order_qty) {
        if (qty > qty_threshold) {
            large_orders.insert(key);
        }
    }

    // Step 3: 获取符合条件的 orders 信息
    struct Q18Result {
        int32_t custkey;
        int32_t orderkey;
        int32_t orderdate;
        int64_t totalprice;
        int64_t sum_qty;
    };

    std::vector<Q18Result> results;
    results.reserve(large_orders.size());

    for (size_t j = 0; j < ord.count; ++j) {
        if (large_orders.count(ord.o_orderkey[j])) {
            Q18Result r;
            r.orderkey = ord.o_orderkey[j];
            r.custkey = ord.o_custkey[j];
            r.orderdate = ord.o_orderdate[j];
            r.totalprice = ord.o_totalprice[j];
            r.sum_qty = order_qty[r.orderkey];
            results.push_back(r);
        }
    }

    // Step 4: 排序 (totalprice DESC, orderdate ASC) 并取前 100
    std::partial_sort(results.begin(),
                      results.begin() + std::min<size_t>(100, results.size()),
                      results.end(),
                      [](const Q18Result& a, const Q18Result& b) {
                          if (a.totalprice != b.totalprice)
                              return a.totalprice > b.totalprice;
                          return a.orderdate < b.orderdate;
                      });

    if (results.size() > 100) {
        results.resize(100);
    }
}

// ============================================================================
// Category B: 部分可优化的查询
// ============================================================================

void run_q2(TPCHDataLoader& loader) {
    // Q2 包含相关子查询，使用简化实现
    // 实际应用中可以部分优化 Join
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // 找到 EUROPE region
    int32_t europe_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::EUROPE) {
            europe_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    // 找到 EUROPE 的 nations
    std::unordered_set<int32_t> europe_nations;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == europe_regionkey) {
            europe_nations.insert(nat.n_nationkey[i]);
        }
    }

    // 过滤 supplier
    std::unordered_set<int32_t> europe_suppliers;
    for (size_t i = 0; i < supp.count; ++i) {
        if (europe_nations.count(supp.s_nationkey[i])) {
            europe_suppliers.insert(supp.s_suppkey[i]);
        }
    }

    // 过滤 part (p_size = 15, p_type LIKE '%BRASS')
    std::unordered_set<int32_t> valid_parts;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_size[i] == 15 &&
            part.p_type[i].find("BRASS") != std::string::npos) {
            valid_parts.insert(part.p_partkey[i]);
        }
    }

    // 简化的结果计算...
}

void run_q4(TPCHDataLoader& loader) {
    // Q4: EXISTS 子查询 - 使用 GPU SEMI Join
    const auto& ord = loader.orders();
    const auto& li = loader.lineitem();

    // 日期范围: 1993-07-01 to 1993-10-01
    constexpr int32_t date_lo = dates::DATE_1993_07_01;
    constexpr int32_t date_hi = dates::DATE_1993_10_01;

    // Step 1: 构建满足条件的 lineitem orderkeys (l_commitdate < l_receiptdate)
    std::vector<int32_t> late_orderkeys;
    late_orderkeys.reserve(li.count / 4);  // 预估约 25% 的行满足条件
    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_commitdate[i] < li.l_receiptdate[i]) {
            late_orderkeys.push_back(li.l_orderkey[i]);
        }
    }

    // Step 2: 过滤 orders 并提取 orderkeys 和索引
    std::vector<int32_t> filtered_orderkeys;
    std::vector<uint32_t> filtered_indices;
    filtered_orderkeys.reserve(ord.count / 4);
    filtered_indices.reserve(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            filtered_orderkeys.push_back(ord.o_orderkey[i]);
            filtered_indices.push_back(static_cast<uint32_t>(i));
        }
    }

    // Step 3: GPU SEMI Join - 找出存在于 late_orderkeys 中的 filtered orders
    std::vector<uint32_t> semi_matches;
    ops::semi_join_i32(
        late_orderkeys.data(), late_orderkeys.size(),
        filtered_orderkeys.data(), filtered_orderkeys.size(),
        semi_matches
    );

    // Step 4: 聚合 - 根据匹配索引统计 orderpriority
    std::unordered_map<int8_t, int64_t> results;
    for (uint32_t match_idx : semi_matches) {
        uint32_t ord_idx = filtered_indices[match_idx];
        results[ord.o_orderpriority[ord_idx]]++;
    }
}

void run_q11(TPCHDataLoader& loader) {
    // Q11 包含 HAVING 子查询
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    // 找到 GERMANY
    int32_t germany_key = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::GERMANY) {
            germany_key = nat.n_nationkey[i];
            break;
        }
    }

    // 找到 GERMANY 的 suppliers
    std::unordered_set<int32_t> germany_suppliers;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_nationkey[i] == germany_key) {
            germany_suppliers.insert(supp.s_suppkey[i]);
        }
    }

    // 计算 value
    std::unordered_map<int32_t, int64_t> part_value;
    __int128 total_value = 0;

    for (size_t i = 0; i < partsupp.count; ++i) {
        if (germany_suppliers.count(partsupp.ps_suppkey[i])) {
            __int128 val = (__int128)partsupp.ps_supplycost[i] * partsupp.ps_availqty[i] / 10000;
            part_value[partsupp.ps_partkey[i]] += val;
            total_value += val;
        }
    }

    // 阈值
    int64_t threshold = static_cast<int64_t>(total_value / 10000);  // 0.0001

    // 过滤
    std::vector<std::pair<int32_t, int64_t>> results;
    for (const auto& [partkey, value] : part_value) {
        if (value > threshold) {
            results.push_back({partkey, value});
        }
    }

    // 排序
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
}

void run_q15(TPCHDataLoader& loader) {
    // Q15 包含 CTE
    const auto& li = loader.lineitem();
    const auto& supp = loader.supplier();

    // 日期范围: 1996-01-01 to 1996-04-01
    constexpr int32_t date_lo = dates::DATE_1996_01_01;
    constexpr int32_t date_hi = dates::DATE_1996_04_01;

    // 计算每个 supplier 的 revenue
    std::unordered_map<int32_t, int64_t> supp_revenue;
    for (size_t i = 0; i < li.count; ++i) {
        if (li.l_shipdate[i] >= date_lo && li.l_shipdate[i] < date_hi) {
            __int128 rev = (__int128)li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;
            supp_revenue[li.l_suppkey[i]] += rev;
        }
    }

    // 找到最大 revenue
    int64_t max_revenue = 0;
    for (const auto& [suppkey, revenue] : supp_revenue) {
        max_revenue = std::max(max_revenue, revenue);
    }

    // 找到最大 revenue 的 supplier
    std::vector<int32_t> top_suppliers;
    for (const auto& [suppkey, revenue] : supp_revenue) {
        if (revenue == max_revenue) {
            top_suppliers.push_back(suppkey);
        }
    }
}

void run_q16(TPCHDataLoader& loader) {
    // Q16 包含 NOT IN 子查询
    const auto& part = loader.part();
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();

    // 找到有 complaint 的 suppliers
    std::unordered_set<int32_t> complaint_suppliers;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_comment[i].find("Customer") != std::string::npos &&
            supp.s_comment[i].find("Complaints") != std::string::npos) {
            complaint_suppliers.insert(supp.s_suppkey[i]);
        }
    }

    // 过滤 part 并构建哈希索引 (修复 O(n²) → O(n))
    std::unordered_set<int32_t> valid_sizes = {49, 14, 23, 45, 19, 3, 36, 9};

    // 构建 partkey -> part index 哈希表
    struct PartInfo {
        std::string brand;
        std::string type;
        int32_t size;
    };
    std::unordered_map<int32_t, PartInfo> part_info_map;
    part_info_map.reserve(part.count);

    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_brand[i] != "Brand#45" &&
            part.p_type[i].find("MEDIUM POLISHED") != 0 &&
            valid_sizes.count(part.p_size[i])) {
            part_info_map[part.p_partkey[i]] = {
                part.p_brand[i], part.p_type[i], part.p_size[i]
            };
        }
    }

    // 聚合 (现在是 O(n) 查找)
    struct Q16Key {
        std::string brand;
        std::string type;
        int32_t size;

        bool operator==(const Q16Key& o) const {
            return brand == o.brand && type == o.type && size == o.size;
        }
    };

    struct Q16KeyHash {
        size_t operator()(const Q16Key& k) const {
            return std::hash<std::string>()(k.brand) ^
                   (std::hash<std::string>()(k.type) << 1) ^
                   (std::hash<int32_t>()(k.size) << 2);
        }
    };

    std::unordered_map<Q16Key, std::unordered_set<int32_t>, Q16KeyHash> results;

    for (size_t i = 0; i < partsupp.count; ++i) {
        // O(1) 哈希查找替代 O(n) 循环
        auto it = part_info_map.find(partsupp.ps_partkey[i]);
        if (it == part_info_map.end()) continue;
        if (complaint_suppliers.count(partsupp.ps_suppkey[i])) continue;

        const auto& pi = it->second;
        Q16Key key{pi.brand, pi.type, pi.size};
        results[key].insert(partsupp.ps_suppkey[i]);
    }
}

void run_q19(TPCHDataLoader& loader) {
    // Q19 复杂 OR 条件
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    // 构建 part 信息映射
    struct PartInfo {
        std::string brand;
        std::string container;
        int32_t size;
    };
    std::unordered_map<int32_t, PartInfo> part_info;
    for (size_t i = 0; i < part.count; ++i) {
        part_info[part.p_partkey[i]] = {part.p_brand[i], part.p_container[i], part.p_size[i]};
    }

    // shipmode: AIR=1, REG AIR=0
    // shipinstruct: DELIVER IN PERSON=0

    __int128 revenue = 0;

    for (size_t i = 0; i < li.count; ++i) {
        // 公共条件
        if (li.l_shipinstruct[i] != 0) continue;  // DELIVER IN PERSON
        if (li.l_shipmode[i] != 0 && li.l_shipmode[i] != 1) continue;  // AIR or REG AIR

        auto it = part_info.find(li.l_partkey[i]);
        if (it == part_info.end()) continue;

        const auto& p = it->second;
        int64_t qty = li.l_quantity[i];

        bool match = false;

        // 条件 1: Brand#12, SM*, qty 1-11, size 1-5
        if (p.brand == "Brand#12" &&
            (p.container == "SM CASE" || p.container == "SM BOX" ||
             p.container == "SM PACK" || p.container == "SM PKG") &&
            qty >= 10000 && qty <= 110000 &&
            p.size >= 1 && p.size <= 5) {
            match = true;
        }

        // 条件 2: Brand#23, MED*, qty 10-20, size 1-10
        if (!match && p.brand == "Brand#23" &&
            (p.container == "MED BAG" || p.container == "MED BOX" ||
             p.container == "MED PKG" || p.container == "MED PACK") &&
            qty >= 100000 && qty <= 200000 &&
            p.size >= 1 && p.size <= 10) {
            match = true;
        }

        // 条件 3: Brand#34, LG*, qty 20-30, size 1-15
        if (!match && p.brand == "Brand#34" &&
            (p.container == "LG CASE" || p.container == "LG BOX" ||
             p.container == "LG PACK" || p.container == "LG PKG") &&
            qty >= 200000 && qty <= 300000 &&
            p.size >= 1 && p.size <= 15) {
            match = true;
        }

        if (match) {
            revenue += (__int128)li.l_extendedprice[i] * (10000 - li.l_discount[i]) / 10000;
        }
    }

    volatile double result = static_cast<double>(revenue) / 10000.0;
    (void)result;
}

} // namespace queries

// ============================================================================
// 外部接口
// ============================================================================

QueryImplFunc get_thunderduck_impl(const std::string& query_id) {
    static bool initialized = false;
    if (!initialized) {
        queries::register_all_queries();
        initialized = true;
    }
    return queries::get_query_impl(query_id);
}

bool has_thunderduck_impl(const std::string& query_id) {
    static bool initialized = false;
    if (!initialized) {
        queries::register_all_queries();
        initialized = true;
    }
    return queries::has_optimized_impl(query_id);
}

} // namespace tpch
} // namespace thunderduck
