/**
 * ThunderDuck TPC-H Operators V55 Implementation
 *
 * @version 55
 * @date 2026-01-30
 */

#include "tpch_operators_v55.h"
#include "tpch_constants.h"      // 统一常量定义
#include <iostream>
#include <iomanip>
#include <thread>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v55 {

const char* V55_VERSION = "V55-GenericOperators";
const char* V55_DATE = "2026-01-30";

const char* V55_FEATURES[] = {
    "SubqueryDecorrelation (Q2, Q17)",
    "GenericParallelMultiJoin (Q8)",
    "GenericTwoPhaseAgg (Q17)",
    "Adaptive strategy selection"
};

// ============================================================================
// Q2: 子查询解关联优化
// ============================================================================

void run_q2_v55(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& partsupp = loader.partsupp();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // Step 1: 找到 EUROPE region 的 nationkeys
    std::unordered_set<int32_t> europe_nations;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::EUROPE) {
            int32_t regionkey = reg.r_regionkey[i];
            for (size_t j = 0; j < nat.count; ++j) {
                if (nat.n_regionkey[j] == regionkey) {
                    europe_nations.insert(nat.n_nationkey[j]);
                }
            }
            break;
        }
    }

    // Step 2: 找到 EUROPE 的 suppliers
    std::unordered_set<int32_t> europe_suppliers;
    std::unordered_map<int32_t, int32_t> supp_nation;  // suppkey -> nationkey
    for (size_t i = 0; i < supp.count; ++i) {
        if (europe_nations.count(supp.s_nationkey[i])) {
            europe_suppliers.insert(supp.s_suppkey[i]);
            supp_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
        }
    }

    // Step 3: 使用 SubqueryDecorrelation 预计算 MIN(ps_supplycost) per partkey
    // 只考虑 EUROPE suppliers
    SubqueryDecorrelation<int32_t, int64_t> min_cost_decorrelator;

    min_cost_decorrelator.precompute(
        partsupp.ps_partkey.data(),
        partsupp.ps_supplycost.data(),
        partsupp.count,
        [](int64_t a, int64_t b) { return std::min(a, b); },  // MIN
        [&](size_t i) {
            return europe_suppliers.count(partsupp.ps_suppkey[i]) > 0;
        }
    );

    // Step 4: 过滤 part (p_size = 15, p_type LIKE '%BRASS')
    std::unordered_map<int32_t, size_t> valid_parts;  // partkey -> part index
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_size[i] == 15) {
            const auto& ptype = part.p_type[i];
            if (ptype.size() >= 5 &&
                ptype.substr(ptype.size() - 5) == "BRASS") {
                valid_parts[part.p_partkey[i]] = i;
            }
        }
    }

    // Step 5: 主查询 - 使用解关联后的 min_cost 查找
    struct Q2Result {
        int64_t s_acctbal;
        int32_t suppkey;
        int32_t partkey;
        int32_t nationkey;
    };
    std::vector<Q2Result> results;
    results.reserve(100);

    for (size_t i = 0; i < partsupp.count; ++i) {
        int32_t partkey = partsupp.ps_partkey[i];
        int32_t suppkey = partsupp.ps_suppkey[i];
        int64_t supplycost = partsupp.ps_supplycost[i];

        // 检查 partkey 是否有效
        if (valid_parts.count(partkey) == 0) continue;

        // 检查 supplier 是否在 EUROPE
        auto supp_it = supp_nation.find(suppkey);
        if (supp_it == supp_nation.end()) continue;

        // 使用解关联结果: supplycost = MIN(supplycost)
        int64_t min_cost;
        if (!min_cost_decorrelator.lookup(partkey, min_cost)) continue;
        if (supplycost != min_cost) continue;

        // 找到匹配的 supplier 信息
        for (size_t s = 0; s < supp.count; ++s) {
            if (supp.s_suppkey[s] == suppkey) {
                results.push_back({
                    supp.s_acctbal[s],
                    suppkey,
                    partkey,
                    supp_it->second
                });
                break;
            }
        }
    }

    // Step 6: 排序 (s_acctbal DESC, n_name, s_name, p_partkey)
    std::sort(results.begin(), results.end(), [](const Q2Result& a, const Q2Result& b) {
        if (a.s_acctbal != b.s_acctbal) return a.s_acctbal > b.s_acctbal;
        if (a.nationkey != b.nationkey) return a.nationkey < b.nationkey;
        if (a.suppkey != b.suppkey) return a.suppkey < b.suppkey;
        return a.partkey < b.partkey;
    });

    // 取前 100
    if (results.size() > 100) results.resize(100);

    // 防止优化
    volatile size_t sink = results.size();
    (void)sink;
}

// ============================================================================
// Q8: 通用并行多表 JOIN
// ============================================================================

void run_q8_v55(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // 日期范围: 1995-01-01 to 1996-12-31
    constexpr int32_t date_lo = dates::D1995_01_01;
    constexpr int32_t date_hi = dates::D1996_12_31;

    // Step 1: 构建维度表过滤器

    // 找到 AMERICA region
    int32_t america_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::AMERICA) {
            america_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    // 找到 AMERICA 的 nations
    std::unordered_set<int32_t> america_nations;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == america_regionkey) {
            america_nations.insert(nat.n_nationkey[i]);
        }
    }

    // 找到 BRAZIL nationkey
    int32_t brazil_nationkey = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == nations::BRAZIL) {
            brazil_nationkey = nat.n_nationkey[i];
            break;
        }
    }

    // 过滤 ECONOMY ANODIZED STEEL parts
    std::unordered_set<int32_t> valid_parts;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == "ECONOMY ANODIZED STEEL") {
            valid_parts.insert(part.p_partkey[i]);
        }
    }

    // 过滤 AMERICA customers
    std::unordered_set<int32_t> america_customers;
    for (size_t i = 0; i < cust.count; ++i) {
        if (america_nations.count(cust.c_nationkey[i])) {
            america_customers.insert(cust.c_custkey[i]);
        }
    }

    // supplier -> nationkey 映射
    std::unordered_map<int32_t, int32_t> supp_nation;
    for (size_t i = 0; i < supp.count; ++i) {
        supp_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
    }

    // Step 2: 过滤 orders (日期 + AMERICA customers)
    std::unordered_map<int32_t, int32_t> valid_orders;  // orderkey -> year
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] <= date_hi) {
            if (america_customers.count(ord.o_custkey[i])) {
                int32_t year = 1970 + ord.o_orderdate[i] / 365;
                valid_orders[ord.o_orderkey[i]] = year;
            }
        }
    }

    // Step 3: 并行扫描 lineitem 并聚合
    struct YearVolume {
        int64_t total_volume = 0;
        int64_t brazil_volume = 0;
    };

    constexpr size_t NUM_THREADS = 8;
    std::vector<std::unordered_map<int32_t, YearVolume>> thread_results(NUM_THREADS);
    std::vector<std::thread> threads;

    size_t chunk = (li.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, li.count);

        threads.emplace_back([&, t, start, end]() {
            auto& local = thread_results[t];

            for (size_t i = start; i < end; ++i) {
                // 检查 part
                if (!valid_parts.count(li.l_partkey[i])) continue;

                // 检查 order
                auto ord_it = valid_orders.find(li.l_orderkey[i]);
                if (ord_it == valid_orders.end()) continue;

                int32_t year = ord_it->second;

                // 计算 volume
                __int128 volume = (__int128)li.l_extendedprice[i] *
                                  (10000 - li.l_discount[i]) / 10000;

                auto& yv = local[year];
                yv.total_volume += volume;

                // 检查 supplier nation
                auto supp_it = supp_nation.find(li.l_suppkey[i]);
                if (supp_it != supp_nation.end() &&
                    supp_it->second == brazil_nationkey) {
                    yv.brazil_volume += volume;
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // Step 4: 合并结果
    std::unordered_map<int32_t, YearVolume> results;
    for (const auto& local : thread_results) {
        for (const auto& [year, yv] : local) {
            auto& r = results[year];
            r.total_volume += yv.total_volume;
            r.brazil_volume += yv.brazil_volume;
        }
    }

    // 计算市场份额
    volatile double sink = 0;
    for (const auto& [year, yv] : results) {
        double mkt_share = yv.total_volume > 0
            ? 100.0 * static_cast<double>(yv.brazil_volume) / static_cast<double>(yv.total_volume)
            : 0.0;
        sink += mkt_share;
    }
    (void)sink;
}

// ============================================================================
// Q17: 通用两阶段聚合
// ============================================================================

void run_q17_v55(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    // Step 1: 过滤 BRAND#23, MED BOX parts
    std::unordered_set<int32_t> valid_parts;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_brand[i] == "Brand#23" &&
            part.p_container[i] == "MED BOX") {
            valid_parts.insert(part.p_partkey[i]);
        }
    }

    // Step 2: 使用 GenericTwoPhaseAgg 预计算 AVG(l_quantity) per partkey
    GenericTwoPhaseAgg<int32_t, int64_t> avg_qty_agg;

    TwoPhaseAggConfig config;
    config.num_threads = 8;
    config.use_thread_local = true;

    avg_qty_agg.phase1_precompute(
        li.l_partkey.data(),
        li.l_quantity.data(),
        li.count,
        [&](size_t i) {
            return valid_parts.count(li.l_partkey[i]) > 0;
        },
        config
    );

    // Step 3: Phase 2 - 过滤并聚合
    // WHERE l_quantity < 0.2 * AVG(l_quantity)
    __int128 total_price = 0;

    for (size_t i = 0; i < li.count; ++i) {
        int32_t partkey = li.l_partkey[i];
        if (!valid_parts.count(partkey)) continue;

        // 查找预计算的 AVG
        typename GenericTwoPhaseAgg<int32_t, int64_t>::AggResult agg_result;
        if (!avg_qty_agg.lookup(partkey, agg_result)) continue;

        // 阈值: 0.2 * AVG(l_quantity)
        // 注意: quantity 是定点数 (scale=10000)
        int64_t threshold = agg_result.avg() / 5;  // 0.2x

        if (li.l_quantity[i] < threshold) {
            total_price += li.l_extendedprice[i];
        }
    }

    // 结果: AVG(l_extendedprice) / 7.0
    // 这里简化为 sum / 7
    volatile double result = static_cast<double>(total_price) / 70000.0;
    (void)result;
}

} // namespace ops_v55
} // namespace tpch
} // namespace thunderduck
