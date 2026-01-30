/**
 * ThunderDuck TPC-H Operators V56 Implementation
 *
 * @version 56
 * @date 2026-01-30
 */

#include "tpch_operators_v56.h"
#include "tpch_constants.h"      // 统一常量定义
#include <iostream>
#include <iomanip>
#include <thread>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v56 {

const char* V56_VERSION = "V56-OptimizedGenericOperators";
const char* V56_DATE = "2026-01-30";

// ============================================================================
// Q5: 本地供应商收入 (V56 - 消除热路径第 3 次 hash 查找)
// ============================================================================

void run_q5_v56(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    constexpr int32_t date_lo = dates::D1994_01_01;
    constexpr int32_t date_hi = dates::D1995_01_01;

    // Phase 1: 找到 ASIA region 的 nations
    int32_t asia_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::ASIA) {
            asia_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    std::unordered_set<int32_t> asia_nation_set;
    for (size_t j = 0; j < nat.count; ++j) {
        if (nat.n_regionkey[j] == asia_regionkey) {
            asia_nation_set.insert(nat.n_nationkey[j]);
        }
    }

    // Phase 2: 构建 supplier → nation (使用 CompactHashTable，无 Bloom Filter 开销)
    CompactHashTable<int32_t> supp_to_nation;
    supp_to_nation.init(supp.count / 5);
    for (size_t i = 0; i < supp.count; ++i) {
        if (asia_nation_set.count(supp.s_nationkey[i])) {
            supp_to_nation.insert(supp.s_suppkey[i], supp.s_nationkey[i]);
        }
    }

    // Phase 3: 构建 customer → nation (临时 map)
    std::unordered_map<int32_t, int32_t> cust_to_nation;
    cust_to_nation.reserve(cust.count / 5);
    for (size_t i = 0; i < cust.count; ++i) {
        if (asia_nation_set.count(cust.c_nationkey[i])) {
            cust_to_nation[cust.c_custkey[i]] = cust.c_nationkey[i];
        }
    }

    // Phase 4: 构建 orderkey → cust_nation (关键优化!)
    // 直接存储 nationkey，避免热路径的第 3 次查找
    // 这是 V56 vs V32 的核心区别
    CompactHashTable<int32_t> order_to_cust_nation;
    order_to_cust_nation.init(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            auto it = cust_to_nation.find(ord.o_custkey[i]);
            if (it != cust_to_nation.end()) {
                // 直接存储 nationkey 而不是 custkey
                order_to_cust_nation.insert(ord.o_orderkey[i], it->second);
            }
        }
    }

    // 释放临时 map
    cust_to_nation.clear();

    // Phase 5: 并行扫描 lineitem - 只需 2 次查找 (vs V32 的 3 次)
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<ThreadLocalAggregator<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) agg.init(25);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([&, t, start, end]() {
            auto& local_agg = thread_aggs[t];

            // 批量处理 (8 路)
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                alignas(32) int32_t orderkeys[8];
                alignas(32) int32_t suppkeys[8];
                for (int j = 0; j < 8; ++j) {
                    orderkeys[j] = li.l_orderkey[i + j];
                    suppkeys[j] = li.l_suppkey[i + j];
                }

                // 批量查找 orderkey → cust_nation
                const int32_t* cust_nat_results[8];
                order_to_cust_nation.batch_find(orderkeys, cust_nat_results);

                // 批量查找 suppkey → nation
                const int32_t* supp_nat_results[8];
                supp_to_nation.batch_find(suppkeys, supp_nat_results);

                // 处理结果 - 只需比较，不需要第 3 次查找!
                for (int j = 0; j < 8; ++j) {
                    if (!cust_nat_results[j] || !supp_nat_results[j]) continue;

                    int32_t cust_nat = *cust_nat_results[j];
                    int32_t supp_nat = *supp_nat_results[j];

                    if (cust_nat != supp_nat) continue;

                    int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i + j]) *
                                      (10000 - li.l_discount[i + j]) / 10000;
                    local_agg.add(cust_nat, revenue);
                }
            }

            // 处理剩余
            for (; i < end; ++i) {
                const int32_t* cust_nat_ptr = order_to_cust_nation.find(li.l_orderkey[i]);
                if (!cust_nat_ptr) continue;

                const int32_t* supp_nat_ptr = supp_to_nation.find(li.l_suppkey[i]);
                if (!supp_nat_ptr) continue;

                if (*cust_nat_ptr != *supp_nat_ptr) continue;

                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_agg.add(*cust_nat_ptr, revenue);
            }
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    CompactHashTable<int64_t> nation_revenue;
    nation_revenue.init(25);
    for (auto& agg : thread_aggs) {
        agg.table.for_each([&](int32_t nat, int64_t rev) {
            nation_revenue.add_or_update(nat, rev);
        });
    }

    volatile int64_t sink = 0;
    nation_revenue.for_each([&sink](int32_t, int64_t rev) { sink += rev; });
    (void)sink;
}

// ============================================================================
// Q2: 最小成本供应商 (V56 - Direct Array 解关联)
// ============================================================================

void run_q2_v56(TPCHDataLoader& loader) {
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

    // Step 2: 构建 EUROPE suppliers 的直接数组索引
    std::vector<int32_t> supp_nationkey(supp.count + 1, -1);  // suppkey → nationkey
    std::vector<int64_t> supp_acctbal(supp.count + 1, 0);     // suppkey → acctbal

    for (size_t i = 0; i < supp.count; ++i) {
        if (europe_nations.count(supp.s_nationkey[i])) {
            int32_t suppkey = supp.s_suppkey[i];
            if (suppkey >= 0 && static_cast<size_t>(suppkey) < supp_nationkey.size()) {
                supp_nationkey[suppkey] = supp.s_nationkey[i];
                supp_acctbal[suppkey] = supp.s_acctbal[i];
            }
        }
    }

    // Step 3: 使用 DirectArrayDecorrelation 预计算 MIN(ps_supplycost) per partkey
    DirectArrayDecorrelation<int64_t> min_cost_decorrelator;

    min_cost_decorrelator.precompute(
        partsupp.ps_partkey.data(),
        partsupp.ps_supplycost.data(),
        partsupp.count,
        [](int64_t a, int64_t b) { return std::min(a, b); },
        [&](size_t i) {
            int32_t suppkey = partsupp.ps_suppkey[i];
            return suppkey >= 0 &&
                   static_cast<size_t>(suppkey) < supp_nationkey.size() &&
                   supp_nationkey[suppkey] >= 0;
        }
    );

    // Step 4: 构建有效 part 的直接数组 (p_size = 15, p_type LIKE '%BRASS')
    std::vector<bool> valid_parts(part.count + 1, false);
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_size[i] == 15) {
            const auto& ptype = part.p_type[i];
            if (ptype.size() >= 5 &&
                ptype.substr(ptype.size() - 5) == "BRASS") {
                int32_t partkey = part.p_partkey[i];
                if (partkey >= 0 && static_cast<size_t>(partkey) < valid_parts.size()) {
                    valid_parts[partkey] = true;
                }
            }
        }
    }

    // Step 5: 主查询 - 使用直接数组和预计算的 min_cost
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

        // 检查 partkey 是否有效 (O(1) 直接数组)
        if (partkey < 0 || static_cast<size_t>(partkey) >= valid_parts.size() ||
            !valid_parts[partkey]) continue;

        // 检查 supplier 是否在 EUROPE (O(1) 直接数组)
        if (suppkey < 0 || static_cast<size_t>(suppkey) >= supp_nationkey.size() ||
            supp_nationkey[suppkey] < 0) continue;

        // 使用 DirectArray 解关联结果: supplycost = MIN(supplycost)
        int64_t min_cost;
        if (!min_cost_decorrelator.lookup(partkey, min_cost)) continue;
        if (supplycost != min_cost) continue;

        results.push_back({
            supp_acctbal[suppkey],
            suppkey,
            partkey,
            supp_nationkey[suppkey]
        });
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

    volatile size_t sink = results.size();
    (void)sink;
}

// ============================================================================
// Q8: 市场份额 (V56 - Bloom Filter + 预计算维度)
// ============================================================================

void run_q8_v56(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    constexpr int32_t date_lo = dates::D1995_01_01;
    constexpr int32_t date_hi = dates::D1996_12_31;

    // Step 1: 预计算维度表 (全部用直接数组)

    // 找到 AMERICA region
    int32_t america_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == regions::AMERICA) {
            america_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    // 构建 nation → is_america, nation → is_brazil
    std::vector<bool> is_america_nation(25, false);
    int32_t brazil_nationkey = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == america_regionkey) {
            is_america_nation[nat.n_nationkey[i]] = true;
        }
        if (nat.n_name[i] == nations::BRAZIL) {
            brazil_nationkey = nat.n_nationkey[i];
        }
    }

    // 构建 part → is_valid (ECONOMY ANODIZED STEEL) 用 Bitmap
    std::vector<bool> valid_parts(part.count + 1, false);
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == "ECONOMY ANODIZED STEEL") {
            int32_t partkey = part.p_partkey[i];
            if (partkey >= 0 && static_cast<size_t>(partkey) < valid_parts.size()) {
                valid_parts[partkey] = true;
            }
        }
    }

    // 构建 customer → is_america (直接数组)
    std::vector<bool> america_customers(cust.count + 1, false);
    for (size_t i = 0; i < cust.count; ++i) {
        int32_t nationkey = cust.c_nationkey[i];
        if (nationkey >= 0 && nationkey < 25 && is_america_nation[nationkey]) {
            int32_t custkey = cust.c_custkey[i];
            if (custkey >= 0 && static_cast<size_t>(custkey) < america_customers.size()) {
                america_customers[custkey] = true;
            }
        }
    }

    // 构建 supplier → nationkey (直接数组)
    std::vector<int32_t> supp_nation(supp.count + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        int32_t suppkey = supp.s_suppkey[i];
        if (suppkey >= 0 && static_cast<size_t>(suppkey) < supp_nation.size()) {
            supp_nation[suppkey] = supp.s_nationkey[i];
        }
    }

    // Step 2: 构建 orderkey → year (只保留有效 orders)
    struct OrderInfo {
        int16_t year;
    };
    CompactHashTable<OrderInfo> valid_orders;
    valid_orders.init(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] <= date_hi) {
            int32_t custkey = ord.o_custkey[i];
            if (custkey >= 0 && static_cast<size_t>(custkey) < america_customers.size() &&
                america_customers[custkey]) {
                int16_t year = static_cast<int16_t>(1970 + ord.o_orderdate[i] / 365);
                valid_orders.insert(ord.o_orderkey[i], {year});
            }
        }
    }

    // Step 3: 并行扫描 lineitem
    struct YearVolume {
        int64_t total_volume = 0;
        int64_t brazil_volume = 0;
    };

    constexpr size_t NUM_THREADS = 8;
    std::vector<std::array<YearVolume, 3>> thread_results(NUM_THREADS);  // years: 1995, 1996, other
    std::vector<std::thread> threads;

    size_t chunk = (li.count + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(start + chunk, li.count);

        threads.emplace_back([&, t, start, end]() {
            auto& local = thread_results[t];

            for (size_t i = start; i < end; ++i) {
                int32_t partkey = li.l_partkey[i];

                // O(1) 直接数组检查 part
                if (partkey < 0 || static_cast<size_t>(partkey) >= valid_parts.size() ||
                    !valid_parts[partkey]) continue;

                // Hash 查找 order
                const OrderInfo* ord_info = valid_orders.find(li.l_orderkey[i]);
                if (!ord_info) continue;

                int year_idx = ord_info->year - 1995;  // 0=1995, 1=1996
                if (year_idx < 0 || year_idx > 1) continue;

                // 计算 volume
                int64_t volume = static_cast<int64_t>(li.l_extendedprice[i]) *
                                 (10000 - li.l_discount[i]) / 10000;

                local[year_idx].total_volume += volume;

                // O(1) 直接数组检查 supplier nation
                int32_t suppkey = li.l_suppkey[i];
                if (suppkey >= 0 && static_cast<size_t>(suppkey) < supp_nation.size() &&
                    supp_nation[suppkey] == brazil_nationkey) {
                    local[year_idx].brazil_volume += volume;
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // Step 4: 合并结果
    std::array<YearVolume, 2> results{};
    for (const auto& local : thread_results) {
        for (int y = 0; y < 2; ++y) {
            results[y].total_volume += local[y].total_volume;
            results[y].brazil_volume += local[y].brazil_volume;
        }
    }

    volatile double sink = 0;
    for (int y = 0; y < 2; ++y) {
        double mkt_share = results[y].total_volume > 0
            ? 100.0 * static_cast<double>(results[y].brazil_volume) /
                      static_cast<double>(results[y].total_volume)
            : 0.0;
        sink += mkt_share;
    }
    (void)sink;
}

// ============================================================================
// Q17: 小订单收入 (V56 - Direct Array 两阶段聚合)
// ============================================================================

void run_q17_v56(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    // Step 1: 构建有效 part 的直接数组 (Brand#23, MED BOX)
    std::vector<bool> valid_parts(part.count + 1, false);
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_brand[i] == "Brand#23" &&
            part.p_container[i] == "MED BOX") {
            int32_t partkey = part.p_partkey[i];
            if (partkey >= 0 && static_cast<size_t>(partkey) < valid_parts.size()) {
                valid_parts[partkey] = true;
            }
        }
    }

    // Step 2: 使用 DirectArrayTwoPhaseAgg 预计算 AVG(l_quantity) per partkey
    DirectArrayTwoPhaseAgg<int64_t> avg_qty_agg;

    avg_qty_agg.phase1_precompute(
        li.l_partkey.data(),
        li.l_quantity.data(),
        li.count,
        [&](size_t i) {
            int32_t partkey = li.l_partkey[i];
            return partkey >= 0 &&
                   static_cast<size_t>(partkey) < valid_parts.size() &&
                   valid_parts[partkey];
        }
    );

    // Step 3: Phase 2 - 过滤并聚合
    // WHERE l_quantity < 0.2 * AVG(l_quantity)
    __int128 total_price = 0;

    for (size_t i = 0; i < li.count; ++i) {
        int32_t partkey = li.l_partkey[i];

        // O(1) 检查 part 有效性
        if (partkey < 0 || static_cast<size_t>(partkey) >= valid_parts.size() ||
            !valid_parts[partkey]) continue;

        // O(1) 查找预计算的 AVG
        typename DirectArrayTwoPhaseAgg<int64_t>::AggResult agg_result;
        if (!avg_qty_agg.lookup(partkey, agg_result)) continue;

        // 阈值: 0.2 * AVG(l_quantity)
        int64_t threshold = agg_result.avg() / 5;

        if (li.l_quantity[i] < threshold) {
            total_price += li.l_extendedprice[i];
        }
    }

    // 结果: AVG(l_extendedprice) / 7.0
    volatile double result = static_cast<double>(total_price) / 70000.0;
    (void)result;
}

} // namespace ops_v56
} // namespace tpch
} // namespace thunderduck
