/**
 * ThunderDuck TPC-H V42 优化算子实现
 *
 * Q8 并行化 + 线程局部聚合
 *
 * @version 42.0
 * @date 2026-01-29
 */

#include "tpch_operators_v42.h"
#include "tpch_config_v33.h"
#include <algorithm>
#include <thread>
#include <vector>
#include <array>

namespace thunderduck {
namespace tpch {
namespace ops_v42 {

// ============================================================================
// Q8 优化实现 V42 - 并行化版本
// ============================================================================

void run_q8_v42(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // ========== 参数 ==========
    const std::string target_nation = "BRAZIL";
    const std::string target_region = "AMERICA";
    const std::string target_part_type = "ECONOMY ANODIZED STEEL";
    const int32_t date_lo = 9131;   // 1995-01-01
    const int32_t date_hi = 9861;   // 1996-12-31

    // ========================================================================
    // Phase 1: 预计算查找表
    // ========================================================================

    // 1a. 目标区域和国家
    int32_t target_regionkey = -1;
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == target_region) {
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }
    if (target_regionkey < 0) return;

    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == target_nation) {
            target_nationkey = nat.n_nationkey[i];
            break;
        }
    }

    // 1b. nation → is_in_region
    std::vector<bool> nation_in_region(26, false);
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == target_regionkey) {
            nation_in_region[nat.n_nationkey[i]] = true;
        }
    }

    // 1c. supplier → nationkey
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supp.count; ++i) {
        if (supp.s_suppkey[i] > max_suppkey) max_suppkey = supp.s_suppkey[i];
    }
    std::vector<int32_t> supp_nation(max_suppkey + 1, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        supp_nation[supp.s_suppkey[i]] = supp.s_nationkey[i];
    }

    // 1d. customer → nationkey
    int32_t max_custkey = 0;
    for (size_t i = 0; i < cust.count; ++i) {
        if (cust.c_custkey[i] > max_custkey) max_custkey = cust.c_custkey[i];
    }
    std::vector<int32_t> cust_nation(max_custkey + 1, -1);
    for (size_t i = 0; i < cust.count; ++i) {
        cust_nation[cust.c_custkey[i]] = cust.c_nationkey[i];
    }

    // 1e. part → is_valid
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_partkey[i] > max_partkey) max_partkey = part.p_partkey[i];
    }
    std::vector<uint8_t> valid_part(max_partkey + 1, 0);
    size_t valid_count = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == target_part_type) {
            valid_part[part.p_partkey[i]] = 1;
            valid_count++;
        }
    }
    if (valid_count == 0) return;

    // 1f. orders → (year, is_america_customer)
    int32_t max_orderkey = 0;
    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderkey[i] > max_orderkey) max_orderkey = ord.o_orderkey[i];
    }

    struct OrderInfo {
        int16_t year;
        int8_t is_america;  // -1=invalid, 0=no, 1=yes
    };
    std::vector<OrderInfo> order_info(max_orderkey + 1, {0, -1});

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t odate = ord.o_orderdate[i];
        if (odate >= date_lo && odate <= date_hi) {
            int32_t ck = ord.o_custkey[i];
            int32_t cust_nkey = (ck <= max_custkey) ? cust_nation[ck] : -1;
            bool is_america = (cust_nkey >= 0 && cust_nkey < 26) ? nation_in_region[cust_nkey] : false;

            int16_t year = static_cast<int16_t>(1970 + odate / 365);

            order_info[ord.o_orderkey[i]] = {year, is_america ? int8_t(1) : int8_t(0)};
        }
    }

    // ========================================================================
    // Phase 2: 并行扫描 lineitem + 线程局部聚合
    // ========================================================================

    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    if (num_threads > 8) num_threads = 8;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 线程局部聚合结果
    struct ThreadLocalResult {
        std::array<int64_t, 2> brazil_vol{};
        std::array<int64_t, 2> total_vol{};
    };

    std::vector<ThreadLocalResult> thread_results(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 捕获所有需要的数据的指针/引用
    const int32_t* l_partkey = li.l_partkey.data();
    const int32_t* l_orderkey = li.l_orderkey.data();
    const int32_t* l_suppkey = li.l_suppkey.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();
    const uint8_t* valid_part_ptr = valid_part.data();
    const OrderInfo* order_info_ptr = order_info.data();
    const int32_t* supp_nation_ptr = supp_nation.data();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([=, &thread_results]() {
            auto& local = thread_results[t];

            for (size_t i = start; i < end; ++i) {
                // 快速过滤: part
                int32_t pk = l_partkey[i];
                if (pk > max_partkey || !valid_part_ptr[pk]) continue;

                // 快速过滤: order
                int32_t ok = l_orderkey[i];
                if (ok > max_orderkey) continue;

                const auto& oi = order_info_ptr[ok];
                if (oi.is_america != 1) continue;

                // 快速过滤: supplier
                int32_t sk = l_suppkey[i];
                if (sk > max_suppkey) continue;
                int32_t snk = supp_nation_ptr[sk];
                if (snk < 0) continue;

                // 计算 volume
                int64_t volume = static_cast<int64_t>(l_extendedprice[i]) *
                                 (10000 - l_discount[i]) / 10000;

                int year_idx = oi.year - 1995;
                if (year_idx < 0 || year_idx > 1) continue;

                local.total_vol[year_idx] += volume;
                if (snk == target_nationkey) {
                    local.brazil_vol[year_idx] += volume;
                }
            }
        });
    }

    // 等待所有线程完成
    for (auto& th : threads) th.join();

    // ========================================================================
    // Phase 3: 合并聚合结果
    // ========================================================================

    std::array<int64_t, 2> brazil_vol{};
    std::array<int64_t, 2> total_vol{};

    for (const auto& local : thread_results) {
        brazil_vol[0] += local.brazil_vol[0];
        brazil_vol[1] += local.brazil_vol[1];
        total_vol[0] += local.total_vol[0];
        total_vol[1] += local.total_vol[1];
    }

    // 计算市场份额
    double mkt_share_1995 = total_vol[0] > 0 ?
        static_cast<double>(brazil_vol[0]) / total_vol[0] : 0.0;
    double mkt_share_1996 = total_vol[1] > 0 ?
        static_cast<double>(brazil_vol[1]) / total_vol[1] : 0.0;

    // 防止优化器消除
    volatile double sink = mkt_share_1995 + mkt_share_1996;
    (void)sink;
}

}  // namespace ops_v42
}  // namespace tpch
}  // namespace thunderduck
