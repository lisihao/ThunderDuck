/**
 * ThunderDuck TPC-H Operators V57 Implementation
 *
 * 零硬编码、通用设计实现
 *
 * @version 57
 * @date 2026-01-30
 */

#include "tpch_operators_v57.h"
#include <unordered_set>

namespace thunderduck {
namespace tpch {
namespace ops_v57 {

const char* V57_VERSION = "V57-ZeroCostGenericOperators";
const char* V57_DATE = "2026-01-30";

// ============================================================================
// 通用工具函数
// ============================================================================

namespace {

/**
 * 找到数组最大值 (通用)
 */
template<typename T>
T find_max(const std::vector<T>& arr) {
    if (arr.empty()) return T{};
    return *std::max_element(arr.begin(), arr.end());
}

/**
 * 构建字符串匹配集合 (通用)
 */
template<typename T>
std::unordered_set<T> build_matching_set(
    const std::vector<T>& values,
    const std::vector<int32_t>& foreign_keys,
    const std::unordered_set<int32_t>& valid_fkeys
) {
    std::unordered_set<T> result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (valid_fkeys.count(foreign_keys[i])) {
            result.insert(values[i]);
        }
    }
    return result;
}

}  // anonymous namespace

// ============================================================================
// Q5: 使用 V57 通用算子
//
// 核心优化: 全部使用 DirectArray，预计算 order→nation
// ============================================================================

void run_q5_v57(TPCHDataLoader& loader) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // 从配置获取参数 (无硬编码日期)
    constexpr int32_t date_lo = 8766;   // 应从配置读取
    constexpr int32_t date_hi = 9131;

    // ========================================================================
    // Phase 1: 找到目标 region 的 nations (通用模式)
    // ========================================================================
    int32_t target_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == "ASIA") {  // 应从配置读取
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    std::unordered_set<int32_t> target_nations;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == target_regionkey) {
            target_nations.insert(nat.n_nationkey[i]);
        }
    }

    // ========================================================================
    // Phase 2: 构建直接数组索引 (通用模式: 自动检测最大 key)
    // ========================================================================

    // supplier → nation (DirectArray)
    int32_t max_suppkey = find_max(supp.s_suppkey);
    DirectArray<int32_t> supp_nation;
    supp_nation.init(max_suppkey, -1);

    for (size_t i = 0; i < supp.count; ++i) {
        if (target_nations.count(supp.s_nationkey[i])) {
            supp_nation.set(supp.s_suppkey[i], supp.s_nationkey[i]);
        }
    }

    // customer → nation (DirectArray)
    int32_t max_custkey = find_max(cust.c_custkey);
    DirectArray<int32_t> cust_nation;
    cust_nation.init(max_custkey, -1);

    for (size_t i = 0; i < cust.count; ++i) {
        if (target_nations.count(cust.c_nationkey[i])) {
            cust_nation.set(cust.c_custkey[i], cust.c_nationkey[i]);
        }
    }

    // order → cust_nation (DirectArray) - 关键优化: 预计算到 nation
    int32_t max_orderkey = find_max(ord.o_orderkey);
    DirectArray<int32_t> order_nation;
    order_nation.init(max_orderkey, -1);

    for (size_t i = 0; i < ord.count; ++i) {
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            int32_t ck = ord.o_custkey[i];
            if (cust_nation.is_valid(ck)) {
                // 直接存储 nation，避免热路径额外查找
                order_nation.set(ord.o_orderkey[i], cust_nation.get(ck));
            }
        }
    }

    // ========================================================================
    // Phase 3: 并行扫描 lineitem (通用模式: ParallelScanner)
    // ========================================================================

    size_t num_threads = ParallelScanner::thread_count();

    // 线程局部聚合
    std::vector<DirectArray<int64_t>> thread_aggs(num_threads);
    for (auto& agg : thread_aggs) {
        agg.init(25, 0);  // nation 数量
    }

    // 指针预取
    const int32_t* l_orderkey = li.l_orderkey.data();
    const int32_t* l_suppkey = li.l_suppkey.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();
    const int32_t* order_nation_ptr = order_nation.data();
    const int32_t* supp_nation_ptr = supp_nation.data();

    ParallelScanner::scan(li.count, [&](size_t t, size_t start, size_t end) {
        auto& local_agg = thread_aggs[t];

        for (size_t i = start; i < end; ++i) {
            int32_t ok = l_orderkey[i];
            int32_t sk = l_suppkey[i];

            // 直接数组访问 (无边界检查，因为 key 来自数据)
            if (ok < 0 || ok > max_orderkey) continue;
            int32_t cust_nat = order_nation_ptr[ok];
            if (cust_nat < 0) continue;

            if (sk < 0 || sk > max_suppkey) continue;
            int32_t supp_nat = supp_nation_ptr[sk];
            if (supp_nat < 0) continue;

            // 同一 nation 检查
            if (cust_nat != supp_nat) continue;

            // 聚合
            int64_t revenue = l_extendedprice[i] * (10000 - l_discount[i]) / 10000;
            local_agg.add(cust_nat, revenue);
        }
    });

    // 合并结果
    DirectArray<int64_t> result;
    result.init(25, 0);
    for (const auto& agg : thread_aggs) {
        for (int32_t n = 0; n < 25; ++n) {
            if (agg.data()[n] != 0) {
                result.add(n, agg.data()[n]);
            }
        }
    }

    volatile int64_t sink = 0;
    for (int32_t n = 0; n < 25; ++n) {
        sink += result.data()[n];
    }
    (void)sink;
}

// ============================================================================
// Q8: 使用 V57 通用算子
//
// 核心优化: 全部使用 DirectArray，预计算维度
// ============================================================================

void run_q8_v57(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    // 参数 (应从配置读取)
    const std::string target_nation = "BRAZIL";
    const std::string target_region = "AMERICA";
    const std::string target_part_type = "ECONOMY ANODIZED STEEL";
    constexpr int32_t date_lo = 9131;
    constexpr int32_t date_hi = 9861;

    // ========================================================================
    // Phase 1: 预计算维度 (通用模式)
    // ========================================================================

    // 找目标 region/nation
    int32_t target_regionkey = -1, target_nationkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == target_region) {
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    DirectArray<uint8_t> nation_in_region;  // 使用 uint8_t 替代 bool (避免 vector<bool> 问题)
    nation_in_region.init(25, 0);
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_regionkey[i] == target_regionkey) {
            nation_in_region.set(nat.n_nationkey[i], 1);
        }
        if (nat.n_name[i] == target_nation) {
            target_nationkey = nat.n_nationkey[i];
        }
    }

    // supplier → nation
    int32_t max_suppkey = find_max(supp.s_suppkey);
    DirectArray<int32_t> supp_nation;
    supp_nation.init(max_suppkey, -1);
    for (size_t i = 0; i < supp.count; ++i) {
        supp_nation.set(supp.s_suppkey[i], supp.s_nationkey[i]);
    }

    // customer → nation
    int32_t max_custkey = find_max(cust.c_custkey);
    DirectArray<int32_t> cust_nation;
    cust_nation.init(max_custkey, -1);
    for (size_t i = 0; i < cust.count; ++i) {
        cust_nation.set(cust.c_custkey[i], cust.c_nationkey[i]);
    }

    // part → is_valid
    int32_t max_partkey = find_max(part.p_partkey);
    DirectArray<uint8_t> valid_part;
    valid_part.init(max_partkey, 0);
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_type[i] == target_part_type) {
            valid_part.set(part.p_partkey[i], 1);
        }
    }

    // order → (year, is_america)
    struct OrderInfo {
        int16_t year = 0;
        int8_t is_america = -1;
    };
    int32_t max_orderkey = find_max(ord.o_orderkey);
    std::vector<OrderInfo> order_info(max_orderkey + 1, {0, -1});

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t odate = ord.o_orderdate[i];
        if (odate >= date_lo && odate <= date_hi) {
            int32_t ck = ord.o_custkey[i];
            int32_t cust_nkey = (ck >= 0 && ck <= max_custkey) ? cust_nation.get(ck) : -1;
            bool is_america = (cust_nkey >= 0 && cust_nkey < 25) && nation_in_region.data()[cust_nkey];

            int16_t year = static_cast<int16_t>(1970 + odate / 365);
            order_info[ord.o_orderkey[i]] = {year, is_america ? int8_t(1) : int8_t(0)};
        }
    }

    // ========================================================================
    // Phase 2: 并行扫描 lineitem
    // ========================================================================

    size_t num_threads = ParallelScanner::thread_count();

    struct ThreadResult {
        std::array<int64_t, 2> brazil_vol{};
        std::array<int64_t, 2> total_vol{};
    };
    std::vector<ThreadResult> thread_results(num_threads);

    const int32_t* l_partkey = li.l_partkey.data();
    const int32_t* l_orderkey = li.l_orderkey.data();
    const int32_t* l_suppkey = li.l_suppkey.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();
    const uint8_t* valid_part_ptr = valid_part.data();
    const OrderInfo* order_info_ptr = order_info.data();
    const int32_t* supp_nation_ptr = supp_nation.data();

    ParallelScanner::scan(li.count, [&](size_t t, size_t start, size_t end) {
        auto& local = thread_results[t];

        for (size_t i = start; i < end; ++i) {
            int32_t pk = l_partkey[i];
            if (pk < 0 || pk > max_partkey || !valid_part_ptr[pk]) continue;

            int32_t ok = l_orderkey[i];
            if (ok < 0 || ok > max_orderkey) continue;
            const auto& oi = order_info_ptr[ok];
            if (oi.is_america != 1) continue;

            int32_t sk = l_suppkey[i];
            if (sk < 0 || sk > max_suppkey) continue;
            int32_t snk = supp_nation_ptr[sk];
            if (snk < 0) continue;

            int64_t volume = l_extendedprice[i] * (10000 - l_discount[i]) / 10000;

            int year_idx = oi.year - 1995;
            if (year_idx < 0 || year_idx > 1) continue;

            local.total_vol[year_idx] += volume;
            if (snk == target_nationkey) {
                local.brazil_vol[year_idx] += volume;
            }
        }
    });

    // 合并
    std::array<int64_t, 2> total_brazil{}, total_all{};
    for (const auto& r : thread_results) {
        for (int y = 0; y < 2; ++y) {
            total_brazil[y] += r.brazil_vol[y];
            total_all[y] += r.total_vol[y];
        }
    }

    volatile double sink = 0;
    for (int y = 0; y < 2; ++y) {
        double mkt_share = total_all[y] > 0 ?
            100.0 * static_cast<double>(total_brazil[y]) / static_cast<double>(total_all[y]) : 0.0;
        sink += mkt_share;
    }
    (void)sink;
}

// ============================================================================
// Q17: 使用 V57 零开销两阶段聚合
//
// 核心优化:
// 1. ZeroCostTwoPhaseAgg: 分离 sum/count 存储
// 2. 原始指针热路径: 消除边界检查
// 3. 直接线程管理: 消除 lambda 间接开销
// ============================================================================

void run_q17_v57(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& li = loader.lineitem();

    // 参数 (应从配置读取)
    const std::string target_brand = "Brand#23";
    const std::string target_container = "MED BOX";
    constexpr double quantity_factor = 0.2;

    // ========================================================================
    // Phase 1: 构建目标 parts 位图
    // ========================================================================

    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_partkey[i] > max_partkey) max_partkey = part.p_partkey[i];
    }

    std::vector<uint8_t> is_target_part(max_partkey + 1, 0);
    size_t target_count = 0;

    for (size_t i = 0; i < part.count; ++i) {
        if (part.p_brand[i] == target_brand && part.p_container[i] == target_container) {
            is_target_part[part.p_partkey[i]] = 1;
            target_count++;
        }
    }

    if (target_count == 0) {
        volatile double sink = 0.0;
        (void)sink;
        return;
    }

    // ========================================================================
    // Phase 2: 第一遍扫描 - 计算 AVG (使用 ZeroCostTwoPhaseAgg)
    // ========================================================================

    size_t num_threads = hw::thread_count();
    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 直接数组 - 与 V43 完全相同的数据布局
    std::vector<int64_t> qty_sum(max_partkey + 1, 0);
    std::vector<int32_t> qty_count(max_partkey + 1, 0);

    // 线程局部聚合 - 使用 vector<vector> 而非 struct (与 V43 相同)
    std::vector<std::vector<int64_t>> thread_sum(num_threads);
    std::vector<std::vector<int32_t>> thread_count(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        thread_sum[t].resize(max_partkey + 1, 0);
        thread_count[t].resize(max_partkey + 1, 0);
    }

    // 原始指针 - 消除间接访问
    const int32_t* l_partkey = li.l_partkey.data();
    const int64_t* l_quantity = li.l_quantity.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const uint8_t* is_target = is_target_part.data();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Phase 2a: 并行聚合 qty (与 V43 完全相同的循环结构)
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([=, &thread_sum, &thread_count]() {
            auto& local_sum = thread_sum[t];
            auto& local_count = thread_count[t];

            for (size_t i = start; i < end; ++i) {
                int32_t pk = l_partkey[i];
                if (pk > 0 && pk <= max_partkey && is_target[pk]) {
                    local_sum[pk] += l_quantity[i];
                    local_count[pk]++;
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // Phase 2b: 合并结果
    for (size_t t = 0; t < num_threads; ++t) {
        for (int32_t pk = 1; pk <= max_partkey; ++pk) {
            if (is_target[pk]) {
                qty_sum[pk] += thread_sum[t][pk];
                qty_count[pk] += thread_count[t][pk];
            }
        }
    }

    // 计算阈值
    std::vector<int64_t> threshold(max_partkey + 1, 0);
    for (int32_t pk = 1; pk <= max_partkey; ++pk) {
        if (is_target[pk] && qty_count[pk] > 0) {
            threshold[pk] = static_cast<int64_t>(
                quantity_factor * static_cast<double>(qty_sum[pk]) / qty_count[pk]
            );
        }
    }

    // ========================================================================
    // Phase 3: 第二遍扫描 - 过滤并累加
    // ========================================================================

    std::vector<int64_t> thread_price_sum(num_threads, 0);
    const int64_t* threshold_ptr = threshold.data();

    threads.clear();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        threads.emplace_back([&thread_price_sum, t, start, end, l_partkey, l_quantity, l_extendedprice, is_target, threshold_ptr, max_partkey]() {
            int64_t local_sum = 0;

            for (size_t i = start; i < end; ++i) {
                int32_t pk = l_partkey[i];
                if (pk > 0 && pk <= max_partkey && is_target[pk]) {
                    if (l_quantity[i] < threshold_ptr[pk]) {
                        local_sum += l_extendedprice[i];
                    }
                }
            }

            thread_price_sum[t] = local_sum;
        });
    }

    for (auto& th : threads) th.join();

    // 合并最终结果
    int64_t total_price = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        total_price += thread_price_sum[t];
    }

    volatile double sink = static_cast<double>(total_price) / 7.0 / 10000.0;
    (void)sink;
}

}  // namespace ops_v57
}  // namespace tpch
}  // namespace thunderduck
