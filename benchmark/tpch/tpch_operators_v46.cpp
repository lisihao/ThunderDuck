/**
 * ThunderDuck TPC-H V46 通用化算子实现
 *
 * 基于 V45，消除所有硬编码，使用通用算子模板
 *
 * @version 46.0
 * @date 2026-01-29
 */

#include "tpch_operators_v46.h"
#include "tpch_operators_v25.h"  // ThreadPool

#include <algorithm>
#include <future>

namespace thunderduck {
namespace tpch {
namespace ops_v46 {

// ============================================================================
// Q14 V46: 通用化直接数组
// ============================================================================

void run_q14_v46(TPCHDataLoader& loader, const Q14Config& config) {
    const auto& li = loader.lineitem();
    const auto& part = loader.part();

    const int32_t date_lo = config.date_range.lo;
    const int32_t date_hi = config.date_range.hi;
    const std::string& type_prefix = config.type_prefix;
    const size_t prefix_len = type_prefix.length();

    // Phase 1: 使用通用 DirectArrayFilter 构建 part_is_promo
    DirectArrayFilter<uint8_t> part_filter;
    part_filter.set_default(0);  // unknown
    part_filter.set_match(1);    // promo

    // 带值构建: 0=unknown, 1=promo, 2=not promo
    part_filter.build_with_value(
        part.p_partkey.data(), part.count,
        [&](size_t i) -> uint8_t {
            bool is_promo = (part.p_type[i].size() >= prefix_len &&
                             part.p_type[i].compare(0, prefix_len, type_prefix) == 0);
            return is_promo ? 1 : 2;
        }
    );

    const int32_t max_partkey = part_filter.max_key();
    const uint8_t* promo_arr = part_filter.data();

    // Phase 2: 并行扫描 lineitem
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    std::vector<__int128> thread_promo(num_threads, 0);
    std::vector<__int128> thread_total(num_threads, 0);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    const int32_t* l_shipdate = li.l_shipdate.data();
    const int32_t* l_partkey = li.l_partkey.data();
    const int64_t* l_extendedprice = li.l_extendedprice.data();
    const int64_t* l_discount = li.l_discount.data();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([=, &thread_promo, &thread_total]() {
            __int128 local_promo = 0;
            __int128 local_total = 0;

            for (size_t i = start; i < end; ++i) {
                int32_t shipdate = l_shipdate[i];
                if (shipdate < date_lo || shipdate >= date_hi) continue;

                int32_t pkey = l_partkey[i];
                if (pkey < 0 || pkey > max_partkey) continue;

                uint8_t promo_flag = promo_arr[pkey];
                if (promo_flag == 0) continue;

                __int128 val = (__int128)l_extendedprice[i] *
                               (10000 - l_discount[i]) / 10000;
                local_total += val;
                if (promo_flag == 1) local_promo += val;
            }

            thread_promo[t] = local_promo;
            thread_total[t] = local_total;
        }));
    }

    for (auto& f : futures) f.get();

    __int128 sum_promo = 0, sum_total = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        sum_promo += thread_promo[t];
        sum_total += thread_total[t];
    }

    volatile double result = 100.0 * static_cast<double>(sum_promo) /
                             static_cast<double>(sum_total);
    (void)result;
}

// ============================================================================
// Q11 V46: 通用化位图 + 直接数组聚合
// ============================================================================

void run_q11_v46(TPCHDataLoader& loader, const Q11Config& config) {
    const auto& partsupp = loader.partsupp();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();

    const std::string& target_nation = config.target_nation;
    const double threshold_factor = config.threshold_factor;

    // Step 1: 找到目标国家 nationkey (通用化)
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nat.count; ++i) {
        if (nat.n_name[i] == target_nation) {
            target_nationkey = nat.n_nationkey[i];
            break;
        }
    }

    if (target_nationkey < 0) {
        return;  // 未找到目标国家
    }

    // Step 2: 使用通用 BitmapMembershipFilter 构建 supplier 过滤器
    BitmapMembershipFilter supp_filter;
    supp_filter.build(
        supp.s_suppkey.data(), supp.count,
        [&](size_t i) { return supp.s_nationkey[i] == target_nationkey; }
    );

    const int32_t max_suppkey = supp_filter.max_key();
    const uint8_t* bitmap = supp_filter.data();

    // Step 3: 使用通用 DirectArrayAggregator
    DirectArrayAggregator<int64_t> partkey_agg;
    partkey_agg.init_from_keys(partsupp.ps_partkey.data(), partsupp.count);
    const int32_t max_partkey = partkey_agg.max_key();

    // Step 4: 并行扫描
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, partsupp.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (partsupp.count + num_threads - 1) / num_threads;

    struct ThreadLocalData {
        DirectArrayAggregator<int64_t> agg;
        int64_t total = 0;
    };
    std::vector<ThreadLocalData> thread_data(num_threads);
    for (auto& td : thread_data) {
        td.agg.init(max_partkey);
    }

    const int32_t* ps_suppkey = partsupp.ps_suppkey.data();
    const int32_t* ps_partkey = partsupp.ps_partkey.data();
    const int64_t* ps_supplycost = partsupp.ps_supplycost.data();
    const int32_t* ps_availqty = partsupp.ps_availqty.data();

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, partsupp.count);
        if (start >= partsupp.count) break;

        futures.push_back(pool.submit([=, &thread_data]() {
            auto& local = thread_data[t];
            int64_t local_total = 0;

            for (size_t i = start; i < end; ++i) {
                int32_t sk = ps_suppkey[i];
                if (sk < 0 || sk > max_suppkey) continue;
                if (!(bitmap[sk >> 3] & (1u << (sk & 7)))) continue;

                int64_t val = ps_supplycost[i] * ps_availqty[i] / 10000;
                local.agg.add(ps_partkey[i], val);
                local_total += val;
            }

            local.total = local_total;
        }));
    }

    for (auto& f : futures) f.get();

    // Step 5: 合并
    int64_t total_value = 0;
    for (const auto& td : thread_data) {
        total_value += td.total;
    }

    DirectArrayAggregator<int64_t> merged_agg;
    merged_agg.init(max_partkey);
    for (const auto& td : thread_data) {
        merged_agg.merge(td.agg);
    }

    // Step 6: 后置过滤 (通用化阈值因子)
    int64_t threshold = static_cast<int64_t>(total_value * threshold_factor);

    std::vector<std::pair<int32_t, int64_t>> results;
    results.reserve(1000);

    merged_agg.for_each_nonzero([&](int32_t pk, int64_t val) {
        if (val > threshold) {
            results.emplace_back(pk, val);
        }
    });

    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    volatile size_t sink = results.size();
    (void)sink;
}

// ============================================================================
// Q5 V46: 通用化直接数组维度表
// ============================================================================

void run_q5_v46(TPCHDataLoader& loader, const Q5Config& config) {
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& cust = loader.customer();
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& reg = loader.region();

    const int32_t date_lo = config.date_range.lo;
    const int32_t date_hi = config.date_range.hi;
    const std::string& target_region = config.target_region;

    // Phase 1: 找到目标区域的 nations (通用化)
    int32_t target_regionkey = -1;
    for (size_t i = 0; i < reg.count; ++i) {
        if (reg.r_name[i] == target_region) {
            target_regionkey = reg.r_regionkey[i];
            break;
        }
    }

    if (target_regionkey < 0) {
        return;  // 未找到目标区域
    }

    // 动态检测 max_nationkey (消除硬编码 25)
    int32_t max_nationkey = 0;
    for (size_t i = 0; i < nat.count; ++i) {
        max_nationkey = std::max(max_nationkey, nat.n_nationkey[i]);
    }

    // 使用通用 DirectArrayFilter 标记目标区域的 nations
    std::vector<uint8_t> is_target_nation(max_nationkey + 1, 0);
    for (size_t j = 0; j < nat.count; ++j) {
        if (nat.n_regionkey[j] == target_regionkey) {
            is_target_nation[nat.n_nationkey[j]] = 1;
        }
    }

    // Phase 2: 构建 supp_nation 直接数组
    DirectArrayFilter<int8_t> supp_nation_filter;
    supp_nation_filter.set_default(-1);  // not in target region
    supp_nation_filter.build_with_value(
        supp.s_suppkey.data(), supp.count,
        [&](size_t i) -> int8_t {
            int32_t nkey = supp.s_nationkey[i];
            if (nkey >= 0 && nkey <= max_nationkey && is_target_nation[nkey]) {
                return static_cast<int8_t>(nkey);
            }
            return -1;
        }
    );

    const int32_t max_suppkey = supp_nation_filter.max_key();
    const int8_t* sn_arr = supp_nation_filter.data();

    // Phase 3: 构建 cust_nation 直接数组
    DirectArrayFilter<int8_t> cust_nation_filter;
    cust_nation_filter.set_default(-1);
    cust_nation_filter.build_with_value(
        cust.c_custkey.data(), cust.count,
        [&](size_t i) -> int8_t {
            int32_t nkey = cust.c_nationkey[i];
            if (nkey >= 0 && nkey <= max_nationkey && is_target_nation[nkey]) {
                return static_cast<int8_t>(nkey);
            }
            return -1;
        }
    );

    const int32_t max_custkey = cust_nation_filter.max_key();
    const int8_t* cn_arr = cust_nation_filter.data();

    // Phase 4: 构建 order_to_cust (使用 unordered_map，避免大数组)
    std::unordered_map<int32_t, int32_t> order_to_cust;
    order_to_cust.reserve(ord.count / 4);

    for (size_t i = 0; i < ord.count; ++i) {
        int32_t ck = ord.o_custkey[i];
        if (ord.o_orderdate[i] >= date_lo && ord.o_orderdate[i] < date_hi) {
            if (ck >= 0 && ck <= max_custkey && cn_arr[ck] >= 0) {
                order_to_cust[ord.o_orderkey[i]] = ck;
            }
        }
    }

    // Phase 5: 并行扫描 lineitem
    auto& pool = ops_v25::ThreadPool::instance();
    pool.prewarm(8, li.count / 8);
    size_t num_threads = pool.size();
    if (num_threads == 0) num_threads = 1;

    size_t chunk_size = (li.count + num_threads - 1) / num_threads;

    // 动态分配 nation 收入数组
    std::vector<std::vector<int64_t>> thread_revenues(num_threads);
    for (auto& arr : thread_revenues) {
        arr.assign(max_nationkey + 1, 0);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, li.count);
        if (start >= li.count) break;

        futures.push_back(pool.submit([=, &thread_revenues, &order_to_cust]() {
            auto& local_rev = thread_revenues[t];

            for (size_t i = start; i < end; ++i) {
                int32_t sk = li.l_suppkey[i];
                if (sk < 0 || sk > max_suppkey) continue;
                int8_t s_nat = sn_arr[sk];
                if (s_nat < 0) continue;

                auto it = order_to_cust.find(li.l_orderkey[i]);
                if (it == order_to_cust.end()) continue;
                int32_t ck = it->second;

                if (ck < 0 || ck > max_custkey) continue;
                int8_t c_nat = cn_arr[ck];
                if (c_nat < 0 || s_nat != c_nat) continue;

                int64_t revenue = static_cast<int64_t>(li.l_extendedprice[i]) *
                                  (10000 - li.l_discount[i]) / 10000;
                local_rev[s_nat] += revenue;
            }
        }));
    }

    for (auto& f : futures) f.get();

    // 合并结果
    std::vector<int64_t> nation_revenue(max_nationkey + 1, 0);
    for (const auto& local : thread_revenues) {
        for (int32_t n = 0; n <= max_nationkey; ++n) {
            nation_revenue[n] += local[n];
        }
    }

    volatile int64_t sink = 0;
    for (int64_t r : nation_revenue) sink += r;
    (void)sink;
}

}  // namespace ops_v46
}  // namespace tpch
}  // namespace thunderduck
