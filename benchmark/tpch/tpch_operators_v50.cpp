/**
 * ThunderDuck TPC-H V50 - Q21 查询重写实现
 *
 * @version 50.0
 * @date 2026-01-29
 */

#include "tpch_operators_v50.h"
#include <algorithm>
#include <future>
#include <cstring>

#ifdef __aarch64__
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v50 {

// ============================================================================
// Q21RewriteOptimizer 实现
// ============================================================================

Q21RewriteOptimizer::Result Q21RewriteOptimizer::execute(
    const int32_t* s_suppkey,
    const int32_t* s_nationkey,
    const std::vector<std::string>& s_name,
    size_t supplier_count,
    const int32_t* l_orderkey,
    const int32_t* l_suppkey,
    const int32_t* l_commitdate,
    const int32_t* l_receiptdate,
    size_t lineitem_count,
    const int32_t* o_orderkey,
    const int8_t* o_orderstatus,
    size_t orders_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const Config& config
) {
    Result result;

    // ========================================================================
    // Phase 1: 预计算 - 目标国家供应商位图
    // ========================================================================

    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == config.target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // 找最大 suppkey
    int32_t max_suppkey = 0;
    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_suppkey[i] > max_suppkey) max_suppkey = s_suppkey[i];
    }

    // 目标国家供应商位图 + 名称索引
    std::vector<bool> is_target_supplier(max_suppkey + 1, false);
    std::vector<size_t> suppkey_to_idx(max_suppkey + 1, SIZE_MAX);

    for (size_t i = 0; i < supplier_count; ++i) {
        suppkey_to_idx[s_suppkey[i]] = i;
        if (s_nationkey[i] == target_nationkey) {
            is_target_supplier[s_suppkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 预计算 - 有效订单位图 (orderstatus = 'F')
    // ========================================================================

    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderkey[i] > max_orderkey) max_orderkey = o_orderkey[i];
    }

    std::vector<bool> is_valid_order(max_orderkey + 1, false);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == config.order_status) {
            is_valid_order[o_orderkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 3: 核心重写 - 预聚合每个 orderkey 的供应商统计
    // ========================================================================

    // 使用 Generation Counter 技术进行去重计数
    // 每个 orderkey 需要: total_suppliers (不同供应商数), late_suppliers (迟到的不同供应商数)

    struct OrderStats {
        uint16_t total_suppliers = 0;
        uint16_t late_suppliers = 0;
        int32_t target_late_suppkey = -1;  // 目标国家的迟到供应商
    };

    // 并行计算订单统计
    auto& pool = ThreadPool::instance();
    pool.prewarm(8, lineitem_count / 8);
    size_t num_threads = pool.size();

    // 先用计数排序将 lineitem 按 orderkey 分组
    std::vector<uint32_t> order_counts(max_orderkey + 2, 0);

    // 第一遍: 计数有效行
    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok > 0 && ok <= max_orderkey && is_valid_order[ok]) {
            order_counts[ok + 1]++;
        }
    }

    // 前缀和
    for (int32_t ok = 1; ok <= max_orderkey + 1; ++ok) {
        order_counts[ok] += order_counts[ok - 1];
    }

    // 第二遍: 存储紧凑数据
    struct CompactLineitem {
        int32_t suppkey;
        int8_t is_late;
    };

    size_t total_valid = order_counts[max_orderkey + 1];
    std::vector<CompactLineitem> sorted_li(total_valid);
    std::vector<uint32_t> offsets = order_counts;

    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t ok = l_orderkey[i];
        if (ok > 0 && ok <= max_orderkey && is_valid_order[ok]) {
            sorted_li[offsets[ok]++] = {
                l_suppkey[i],
                static_cast<int8_t>(l_receiptdate[i] > l_commitdate[i] ? 1 : 0)
            };
        }
    }

    // ========================================================================
    // Phase 4: 计算每个 orderkey 的统计并过滤
    // ========================================================================

    // Generation counter 去重
    struct SuppState {
        uint32_t seen_gen = 0;
        uint32_t late_gen = 0;
    };
    std::vector<SuppState> supp_state(max_suppkey + 1);

    // 收集满足条件的 (orderkey, target_suppkey) 对
    std::vector<std::pair<int32_t, int32_t>> valid_pairs;
    valid_pairs.reserve(orders_count / 10);

    uint32_t current_gen = 1;

    // 收集有效订单列表
    std::vector<int32_t> valid_orders;
    valid_orders.reserve(orders_count / 2);
    for (size_t i = 0; i < orders_count; ++i) {
        if (o_orderstatus[i] == config.order_status) {
            valid_orders.push_back(o_orderkey[i]);
        }
    }

    for (int32_t ok : valid_orders) {
        if (ok <= 0 || ok > max_orderkey) continue;

        uint32_t start = order_counts[ok];
        uint32_t end = order_counts[ok + 1];
        if (start == end) continue;

        uint16_t total_suppliers = 0;
        uint16_t late_suppliers = 0;
        int32_t target_late_suppkey = -1;

        for (uint32_t idx = start; idx < end; ++idx) {
            const auto& li = sorted_li[idx];
            int32_t sk = li.suppkey;

            if (sk <= 0 || sk > max_suppkey) continue;

            auto& state = supp_state[sk];

            // 计算 total_suppliers (不同供应商数)
            if (state.seen_gen != current_gen) {
                state.seen_gen = current_gen;
                total_suppliers++;
            }

            // 计算 late_suppliers (迟到的不同供应商数)
            if (li.is_late && state.late_gen != current_gen) {
                state.late_gen = current_gen;
                late_suppliers++;

                // 记录目标国家的迟到供应商
                if (is_target_supplier[sk]) {
                    target_late_suppkey = sk;
                }
            }
        }

        // 核心条件 (重写的关键):
        // - total_suppliers > 1: EXISTS (有其他供应商)
        // - late_suppliers == 1: NOT EXISTS (只有一个迟到) + 当前供应商迟到
        // - target_late_suppkey > 0: 迟到的是目标国家供应商
        if (total_suppliers > 1 && late_suppliers == 1 && target_late_suppkey > 0) {
            valid_pairs.emplace_back(ok, target_late_suppkey);
        }

        ++current_gen;
    }

    // ========================================================================
    // Phase 5: 聚合 - GROUP BY s_name, COUNT(*)
    // ========================================================================

    std::vector<int64_t> suppkey_counts(max_suppkey + 1, 0);

    for (const auto& [ok, sk] : valid_pairs) {
        suppkey_counts[sk]++;
    }

    // ========================================================================
    // Phase 6: Top-N 排序
    // ========================================================================

    std::vector<std::pair<int64_t, int32_t>> sorted_results;
    for (int32_t sk = 1; sk <= max_suppkey; ++sk) {
        if (suppkey_counts[sk] > 0 && is_target_supplier[sk]) {
            sorted_results.emplace_back(suppkey_counts[sk], sk);
        }
    }

    // ORDER BY numwait DESC, s_name ASC
    std::sort(sorted_results.begin(), sorted_results.end(),
        [&](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            size_t ia = suppkey_to_idx[a.second];
            size_t ib = suppkey_to_idx[b.second];
            return s_name[ia] < s_name[ib];
        });

    // LIMIT
    size_t result_count = std::min(config.limit, sorted_results.size());
    result.suppliers.reserve(result_count);

    for (size_t i = 0; i < result_count; ++i) {
        int32_t sk = sorted_results[i].second;
        int64_t cnt = sorted_results[i].first;
        size_t sidx = suppkey_to_idx[sk];
        result.suppliers.emplace_back(s_name[sidx], cnt);
    }

    return result;
}

// ============================================================================
// 入口函数
// ============================================================================

void run_q21_v50(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& li = loader.lineitem();
    const auto& ord = loader.orders();
    const auto& nat = loader.nation();

    Q21RewriteOptimizer::Config config;
    config.target_nation = "SAUDI ARABIA";
    config.order_status = 0;  // 'F'
    config.limit = 100;

    auto result = Q21RewriteOptimizer::execute(
        supp.s_suppkey.data(),
        supp.s_nationkey.data(),
        supp.s_name,
        supp.count,
        li.l_orderkey.data(),
        li.l_suppkey.data(),
        li.l_commitdate.data(),
        li.l_receiptdate.data(),
        li.count,
        ord.o_orderkey.data(),
        ord.o_orderstatus.data(),
        ord.count,
        nat.n_nationkey.data(),
        nat.n_name,
        nat.count,
        config
    );

    // 防止优化器消除
    volatile size_t sink = result.suppliers.size();
    (void)sink;
}

} // namespace ops_v50
} // namespace tpch
} // namespace thunderduck
