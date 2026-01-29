/**
 * ThunderDuck TPC-H V36 优化算子实现
 *
 * 相关子查询解关联优化
 */

#include "tpch_operators_v36.h"
#include <algorithm>
#include <cstring>

namespace thunderduck {
namespace tpch {
namespace ops_v36 {

// ============================================================================
// Q17 优化实现
// ============================================================================

Q17Optimizer::Result Q17Optimizer::execute(
    const int32_t* p_partkey,
    const std::vector<std::string>& p_brand,
    const std::vector<std::string>& p_container,
    size_t part_count,
    const int32_t* l_partkey,
    const int64_t* l_quantity,
    const int64_t* l_extendedprice,
    size_t lineitem_count,
    const std::string& target_brand,
    const std::string& target_container,
    double quantity_factor,
    size_t num_threads
) {
    // ========================================================================
    // Phase 1: 找出目标 part (p_brand = target, p_container = target)
    // ========================================================================
    std::unordered_set<int32_t> target_parts;
    target_parts.reserve(part_count / 1000);

    for (size_t i = 0; i < part_count; ++i) {
        if (p_brand[i] == target_brand && p_container[i] == target_container) {
            target_parts.insert(p_partkey[i]);
        }
    }

    if (target_parts.empty()) {
        return {0, 0.0};
    }

    // ========================================================================
    // Phase 2: 并行扫描 lineitem - 计算 AVG 和收集待评估行
    // ========================================================================
    std::vector<ThreadLocalState> thread_states(num_threads);
    std::vector<std::thread> threads;

    size_t chunk_size = (lineitem_count + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, lineitem_count);

        threads.emplace_back([&, t, start, end]() {
            auto& state = thread_states[t];

            // 预分配
            state.qty_states.reserve(target_parts.size());
            state.pending_rows.reserve(target_parts.size());

            // 8 路展开扫描
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                // 预取
                __builtin_prefetch(l_partkey + i + 64, 0, 3);
                __builtin_prefetch(l_quantity + i + 64, 0, 3);
                __builtin_prefetch(l_extendedprice + i + 64, 0, 3);

                #define PROCESS_ROW(idx) \
                    do { \
                        int32_t pk = l_partkey[i + idx]; \
                        if (target_parts.count(pk)) { \
                            auto& qs = state.qty_states[pk]; \
                            qs.sum += l_quantity[i + idx]; \
                            qs.count++; \
                            state.pending_rows[pk].push_back({ \
                                l_quantity[i + idx], \
                                l_extendedprice[i + idx] \
                            }); \
                        } \
                    } while (0)

                PROCESS_ROW(0);
                PROCESS_ROW(1);
                PROCESS_ROW(2);
                PROCESS_ROW(3);
                PROCESS_ROW(4);
                PROCESS_ROW(5);
                PROCESS_ROW(6);
                PROCESS_ROW(7);

                #undef PROCESS_ROW
            }

            // 处理剩余
            for (; i < end; ++i) {
                int32_t pk = l_partkey[i];
                if (target_parts.count(pk)) {
                    auto& qs = state.qty_states[pk];
                    qs.sum += l_quantity[i];
                    qs.count++;
                    state.pending_rows[pk].push_back({l_quantity[i], l_extendedprice[i]});
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // ========================================================================
    // Phase 3: 合并线程结果
    // ========================================================================
    CompactHashTable<int32_t, QuantityState> merged_states;
    merged_states.reserve(target_parts.size());

    // 合并 qty_states
    for (size_t t = 0; t < num_threads; ++t) {
        thread_states[t].qty_states.for_each([&merged_states](int32_t pk, const QuantityState& qs) {
            auto& m = merged_states[pk];
            m.sum += qs.sum;
            m.count += qs.count;
        });
    }

    // ========================================================================
    // Phase 4: 并行过滤并聚合
    // ========================================================================
    std::atomic<int64_t> total_price{0};

    // 收集所有 partkey
    std::vector<int32_t> partkeys;
    partkeys.reserve(target_parts.size());
    merged_states.for_each([&partkeys](int32_t pk, const QuantityState&) {
        partkeys.push_back(pk);
    });

    size_t pk_chunk = (partkeys.size() + num_threads - 1) / num_threads;
    threads.clear();

    for (size_t t = 0; t < num_threads; ++t) {
        size_t pk_start = t * pk_chunk;
        size_t pk_end = std::min(pk_start + pk_chunk, partkeys.size());

        threads.emplace_back([&, t, pk_start, pk_end]() {
            int64_t local_sum = 0;

            for (size_t pi = pk_start; pi < pk_end; ++pi) {
                int32_t pk = partkeys[pi];

                // 获取阈值
                const auto* qs = merged_states.find(pk);
                if (!qs || qs->count == 0) continue;

                int64_t threshold = static_cast<int64_t>(qs->avg() * quantity_factor);

                // 遍历所有线程的 pending_rows
                for (size_t ts = 0; ts < num_threads; ++ts) {
                    const auto* rows = thread_states[ts].pending_rows.find(pk);
                    if (!rows) continue;

                    // SIMD 优化: 批量比较
                    #ifdef __aarch64__
                    if (rows->size() >= 4) {
                        size_t ri = 0;
                        int64x2_t threshold_vec = vdupq_n_s64(threshold);
                        int64x2_t sum_vec = vdupq_n_s64(0);

                        for (; ri + 2 <= rows->size(); ri += 2) {
                            int64_t qty0 = (*rows)[ri].quantity;
                            int64_t qty1 = (*rows)[ri + 1].quantity;
                            int64_t price0 = (*rows)[ri].extendedprice;
                            int64_t price1 = (*rows)[ri + 1].extendedprice;

                            int64x2_t qty_vec = {qty0, qty1};
                            int64x2_t price_vec = {price0, price1};

                            // qty < threshold
                            uint64x2_t mask = vcltq_s64(qty_vec, threshold_vec);

                            // 条件累加
                            int64x2_t masked = vandq_s64(vreinterpretq_s64_u64(mask), price_vec);
                            sum_vec = vaddq_s64(sum_vec, masked);
                        }

                        local_sum += vgetq_lane_s64(sum_vec, 0) + vgetq_lane_s64(sum_vec, 1);

                        // 处理剩余
                        for (; ri < rows->size(); ++ri) {
                            if ((*rows)[ri].quantity < threshold) {
                                local_sum += (*rows)[ri].extendedprice;
                            }
                        }
                    } else
                    #endif
                    {
                        // 标量版本
                        for (const auto& row : *rows) {
                            if (row.quantity < threshold) {
                                local_sum += row.extendedprice;
                            }
                        }
                    }
                }
            }

            total_price.fetch_add(local_sum, std::memory_order_relaxed);
        });
    }

    for (auto& th : threads) th.join();

    // ========================================================================
    // 返回结果
    // ========================================================================
    Result result;
    result.sum_extendedprice = total_price.load();
    // 定点数转换: l_extendedprice 是 * 10000 的
    result.avg_yearly = static_cast<double>(result.sum_extendedprice) / 7.0 / 10000.0;
    return result;
}

// ============================================================================
// Q20 优化实现
// ============================================================================

Q20Optimizer::Result Q20Optimizer::execute(
    const int32_t* s_suppkey,
    const int32_t* s_nationkey,
    const std::vector<std::string>& s_name,
    const std::vector<std::string>& s_address,
    size_t supplier_count,
    const int32_t* n_nationkey,
    const std::vector<std::string>& n_name,
    size_t nation_count,
    const int32_t* p_partkey,
    const std::vector<std::string>& p_name,
    size_t part_count,
    const int32_t* ps_partkey,
    const int32_t* ps_suppkey,
    const int32_t* ps_availqty,
    size_t partsupp_count,
    const int32_t* l_partkey,
    const int32_t* l_suppkey,
    const int64_t* l_quantity,
    const int32_t* l_shipdate,
    size_t lineitem_count,
    const std::string& part_prefix,
    const std::string& target_nation,
    int32_t date_lo,
    int32_t date_hi,
    double quantity_factor
) {
    Result result;

    // ========================================================================
    // Phase 1: 找目标国家
    // ========================================================================
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // ========================================================================
    // Phase 2: 找 "forest%" 开头的 part
    // ========================================================================
    std::unordered_set<int32_t> forest_parts;
    for (size_t i = 0; i < part_count; ++i) {
        if (p_name[i].compare(0, part_prefix.size(), part_prefix) == 0) {
            forest_parts.insert(p_partkey[i]);
        }
    }

    // ========================================================================
    // Phase 3: 计算 SUM(l_quantity) GROUP BY (l_partkey, l_suppkey) WHERE 日期
    // ========================================================================
    std::unordered_map<int64_t, int64_t> qty_sum;  // key = (partkey << 32) | suppkey

    for (size_t i = 0; i < lineitem_count; ++i) {
        if (l_shipdate[i] >= date_lo && l_shipdate[i] < date_hi) {
            if (forest_parts.count(l_partkey[i])) {
                int64_t key = (static_cast<int64_t>(l_partkey[i]) << 32) |
                              static_cast<uint32_t>(l_suppkey[i]);
                qty_sum[key] += l_quantity[i];
            }
        }
    }

    // ========================================================================
    // Phase 4: 找满足条件的 suppkey
    // ========================================================================
    std::unordered_set<int32_t> valid_suppkeys;

    for (size_t i = 0; i < partsupp_count; ++i) {
        if (!forest_parts.count(ps_partkey[i])) continue;

        int64_t key = (static_cast<int64_t>(ps_partkey[i]) << 32) |
                      static_cast<uint32_t>(ps_suppkey[i]);

        auto it = qty_sum.find(key);
        if (it != qty_sum.end()) {
            // ps_availqty > 0.5 * SUM(l_quantity)
            int64_t threshold = static_cast<int64_t>(it->second * quantity_factor);
            if (ps_availqty[i] > threshold) {
                valid_suppkeys.insert(ps_suppkey[i]);
            }
        }
    }

    // ========================================================================
    // Phase 5: 过滤 supplier (国家 + IN valid_suppkeys)
    // ========================================================================
    std::vector<std::pair<std::string, std::string>> suppliers;

    for (size_t i = 0; i < supplier_count; ++i) {
        if (s_nationkey[i] == target_nationkey && valid_suppkeys.count(s_suppkey[i])) {
            suppliers.emplace_back(s_name[i], s_address[i]);
        }
    }

    // 排序
    std::sort(suppliers.begin(), suppliers.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    result.suppliers = std::move(suppliers);
    return result;
}

// ============================================================================
// V36 查询入口
// ============================================================================

void run_q17_v36(TPCHDataLoader& loader) {
    const auto& part = loader.part();
    const auto& li = loader.lineitem();

    auto result = Q17Optimizer::execute(
        part.p_partkey.data(),
        part.p_brand,
        part.p_container,
        part.count,
        li.l_partkey.data(),
        li.l_quantity.data(),
        li.l_extendedprice.data(),
        li.count,
        "Brand#23",
        "MED BOX",
        0.2,
        8  // num_threads
    );

    // 结果用于验证
    volatile double avg_yearly = result.avg_yearly;
    (void)avg_yearly;
}

void run_q20_v36(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& part = loader.part();
    const auto& ps = loader.partsupp();
    const auto& li = loader.lineitem();

    // 日期常量: 1994-01-01 和 1995-01-01
    constexpr int32_t DATE_1994_01_01 = 8766;
    constexpr int32_t DATE_1995_01_01 = 9131;

    auto result = Q20Optimizer::execute(
        supp.s_suppkey.data(),
        supp.s_nationkey.data(),
        supp.s_name,
        supp.s_address,
        supp.count,
        nat.n_nationkey.data(),
        nat.n_name,
        nat.count,
        part.p_partkey.data(),
        part.p_name,
        part.count,
        ps.ps_partkey.data(),
        ps.ps_suppkey.data(),
        ps.ps_availqty.data(),
        ps.count,
        li.l_partkey.data(),
        li.l_suppkey.data(),
        li.l_quantity.data(),
        li.l_shipdate.data(),
        li.count,
        "forest",
        "CANADA",
        DATE_1994_01_01,
        DATE_1995_01_01,
        0.5
    );

    // 结果用于验证
    volatile size_t count = result.suppliers.size();
    (void)count;
}

} // namespace ops_v36
} // namespace tpch
} // namespace thunderduck
