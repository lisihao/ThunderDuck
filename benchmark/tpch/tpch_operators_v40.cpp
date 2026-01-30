/**
 * ThunderDuck TPC-H V40 优化算子实现
 *
 * 通用算子框架实现，消除硬编码
 *
 * @version 40.0
 * @date 2026-01-29
 */

#include "tpch_operators_v40.h"
#include "tpch_constants.h"      // 统一常量定义
#include <algorithm>
#include <cstring>

using namespace thunderduck::tpch::constants;

namespace thunderduck {
namespace tpch {
namespace ops_v40 {

// ============================================================================
// Q20 优化实现 V5 - 通用算子框架版本
// ============================================================================

Q20OptimizerV5::Result Q20OptimizerV5::execute(
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
    // Phase 1: 预计算
    // ========================================================================

    // 1.1 找目标国家
    int32_t target_nationkey = -1;
    for (size_t i = 0; i < nation_count; ++i) {
        if (n_name[i] == target_nation) {
            target_nationkey = n_nationkey[i];
            break;
        }
    }
    if (target_nationkey < 0) return result;

    // 1.2 使用 DynamicBitmapFilter 构建 forest% parts (无硬编码范围!)
    // 优化: 单遍扫描同时收集 partkey 和构建位图
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part_count; ++i) {
        if (p_partkey[i] > max_partkey) max_partkey = p_partkey[i];
    }

    std::vector<bool> is_forest_part(max_partkey + 1, false);
    for (size_t i = 0; i < part_count; ++i) {
        if (p_name[i].compare(0, part_prefix.size(), part_prefix) == 0) {
            is_forest_part[p_partkey[i]] = true;
        }
    }

    // ========================================================================
    // Phase 2: 收集 lineitem (partkey, suppkey, quantity) 并排序
    // ========================================================================

    // 直接使用紧凑结构，避免包装类开销
    struct LIRecord {
        int32_t partkey;
        int32_t suppkey;
        int64_t quantity;

        bool operator<(const LIRecord& o) const {
            if (partkey != o.partkey) return partkey < o.partkey;
            return suppkey < o.suppkey;
        }
    };

    std::vector<LIRecord> li_records;
    li_records.reserve(lineitem_count / 10);

    for (size_t i = 0; i < lineitem_count; ++i) {
        if (l_shipdate[i] < date_lo || l_shipdate[i] >= date_hi) continue;

        int32_t pk = l_partkey[i];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        li_records.push_back({pk, l_suppkey[i], l_quantity[i]});
    }

    std::sort(li_records.begin(), li_records.end());

    // ========================================================================
    // Phase 3: 单遍聚合 SUM(quantity) per (partkey, suppkey)
    // ========================================================================

    struct PSSum {
        int32_t partkey;
        int32_t suppkey;
        int64_t sum_qty;
    };

    std::vector<PSSum> qty_sums;
    qty_sums.reserve(li_records.size() / 4);

    size_t i = 0;
    while (i < li_records.size()) {
        int32_t pk = li_records[i].partkey;
        int32_t sk = li_records[i].suppkey;
        int64_t sum = 0;

        while (i < li_records.size() &&
               li_records[i].partkey == pk &&
               li_records[i].suppkey == sk) {
            sum += li_records[i].quantity;
            ++i;
        }

        qty_sums.push_back({pk, sk, sum});
    }

    // ========================================================================
    // Phase 4: 收集并排序 partsupp
    // ========================================================================

    struct PSRecord {
        int32_t partkey;
        int32_t suppkey;
        int32_t availqty;

        bool operator<(const PSRecord& o) const {
            if (partkey != o.partkey) return partkey < o.partkey;
            return suppkey < o.suppkey;
        }
    };

    std::vector<PSRecord> ps_records;
    ps_records.reserve(partsupp_count / 100);

    for (size_t j = 0; j < partsupp_count; ++j) {
        int32_t pk = ps_partkey[j];
        if (pk <= 0 || pk > max_partkey || !is_forest_part[pk]) continue;

        ps_records.push_back({pk, ps_suppkey[j], ps_availqty[j]});
    }

    std::sort(ps_records.begin(), ps_records.end());

    // ========================================================================
    // Phase 5: 归并比较 - 找满足条件的 suppkey
    // ========================================================================

    // 使用 DynamicBitmapFilter 存储结果 (无硬编码!)
    generic::DynamicBitmapFilter valid_suppkeys;

    // 先收集满足条件的 suppkey
    std::vector<int32_t> matching_suppkeys;
    matching_suppkeys.reserve(qty_sums.size());

    size_t qi = 0, pi = 0;
    while (qi < qty_sums.size() && pi < ps_records.size()) {
        const auto& q = qty_sums[qi];
        const auto& p = ps_records[pi];

        if (q.partkey < p.partkey || (q.partkey == p.partkey && q.suppkey < p.suppkey)) {
            ++qi;
        } else if (q.partkey > p.partkey || (q.partkey == p.partkey && q.suppkey > p.suppkey)) {
            ++pi;
        } else {
            // 匹配
            int64_t threshold = static_cast<int64_t>(q.sum_qty * quantity_factor);
            if (p.availqty > threshold) {
                matching_suppkeys.push_back(p.suppkey);
            }
            ++qi;
            ++pi;
        }
    }

    // 构建动态位图 (自动检测范围，无硬编码!)
    valid_suppkeys.build(matching_suppkeys.data(), matching_suppkeys.size());

    // ========================================================================
    // Phase 6: 过滤 supplier
    // ========================================================================

    std::vector<std::pair<std::string, std::string>> suppliers;
    suppliers.reserve(matching_suppkeys.size());

    for (size_t j = 0; j < supplier_count; ++j) {
        int32_t sk = s_suppkey[j];
        if (s_nationkey[j] == target_nationkey && valid_suppkeys.test(sk)) {
            suppliers.emplace_back(s_name[j], s_address[j]);
        }
    }

    std::sort(suppliers.begin(), suppliers.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    result.suppliers = std::move(suppliers);
    return result;
}

// ============================================================================
// V40 查询入口
// ============================================================================

void run_q20_v40(TPCHDataLoader& loader) {
    const auto& supp = loader.supplier();
    const auto& nat = loader.nation();
    const auto& part = loader.part();
    const auto& ps = loader.partsupp();
    const auto& li = loader.lineitem();

    // TPC-H Q20 标准参数
    constexpr int32_t DATE_1994_01_01 = dates::D1994_01_01;
    constexpr int32_t DATE_1995_01_01 = dates::D1995_01_01;

    auto result = Q20OptimizerV5::execute(
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
        query_params::q20::COLOR_PREFIX,  // part_prefix
        nations::CANADA,                   // target_nation
        dates::D1994_01_01,
        dates::D1995_01_01,
        0.5                                // quantity_factor
    );

    // 防止编译器优化掉结果
    volatile size_t count = result.suppliers.size();
    (void)count;
}

} // namespace ops_v40
} // namespace tpch
} // namespace thunderduck
