/**
 * ThunderDuck TPC-H Operators V92
 *
 * Q16 优化: 并行扫描 + 基数排序 + 无锁聚合
 *
 * 目标: 从 1.52x 提升到 2.5x+
 *
 * @version 92
 * @date 2026-02-02
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_constants.h"
#include <vector>
#include <cstdint>
#include <thread>
#include <atomic>
#include <algorithm>

namespace thunderduck {
namespace tpch {
namespace ops_v92 {

// ============================================================================
// 版本信息
// ============================================================================

constexpr const char* V92_VERSION = "V92-ParallelRadixQ16";
constexpr const char* V92_DATE = "2026-02-02";

// ============================================================================
// 编码的 Group Key (与 V27 兼容)
// ============================================================================

struct EncodedGroupKey {
    int64_t key;

    static EncodedGroupKey make(int16_t brand_id, int16_t type_id, int32_t size) {
        // brand: 6 bits, type: 12 bits, size: 8 bits
        int64_t encoded = (static_cast<int64_t>(brand_id) << 20) |
                         (static_cast<int64_t>(type_id) << 8) |
                         (static_cast<int64_t>(size) & 0xFF);
        return {encoded};
    }

    int16_t get_brand() const { return static_cast<int16_t>((key >> 20) & 0x3F); }
    int16_t get_type() const { return static_cast<int16_t>((key >> 8) & 0xFFF); }
    int32_t get_size() const { return static_cast<int32_t>(key & 0xFF); }
};

// ============================================================================
// Group-Suppkey 对 (用于排序)
// ============================================================================

struct GroupSuppPair {
    int64_t group_key;
    int32_t suppkey;

    bool operator<(const GroupSuppPair& o) const {
        if (group_key != o.group_key) return group_key < o.group_key;
        return suppkey < o.suppkey;
    }
};

// ============================================================================
// Part 表预计算缓存
// ============================================================================

struct PartInfo {
    int16_t brand_id;
    int16_t type_id;
    int32_t size;
};

class Q16PartCache {
public:
    std::vector<PartInfo> partkey_to_info;
    size_t max_partkey = 0;
    bool initialized = false;

    // 字典用于最终解码
    std::vector<std::string> brand_dict;
    std::vector<std::string> type_dict;

    void precompute(const tpch::PartColumns& part);
};

// ============================================================================
// 基数排序 (针对 GroupSuppPair)
// ============================================================================

void radix_sort_pairs(std::vector<GroupSuppPair>& pairs);

// ============================================================================
// Q16 V92: 并行扫描 + 基数排序
// ============================================================================

/**
 * Q16: Parts/Supplier Relationship (V92 优化)
 *
 * SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey)
 * FROM partsupp, part
 * WHERE p_partkey = ps_partkey
 *   AND p_brand <> 'Brand#45'
 *   AND p_type NOT LIKE 'MEDIUM POLISHED%'
 *   AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
 *   AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
 * GROUP BY p_brand, p_type, p_size
 * ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
 *
 * V92 优化:
 * 1. 并行扫描 partsupp 表 (8 线程)
 * 2. 线程本地收集器 (无锁)
 * 3. 基数排序替代 std::sort
 * 4. 缓存预取
 */
void run_q16_v92(TPCHDataLoader& loader);

/**
 * 检查 V92 是否适用
 */
bool is_v92_q16_applicable(size_t partsupp_count);

/**
 * 估算 V92 执行时间
 */
double estimate_v92_q16_time_ms(size_t partsupp_count);

} // namespace ops_v92
} // namespace tpch
} // namespace thunderduck
