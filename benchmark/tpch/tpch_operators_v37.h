/**
 * ThunderDuck TPC-H V37 优化算子
 *
 * 核心优化:
 * - Bitmap Anti-Join (Q22)
 * - OrderKeyState 预计算 (Q21)
 * - 复合键 Hash 优化 (Q20)
 * - 批量 SIMD 过滤 (Q17)
 *
 * @deprecated 专用类命名 (如 Q22Optimizer) 已废弃，请使用通用别名:
 *   - Q22Optimizer → BitmapAntiJoinOptimizer
 *   - Q21Optimizer → OrderKeyStateOptimizer
 *   - Q20OptimizerV2 → CompositeKeyHashOptimizer
 *   - Q17OptimizerV2 → BatchSIMDFilterOptimizer
 *   - Q8Optimizer → MultiTableJoinOptimizer
 */

#pragma once

#include "tpch_data_loader.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <algorithm>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v37 {

// ============================================================================
// Bitmap 存在性集合 - 用于稠密键的快速存在性检查
// ============================================================================

class BitmapExistenceSet {
public:
    BitmapExistenceSet() = default;

    /**
     * 构建位图
     * @param keys 键数组
     * @param count 数量
     * @param min_key 最小键值 (用于偏移)
     * @param max_key 最大键值
     */
    void build(const int32_t* keys, size_t count, int32_t min_key, int32_t max_key) {
        min_key_ = min_key;
        size_t range = static_cast<size_t>(max_key - min_key + 1);
        size_t word_count = (range + 63) / 64;
        bitmap_.resize(word_count, 0);

        // 并行填充 (8路展开)
        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            set_bit(keys[i]);
            set_bit(keys[i + 1]);
            set_bit(keys[i + 2]);
            set_bit(keys[i + 3]);
            set_bit(keys[i + 4]);
            set_bit(keys[i + 5]);
            set_bit(keys[i + 6]);
            set_bit(keys[i + 7]);
        }
        for (; i < count; ++i) {
            set_bit(keys[i]);
        }
    }

    /**
     * 检查键是否存在
     */
    __attribute__((always_inline))
    bool exists(int32_t key) const {
        int32_t idx = key - min_key_;
        if (idx < 0 || static_cast<size_t>(idx) >= bitmap_.size() * 64) return false;
        return (bitmap_[idx >> 6] >> (idx & 63)) & 1;
    }

    /**
     * 检查键是否不存在
     */
    __attribute__((always_inline))
    bool not_exists(int32_t key) const {
        return !exists(key);
    }

    /**
     * 批量检查 - 返回不存在的索引
     */
    void batch_anti_join(const int32_t* keys, size_t count,
                         std::vector<uint32_t>& result) const {
        result.clear();
        result.reserve(count / 10);  // 估计 10% 不存在

        for (size_t i = 0; i < count; ++i) {
            if (not_exists(keys[i])) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
    }

    size_t memory_bytes() const {
        return bitmap_.size() * sizeof(uint64_t);
    }

private:
    std::vector<uint64_t> bitmap_;
    int32_t min_key_ = 0;

    __attribute__((always_inline))
    void set_bit(int32_t key) {
        int32_t idx = key - min_key_;
        bitmap_[idx >> 6] |= (1ULL << (idx & 63));
    }
};

// ============================================================================
// Q22 优化器 - Bitmap Anti-Join + 融合过滤
// ============================================================================

class Q22Optimizer {
public:
    struct Result {
        std::array<int64_t, 7> counts{};      // 每个国家码的计数
        std::array<int64_t, 7> sums{};        // 每个国家码的总余额
        std::array<std::string, 7> codes;     // 国家码字符串
    };

    static Result execute(
        // Customer 表
        const int32_t* c_custkey,
        const int64_t* c_acctbal,
        const std::vector<std::string>& c_phone,
        size_t customer_count,
        // Orders 表
        const int32_t* o_custkey,
        size_t orders_count,
        // 参数
        const std::vector<std::string>& country_codes,
        int32_t substring_start = 1,
        int32_t substring_length = 2
    );
};

// ============================================================================
// Q21 OrderKeyState - 预计算订单供应商状态
// ============================================================================

struct OrderKeyState {
    uint16_t total_suppliers;       // 该订单的供应商数量
    uint16_t late_suppliers;        // 延迟交付的供应商数量
    int32_t single_late_suppkey;    // 如果只有1个延迟供应商，记录其 suppkey
};

class Q21Optimizer {
public:
    struct Result {
        std::vector<std::pair<std::string, int64_t>> suppliers;  // (s_name, numwait)
    };

    static Result execute(
        // Supplier 表
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        size_t supplier_count,
        // Lineitem 表
        const int32_t* l_orderkey,
        const int32_t* l_suppkey,
        const int32_t* l_commitdate,
        const int32_t* l_receiptdate,
        size_t lineitem_count,
        // Orders 表
        const int32_t* o_orderkey,
        const int8_t* o_orderstatus,  // F=0, O=1, P=2
        size_t orders_count,
        // Nation 表
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // 参数
        const std::string& target_nation,
        size_t limit = 100
    );

private:
    // 紧凑 Hash Table 用于 OrderKeyState
    static constexpr size_t EMPTY_KEY = static_cast<size_t>(-1);

    struct StateTable {
        std::vector<int32_t> keys;
        std::vector<OrderKeyState> values;
        size_t mask = 0;
        size_t size_ = 0;

        void reserve(size_t capacity) {
            size_t new_cap = 1;
            while (new_cap < capacity * 2) new_cap <<= 1;
            keys.resize(new_cap, static_cast<int32_t>(EMPTY_KEY));
            values.resize(new_cap);
            mask = new_cap - 1;
        }

        OrderKeyState& operator[](int32_t key) {
            size_t idx = hash(key) & mask;
            while (keys[idx] != static_cast<int32_t>(EMPTY_KEY) && keys[idx] != key) {
                idx = (idx + 1) & mask;
            }
            if (keys[idx] == static_cast<int32_t>(EMPTY_KEY)) {
                keys[idx] = key;
                values[idx] = {};
                size_++;
            }
            return values[idx];
        }

        const OrderKeyState* find(int32_t key) const {
            size_t idx = hash(key) & mask;
            while (keys[idx] != static_cast<int32_t>(EMPTY_KEY)) {
                if (keys[idx] == key) return &values[idx];
                idx = (idx + 1) & mask;
            }
            return nullptr;
        }

    private:
        static size_t hash(int32_t key) {
            size_t h = static_cast<size_t>(key);
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            return h;
        }
    };
};

// ============================================================================
// Q20 优化器 - 改进的复合键实现
// ============================================================================

class Q20OptimizerV2 {
public:
    struct Result {
        std::vector<std::pair<std::string, std::string>> suppliers;  // (s_name, s_address)
    };

    static Result execute(
        // Supplier 表
        const int32_t* s_suppkey,
        const int32_t* s_nationkey,
        const std::vector<std::string>& s_name,
        const std::vector<std::string>& s_address,
        size_t supplier_count,
        // Nation 表
        const int32_t* n_nationkey,
        const std::vector<std::string>& n_name,
        size_t nation_count,
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_name,
        size_t part_count,
        // PartSupp 表
        const int32_t* ps_partkey,
        const int32_t* ps_suppkey,
        const int32_t* ps_availqty,
        size_t partsupp_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int32_t* l_suppkey,
        const int64_t* l_quantity,
        const int32_t* l_shipdate,
        size_t lineitem_count,
        // 参数
        const std::string& part_prefix,
        const std::string& target_nation,
        int32_t date_lo,
        int32_t date_hi,
        double quantity_factor
    );
};

// ============================================================================
// Q17 优化器 V2 - 批量处理版本
// ============================================================================

class Q17OptimizerV2 {
public:
    struct Result {
        int64_t sum_extendedprice;
        double avg_yearly;
    };

    static Result execute(
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_brand,
        const std::vector<std::string>& p_container,
        size_t part_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int64_t* l_quantity,
        const int64_t* l_extendedprice,
        size_t lineitem_count,
        // 参数
        const std::string& target_brand,
        const std::string& target_container,
        double quantity_factor
    );
};

// ============================================================================
// Q8 优化器 - 优化的 Join 顺序
// ============================================================================

class Q8Optimizer {
public:
    struct Result {
        std::vector<std::pair<int32_t, double>> year_shares;  // (year, mkt_share)
    };

    static Result execute(
        TPCHDataLoader& loader,
        const std::string& target_region,
        const std::string& target_nation,
        const std::string& target_type,
        int32_t date_lo,
        int32_t date_hi
    );
};

// ============================================================================
// V37 查询入口
// ============================================================================

void run_q22_v37(TPCHDataLoader& loader);
void run_q21_v37(TPCHDataLoader& loader);
void run_q20_v37(TPCHDataLoader& loader);
void run_q17_v37(TPCHDataLoader& loader);
void run_q8_v37(TPCHDataLoader& loader);

// ============================================================================
// 通用别名 (推荐使用，取代查询专用命名)
// ============================================================================

using BitmapAntiJoinOptimizer = Q22Optimizer;
using OrderKeyStateOptimizer = Q21Optimizer;
using CompositeKeyHashOptimizer = Q20OptimizerV2;
using BatchSIMDFilterOptimizer = Q17OptimizerV2;
using MultiTableJoinOptimizer = Q8Optimizer;

} // namespace ops_v37
} // namespace tpch
} // namespace thunderduck
