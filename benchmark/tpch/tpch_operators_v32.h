/**
 * ThunderDuck TPC-H 算子封装 V32
 *
 * V32 新性能基线:
 * - 自适应策略: SF < 5 使用 V27 直接数组, SF >= 5 使用紧凑 Hash
 * - 批量哈希 + SIMD 优化: 减少哈希计算开销
 * - Q3 保留 V31 (Bloom Filter 收益明显)
 *
 * 核心优化:
 * - 紧凑 Hash Table: 只存储有效条目，开放寻址 + CRC32 硬件哈希
 * - 批量查找: 8 路并行预取 + SIMD 哈希
 * - Thread-Local 聚合: 消除 atomic 竞争
 *
 * @version 32.1
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V32_H
#define TPCH_OPERATORS_V32_H

#include "tpch_data_loader.h"
#include "tpch_operators_v27.h"
#include "tpch_queries.h"
#include <cstdint>
#include <vector>
#include <atomic>
#include <thread>
#include <array>

#ifdef __aarch64__
#include <arm_acle.h>
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v32 {

using ::thunderduck::tpch::TPCHDataLoader;
using namespace ops_v27;  // 继承 V27 的工具类
namespace queries = ::thunderduck::tpch::queries;

// ============================================================================
// 自适应策略阈值
// ============================================================================

namespace adaptive {
    // SF 阈值: 低于此值使用直接数组，高于使用紧凑 Hash
    constexpr size_t SF_THRESHOLD_ORDERS = 5'000'000;    // ~SF=3.5
    constexpr size_t SF_THRESHOLD_LINEITEM = 20'000'000; // ~SF=3.5

    // 批量处理大小
    constexpr size_t BATCH_SIZE = 8;

    // 判断是否使用紧凑结构
    inline bool use_compact_structure(size_t order_count) {
        return order_count >= SF_THRESHOLD_ORDERS;
    }
}

// ============================================================================
// 批量哈希工具 (SIMD 优化)
// ============================================================================

/**
 * 批量 CRC32 哈希计算
 * 利用指令级并行，8 路同时计算
 */
class BatchHasher {
public:
    static constexpr size_t BATCH_SIZE = 8;

    /**
     * 批量计算 CRC32 哈希
     */
    static inline void batch_hash(const int32_t* keys, uint32_t* hashes) {
#ifdef __aarch64__
        // ARM: 8 路 CRC32 并行
        hashes[0] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[0]));
        hashes[1] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[1]));
        hashes[2] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[2]));
        hashes[3] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[3]));
        hashes[4] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[4]));
        hashes[5] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[5]));
        hashes[6] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[6]));
        hashes[7] = __builtin_arm_crc32w(0, static_cast<uint32_t>(keys[7]));
#else
        for (int i = 0; i < 8; ++i) {
            uint32_t k = static_cast<uint32_t>(keys[i]);
            k ^= k >> 16;
            k *= 0x85ebca6b;
            k ^= k >> 13;
            k *= 0xc2b2ae35;
            k ^= k >> 16;
            hashes[i] = k;
        }
#endif
    }

    /**
     * 单个 CRC32 哈希
     */
    static inline uint32_t hash_single(int32_t key) {
#ifdef __aarch64__
        return __builtin_arm_crc32w(0, static_cast<uint32_t>(key));
#else
        uint32_t k = static_cast<uint32_t>(key);
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k;
#endif
    }
};

// ============================================================================
// 紧凑 Hash Table (带批量查找优化)
// ============================================================================

/**
 * 紧凑 Hash Table (开放寻址 + 批量优化)
 *
 * 特点:
 * - 只存储有效条目，不预分配全量空间
 * - CRC32 硬件哈希 (ARM)
 * - 批量查找: 预取 + 8 路并行
 * - 内存: 通常节省 85-97%
 */
template<typename Value>
class CompactHashTable {
public:
    static constexpr int32_t EMPTY_KEY = INT32_MIN;

    struct Entry {
        int32_t key = EMPTY_KEY;
        Value value{};
    };

    CompactHashTable() = default;

    void init(size_t expected_count, double load_factor = 0.5) {
        size_t min_capacity = static_cast<size_t>(expected_count / load_factor) + 1;
        capacity_ = 1;
        while (capacity_ < min_capacity) capacity_ <<= 1;
        mask_ = static_cast<uint32_t>(capacity_ - 1);
        entries_.resize(capacity_);
        count_ = 0;
    }

    void insert(int32_t key, const Value& value) {
        uint32_t pos = BatchHasher::hash_single(key) & mask_;
        while (entries_[pos].key != EMPTY_KEY && entries_[pos].key != key) {
            pos = (pos + 1) & mask_;
        }
        if (entries_[pos].key == EMPTY_KEY) count_++;
        entries_[pos].key = key;
        entries_[pos].value = value;
    }

    const Value* find(int32_t key) const {
        uint32_t pos = BatchHasher::hash_single(key) & mask_;
        while (entries_[pos].key != EMPTY_KEY) {
            if (entries_[pos].key == key) return &entries_[pos].value;
            pos = (pos + 1) & mask_;
        }
        return nullptr;
    }

    /**
     * 带预计算哈希的查找 (用于批量操作)
     */
    const Value* find_with_hash(int32_t key, uint32_t hash) const {
        uint32_t pos = hash & mask_;
        while (entries_[pos].key != EMPTY_KEY) {
            if (entries_[pos].key == key) return &entries_[pos].value;
            pos = (pos + 1) & mask_;
        }
        return nullptr;
    }

    /**
     * 批量查找 (8 路并行)
     *
     * @param keys 输入键数组 (至少 8 个)
     * @param results 输出结果指针数组 (nullptr 表示未找到)
     */
    void batch_find(const int32_t* keys, const Value** results) const {
        alignas(32) uint32_t hashes[8];
        BatchHasher::batch_hash(keys, hashes);

        // 批量预取
        __builtin_prefetch(&entries_[hashes[0] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[1] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[2] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[3] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[4] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[5] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[6] & mask_], 0, 1);
        __builtin_prefetch(&entries_[hashes[7] & mask_], 0, 1);

        // 批量查找
        results[0] = find_with_hash(keys[0], hashes[0]);
        results[1] = find_with_hash(keys[1], hashes[1]);
        results[2] = find_with_hash(keys[2], hashes[2]);
        results[3] = find_with_hash(keys[3], hashes[3]);
        results[4] = find_with_hash(keys[4], hashes[4]);
        results[5] = find_with_hash(keys[5], hashes[5]);
        results[6] = find_with_hash(keys[6], hashes[6]);
        results[7] = find_with_hash(keys[7], hashes[7]);
    }

    template<typename T>
    void add_or_update(int32_t key, T delta) {
        uint32_t pos = BatchHasher::hash_single(key) & mask_;
        while (entries_[pos].key != EMPTY_KEY && entries_[pos].key != key) {
            pos = (pos + 1) & mask_;
        }
        if (entries_[pos].key == EMPTY_KEY) {
            entries_[pos].key = key;
            entries_[pos].value = static_cast<Value>(delta);
            count_++;
        } else {
            entries_[pos].value += static_cast<Value>(delta);
        }
    }

    template<typename Func>
    void for_each(Func&& func) const {
        for (const auto& e : entries_) {
            if (e.key != EMPTY_KEY) func(e.key, e.value);
        }
    }

    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }
    uint32_t get_mask() const { return mask_; }
    const Entry* data() const { return entries_.data(); }

private:
    std::vector<Entry> entries_;
    size_t capacity_ = 0;
    uint32_t mask_ = 0;
    size_t count_ = 0;
};

// ============================================================================
// 单 Hash Bloom Filter (带批量测试)
// ============================================================================

class SingleHashBloomFilter {
public:
    SingleHashBloomFilter() = default;

    void init(size_t expected_count, size_t bits_per_element = 2) {
        size_t num_bits = expected_count * bits_per_element;
        num_bits = ((num_bits + 63) / 64) * 64;
        bits_.resize(num_bits / 64, 0);
        mask_ = static_cast<uint32_t>(num_bits - 1);
    }

    inline void insert(int32_t key) {
        uint32_t hash = BatchHasher::hash_single(key);
        bits_[(hash & mask_) >> 6] |= (1ULL << ((hash & mask_) & 63));
    }

    inline bool may_contain(int32_t key) const {
        uint32_t hash = BatchHasher::hash_single(key);
        return (bits_[(hash & mask_) >> 6] >> ((hash & mask_) & 63)) & 1;
    }

    /**
     * 带预计算哈希的测试
     */
    inline bool may_contain_with_hash(uint32_t hash) const {
        return (bits_[(hash & mask_) >> 6] >> ((hash & mask_) & 63)) & 1;
    }

    /**
     * 批量测试 (返回位掩码，bit i = 1 表示 keys[i] 可能存在)
     */
    inline uint8_t batch_test(const int32_t* keys) const {
        alignas(32) uint32_t hashes[8];
        BatchHasher::batch_hash(keys, hashes);

        uint8_t result = 0;
        result |= (may_contain_with_hash(hashes[0]) ? 0x01 : 0);
        result |= (may_contain_with_hash(hashes[1]) ? 0x02 : 0);
        result |= (may_contain_with_hash(hashes[2]) ? 0x04 : 0);
        result |= (may_contain_with_hash(hashes[3]) ? 0x08 : 0);
        result |= (may_contain_with_hash(hashes[4]) ? 0x10 : 0);
        result |= (may_contain_with_hash(hashes[5]) ? 0x20 : 0);
        result |= (may_contain_with_hash(hashes[6]) ? 0x40 : 0);
        result |= (may_contain_with_hash(hashes[7]) ? 0x80 : 0);
        return result;
    }

private:
    std::vector<uint64_t> bits_;
    uint32_t mask_ = 0;
};

// ============================================================================
// Thread-Local 聚合器
// ============================================================================

template<typename Value>
struct alignas(128) ThreadLocalAggregator {
    CompactHashTable<Value> table;

    void init(size_t expected_count) { table.init(expected_count); }
    void add(int32_t key, Value delta) { table.add_or_update(key, delta); }
    void clear() { table = CompactHashTable<Value>(); }
};

// ============================================================================
// 自适应查询实现声明
// ============================================================================

/**
 * Q5 自适应版本
 * - SF < 3.5: 使用 V27 直接数组
 * - SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
 */
void run_q5_adaptive(TPCHDataLoader& loader);

/**
 * Q7 自适应版本
 * - SF < 3.5: 使用 V27 直接数组
 * - SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
 */
void run_q7_adaptive(TPCHDataLoader& loader);

/**
 * Q9 自适应版本
 * - SF < 3.5: 使用 V27 直接数组 (V25 优化)
 * - SF >= 3.5: 使用 V32 紧凑 Hash + 批量优化
 */
void run_q9_adaptive(TPCHDataLoader& loader);

/**
 * Q18 自适应版本
 * - SF < 3.5: 使用 V27 直接数组
 * - SF >= 3.5: 使用 V32 Thread-local 紧凑 Hash
 */
void run_q18_adaptive(TPCHDataLoader& loader);

// ============================================================================
// V32 紧凑 Hash 版本 (带批量优化，用于大数据量)
// ============================================================================

void run_q5_v32_batch(TPCHDataLoader& loader);
void run_q7_v32_batch(TPCHDataLoader& loader);
void run_q9_v32_batch(TPCHDataLoader& loader);
void run_q18_v32_batch(TPCHDataLoader& loader);

// ============================================================================
// V32 查询入口 - 使用自适应策略
// ============================================================================

// Category A
inline void run_q1_v32(TPCHDataLoader& loader)  { queries::run_q1(loader); }
inline void run_q3_v32(TPCHDataLoader& loader)  { run_q3_v31(loader); }  // V31 Q3 最优，保持不变
inline void run_q5_v32(TPCHDataLoader& loader)  { run_q5_adaptive(loader); }
inline void run_q6_v32(TPCHDataLoader& loader)  { run_q6_v26(loader); }
inline void run_q7_v32(TPCHDataLoader& loader)  { run_q7_adaptive(loader); }
inline void run_q9_v32(TPCHDataLoader& loader)  { run_q9_adaptive(loader); }
inline void run_q10_v32(TPCHDataLoader& loader) { run_q10_v26(loader); }
inline void run_q12_v32(TPCHDataLoader& loader) { run_q12_v27(loader); }
inline void run_q14_v32(TPCHDataLoader& loader) { run_q14_v26(loader); }
inline void run_q18_v32(TPCHDataLoader& loader) { run_q18_adaptive(loader); }

// Category B - 沿用 V27
inline void run_q2_v32(TPCHDataLoader& loader)  { queries::run_q2(loader); }
inline void run_q4_v32(TPCHDataLoader& loader)  { run_q4_v27(loader); }
inline void run_q11_v32(TPCHDataLoader& loader) { run_q11_v27(loader); }
inline void run_q15_v32(TPCHDataLoader& loader) { run_q15_v27(loader); }
inline void run_q16_v32(TPCHDataLoader& loader) { run_q16_v27(loader); }
void run_q19_v32(TPCHDataLoader& loader);  // V32: PredicatePrecomputer + 直接数组 + 并行

} // namespace ops_v32
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V32_H
