/**
 * ThunderDuck V13 - Hash Join 两阶段优化
 *
 * 核心优化:
 * 1. 两阶段算法: 先计数后填充，消除动态扩容
 * 2. 紧凑哈希表: 开放寻址，缓存友好
 * 3. 批量预取: 减少 cache miss
 *
 * 目标: 0.06x → 1.5x+
 */

#include "thunderduck/join.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <vector>

namespace thunderduck {
namespace join {

// ============================================================================
// 紧凑哈希表 (开放寻址，线性探测)
// ============================================================================

namespace {

// 哈希表条目
struct HashEntry {
    int32_t key;
    uint32_t index;
    uint32_t next;  // 用于冲突链
};

// 紧凑哈希表
class CompactHashTable {
public:
    CompactHashTable(size_t build_count) {
        // 使用 2x 容量，保证负载因子 < 0.5
        capacity_ = 1;
        while (capacity_ < build_count * 2) capacity_ <<= 1;
        mask_ = capacity_ - 1;

        // 分配并初始化
        table_.resize(capacity_);
        for (auto& e : table_) {
            e.key = INT32_MIN;  // 空标记
            e.index = UINT32_MAX;
            e.next = UINT32_MAX;
        }

        // 溢出链
        overflow_.reserve(build_count / 4);
    }

    // 插入 key-index 对
    void insert(int32_t key, uint32_t index) {
        uint32_t slot = hash(key) & mask_;

        if (table_[slot].key == INT32_MIN) {
            // 空槽，直接插入
            table_[slot].key = key;
            table_[slot].index = index;
        } else if (table_[slot].key == key) {
            // 相同 key，添加到溢出链
            uint32_t overflow_idx = overflow_.size();
            overflow_.push_back({key, index, table_[slot].next});
            table_[slot].next = overflow_idx;
        } else {
            // 冲突，线性探测
            uint32_t probe = (slot + 1) & mask_;
            while (table_[probe].key != INT32_MIN && table_[probe].key != key) {
                probe = (probe + 1) & mask_;
            }
            if (table_[probe].key == INT32_MIN) {
                table_[probe].key = key;
                table_[probe].index = index;
            } else {
                // 相同 key
                uint32_t overflow_idx = overflow_.size();
                overflow_.push_back({key, index, table_[probe].next});
                table_[probe].next = overflow_idx;
            }
        }
    }

    // 计数匹配数量 (Phase 1)
    inline size_t count(int32_t key) const {
        uint32_t slot = hash(key) & mask_;
        uint32_t probe_count = 0;
        const uint32_t max_probe = 32;  // 限制探测次数

        while (probe_count < max_probe) {
            if (table_[slot].key == INT32_MIN) return 0;
            if (table_[slot].key == key) {
                // 找到，计算溢出链长度
                size_t cnt = 1;
                uint32_t next = table_[slot].next;
                while (next != UINT32_MAX) {
                    cnt++;
                    next = overflow_[next].next;
                }
                return cnt;
            }
            slot = (slot + 1) & mask_;
            probe_count++;
        }
        return 0;
    }

    // 查找并填充结果 (Phase 2)
    inline size_t find_and_fill(int32_t key, uint32_t probe_idx,
                                uint32_t* left_out, uint32_t* right_out,
                                size_t write_idx) const {
        uint32_t slot = hash(key) & mask_;
        uint32_t probe_count = 0;
        const uint32_t max_probe = 32;

        while (probe_count < max_probe) {
            if (table_[slot].key == INT32_MIN) return 0;
            if (table_[slot].key == key) {
                // 写入主条目
                left_out[write_idx] = table_[slot].index;
                right_out[write_idx] = probe_idx;
                size_t cnt = 1;

                // 写入溢出链
                uint32_t next = table_[slot].next;
                while (next != UINT32_MAX) {
                    left_out[write_idx + cnt] = overflow_[next].index;
                    right_out[write_idx + cnt] = probe_idx;
                    cnt++;
                    next = overflow_[next].next;
                }
                return cnt;
            }
            slot = (slot + 1) & mask_;
            probe_count++;
        }
        return 0;
    }

private:
    inline uint32_t hash(int32_t key) const {
        // Murmur3 finalizer
        uint32_t h = static_cast<uint32_t>(key);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    std::vector<HashEntry> table_;
    std::vector<HashEntry> overflow_;
    size_t capacity_;
    uint32_t mask_;
};

}  // anonymous namespace

// ============================================================================
// V13 两阶段 Hash Join
// ============================================================================

size_t hash_join_i32_v13(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // ========== 阶段 0: 构建哈希表 ==========
    CompactHashTable ht(build_count);
    for (size_t i = 0; i < build_count; i++) {
        ht.insert(build_keys[i], static_cast<uint32_t>(i));
    }

    // ========== 阶段 1: 计数 ==========
    size_t total_matches = 0;

    // SIMD 预取 + 批量计数
    constexpr size_t BATCH_SIZE = 8;
    size_t i = 0;

    for (; i + BATCH_SIZE <= probe_count; i += BATCH_SIZE) {
        // 预取下一批
        if (i + BATCH_SIZE * 2 < probe_count) {
            __builtin_prefetch(&probe_keys[i + BATCH_SIZE * 2], 0, 0);
        }

        // 批量计数
        for (size_t j = 0; j < BATCH_SIZE; j++) {
            total_matches += ht.count(probe_keys[i + j]);
        }
    }

    // 处理剩余
    for (; i < probe_count; i++) {
        total_matches += ht.count(probe_keys[i]);
    }

    // 无匹配，提前返回
    if (total_matches == 0) {
        result->count = 0;
        return 0;
    }

    // ========== 阶段 2: 预分配 + 填充 ==========
    // 使用 ensure 确保精确容量，避免多次扩容
    ensure_join_result_capacity(result, total_matches);

    size_t write_idx = 0;
    i = 0;

    for (; i + BATCH_SIZE <= probe_count; i += BATCH_SIZE) {
        // 预取下一批
        if (i + BATCH_SIZE * 2 < probe_count) {
            __builtin_prefetch(&probe_keys[i + BATCH_SIZE * 2], 0, 0);
        }

        // 批量填充
        for (size_t j = 0; j < BATCH_SIZE; j++) {
            size_t cnt = ht.find_and_fill(probe_keys[i + j],
                                          static_cast<uint32_t>(i + j),
                                          result->left_indices,
                                          result->right_indices,
                                          write_idx);
            write_idx += cnt;
        }
    }

    // 处理剩余
    for (; i < probe_count; i++) {
        size_t cnt = ht.find_and_fill(probe_keys[i],
                                      static_cast<uint32_t>(i),
                                      result->left_indices,
                                      result->right_indices,
                                      write_idx);
        write_idx += cnt;
    }

    result->count = write_idx;
    return write_idx;
}

// ============================================================================
// V13 带配置版本
// ============================================================================

size_t hash_join_i32_v13_config(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinType join_type, JoinResult* result,
                                 const JoinConfig& config) {
    // 目前只支持 INNER JOIN
    if (join_type != JoinType::INNER) {
        // 回退到 V11
        return hash_join_i32_v11(build_keys, build_count,
                                 probe_keys, probe_count,
                                 join_type, result);
    }

    return hash_join_i32_v13(build_keys, build_count,
                              probe_keys, probe_count,
                              join_type, result);
}

}  // namespace join
}  // namespace thunderduck
