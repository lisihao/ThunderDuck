/**
 * ThunderDuck - Robin Hood Hash Join Implementation
 *
 * 优化特性：
 * - Robin Hood 哈希表减少探测长度方差
 * - 批量预取减少缓存未命中
 * - SIMD 键比较加速匹配
 */

#include "thunderduck/join.h"
#include "thunderduck/memory.h"
#include <vector>
#include <cstring>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {

// ============================================================================
// Robin Hood 哈希表
// ============================================================================

class RobinHoodHashTable {
public:
    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
    static constexpr uint8_t MAX_PSL = 64;  // 最大探测序列长度

    struct Entry {
        int32_t key;
        uint32_t row_idx;
        uint8_t psl;  // Probe Sequence Length
    };

    RobinHoodHashTable(size_t expected_size) {
        // 选择 2 的幂，负载因子 0.7
        capacity_ = 1;
        while (capacity_ < expected_size * 1.5) {
            capacity_ *= 2;
        }
        mask_ = capacity_ - 1;
        size_ = 0;

        entries_ = static_cast<Entry*>(
            aligned_alloc(capacity_ * sizeof(Entry), CACHE_LINE_SIZE));

        // 初始化所有条目
        for (size_t i = 0; i < capacity_; ++i) {
            entries_[i].key = EMPTY_KEY;
            entries_[i].psl = 0;
        }
    }

    ~RobinHoodHashTable() {
        aligned_free(entries_);
    }

    void build(const int32_t* keys, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            insert(keys[i], static_cast<uint32_t>(i));
        }
    }

    void insert(int32_t key, uint32_t row_idx) {
        uint32_t hash = hash_key(key);
        size_t idx = hash & mask_;
        uint8_t psl = 0;

        Entry entry = {key, row_idx, psl};

        while (true) {
            // 空槽位，直接插入
            if (entries_[idx].key == EMPTY_KEY) {
                entries_[idx] = entry;
                ++size_;
                return;
            }

            // Robin Hood: 如果当前条目的 PSL 更小，抢夺位置
            if (entries_[idx].psl < psl) {
                std::swap(entry, entries_[idx]);
            }

            // 继续探测
            idx = (idx + 1) & mask_;
            ++psl;
            entry.psl = psl;

            // 防止无限循环
            if (psl >= MAX_PSL) {
                // 需要重新哈希，这里简化处理
                break;
            }
        }
    }

    size_t probe(const int32_t* probe_keys, size_t probe_count,
                 uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
        size_t match_count = 0;

        // 批量处理，带预取
        constexpr size_t BATCH_SIZE = 8;
        constexpr size_t PREFETCH_DIST = 16;

        for (size_t i = 0; i < probe_count; i += BATCH_SIZE) {
            // 预取下一批
            if (i + PREFETCH_DIST < probe_count) {
                for (size_t j = 0; j < BATCH_SIZE && i + PREFETCH_DIST + j < probe_count; ++j) {
                    uint32_t h = hash_key(probe_keys[i + PREFETCH_DIST + j]);
                    __builtin_prefetch(&entries_[h & mask_], 0, 3);
                }
            }

            // 处理当前批次
            size_t batch_end = std::min(i + BATCH_SIZE, probe_count);
            for (size_t j = i; j < batch_end; ++j) {
                int32_t probe_key = probe_keys[j];
                uint32_t hash = hash_key(probe_key);
                size_t idx = hash & mask_;
                uint8_t psl = 0;

                while (entries_[idx].key != EMPTY_KEY && psl <= entries_[idx].psl) {
                    if (entries_[idx].key == probe_key) {
                        out_build_indices[match_count] = entries_[idx].row_idx;
                        out_probe_indices[match_count] = static_cast<uint32_t>(j);
                        ++match_count;
                    }
                    idx = (idx + 1) & mask_;
                    ++psl;
                }
            }
        }

        return match_count;
    }

#ifdef __aarch64__
    // SIMD 加速的探测 (SOA 布局)
    size_t probe_simd(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
        size_t match_count = 0;

        for (size_t i = 0; i < probe_count; ++i) {
            int32_t probe_key = probe_keys[i];
            int32x4_t probe_vec = vdupq_n_s32(probe_key);
            uint32_t hash = hash_key(probe_key);
            size_t idx = hash & mask_;
            uint8_t psl = 0;

            // 检查连续 4 个槽位
            while (idx + 4 <= capacity_ && psl < MAX_PSL) {
                // 加载 4 个键
                alignas(16) int32_t keys[4];
                keys[0] = entries_[idx].key;
                keys[1] = entries_[(idx + 1) & mask_].key;
                keys[2] = entries_[(idx + 2) & mask_].key;
                keys[3] = entries_[(idx + 3) & mask_].key;

                int32x4_t key_vec = vld1q_s32(keys);
                uint32x4_t eq_mask = vceqq_s32(key_vec, probe_vec);

                // 检查是否有匹配
                if (vmaxvq_u32(eq_mask)) {
                    for (int k = 0; k < 4; ++k) {
                        size_t check_idx = (idx + k) & mask_;
                        if (entries_[check_idx].key == probe_key) {
                            out_build_indices[match_count] = entries_[check_idx].row_idx;
                            out_probe_indices[match_count] = static_cast<uint32_t>(i);
                            ++match_count;
                        }
                    }
                }

                // 检查是否应该继续
                bool should_continue = false;
                for (int k = 0; k < 4; ++k) {
                    size_t check_idx = (idx + k) & mask_;
                    if (entries_[check_idx].key != EMPTY_KEY &&
                        entries_[check_idx].psl >= psl + k) {
                        should_continue = true;
                    }
                }

                if (!should_continue) break;

                idx = (idx + 4) & mask_;
                psl += 4;
            }

            // 处理剩余
            while (entries_[idx].key != EMPTY_KEY && psl <= entries_[idx].psl) {
                if (entries_[idx].key == probe_key) {
                    out_build_indices[match_count] = entries_[idx].row_idx;
                    out_probe_indices[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
                idx = (idx + 1) & mask_;
                ++psl;
            }
        }

        return match_count;
    }
#endif

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    float load_factor() const { return static_cast<float>(size_) / capacity_; }

private:
    static uint32_t hash_key(int32_t key) {
#ifdef __aarch64__
        return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
#else
        uint32_t h = static_cast<uint32_t>(key);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
#endif
    }

    Entry* entries_;
    size_t capacity_;
    size_t mask_;
    size_t size_;
};

// ============================================================================
// 优化的 Hash Join
// ============================================================================

size_t hash_join_i32_v2(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result) {
    // 构建 Robin Hood 哈希表
    RobinHoodHashTable ht(build_count);
    ht.build(build_keys, build_count);

    // 确保结果缓冲区足够大
    size_t estimated_matches = std::min(build_count * probe_count, size_t(10000000));
    if (result->capacity < estimated_matches) {
        grow_join_result(result, estimated_matches);
    }

    // 探测
#ifdef __aarch64__
    size_t match_count = ht.probe_simd(probe_keys, probe_count,
                                        result->left_indices, result->right_indices);
#else
    size_t match_count = ht.probe(probe_keys, probe_count,
                                   result->left_indices, result->right_indices);
#endif

    result->count = match_count;

    // TODO: 处理 LEFT/RIGHT/FULL JOIN
    (void)join_type;

    return match_count;
}

// ============================================================================
// 分区 Hash Join (用于超大表)
// ============================================================================

size_t partitioned_hash_join_i32(const int32_t* build_keys, size_t build_count,
                                  const int32_t* probe_keys, size_t probe_count,
                                  JoinType join_type,
                                  JoinResult* result) {
    // 对于较小的表，使用单分区
    if (build_count < 1000000) {
        return hash_join_i32_v2(build_keys, build_count,
                                 probe_keys, probe_count, join_type, result);
    }

    // 分区数 (选择 2 的幂)
    constexpr size_t NUM_PARTITIONS = 16;
    constexpr size_t PARTITION_MASK = NUM_PARTITIONS - 1;

    // 分区 build 表
    std::vector<std::vector<int32_t>> build_partitions(NUM_PARTITIONS);
    std::vector<std::vector<uint32_t>> build_indices(NUM_PARTITIONS);

    for (size_t i = 0; i < build_count; ++i) {
#ifdef __aarch64__
        uint32_t hash = __crc32cw(0, static_cast<uint32_t>(build_keys[i]));
#else
        uint32_t hash = static_cast<uint32_t>(build_keys[i]);
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
#endif
        size_t partition = hash & PARTITION_MASK;
        build_partitions[partition].push_back(build_keys[i]);
        build_indices[partition].push_back(static_cast<uint32_t>(i));
    }

    // 分区 probe 表
    std::vector<std::vector<int32_t>> probe_partitions(NUM_PARTITIONS);
    std::vector<std::vector<uint32_t>> probe_indices_vec(NUM_PARTITIONS);

    for (size_t i = 0; i < probe_count; ++i) {
#ifdef __aarch64__
        uint32_t hash = __crc32cw(0, static_cast<uint32_t>(probe_keys[i]));
#else
        uint32_t hash = static_cast<uint32_t>(probe_keys[i]);
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
#endif
        size_t partition = hash & PARTITION_MASK;
        probe_partitions[partition].push_back(probe_keys[i]);
        probe_indices_vec[partition].push_back(static_cast<uint32_t>(i));
    }

    // 对每个分区进行 Join
    size_t total_matches = 0;

    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        if (build_partitions[p].empty() || probe_partitions[p].empty()) {
            continue;
        }

        RobinHoodHashTable ht(build_partitions[p].size());

        // 构建时使用原始索引
        for (size_t i = 0; i < build_partitions[p].size(); ++i) {
            ht.insert(build_partitions[p][i], build_indices[p][i]);
        }

        // 临时结果
        std::vector<uint32_t> temp_build(probe_partitions[p].size());
        std::vector<uint32_t> temp_probe(probe_partitions[p].size());

        size_t matches = ht.probe(probe_partitions[p].data(), probe_partitions[p].size(),
                                   temp_build.data(), temp_probe.data());

        // 复制结果，转换回原始索引
        if (result->count + matches > result->capacity) {
            grow_join_result(result, result->count + matches);
        }

        for (size_t i = 0; i < matches; ++i) {
            result->left_indices[result->count + i] = temp_build[i];
            result->right_indices[result->count + i] = probe_indices_vec[p][temp_probe[i]];
        }
        result->count += matches;
        total_matches += matches;
    }

    return total_matches;
}

} // namespace join
} // namespace thunderduck
