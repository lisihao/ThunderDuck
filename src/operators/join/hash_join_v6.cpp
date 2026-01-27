/**
 * ThunderDuck - Hash Join v6.0 Implementation
 *
 * 针对 Apple M4 处理器深度优化:
 * 1. 自适应策略 - 小表跳过分区，直接 Join
 * 2. 超激进预取 - 多级预取，隐藏内存延迟
 * 3. SOA 哈希表 - 缓存对齐，减少 cache miss
 * 4. 批量探测 - 8 路展开，最大化 ILP
 * 5. 最小化内存分配 - 预分配所有缓冲区
 *
 * 目标: 超越 DuckDB 的 Hash Join 性能
 */

#include "thunderduck/join.h"
#include "thunderduck/memory.h"

#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

// 缓存行大小 (M4 = 128 bytes)
constexpr size_t M4_CACHE_LINE = 128;

// 空键标记
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();

// 批量大小
constexpr size_t BATCH_SIZE = 8;

// 预取距离 (减少预取计算开销)
constexpr size_t PREFETCH_DISTANCE = 16;   // 单级预取

// 自适应阈值 (降低以使用分区优化)
constexpr size_t DIRECT_JOIN_THRESHOLD = 10000;  // < 10K build 直接 Join

// 完美哈希密度阈值
constexpr double PERFECT_HASH_DENSITY = 2.0;

} // anonymous namespace

// ============================================================================
// 哈希函数 - 使用 CRC32 硬件加速
// ============================================================================

namespace {

#ifdef __aarch64__
// 单键哈希
__attribute__((always_inline))
inline uint32_t hash_key(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

// 批量哈希 - 8 路展开
__attribute__((always_inline))
inline void hash_batch_8(const int32_t* keys, uint32_t* hashes) {
    hashes[0] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[0]));
    hashes[1] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[1]));
    hashes[2] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[2]));
    hashes[3] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[3]));
    hashes[4] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[4]));
    hashes[5] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[5]));
    hashes[6] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[6]));
    hashes[7] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[7]));
}
#else
inline uint32_t hash_key(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
#endif

} // anonymous namespace

// ============================================================================
// V6 SOA 哈希表 - 超激进预取版
// ============================================================================

class HashTableV6 {
public:
    HashTableV6() = default;

    void build(const int32_t* keys, size_t count) {
        if (count == 0) return;

        // 容量 = 2^n, 负载因子 ~0.6 (与 V3 一致)
        capacity_ = 16;
        while (capacity_ < count * 1.7) {
            capacity_ *= 2;
        }
        mask_ = capacity_ - 1;
        size_ = count;

        // 分配 128-byte 对齐的数组
        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        // 插入 - 线性探测
        for (size_t i = 0; i < count; ++i) {
            uint32_t h = hash_key(keys[i]);
            size_t idx = h & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                idx = (idx + 1) & mask_;
            }

            keys_[idx] = keys[i];
            row_indices_[idx] = static_cast<uint32_t>(i);
        }
    }

#ifdef __aarch64__
    /**
     * 优化预取探测
     *
     * 关键优化:
     * 1. 单级预取: 简化逻辑，减少预取计算开销
     * 2. 批量哈希: 8 个键一次计算
     * 3. 循环展开: 减少分支
     */
    size_t probe_aggressive(const int32_t* probe_keys, size_t probe_count,
                            uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;
        const size_t batch = BATCH_SIZE;

        size_t i = 0;
        for (; i + batch <= probe_count; i += batch) {
            // ===== 预取下一批 (简化为单级) =====
            if (i + PREFETCH_DISTANCE < probe_count) {
                // 只预取 4 个，减少开销
                uint32_t h0 = hash_key(probe_keys[i + PREFETCH_DISTANCE]);
                uint32_t h1 = hash_key(probe_keys[i + PREFETCH_DISTANCE + 1]);
                uint32_t h2 = hash_key(probe_keys[i + PREFETCH_DISTANCE + 2]);
                uint32_t h3 = hash_key(probe_keys[i + PREFETCH_DISTANCE + 3]);
                __builtin_prefetch(&keys_[h0 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h1 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h2 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h3 & mask_], 0, 3);
            }

            // ===== 批量哈希 =====
            alignas(64) uint32_t hashes[batch];
            hash_batch_8(probe_keys + i, hashes);

            // ===== 探测当前批次 =====
            for (size_t j = 0; j < batch; ++j) {
                int32_t key = probe_keys[i + j];
                size_t idx = hashes[j] & mask_;

                // 线性探测，直到空槽
                while (keys_[idx] != EMPTY_KEY) {
                    if (keys_[idx] == key) {
                        out_build[match_count] = row_indices_[idx];
                        out_probe[match_count] = static_cast<uint32_t>(i + j);
                        ++match_count;
                    }
                    idx = (idx + 1) & mask_;
                }
            }
        }

        // 处理剩余
        for (; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            uint32_t h = hash_key(key);
            size_t idx = h & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) {
                    out_build[match_count] = row_indices_[idx];
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
                idx = (idx + 1) & mask_;
            }
        }

        return match_count;
    }
#else
    size_t probe_aggressive(const int32_t* probe_keys, size_t probe_count,
                            uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            uint32_t h = hash_key(key);
            size_t idx = h & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) {
                    out_build[match_count] = row_indices_[idx];
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
                idx = (idx + 1) & mask_;
            }
        }

        return match_count;
    }
#endif

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    alignas(M4_CACHE_LINE) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// V6 完美哈希表 - SIMD 优化版
// ============================================================================

class PerfectHashV6 {
public:
    bool try_build(const int32_t* keys, size_t count) {
        if (count == 0) return false;

        // 找键范围
        int32_t min_key = keys[0];
        int32_t max_key = keys[0];
        for (size_t i = 1; i < count; ++i) {
            if (keys[i] < min_key) min_key = keys[i];
            if (keys[i] > max_key) max_key = keys[i];
        }

        // 检查密度
        int64_t range = static_cast<int64_t>(max_key) - min_key + 1;
        if (range <= 0 || range > static_cast<int64_t>(count * PERFECT_HASH_DENSITY)) {
            return false;
        }
        if (range > 10000000) {
            return false;
        }

        // 检查重复键
        std::vector<uint8_t> seen(static_cast<size_t>(range), 0);
        for (size_t i = 0; i < count; ++i) {
            size_t idx = static_cast<size_t>(keys[i] - min_key);
            if (seen[idx]) return false;
            seen[idx] = 1;
        }

        min_key_ = min_key;
        size_ = static_cast<size_t>(range);

        // 分配直接索引表
        indices_.resize(size_, UINT32_MAX);
        for (size_t i = 0; i < count; ++i) {
            indices_[keys[i] - min_key_] = static_cast<uint32_t>(i);
        }

        built_ = true;
        return true;
    }

#ifdef __aarch64__
    size_t probe(const int32_t* probe_keys, size_t probe_count,
                 uint32_t* out_build, uint32_t* out_probe) const {
        if (!built_) return 0;

        size_t match_count = 0;
        const int32_t max_offset = static_cast<int32_t>(size_ - 1);

        // SIMD 批量探测
        int32x4_t min_vec = vdupq_n_s32(min_key_);
        int32x4_t zero_vec = vdupq_n_s32(0);
        int32x4_t max_vec = vdupq_n_s32(max_offset);

        size_t i = 0;
        for (; i + 4 <= probe_count; i += 4) {
            // 预取下一批
            if (i + 16 < probe_count) {
                int32_t k0 = probe_keys[i + 16] - min_key_;
                int32_t k1 = probe_keys[i + 17] - min_key_;
                if (k0 >= 0 && k0 <= max_offset) {
                    __builtin_prefetch(&indices_[k0], 0, 3);
                }
                if (k1 >= 0 && k1 <= max_offset) {
                    __builtin_prefetch(&indices_[k1], 0, 3);
                }
            }

            int32x4_t keys_vec = vld1q_s32(probe_keys + i);
            int32x4_t offsets = vsubq_s32(keys_vec, min_vec);

            // 范围检查
            uint32x4_t valid = vandq_u32(
                vcgeq_s32(offsets, zero_vec),
                vcleq_s32(offsets, max_vec)
            );

            if (vmaxvq_u32(valid)) {
                // 有效范围内的探测
                for (int j = 0; j < 4; ++j) {
                    int32_t offset = probe_keys[i + j] - min_key_;
                    if (offset >= 0 && offset <= max_offset) {
                        uint32_t idx = indices_[offset];
                        if (idx != UINT32_MAX) {
                            out_build[match_count] = idx;
                            out_probe[match_count] = static_cast<uint32_t>(i + j);
                            ++match_count;
                        }
                    }
                }
            }
        }

        // 剩余
        for (; i < probe_count; ++i) {
            int32_t offset = probe_keys[i] - min_key_;
            if (offset >= 0 && offset <= max_offset) {
                uint32_t idx = indices_[offset];
                if (idx != UINT32_MAX) {
                    out_build[match_count] = idx;
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
            }
        }

        return match_count;
    }
#else
    size_t probe(const int32_t* probe_keys, size_t probe_count,
                 uint32_t* out_build, uint32_t* out_probe) const {
        if (!built_) return 0;

        size_t match_count = 0;
        const int32_t max_offset = static_cast<int32_t>(size_ - 1);

        for (size_t i = 0; i < probe_count; ++i) {
            int32_t offset = probe_keys[i] - min_key_;
            if (offset >= 0 && offset <= max_offset) {
                uint32_t idx = indices_[offset];
                if (idx != UINT32_MAX) {
                    out_build[match_count] = idx;
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
            }
        }

        return match_count;
    }
#endif

    bool is_built() const { return built_; }

private:
    std::vector<uint32_t> indices_;
    int32_t min_key_ = 0;
    size_t size_ = 0;
    bool built_ = false;
};

// ============================================================================
// V6 分区 Join (大表优化)
// ============================================================================

namespace {

constexpr int RADIX_BITS = 4;
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;  // 16
constexpr size_t PARTITION_MASK = NUM_PARTITIONS - 1;

inline size_t get_partition(uint32_t hash) {
    return (hash >> (32 - RADIX_BITS)) & PARTITION_MASK;
}

struct Partition {
    std::vector<int32_t> keys;
    std::vector<uint32_t> indices;
};

/**
 * 并行分区 Join - 优化版
 *
 * 优化:
 * 1. 单次直方图计算
 * 2. 预分配所有内存
 * 3. 减少动态分配
 */
size_t partitioned_join(const int32_t* build_keys, size_t build_count,
                        const int32_t* probe_keys, size_t probe_count,
                        JoinResult* result) {
    // 计算直方图
    std::array<size_t, NUM_PARTITIONS> build_hist{};
    std::array<size_t, NUM_PARTITIONS> probe_hist{};

    for (size_t i = 0; i < build_count; ++i) {
        build_hist[get_partition(hash_key(build_keys[i]))]++;
    }
    for (size_t i = 0; i < probe_count; ++i) {
        probe_hist[get_partition(hash_key(probe_keys[i]))]++;
    }

    // 分配分区内存
    std::array<Partition, NUM_PARTITIONS> build_parts;
    std::array<Partition, NUM_PARTITIONS> probe_parts;

    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        build_parts[p].keys.resize(build_hist[p]);
        build_parts[p].indices.resize(build_hist[p]);
        probe_parts[p].keys.resize(probe_hist[p]);
        probe_parts[p].indices.resize(probe_hist[p]);
    }

    // 分散数据到分区
    std::array<size_t, NUM_PARTITIONS> build_pos{};
    std::array<size_t, NUM_PARTITIONS> probe_pos{};

    for (size_t i = 0; i < build_count; ++i) {
        size_t p = get_partition(hash_key(build_keys[i]));
        size_t pos = build_pos[p]++;
        build_parts[p].keys[pos] = build_keys[i];
        build_parts[p].indices[pos] = static_cast<uint32_t>(i);
    }

    for (size_t i = 0; i < probe_count; ++i) {
        size_t p = get_partition(hash_key(probe_keys[i]));
        size_t pos = probe_pos[p]++;
        probe_parts[p].keys[pos] = probe_keys[i];
        probe_parts[p].indices[pos] = static_cast<uint32_t>(i);
    }

    // 每分区结果
    std::array<std::vector<uint32_t>, NUM_PARTITIONS> part_build_results;
    std::array<std::vector<uint32_t>, NUM_PARTITIONS> part_probe_results;

    // 并行处理分区
    std::atomic<size_t> next_partition{0};
    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size_t(4));

    auto worker = [&]() {
        while (true) {
            size_t p = next_partition.fetch_add(1);
            if (p >= NUM_PARTITIONS) break;

            if (build_parts[p].keys.empty() || probe_parts[p].keys.empty()) {
                continue;
            }

            // 构建分区哈希表
            HashTableV6 ht;
            ht.build(build_parts[p].keys.data(), build_parts[p].keys.size());

            // 预分配结果
            size_t estimated = probe_parts[p].keys.size();
            std::vector<uint32_t> temp_build(estimated);
            std::vector<uint32_t> temp_probe(estimated);

            // 探测
            size_t matches = ht.probe_aggressive(
                probe_parts[p].keys.data(), probe_parts[p].keys.size(),
                temp_build.data(), temp_probe.data());

            // 转换为原始索引并存储
            if (matches > 0) {
                part_build_results[p].resize(matches);
                part_probe_results[p].resize(matches);

                for (size_t i = 0; i < matches; ++i) {
                    part_build_results[p][i] = build_parts[p].indices[temp_build[i]];
                    part_probe_results[p][i] = probe_parts[p].indices[temp_probe[i]];
                }
            }
        }
    };

    // 启动线程
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    // 计算总匹配数
    size_t total_matches = 0;
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        total_matches += part_build_results[p].size();
    }

    // 确保结果缓冲区
    if (result->capacity < total_matches) {
        grow_join_result(result, total_matches);
    }

    // 合并结果
    size_t offset = 0;
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        size_t count = part_build_results[p].size();
        if (count > 0) {
            std::memcpy(result->left_indices + offset,
                       part_build_results[p].data(), count * sizeof(uint32_t));
            std::memcpy(result->right_indices + offset,
                       part_probe_results[p].data(), count * sizeof(uint32_t));
            offset += count;
        }
    }
    result->count = total_matches;

    return total_matches;
}

} // anonymous namespace

// ============================================================================
// V6 公开接口
// ============================================================================

size_t hash_join_i32_v6(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 预分配结果缓冲区
    size_t estimated = std::max(build_count, probe_count) * 2;
    if (result->capacity < estimated) {
        grow_join_result(result, estimated);
    }

    // ===== 策略 1: 完美哈希 (密集整数键) =====
    PerfectHashV6 perfect_ht;
    if (perfect_ht.try_build(build_keys, build_count)) {
        size_t matches = perfect_ht.probe(probe_keys, probe_count,
                                           result->left_indices, result->right_indices);
        result->count = matches;
        return matches;
    }

    // ===== 策略 2: 直接 Join (小表) =====
    if (build_count < DIRECT_JOIN_THRESHOLD) {
        HashTableV6 ht;
        ht.build(build_keys, build_count);

        size_t matches = ht.probe_aggressive(probe_keys, probe_count,
                                              result->left_indices, result->right_indices);
        result->count = matches;
        return matches;
    }

    // ===== 策略 3: 分区 Join (大表) =====
    size_t matches = partitioned_join(build_keys, build_count,
                                       probe_keys, probe_count, result);

    (void)join_type;  // TODO: LEFT/RIGHT/FULL
    return matches;
}

size_t hash_join_i32_v6_config(const int32_t* build_keys, size_t build_count,
                                const int32_t* probe_keys, size_t probe_count,
                                JoinType join_type,
                                JoinResult* result,
                                const JoinConfig& config) {
    // 使用配置 (预留接口)
    (void)config;
    return hash_join_i32_v6(build_keys, build_count, probe_keys, probe_count,
                            join_type, result);
}

} // namespace join
} // namespace thunderduck
