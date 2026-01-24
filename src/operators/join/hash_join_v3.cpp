/**
 * ThunderDuck - Hash Join v3.0 Implementation
 *
 * 针对 Apple M4 处理器深度优化:
 * 1. SOA 哈希表布局 - 缓存友好
 * 2. SIMD 批量探测 - Neon 加速
 * 3. Radix Partitioning - L1/L2 适配
 * 4. 完美哈希优化 - O(1) 小整数键
 * 5. Morsel-driven 多核并行
 * 6. AMX 实验支持 (通过 Accelerate)
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

// AMX 通过 Accelerate 框架使用 (可选)
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace thunderduck {
namespace join {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

// 缓存行大小 (M4 = 128 bytes)
constexpr size_t M4_CACHE_LINE = 128;

// 每个缓存行的 int32 数量
constexpr size_t KEYS_PER_CACHELINE = M4_CACHE_LINE / sizeof(int32_t);  // 32

// Radix 分区配置
constexpr int RADIX_BITS = 4;
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;  // 16
constexpr size_t PARTITION_MASK = NUM_PARTITIONS - 1;

// Morsel 大小 (DuckDB 标准)
constexpr size_t MORSEL_SIZE = 2048;

// 完美哈希阈值 (键范围 <= 2x 数据量时启用)
constexpr double PERFECT_HASH_DENSITY_THRESHOLD = 2.0;

// 并行阈值
constexpr size_t PARALLEL_THRESHOLD = 100000;

// 空键标记
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();

// 批量大小
constexpr size_t SIMD_BATCH_SIZE = 8;
constexpr size_t PREFETCH_DISTANCE = 16;

} // anonymous namespace

// ============================================================================
// 哈希函数
// ============================================================================

namespace {

#ifdef __aarch64__
inline uint32_t crc32_hash(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

// 批量哈希 - 4 个键
inline void hash_4_keys(const int32_t* keys, uint32_t* hashes) {
    hashes[0] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[0]));
    hashes[1] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[1]));
    hashes[2] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[2]));
    hashes[3] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[3]));
}

// 批量哈希 - 8 个键 (展开)
inline void hash_8_keys(const int32_t* keys, uint32_t* hashes) {
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
inline uint32_t crc32_hash(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
#endif

// 获取分区索引 (使用高位)
inline size_t get_partition(uint32_t hash) {
    return (hash >> (32 - RADIX_BITS)) & PARTITION_MASK;
}

} // anonymous namespace

// ============================================================================
// SOA 哈希表 (Structure of Arrays)
// ============================================================================

class SOAHashTable {
public:
    SOAHashTable() = default;

    void build(const int32_t* keys, size_t count) {
        if (count == 0) return;

        // 计算容量 (2的幂，负载因子 0.6)
        capacity_ = 16;
        while (capacity_ < count * 1.7) {
            capacity_ *= 2;
        }
        mask_ = capacity_ - 1;
        size_ = count;

        // 分配 128-byte 对齐的数组
        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        // 批量插入
        for (size_t i = 0; i < count; ++i) {
            insert(keys[i], static_cast<uint32_t>(i));
        }
    }

    void insert(int32_t key, uint32_t row_idx) {
        uint32_t hash = crc32_hash(key);
        size_t idx = hash & mask_;

        // 线性探测
        while (keys_[idx] != EMPTY_KEY) {
            idx = (idx + 1) & mask_;
        }

        keys_[idx] = key;
        row_indices_[idx] = row_idx;
    }

    // 标量探测 (基准)
    size_t probe_scalar(const int32_t* probe_keys, size_t probe_count,
                        uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            uint32_t hash = crc32_hash(key);
            size_t idx = hash & mask_;

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

#ifdef __aarch64__
    // SIMD 批量探测 - 使用预取优化但保持正确的线性探测语义
    size_t probe_simd(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        // 批量处理
        size_t i = 0;
        for (; i + SIMD_BATCH_SIZE <= probe_count; i += SIMD_BATCH_SIZE) {
            // 预取下一批哈希表位置
            if (i + PREFETCH_DISTANCE < probe_count) {
                uint32_t h0 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE]);
                uint32_t h1 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE + 1]);
                uint32_t h2 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE + 2]);
                uint32_t h3 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE + 3]);
                __builtin_prefetch(&keys_[h0 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h1 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h2 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h3 & mask_], 0, 3);
            }

            // 批量哈希
            alignas(32) uint32_t hashes[SIMD_BATCH_SIZE];
            hash_8_keys(probe_keys + i, hashes);

            // 探测每个键 - 使用正确的线性探测语义
            for (size_t j = 0; j < SIMD_BATCH_SIZE; ++j) {
                int32_t key = probe_keys[i + j];
                size_t idx = hashes[j] & mask_;

                // 标准线性探测，直到遇到空槽
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
            uint32_t hash = crc32_hash(key);
            size_t idx = hash & mask_;

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
    const int32_t* keys() const { return keys_.data(); }
    const uint32_t* row_indices() const { return row_indices_.data(); }

private:
    alignas(M4_CACHE_LINE) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// 分区数据结构
// ============================================================================

struct PartitionedData {
    struct Partition {
        std::vector<int32_t> keys;
        std::vector<uint32_t> indices;

        void reserve(size_t n) {
            keys.reserve(n);
            indices.reserve(n);
        }

        void clear() {
            keys.clear();
            indices.clear();
        }
    };

    std::array<Partition, NUM_PARTITIONS> partitions;
    std::array<size_t, NUM_PARTITIONS> histogram{};

    void clear() {
        for (auto& p : partitions) p.clear();
        histogram.fill(0);
    }
};

// ============================================================================
// Radix Partitioning
// ============================================================================

namespace {

void compute_histogram(const int32_t* keys, size_t count,
                       std::array<size_t, NUM_PARTITIONS>& histogram) {
    histogram.fill(0);

#ifdef __aarch64__
    // 批量处理
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint32_t h0 = crc32_hash(keys[i]);
        uint32_t h1 = crc32_hash(keys[i + 1]);
        uint32_t h2 = crc32_hash(keys[i + 2]);
        uint32_t h3 = crc32_hash(keys[i + 3]);

        histogram[get_partition(h0)]++;
        histogram[get_partition(h1)]++;
        histogram[get_partition(h2)]++;
        histogram[get_partition(h3)]++;
    }
    for (; i < count; ++i) {
        histogram[get_partition(crc32_hash(keys[i]))]++;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        histogram[get_partition(crc32_hash(keys[i]))]++;
    }
#endif
}

void scatter_to_partitions(const int32_t* keys, size_t count,
                           PartitionedData& out, bool with_indices = true) {
    // Phase 1: 直方图
    compute_histogram(keys, count, out.histogram);

    // Phase 2: 预分配
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        out.partitions[p].keys.resize(out.histogram[p]);
        if (with_indices) {
            out.partitions[p].indices.resize(out.histogram[p]);
        }
    }

    // Phase 3: 分散写入
    std::array<size_t, NUM_PARTITIONS> write_pos{};

#ifdef __aarch64__
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint32_t h0 = crc32_hash(keys[i]);
        uint32_t h1 = crc32_hash(keys[i + 1]);
        uint32_t h2 = crc32_hash(keys[i + 2]);
        uint32_t h3 = crc32_hash(keys[i + 3]);

        size_t p0 = get_partition(h0);
        size_t p1 = get_partition(h1);
        size_t p2 = get_partition(h2);
        size_t p3 = get_partition(h3);

        size_t pos0 = write_pos[p0]++;
        size_t pos1 = write_pos[p1]++;
        size_t pos2 = write_pos[p2]++;
        size_t pos3 = write_pos[p3]++;

        out.partitions[p0].keys[pos0] = keys[i];
        out.partitions[p1].keys[pos1] = keys[i + 1];
        out.partitions[p2].keys[pos2] = keys[i + 2];
        out.partitions[p3].keys[pos3] = keys[i + 3];

        if (with_indices) {
            out.partitions[p0].indices[pos0] = static_cast<uint32_t>(i);
            out.partitions[p1].indices[pos1] = static_cast<uint32_t>(i + 1);
            out.partitions[p2].indices[pos2] = static_cast<uint32_t>(i + 2);
            out.partitions[p3].indices[pos3] = static_cast<uint32_t>(i + 3);
        }
    }
    for (; i < count; ++i) {
        size_t p = get_partition(crc32_hash(keys[i]));
        size_t pos = write_pos[p]++;
        out.partitions[p].keys[pos] = keys[i];
        if (with_indices) {
            out.partitions[p].indices[pos] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        size_t p = get_partition(crc32_hash(keys[i]));
        size_t pos = write_pos[p]++;
        out.partitions[p].keys[pos] = keys[i];
        if (with_indices) {
            out.partitions[p].indices[pos] = static_cast<uint32_t>(i);
        }
    }
#endif
}

} // anonymous namespace

// ============================================================================
// 完美哈希表 (直接索引)
// ============================================================================

class PerfectHashTable {
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
        if (range <= 0 || range > static_cast<int64_t>(count * PERFECT_HASH_DENSITY_THRESHOLD)) {
            return false;
        }
        if (range > 10000000) {  // 限制内存使用
            return false;
        }

        // 检查是否有重复键 - 完美哈希只适用于唯一键
        // 使用临时计数数组检测
        std::vector<uint8_t> key_count(static_cast<size_t>(range), 0);
        bool has_duplicates = false;
        for (size_t i = 0; i < count && !has_duplicates; ++i) {
            size_t idx = static_cast<size_t>(keys[i] - min_key);
            if (key_count[idx] > 0) {
                has_duplicates = true;
            }
            key_count[idx] = 1;
        }

        if (has_duplicates) {
            return false;  // 有重复键，放弃完美哈希
        }

        min_key_ = min_key;
        size_ = static_cast<size_t>(range);

        // 分配直接索引表
        indices_.resize(size_, UINT32_MAX);

        // 填充
        for (size_t i = 0; i < count; ++i) {
            size_t idx = static_cast<size_t>(keys[i] - min_key_);
            indices_[idx] = static_cast<uint32_t>(i);
        }

        built_ = true;
        return true;
    }

    size_t probe(const int32_t* probe_keys, size_t probe_count,
                 uint32_t* out_build, uint32_t* out_probe) const {
        if (!built_) return 0;

        size_t match_count = 0;

#ifdef __aarch64__
        // SIMD 批量探测
        int32x4_t min_key_vec = vdupq_n_s32(min_key_);
        int32x4_t max_offset_vec = vdupq_n_s32(static_cast<int32_t>(size_ - 1));
        uint32x4_t invalid_vec = vdupq_n_u32(UINT32_MAX);

        size_t i = 0;
        for (; i + 4 <= probe_count; i += 4) {
            int32x4_t keys = vld1q_s32(probe_keys + i);

            // 计算偏移
            int32x4_t offsets = vsubq_s32(keys, min_key_vec);

            // 范围检查
            uint32x4_t valid_mask = vandq_u32(
                vcgeq_s32(offsets, vdupq_n_s32(0)),
                vcleq_s32(offsets, max_offset_vec)
            );

            if (vmaxvq_u32(valid_mask)) {
                // 有潜在匹配，逐个处理
                for (int j = 0; j < 4; ++j) {
                    int32_t key = probe_keys[i + j];
                    int32_t offset = key - min_key_;
                    if (offset >= 0 && static_cast<size_t>(offset) < size_) {
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

        // 处理剩余
        for (; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            int32_t offset = key - min_key_;
            if (offset >= 0 && static_cast<size_t>(offset) < size_) {
                uint32_t idx = indices_[offset];
                if (idx != UINT32_MAX) {
                    out_build[match_count] = idx;
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
            }
        }
#else
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            int32_t offset = key - min_key_;
            if (offset >= 0 && static_cast<size_t>(offset) < size_) {
                uint32_t idx = indices_[offset];
                if (idx != UINT32_MAX) {
                    out_build[match_count] = idx;
                    out_probe[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
            }
        }
#endif

        return match_count;
    }

    bool is_built() const { return built_; }

private:
    std::vector<uint32_t> indices_;
    int32_t min_key_ = 0;
    size_t size_ = 0;
    bool built_ = false;
};

// ============================================================================
// 线程本地结果缓冲
// ============================================================================

struct ThreadLocalResult {
    std::vector<uint32_t> build_indices;
    std::vector<uint32_t> probe_indices;
    size_t count = 0;

    void reserve(size_t n) {
        build_indices.reserve(n);
        probe_indices.reserve(n);
    }

    void add(uint32_t build_idx, uint32_t probe_idx) {
        build_indices.push_back(build_idx);
        probe_indices.push_back(probe_idx);
        ++count;
    }

    void clear() {
        build_indices.clear();
        probe_indices.clear();
        count = 0;
    }
};

// ============================================================================
// Hash Join v3.0 核心实现
// ============================================================================

namespace {

// 单分区 Join (带预取)
size_t join_single_partition(const int32_t* build_keys, const uint32_t* build_indices, size_t build_count,
                             const int32_t* probe_keys, const uint32_t* probe_indices, size_t probe_count,
                             uint32_t* out_build, uint32_t* out_probe) {
    if (build_count == 0 || probe_count == 0) return 0;

    // 构建 SOA 哈希表
    SOAHashTable ht;
    ht.build(build_keys, build_count);

    // 临时结果
    std::vector<uint32_t> temp_build(probe_count);
    std::vector<uint32_t> temp_probe(probe_count);

    // 探测
#ifdef __aarch64__
    size_t matches = ht.probe_simd(probe_keys, probe_count,
                                    temp_build.data(), temp_probe.data());
#else
    size_t matches = ht.probe_scalar(probe_keys, probe_count,
                                      temp_build.data(), temp_probe.data());
#endif

    // 转换回原始索引
    for (size_t i = 0; i < matches; ++i) {
        out_build[i] = build_indices[temp_build[i]];
        out_probe[i] = probe_indices[temp_probe[i]];
    }

    return matches;
}

// 多线程并行 Join
size_t join_parallel(const PartitionedData& build_data,
                     const PartitionedData& probe_data,
                     JoinResult* result,
                     size_t num_threads) {

    // 结果缓冲（每分区）
    std::array<ThreadLocalResult, NUM_PARTITIONS> partition_results;

    // 原子分区计数器
    std::atomic<size_t> next_partition{0};

    // 工作函数
    auto worker = [&]() {
        while (true) {
            size_t p = next_partition.fetch_add(1);
            if (p >= NUM_PARTITIONS) break;

            const auto& build_part = build_data.partitions[p];
            const auto& probe_part = probe_data.partitions[p];

            if (build_part.keys.empty() || probe_part.keys.empty()) {
                continue;
            }

            // 为分区结果预分配空间
            partition_results[p].reserve(probe_part.keys.size());

            // 构建分区哈希表
            SOAHashTable ht;
            ht.build(build_part.keys.data(), build_part.keys.size());

            // 探测
            std::vector<uint32_t> temp_build(probe_part.keys.size());
            std::vector<uint32_t> temp_probe(probe_part.keys.size());

#ifdef __aarch64__
            size_t matches = ht.probe_simd(probe_part.keys.data(), probe_part.keys.size(),
                                            temp_build.data(), temp_probe.data());
#else
            size_t matches = ht.probe_scalar(probe_part.keys.data(), probe_part.keys.size(),
                                              temp_build.data(), temp_probe.data());
#endif

            // 存储结果（转换为原始索引）
            for (size_t i = 0; i < matches; ++i) {
                uint32_t orig_build = build_part.indices[temp_build[i]];
                uint32_t orig_probe = probe_part.indices[temp_probe[i]];
                partition_results[p].add(orig_build, orig_probe);
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

    // 合并结果
    size_t total_matches = 0;
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        total_matches += partition_results[p].count;
    }

    // 确保结果缓冲区足够大
    if (result->capacity < total_matches) {
        grow_join_result(result, total_matches);
    }

    // 复制结果
    size_t offset = 0;
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        const auto& pr = partition_results[p];
        if (pr.count > 0) {
            std::memcpy(result->left_indices + offset,
                       pr.build_indices.data(), pr.count * sizeof(uint32_t));
            std::memcpy(result->right_indices + offset,
                       pr.probe_indices.data(), pr.count * sizeof(uint32_t));
            offset += pr.count;
        }
    }
    result->count = total_matches;

    return total_matches;
}

} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 确保结果缓冲区 - 考虑重复键可能产生大量匹配
    // 最坏情况：每个 probe key 匹配所有 build rows (笛卡尔积)
    // 保守估算：probe_count * 平均匹配数，取 max(probe, build) * 4 作为初始值
    size_t estimated_matches = std::max(build_count, probe_count) * 4;
    if (result->capacity < estimated_matches) {
        grow_join_result(result, estimated_matches);
    }

    // 策略 1: 尝试完美哈希 (小整数键，最快)
    PerfectHashTable perfect_ht;
    if (perfect_ht.try_build(build_keys, build_count)) {
        size_t matches = perfect_ht.probe(probe_keys, probe_count,
                                           result->left_indices, result->right_indices);
        result->count = matches;
        return matches;
    }

    // 策略 2: 小表直接 Join
    if (build_count < 10000 && probe_count < 50000) {
        SOAHashTable ht;
        ht.build(build_keys, build_count);

#ifdef __aarch64__
        size_t matches = ht.probe_simd(probe_keys, probe_count,
                                        result->left_indices, result->right_indices);
#else
        size_t matches = ht.probe_scalar(probe_keys, probe_count,
                                          result->left_indices, result->right_indices);
#endif
        result->count = matches;
        return matches;
    }

    // 策略 3: Radix Partitioning + 并行
    PartitionedData build_data, probe_data;

    // 分区
    scatter_to_partitions(build_keys, build_count, build_data, true);
    scatter_to_partitions(probe_keys, probe_count, probe_data, true);

    // 确定线程数
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads > 4) num_threads = 4;  // 使用 P-cores
    if (num_threads < 1) num_threads = 1;

    // 对于小任务使用单线程
    if (build_count + probe_count < PARALLEL_THRESHOLD) {
        num_threads = 1;
    }

    // 并行 Join
    size_t matches = join_parallel(build_data, probe_data, result, num_threads);

    // TODO: 处理 LEFT/RIGHT/FULL JOIN
    (void)join_type;

    return matches;
}

// ============================================================================
// 配置版本
// ============================================================================

size_t hash_join_i32_v3_config(const int32_t* build_keys, size_t build_count,
                                const int32_t* probe_keys, size_t probe_count,
                                JoinType join_type,
                                JoinResult* result,
                                const JoinConfig& config) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 确保结果缓冲区 - 考虑重复键可能产生大量匹配
    size_t estimated_matches = std::max(build_count, probe_count) * 4;
    if (result->capacity < estimated_matches) {
        grow_join_result(result, estimated_matches);
    }

    // 完美哈希优化
    if (config.enable_perfect_hash) {
        PerfectHashTable perfect_ht;
        if (perfect_ht.try_build(build_keys, build_count)) {
            size_t matches = perfect_ht.probe(probe_keys, probe_count,
                                               result->left_indices, result->right_indices);
            result->count = matches;
            return matches;
        }
    }

    // 分区
    PartitionedData build_data, probe_data;
    scatter_to_partitions(build_keys, build_count, build_data, true);
    scatter_to_partitions(probe_keys, probe_count, probe_data, true);

    // 并行 Join
    size_t num_threads = config.num_threads;
    if (num_threads < 1) num_threads = 1;
    if (build_count + probe_count < PARALLEL_THRESHOLD) {
        num_threads = 1;
    }

    size_t matches = join_parallel(build_data, probe_data, result, num_threads);

    (void)join_type;
    return matches;
}

} // namespace join
} // namespace thunderduck
