/**
 * ThunderDuck - Hash Join v4 RADIX256 Strategy
 *
 * 256 分区 (8-bit) 优化:
 * - 每分区 ~53KB 适配 L1 缓存 (M4 L1D = 192KB)
 * - 更好的缓存局部性
 * - 复用 v3 的 SOA 哈希表和 SIMD 探测
 */

#include "thunderduck/join.h"
#include "thunderduck/memory.h"

#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {
namespace v4 {

// ============================================================================
// 配置常量 - RADIX256
// ============================================================================

namespace {

// 缓存行大小 (M4 = 128 bytes)
constexpr size_t M4_CACHE_LINE = 128;

// RADIX256: 8-bit 分区
constexpr int RADIX_BITS_256 = 8;
constexpr size_t NUM_PARTITIONS_256 = 1 << RADIX_BITS_256;  // 256
constexpr size_t PARTITION_MASK_256 = NUM_PARTITIONS_256 - 1;

// 空键标记
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();

// 批处理大小
constexpr size_t SIMD_BATCH_SIZE = 8;
constexpr size_t PREFETCH_DISTANCE = 16;

// 并行阈值
constexpr size_t PARALLEL_THRESHOLD = 100000;

} // anonymous namespace

// ============================================================================
// 哈希函数
// ============================================================================

namespace {

#ifdef __aarch64__
inline uint32_t crc32_hash(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

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

// 获取 256 分区索引 (使用高 8 位)
inline size_t get_partition_256(uint32_t hash) {
    return (hash >> (32 - RADIX_BITS_256)) & PARTITION_MASK_256;
}

} // anonymous namespace

// ============================================================================
// SOA 哈希表 (复用 v3 设计)
// ============================================================================

class SOAHashTable256 {
public:
    SOAHashTable256() = default;

    void build(const int32_t* keys, size_t count) {
        if (count == 0) return;

        // 计算容量 (2的幂，负载因子 0.6)
        capacity_ = 16;
        while (capacity_ < count * 1.7) {
            capacity_ *= 2;
        }
        mask_ = capacity_ - 1;
        size_ = count;

        // 分配对齐数组
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

#ifdef __aarch64__
    // SIMD 批量探测
    size_t probe_simd(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        size_t i = 0;
        for (; i + SIMD_BATCH_SIZE <= probe_count; i += SIMD_BATCH_SIZE) {
            // 预取下一批
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

            // 逐个探测
            for (size_t j = 0; j < SIMD_BATCH_SIZE; ++j) {
                int32_t key = probe_keys[i + j];
                size_t idx = hashes[j] & mask_;

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
// 256 分区数据结构
// ============================================================================

struct PartitionedData256 {
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

    std::array<Partition, NUM_PARTITIONS_256> partitions;
    std::array<size_t, NUM_PARTITIONS_256> histogram{};

    void clear() {
        for (auto& p : partitions) p.clear();
        histogram.fill(0);
    }
};

// ============================================================================
// Radix Partitioning - 256 分区
// ============================================================================

namespace {

void compute_histogram_256(const int32_t* keys, size_t count,
                            std::array<size_t, NUM_PARTITIONS_256>& histogram) {
    histogram.fill(0);

#ifdef __aarch64__
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint32_t h0 = crc32_hash(keys[i]);
        uint32_t h1 = crc32_hash(keys[i + 1]);
        uint32_t h2 = crc32_hash(keys[i + 2]);
        uint32_t h3 = crc32_hash(keys[i + 3]);

        histogram[get_partition_256(h0)]++;
        histogram[get_partition_256(h1)]++;
        histogram[get_partition_256(h2)]++;
        histogram[get_partition_256(h3)]++;
    }
    for (; i < count; ++i) {
        histogram[get_partition_256(crc32_hash(keys[i]))]++;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        histogram[get_partition_256(crc32_hash(keys[i]))]++;
    }
#endif
}

void scatter_to_partitions_256(const int32_t* keys, size_t count,
                                PartitionedData256& out, bool with_indices = true) {
    // Phase 1: 直方图
    compute_histogram_256(keys, count, out.histogram);

    // Phase 2: 预分配
    for (size_t p = 0; p < NUM_PARTITIONS_256; ++p) {
        out.partitions[p].keys.resize(out.histogram[p]);
        if (with_indices) {
            out.partitions[p].indices.resize(out.histogram[p]);
        }
    }

    // Phase 3: 分散写入
    std::array<size_t, NUM_PARTITIONS_256> write_pos{};

#ifdef __aarch64__
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        uint32_t h0 = crc32_hash(keys[i]);
        uint32_t h1 = crc32_hash(keys[i + 1]);
        uint32_t h2 = crc32_hash(keys[i + 2]);
        uint32_t h3 = crc32_hash(keys[i + 3]);

        size_t p0 = get_partition_256(h0);
        size_t p1 = get_partition_256(h1);
        size_t p2 = get_partition_256(h2);
        size_t p3 = get_partition_256(h3);

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
        size_t p = get_partition_256(crc32_hash(keys[i]));
        size_t pos = write_pos[p]++;
        out.partitions[p].keys[pos] = keys[i];
        if (with_indices) {
            out.partitions[p].indices[pos] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        size_t p = get_partition_256(crc32_hash(keys[i]));
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
// 线程本地结果缓冲
// ============================================================================

struct ThreadLocalResult256 {
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
// 并行 Join
// ============================================================================

namespace {

size_t join_parallel_256(const PartitionedData256& build_data,
                          const PartitionedData256& probe_data,
                          JoinResult* result,
                          size_t num_threads) {
    // 结果缓冲（每分区）
    std::array<ThreadLocalResult256, NUM_PARTITIONS_256> partition_results;

    // 原子分区计数器
    std::atomic<size_t> next_partition{0};

    // 工作函数
    auto worker = [&]() {
        while (true) {
            size_t p = next_partition.fetch_add(1);
            if (p >= NUM_PARTITIONS_256) break;

            const auto& build_part = build_data.partitions[p];
            const auto& probe_part = probe_data.partitions[p];

            if (build_part.keys.empty() || probe_part.keys.empty()) {
                continue;
            }

            // 预分配结果空间
            partition_results[p].reserve(probe_part.keys.size());

            // 构建分区哈希表
            SOAHashTable256 ht;
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

            // 转换为原始索引
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
    for (size_t p = 0; p < NUM_PARTITIONS_256; ++p) {
        total_matches += partition_results[p].count;
    }

    // 确保结果缓冲区足够大
    if (result->capacity < total_matches) {
        grow_join_result(result, total_matches);
    }

    // 复制结果
    size_t offset = 0;
    for (size_t p = 0; p < NUM_PARTITIONS_256; ++p) {
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
// 无分区直接 Join (小数据量优化)
// ============================================================================

namespace {

size_t join_direct_no_partition(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinResult* result) {

    // 直接构建哈希表，不分区
    SOAHashTable256 ht;
    ht.build(build_keys, build_count);

    // 确保结果缓冲区
    if (result->capacity < probe_count) {
        grow_join_result(result, probe_count * 2);
    }

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

} // anonymous namespace

// ============================================================================
// RADIX256 策略入口 (自适应分区)
// ============================================================================

size_t hash_join_radix256(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

    size_t total = build_count + probe_count;

    // ========== P1 优化: 自适应分区 ==========
    // 小数据量不需要分区，直接 Join 更快
    if (total < 100000) {
        // 不分区，直接使用哈希表
        return join_direct_no_partition(build_keys, build_count,
                                         probe_keys, probe_count, result);
    }

    // 中等数据量使用 16 分区 (回退到 v3 实现)
    if (total < 500000) {
        return hash_join_i32_v3(build_keys, build_count,
                                 probe_keys, probe_count,
                                 join_type, result);
    }

    // 大数据量使用 256 分区
    PartitionedData256 build_data, probe_data;
    scatter_to_partitions_256(build_keys, build_count, build_data, true);
    scatter_to_partitions_256(probe_keys, probe_count, probe_data, true);

    // 确定线程数
    size_t num_threads = config.num_threads;
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 8) num_threads = 8;

    // 小任务使用单线程
    if (total < PARALLEL_THRESHOLD) {
        num_threads = 1;
    }

    // 并行 Join
    size_t matches = join_parallel_256(build_data, probe_data, result, num_threads);

    (void)join_type;
    return matches;
}

} // namespace v4
} // namespace join
} // namespace thunderduck
