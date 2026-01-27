/**
 * ThunderDuck - Hash Join v10.1 Implementation
 *
 * 零拷贝优化版本:
 * 1. 单次哈希计算 + 缓存
 * 2. 指针数组代替数据拷贝
 * 3. 直接写入最终结果缓冲区
 * 4. 消除索引转换开销
 *
 * 内存优化:
 * - V3: ~9MB 数据拷贝 (100K build + 1M probe)
 * - V10.1: ~4MB 哈希缓存 (无数据拷贝)
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

constexpr size_t M4_CACHE_LINE = 128;
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();

// 分区配置
constexpr int RADIX_BITS = 4;
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;  // 16
constexpr size_t PARTITION_MASK = NUM_PARTITIONS - 1;

// 批量大小
constexpr size_t BATCH_SIZE = 8;
constexpr size_t PREFETCH_DISTANCE = 16;

// 阈值
constexpr size_t SMALL_TABLE_THRESHOLD = 10000;  // < 10K 直接 Join

#ifdef __aarch64__
__attribute__((always_inline))
inline uint32_t hash_key(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

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
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
#endif

inline size_t get_partition(uint32_t hash) {
    return (hash >> (32 - RADIX_BITS)) & PARTITION_MASK;
}

} // anonymous namespace

// ============================================================================
// V10.1 零拷贝分区结构
// ============================================================================

/**
 * 零拷贝分区视图
 *
 * 不拷贝实际数据，只存储:
 * - 指向原始数据的指针
 * - 原始索引数组
 * - 预计算的哈希值
 */
struct PartitionView {
    std::vector<uint32_t> indices;   // 原始数据索引
    std::vector<uint32_t> hashes;    // 预计算的哈希值
    size_t count = 0;

    void reserve(size_t n) {
        indices.reserve(n);
        hashes.reserve(n);
    }

    void add(uint32_t idx, uint32_t hash) {
        indices.push_back(idx);
        hashes.push_back(hash);
        ++count;
    }

    void clear() {
        indices.clear();
        hashes.clear();
        count = 0;
    }
};

// ============================================================================
// V10.1 零拷贝哈希表
// ============================================================================

/**
 * 零拷贝哈希表
 *
 * 特点:
 * - 不拷贝键，只存储指针和索引
 * - 使用预计算的哈希值
 * - 直接引用原始数据
 */
class ZeroCopyHashTable {
public:
    ZeroCopyHashTable() = default;

    /**
     * 构建哈希表 (零拷贝)
     *
     * @param keys 原始键数组的指针 (不拷贝)
     * @param indices 分区内的原始索引
     * @param hashes 预计算的哈希值
     * @param count 元素数量
     */
    void build(const int32_t* keys, const uint32_t* indices,
               const uint32_t* hashes, size_t count) {
        if (count == 0) return;

        keys_ptr_ = keys;      // 只存储指针，不拷贝
        count_ = count;

        // 计算容量
        capacity_ = 16;
        while (capacity_ < count * 1.7) capacity_ *= 2;
        mask_ = capacity_ - 1;

        // 分配槽位数组
        slots_.resize(capacity_);
        for (auto& slot : slots_) {
            slot.key_idx = UINT32_MAX;  // 空标记
        }

        // 插入 (使用预计算的哈希)
        for (size_t i = 0; i < count; ++i) {
            uint32_t hash = hashes[i];
            size_t slot_idx = hash & mask_;

            // 线性探测
            while (slots_[slot_idx].key_idx != UINT32_MAX) {
                slot_idx = (slot_idx + 1) & mask_;
            }

            slots_[slot_idx].key_idx = indices[i];  // 原始索引
            slots_[slot_idx].hash = hash;
        }
    }

    /**
     * 探测哈希表 (零拷贝)
     *
     * @param keys 原始探测键数组
     * @param probe_indices 分区内的探测索引
     * @param probe_hashes 预计算的哈希值
     * @param probe_count 探测数量
     * @param out_build 输出: build 侧原始索引
     * @param out_probe 输出: probe 侧原始索引
     * @return 匹配数量
     */
    size_t probe(const int32_t* probe_keys, const uint32_t* probe_indices,
                 const uint32_t* probe_hashes, size_t probe_count,
                 uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

#ifdef __aarch64__
        size_t i = 0;
        for (; i + BATCH_SIZE <= probe_count; i += BATCH_SIZE) {
            // 预取
            if (i + PREFETCH_DISTANCE < probe_count) {
                __builtin_prefetch(&slots_[probe_hashes[i + PREFETCH_DISTANCE] & mask_], 0, 3);
                __builtin_prefetch(&slots_[probe_hashes[i + PREFETCH_DISTANCE + 1] & mask_], 0, 3);
                __builtin_prefetch(&slots_[probe_hashes[i + PREFETCH_DISTANCE + 2] & mask_], 0, 3);
                __builtin_prefetch(&slots_[probe_hashes[i + PREFETCH_DISTANCE + 3] & mask_], 0, 3);
            }

            // 批量探测
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                uint32_t probe_idx = probe_indices[i + j];
                int32_t probe_key = probe_keys[probe_idx];
                uint32_t hash = probe_hashes[i + j];
                size_t slot_idx = hash & mask_;

                while (slots_[slot_idx].key_idx != UINT32_MAX) {
                    uint32_t build_idx = slots_[slot_idx].key_idx;
                    if (keys_ptr_[build_idx] == probe_key) {
                        out_build[match_count] = build_idx;
                        out_probe[match_count] = probe_idx;
                        ++match_count;
                    }
                    slot_idx = (slot_idx + 1) & mask_;
                }
            }
        }

        // 处理剩余
        for (; i < probe_count; ++i) {
            uint32_t probe_idx = probe_indices[i];
            int32_t probe_key = probe_keys[probe_idx];
            uint32_t hash = probe_hashes[i];
            size_t slot_idx = hash & mask_;

            while (slots_[slot_idx].key_idx != UINT32_MAX) {
                uint32_t build_idx = slots_[slot_idx].key_idx;
                if (keys_ptr_[build_idx] == probe_key) {
                    out_build[match_count] = build_idx;
                    out_probe[match_count] = probe_idx;
                    ++match_count;
                }
                slot_idx = (slot_idx + 1) & mask_;
            }
        }
#else
        for (size_t i = 0; i < probe_count; ++i) {
            uint32_t probe_idx = probe_indices[i];
            int32_t probe_key = probe_keys[probe_idx];
            uint32_t hash = probe_hashes[i];
            size_t slot_idx = hash & mask_;

            while (slots_[slot_idx].key_idx != UINT32_MAX) {
                uint32_t build_idx = slots_[slot_idx].key_idx;
                if (keys_ptr_[build_idx] == probe_key) {
                    out_build[match_count] = build_idx;
                    out_probe[match_count] = probe_idx;
                    ++match_count;
                }
                slot_idx = (slot_idx + 1) & mask_;
            }
        }
#endif

        return match_count;
    }

private:
    struct Slot {
        uint32_t key_idx;   // 原始键索引 (UINT32_MAX = 空)
        uint32_t hash;      // 哈希值 (用于快速比较)
    };

    const int32_t* keys_ptr_ = nullptr;   // 指向原始数据，不拷贝
    std::vector<Slot> slots_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t count_ = 0;
};

// ============================================================================
// V10.1 单遍分区 (哈希计算仅一次)
// ============================================================================

namespace {

/**
 * 单遍分区 + 哈希缓存
 *
 * 优化:
 * 1. 哈希只计算一次，缓存起来
 * 2. 不拷贝键，只存储索引
 * 3. 两遍扫描: 计数 → 分配 → 写入
 */
void partition_with_hash_cache(const int32_t* keys, size_t count,
                                std::array<PartitionView, NUM_PARTITIONS>& partitions,
                                std::vector<uint32_t>& hash_cache) {
    // 分配哈希缓存
    hash_cache.resize(count);

    // Phase 1: 计算哈希 + 直方图 (一次遍历)
    std::array<size_t, NUM_PARTITIONS> histogram{};

#ifdef __aarch64__
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        alignas(32) uint32_t hashes[8];
        hash_batch_8(keys + i, hashes);

        for (int j = 0; j < 8; ++j) {
            hash_cache[i + j] = hashes[j];
            histogram[get_partition(hashes[j])]++;
        }
    }
    for (; i < count; ++i) {
        uint32_t h = hash_key(keys[i]);
        hash_cache[i] = h;
        histogram[get_partition(h)]++;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        uint32_t h = hash_key(keys[i]);
        hash_cache[i] = h;
        histogram[get_partition(h)]++;
    }
#endif

    // Phase 2: 预分配分区空间
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        partitions[p].reserve(histogram[p]);
        partitions[p].count = 0;
    }

    // Phase 3: 分配到分区 (不拷贝键，只存索引和哈希)
    for (size_t i = 0; i < count; ++i) {
        uint32_t h = hash_cache[i];
        size_t p = get_partition(h);
        partitions[p].add(static_cast<uint32_t>(i), h);
    }
}

} // anonymous namespace

// ============================================================================
// V10.1 并行 Join (直接写入最终缓冲区)
// ============================================================================

namespace {

/**
 * 计算每个分区的结果偏移量
 *
 * 用于直接写入最终缓冲区，避免合并拷贝
 */
struct PartitionOffsets {
    std::array<std::atomic<size_t>, NUM_PARTITIONS> offsets;
    std::array<size_t, NUM_PARTITIONS> capacities;

    void init(const std::array<PartitionView, NUM_PARTITIONS>& build_parts,
              const std::array<PartitionView, NUM_PARTITIONS>& probe_parts) {
        size_t offset = 0;
        for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
            offsets[p].store(offset, std::memory_order_relaxed);
            // 估算: 每个 probe 最多匹配一次 (实际可能更多)
            capacities[p] = std::min(build_parts[p].count, probe_parts[p].count) * 2 + 1024;
            offset += capacities[p];
        }
    }

    size_t total_capacity() const {
        size_t total = 0;
        for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
            total += capacities[p];
        }
        return total;
    }
};

/**
 * 并行 Join (零拷贝版)
 *
 * 优化:
 * 1. 使用预计算的哈希值
 * 2. 直接写入最终结果缓冲区
 * 3. 消除索引转换 (已经是原始索引)
 */
size_t join_parallel_zerocopy(
    const int32_t* build_keys, const std::array<PartitionView, NUM_PARTITIONS>& build_parts,
    const int32_t* probe_keys, const std::array<PartitionView, NUM_PARTITIONS>& probe_parts,
    JoinResult* result, size_t num_threads) {

    // 计算分区偏移
    PartitionOffsets offsets;
    offsets.init(build_parts, probe_parts);

    // 确保结果缓冲区
    size_t total_capacity = offsets.total_capacity();
    if (result->capacity < total_capacity) {
        grow_join_result(result, total_capacity);
    }

    // 每分区的实际匹配数
    std::array<std::atomic<size_t>, NUM_PARTITIONS> match_counts{};

    // 原子分区计数器
    std::atomic<size_t> next_partition{0};

    // 工作函数
    auto worker = [&]() {
        while (true) {
            size_t p = next_partition.fetch_add(1);
            if (p >= NUM_PARTITIONS) break;

            const auto& build_part = build_parts[p];
            const auto& probe_part = probe_parts[p];

            if (build_part.count == 0 || probe_part.count == 0) {
                continue;
            }

            // 构建零拷贝哈希表
            ZeroCopyHashTable ht;
            ht.build(build_keys, build_part.indices.data(),
                     build_part.hashes.data(), build_part.count);

            // 获取此分区在最终缓冲区的位置
            size_t base_offset = offsets.offsets[p].load(std::memory_order_relaxed);

            // 直接写入最终缓冲区
            size_t matches = ht.probe(
                probe_keys, probe_part.indices.data(),
                probe_part.hashes.data(), probe_part.count,
                result->left_indices + base_offset,
                result->right_indices + base_offset);

            match_counts[p].store(matches, std::memory_order_relaxed);
        }
    };

    // 启动线程
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    // 计算总匹配数并压缩结果
    size_t total_matches = 0;
    size_t write_pos = 0;

    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        size_t matches = match_counts[p].load(std::memory_order_relaxed);
        if (matches == 0) continue;

        size_t read_pos = offsets.offsets[p].load(std::memory_order_relaxed);

        // 如果位置不连续，需要移动
        if (write_pos != read_pos && matches > 0) {
            std::memmove(result->left_indices + write_pos,
                        result->left_indices + read_pos,
                        matches * sizeof(uint32_t));
            std::memmove(result->right_indices + write_pos,
                        result->right_indices + read_pos,
                        matches * sizeof(uint32_t));
        }
        write_pos += matches;
        total_matches += matches;
    }

    result->count = total_matches;
    return total_matches;
}

} // anonymous namespace

// ============================================================================
// V10.1 小表直接 Join (无分区)
// ============================================================================

namespace {

/**
 * 小表直接 Join
 *
 * 对于小表，分区开销大于收益，直接构建全局哈希表
 */
size_t join_direct_zerocopy(const int32_t* build_keys, size_t build_count,
                             const int32_t* probe_keys, size_t probe_count,
                             JoinResult* result) {
    // 预计算哈希
    std::vector<uint32_t> build_hashes(build_count);
    std::vector<uint32_t> probe_hashes(probe_count);
    std::vector<uint32_t> build_indices(build_count);
    std::vector<uint32_t> probe_indices(probe_count);

#ifdef __aarch64__
    // 批量哈希 build
    size_t i = 0;
    for (; i + 8 <= build_count; i += 8) {
        hash_batch_8(build_keys + i, build_hashes.data() + i);
    }
    for (; i < build_count; ++i) {
        build_hashes[i] = hash_key(build_keys[i]);
    }

    // 批量哈希 probe
    i = 0;
    for (; i + 8 <= probe_count; i += 8) {
        hash_batch_8(probe_keys + i, probe_hashes.data() + i);
    }
    for (; i < probe_count; ++i) {
        probe_hashes[i] = hash_key(probe_keys[i]);
    }
#else
    for (size_t i = 0; i < build_count; ++i) {
        build_hashes[i] = hash_key(build_keys[i]);
    }
    for (size_t i = 0; i < probe_count; ++i) {
        probe_hashes[i] = hash_key(probe_keys[i]);
    }
#endif

    // 初始化索引
    for (size_t i = 0; i < build_count; ++i) build_indices[i] = i;
    for (size_t i = 0; i < probe_count; ++i) probe_indices[i] = i;

    // 构建哈希表
    ZeroCopyHashTable ht;
    ht.build(build_keys, build_indices.data(), build_hashes.data(), build_count);

    // 确保结果缓冲区
    size_t estimated = std::max(build_count, probe_count) * 2;
    if (result->capacity < estimated) {
        grow_join_result(result, estimated);
    }

    // 探测
    size_t matches = ht.probe(probe_keys, probe_indices.data(),
                               probe_hashes.data(), probe_count,
                               result->left_indices, result->right_indices);

    result->count = matches;
    return matches;
}

} // anonymous namespace

// ============================================================================
// V10.1 公开接口
// ============================================================================

size_t hash_join_i32_v10_1(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinType join_type,
                            JoinResult* result) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 小表直接 Join
    if (build_count < SMALL_TABLE_THRESHOLD) {
        return join_direct_zerocopy(build_keys, build_count,
                                     probe_keys, probe_count, result);
    }

    // 大表使用分区
    std::array<PartitionView, NUM_PARTITIONS> build_parts;
    std::array<PartitionView, NUM_PARTITIONS> probe_parts;
    std::vector<uint32_t> build_hash_cache, probe_hash_cache;

    // 单遍分区 + 哈希缓存
    partition_with_hash_cache(build_keys, build_count, build_parts, build_hash_cache);
    partition_with_hash_cache(probe_keys, probe_count, probe_parts, probe_hash_cache);

    // 确定线程数
    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size_t(4));
    if (num_threads < 1) num_threads = 1;

    // 并行 Join
    size_t matches = join_parallel_zerocopy(
        build_keys, build_parts,
        probe_keys, probe_parts,
        result, num_threads);

    (void)join_type;  // TODO: LEFT/RIGHT/FULL/SEMI/ANTI
    return matches;
}

// ============================================================================
// 版本信息
// ============================================================================

const char* get_v10_1_version_info() {
    return "V10.1 - 零拷贝优化 (单次哈希 + 指针数组 + 直接写入)";
}

} // namespace join
} // namespace thunderduck
