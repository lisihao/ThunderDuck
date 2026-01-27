/**
 * ThunderDuck - Hash Join v10.2 Implementation
 *
 * 优化方向 (基于 V10.1 的教训):
 * 1. 保留数据拷贝 - 缓存局部性比零拷贝更重要
 * 2. 单次哈希计算 - 分区时缓存哈希值，build 时复用
 * 3. 直接写入最终缓冲区 - 消除结果合并的 memcpy
 * 4. 更紧凑的哈希表 - 减少 cache miss
 *
 * 核心思想: 数据拷贝是值得的，因为它改善了 join 阶段的缓存访问模式
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
constexpr size_t SMALL_TABLE_THRESHOLD = 10000;

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
// V10.2 紧凑分区结构 (带哈希缓存)
// ============================================================================

struct PartitionData {
    std::vector<int32_t> keys;      // 键 (拷贝，保证缓存局部性)
    std::vector<uint32_t> indices;  // 原始索引
    std::vector<uint32_t> hashes;   // 预计算哈希 (复用于 build)
    size_t count = 0;

    void resize(size_t n) {
        keys.resize(n);
        indices.resize(n);
        hashes.resize(n);
        count = n;
    }

    void clear() {
        keys.clear();
        indices.clear();
        hashes.clear();
        count = 0;
    }
};

// ============================================================================
// V10.2 紧凑哈希表 (使用预计算哈希)
// ============================================================================

class CompactHashTable {
public:
    /**
     * 构建哈希表 (使用预计算哈希)
     *
     * 优化: 哈希值直接复用，无需重新计算
     */
    void build(const int32_t* keys, const uint32_t* indices,
               const uint32_t* hashes, size_t count) {
        if (count == 0) return;

        count_ = count;

        // 计算容量 (负载因子 ~0.6)
        capacity_ = 16;
        while (capacity_ < count * 1.7) capacity_ *= 2;
        mask_ = capacity_ - 1;

        // 分配紧凑的槽位数组 (key + index 在一起，缓存友好)
        slots_.resize(capacity_);
        for (auto& slot : slots_) {
            slot.key = EMPTY_KEY;
        }

        // 插入 (使用预计算哈希)
        for (size_t i = 0; i < count; ++i) {
            uint32_t h = hashes[i];
            size_t slot_idx = h & mask_;

            while (slots_[slot_idx].key != EMPTY_KEY) {
                slot_idx = (slot_idx + 1) & mask_;
            }

            slots_[slot_idx].key = keys[i];      // 直接存储键
            slots_[slot_idx].index = indices[i]; // 原始索引
        }
    }

    /**
     * 探测 (使用预计算哈希，直接写入最终缓冲区)
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
                int32_t probe_key = probe_keys[i + j];
                uint32_t probe_idx = probe_indices[i + j];
                size_t slot_idx = probe_hashes[i + j] & mask_;

                while (slots_[slot_idx].key != EMPTY_KEY) {
                    if (slots_[slot_idx].key == probe_key) {
                        out_build[match_count] = slots_[slot_idx].index;
                        out_probe[match_count] = probe_idx;
                        ++match_count;
                    }
                    slot_idx = (slot_idx + 1) & mask_;
                }
            }
        }

        // 处理剩余
        for (; i < probe_count; ++i) {
            int32_t probe_key = probe_keys[i];
            uint32_t probe_idx = probe_indices[i];
            size_t slot_idx = probe_hashes[i] & mask_;

            while (slots_[slot_idx].key != EMPTY_KEY) {
                if (slots_[slot_idx].key == probe_key) {
                    out_build[match_count] = slots_[slot_idx].index;
                    out_probe[match_count] = probe_idx;
                    ++match_count;
                }
                slot_idx = (slot_idx + 1) & mask_;
            }
        }
#else
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t probe_key = probe_keys[i];
            uint32_t probe_idx = probe_indices[i];
            size_t slot_idx = probe_hashes[i] & mask_;

            while (slots_[slot_idx].key != EMPTY_KEY) {
                if (slots_[slot_idx].key == probe_key) {
                    out_build[match_count] = slots_[slot_idx].index;
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
    // 紧凑槽位: key 和 index 在一起，一次 cache line 可以加载多个
    struct alignas(8) Slot {
        int32_t key;      // 键 (EMPTY_KEY = 空)
        uint32_t index;   // 原始索引
    };

    std::vector<Slot> slots_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t count_ = 0;
};

// ============================================================================
// V10.2 单遍分区 (哈希计算一次 + 数据拷贝)
// ============================================================================

namespace {

/**
 * 单遍分区
 *
 * 优化:
 * 1. 哈希只计算一次
 * 2. 键拷贝到分区 (保证缓存局部性)
 * 3. 哈希值缓存 (复用于 build)
 */
void partition_single_pass(const int32_t* keys, size_t count,
                           std::array<PartitionData, NUM_PARTITIONS>& partitions) {
    // Phase 1: 计算哈希 + 直方图 (一次遍历)
    std::array<size_t, NUM_PARTITIONS> histogram{};
    std::vector<uint32_t> hash_cache(count);

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

    // Phase 2: 预分配
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        partitions[p].resize(histogram[p]);
    }

    // Phase 3: 分散 (拷贝键 + 索引 + 哈希)
    std::array<size_t, NUM_PARTITIONS> write_pos{};

    for (size_t i = 0; i < count; ++i) {
        uint32_t h = hash_cache[i];
        size_t p = get_partition(h);
        size_t pos = write_pos[p]++;

        partitions[p].keys[pos] = keys[i];
        partitions[p].indices[pos] = static_cast<uint32_t>(i);
        partitions[p].hashes[pos] = h;
    }
}

} // anonymous namespace

// ============================================================================
// V10.2 并行 Join (直接写入最终缓冲区)
// ============================================================================

namespace {

/**
 * 并行 Join
 *
 * 优化:
 * 1. 使用预计算哈希构建哈希表 (无需重新计算)
 * 2. 直接写入最终缓冲区
 * 3. 最后压缩结果 (消除空隙)
 */
size_t join_parallel_v10_2(
    const std::array<PartitionData, NUM_PARTITIONS>& build_parts,
    const std::array<PartitionData, NUM_PARTITIONS>& probe_parts,
    JoinResult* result, size_t num_threads) {

    // 计算每分区的估计容量和偏移
    std::array<size_t, NUM_PARTITIONS> offsets;
    size_t total_capacity = 0;
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        offsets[p] = total_capacity;
        // 估计: 最大匹配数 = min(build, probe) (假设 1:1)
        size_t estimated = std::min(build_parts[p].count, probe_parts[p].count);
        if (estimated > 0) estimated = estimated * 2 + 1024;  // 留余量
        total_capacity += estimated;
    }

    // 确保结果缓冲区
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

            // 构建哈希表 (使用预计算哈希)
            CompactHashTable ht;
            ht.build(build_part.keys.data(), build_part.indices.data(),
                     build_part.hashes.data(), build_part.count);

            // 直接写入最终缓冲区
            size_t matches = ht.probe(
                probe_part.keys.data(), probe_part.indices.data(),
                probe_part.hashes.data(), probe_part.count,
                result->left_indices + offsets[p],
                result->right_indices + offsets[p]);

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

    // 压缩结果 (移除空隙)
    size_t total_matches = 0;
    size_t write_pos = 0;

    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        size_t matches = match_counts[p].load(std::memory_order_relaxed);
        if (matches == 0) continue;

        if (write_pos != offsets[p] && matches > 0) {
            std::memmove(result->left_indices + write_pos,
                        result->left_indices + offsets[p],
                        matches * sizeof(uint32_t));
            std::memmove(result->right_indices + write_pos,
                        result->right_indices + offsets[p],
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
// V10.2 小表直接 Join
// ============================================================================

namespace {

size_t join_direct_v10_2(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinResult* result) {
    // 预计算哈希
    std::vector<uint32_t> build_hashes(build_count);
    std::vector<uint32_t> probe_hashes(probe_count);
    std::vector<uint32_t> build_indices(build_count);
    std::vector<uint32_t> probe_indices(probe_count);

#ifdef __aarch64__
    size_t i = 0;
    for (; i + 8 <= build_count; i += 8) {
        hash_batch_8(build_keys + i, build_hashes.data() + i);
    }
    for (; i < build_count; ++i) {
        build_hashes[i] = hash_key(build_keys[i]);
    }

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

    for (size_t i = 0; i < build_count; ++i) build_indices[i] = i;
    for (size_t i = 0; i < probe_count; ++i) probe_indices[i] = i;

    // 构建哈希表
    CompactHashTable ht;
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
// V10.2 公开接口
// ============================================================================

size_t hash_join_i32_v10_2(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinType join_type,
                            JoinResult* result) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 小表直接 Join
    if (build_count < SMALL_TABLE_THRESHOLD) {
        return join_direct_v10_2(build_keys, build_count,
                                  probe_keys, probe_count, result);
    }

    // 大表使用分区
    std::array<PartitionData, NUM_PARTITIONS> build_parts;
    std::array<PartitionData, NUM_PARTITIONS> probe_parts;

    // 单遍分区 (哈希只计算一次)
    partition_single_pass(build_keys, build_count, build_parts);
    partition_single_pass(probe_keys, probe_count, probe_parts);

    // 确定线程数
    size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size_t(4));
    if (num_threads < 1) num_threads = 1;

    // 并行 Join
    size_t matches = join_parallel_v10_2(build_parts, probe_parts, result, num_threads);

    (void)join_type;
    return matches;
}

const char* get_v10_2_version_info() {
    return "V10.2 - 单次哈希 + 数据拷贝 + 直接写入";
}

} // namespace join
} // namespace thunderduck
