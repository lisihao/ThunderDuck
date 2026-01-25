/**
 * ThunderDuck - Hash Join v4 BloomFilter Strategy
 *
 * Bloom Filter 预过滤优化:
 * - 使用 Bloom Filter 快速排除不可能匹配的 probe keys
 * - 减少哈希表探测次数
 * - 适用于高选择性 join (匹配率低)
 */

#include "thunderduck/join.h"
#include "thunderduck/bloom_filter.h"
#include "thunderduck/memory.h"

#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <memory>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {
namespace v4 {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

// 缓存行大小
constexpr size_t M4_CACHE_LINE = 128;

// 空键标记
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();

// 批处理大小
constexpr size_t SIMD_BATCH_SIZE = 8;
constexpr size_t PREFETCH_DISTANCE = 16;

// Bloom 过滤批次大小
constexpr size_t BLOOM_BATCH_SIZE = 8192;

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

} // anonymous namespace

// ============================================================================
// SOA 哈希表 (复用设计)
// ============================================================================

class SOAHashTableBloom {
public:
    SOAHashTableBloom() = default;

    void build(const int32_t* keys, size_t count) {
        if (count == 0) return;

        capacity_ = 16;
        while (capacity_ < count * 1.7) {
            capacity_ *= 2;
        }
        mask_ = capacity_ - 1;
        size_ = count;

        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        for (size_t i = 0; i < count; ++i) {
            insert(keys[i], static_cast<uint32_t>(i));
        }
    }

    void insert(int32_t key, uint32_t row_idx) {
        uint32_t hash = crc32_hash(key);
        size_t idx = hash & mask_;

        while (keys_[idx] != EMPTY_KEY) {
            idx = (idx + 1) & mask_;
        }

        keys_[idx] = key;
        row_indices_[idx] = row_idx;
    }

#ifdef __aarch64__
    size_t probe_simd(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        size_t i = 0;
        for (; i + SIMD_BATCH_SIZE <= probe_count; i += SIMD_BATCH_SIZE) {
            if (i + PREFETCH_DISTANCE < probe_count) {
                uint32_t h0 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE]);
                uint32_t h1 = crc32_hash(probe_keys[i + PREFETCH_DISTANCE + 1]);
                __builtin_prefetch(&keys_[h0 & mask_], 0, 3);
                __builtin_prefetch(&keys_[h1 & mask_], 0, 3);
            }

            alignas(32) uint32_t hashes[SIMD_BATCH_SIZE];
            hash_8_keys(probe_keys + i, hashes);

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

    // 只对通过 Bloom Filter 的 probe keys 进行探测
    size_t probe_filtered(const int32_t* all_probe_keys,
                          const uint32_t* filtered_indices, size_t filtered_count,
                          uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;

        for (size_t i = 0; i < filtered_count; ++i) {
            uint32_t orig_idx = filtered_indices[i];
            int32_t key = all_probe_keys[orig_idx];
            uint32_t hash = crc32_hash(key);
            size_t idx = hash & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) {
                    out_build[match_count] = row_indices_[idx];
                    out_probe[match_count] = orig_idx;
                    ++match_count;
                }
                idx = (idx + 1) & mask_;
            }
        }

        return match_count;
    }

    size_t size() const { return size_; }

private:
    alignas(M4_CACHE_LINE) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// BloomFilter 策略实现
// ============================================================================

namespace {

size_t join_with_bloom(const int32_t* build_keys, size_t build_count,
                       const int32_t* probe_keys, size_t probe_count,
                       JoinResult* result,
                       double bloom_fpr) {

    // 1. 构建 Bloom Filter
    std::unique_ptr<bloom::BloomFilter> bloom_filter(
        bloom::build_bloom_filter(build_keys, build_count, bloom_fpr));

    // 2. 构建哈希表
    SOAHashTableBloom ht;
    ht.build(build_keys, build_count);

    // 3. 批量过滤 + 探测
    size_t total_matches = 0;
    std::vector<uint32_t> filtered_indices(BLOOM_BATCH_SIZE);
    std::vector<uint32_t> temp_build(BLOOM_BATCH_SIZE);
    std::vector<uint32_t> temp_probe(BLOOM_BATCH_SIZE);

    for (size_t i = 0; i < probe_count; i += BLOOM_BATCH_SIZE) {
        size_t batch_size = std::min(BLOOM_BATCH_SIZE, probe_count - i);

        // 3a. Bloom 过滤
        size_t filtered_count = bloom_filter->filter_batch(
            probe_keys + i, batch_size, filtered_indices.data());

        if (filtered_count == 0) {
            continue;  // 整个批次都被过滤掉
        }

        // 调整索引为全局索引
        for (size_t j = 0; j < filtered_count; ++j) {
            filtered_indices[j] += static_cast<uint32_t>(i);
        }

        // 3b. 对通过过滤的 keys 进行哈希表探测
        size_t batch_matches = ht.probe_filtered(
            probe_keys, filtered_indices.data(), filtered_count,
            temp_build.data(), temp_probe.data());

        // 3c. 确保结果缓冲区足够
        if (result->capacity < total_matches + batch_matches) {
            grow_join_result(result, (total_matches + batch_matches) * 2);
        }

        // 3d. 复制结果
        std::memcpy(result->left_indices + total_matches,
                   temp_build.data(), batch_matches * sizeof(uint32_t));
        std::memcpy(result->right_indices + total_matches,
                   temp_probe.data(), batch_matches * sizeof(uint32_t));

        total_matches += batch_matches;
    }

    result->count = total_matches;
    return total_matches;
}

// 多线程版本
size_t join_with_bloom_parallel(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinResult* result,
                                 double bloom_fpr,
                                 size_t num_threads) {

    // 1. 构建共享的 Bloom Filter
    std::unique_ptr<bloom::BloomFilter> bloom_filter(
        bloom::build_bloom_filter(build_keys, build_count, bloom_fpr));

    // 2. 构建共享的哈希表
    SOAHashTableBloom ht;
    ht.build(build_keys, build_count);

    // 3. 每线程的结果
    struct ThreadResult {
        std::vector<uint32_t> build_indices;
        std::vector<uint32_t> probe_indices;
        size_t count = 0;
    };
    std::vector<ThreadResult> thread_results(num_threads);

    // 4. 原子 probe 索引
    std::atomic<size_t> next_batch{0};
    const size_t batch_size = BLOOM_BATCH_SIZE;

    // 5. 工作函数
    auto worker = [&](size_t thread_id) {
        auto& local_result = thread_results[thread_id];
        local_result.build_indices.reserve(probe_count / num_threads);
        local_result.probe_indices.reserve(probe_count / num_threads);

        std::vector<uint32_t> filtered_indices(batch_size);
        std::vector<uint32_t> temp_build(batch_size);
        std::vector<uint32_t> temp_probe(batch_size);

        while (true) {
            size_t batch_start = next_batch.fetch_add(batch_size);
            if (batch_start >= probe_count) break;

            size_t current_batch_size = std::min(batch_size, probe_count - batch_start);

            // Bloom 过滤
            size_t filtered_count = bloom_filter->filter_batch(
                probe_keys + batch_start, current_batch_size, filtered_indices.data());

            if (filtered_count == 0) continue;

            // 调整为全局索引
            for (size_t j = 0; j < filtered_count; ++j) {
                filtered_indices[j] += static_cast<uint32_t>(batch_start);
            }

            // 哈希表探测
            size_t matches = ht.probe_filtered(
                probe_keys, filtered_indices.data(), filtered_count,
                temp_build.data(), temp_probe.data());

            // 保存到线程本地结果
            for (size_t i = 0; i < matches; ++i) {
                local_result.build_indices.push_back(temp_build[i]);
                local_result.probe_indices.push_back(temp_probe[i]);
            }
            local_result.count += matches;
        }
    };

    // 6. 启动线程
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) {
        t.join();
    }

    // 7. 合并结果
    size_t total_matches = 0;
    for (const auto& tr : thread_results) {
        total_matches += tr.count;
    }

    if (result->capacity < total_matches) {
        grow_join_result(result, total_matches);
    }

    size_t offset = 0;
    for (const auto& tr : thread_results) {
        if (tr.count > 0) {
            std::memcpy(result->left_indices + offset,
                       tr.build_indices.data(), tr.count * sizeof(uint32_t));
            std::memcpy(result->right_indices + offset,
                       tr.probe_indices.data(), tr.count * sizeof(uint32_t));
            offset += tr.count;
        }
    }
    result->count = total_matches;

    return total_matches;
}

} // anonymous namespace

// ============================================================================
// 直接 Hash Join (不使用 Bloom Filter)
// ============================================================================

namespace {

size_t join_direct_hash(const int32_t* build_keys, size_t build_count,
                        const int32_t* probe_keys, size_t probe_count,
                        JoinResult* result) {

    SOAHashTableBloom ht;
    ht.build(build_keys, build_count);

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

// 采样估计选择率
double estimate_selectivity(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count) {
    // 构建采样用的小 Bloom Filter
    constexpr size_t SAMPLE_SIZE = 1000;
    constexpr double SAMPLE_FPR = 0.01;

    size_t sample_count = std::min(SAMPLE_SIZE, probe_count);

    // 构建 build 端的 Bloom Filter
    std::unique_ptr<bloom::BloomFilter> bloom(
        bloom::build_bloom_filter(build_keys, build_count, SAMPLE_FPR));

    // 采样 probe 端，估计匹配率
    size_t hits = 0;
    size_t step = probe_count / sample_count;
    if (step < 1) step = 1;

    for (size_t i = 0; i < probe_count && (i / step) < sample_count; i += step) {
        if (bloom->maybe_contains(probe_keys[i])) {
            hits++;
        }
    }

    double selectivity = static_cast<double>(hits) / sample_count;
    return selectivity;
}

} // anonymous namespace

// ============================================================================
// BloomFilter 策略入口 (选择率感知)
// ============================================================================

size_t hash_join_bloom(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

    // ========== P2 优化: 选择率感知 ==========
    // Bloom Filter 只有在选择率低时才有价值
    // 高选择率 (>30%) 时直接使用哈希表更快

    constexpr double MAX_SELECTIVITY_FOR_BLOOM = 0.30;  // 30%

    double selectivity = estimate_selectivity(build_keys, build_count,
                                               probe_keys, probe_count);

    if (selectivity > MAX_SELECTIVITY_FOR_BLOOM) {
        // 高选择率，跳过 Bloom Filter，直接使用哈希表
        return join_direct_hash(build_keys, build_count,
                                probe_keys, probe_count, result);
    }

    // 低选择率，使用 Bloom Filter 预过滤
    size_t num_threads = config.num_threads;
    if (num_threads < 1) num_threads = 1;

    // 小数据量使用单线程
    if (build_count + probe_count < 200000 || num_threads == 1) {
        return join_with_bloom(build_keys, build_count,
                               probe_keys, probe_count,
                               result, config.bloom_fpr);
    }

    // 大数据量使用多线程
    return join_with_bloom_parallel(build_keys, build_count,
                                     probe_keys, probe_count,
                                     result, config.bloom_fpr,
                                     num_threads);
}

} // namespace v4
} // namespace join
} // namespace thunderduck
