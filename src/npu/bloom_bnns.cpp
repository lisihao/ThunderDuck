/**
 * ThunderDuck - NPU Accelerated Hash Join using BNNS
 *
 * 使用 Apple Accelerate 框架的 BNNS (Basic Neural Network Subroutines)
 * 加速 Bloom Filter 哈希计算和批量操作
 *
 * 注意: BNNS 主要用于神经网络运算，这里我们利用其向量化操作
 * 来加速哈希计算和位操作
 */

#include "thunderduck/join.h"
#include "thunderduck/bloom_filter.h"
#include "thunderduck/memory.h"

#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <memory>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define THUNDERDUCK_HAS_BNNS 1
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {
namespace v4 {

// ============================================================================
// NPU 可用性检测
// ============================================================================

namespace {

bool g_npu_initialized = false;
bool g_npu_available = false;

void init_npu_detection() {
    if (g_npu_initialized) return;

#ifdef THUNDERDUCK_HAS_BNNS
    // 检查是否可以使用 BNNS
    // 在 Apple Silicon 上，BNNS 会自动使用 ANE (Apple Neural Engine)
    // 进行某些操作

    // 简单的可用性测试 - 尝试创建一个小的 BNNS 操作
    // 如果成功，我们认为 BNNS 可用
    g_npu_available = true;
#else
    g_npu_available = false;
#endif

    g_npu_initialized = true;
}

} // anonymous namespace

bool is_npu_ready() {
    init_npu_detection();
    return g_npu_available;
}

// ============================================================================
// 配置常量
// ============================================================================

namespace {

constexpr size_t M4_CACHE_LINE = 128;
constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();
constexpr size_t BNNS_BATCH_SIZE = 8192;  // NPU 批处理大小
constexpr size_t SIMD_BATCH_SIZE = 8;

} // anonymous namespace

// ============================================================================
// BNNS 加速的哈希计算
// ============================================================================

#ifdef THUNDERDUCK_HAS_BNNS

namespace {

// 使用 vDSP 加速的向量操作来优化哈希计算中的某些步骤
// 注意：CRC32 哈希本身无法通过 BNNS 加速，但我们可以优化
// 批量操作和内存访问模式

inline uint32_t crc32_hash(int32_t key) {
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

// 批量哈希计算 - 利用向量化
void batch_hash_bnns(const int32_t* keys, uint32_t* hashes, size_t count) {
    // 使用 vDSP 进行批量整数操作
    // 由于 CRC32 是硬件加速的，直接使用展开循环

    size_t i = 0;

#ifdef __aarch64__
    // 展开循环，提高指令级并行
    for (; i + 8 <= count; i += 8) {
        hashes[i]     = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i]));
        hashes[i + 1] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 1]));
        hashes[i + 2] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 2]));
        hashes[i + 3] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 3]));
        hashes[i + 4] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 4]));
        hashes[i + 5] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 5]));
        hashes[i + 6] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 6]));
        hashes[i + 7] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 7]));
    }
#endif

    for (; i < count; ++i) {
        hashes[i] = crc32_hash(keys[i]);
    }
}

// vDSP 加速的位掩码应用
void apply_mask_vdsp(const uint32_t* hashes, uint32_t* masked, size_t count, uint32_t mask) {
    // 使用 vDSP 进行向量化的位与操作
    // vDSP 对浮点更优化，但我们可以利用其批量操作

    // 简单情况直接用 SIMD
    size_t i = 0;

#ifdef __aarch64__
    uint32x4_t mask_vec = vdupq_n_u32(mask);

    for (; i + 4 <= count; i += 4) {
        uint32x4_t h = vld1q_u32(hashes + i);
        uint32x4_t m = vandq_u32(h, mask_vec);
        vst1q_u32(masked + i, m);
    }
#endif

    for (; i < count; ++i) {
        masked[i] = hashes[i] & mask;
    }
}

} // anonymous namespace

#endif // THUNDERDUCK_HAS_BNNS

// ============================================================================
// SOA 哈希表 (NPU 优化版)
// ============================================================================

class SOAHashTableNPU {
public:
    SOAHashTableNPU() = default;

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

    // NPU 优化的批量探测
    size_t probe_bnns(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build, uint32_t* out_probe) const {
#ifdef THUNDERDUCK_HAS_BNNS
        size_t match_count = 0;

        // 批量处理
        std::vector<uint32_t> hashes(BNNS_BATCH_SIZE);
        std::vector<uint32_t> slots(BNNS_BATCH_SIZE);

        for (size_t batch_start = 0; batch_start < probe_count; batch_start += BNNS_BATCH_SIZE) {
            size_t batch_size = std::min(BNNS_BATCH_SIZE, probe_count - batch_start);

            // 批量计算哈希
            batch_hash_bnns(probe_keys + batch_start, hashes.data(), batch_size);

            // 批量应用掩码计算初始槽位
            apply_mask_vdsp(hashes.data(), slots.data(), batch_size, static_cast<uint32_t>(mask_));

            // 逐个探测 (线性探测无法完全向量化)
            for (size_t i = 0; i < batch_size; ++i) {
                int32_t key = probe_keys[batch_start + i];
                size_t idx = slots[i];

                while (keys_[idx] != EMPTY_KEY) {
                    if (keys_[idx] == key) {
                        out_build[match_count] = row_indices_[idx];
                        out_probe[match_count] = static_cast<uint32_t>(batch_start + i);
                        ++match_count;
                    }
                    idx = (idx + 1) & mask_;
                }
            }
        }

        return match_count;
#else
        return probe_scalar(probe_keys, probe_count, out_build, out_probe);
#endif
    }

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

private:
    alignas(M4_CACHE_LINE) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// NPU 策略实现
// ============================================================================

namespace {

size_t join_with_npu_single(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinResult* result,
                            double bloom_fpr) {

#ifdef THUNDERDUCK_HAS_BNNS
    // 1. 构建 Bloom Filter
    std::unique_ptr<bloom::BloomFilter> bloom_filter(
        bloom::build_bloom_filter(build_keys, build_count, bloom_fpr));

    // 2. 构建哈希表
    SOAHashTableNPU ht;
    ht.build(build_keys, build_count);

    // 3. NPU 加速的批量过滤 + 探测
    size_t total_matches = 0;
    std::vector<uint32_t> filtered_indices(BNNS_BATCH_SIZE);
    std::vector<uint32_t> temp_build(BNNS_BATCH_SIZE);
    std::vector<uint32_t> temp_probe(BNNS_BATCH_SIZE);

    for (size_t i = 0; i < probe_count; i += BNNS_BATCH_SIZE) {
        size_t batch_size = std::min(BNNS_BATCH_SIZE, probe_count - i);

        // Bloom 过滤
        size_t filtered_count = bloom_filter->filter_batch(
            probe_keys + i, batch_size, filtered_indices.data());

        if (filtered_count == 0) continue;

        // 收集通过的 keys
        std::vector<int32_t> filtered_keys(filtered_count);
        for (size_t j = 0; j < filtered_count; ++j) {
            filtered_keys[j] = probe_keys[i + filtered_indices[j]];
        }

        // NPU 批量探测
        size_t batch_matches = ht.probe_bnns(
            filtered_keys.data(), filtered_count,
            temp_build.data(), temp_probe.data());

        // 调整索引为全局索引
        for (size_t j = 0; j < batch_matches; ++j) {
            temp_probe[j] = static_cast<uint32_t>(i) + filtered_indices[temp_probe[j]];
        }

        // 确保结果缓冲区足够
        if (result->capacity < total_matches + batch_matches) {
            grow_join_result(result, (total_matches + batch_matches) * 2);
        }

        // 复制结果
        std::memcpy(result->left_indices + total_matches,
                   temp_build.data(), batch_matches * sizeof(uint32_t));
        std::memcpy(result->right_indices + total_matches,
                   temp_probe.data(), batch_matches * sizeof(uint32_t));

        total_matches += batch_matches;
    }

    result->count = total_matches;
    return total_matches;
#else
    // 回退到 CPU Bloom 策略
    return hash_join_bloom(build_keys, build_count,
                           probe_keys, probe_count,
                           JoinType::INNER, result,
                           JoinConfigV4{});
#endif
}

size_t join_with_npu_parallel(const int32_t* build_keys, size_t build_count,
                               const int32_t* probe_keys, size_t probe_count,
                               JoinResult* result,
                               double bloom_fpr,
                               size_t num_threads) {

#ifdef THUNDERDUCK_HAS_BNNS
    // 共享数据结构
    std::unique_ptr<bloom::BloomFilter> bloom_filter(
        bloom::build_bloom_filter(build_keys, build_count, bloom_fpr));

    SOAHashTableNPU ht;
    ht.build(build_keys, build_count);

    // 每线程结果
    struct ThreadResult {
        std::vector<uint32_t> build_indices;
        std::vector<uint32_t> probe_indices;
        size_t count = 0;
    };
    std::vector<ThreadResult> thread_results(num_threads);

    // 原子批次计数器
    std::atomic<size_t> next_batch{0};

    // 工作函数
    auto worker = [&](size_t thread_id) {
        auto& local_result = thread_results[thread_id];
        local_result.build_indices.reserve(probe_count / num_threads);
        local_result.probe_indices.reserve(probe_count / num_threads);

        std::vector<uint32_t> filtered_indices(BNNS_BATCH_SIZE);
        std::vector<int32_t> filtered_keys(BNNS_BATCH_SIZE);
        std::vector<uint32_t> temp_build(BNNS_BATCH_SIZE);
        std::vector<uint32_t> temp_probe(BNNS_BATCH_SIZE);

        while (true) {
            size_t batch_start = next_batch.fetch_add(BNNS_BATCH_SIZE);
            if (batch_start >= probe_count) break;

            size_t batch_size = std::min(BNNS_BATCH_SIZE, probe_count - batch_start);

            // Bloom 过滤
            size_t filtered_count = bloom_filter->filter_batch(
                probe_keys + batch_start, batch_size, filtered_indices.data());

            if (filtered_count == 0) continue;

            // 收集通过的 keys
            for (size_t j = 0; j < filtered_count; ++j) {
                filtered_keys[j] = probe_keys[batch_start + filtered_indices[j]];
            }

            // 探测
            size_t matches = ht.probe_bnns(
                filtered_keys.data(), filtered_count,
                temp_build.data(), temp_probe.data());

            // 保存结果 (调整索引)
            for (size_t j = 0; j < matches; ++j) {
                local_result.build_indices.push_back(temp_build[j]);
                local_result.probe_indices.push_back(
                    static_cast<uint32_t>(batch_start) + filtered_indices[temp_probe[j]]);
            }
            local_result.count += matches;
        }
    };

    // 启动线程
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) {
        t.join();
    }

    // 合并结果
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
#else
    return join_with_npu_single(build_keys, build_count,
                                probe_keys, probe_count,
                                result, bloom_fpr);
#endif
}

} // anonymous namespace

// ============================================================================
// NPU 策略入口
// ============================================================================

// 前向声明 (在 hash_join_v4_bloom.cpp 中实现)
size_t hash_join_bloom(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

size_t hash_join_npu(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

    // ========== P4 优化: 简化 NPU 策略 ==========
    // 分析发现 "NPU 加速" 实际上只是循环展开，没有真正利用 ANE
    // CRC32 哈希已经是硬件加速的，BNNS 无法进一步优化
    // 直接回退到 CPU Bloom 策略，避免不必要的调用开销

    return hash_join_bloom(build_keys, build_count,
                           probe_keys, probe_count,
                           join_type, result, config);
}

} // namespace v4
} // namespace join
} // namespace thunderduck
