/**
 * ThunderDuck - Hash Join v15.0 Implementation
 *
 * V15 优化策略：简化版 - 直接使用 V3 的分区逻辑，仅优化探测阶段
 */

#include "thunderduck/join.h"
#include "thunderduck/memory.h"

#include <vector>
#include <array>
#include <thread>
#include <algorithm>
#include <cstring>
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {

namespace {

constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();
constexpr size_t NUM_THREADS_V15 = 4;
constexpr size_t MIN_PROBE_PER_THREAD = 100000;

const char* V15_VERSION = "V15.0 - 并行探测";

#ifdef __aarch64__
inline uint32_t crc32_hash(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}
#endif

// 简单哈希表 (单分区)
class SimpleHashTableV15 {
public:
    void build(const int32_t* keys, size_t count, float load_factor = 1.7f) {
        if (count == 0) {
            capacity_ = 0;
            return;
        }

        capacity_ = 1;
        while (capacity_ < count * load_factor) capacity_ <<= 1;
        mask_ = capacity_ - 1;

        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

#ifdef __aarch64__
        for (size_t i = 0; i < count; ++i) {
            int32_t key = keys[i];
            uint32_t hash = crc32_hash(key);
            size_t idx = hash & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                idx = (idx + 1) & mask_;
            }
            keys_[idx] = key;
            row_indices_[idx] = static_cast<uint32_t>(i);
        }
#endif
    }

    size_t probe_range(const int32_t* probe_keys, size_t start, size_t end,
                       uint32_t* out_build, uint32_t* out_probe) const {
        if (capacity_ == 0) return 0;

        size_t match_count = 0;

#ifdef __aarch64__
        for (size_t i = start; i < end; ++i) {
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
#endif
        return match_count;
    }

    // 公开成员以供并行访问
    std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
};

} // anonymous namespace

// ============================================================================
// V15 Hash Join 实现 - 简化版
// ============================================================================

size_t hash_join_i32_v15(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          uint32_t* out_build_indices, uint32_t* out_probe_indices,
                          JoinResult* result) {
    (void)result;  // 不使用

    if (build_count == 0 || probe_count == 0) return 0;

    // 1. 构建哈希表
    SimpleHashTableV15 hash_table;
    hash_table.build(build_keys, build_count);

    // 2. 选择线程数
    size_t num_threads = NUM_THREADS_V15;
    if (probe_count < MIN_PROBE_PER_THREAD * 2) {
        num_threads = 1;
    }

    if (num_threads == 1) {
        // 单线程直接探测
        return hash_table.probe_range(probe_keys, 0, probe_count,
                                       out_build_indices, out_probe_indices);
    }

    // 3. 并行探测 - 每线程本地缓冲
    struct ThreadResult {
        std::vector<uint32_t> build_indices;
        std::vector<uint32_t> probe_indices;
    };
    std::vector<ThreadResult> thread_results(num_threads);

    size_t chunk_size = (probe_count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, probe_count);

        if (start >= end) continue;

        thread_results[t].build_indices.reserve((end - start));
        thread_results[t].probe_indices.reserve((end - start));

        threads.emplace_back([&, t, start, end]() {
            auto& local_build = thread_results[t].build_indices;
            auto& local_probe = thread_results[t].probe_indices;

#ifdef __aarch64__
            for (size_t i = start; i < end; ++i) {
                int32_t key = probe_keys[i];
                uint32_t hash = crc32_hash(key);
                size_t idx = hash & (hash_table.capacity_ - 1);

                while (hash_table.keys_[idx] != EMPTY_KEY) {
                    if (hash_table.keys_[idx] == key) {
                        local_build.push_back(hash_table.row_indices_[idx]);
                        local_probe.push_back(static_cast<uint32_t>(i));
                    }
                    idx = (idx + 1) & (hash_table.capacity_ - 1);
                }
            }
#endif
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // 4. 合并结果
    size_t total = 0;
    for (const auto& tr : thread_results) {
        total += tr.build_indices.size();
    }

    size_t offset = 0;
    for (const auto& tr : thread_results) {
        if (!tr.build_indices.empty()) {
            std::memcpy(out_build_indices + offset, tr.build_indices.data(),
                       tr.build_indices.size() * sizeof(uint32_t));
            std::memcpy(out_probe_indices + offset, tr.probe_indices.data(),
                       tr.probe_indices.size() * sizeof(uint32_t));
            offset += tr.build_indices.size();
        }
    }

    return total;
}

const char* get_hash_join_v15_version() {
    return V15_VERSION;
}

} // namespace join
} // namespace thunderduck
