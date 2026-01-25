/**
 * ThunderDuck - Perfect Hash Table
 *
 * O(1) 直接索引哈希表，适用于连续或密集整数键
 * 当键范围 <= 2x 数据量时启用，避免哈希碰撞
 */

#ifndef THUNDERDUCK_PERFECT_HASH_H
#define THUNDERDUCK_PERFECT_HASH_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace join {

/**
 * 完美哈希表 - 直接索引
 *
 * 适用场景:
 * - 键为连续或近似连续整数
 * - 键范围 <= 2x 数据量
 * - 键范围 <= 10M (内存限制)
 * - 无重复键
 */
class PerfectHashTable {
public:
    static constexpr double DENSITY_THRESHOLD = 2.0;
    static constexpr size_t MAX_RANGE = 10000000;

    PerfectHashTable() = default;

    /**
     * 尝试构建完美哈希表
     *
     * @param keys 键数组
     * @param count 元素数量
     * @return true 如果成功构建，false 如果不适合使用完美哈希
     */
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
        if (range <= 0 || range > static_cast<int64_t>(count * DENSITY_THRESHOLD)) {
            return false;
        }
        if (range > static_cast<int64_t>(MAX_RANGE)) {
            return false;
        }

        // 检查是否有重复键 - 完美哈希只适用于唯一键
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
            return false;
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

    /**
     * 探测完美哈希表
     *
     * @param probe_keys 探测键数组
     * @param probe_count 探测数量
     * @param out_build 匹配的 build 索引
     * @param out_probe 匹配的 probe 索引
     * @return 匹配数量
     */
    size_t probe(const int32_t* probe_keys, size_t probe_count,
                 uint32_t* out_build, uint32_t* out_probe) const {
        if (!built_) return 0;

        size_t match_count = 0;

#ifdef __aarch64__
        // SIMD 批量探测
        int32x4_t min_key_vec = vdupq_n_s32(min_key_);
        int32x4_t max_offset_vec = vdupq_n_s32(static_cast<int32_t>(size_ - 1));

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
    size_t range_size() const { return size_; }

private:
    std::vector<uint32_t> indices_;
    int32_t min_key_ = 0;
    size_t size_ = 0;
    bool built_ = false;
};

} // namespace join
} // namespace thunderduck

#endif // THUNDERDUCK_PERFECT_HASH_H
