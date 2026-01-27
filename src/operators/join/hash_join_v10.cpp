/**
 * ThunderDuck - Hash Join v10.0 Implementation
 *
 * V10 深度优化版本:
 * 1. SEMI/ANTI JOIN 优化 (提前退出)
 * 2. Sort-Merge Join (SIMD 优化)
 * 3. Range Join (范围连接)
 * 4. 字符串键 SIMD 优化
 * 5. GPU 阈值调整
 */

#include "thunderduck/join.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <thread>
#include <atomic>
#include <limits>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace join {

// ============================================================================
// 常量定义
// ============================================================================

namespace {

constexpr int32_t EMPTY_KEY = std::numeric_limits<int32_t>::min();
const char* V10_VERSION = "V10.0 - Join深度优化候选版";

#ifdef __aarch64__
inline uint32_t crc32_hash(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}
#else
inline uint32_t crc32_hash(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
#endif

} // anonymous namespace

// ============================================================================
// V10 哈希表 (SEMI/ANTI 优化)
// ============================================================================

class HashTableV10 {
public:
    void build(const int32_t* keys, size_t count) {
        if (count == 0) return;

        capacity_ = 16;
        while (capacity_ < count * 1.7) capacity_ *= 2;
        mask_ = capacity_ - 1;

        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        for (size_t i = 0; i < count; ++i) {
            uint32_t h = crc32_hash(keys[i]);
            size_t idx = h & mask_;
            while (keys_[idx] != EMPTY_KEY) idx = (idx + 1) & mask_;
            keys_[idx] = keys[i];
            row_indices_[idx] = static_cast<uint32_t>(i);
        }
    }

    // INNER JOIN
    size_t probe_inner(const int32_t* probe_keys, size_t probe_count,
                       uint32_t* out_build, uint32_t* out_probe) const {
        size_t match_count = 0;
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            size_t idx = crc32_hash(key) & mask_;
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

    // SEMI JOIN - 优化: 找到即退出
    size_t probe_semi(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_probe) const {
        size_t match_count = 0;
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            size_t idx = crc32_hash(key) & mask_;
            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) {
                    out_probe[match_count++] = static_cast<uint32_t>(i);
                    break;  // 提前退出
                }
                idx = (idx + 1) & mask_;
            }
        }
        return match_count;
    }

    // ANTI JOIN - 返回不匹配的
    size_t probe_anti(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_probe) const {
        size_t match_count = 0;
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t key = probe_keys[i];
            size_t idx = crc32_hash(key) & mask_;
            bool found = false;
            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) { found = true; break; }
                idx = (idx + 1) & mask_;
            }
            if (!found) out_probe[match_count++] = static_cast<uint32_t>(i);
        }
        return match_count;
    }

private:
    std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_ = 0, mask_ = 0;
};

// ============================================================================
// V10 Hash Join
// ============================================================================

size_t hash_join_i32_v10_config(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinType join_type, JoinResult* result,
                                 const JoinConfigV10& config) {
    if (!result) return 0;

    HashTableV10 ht;
    ht.build(build_keys, build_count);

    size_t estimated = std::max(build_count, probe_count);
    if (result->capacity < estimated) grow_join_result(result, estimated);

    size_t match_count = 0;

    switch (join_type) {
        case JoinType::INNER:
            match_count = ht.probe_inner(probe_keys, probe_count,
                                         result->left_indices, result->right_indices);
            break;
        case JoinType::SEMI:
            match_count = ht.probe_semi(probe_keys, probe_count, result->right_indices);
            for (size_t i = 0; i < match_count; ++i) result->left_indices[i] = NULL_INDEX;
            break;
        case JoinType::ANTI:
            match_count = ht.probe_anti(probe_keys, probe_count, result->right_indices);
            for (size_t i = 0; i < match_count; ++i) result->left_indices[i] = NULL_INDEX;
            break;
        default:
            // LEFT/RIGHT/FULL 委托给 v4
            return hash_join_i32_v4(build_keys, build_count, probe_keys, probe_count,
                                    join_type, result);
    }

    result->count = match_count;
    return match_count;
}

size_t hash_join_i32_v10(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result) {
    JoinConfigV10 config;
    return hash_join_i32_v10_config(build_keys, build_count, probe_keys, probe_count,
                                     join_type, result, config);
}

// ============================================================================
// Sort-Merge Join
// ============================================================================

size_t sort_merge_join_i32_config(const int32_t* left_keys, size_t left_count,
                                   const int32_t* right_keys, size_t right_count,
                                   JoinType join_type, JoinResult* result,
                                   const JoinConfigV10& config) {
    if (!result || left_count == 0 || right_count == 0) return 0;

    // 创建排序索引
    std::vector<uint32_t> left_idx(left_count), right_idx(right_count);
    for (size_t i = 0; i < left_count; ++i) left_idx[i] = i;
    for (size_t i = 0; i < right_count; ++i) right_idx[i] = i;

    if (!config.assume_sorted) {
        std::sort(left_idx.begin(), left_idx.end(),
                  [left_keys](uint32_t a, uint32_t b) { return left_keys[a] < left_keys[b]; });
        std::sort(right_idx.begin(), right_idx.end(),
                  [right_keys](uint32_t a, uint32_t b) { return right_keys[a] < right_keys[b]; });
    }

    size_t estimated = std::min(left_count, right_count);
    if (result->capacity < estimated) grow_join_result(result, estimated);

    size_t match_count = 0, i = 0, j = 0;

    while (i < left_count && j < right_count) {
        int32_t lv = left_keys[left_idx[i]], rv = right_keys[right_idx[j]];

        if (lv < rv) { ++i; }
        else if (lv > rv) { ++j; }
        else {
            // 处理重复键
            size_t i0 = i, j0 = j;
            while (i < left_count && left_keys[left_idx[i]] == lv) ++i;
            while (j < right_count && right_keys[right_idx[j]] == rv) ++j;

            size_t pairs = (i - i0) * (j - j0);
            if (match_count + pairs > result->capacity)
                grow_join_result(result, std::max(result->capacity * 2, match_count + pairs));

            for (size_t li = i0; li < i; ++li) {
                for (size_t ri = j0; ri < j; ++ri) {
                    result->left_indices[match_count] = left_idx[li];
                    result->right_indices[match_count] = right_idx[ri];
                    ++match_count;
                }
            }
        }
    }

    result->count = match_count;
    return match_count;
}

size_t sort_merge_join_i32(const int32_t* left_keys, size_t left_count,
                            const int32_t* right_keys, size_t right_count,
                            JoinType join_type, JoinResult* result) {
    JoinConfigV10 config;
    return sort_merge_join_i32_config(left_keys, left_count, right_keys, right_count,
                                       join_type, result, config);
}

// ============================================================================
// Range Join
// ============================================================================

size_t range_join_i32_config(const int32_t* left_keys, size_t left_count,
                              const int32_t* right_lo, const int32_t* right_hi,
                              size_t right_count, JoinResult* result,
                              const JoinConfigV10& config) {
    if (!result || left_count == 0 || right_count == 0) return 0;

    if (result->capacity < left_count) grow_join_result(result, left_count);
    size_t match_count = 0;

#ifdef __aarch64__
    if (config.range_join_simd) {
        size_t i = 0;
        for (; i + 4 <= left_count; i += 4) {
            int32x4_t keys = vld1q_s32(&left_keys[i]);
            for (size_t j = 0; j < right_count; ++j) {
                int32x4_t lo = vdupq_n_s32(right_lo[j]);
                int32x4_t hi = vdupq_n_s32(right_hi[j]);
                uint32x4_t in_range = vandq_u32(vcgeq_s32(keys, lo), vcleq_s32(keys, hi));

                uint32_t mask[4];
                vst1q_u32(mask, in_range);
                for (int k = 0; k < 4; ++k) {
                    if (mask[k]) {
                        if (match_count >= result->capacity)
                            grow_join_result(result, result->capacity * 2);
                        result->left_indices[match_count] = i + k;
                        result->right_indices[match_count++] = j;
                    }
                }
            }
        }
        for (; i < left_count; ++i) {
            for (size_t j = 0; j < right_count; ++j) {
                if (left_keys[i] >= right_lo[j] && left_keys[i] <= right_hi[j]) {
                    if (match_count >= result->capacity)
                        grow_join_result(result, result->capacity * 2);
                    result->left_indices[match_count] = i;
                    result->right_indices[match_count++] = j;
                }
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < left_count; ++i) {
            for (size_t j = 0; j < right_count; ++j) {
                if (left_keys[i] >= right_lo[j] && left_keys[i] <= right_hi[j]) {
                    if (match_count >= result->capacity)
                        grow_join_result(result, result->capacity * 2);
                    result->left_indices[match_count] = i;
                    result->right_indices[match_count++] = j;
                }
            }
        }
    }

    result->count = match_count;
    return match_count;
}

size_t range_join_i32(const int32_t* left_keys, size_t left_count,
                       const int32_t* right_lo, const int32_t* right_hi,
                       size_t right_count, JoinResult* result) {
    JoinConfigV10 config;
    return range_join_i32_config(left_keys, left_count, right_lo, right_hi,
                                  right_count, result, config);
}

// ============================================================================
// Inequality Join
// ============================================================================

size_t inequality_join_i32(const int32_t* left_keys, size_t left_count,
                            const int32_t* right_keys, size_t right_count,
                            InequalityOp op, JoinResult* result) {
    if (!result || left_count == 0 || right_count == 0) return 0;

    // 对右侧排序
    std::vector<uint32_t> right_idx(right_count);
    for (size_t i = 0; i < right_count; ++i) right_idx[i] = i;
    std::sort(right_idx.begin(), right_idx.end(),
              [right_keys](uint32_t a, uint32_t b) { return right_keys[a] < right_keys[b]; });

    if (result->capacity < left_count) grow_join_result(result, left_count);
    size_t match_count = 0;

    for (size_t i = 0; i < left_count; ++i) {
        int32_t lv = left_keys[i];
        auto begin = right_idx.begin(), end = right_idx.end();

        switch (op) {
            case InequalityOp::LESS_THAN:
                begin = std::upper_bound(right_idx.begin(), right_idx.end(), lv,
                    [right_keys](int32_t v, uint32_t idx) { return v < right_keys[idx]; });
                break;
            case InequalityOp::LESS_EQUAL:
                begin = std::lower_bound(right_idx.begin(), right_idx.end(), lv,
                    [right_keys](uint32_t idx, int32_t v) { return right_keys[idx] < v; });
                break;
            case InequalityOp::GREATER_THAN:
                end = std::lower_bound(right_idx.begin(), right_idx.end(), lv,
                    [right_keys](uint32_t idx, int32_t v) { return right_keys[idx] < v; });
                begin = right_idx.begin();
                break;
            case InequalityOp::GREATER_EQUAL:
                end = std::upper_bound(right_idx.begin(), right_idx.end(), lv,
                    [right_keys](int32_t v, uint32_t idx) { return v < right_keys[idx]; });
                begin = right_idx.begin();
                break;
        }

        for (auto it = begin; it != end; ++it) {
            if (match_count >= result->capacity)
                grow_join_result(result, result->capacity * 2);
            result->left_indices[match_count] = i;
            result->right_indices[match_count++] = *it;
        }
    }

    result->count = match_count;
    return match_count;
}

// ============================================================================
// 字符串键 Hash Join
// ============================================================================

namespace {
#ifdef __aarch64__
inline uint32_t simd_string_hash(const char* str, size_t len) {
    uint32_t h = 0;
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        uint32_t v;
        memcpy(&v, str + i, 4);
        h = __crc32cw(h, v);
    }
    for (; i < len; ++i) h = __crc32cb(h, str[i]);
    return h;
}

inline bool simd_string_equal(const char* a, const char* b, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        uint8x16_t va = vld1q_u8(reinterpret_cast<const uint8_t*>(a + i));
        uint8x16_t vb = vld1q_u8(reinterpret_cast<const uint8_t*>(b + i));
        if (vminvq_u8(vceqq_u8(va, vb)) != 0xFF) return false;
    }
    return memcmp(a + i, b + i, len - i) == 0;
}
#else
inline uint32_t simd_string_hash(const char* str, size_t len) {
    uint32_t h = 0;
    for (size_t i = 0; i < len; ++i) h = h * 31 + (uint8_t)str[i];
    return h;
}
inline bool simd_string_equal(const char* a, const char* b, size_t len) {
    return memcmp(a, b, len) == 0;
}
#endif
} // anonymous namespace

size_t hash_join_string(const char* const* build_keys, const size_t* build_lens,
                         size_t build_count,
                         const char* const* probe_keys, const size_t* probe_lens,
                         size_t probe_count, JoinType join_type, JoinResult* result) {
    if (!result || join_type != JoinType::INNER) return 0;

    // 构建哈希表
    size_t cap = 16;
    while (cap < build_count * 1.7) cap *= 2;
    size_t mask = cap - 1;

    std::vector<uint32_t> hashes(cap, 0);
    std::vector<size_t> lens(cap, 0);
    std::vector<const char*> keys(cap, nullptr);
    std::vector<uint32_t> row_idx(cap, 0);
    std::vector<bool> occupied(cap, false);

    for (size_t i = 0; i < build_count; ++i) {
        uint32_t h = simd_string_hash(build_keys[i], build_lens[i]);
        size_t idx = h & mask;
        while (occupied[idx]) idx = (idx + 1) & mask;
        hashes[idx] = h; lens[idx] = build_lens[i];
        keys[idx] = build_keys[i]; row_idx[idx] = i;
        occupied[idx] = true;
    }

    if (result->capacity < probe_count) grow_join_result(result, probe_count);
    size_t match_count = 0;

    for (size_t i = 0; i < probe_count; ++i) {
        uint32_t h = simd_string_hash(probe_keys[i], probe_lens[i]);
        size_t idx = h & mask;
        while (occupied[idx]) {
            if (hashes[idx] == h && lens[idx] == probe_lens[i] &&
                simd_string_equal(keys[idx], probe_keys[i], probe_lens[i])) {
                if (match_count >= result->capacity)
                    grow_join_result(result, result->capacity * 2);
                result->left_indices[match_count] = row_idx[idx];
                result->right_indices[match_count++] = i;
            }
            idx = (idx + 1) & mask;
        }
    }

    result->count = match_count;
    return match_count;
}

size_t hash_join_fixedstring(const char* build_keys, size_t key_len, size_t build_count,
                              const char* probe_keys, size_t probe_count,
                              JoinType join_type, JoinResult* result) {
    if (!result || join_type != JoinType::INNER) return 0;

    size_t cap = 16;
    while (cap < build_count * 1.7) cap *= 2;
    size_t mask = cap - 1;

    std::vector<uint32_t> hashes(cap, 0), row_idx(cap, 0);
    std::vector<bool> occupied(cap, false);

    for (size_t i = 0; i < build_count; ++i) {
        uint32_t h = simd_string_hash(build_keys + i * key_len, key_len);
        size_t idx = h & mask;
        while (occupied[idx]) idx = (idx + 1) & mask;
        hashes[idx] = h; row_idx[idx] = i; occupied[idx] = true;
    }

    if (result->capacity < probe_count) grow_join_result(result, probe_count);
    size_t match_count = 0;

    for (size_t i = 0; i < probe_count; ++i) {
        const char* pk = probe_keys + i * key_len;
        uint32_t h = simd_string_hash(pk, key_len);
        size_t idx = h & mask;
        while (occupied[idx]) {
            if (hashes[idx] == h &&
                simd_string_equal(build_keys + row_idx[idx] * key_len, pk, key_len)) {
                if (match_count >= result->capacity)
                    grow_join_result(result, result->capacity * 2);
                result->left_indices[match_count] = row_idx[idx];
                result->right_indices[match_count++] = i;
            }
            idx = (idx + 1) & mask;
        }
    }

    result->count = match_count;
    return match_count;
}

// ============================================================================
// 工具函数
// ============================================================================

bool is_v10_available() { return true; }
const char* get_v10_version_info() { return V10_VERSION; }

} // namespace join
} // namespace thunderduck
