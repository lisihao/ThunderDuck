/**
 * ThunderDuck - SIMD Hash Join Implementation
 * 
 * ARM Neon 加速的哈希连接实现
 */

#include "thunderduck/join.h"
#include "thunderduck/simd.h"
#include "thunderduck/memory.h"

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>  // for __crc32cw
#endif

#include <vector>
#include <cstring>
#include <algorithm>

namespace thunderduck {
namespace join {

// ============================================================================
// 哈希函数实现
// ============================================================================

namespace {

#ifdef __aarch64__
// 使用 CRC32 指令加速哈希
inline uint32_t crc32_hash_i32(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

inline uint64_t crc32_hash_i64(int64_t key) {
    uint64_t hash = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(key));
    return hash;
}
#else
// 回退到 MurmurHash3 风格的哈希
inline uint32_t fallback_hash_i32(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

inline uint64_t fallback_hash_i64(int64_t key) {
    uint64_t h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
#endif

} // anonymous namespace

inline uint32_t hash_one_i32(int32_t key) {
#ifdef __aarch64__
    return crc32_hash_i32(key);
#else
    return fallback_hash_i32(key);
#endif
}

inline uint64_t hash_one_i64(int64_t key) {
#ifdef __aarch64__
    return crc32_hash_i64(key);
#else
    return fallback_hash_i64(key);
#endif
}

void hash_i32(const int32_t* keys, uint32_t* hashes, size_t count) {
#ifdef __aarch64__
    // 展开循环以利用指令流水线
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        hashes[i + 0] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 0]));
        hashes[i + 1] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 1]));
        hashes[i + 2] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 2]));
        hashes[i + 3] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 3]));
    }
    for (; i < count; ++i) {
        hashes[i] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i]));
    }
#else
    for (size_t i = 0; i < count; ++i) {
        hashes[i] = fallback_hash_i32(keys[i]);
    }
#endif
}

void hash_i64(const int64_t* keys, uint64_t* hashes, size_t count) {
#ifdef __aarch64__
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        hashes[i + 0] = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(keys[i + 0]));
        hashes[i + 1] = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(keys[i + 1]));
        hashes[i + 2] = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(keys[i + 2]));
        hashes[i + 3] = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(keys[i + 3]));
    }
    for (; i < count; ++i) {
        hashes[i] = __crc32cd(0xFFFFFFFF, static_cast<uint64_t>(keys[i]));
    }
#else
    for (size_t i = 0; i < count; ++i) {
        hashes[i] = fallback_hash_i64(keys[i]);
    }
#endif
}

// ============================================================================
// SIMD 键匹配
// ============================================================================

size_t simd_find_matches_i32(const int32_t* candidates, size_t candidate_count,
                             int32_t probe_key, uint32_t* out_matches) {
    size_t match_count = 0;

#ifdef __aarch64__
    int32x4_t probe_vec = vdupq_n_s32(probe_key);
    size_t i = 0;
    
    for (; i + 4 <= candidate_count; i += 4) {
        int32x4_t cand_vec = vld1q_s32(candidates + i);
        uint32x4_t eq_mask = vceqq_s32(cand_vec, probe_vec);
        
        // 检查是否有任何匹配
        if (vmaxvq_u32(eq_mask) != 0) {
            // 提取匹配的索引
            uint32_t mask_bits = 0;
            mask_bits |= (vgetq_lane_u32(eq_mask, 0) ? 1 : 0);
            mask_bits |= (vgetq_lane_u32(eq_mask, 1) ? 2 : 0);
            mask_bits |= (vgetq_lane_u32(eq_mask, 2) ? 4 : 0);
            mask_bits |= (vgetq_lane_u32(eq_mask, 3) ? 8 : 0);
            
            if (mask_bits & 1) out_matches[match_count++] = static_cast<uint32_t>(i);
            if (mask_bits & 2) out_matches[match_count++] = static_cast<uint32_t>(i + 1);
            if (mask_bits & 4) out_matches[match_count++] = static_cast<uint32_t>(i + 2);
            if (mask_bits & 8) out_matches[match_count++] = static_cast<uint32_t>(i + 3);
        }
    }
    
    // 处理剩余元素
    for (; i < candidate_count; ++i) {
        if (candidates[i] == probe_key) {
            out_matches[match_count++] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidates[i] == probe_key) {
            out_matches[match_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return match_count;
}

size_t simd_find_matches_i64(const int64_t* candidates, size_t candidate_count,
                             int64_t probe_key, uint32_t* out_matches) {
    size_t match_count = 0;

#ifdef __aarch64__
    int64x2_t probe_vec = vdupq_n_s64(probe_key);
    size_t i = 0;
    
    for (; i + 2 <= candidate_count; i += 2) {
        int64x2_t cand_vec = vld1q_s64(candidates + i);
        uint64x2_t eq_mask = vceqq_s64(cand_vec, probe_vec);
        
        if (vgetq_lane_u64(eq_mask, 0)) {
            out_matches[match_count++] = static_cast<uint32_t>(i);
        }
        if (vgetq_lane_u64(eq_mask, 1)) {
            out_matches[match_count++] = static_cast<uint32_t>(i + 1);
        }
    }
    
    for (; i < candidate_count; ++i) {
        if (candidates[i] == probe_key) {
            out_matches[match_count++] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidates[i] == probe_key) {
            out_matches[match_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return match_count;
}

// ============================================================================
// 哈希表实现
// ============================================================================

struct HashTable::Impl {
    // 桶结构：链表节点
    struct Entry {
        int64_t key;        // 存储键（支持 i32 和 i64）
        uint32_t row_idx;   // 原始行索引
        uint32_t next;      // 下一个节点索引（0xFFFFFFFF 表示无）
    };
    
    static constexpr uint32_t EMPTY = 0xFFFFFFFF;
    
    std::vector<uint32_t> buckets;  // 桶数组，存储 Entry 索引
    std::vector<Entry> entries;     // 条目数组
    size_t num_buckets;
    size_t entry_count;
    bool is_i32;                    // 键类型
    
    Impl(size_t expected_size) {
        // 选择桶数量为 2 的幂，负载因子约 0.7
        num_buckets = 1;
        while (num_buckets < expected_size * 1.5) {
            num_buckets *= 2;
        }
        buckets.resize(num_buckets, EMPTY);
        entries.reserve(expected_size);
        entry_count = 0;
        is_i32 = true;
    }
    
    void clear() {
        std::fill(buckets.begin(), buckets.end(), EMPTY);
        entries.clear();
        entry_count = 0;
    }
    
    void build_i32(const int32_t* keys, size_t count) {
        is_i32 = true;
        entries.resize(count);
        
        for (size_t i = 0; i < count; ++i) {
            uint32_t hash = hash_one_i32(keys[i]);
            uint32_t bucket_idx = hash & (num_buckets - 1);
            
            entries[i].key = keys[i];
            entries[i].row_idx = static_cast<uint32_t>(i);
            entries[i].next = buckets[bucket_idx];
            
            buckets[bucket_idx] = static_cast<uint32_t>(i);
        }
        
        entry_count = count;
    }
    
    void build_i64(const int64_t* keys, size_t count) {
        is_i32 = false;
        entries.resize(count);
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t hash = hash_one_i64(keys[i]);
            uint32_t bucket_idx = static_cast<uint32_t>(hash & (num_buckets - 1));
            
            entries[i].key = keys[i];
            entries[i].row_idx = static_cast<uint32_t>(i);
            entries[i].next = buckets[bucket_idx];
            
            buckets[bucket_idx] = static_cast<uint32_t>(i);
        }
        
        entry_count = count;
    }
    
    size_t probe_i32(const int32_t* probe_keys, size_t probe_count,
                     uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
        size_t match_count = 0;
        
        for (size_t i = 0; i < probe_count; ++i) {
            int32_t probe_key = probe_keys[i];
            uint32_t hash = hash_one_i32(probe_key);
            uint32_t bucket_idx = hash & (num_buckets - 1);
            
            uint32_t entry_idx = buckets[bucket_idx];
            while (entry_idx != EMPTY) {
                const Entry& e = entries[entry_idx];
                if (static_cast<int32_t>(e.key) == probe_key) {
                    out_build_indices[match_count] = e.row_idx;
                    out_probe_indices[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
                entry_idx = e.next;
            }
        }
        
        return match_count;
    }
    
    size_t probe_i64(const int64_t* probe_keys, size_t probe_count,
                     uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
        size_t match_count = 0;
        
        for (size_t i = 0; i < probe_count; ++i) {
            int64_t probe_key = probe_keys[i];
            uint64_t hash = hash_one_i64(probe_key);
            uint32_t bucket_idx = static_cast<uint32_t>(hash & (num_buckets - 1));
            
            uint32_t entry_idx = buckets[bucket_idx];
            while (entry_idx != EMPTY) {
                const Entry& e = entries[entry_idx];
                if (e.key == probe_key) {
                    out_build_indices[match_count] = e.row_idx;
                    out_probe_indices[match_count] = static_cast<uint32_t>(i);
                    ++match_count;
                }
                entry_idx = e.next;
            }
        }
        
        return match_count;
    }
};

HashTable::HashTable(size_t expected_size) : impl_(new Impl(expected_size)) {}

HashTable::~HashTable() { delete impl_; }

HashTable::HashTable(HashTable&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

HashTable& HashTable::operator=(HashTable&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void HashTable::build_i32(const int32_t* keys, size_t count) {
    impl_->build_i32(keys, count);
}

void HashTable::build_i64(const int64_t* keys, size_t count) {
    impl_->build_i64(keys, count);
}

size_t HashTable::probe_i32(const int32_t* probe_keys, size_t probe_count,
                            uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
    return impl_->probe_i32(probe_keys, probe_count, out_build_indices, out_probe_indices);
}

size_t HashTable::probe_i64(const int64_t* probe_keys, size_t probe_count,
                            uint32_t* out_build_indices, uint32_t* out_probe_indices) const {
    return impl_->probe_i64(probe_keys, probe_count, out_build_indices, out_probe_indices);
}

size_t HashTable::size() const { return impl_->entry_count; }
size_t HashTable::bucket_count() const { return impl_->num_buckets; }
float HashTable::load_factor() const { 
    return static_cast<float>(impl_->entry_count) / impl_->num_buckets; 
}
void HashTable::clear() { impl_->clear(); }

// ============================================================================
// JoinResult 管理
// ============================================================================

JoinResult* create_join_result(size_t initial_capacity) {
    JoinResult* result = new JoinResult;
    result->left_indices = static_cast<uint32_t*>(
        aligned_alloc(initial_capacity * sizeof(uint32_t), CACHE_LINE_SIZE));
    result->right_indices = static_cast<uint32_t*>(
        aligned_alloc(initial_capacity * sizeof(uint32_t), CACHE_LINE_SIZE));
    result->count = 0;
    result->capacity = initial_capacity;
    return result;
}

void free_join_result(JoinResult* result) {
    if (result) {
        aligned_free(result->left_indices);
        aligned_free(result->right_indices);
        delete result;
    }
}

void grow_join_result(JoinResult* result, size_t min_capacity) {
    if (result->capacity >= min_capacity) return;
    
    size_t new_capacity = result->capacity * 2;
    while (new_capacity < min_capacity) {
        new_capacity *= 2;
    }
    
    uint32_t* new_left = static_cast<uint32_t*>(
        aligned_alloc(new_capacity * sizeof(uint32_t), CACHE_LINE_SIZE));
    uint32_t* new_right = static_cast<uint32_t*>(
        aligned_alloc(new_capacity * sizeof(uint32_t), CACHE_LINE_SIZE));
    
    std::memcpy(new_left, result->left_indices, result->count * sizeof(uint32_t));
    std::memcpy(new_right, result->right_indices, result->count * sizeof(uint32_t));
    
    aligned_free(result->left_indices);
    aligned_free(result->right_indices);
    
    result->left_indices = new_left;
    result->right_indices = new_right;
    result->capacity = new_capacity;
}

// ============================================================================
// Hash Join 实现
// ============================================================================

size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type,
                     JoinResult* result) {
    // 构建哈希表
    HashTable ht(build_count);
    ht.build_i32(build_keys, build_count);
    
    // 分配临时空间
    size_t max_matches = build_count * probe_count;  // 最坏情况
    if (max_matches > 10000000) max_matches = 10000000;  // 限制初始分配
    
    if (result->capacity < max_matches) {
        grow_join_result(result, max_matches);
    }
    
    // 探测
    size_t match_count = ht.probe_i32(probe_keys, probe_count,
                                       result->left_indices, result->right_indices);
    
    result->count = match_count;
    
    // 处理不同连接类型
    if (join_type == JoinType::LEFT || join_type == JoinType::FULL) {
        // TODO: 添加左表未匹配行
    }
    if (join_type == JoinType::RIGHT || join_type == JoinType::FULL) {
        // TODO: 添加右表未匹配行
    }
    
    return match_count;
}

size_t hash_join_i64(const int64_t* build_keys, size_t build_count,
                     const int64_t* probe_keys, size_t probe_count,
                     JoinType join_type,
                     JoinResult* result) {
    HashTable ht(build_count);
    ht.build_i64(build_keys, build_count);
    
    size_t max_matches = std::min(build_count * probe_count, size_t(10000000));
    
    if (result->capacity < max_matches) {
        grow_join_result(result, max_matches);
    }
    
    size_t match_count = ht.probe_i64(probe_keys, probe_count,
                                       result->left_indices, result->right_indices);
    
    result->count = match_count;
    
    return match_count;
}

} // namespace join
} // namespace thunderduck
