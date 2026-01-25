/**
 * ThunderDuck - Hash Join Metal Shader
 *
 * GPU 并行 Hash Join 探测
 * - 每线程处理一个 probe key
 * - 使用原子操作收集匹配结果
 * - 统一内存零拷贝
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 常量
// ============================================================================

constant uint32_t EMPTY_KEY = 0x80000000;  // INT32_MIN

// ============================================================================
// 哈希函数
// ============================================================================

// CRC32 模拟 (Metal 没有 CRC32 指令，使用 MurmurHash3 变种)
inline uint32_t hash_key(int32_t key) {
    uint32_t h = as_type<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// ============================================================================
// 哈希表探测 Kernel
// ============================================================================

/**
 * 批量探测哈希表
 *
 * @param probe_keys     探测键数组
 * @param ht_keys        哈希表键数组
 * @param ht_indices     哈希表行索引数组
 * @param ht_mask        哈希表掩码 (capacity - 1)
 * @param out_build      输出: 匹配的 build 索引
 * @param out_probe      输出: 匹配的 probe 索引
 * @param match_counter  原子计数器 (当前匹配数量)
 * @param max_matches    最大匹配数量 (防止溢出)
 * @param probe_count    probe 键数量
 */
kernel void hash_join_probe(
    device const int32_t* probe_keys [[buffer(0)]],
    device const int32_t* ht_keys [[buffer(1)]],
    device const uint32_t* ht_indices [[buffer(2)]],
    constant uint32_t& ht_mask [[buffer(3)]],
    device uint32_t* out_build [[buffer(4)]],
    device uint32_t* out_probe [[buffer(5)]],
    device atomic_uint* match_counter [[buffer(6)]],
    constant uint32_t& max_matches [[buffer(7)]],
    constant uint32_t& probe_count [[buffer(8)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= probe_count) return;

    int32_t key = probe_keys[thread_id];
    uint32_t hash = hash_key(key);
    uint32_t idx = hash & ht_mask;

    // 线性探测
    while (true) {
        int32_t ht_key = ht_keys[idx];

        if (ht_key == as_type<int32_t>(EMPTY_KEY)) {
            // 空槽，没有匹配
            break;
        }

        if (ht_key == key) {
            // 找到匹配
            uint32_t match_idx = atomic_fetch_add_explicit(
                match_counter, 1u, memory_order_relaxed);

            if (match_idx < max_matches) {
                out_build[match_idx] = ht_indices[idx];
                out_probe[match_idx] = thread_id;
            }
        }

        idx = (idx + 1) & ht_mask;
    }
}

// ============================================================================
// Bloom Filter 检查 Kernel (可选优化)
// ============================================================================

/**
 * Bloom Filter 批量检查
 *
 * @param keys           检查的键数组
 * @param bloom_bits     Bloom Filter 位数组 (64-bit words)
 * @param bloom_mask     Bloom Filter 位掩码
 * @param num_hashes     哈希函数数量
 * @param out_pass       输出: 通过检查的标记 (1=通过, 0=失败)
 * @param count          键数量
 */
kernel void bloom_filter_check(
    device const int32_t* keys [[buffer(0)]],
    device const uint64_t* bloom_bits [[buffer(1)]],
    constant uint32_t& bloom_mask [[buffer(2)]],
    constant uint32_t& num_hashes [[buffer(3)]],
    device uint8_t* out_pass [[buffer(4)]],
    constant uint32_t& count [[buffer(5)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= count) return;

    int32_t key = keys[thread_id];

    // 双重哈希
    uint32_t h1 = hash_key(key);
    uint32_t h2 = hash_key(key ^ 0x12345678);

    bool pass = true;

    for (uint32_t i = 0; i < num_hashes && pass; ++i) {
        uint32_t bit_index = (h1 + i * h2) & bloom_mask;
        uint32_t word_idx = bit_index / 64;
        uint32_t bit_pos = bit_index % 64;

        if ((bloom_bits[word_idx] & (1ULL << bit_pos)) == 0) {
            pass = false;
        }
    }

    out_pass[thread_id] = pass ? 1 : 0;
}

// ============================================================================
// Bloom + Hash Join 组合 Kernel
// ============================================================================

/**
 * Bloom 过滤 + Hash Join 探测组合
 * 减少内存带宽，一次 kernel 完成过滤和探测
 */
kernel void bloom_hash_join_probe(
    device const int32_t* probe_keys [[buffer(0)]],
    device const uint64_t* bloom_bits [[buffer(1)]],
    constant uint32_t& bloom_mask [[buffer(2)]],
    constant uint32_t& num_hashes [[buffer(3)]],
    device const int32_t* ht_keys [[buffer(4)]],
    device const uint32_t* ht_indices [[buffer(5)]],
    constant uint32_t& ht_mask [[buffer(6)]],
    device uint32_t* out_build [[buffer(7)]],
    device uint32_t* out_probe [[buffer(8)]],
    device atomic_uint* match_counter [[buffer(9)]],
    constant uint32_t& max_matches [[buffer(10)]],
    constant uint32_t& probe_count [[buffer(11)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= probe_count) return;

    int32_t key = probe_keys[thread_id];

    // 1. Bloom Filter 检查
    uint32_t h1 = hash_key(key);
    uint32_t h2 = hash_key(key ^ 0x12345678);

    bool bloom_pass = true;
    for (uint32_t i = 0; i < num_hashes && bloom_pass; ++i) {
        uint32_t bit_index = (h1 + i * h2) & bloom_mask;
        uint32_t word_idx = bit_index / 64;
        uint32_t bit_pos = bit_index % 64;

        if ((bloom_bits[word_idx] & (1ULL << bit_pos)) == 0) {
            bloom_pass = false;
        }
    }

    if (!bloom_pass) return;

    // 2. Hash Table 探测
    uint32_t idx = h1 & ht_mask;

    while (true) {
        int32_t ht_key = ht_keys[idx];

        if (ht_key == as_type<int32_t>(EMPTY_KEY)) {
            break;
        }

        if (ht_key == key) {
            uint32_t match_idx = atomic_fetch_add_explicit(
                match_counter, 1u, memory_order_relaxed);

            if (match_idx < max_matches) {
                out_build[match_idx] = ht_indices[idx];
                out_probe[match_idx] = thread_id;
            }
        }

        idx = (idx + 1) & ht_mask;
    }
}
