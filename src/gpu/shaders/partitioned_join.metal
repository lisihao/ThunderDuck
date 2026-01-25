/**
 * ThunderDuck - Partitioned GPU Hash Join
 *
 * 基于 SIGMOD'25 GFTR 模式的优化实现:
 * - Radix 分区实现顺序内存访问
 * - Threadgroup memory 缓存热点数据
 * - SIMD prefix sum 批量收集结果
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 常量
// ============================================================================

constant uint RADIX_BITS = 8;
constant uint NUM_PARTITIONS = 256;  // 2^8
constant uint RADIX_MASK = 255;

constant uint THREADGROUP_SIZE = 256;
constant uint MAX_LOCAL_MATCHES = 16;  // 每线程本地缓存

constant int32_t EMPTY_KEY = 0x80000000;  // INT32_MIN

// ============================================================================
// 哈希函数
// ============================================================================

inline uint32_t hash_key(int32_t key) {
    uint32_t h = as_type<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// 提取 radix 位 (用于分区)
inline uint32_t get_radix(int32_t key) {
    return hash_key(key) & RADIX_MASK;
}

// ============================================================================
// Kernel 1: Radix Histogram
// 计算每个分区的元素数量
// ============================================================================

kernel void radix_histogram(
    device const int32_t* keys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],  // 256 个桶
    constant uint& count [[buffer(2)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= count) return;

    int32_t key = keys[thread_id];
    uint32_t radix = get_radix(key);

    atomic_fetch_add_explicit(&histogram[radix], 1u, memory_order_relaxed);
}

// ============================================================================
// Kernel 2: Radix Scatter
// 根据 radix 分区重新排列数据
// ============================================================================

kernel void radix_scatter(
    device const int32_t* keys [[buffer(0)]],
    device const uint32_t* indices [[buffer(1)]],
    device const uint* prefix_sum [[buffer(2)]],  // 每个分区的起始偏移
    device atomic_uint* partition_counters [[buffer(3)]],  // 每个分区的当前偏移
    device int32_t* out_keys [[buffer(4)]],
    device uint32_t* out_indices [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= count) return;

    int32_t key = keys[thread_id];
    uint32_t idx = indices[thread_id];
    uint32_t radix = get_radix(key);

    // 获取该分区内的写入位置
    uint offset = prefix_sum[radix] + atomic_fetch_add_explicit(
        &partition_counters[radix], 1u, memory_order_relaxed);

    out_keys[offset] = key;
    out_indices[offset] = idx;
}

// ============================================================================
// Kernel 3: Partitioned Hash Join (核心优化)
// 每个 threadgroup 处理一个分区
// ============================================================================

kernel void partitioned_hash_join(
    // Build 侧 (已分区)
    device const int32_t* build_keys [[buffer(0)]],
    device const uint32_t* build_indices [[buffer(1)]],
    // Probe 侧 (已分区)
    device const int32_t* probe_keys [[buffer(2)]],
    device const uint32_t* probe_indices [[buffer(3)]],
    // 分区信息
    device const uint* build_offsets [[buffer(4)]],
    device const uint* build_sizes [[buffer(5)]],
    device const uint* probe_offsets [[buffer(6)]],
    device const uint* probe_sizes [[buffer(7)]],
    // 输出
    device uint32_t* out_build [[buffer(8)]],
    device uint32_t* out_probe [[buffer(9)]],
    device atomic_uint* match_counter [[buffer(10)]],
    constant uint& max_matches [[buffer(11)]],
    // Threadgroup memory (由 host 分配大小)
    threadgroup int32_t* shared_build_keys [[threadgroup(0)]],
    threadgroup uint32_t* shared_build_indices [[threadgroup(1)]],
    // IDs
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // 每个 threadgroup 处理一个分区
    uint partition_id = group_id;
    if (partition_id >= NUM_PARTITIONS) return;

    uint build_offset = build_offsets[partition_id];
    uint build_size = build_sizes[partition_id];
    uint probe_offset = probe_offsets[partition_id];
    uint probe_size = probe_sizes[partition_id];

    // 空分区跳过
    if (build_size == 0 || probe_size == 0) return;

    // ========== 阶段 1: 加载 build 数据到 threadgroup memory ==========
    // 使用协作加载
    for (uint i = local_id; i < build_size; i += threads_per_group) {
        shared_build_keys[i] = build_keys[build_offset + i];
        shared_build_indices[i] = build_indices[build_offset + i];
    }

    // 同步确保所有数据加载完成
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========== 阶段 2: 每个线程处理一部分 probe 数据 ==========
    // 本地结果缓存
    uint32_t local_build[MAX_LOCAL_MATCHES];
    uint32_t local_probe[MAX_LOCAL_MATCHES];
    uint local_count = 0;

    // 每个线程处理多个 probe keys
    for (uint probe_idx = local_id; probe_idx < probe_size; probe_idx += threads_per_group) {
        int32_t probe_key = probe_keys[probe_offset + probe_idx];
        uint32_t probe_original_idx = probe_indices[probe_offset + probe_idx];

        // 在 threadgroup memory 中线性搜索
        // 对于小分区这比哈希更快 (顺序访问 + 缓存)
        for (uint build_idx = 0; build_idx < build_size; build_idx++) {
            if (shared_build_keys[build_idx] == probe_key) {
                // 找到匹配
                if (local_count < MAX_LOCAL_MATCHES) {
                    local_build[local_count] = shared_build_indices[build_idx];
                    local_probe[local_count] = probe_original_idx;
                    local_count++;
                } else {
                    // 本地缓存满，flush 到全局
                    uint global_offset = atomic_fetch_add_explicit(
                        match_counter, local_count, memory_order_relaxed);

                    for (uint i = 0; i < local_count && global_offset + i < max_matches; i++) {
                        out_build[global_offset + i] = local_build[i];
                        out_probe[global_offset + i] = local_probe[i];
                    }
                    local_count = 0;

                    // 存储当前匹配
                    local_build[local_count] = shared_build_indices[build_idx];
                    local_probe[local_count] = probe_original_idx;
                    local_count++;
                }
            }
        }
    }

    // ========== 阶段 3: SIMD 批量写入剩余结果 ==========
    // 使用 SIMD prefix sum 减少原子操作
    uint simd_total = simd_sum(local_count);

    if (simd_total > 0) {
        uint simd_offset = simd_prefix_exclusive_sum(local_count);

        // 只有 simd_lane 0 执行原子操作
        uint global_offset;
        if (simd_is_first()) {
            global_offset = atomic_fetch_add_explicit(
                match_counter, simd_total, memory_order_relaxed);
        }
        // 广播 global_offset 到 simd 组内所有线程
        global_offset = simd_broadcast_first(global_offset);

        // 每个线程写入自己的结果
        for (uint i = 0; i < local_count; i++) {
            uint write_idx = global_offset + simd_offset + i;
            if (write_idx < max_matches) {
                out_build[write_idx] = local_build[i];
                out_probe[write_idx] = local_probe[i];
            }
        }
    }
}

// ============================================================================
// Kernel 4: 简化版本 (小数据量使用)
// 不使用分区，直接在 GPU 上构建哈希表
// ============================================================================

kernel void simple_hash_join(
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

    // 本地缓存 (减少原子操作)
    uint32_t local_build[4];
    uint32_t local_probe[4];
    uint local_count = 0;

    // 线性探测
    for (uint attempts = 0; attempts < 128; attempts++) {
        int32_t ht_key = ht_keys[idx];

        if (ht_key == EMPTY_KEY) {
            break;
        }

        if (ht_key == key) {
            if (local_count < 4) {
                local_build[local_count] = ht_indices[idx];
                local_probe[local_count] = thread_id;
                local_count++;
            }
        }

        idx = (idx + 1) & ht_mask;
    }

    // SIMD 批量写入
    if (local_count > 0) {
        uint simd_total = simd_sum(local_count);
        uint simd_offset = simd_prefix_exclusive_sum(local_count);

        uint global_offset;
        if (simd_is_first()) {
            global_offset = atomic_fetch_add_explicit(
                match_counter, simd_total, memory_order_relaxed);
        }
        global_offset = simd_broadcast_first(global_offset);

        for (uint i = 0; i < local_count; i++) {
            uint write_idx = global_offset + simd_offset + i;
            if (write_idx < max_matches) {
                out_build[write_idx] = local_build[i];
                out_probe[write_idx] = local_probe[i];
            }
        }
    }
}

// ============================================================================
// Kernel 5: 生成原始索引数组
// 用于分区前准备
// ============================================================================

kernel void generate_indices(
    device uint32_t* indices [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= count) return;
    indices[thread_id] = thread_id;
}
