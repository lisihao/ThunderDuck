/**
 * ThunderDuck - V9.2 GPU Group Aggregation
 *
 * 两阶段分组聚合优化:
 * - Phase 1: Threadgroup 本地累加 (共享内存原子)
 * - Phase 2: 全局合并 (设备内存原子)
 *
 * 优势:
 * - 共享内存原子操作比设备内存快 ~10x
 * - 全局原子从 count 降到 num_groups * num_threadgroups
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 常量
// ============================================================================

constant uint THREADGROUP_SIZE = 256;
constant uint MAX_LOCAL_GROUPS = 1024;  // 共享内存中最大分组数

// ============================================================================
// V9.2 两阶段分组求和
// ============================================================================

/**
 * 两阶段分组求和 - 适合分组数 <= MAX_LOCAL_GROUPS
 *
 * 原理:
 * 1. 每个 threadgroup 在共享内存中维护本地累加器
 * 2. 所有线程先累加到本地 (共享内存原子)
 * 3. threadgroup 完成后，合并到全局 (设备内存原子)
 */
kernel void group_sum_i32_twopass(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* global_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& num_groups [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]],
    threadgroup atomic_int* local_sums [[threadgroup(0)]]
) {
    // Phase 1: 初始化本地累加器
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_sums[g], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: 本地累加 (共享内存原子，比设备内存快 ~10x)
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_add_explicit(&local_sums[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并到全局 (每个分组仅一次全局原子)
    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_sums[g], memory_order_relaxed);
        if (local_val != 0) {
            atomic_fetch_add_explicit(&global_sums[g], local_val, memory_order_relaxed);
        }
    }
}

/**
 * 两阶段分组计数
 */
kernel void group_count_twopass(
    device const uint* groups [[buffer(0)]],
    device atomic_uint* global_counts [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]],
    threadgroup atomic_uint* local_counts [[threadgroup(0)]]
) {
    // Phase 1: 初始化
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_counts[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: 本地计数
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        atomic_fetch_add_explicit(&local_counts[group], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并
    for (uint g = lid; g < num_groups; g += tg_size) {
        uint local_val = atomic_load_explicit(&local_counts[g], memory_order_relaxed);
        if (local_val != 0) {
            atomic_fetch_add_explicit(&global_counts[g], local_val, memory_order_relaxed);
        }
    }
}

/**
 * 两阶段分组 MIN
 */
kernel void group_min_i32_twopass(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* global_mins [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& num_groups [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]],
    threadgroup atomic_int* local_mins [[threadgroup(0)]]
) {
    // Phase 1: 初始化为 INT_MAX
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_mins[g], INT_MAX, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: 本地 MIN
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_min_explicit(&local_mins[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并
    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_mins[g], memory_order_relaxed);
        if (local_val != INT_MAX) {
            atomic_fetch_min_explicit(&global_mins[g], local_val, memory_order_relaxed);
        }
    }
}

/**
 * 两阶段分组 MAX
 */
kernel void group_max_i32_twopass(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* global_maxs [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& num_groups [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]],
    threadgroup atomic_int* local_maxs [[threadgroup(0)]]
) {
    // Phase 1: 初始化为 INT_MIN
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_maxs[g], INT_MIN, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: 本地 MAX
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_max_explicit(&local_maxs[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并
    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_maxs[g], memory_order_relaxed);
        if (local_val != INT_MIN) {
            atomic_fetch_max_explicit(&global_maxs[g], local_val, memory_order_relaxed);
        }
    }
}

// ============================================================================
// V9.2 融合分组统计 (SUM + COUNT + MIN + MAX)
// ============================================================================

/**
 * 融合分组统计 - 单次遍历计算所有统计量
 *
 * 结构: [sum0, sum1, ...], [count0, count1, ...], [min0, min1, ...], [max0, max1, ...]
 */
kernel void group_stats_fused_twopass(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* global_sums [[buffer(2)]],
    device atomic_uint* global_counts [[buffer(3)]],
    device atomic_int* global_mins [[buffer(4)]],
    device atomic_int* global_maxs [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant uint& num_groups [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]],
    threadgroup atomic_int* local_sums [[threadgroup(0)]],
    threadgroup atomic_uint* local_counts [[threadgroup(1)]],
    threadgroup atomic_int* local_mins [[threadgroup(2)]],
    threadgroup atomic_int* local_maxs [[threadgroup(3)]]
) {
    // Phase 1: 初始化
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_sums[g], 0, memory_order_relaxed);
        atomic_store_explicit(&local_counts[g], 0u, memory_order_relaxed);
        atomic_store_explicit(&local_mins[g], INT_MAX, memory_order_relaxed);
        atomic_store_explicit(&local_maxs[g], INT_MIN, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: 本地累加
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_add_explicit(&local_sums[group], val, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_counts[group], 1u, memory_order_relaxed);
        atomic_fetch_min_explicit(&local_mins[group], val, memory_order_relaxed);
        atomic_fetch_max_explicit(&local_maxs[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并
    for (uint g = lid; g < num_groups; g += tg_size) {
        int sum_val = atomic_load_explicit(&local_sums[g], memory_order_relaxed);
        uint cnt_val = atomic_load_explicit(&local_counts[g], memory_order_relaxed);
        int min_val = atomic_load_explicit(&local_mins[g], memory_order_relaxed);
        int max_val = atomic_load_explicit(&local_maxs[g], memory_order_relaxed);

        if (cnt_val > 0) {
            atomic_fetch_add_explicit(&global_sums[g], sum_val, memory_order_relaxed);
            atomic_fetch_add_explicit(&global_counts[g], cnt_val, memory_order_relaxed);
            atomic_fetch_min_explicit(&global_mins[g], min_val, memory_order_relaxed);
            atomic_fetch_max_explicit(&global_maxs[g], max_val, memory_order_relaxed);
        }
    }
}
