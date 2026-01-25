/**
 * ThunderDuck - Aggregate GPU Shaders
 *
 * GPU 聚合内核 (SUM, MIN, MAX, COUNT)
 * 利用 UMA 实现零拷贝
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 常量
// ============================================================================

constant uint THREADGROUP_SIZE = 256;
constant uint WARP_SIZE = 32;

// ============================================================================
// 并行归约 - SUM (int32 -> int64)
// ============================================================================

/**
 * 第一阶段: 每个 threadgroup 计算部分和
 *
 * 使用树形归约，结果存储在 block_sums
 */
kernel void reduce_sum_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device int64_t* block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int64_t* shared [[threadgroup(0)]]
) {
    // 每线程加载多个元素
    int64_t sum = 0;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        sum += input[i];
    }

    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 树形归约
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写入块结果
    if (lid == 0) {
        block_sums[tgid] = shared[0];
    }
}

/**
 * 第二阶段: 归约块结果
 */
kernel void reduce_sum_i64_final(
    device int64_t* block_sums [[buffer(0)]],
    device int64_t* result [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup int64_t* shared [[threadgroup(0)]]
) {
    int64_t sum = 0;
    for (uint i = lid; i < num_blocks; i += THREADGROUP_SIZE) {
        sum += block_sums[i];
    }

    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        result[0] = shared[0];
    }
}

// ============================================================================
// 并行归约 - SUM (float -> double)
// ============================================================================

kernel void reduce_sum_f32_phase1(
    device const float* input [[buffer(0)]],
    device float* block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float sum = 0.0f;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        sum += input[i];
    }

    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_sums[tgid] = shared[0];
    }
}

// ============================================================================
// 并行归约 - MIN/MAX
// ============================================================================

kernel void reduce_min_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device int32_t* block_mins [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int32_t* shared [[threadgroup(0)]]
) {
    int32_t min_val = INT_MAX;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        min_val = min(min_val, input[i]);
    }

    shared[lid] = min_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = min(shared[lid], shared[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_mins[tgid] = shared[0];
    }
}

kernel void reduce_max_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device int32_t* block_maxs [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int32_t* shared [[threadgroup(0)]]
) {
    int32_t max_val = INT_MIN;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        max_val = max(max_val, input[i]);
    }

    shared[lid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] = max(shared[lid], shared[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_maxs[tgid] = shared[0];
    }
}

// ============================================================================
// 合并 MIN/MAX (单次遍历)
// ============================================================================

struct MinMaxResult {
    int32_t min_val;
    int32_t max_val;
};

kernel void reduce_minmax_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device MinMaxResult* block_results [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int32_t* shared_min [[threadgroup(0)]],
    threadgroup int32_t* shared_max [[threadgroup(1)]]
) {
    int32_t min_val = INT_MAX;
    int32_t max_val = INT_MIN;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        int32_t val = input[i];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }

    shared_min[lid] = min_val;
    shared_max[lid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_results[tgid].min_val = shared_min[0];
        block_results[tgid].max_val = shared_max[0];
    }
}

// ============================================================================
// 带选择向量的聚合
// ============================================================================

kernel void reduce_sum_i32_sel(
    device const int32_t* input [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device int64_t* block_sums [[buffer(2)]],
    constant uint& sel_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int64_t* shared [[threadgroup(0)]]
) {
    int64_t sum = 0;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < sel_count; i += stride) {
        sum += input[selection[i]];
    }

    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_sums[tgid] = shared[0];
    }
}

// ============================================================================
// 分组聚合
// ============================================================================

/**
 * 分组求和 - 原子版本
 *
 * 适合分组数较少的场景
 */
kernel void group_sum_i32_atomic(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* sums [[buffer(2)]],    // 使用 atomic_int 累加
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint group = groups[gid];
    int32_t val = values[gid];

    atomic_fetch_add_explicit(&sums[group], val, memory_order_relaxed);
}

/**
 * 分组计数 - 原子版本
 */
kernel void group_count_atomic(
    device const uint* groups [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint group = groups[gid];
    atomic_fetch_add_explicit(&counts[group], 1u, memory_order_relaxed);
}

/**
 * 分组 MIN - 原子版本
 */
kernel void group_min_i32_atomic(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* mins [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint group = groups[gid];
    int32_t val = values[gid];

    atomic_fetch_min_explicit(&mins[group], val, memory_order_relaxed);
}

/**
 * 分组 MAX - 原子版本
 */
kernel void group_max_i32_atomic(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* maxs [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint group = groups[gid];
    int32_t val = values[gid];

    atomic_fetch_max_explicit(&maxs[group], val, memory_order_relaxed);
}

// ============================================================================
// 一次遍历计算所有统计量
// ============================================================================

struct AllStats {
    int64_t sum;
    int32_t min_val;
    int32_t max_val;
    uint count;
};

kernel void aggregate_all_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device AllStats* block_stats [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int64_t* shared_sum [[threadgroup(0)]],
    threadgroup int32_t* shared_min [[threadgroup(1)]],
    threadgroup int32_t* shared_max [[threadgroup(2)]]
) {
    int64_t sum = 0;
    int32_t min_val = INT_MAX;
    int32_t max_val = INT_MIN;
    uint stride = tg_size * gridDim.x;

    for (uint i = gid; i < count; i += stride) {
        int32_t val = input[i];
        sum += val;
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }

    shared_sum[lid] = sum;
    shared_min[lid] = min_val;
    shared_max[lid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_stats[tgid].sum = shared_sum[0];
        block_stats[tgid].min_val = shared_min[0];
        block_stats[tgid].max_val = shared_max[0];
        block_stats[tgid].count = min(tg_size, count - tgid * tg_size);
    }
}

/**
 * 第二阶段: 归约块结果获得最终统计量
 */
kernel void aggregate_all_i32_final(
    device AllStats* block_stats [[buffer(0)]],
    device AllStats* result [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup int64_t* shared_sum [[threadgroup(0)]],
    threadgroup int32_t* shared_min [[threadgroup(1)]],
    threadgroup int32_t* shared_max [[threadgroup(2)]]
) {
    int64_t sum = 0;
    int32_t min_val = INT_MAX;
    int32_t max_val = INT_MIN;

    for (uint i = lid; i < num_blocks; i += THREADGROUP_SIZE) {
        sum += block_stats[i].sum;
        min_val = min(min_val, block_stats[i].min_val);
        max_val = max(max_val, block_stats[i].max_val);
    }

    shared_sum[lid] = sum;
    shared_min[lid] = min_val;
    shared_max[lid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = THREADGROUP_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        result[0].sum = shared_sum[0];
        result[0].min_val = shared_min[0];
        result[0].max_val = shared_max[0];
        result[0].count = num_blocks;  // 用于验证
    }
}
