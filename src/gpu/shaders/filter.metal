/**
 * ThunderDuck - Filter GPU Shaders
 *
 * GPU 过滤内核，利用 UMA 实现零拷贝
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 比较操作枚举
// ============================================================================

enum CompareOp : uint {
    EQ = 0,  // ==
    NE = 1,  // !=
    LT = 2,  // <
    LE = 3,  // <=
    GT = 4,  // >
    GE = 5   // >=
};

// ============================================================================
// 过滤参数
// ============================================================================

struct FilterParams {
    uint count;         // 元素数量
    uint op;            // 比较操作
    int value;          // 比较值 (int32)
    uint pad;           // 填充对齐
};

struct FilterParamsF32 {
    uint count;         // 元素数量
    uint op;            // 比较操作
    float value;        // 比较值 (float)
    uint pad;           // 填充对齐
};

// ============================================================================
// 位图过滤内核 - 生成位图
// ============================================================================

/**
 * 生成过滤位图 (int32)
 *
 * 每个线程处理 64 个元素，生成 1 个 64-bit mask
 * 使用 threadgroup 内存累加小批量结果
 */
kernel void filter_to_bitmap_i32(
    device const int32_t* input [[buffer(0)]],
    device atomic_uint* bitmap [[buffer(1)]],      // 使用 atomic_uint 组成 64-bit
    constant FilterParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint base_idx = gid * 32;  // 每线程处理 32 个元素
    if (base_idx >= params.count) return;

    uint mask = 0;
    uint end_idx = min(base_idx + 32, params.count);

    int cmp_value = params.value;
    CompareOp op = static_cast<CompareOp>(params.op);

    for (uint i = base_idx; i < end_idx; i++) {
        int val = input[i];
        bool match = false;

        switch (op) {
            case EQ: match = (val == cmp_value); break;
            case NE: match = (val != cmp_value); break;
            case LT: match = (val < cmp_value); break;
            case LE: match = (val <= cmp_value); break;
            case GT: match = (val > cmp_value); break;
            case GE: match = (val >= cmp_value); break;
        }

        if (match) {
            mask |= (1u << (i - base_idx));
        }
    }

    // 原子写入位图
    if (mask != 0) {
        atomic_fetch_or_explicit(&bitmap[gid], mask, memory_order_relaxed);
    }
}

// ============================================================================
// 直接索引过滤内核 - 输出匹配索引
// ============================================================================

/**
 * 过滤并输出匹配索引 (int32)
 *
 * 两阶段方法:
 * Phase 1: 计算每个线程的匹配数 (prefix sum 准备)
 * Phase 2: 写入索引
 */
kernel void filter_count_i32(
    device const int32_t* input [[buffer(0)]],
    device uint* counts [[buffer(1)]],           // 每线程的匹配数
    constant FilterParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 64;  // 每线程处理 64 个元素
    if (base_idx >= params.count) {
        counts[gid] = 0;
        return;
    }

    uint count = 0;
    uint end_idx = min(base_idx + 64, params.count);

    int cmp_value = params.value;
    CompareOp op = static_cast<CompareOp>(params.op);

    for (uint i = base_idx; i < end_idx; i++) {
        int val = input[i];
        bool match = false;

        switch (op) {
            case EQ: match = (val == cmp_value); break;
            case NE: match = (val != cmp_value); break;
            case LT: match = (val < cmp_value); break;
            case LE: match = (val <= cmp_value); break;
            case GT: match = (val > cmp_value); break;
            case GE: match = (val >= cmp_value); break;
        }

        if (match) count++;
    }

    counts[gid] = count;
}

/**
 * 前缀和计算 (Blelloch scan)
 */
kernel void prefix_sum(
    device uint* data [[buffer(0)]],
    device uint* block_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    // Load into shared memory
    if (gid < n) {
        shared[lid] = data[gid];
    } else {
        shared[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < tg_size) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block sum and clear last element
    if (lid == tg_size - 1) {
        if (block_sums) {
            block_sums[tgid] = shared[lid];
        }
        shared[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = tg_size / 2; stride >= 1; stride /= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < tg_size) {
            uint t = shared[idx - stride];
            shared[idx - stride] = shared[idx];
            shared[idx] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (gid < n) {
        data[gid] = shared[lid];
    }
}

/**
 * 写入过滤索引
 */
kernel void filter_write_indices_i32(
    device const int32_t* input [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],     // 前缀和结果
    device uint* out_indices [[buffer(2)]],
    constant FilterParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 64;
    if (base_idx >= params.count) return;

    uint offset = offsets[gid];
    uint end_idx = min(base_idx + 64, params.count);

    int cmp_value = params.value;
    CompareOp op = static_cast<CompareOp>(params.op);

    for (uint i = base_idx; i < end_idx; i++) {
        int val = input[i];
        bool match = false;

        switch (op) {
            case EQ: match = (val == cmp_value); break;
            case NE: match = (val != cmp_value); break;
            case LT: match = (val < cmp_value); break;
            case LE: match = (val <= cmp_value); break;
            case GT: match = (val > cmp_value); break;
            case GE: match = (val >= cmp_value); break;
        }

        if (match) {
            out_indices[offset++] = i;
        }
    }
}

// ============================================================================
// 原子版本 - 单 pass，适合小选择率
// ============================================================================

/**
 * 原子版过滤 - 单 pass
 *
 * 适合选择率 < 10% 的场景
 */
kernel void filter_atomic_i32(
    device const int32_t* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant FilterParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;

    int val = input[gid];
    int cmp_value = params.value;
    CompareOp op = static_cast<CompareOp>(params.op);

    bool match = false;
    switch (op) {
        case EQ: match = (val == cmp_value); break;
        case NE: match = (val != cmp_value); break;
        case LT: match = (val < cmp_value); break;
        case LE: match = (val <= cmp_value); break;
        case GT: match = (val > cmp_value); break;
        case GE: match = (val >= cmp_value); break;
    }

    if (match) {
        uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        out_indices[idx] = gid;
    }
}

// ============================================================================
// 向量化过滤 - SIMD within GPU
// ============================================================================

/**
 * 向量化过滤 (4 元素/线程)
 *
 * 利用 GPU 的 SIMD 单元处理 4 个元素
 */
kernel void filter_simd4_i32(
    device const int4* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant FilterParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 4;
    if (base_idx >= params.count) return;

    int4 vals = input[gid];
    int4 cmp = int4(params.value);
    bool4 mask;

    CompareOp op = static_cast<CompareOp>(params.op);
    switch (op) {
        case EQ: mask = (vals == cmp); break;
        case NE: mask = (vals != cmp); break;
        case LT: mask = (vals < cmp); break;
        case LE: mask = (vals <= cmp); break;
        case GT: mask = (vals > cmp); break;
        case GE: mask = (vals >= cmp); break;
    }

    // 处理边界
    uint remaining = min(4u, params.count - base_idx);

    // 写入匹配的索引
    for (uint i = 0; i < remaining; i++) {
        if (mask[i]) {
            uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            out_indices[idx] = base_idx + i;
        }
    }
}

// ============================================================================
// 范围过滤
// ============================================================================

struct RangeParams {
    uint count;
    int low;
    int high;
    uint pad;
};

kernel void filter_range_i32(
    device const int32_t* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant RangeParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;

    int val = input[gid];

    if (val >= params.low && val < params.high) {
        uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        out_indices[idx] = gid;
    }
}

// ============================================================================
// Float 过滤
// ============================================================================

kernel void filter_atomic_f32(
    device const float* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant FilterParamsF32& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;

    float val = input[gid];
    float cmp_value = params.value;
    CompareOp op = static_cast<CompareOp>(params.op);

    bool match = false;
    switch (op) {
        case EQ: match = (val == cmp_value); break;
        case NE: match = (val != cmp_value); break;
        case LT: match = (val < cmp_value); break;
        case LE: match = (val <= cmp_value); break;
        case GT: match = (val > cmp_value); break;
        case GE: match = (val >= cmp_value); break;
    }

    if (match) {
        uint idx = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        out_indices[idx] = gid;
    }
}

// ============================================================================
// 选择向量应用
// ============================================================================

/**
 * 应用选择向量到数据列
 *
 * 从 input 中根据 selection 提取数据到 output
 */
kernel void apply_selection_i32(
    device const int32_t* input [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device int32_t* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = input[selection[gid]];
}

kernel void apply_selection_i64(
    device const int64_t* input [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device int64_t* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = input[selection[gid]];
}

kernel void apply_selection_f32(
    device const float* input [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = input[selection[gid]];
}

kernel void apply_selection_f64(
    device const double* input [[buffer(0)]],
    device const uint* selection [[buffer(1)]],
    device double* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = input[selection[gid]];
}
