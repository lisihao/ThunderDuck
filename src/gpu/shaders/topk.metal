/**
 * ThunderDuck - TopK GPU Shaders
 *
 * GPU TopK 内核:
 * - 并行过滤 (小 K)
 * - Bitonic Sort (大 K)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 常量
// ============================================================================

constant uint THREADGROUP_SIZE = 256;

// ============================================================================
// TopK 参数
// ============================================================================

struct TopKParams {
    uint count;         // 元素数量
    uint k;             // K 值
    int threshold;      // 阈值 (用于过滤)
    uint is_max;        // 1 = max, 0 = min
};

// ============================================================================
// 采样估计阈值
// ============================================================================

/**
 * 采样并估计第 K 大/小值
 *
 * 策略: 随机采样 sqrt(N) 个元素，找第 K/factor 个
 */
kernel void sample_threshold(
    device const int32_t* input [[buffer(0)]],
    device int32_t* samples [[buffer(1)]],
    constant TopKParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // 每个 threadgroup 采样一个元素
    uint sample_stride = params.count / THREADGROUP_SIZE;
    if (sample_stride == 0) sample_stride = 1;

    uint sample_idx = tgid * sample_stride;
    if (sample_idx < params.count) {
        samples[tgid] = input[sample_idx];
    }
}

// ============================================================================
// 并行过滤 - 计算每线程候选数
// ============================================================================

kernel void topk_filter_count(
    device const int32_t* input [[buffer(0)]],
    device uint* counts [[buffer(1)]],          // 每线程候选数
    constant TopKParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 64;  // 每线程处理 64 个元素
    if (base_idx >= params.count) {
        counts[gid] = 0;
        return;
    }

    uint count = 0;
    uint end_idx = min(base_idx + 64, params.count);
    int threshold = params.threshold;
    bool is_max = params.is_max != 0;

    for (uint i = base_idx; i < end_idx; i++) {
        int val = input[i];
        bool match = is_max ? (val >= threshold) : (val <= threshold);
        if (match) count++;
    }

    counts[gid] = count;
}

/**
 * 写入候选元素
 */
kernel void topk_filter_write(
    device const int32_t* input [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],     // 前缀和
    device int32_t* candidates [[buffer(2)]],      // 候选值
    device uint* candidate_indices [[buffer(3)]], // 候选索引
    constant TopKParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 64;
    if (base_idx >= params.count) return;

    uint offset = offsets[gid];
    uint end_idx = min(base_idx + 64, params.count);
    int threshold = params.threshold;
    bool is_max = params.is_max != 0;

    for (uint i = base_idx; i < end_idx; i++) {
        int val = input[i];
        bool match = is_max ? (val >= threshold) : (val <= threshold);
        if (match) {
            candidates[offset] = val;
            candidate_indices[offset] = i;
            offset++;
        }
    }
}

// ============================================================================
// 原子版 TopK - 适合小 K
// ============================================================================

/**
 * 原子版 TopK
 *
 * 使用全局最小堆，每线程尝试插入
 */
kernel void topk_atomic(
    device const int32_t* input [[buffer(0)]],
    device int32_t* heap [[buffer(1)]],           // 大小为 K 的堆
    device uint* heap_indices [[buffer(2)]],      // 对应索引
    device atomic_int* heap_min [[buffer(3)]],    // 堆的当前最小值
    constant TopKParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;

    int val = input[gid];
    int current_min = atomic_load_explicit(heap_min, memory_order_relaxed);
    bool is_max = params.is_max != 0;

    // 快速跳过
    if (is_max && val <= current_min) return;
    if (!is_max && val >= current_min) return;

    // 尝试更新堆最小值 (简化版，真正的堆更新需要更复杂的逻辑)
    atomic_compare_exchange_weak_explicit(heap_min, &current_min, val,
                                          memory_order_relaxed, memory_order_relaxed);
}

// ============================================================================
// Bitonic Sort 相关
// ============================================================================

/**
 * Bitonic 比较交换
 */
inline void compare_and_swap(
    device int32_t* data,
    device uint* indices,
    uint i, uint j,
    bool ascending
) {
    if ((data[i] > data[j]) == ascending) {
        // 交换值
        int32_t temp_val = data[i];
        data[i] = data[j];
        data[j] = temp_val;

        // 交换索引
        uint temp_idx = indices[i];
        indices[i] = indices[j];
        indices[j] = temp_idx;
    }
}

/**
 * Bitonic Sort 单步
 */
kernel void bitonic_sort_step(
    device int32_t* data [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& step [[buffer(2)]],
    constant uint& stage [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& is_ascending [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_distance = 1u << step;
    uint block_size = 1u << (stage + 1);

    uint partner;
    bool direction;

    uint block_id = gid / pair_distance;
    uint offset = gid % pair_distance;

    // 确定配对
    uint left_idx = block_id * pair_distance * 2 + offset;
    partner = left_idx + pair_distance;

    if (partner >= count) return;

    // 确定方向
    uint block_start = (left_idx / block_size) * block_size;
    direction = ((block_start / (block_size / 2)) % 2 == 0) == (is_ascending != 0);

    compare_and_swap(data, indices, left_idx, partner, direction);
}

/**
 * Threadgroup 内 Bitonic Sort
 *
 * 适合小数组 (< 1024 元素)
 */
kernel void bitonic_sort_local(
    device int32_t* data [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& is_ascending [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int32_t* shared_data [[threadgroup(0)]],
    threadgroup uint* shared_indices [[threadgroup(1)]]
) {
    // 加载到共享内存
    if (lid < count) {
        shared_data[lid] = data[lid];
        shared_indices[lid] = indices[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic Sort
    bool ascending = is_ascending != 0;
    uint n = count;

    for (uint stage = 0; (1u << stage) < n; stage++) {
        for (int step = int(stage); step >= 0; step--) {
            uint pair_distance = 1u << step;
            uint block_size = 1u << (stage + 1);

            if (lid < n / 2) {
                uint block_id = lid / pair_distance;
                uint offset = lid % pair_distance;
                uint left_idx = block_id * pair_distance * 2 + offset;
                uint right_idx = left_idx + pair_distance;

                if (right_idx < n) {
                    uint block_start = (left_idx / block_size) * block_size;
                    bool direction = ((block_start / (block_size / 2)) % 2 == 0) == ascending;

                    if ((shared_data[left_idx] > shared_data[right_idx]) == direction) {
                        int32_t temp_val = shared_data[left_idx];
                        shared_data[left_idx] = shared_data[right_idx];
                        shared_data[right_idx] = temp_val;

                        uint temp_idx = shared_indices[left_idx];
                        shared_indices[left_idx] = shared_indices[right_idx];
                        shared_indices[right_idx] = temp_idx;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 写回
    if (lid < count) {
        data[lid] = shared_data[lid];
        indices[lid] = shared_indices[lid];
    }
}

// ============================================================================
// 初始化索引
// ============================================================================

kernel void init_indices(
    device uint* indices [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        indices[gid] = gid;
    }
}

// ============================================================================
// 复制 TopK 结果
// ============================================================================

kernel void copy_topk_result(
    device const int32_t* sorted_data [[buffer(0)]],
    device const uint* sorted_indices [[buffer(1)]],
    device int32_t* out_values [[buffer(2)]],
    device uint* out_indices [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& is_max [[buffer(5)]],
    constant uint& total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= k) return;

    uint src_idx;
    if (is_max != 0) {
        // Max: 取末尾 K 个 (降序)
        src_idx = total - 1 - gid;
    } else {
        // Min: 取开头 K 个 (升序)
        src_idx = gid;
    }

    out_values[gid] = sorted_data[src_idx];
    if (out_indices) {
        out_indices[gid] = sorted_indices[src_idx];
    }
}
