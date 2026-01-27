/**
 * ThunderDuck V13 - GPU TopK 并行版本
 *
 * 算法: 分层选择 + Bitonic Sort
 *
 * Phase 1: 每个 threadgroup 找本地 TopK (并行)
 * Phase 2: 合并所有 threadgroup 的 TopK
 * Phase 3: Bitonic Sort 最终结果
 *
 * 目标: CPU 4x → GPU 5x+
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "thunderduck/sort.h"
#include <vector>
#include <algorithm>

namespace thunderduck {
namespace sort {

// ============================================================================
// Metal Shader 源码
// ============================================================================

static const char* kTopKV13Shader = R"(
#include <metal_stdlib>
using namespace metal;

// Metal 常量定义
constant int32_t METAL_INT32_MIN = -2147483648;
constant uint32_t METAL_UINT32_MAX = 4294967295u;

// 每个 threadgroup 的本地 TopK 堆
struct HeapElement {
    int32_t value;
    uint32_t index;
};

// Phase 1: 每个 threadgroup 找本地 TopK
kernel void topk_local_max(
    device const int32_t* data [[buffer(0)]],
    device HeapElement* local_topk [[buffer(1)]],  // [num_threadgroups][k]
    constant uint32_t& count [[buffer(2)]],
    constant uint32_t& k [[buffer(3)]],
    constant uint32_t& elements_per_tg [[buffer(4)]],
    threadgroup HeapElement* heap [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // 计算本 threadgroup 处理的数据范围
    uint start = tg_id * elements_per_tg;
    uint end = min(start + elements_per_tg, count);

    // 初始化堆 (最小堆，堆顶是最小值)
    if (tid < k) {
        heap[tid].value = METAL_INT32_MIN;
        heap[tid].index = METAL_UINT32_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个线程处理一部分数据
    for (uint i = start + tid; i < end; i += tg_size) {
        int32_t val = data[i];

        // 如果比堆顶大，尝试替换
        if (val > heap[0].value) {
            // 原子比较交换 (简化: 使用锁)
            // 实际实现需要更精细的同步
            if (tid == 0) {
                // 单线程更新堆
                for (uint j = start; j < end; j++) {
                    int32_t v = data[j];
                    if (v > heap[0].value) {
                        // 替换堆顶
                        heap[0].value = v;
                        heap[0].index = j;
                        // 下沉
                        uint pos = 0;
                        while (true) {
                            uint left = 2 * pos + 1;
                            uint right = 2 * pos + 2;
                            uint smallest = pos;
                            if (left < k && heap[left].value < heap[smallest].value) {
                                smallest = left;
                            }
                            if (right < k && heap[right].value < heap[smallest].value) {
                                smallest = right;
                            }
                            if (smallest == pos) break;
                            HeapElement tmp = heap[pos];
                            heap[pos] = heap[smallest];
                            heap[smallest] = tmp;
                            pos = smallest;
                        }
                    }
                }
            }
            break;  // 只处理一次
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 写回 local_topk
    if (tid < k) {
        uint offset = tg_id * k;
        local_topk[offset + tid] = heap[tid];
    }
}

// Phase 2: Bitonic Sort 比较交换
kernel void bitonic_compare_swap(
    device HeapElement* data [[buffer(0)]],
    constant uint32_t& stage [[buffer(1)]],
    constant uint32_t& step [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    constant bool& ascending [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count / 2) return;

    uint pair_distance = 1 << step;
    uint block_size = 2 << stage;

    uint left_idx = (tid / pair_distance) * (pair_distance * 2) + (tid % pair_distance);
    uint right_idx = left_idx + pair_distance;

    if (right_idx >= count) return;

    // 判断排序方向
    bool should_swap;
    uint block_idx = left_idx / block_size;
    bool block_ascending = (block_idx % 2 == 0) ? ascending : !ascending;

    if (block_ascending) {
        should_swap = data[left_idx].value < data[right_idx].value;  // Max: 大的在前
    } else {
        should_swap = data[left_idx].value > data[right_idx].value;
    }

    if (should_swap) {
        HeapElement tmp = data[left_idx];
        data[left_idx] = data[right_idx];
        data[right_idx] = tmp;
    }
}

// 简化版: 单 threadgroup 本地 TopK
kernel void topk_simple(
    device const int32_t* data [[buffer(0)]],
    device int32_t* out_values [[buffer(1)]],
    device uint32_t* out_indices [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    threadgroup int32_t* local_vals [[threadgroup(0)]],
    threadgroup uint32_t* local_idx [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // 初始化本地缓冲区
    if (tid < k) {
        local_vals[tid] = METAL_INT32_MIN;
        local_idx[tid] = METAL_UINT32_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 每个线程扫描一部分数据
    for (uint i = tid; i < count; i += tg_size) {
        int32_t val = data[i];

        // 检查是否应该加入 TopK
        if (val > local_vals[k - 1]) {
            // 找到插入位置 (线性搜索，因为 K 很小)
            for (int j = k - 1; j >= 0; j--) {
                if (j == 0 || val <= local_vals[j - 1]) {
                    // 插入位置 j，移动后面的元素
                    for (int m = k - 1; m > j; m--) {
                        local_vals[m] = local_vals[m - 1];
                        local_idx[m] = local_idx[m - 1];
                    }
                    local_vals[j] = val;
                    local_idx[j] = i;
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写回结果
    if (tid < k) {
        out_values[tid] = local_vals[tid];
        out_indices[tid] = local_idx[tid];
    }
}
)";

// ============================================================================
// GPU 资源管理
// ============================================================================

static id<MTLDevice> g_topk_device_v13 = nil;
static id<MTLLibrary> g_topk_library_v13 = nil;
static id<MTLComputePipelineState> g_topk_simple_pipeline = nil;
static id<MTLComputePipelineState> g_bitonic_pipeline = nil;
static id<MTLCommandQueue> g_topk_queue_v13 = nil;

static bool init_topk_v13_gpu() {
    if (g_topk_device_v13) return true;

    @autoreleasepool {
        g_topk_device_v13 = MTLCreateSystemDefaultDevice();
        if (!g_topk_device_v13) return false;

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_4;

        NSString* source = [NSString stringWithUTF8String:kTopKV13Shader];
        g_topk_library_v13 = [g_topk_device_v13 newLibraryWithSource:source options:options error:&error];
        if (!g_topk_library_v13) {
            NSLog(@"V13 TopK shader compile failed: %@", error);
            return false;
        }

        id<MTLFunction> simpleFunc = [g_topk_library_v13 newFunctionWithName:@"topk_simple"];
        id<MTLFunction> bitonicFunc = [g_topk_library_v13 newFunctionWithName:@"bitonic_compare_swap"];

        g_topk_simple_pipeline = [g_topk_device_v13 newComputePipelineStateWithFunction:simpleFunc error:&error];
        g_bitonic_pipeline = [g_topk_device_v13 newComputePipelineStateWithFunction:bitonicFunc error:&error];

        g_topk_queue_v13 = [g_topk_device_v13 newCommandQueue];

        return g_topk_simple_pipeline && g_topk_queue_v13;
    }
}

// ============================================================================
// V13 GPU TopK 实现
// ============================================================================

bool is_topk_v13_gpu_available() {
    return init_topk_v13_gpu();
}

void topk_max_i32_v13(const int32_t* data, size_t count, size_t k,
                       int32_t* out_values, uint32_t* out_indices) {
    // GPU 只对大数据集有优势
    if (!init_topk_v13_gpu() || count < 1000000 || k > 100) {
        // 回退到 CPU V8
        topk_max_i32_v5(data, count, k, out_values, out_indices);
        return;
    }

    @autoreleasepool {
        // 分配 GPU 缓冲区
        id<MTLBuffer> dataBuffer = [g_topk_device_v13 newBufferWithBytes:data
                                                                  length:count * sizeof(int32_t)
                                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> valuesBuffer = [g_topk_device_v13 newBufferWithLength:k * sizeof(int32_t)
                                                                    options:MTLResourceStorageModeShared];

        id<MTLBuffer> indicesBuffer = [g_topk_device_v13 newBufferWithLength:k * sizeof(uint32_t)
                                                                     options:MTLResourceStorageModeShared];

        // 创建命令缓冲区
        id<MTLCommandBuffer> cmdBuffer = [g_topk_queue_v13 commandBuffer];

        {
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            [encoder setComputePipelineState:g_topk_simple_pipeline];

            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:valuesBuffer offset:0 atIndex:1];
            [encoder setBuffer:indicesBuffer offset:0 atIndex:2];

            uint32_t count32 = (uint32_t)count;
            uint32_t k32 = (uint32_t)k;
            [encoder setBytes:&count32 length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&k32 length:sizeof(uint32_t) atIndex:4];

            // Threadgroup 内存
            [encoder setThreadgroupMemoryLength:k * sizeof(int32_t) atIndex:0];
            [encoder setThreadgroupMemoryLength:k * sizeof(uint32_t) atIndex:1];

            // 使用单个 threadgroup，多线程并行扫描
            NSUInteger tg_size = std::min((NSUInteger)1024, g_topk_simple_pipeline.maxTotalThreadsPerThreadgroup);
            MTLSize gridSize = MTLSizeMake(tg_size, 1, 1);
            MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);

            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tgSize];
            [encoder endEncoding];
        }

        // 提交并等待
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 拷贝结果
        memcpy(out_values, [valuesBuffer contents], k * sizeof(int32_t));
        if (out_indices) {
            memcpy(out_indices, [indicesBuffer contents], k * sizeof(uint32_t));
        }
    }
}

void topk_min_i32_v13(const int32_t* data, size_t count, size_t k,
                       int32_t* out_values, uint32_t* out_indices) {
    // 简化: 使用 CPU 版本
    topk_min_i32_v5(data, count, k, out_values, out_indices);
}

}  // namespace sort
}  // namespace thunderduck
