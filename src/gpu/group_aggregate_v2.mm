/**
 * ThunderDuck - V9.2 GPU Group Aggregation
 *
 * 两阶段分组聚合优化:
 * - Phase 1: Threadgroup 本地累加 (共享内存原子)
 * - Phase 2: 全局合并 (设备内存原子)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/aggregate.h"
#include "thunderduck/uma_memory.h"
#include <algorithm>
#include <mutex>
#include <limits>
#include <cstring>

namespace thunderduck {
namespace aggregate {

// ============================================================================
// V9.2 Metal Context
// ============================================================================

namespace {

struct GroupAggregateV2Context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    // V9.2 两阶段内核
    id<MTLComputePipelineState> groupSumTwopass = nil;
    id<MTLComputePipelineState> groupCountTwopass = nil;
    id<MTLComputePipelineState> groupMinTwopass = nil;
    id<MTLComputePipelineState> groupMaxTwopass = nil;
    id<MTLComputePipelineState> groupStatsFused = nil;

    bool initialized = false;

    bool initialize() {
        if (initialized) return device != nil;

        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                initialized = true;
                return false;
            }

            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                device = nil;
                initialized = true;
                return false;
            }

            // V9.2 两阶段分组聚合 shader
            NSError* error = nil;
            NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;
constant uint MAX_LOCAL_GROUPS = 1024;

// V9.2 两阶段分组求和
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

    // Phase 2: 本地累加
    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_add_explicit(&local_sums[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 合并到全局
    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_sums[g], memory_order_relaxed);
        if (local_val != 0) {
            atomic_fetch_add_explicit(&global_sums[g], local_val, memory_order_relaxed);
        }
    }
}

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
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_counts[g], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        atomic_fetch_add_explicit(&local_counts[group], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = lid; g < num_groups; g += tg_size) {
        uint local_val = atomic_load_explicit(&local_counts[g], memory_order_relaxed);
        if (local_val != 0) {
            atomic_fetch_add_explicit(&global_counts[g], local_val, memory_order_relaxed);
        }
    }
}

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
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_mins[g], INT_MAX, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_min_explicit(&local_mins[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_mins[g], memory_order_relaxed);
        if (local_val != INT_MAX) {
            atomic_fetch_min_explicit(&global_mins[g], local_val, memory_order_relaxed);
        }
    }
}

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
    for (uint g = lid; g < num_groups; g += tg_size) {
        atomic_store_explicit(&local_maxs[g], INT_MIN, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = gid; i < count; i += grid_size) {
        uint group = groups[i];
        int32_t val = values[i];
        atomic_fetch_max_explicit(&local_maxs[group], val, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint g = lid; g < num_groups; g += tg_size) {
        int local_val = atomic_load_explicit(&local_maxs[g], memory_order_relaxed);
        if (local_val != INT_MIN) {
            atomic_fetch_max_explicit(&global_maxs[g], local_val, memory_order_relaxed);
        }
    }
}
)";

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }

            library = [device newLibraryWithSource:shaderSource options:options error:&error];

            if (!library) {
                NSLog(@"V9.2 Group Aggregate shader compilation failed: %@", error);
                initialized = true;
                return false;
            }

            auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (!func) return nil;
                return [device newComputePipelineStateWithFunction:func error:&error];
            };

            groupSumTwopass = createPipeline("group_sum_i32_twopass");
            groupCountTwopass = createPipeline("group_count_twopass");
            groupMinTwopass = createPipeline("group_min_i32_twopass");
            groupMaxTwopass = createPipeline("group_max_i32_twopass");

            initialized = true;
            return groupSumTwopass != nil;
        }
    }
};

GroupAggregateV2Context& getV2Context() {
    static GroupAggregateV2Context ctx;
    return ctx;
}

constexpr size_t THREADGROUP_SIZE = 256;
constexpr size_t MAX_LOCAL_GROUPS = 1024;  // 共享内存限制
constexpr size_t GPU_MIN_COUNT = 100000;   // GPU 最小数据量

} // anonymous namespace

// ============================================================================
// V9.2 公共 API
// ============================================================================

bool is_group_aggregate_v2_available() {
    auto& ctx = getV2Context();
    return ctx.initialize() && ctx.groupSumTwopass != nil;
}

void group_sum_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    // 条件: 数据量足够大, 分组数不超过共享内存限制
    if (count < GPU_MIN_COUNT || num_groups > MAX_LOCAL_GROUPS) {
        // 回退到 CPU v4 多线程实现
        group_sum_i32_v4_parallel(values, groups, count, num_groups, out_sums);
        return;
    }

    auto& ctx = getV2Context();
    if (!ctx.initialize() || !ctx.groupSumTwopass) {
        group_sum_i32_v4_parallel(values, groups, count, num_groups, out_sums);
        return;
    }

    @autoreleasepool {
        // 准备输入缓冲区
        id<MTLBuffer> valuesBuffer = nil;
        if (uma::is_page_aligned(values)) {
            valuesBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)values
                                                          length:count * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!valuesBuffer) {
            valuesBuffer = [ctx.device newBufferWithBytes:values
                                                   length:count * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> groupsBuffer = nil;
        if (uma::is_page_aligned(groups)) {
            groupsBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)groups
                                                          length:count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!groupsBuffer) {
            groupsBuffer = [ctx.device newBufferWithBytes:groups
                                                   length:count * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        // 输出缓冲区 (使用 int32 原子, Metal 不支持 int64 原子)
        id<MTLBuffer> sumsBuffer = [ctx.device newBufferWithLength:num_groups * sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];
        std::memset([sumsBuffer contents], 0, num_groups * sizeof(int32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);
        uint32_t numGroupsU32 = static_cast<uint32_t>(num_groups);

        // 计算 grid 配置
        // 使用多个 threadgroup 以利用并行
        size_t numThreadgroups = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                          (size_t)64);  // 限制 threadgroup 数量

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupSumTwopass];
        [encoder setBuffer:valuesBuffer offset:0 atIndex:0];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:1];
        [encoder setBuffer:sumsBuffer offset:0 atIndex:2];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:3];
        [encoder setBytes:&numGroupsU32 length:sizeof(numGroupsU32) atIndex:4];

        // 共享内存: 每个分组一个 atomic_int
        [encoder setThreadgroupMemoryLength:num_groups * sizeof(int32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numThreadgroups * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 转换 int32 -> int64
        int32_t* sums32 = (int32_t*)[sumsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            out_sums[i] = sums32[i];
        }
    }
}

void group_count_v5(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    if (count < GPU_MIN_COUNT || num_groups > MAX_LOCAL_GROUPS) {
        group_count_v4_parallel(groups, count, num_groups, out_counts);
        return;
    }

    auto& ctx = getV2Context();
    if (!ctx.initialize() || !ctx.groupCountTwopass) {
        group_count_v4_parallel(groups, count, num_groups, out_counts);
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> groupsBuffer = nil;
        if (uma::is_page_aligned(groups)) {
            groupsBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)groups
                                                          length:count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!groupsBuffer) {
            groupsBuffer = [ctx.device newBufferWithBytes:groups
                                                   length:count * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> countsBuffer = [ctx.device newBufferWithLength:num_groups * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        std::memset([countsBuffer contents], 0, num_groups * sizeof(uint32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);
        uint32_t numGroupsU32 = static_cast<uint32_t>(num_groups);

        size_t numThreadgroups = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                          (size_t)64);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupCountTwopass];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:0];
        [encoder setBuffer:countsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setBytes:&numGroupsU32 length:sizeof(numGroupsU32) atIndex:3];
        [encoder setThreadgroupMemoryLength:num_groups * sizeof(uint32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numThreadgroups * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        uint32_t* counts32 = (uint32_t*)[countsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            out_counts[i] = counts32[i];
        }
    }
}

void group_min_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins) {
    if (count == 0 || !values || !groups || !out_mins) return;

    if (count < GPU_MIN_COUNT || num_groups > MAX_LOCAL_GROUPS) {
        group_min_i32_v4(values, groups, count, num_groups, out_mins);
        return;
    }

    auto& ctx = getV2Context();
    if (!ctx.initialize() || !ctx.groupMinTwopass) {
        group_min_i32_v4(values, groups, count, num_groups, out_mins);
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> valuesBuffer = nil;
        if (uma::is_page_aligned(values)) {
            valuesBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)values
                                                          length:count * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!valuesBuffer) {
            valuesBuffer = [ctx.device newBufferWithBytes:values
                                                   length:count * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> groupsBuffer = nil;
        if (uma::is_page_aligned(groups)) {
            groupsBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)groups
                                                          length:count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!groupsBuffer) {
            groupsBuffer = [ctx.device newBufferWithBytes:groups
                                                   length:count * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> minsBuffer = [ctx.device newBufferWithLength:num_groups * sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];
        // 初始化为 INT_MAX
        int32_t* minsInit = (int32_t*)[minsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            minsInit[i] = std::numeric_limits<int32_t>::max();
        }

        uint32_t countU32 = static_cast<uint32_t>(count);
        uint32_t numGroupsU32 = static_cast<uint32_t>(num_groups);

        size_t numThreadgroups = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                          (size_t)64);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupMinTwopass];
        [encoder setBuffer:valuesBuffer offset:0 atIndex:0];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:1];
        [encoder setBuffer:minsBuffer offset:0 atIndex:2];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:3];
        [encoder setBytes:&numGroupsU32 length:sizeof(numGroupsU32) atIndex:4];
        [encoder setThreadgroupMemoryLength:num_groups * sizeof(int32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numThreadgroups * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        std::memcpy(out_mins, [minsBuffer contents], num_groups * sizeof(int32_t));
    }
}

void group_max_i32_v5(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs) {
    if (count == 0 || !values || !groups || !out_maxs) return;

    if (count < GPU_MIN_COUNT || num_groups > MAX_LOCAL_GROUPS) {
        group_max_i32_v4(values, groups, count, num_groups, out_maxs);
        return;
    }

    auto& ctx = getV2Context();
    if (!ctx.initialize() || !ctx.groupMaxTwopass) {
        group_max_i32_v4(values, groups, count, num_groups, out_maxs);
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> valuesBuffer = nil;
        if (uma::is_page_aligned(values)) {
            valuesBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)values
                                                          length:count * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!valuesBuffer) {
            valuesBuffer = [ctx.device newBufferWithBytes:values
                                                   length:count * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> groupsBuffer = nil;
        if (uma::is_page_aligned(groups)) {
            groupsBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)groups
                                                          length:count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }
        if (!groupsBuffer) {
            groupsBuffer = [ctx.device newBufferWithBytes:groups
                                                   length:count * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> maxsBuffer = [ctx.device newBufferWithLength:num_groups * sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];
        // 初始化为 INT_MIN
        int32_t* maxsInit = (int32_t*)[maxsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            maxsInit[i] = std::numeric_limits<int32_t>::min();
        }

        uint32_t countU32 = static_cast<uint32_t>(count);
        uint32_t numGroupsU32 = static_cast<uint32_t>(num_groups);

        size_t numThreadgroups = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                          (size_t)64);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupMaxTwopass];
        [encoder setBuffer:valuesBuffer offset:0 atIndex:0];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:1];
        [encoder setBuffer:maxsBuffer offset:0 atIndex:2];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:3];
        [encoder setBytes:&numGroupsU32 length:sizeof(numGroupsU32) atIndex:4];
        [encoder setThreadgroupMemoryLength:num_groups * sizeof(int32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numThreadgroups * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        std::memcpy(out_maxs, [maxsBuffer contents], num_groups * sizeof(int32_t));
    }
}

} // namespace aggregate
} // namespace thunderduck
