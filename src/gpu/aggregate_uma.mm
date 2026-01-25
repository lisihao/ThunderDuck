/**
 * ThunderDuck - Aggregate v3 UMA Implementation
 *
 * GPU 加速聚合 + 零拷贝数据传输
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/aggregate.h"
#include "thunderduck/uma_memory.h"
#include "thunderduck/adaptive_strategy.h"
#include <algorithm>
#include <mutex>
#include <vector>
#include <limits>
#include <cstring>

namespace thunderduck {
namespace aggregate {

// ============================================================================
// Metal 资源管理
// ============================================================================

namespace {

struct AggregateMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    // 聚合内核
    id<MTLComputePipelineState> reduceSumI32Phase1 = nil;
    id<MTLComputePipelineState> reduceSumI64Final = nil;
    id<MTLComputePipelineState> reduceSumF32Phase1 = nil;
    id<MTLComputePipelineState> reduceMinI32Phase1 = nil;
    id<MTLComputePipelineState> reduceMaxI32Phase1 = nil;
    id<MTLComputePipelineState> reduceMinMaxI32Phase1 = nil;
    id<MTLComputePipelineState> reduceSumI32Sel = nil;
    id<MTLComputePipelineState> groupSumI32Atomic = nil;
    id<MTLComputePipelineState> groupCountAtomic = nil;
    id<MTLComputePipelineState> aggregateAllI32Phase1 = nil;
    id<MTLComputePipelineState> aggregateAllI32Final = nil;

    // 缓冲区池
    std::vector<id<MTLBuffer>> bufferPool;
    std::mutex poolMutex;

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

            // 内联 shader 代码 (避免文件路径问题)
            NSError* error = nil;
            NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;
constant uint WARP_SIZE = 32;

// ============================================================================
// 并行归约 - SUM (int32 -> int64)
// ============================================================================

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
    int64_t sum = 0;
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
// 并行归约 - SUM (float)
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
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
    uint num_groups = (sel_count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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

kernel void group_sum_i32_atomic(
    device const int32_t* values [[buffer(0)]],
    device const uint* groups [[buffer(1)]],
    device atomic_int* sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint group = groups[gid];
    int32_t val = values[gid];

    atomic_fetch_add_explicit(&sums[group], val, memory_order_relaxed);
}

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

// ============================================================================
// 一次遍历计算所有统计量 (融合 SUM+MIN+MAX)
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
    uint num_groups = (count + tg_size - 1) / tg_size;
    uint stride = tg_size * num_groups;

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
        result[0].count = num_blocks;
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
                NSLog(@"Warning: Aggregate Metal shader compilation failed: %@", error);
                initialized = true;
                return false;
            }

            // 创建计算管线
            auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (!func) return nil;
                return [device newComputePipelineStateWithFunction:func error:&error];
            };

            reduceSumI32Phase1 = createPipeline("reduce_sum_i32_phase1");
            reduceSumI64Final = createPipeline("reduce_sum_i64_final");
            reduceSumF32Phase1 = createPipeline("reduce_sum_f32_phase1");
            reduceMinI32Phase1 = createPipeline("reduce_min_i32_phase1");
            reduceMaxI32Phase1 = createPipeline("reduce_max_i32_phase1");
            reduceMinMaxI32Phase1 = createPipeline("reduce_minmax_i32_phase1");
            reduceSumI32Sel = createPipeline("reduce_sum_i32_sel");
            groupSumI32Atomic = createPipeline("group_sum_i32_atomic");
            groupCountAtomic = createPipeline("group_count_atomic");
            aggregateAllI32Phase1 = createPipeline("aggregate_all_i32_phase1");
            aggregateAllI32Final = createPipeline("aggregate_all_i32_final");

            initialized = true;
            return reduceSumI32Phase1 != nil;
        }
    }

    id<MTLBuffer> acquireBuffer(size_t size) {
        std::lock_guard<std::mutex> lock(poolMutex);

        for (auto it = bufferPool.begin(); it != bufferPool.end(); ++it) {
            if ((*it).length >= size) {
                id<MTLBuffer> buffer = *it;
                bufferPool.erase(it);
                return buffer;
            }
        }

        return [device newBufferWithLength:size
                                   options:MTLResourceStorageModeShared];
    }

    void releaseBuffer(id<MTLBuffer> buffer) {
        if (!buffer) return;
        std::lock_guard<std::mutex> lock(poolMutex);

        if (bufferPool.size() < 16) {
            bufferPool.push_back(buffer);
        }
    }
};

AggregateMetalContext& getContext() {
    static AggregateMetalContext ctx;
    return ctx;
}

constexpr size_t THREADGROUP_SIZE = 256;

// 直接使用策略选择器的阈值
using strategy::thresholds::AGGREGATE_GPU_MIN;

// 兼容性常量 - 用于非 config 版本的函数
constexpr size_t GPU_THRESHOLD = AGGREGATE_GPU_MIN;

// 策略选择 - 使用自适应策略选择器
AggregateStrategy selectStrategy(size_t count, AggregateStrategy requested,
                                   strategy::OperatorType op_type = strategy::OperatorType::AGGREGATE_SUM,
                                   bool is_page_aligned = false) {
    if (requested != AggregateStrategy::AUTO) {
        return requested;
    }

    // 使用自适应策略选择器
    strategy::DataCharacteristics data_chars;
    data_chars.row_count = count;
    data_chars.column_count = 1;
    data_chars.element_size = sizeof(int32_t);
    data_chars.selectivity = -1.0f;
    data_chars.cardinality_ratio = -1.0f;
    data_chars.is_page_aligned = is_page_aligned;

    auto executor = strategy::StrategySelector::instance().select(op_type, data_chars);

    if (executor == strategy::Executor::CPU_SIMD ||
        executor == strategy::Executor::CPU_SCALAR) {
        return AggregateStrategy::CPU_SIMD;
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceSumI32Phase1) {
        return AggregateStrategy::CPU_SIMD;
    }

    return AggregateStrategy::GPU;
}

} // anonymous namespace

// ============================================================================
// 公共 API
// ============================================================================

bool is_aggregate_gpu_available() {
    auto& ctx = getContext();
    return ctx.initialize() && ctx.reduceSumI32Phase1 != nil;
}

int64_t sum_i32_v3(const int32_t* input, size_t count) {
    AggregateConfigV3 config;
    return sum_i32_v3_config(input, count, config);
}

int64_t sum_i32_v3_config(const int32_t* input, size_t count,
                           const AggregateConfigV3& config) {
    if (count == 0 || !input) return 0;

    bool is_page_aligned = uma::is_page_aligned(input);
    AggregateStrategy strategy = selectStrategy(count, config.strategy,
                                                  strategy::OperatorType::AGGREGATE_SUM,
                                                  is_page_aligned);

    // CPU 路径
    if (strategy == AggregateStrategy::CPU_SIMD) {
        return sum_i32_v2(input, count);
    }

    // GPU 路径
    auto& ctx = getContext();
    if (!ctx.initialize()) {
        return sum_i32_v2(input, count);
    }

    @autoreleasepool {
        // 准备输入缓冲区
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        // 计算块数
        size_t numBlocks = (count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        numBlocks = std::min(numBlocks, (size_t)1024);  // 限制最大块数

        // 块结果缓冲区
        id<MTLBuffer> blockSumsBuffer = ctx.acquireBuffer(numBlocks * sizeof(int64_t));
        id<MTLBuffer> resultBuffer = ctx.acquireBuffer(sizeof(int64_t));

        uint32_t countU32 = static_cast<uint32_t>(count);

        // Phase 1: 每块归约
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceSumI32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockSumsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int64_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        // Phase 2: 最终归约
        encoder = [cmdBuffer computeCommandEncoder];
        [encoder setComputePipelineState:ctx.reduceSumI64Final];
        [encoder setBuffer:blockSumsBuffer offset:0 atIndex:0];
        [encoder setBuffer:resultBuffer offset:0 atIndex:1];
        uint32_t numBlocksU32 = static_cast<uint32_t>(numBlocks);
        [encoder setBytes:&numBlocksU32 length:sizeof(numBlocksU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int64_t) atIndex:0];

        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(THREADGROUP_SIZE, 1, 1)];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        int64_t result = *(int64_t*)[resultBuffer contents];

        ctx.releaseBuffer(blockSumsBuffer);
        ctx.releaseBuffer(resultBuffer);

        return result;
    }
}

double sum_f32_v3(const float* input, size_t count) {
    if (count == 0 || !input) return 0.0;

    if (count < GPU_THRESHOLD) {
        return sum_f32_v2(input, count);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceSumF32Phase1) {
        return sum_f32_v2(input, count);
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(float)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        }

        size_t numBlocks = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                    (size_t)1024);

        id<MTLBuffer> blockSumsBuffer = ctx.acquireBuffer(numBlocks * sizeof(float));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceSumF32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockSumsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(float) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // CPU 最终归约 (块数较少)
        float* blockSums = (float*)[blockSumsBuffer contents];
        double result = 0.0;
        for (size_t i = 0; i < numBlocks; i++) {
            result += blockSums[i];
        }

        ctx.releaseBuffer(blockSumsBuffer);

        return result;
    }
}

int32_t min_i32_v3(const int32_t* input, size_t count) {
    if (count == 0 || !input) return std::numeric_limits<int32_t>::max();

    if (count < GPU_THRESHOLD) {
        return min_i32(input, count);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceMinI32Phase1) {
        return min_i32(input, count);
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        size_t numBlocks = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                    (size_t)1024);

        id<MTLBuffer> blockMinsBuffer = ctx.acquireBuffer(numBlocks * sizeof(int32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceMinI32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockMinsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        int32_t* blockMins = (int32_t*)[blockMinsBuffer contents];
        int32_t result = std::numeric_limits<int32_t>::max();
        for (size_t i = 0; i < numBlocks; i++) {
            result = std::min(result, blockMins[i]);
        }

        ctx.releaseBuffer(blockMinsBuffer);

        return result;
    }
}

int32_t max_i32_v3(const int32_t* input, size_t count) {
    if (count == 0 || !input) return std::numeric_limits<int32_t>::min();

    if (count < GPU_THRESHOLD) {
        return max_i32(input, count);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceMaxI32Phase1) {
        return max_i32(input, count);
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        size_t numBlocks = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                    (size_t)1024);

        id<MTLBuffer> blockMaxsBuffer = ctx.acquireBuffer(numBlocks * sizeof(int32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceMaxI32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockMaxsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        int32_t* blockMaxs = (int32_t*)[blockMaxsBuffer contents];
        int32_t result = std::numeric_limits<int32_t>::min();
        for (size_t i = 0; i < numBlocks; i++) {
            result = std::max(result, blockMaxs[i]);
        }

        ctx.releaseBuffer(blockMaxsBuffer);

        return result;
    }
}

void minmax_i32_v3(const int32_t* input, size_t count,
                    int32_t* out_min, int32_t* out_max) {
    if (count == 0 || !input) {
        if (out_min) *out_min = std::numeric_limits<int32_t>::max();
        if (out_max) *out_max = std::numeric_limits<int32_t>::min();
        return;
    }

    if (count < GPU_THRESHOLD) {
        minmax_i32(input, count, out_min, out_max);
        return;
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceMinMaxI32Phase1) {
        minmax_i32(input, count, out_min, out_max);
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        size_t numBlocks = std::min((count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                    (size_t)1024);

        struct MinMaxResult {
            int32_t min_val;
            int32_t max_val;
        };

        id<MTLBuffer> blockResultsBuffer = ctx.acquireBuffer(numBlocks * sizeof(MinMaxResult));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceMinMaxI32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockResultsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:0];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:1];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        MinMaxResult* blockResults = (MinMaxResult*)[blockResultsBuffer contents];
        int32_t minResult = std::numeric_limits<int32_t>::max();
        int32_t maxResult = std::numeric_limits<int32_t>::min();
        for (size_t i = 0; i < numBlocks; i++) {
            minResult = std::min(minResult, blockResults[i].min_val);
            maxResult = std::max(maxResult, blockResults[i].max_val);
        }

        if (out_min) *out_min = minResult;
        if (out_max) *out_max = maxResult;

        ctx.releaseBuffer(blockResultsBuffer);
    }
}

int64_t sum_i32_sel_v3(const int32_t* input, const uint32_t* sel, size_t sel_count) {
    if (sel_count == 0 || !input || !sel) return 0;

    if (sel_count < GPU_THRESHOLD) {
        return sum_i32_sel(input, sel, sel_count);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.reduceSumI32Sel) {
        return sum_i32_sel(input, sel, sel_count);
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:sel_count * sizeof(int32_t) * 2  // 估计大小
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            // 无法确定 input 的实际大小，使用选择向量的最大索引
            // 这里简化处理，直接复制
            size_t maxInputSize = sel_count * 10 * sizeof(int32_t);  // 估计
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:maxInputSize
                                                 options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> selBuffer = nil;
        if (uma::is_page_aligned(sel)) {
            selBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)sel
                                                       length:sel_count * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        }
        if (!selBuffer) {
            selBuffer = [ctx.device newBufferWithBytes:sel
                                                length:sel_count * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
        }

        size_t numBlocks = std::min((sel_count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE,
                                    (size_t)1024);

        id<MTLBuffer> blockSumsBuffer = ctx.acquireBuffer(numBlocks * sizeof(int64_t));

        uint32_t selCountU32 = static_cast<uint32_t>(sel_count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.reduceSumI32Sel];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:selBuffer offset:0 atIndex:1];
        [encoder setBuffer:blockSumsBuffer offset:0 atIndex:2];
        [encoder setBytes:&selCountU32 length:sizeof(selCountU32) atIndex:3];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int64_t) atIndex:0];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        int64_t* blockSums = (int64_t*)[blockSumsBuffer contents];
        int64_t result = 0;
        for (size_t i = 0; i < numBlocks; i++) {
            result += blockSums[i];
        }

        ctx.releaseBuffer(blockSumsBuffer);

        return result;
    }
}

void group_sum_i32_v3(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    if (count == 0 || !values || !groups || !out_sums) return;

    // 分组聚合在 GPU 上需要原子操作，小分组数时效果不佳
    if (count < GPU_THRESHOLD || num_groups > 10000) {
        group_sum_i32(values, groups, count, num_groups, out_sums);
        return;
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.groupSumI32Atomic) {
        group_sum_i32(values, groups, count, num_groups, out_sums);
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

        // 使用 int32 原子操作 (Metal 不支持 int64 原子)
        id<MTLBuffer> sumsBuffer = ctx.acquireBuffer(num_groups * sizeof(int32_t));
        std::memset([sumsBuffer contents], 0, num_groups * sizeof(int32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupSumI32Atomic];
        [encoder setBuffer:valuesBuffer offset:0 atIndex:0];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:1];
        [encoder setBuffer:sumsBuffer offset:0 atIndex:2];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:3];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.groupSumI32Atomic.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 转换 int32 -> int64
        int32_t* sums32 = (int32_t*)[sumsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            out_sums[i] = sums32[i];
        }

        ctx.releaseBuffer(sumsBuffer);
    }
}

void group_count_v3(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    if (count < GPU_THRESHOLD || num_groups > 10000) {
        group_count(groups, count, num_groups, out_counts);
        return;
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.groupCountAtomic) {
        group_count(groups, count, num_groups, out_counts);
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

        id<MTLBuffer> countsBuffer = ctx.acquireBuffer(num_groups * sizeof(uint32_t));
        std::memset([countsBuffer contents], 0, num_groups * sizeof(uint32_t));

        uint32_t countU32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.groupCountAtomic];
        [encoder setBuffer:groupsBuffer offset:0 atIndex:0];
        [encoder setBuffer:countsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.groupCountAtomic.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 转换 uint32 -> size_t
        uint32_t* counts32 = (uint32_t*)[countsBuffer contents];
        for (size_t i = 0; i < num_groups; i++) {
            out_counts[i] = counts32[i];
        }

        ctx.releaseBuffer(countsBuffer);
    }
}

AggregateStats aggregate_all_i32_v3(const int32_t* input, size_t count) {
    AggregateStats result = {0, 0,
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min()};

    if (count == 0 || !input) return result;

    if (count < GPU_THRESHOLD) {
        return aggregate_all_i32(input, count);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.aggregateAllI32Phase1 || !ctx.aggregateAllI32Final) {
        return aggregate_all_i32(input, count);
    }

    @autoreleasepool {
        // 准备输入缓冲区
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(input)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        // AllStats 结构: sum(8) + min(4) + max(4) + count(4) = 20 bytes, 对齐到 24
        struct GPUAllStats {
            int64_t sum;
            int32_t min_val;
            int32_t max_val;
            uint32_t count;
            uint32_t pad;  // 对齐
        };

        // 计算块数
        size_t numBlocks = (count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        numBlocks = std::min(numBlocks, (size_t)1024);

        // 块结果缓冲区
        id<MTLBuffer> blockStatsBuffer = ctx.acquireBuffer(numBlocks * sizeof(GPUAllStats));
        id<MTLBuffer> resultBuffer = ctx.acquireBuffer(sizeof(GPUAllStats));

        uint32_t countU32 = static_cast<uint32_t>(count);

        // Phase 1: 每块计算统计量 (融合 SUM+MIN+MAX)
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.aggregateAllI32Phase1];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:blockStatsBuffer offset:0 atIndex:1];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:2];
        // 3个共享内存区: sum(int64), min(int32), max(int32)
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int64_t) atIndex:0];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:1];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:2];

        MTLSize gridSize = MTLSizeMake(numBlocks * THREADGROUP_SIZE, 1, 1);
        MTLSize groupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        // Phase 2: 最终归约
        encoder = [cmdBuffer computeCommandEncoder];
        [encoder setComputePipelineState:ctx.aggregateAllI32Final];
        [encoder setBuffer:blockStatsBuffer offset:0 atIndex:0];
        [encoder setBuffer:resultBuffer offset:0 atIndex:1];
        uint32_t numBlocksU32 = static_cast<uint32_t>(numBlocks);
        [encoder setBytes:&numBlocksU32 length:sizeof(numBlocksU32) atIndex:2];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int64_t) atIndex:0];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:1];
        [encoder setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(int32_t) atIndex:2];

        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(THREADGROUP_SIZE, 1, 1)];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 读取结果
        GPUAllStats* gpuResult = (GPUAllStats*)[resultBuffer contents];
        result.sum = gpuResult->sum;
        result.min_val = gpuResult->min_val;
        result.max_val = gpuResult->max_val;
        result.count = count;

        ctx.releaseBuffer(blockStatsBuffer);
        ctx.releaseBuffer(resultBuffer);

        return result;
    }
}

} // namespace aggregate
} // namespace thunderduck
