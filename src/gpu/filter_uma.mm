/**
 * ThunderDuck - Filter v4 UMA Implementation
 *
 * GPU 加速过滤 + 零拷贝数据传输
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/filter.h"
#include "thunderduck/uma_memory.h"
#include "thunderduck/adaptive_strategy.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstring>

namespace thunderduck {
namespace filter {

// ============================================================================
// Metal 资源管理
// ============================================================================

namespace {

struct FilterMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    // 过滤内核
    id<MTLComputePipelineState> filterAtomicI32 = nil;
    id<MTLComputePipelineState> filterSimd4I32 = nil;
    id<MTLComputePipelineState> filterRangeI32 = nil;
    id<MTLComputePipelineState> filterAtomicF32 = nil;

    // 前缀和相关
    id<MTLComputePipelineState> filterCountI32 = nil;
    id<MTLComputePipelineState> prefixSum = nil;
    id<MTLComputePipelineState> filterWriteIndices = nil;

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

enum CompareOp : uint {
    EQ = 0, NE = 1, LT = 2, LE = 3, GT = 4, GE = 5
};

struct FilterParams {
    uint count;
    uint op;
    int value;
    uint pad;
};

// 原子版过滤 - 单 pass
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

// 向量化过滤 (4 元素/线程) - 使用 threadgroup 级别优化
kernel void filter_simd4_i32(
    device const int4* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant FilterParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    threadgroup uint* tg_counts [[threadgroup(0)]]
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

    uint remaining = min(4u, params.count - base_idx);
    uint local_count = 0;
    uint local_indices[4];

    for (uint i = 0; i < remaining; i++) {
        if (mask[i]) {
            local_indices[local_count++] = base_idx + i;
        }
    }

    // Threadgroup 前缀和减少原子争用
    tg_counts[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < group_size; stride *= 2) {
        uint val = 0;
        if (lid >= stride) val = tg_counts[lid - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_counts[lid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint local_offset = (lid > 0) ? tg_counts[lid - 1] : 0;
    uint tg_total = tg_counts[group_size - 1];

    threadgroup uint tg_global_offset;
    if (lid == group_size - 1) {
        tg_global_offset = atomic_fetch_add_explicit(counter, tg_total, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint global_offset = tg_global_offset + local_offset;
    for (uint i = 0; i < local_count; i++) {
        out_indices[global_offset + i] = local_indices[i];
    }
}

// 范围过滤
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

// Float 过滤
struct FilterParamsF32 {
    uint count;
    uint op;
    float value;
    uint pad;
};

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
                NSLog(@"Warning: Filter Metal shader compilation failed: %@", error);
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

            filterAtomicI32 = createPipeline("filter_atomic_i32");
            filterSimd4I32 = createPipeline("filter_simd4_i32");
            filterRangeI32 = createPipeline("filter_range_i32");
            filterAtomicF32 = createPipeline("filter_atomic_f32");
            filterCountI32 = createPipeline("filter_count_i32");
            prefixSum = createPipeline("prefix_sum");
            filterWriteIndices = createPipeline("filter_write_indices_i32");

            initialized = true;
            return filterAtomicI32 != nil;
        }
    }

    id<MTLBuffer> acquireBuffer(size_t size) {
        std::lock_guard<std::mutex> lock(poolMutex);

        // 查找合适的缓冲区
        for (auto it = bufferPool.begin(); it != bufferPool.end(); ++it) {
            if ((*it).length >= size) {
                id<MTLBuffer> buffer = *it;
                bufferPool.erase(it);
                return buffer;
            }
        }

        // 创建新缓冲区
        return [device newBufferWithLength:size
                                   options:MTLResourceStorageModeShared];
    }

    void releaseBuffer(id<MTLBuffer> buffer) {
        if (!buffer) return;
        std::lock_guard<std::mutex> lock(poolMutex);

        // 限制池大小
        if (bufferPool.size() < 16) {
            bufferPool.push_back(buffer);
        }
    }
};

FilterMetalContext& getContext() {
    static FilterMetalContext ctx;
    return ctx;
}

// 直接使用策略选择器的阈值
using strategy::thresholds::FILTER_GPU_MIN;

// 兼容性常量 - 用于非 config 版本的函数
constexpr size_t GPU_THRESHOLD = FILTER_GPU_MIN;

// 策略选择 - 使用自适应策略选择器
FilterStrategy selectStrategy(size_t count, float selectivity_hint,
                               FilterStrategy requested, bool is_page_aligned) {
    if (requested != FilterStrategy::AUTO) {
        return requested;
    }

    // 使用自适应策略选择器
    strategy::DataCharacteristics data_chars;
    data_chars.row_count = count;
    data_chars.column_count = 1;
    data_chars.element_size = sizeof(int32_t);
    data_chars.selectivity = selectivity_hint;
    data_chars.cardinality_ratio = -1.0f;
    data_chars.is_page_aligned = is_page_aligned;

    auto executor = strategy::StrategySelector::instance().select(
        strategy::OperatorType::FILTER, data_chars);

    // GPU 不可用或策略选择 CPU
    if (executor == strategy::Executor::CPU_SIMD ||
        executor == strategy::Executor::CPU_SCALAR) {
        return FilterStrategy::CPU_SIMD;
    }

    // GPU 可用性检查
    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.filterAtomicI32) {
        return FilterStrategy::CPU_SIMD;
    }

    // 选择率未知时，CPU 更稳定
    if (selectivity_hint < 0) {
        return FilterStrategy::CPU_SIMD;
    }

    // 根据选择率选择 GPU 策略
    // 低选择率 (<10%): GPU 原子版高效
    // 中高选择率 (>=10%): CPU 更优 (GPU 原子争用)
    if (selectivity_hint < 0.1f) {
        return FilterStrategy::GPU_ATOMIC;
    }

    // 高选择率时 CPU 更高效
    return FilterStrategy::CPU_SIMD;
}

} // anonymous namespace

// ============================================================================
// 公共 API
// ============================================================================

bool is_filter_gpu_available() {
    auto& ctx = getContext();
    return ctx.initialize() && ctx.filterAtomicI32 != nil;
}

size_t filter_i32_v4(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices) {
    FilterConfigV4 config;
    return filter_i32_v4_config(input, count, op, value, out_indices, config);
}

size_t filter_i32_v4_config(const int32_t* input, size_t count,
                             CompareOp op, int32_t value,
                             uint32_t* out_indices,
                             const FilterConfigV4& config) {
    if (count == 0 || !input || !out_indices) return 0;

    bool is_page_aligned = uma::is_page_aligned(input);
    FilterStrategy strategy = selectStrategy(count, config.selectivity_hint,
                                               config.strategy, is_page_aligned);

    // CPU 路径
    if (strategy == FilterStrategy::CPU_SIMD) {
        return filter_i32_v3(input, count, op, value, out_indices);
    }

    // GPU 路径
    auto& ctx = getContext();
    if (!ctx.initialize()) {
        return filter_i32_v3(input, count, op, value, out_indices);
    }

    @autoreleasepool {
        auto& umaMgr = uma::UMAMemoryManager::instance();

        // 准备输入缓冲区 (零拷贝或复制)
        id<MTLBuffer> inputBuffer = nil;
        bool inputOwned = false;

        if (uma::is_page_aligned(input)) {
            // 零拷贝
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)input
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }

        if (!inputBuffer) {
            // 需要复制
            inputBuffer = [ctx.device newBufferWithBytes:input
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
            inputOwned = true;
        }

        // 输出缓冲区
        id<MTLBuffer> outputBuffer = nil;
        bool outputOwned = false;

        if (uma::is_page_aligned(out_indices)) {
            outputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)out_indices
                                                          length:count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        }

        if (!outputBuffer) {
            outputBuffer = ctx.acquireBuffer(count * sizeof(uint32_t));
            outputOwned = true;
        }

        // 计数器缓冲区
        id<MTLBuffer> counterBuffer = ctx.acquireBuffer(sizeof(uint32_t));
        *(uint32_t*)[counterBuffer contents] = 0;

        // 参数缓冲区
        struct FilterParams {
            uint32_t count;
            uint32_t op;
            int32_t value;
            uint32_t pad;
        };

        FilterParams params = {
            static_cast<uint32_t>(count),
            static_cast<uint32_t>(op),
            value,
            0
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // 创建命令缓冲区
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        // 选择内核
        id<MTLComputePipelineState> pipeline = nil;
        size_t threadsPerElement = 1;

        if (strategy == FilterStrategy::GPU_ATOMIC) {
            pipeline = ctx.filterAtomicI32;
            threadsPerElement = 1;
        } else {
            // SIMD4 版本
            pipeline = ctx.filterSimd4I32;
            threadsPerElement = 4;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:counterBuffer offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

        // 计算线程配置
        NSUInteger threadCount = (count + threadsPerElement - 1) / threadsPerElement;
        NSUInteger threadgroupSize = std::min((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(threadCount, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 读取结果
        uint32_t resultCount = *(uint32_t*)[counterBuffer contents];

        // 如果输出需要复制
        if (outputOwned && resultCount > 0) {
            std::memcpy(out_indices, [outputBuffer contents], resultCount * sizeof(uint32_t));
        }

        // 释放资源
        ctx.releaseBuffer(counterBuffer);
        if (outputOwned) {
            ctx.releaseBuffer(outputBuffer);
        }

        return resultCount;
    }
}

size_t count_i32_v4(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    // 计数操作 GPU 优势不大，直接使用 CPU v3
    return count_i32_v3(input, count, op, value);
}

size_t filter_i32_range_v4(const int32_t* input, size_t count,
                            int32_t low, int32_t high,
                            uint32_t* out_indices) {
    if (count == 0 || !input || !out_indices) return 0;

    // 小数据量使用 CPU
    if (count < GPU_THRESHOLD) {
        return filter_i32_range(input, count, low, high, out_indices);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.filterRangeI32) {
        return filter_i32_range(input, count, low, high, out_indices);
    }

    @autoreleasepool {
        // 准备缓冲区
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

        id<MTLBuffer> outputBuffer = ctx.acquireBuffer(count * sizeof(uint32_t));
        id<MTLBuffer> counterBuffer = ctx.acquireBuffer(sizeof(uint32_t));
        *(uint32_t*)[counterBuffer contents] = 0;

        struct RangeParams {
            uint32_t count;
            int32_t low;
            int32_t high;
            uint32_t pad;
        };

        RangeParams params = {
            static_cast<uint32_t>(count),
            low,
            high,
            0
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // 执行
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.filterRangeI32];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:counterBuffer offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.filterRangeI32.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        uint32_t resultCount = *(uint32_t*)[counterBuffer contents];

        if (resultCount > 0) {
            std::memcpy(out_indices, [outputBuffer contents], resultCount * sizeof(uint32_t));
        }

        ctx.releaseBuffer(outputBuffer);
        ctx.releaseBuffer(counterBuffer);

        return resultCount;
    }
}

size_t filter_f32_v4(const float* input, size_t count,
                      CompareOp op, float value,
                      uint32_t* out_indices) {
    if (count == 0 || !input || !out_indices) return 0;

    // 小数据量使用 CPU
    if (count < GPU_THRESHOLD) {
        return filter_f32(input, count, op, value, out_indices);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.filterAtomicF32) {
        return filter_f32(input, count, op, value, out_indices);
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

        id<MTLBuffer> outputBuffer = ctx.acquireBuffer(count * sizeof(uint32_t));
        id<MTLBuffer> counterBuffer = ctx.acquireBuffer(sizeof(uint32_t));
        *(uint32_t*)[counterBuffer contents] = 0;

        struct FilterParamsF32 {
            uint32_t count;
            uint32_t op;
            float value;
            uint32_t pad;
        };

        FilterParamsF32 params = {
            static_cast<uint32_t>(count),
            static_cast<uint32_t>(op),
            value,
            0
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.filterAtomicF32];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:counterBuffer offset:0 atIndex:2];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.filterAtomicF32.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        uint32_t resultCount = *(uint32_t*)[counterBuffer contents];

        if (resultCount > 0) {
            std::memcpy(out_indices, [outputBuffer contents], resultCount * sizeof(uint32_t));
        }

        ctx.releaseBuffer(outputBuffer);
        ctx.releaseBuffer(counterBuffer);

        return resultCount;
    }
}

} // namespace filter
} // namespace thunderduck
