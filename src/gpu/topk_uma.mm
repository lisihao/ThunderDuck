/**
 * ThunderDuck - TopK v6 UMA Implementation
 *
 * GPU 加速 TopK + 零拷贝数据传输
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/sort.h"
#include "thunderduck/uma_memory.h"
#include "thunderduck/adaptive_strategy.h"
#include <algorithm>
#include <mutex>
#include <vector>
#include <cstring>

namespace thunderduck {
namespace sort {

// ============================================================================
// Metal 资源管理
// ============================================================================

namespace {

struct TopKMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    // TopK 内核
    id<MTLComputePipelineState> sampleThreshold = nil;
    id<MTLComputePipelineState> topkFilterCount = nil;
    id<MTLComputePipelineState> topkFilterWrite = nil;
    id<MTLComputePipelineState> bitonicSortStep = nil;
    id<MTLComputePipelineState> bitonicSortLocal = nil;
    id<MTLComputePipelineState> initIndices = nil;
    id<MTLComputePipelineState> copyTopkResult = nil;

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

            NSError* error = nil;

            // 从源码编译
            NSString* sourcePath = @"src/gpu/shaders/topk.metal";
            if ([[NSFileManager defaultManager] fileExistsAtPath:sourcePath]) {
                NSString* source = [NSString stringWithContentsOfFile:sourcePath
                                                             encoding:NSUTF8StringEncoding
                                                                error:&error];
                if (source) {
                    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                    options.fastMathEnabled = YES;
                    library = [device newLibraryWithSource:source options:options error:&error];
                }
            }

            if (!library) {
                NSLog(@"Warning: TopK Metal library not found, GPU topk disabled");
                initialized = true;
                return false;
            }

            auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (!func) return nil;
                return [device newComputePipelineStateWithFunction:func error:&error];
            };

            sampleThreshold = createPipeline("sample_threshold");
            topkFilterCount = createPipeline("topk_filter_count");
            topkFilterWrite = createPipeline("topk_filter_write");
            bitonicSortStep = createPipeline("bitonic_sort_step");
            bitonicSortLocal = createPipeline("bitonic_sort_local");
            initIndices = createPipeline("init_indices");
            copyTopkResult = createPipeline("copy_topk_result");

            initialized = true;
            return bitonicSortStep != nil;
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

TopKMetalContext& getContext() {
    static TopKMetalContext ctx;
    return ctx;
}

constexpr size_t THREADGROUP_SIZE = 256;

// 策略选择 - 使用自适应策略选择器
TopKStrategy selectStrategy(size_t count, size_t k, float cardinality_hint,
                             TopKStrategy requested, bool is_page_aligned = false) {
    if (requested != TopKStrategy::AUTO) {
        return requested;
    }

    // 使用自适应策略选择器
    strategy::DataCharacteristics data_chars;
    data_chars.row_count = count;
    data_chars.column_count = 1;
    data_chars.element_size = sizeof(int32_t);
    // K/count 作为 selectivity 传递给策略选择器
    data_chars.selectivity = (count > 0) ? static_cast<float>(k) / count : 0.0f;
    data_chars.cardinality_ratio = cardinality_hint;
    data_chars.is_page_aligned = is_page_aligned;

    auto executor = strategy::StrategySelector::instance().select(
        strategy::OperatorType::TOPK, data_chars);

    // 策略选择器建议使用 CPU
    if (executor == strategy::Executor::CPU_SIMD ||
        executor == strategy::Executor::CPU_SCALAR) {
        // 根据基数提示选择 v5 或 v4
        if (cardinality_hint >= 0 && cardinality_hint < 0.01f) {
            return TopKStrategy::CPU_COUNT;
        }
        return TopKStrategy::CPU_SAMPLE;
    }

    // GPU 可用性检查
    auto& ctx = getContext();
    if (!ctx.initialize()) {
        return TopKStrategy::CPU_SAMPLE;
    }

    // 大数据量，根据 K 值选择 GPU 策略
    if (k <= 64) {
        return TopKStrategy::GPU_FILTER;
    } else {
        return TopKStrategy::GPU_BITONIC;
    }
}

// CPU 采样预过滤实现
void topk_max_cpu_sample(const int32_t* data, size_t count, size_t k,
                          int32_t* out_values, uint32_t* out_indices) {
    topk_max_i32_v4(data, count, k, out_values, out_indices);
}

void topk_min_cpu_sample(const int32_t* data, size_t count, size_t k,
                          int32_t* out_values, uint32_t* out_indices) {
    topk_min_i32_v4(data, count, k, out_values, out_indices);
}

// GPU 过滤方法 (小 K)
void topk_gpu_filter(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices, bool is_max) {
    auto& ctx = getContext();

    @autoreleasepool {
        // 准备输入缓冲区
        id<MTLBuffer> inputBuffer = nil;
        if (uma::is_page_aligned(data)) {
            inputBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)data
                                                         length:count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!inputBuffer) {
            inputBuffer = [ctx.device newBufferWithBytes:data
                                                  length:count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        // 采样估计阈值 (在 CPU 上做，简单快速)
        size_t sample_size = std::min((size_t)1000, count);
        size_t step = count / sample_size;
        std::vector<int32_t> samples(sample_size);
        for (size_t i = 0; i < sample_size; i++) {
            samples[i] = data[i * step];
        }

        // 快速排序找阈值
        size_t threshold_idx = is_max ? (sample_size - k * sample_size / count - 1)
                                      : (k * sample_size / count);
        threshold_idx = std::clamp(threshold_idx, (size_t)0, sample_size - 1);
        std::nth_element(samples.begin(), samples.begin() + threshold_idx, samples.end());
        int32_t threshold = samples[threshold_idx];

        // 调整阈值以确保足够候选
        if (is_max) {
            threshold = threshold - 1;  // 放宽条件
        } else {
            threshold = threshold + 1;
        }

        // 分配候选缓冲区 (估计大小)
        size_t est_candidates = std::min(count, k * 4);
        id<MTLBuffer> candidatesBuffer = ctx.acquireBuffer(est_candidates * sizeof(int32_t));
        id<MTLBuffer> candidateIndicesBuffer = ctx.acquireBuffer(est_candidates * sizeof(uint32_t));
        id<MTLBuffer> counterBuffer = ctx.acquireBuffer(sizeof(uint32_t));
        *(uint32_t*)[counterBuffer contents] = 0;

        // 准备参数
        struct TopKParams {
            uint32_t count;
            uint32_t k;
            int32_t threshold;
            uint32_t is_max;
        };

        TopKParams params = {
            static_cast<uint32_t>(count),
            static_cast<uint32_t>(k),
            threshold,
            is_max ? 1u : 0u
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // 第一遍: 计算候选数量
        size_t numThreads = (count + 63) / 64;
        id<MTLBuffer> countsBuffer = ctx.acquireBuffer(numThreads * sizeof(uint32_t));

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.topkFilterCount];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:countsBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((NSUInteger)256,
            ctx.topkFilterCount.maxTotalThreadsPerThreadgroup), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 在 CPU 上计算前缀和 (简单快速)
        uint32_t* counts = (uint32_t*)[countsBuffer contents];
        std::vector<uint32_t> offsets(numThreads);
        uint32_t total = 0;
        for (size_t i = 0; i < numThreads; i++) {
            offsets[i] = total;
            total += counts[i];
        }

        if (total == 0) {
            // 没有候选，回退到 CPU
            ctx.releaseBuffer(countsBuffer);
            ctx.releaseBuffer(candidatesBuffer);
            ctx.releaseBuffer(candidateIndicesBuffer);
            ctx.releaseBuffer(counterBuffer);

            if (is_max) {
                topk_max_i32_v5(data, count, k, out_values, out_indices);
            } else {
                topk_min_i32_v5(data, count, k, out_values, out_indices);
            }
            return;
        }

        // 确保缓冲区足够大
        if (total > est_candidates) {
            ctx.releaseBuffer(candidatesBuffer);
            ctx.releaseBuffer(candidateIndicesBuffer);
            candidatesBuffer = ctx.acquireBuffer(total * sizeof(int32_t));
            candidateIndicesBuffer = ctx.acquireBuffer(total * sizeof(uint32_t));
        }

        // 写入偏移
        id<MTLBuffer> offsetsBuffer = [ctx.device newBufferWithBytes:offsets.data()
                                                              length:numThreads * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];

        // 第二遍: 写入候选
        cmdBuffer = [ctx.commandQueue commandBuffer];
        encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.topkFilterWrite];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:offsetsBuffer offset:0 atIndex:1];
        [encoder setBuffer:candidatesBuffer offset:0 atIndex:2];
        [encoder setBuffer:candidateIndicesBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 在 CPU 上对候选进行最终排序 (候选数较少)
        int32_t* candidates = (int32_t*)[candidatesBuffer contents];
        uint32_t* candidateIndices = (uint32_t*)[candidateIndicesBuffer contents];

        // 创建索引排序
        std::vector<size_t> sorted_idx(total);
        for (size_t i = 0; i < total; i++) sorted_idx[i] = i;

        if (is_max) {
            std::partial_sort(sorted_idx.begin(),
                              sorted_idx.begin() + std::min(k, (size_t)total),
                              sorted_idx.end(),
                              [candidates](size_t a, size_t b) {
                                  return candidates[a] > candidates[b];
                              });
        } else {
            std::partial_sort(sorted_idx.begin(),
                              sorted_idx.begin() + std::min(k, (size_t)total),
                              sorted_idx.end(),
                              [candidates](size_t a, size_t b) {
                                  return candidates[a] < candidates[b];
                              });
        }

        // 复制结果
        size_t result_count = std::min(k, (size_t)total);
        for (size_t i = 0; i < result_count; i++) {
            out_values[i] = candidates[sorted_idx[i]];
            if (out_indices) {
                out_indices[i] = candidateIndices[sorted_idx[i]];
            }
        }

        // 释放资源
        ctx.releaseBuffer(countsBuffer);
        ctx.releaseBuffer(candidatesBuffer);
        ctx.releaseBuffer(candidateIndicesBuffer);
        ctx.releaseBuffer(counterBuffer);
    }
}

// GPU Bitonic Sort 方法 (大 K)
void topk_gpu_bitonic(const int32_t* data, size_t count, size_t k,
                       int32_t* out_values, uint32_t* out_indices, bool is_max) {
    auto& ctx = getContext();

    // 对于非常大的数据，仍然使用 CPU
    if (count > 10000000) {
        if (is_max) {
            topk_max_i32_v5(data, count, k, out_values, out_indices);
        } else {
            topk_min_i32_v5(data, count, k, out_values, out_indices);
        }
        return;
    }

    @autoreleasepool {
        // 复制数据到 GPU (需要修改)
        id<MTLBuffer> dataBuffer = [ctx.device newBufferWithBytes:data
                                                           length:count * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];

        // 分配索引缓冲区
        id<MTLBuffer> indicesBuffer = ctx.acquireBuffer(count * sizeof(uint32_t));

        // 初始化索引
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        uint32_t countU32 = static_cast<uint32_t>(count);
        [encoder setComputePipelineState:ctx.initIndices];
        [encoder setBuffer:indicesBuffer offset:0 atIndex:0];
        [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:1];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((NSUInteger)256,
            ctx.initIndices.maxTotalThreadsPerThreadgroup), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        // Bitonic Sort
        uint32_t n = static_cast<uint32_t>(count);
        uint32_t is_ascending = is_max ? 0u : 1u;  // max 需要降序

        // 计算需要的 stages
        uint32_t stages = 0;
        for (uint32_t temp = n - 1; temp > 0; temp >>= 1) stages++;

        for (uint32_t stage = 0; stage < stages; stage++) {
            for (int step = static_cast<int>(stage); step >= 0; step--) {
                encoder = [cmdBuffer computeCommandEncoder];
                [encoder setComputePipelineState:ctx.bitonicSortStep];
                [encoder setBuffer:dataBuffer offset:0 atIndex:0];
                [encoder setBuffer:indicesBuffer offset:0 atIndex:1];

                uint32_t stepU32 = static_cast<uint32_t>(step);
                uint32_t stageU32 = stage;
                [encoder setBytes:&stepU32 length:sizeof(stepU32) atIndex:2];
                [encoder setBytes:&stageU32 length:sizeof(stageU32) atIndex:3];
                [encoder setBytes:&countU32 length:sizeof(countU32) atIndex:4];
                [encoder setBytes:&is_ascending length:sizeof(is_ascending) atIndex:5];

                MTLSize sortGrid = MTLSizeMake(count / 2, 1, 1);
                [encoder dispatchThreads:sortGrid threadsPerThreadgroup:groupSize];
                [encoder endEncoding];
            }
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 复制 TopK 结果
        int32_t* sorted_data = (int32_t*)[dataBuffer contents];
        uint32_t* sorted_indices = (uint32_t*)[indicesBuffer contents];

        if (is_max) {
            // 降序，取前 K 个
            for (size_t i = 0; i < k && i < count; i++) {
                out_values[i] = sorted_data[i];
                if (out_indices) {
                    out_indices[i] = sorted_indices[i];
                }
            }
        } else {
            // 升序，取前 K 个
            for (size_t i = 0; i < k && i < count; i++) {
                out_values[i] = sorted_data[i];
                if (out_indices) {
                    out_indices[i] = sorted_indices[i];
                }
            }
        }

        ctx.releaseBuffer(indicesBuffer);
    }
}

} // anonymous namespace

// ============================================================================
// 公共 API
// ============================================================================

bool is_topk_gpu_available() {
    auto& ctx = getContext();
    return ctx.initialize() && ctx.bitonicSortStep != nil;
}

void topk_max_i32_v6(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices) {
    TopKConfigV6 config;
    topk_max_i32_v6_config(data, count, k, out_values, out_indices, config);
}

void topk_max_i32_v6_config(const int32_t* data, size_t count, size_t k,
                             int32_t* out_values, uint32_t* out_indices,
                             const TopKConfigV6& config) {
    if (count == 0 || k == 0 || !data || !out_values) return;
    if (k > count) k = count;

    bool is_page_aligned = uma::is_page_aligned(data);
    TopKStrategy strategy = selectStrategy(count, k, config.cardinality_hint,
                                           config.strategy, is_page_aligned);

    switch (strategy) {
        case TopKStrategy::CPU_HEAP:
            topk_max_i32_v3(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::CPU_SAMPLE:
            topk_max_i32_v4(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::CPU_COUNT:
            topk_max_i32_v5(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::GPU_FILTER:
            topk_gpu_filter(data, count, k, out_values, out_indices, true);
            break;

        case TopKStrategy::GPU_BITONIC:
            topk_gpu_bitonic(data, count, k, out_values, out_indices, true);
            break;

        default:
            topk_max_i32_v5(data, count, k, out_values, out_indices);
            break;
    }
}

void topk_min_i32_v6(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices) {
    if (count == 0 || k == 0 || !data || !out_values) return;
    if (k > count) k = count;

    TopKConfigV6 config;
    bool is_page_aligned = uma::is_page_aligned(data);
    TopKStrategy strategy = selectStrategy(count, k, config.cardinality_hint,
                                           config.strategy, is_page_aligned);

    switch (strategy) {
        case TopKStrategy::CPU_HEAP:
            topk_min_i32_v3(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::CPU_SAMPLE:
            topk_min_i32_v4(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::CPU_COUNT:
            topk_min_i32_v5(data, count, k, out_values, out_indices);
            break;

        case TopKStrategy::GPU_FILTER:
            topk_gpu_filter(data, count, k, out_values, out_indices, false);
            break;

        case TopKStrategy::GPU_BITONIC:
            topk_gpu_bitonic(data, count, k, out_values, out_indices, false);
            break;

        default:
            topk_min_i32_v5(data, count, k, out_values, out_indices);
            break;
    }
}

} // namespace sort
} // namespace thunderduck
