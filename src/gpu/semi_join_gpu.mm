/**
 * ThunderDuck - GPU SEMI Join Implementation
 *
 * 核心优化: Metal GPU 并行探测
 *
 * 算法:
 * 1. CPU 构建哈希表
 * 2. 拷贝哈希表到 GPU
 * 3. GPU 每线程探测一个 probe 键
 * 4. 原子计数器收集匹配结果
 *
 * 目标: SEMI Join 0.86x → 1.0x+
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/join.h"
#include "thunderduck/uma_memory.h"
#include <vector>
#include <mutex>
#include <cstring>

// GPU SEMI Join 使用软件哈希 (不依赖 ARM CRC32)

namespace thunderduck {
namespace join {

namespace {

// ============================================================================
// Metal 上下文
// ============================================================================

struct SemiJoinMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> semiJoinKernel = nil;
    id<MTLComputePipelineState> antiJoinKernel = nil;
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
            NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// 哈希表参数
struct HashTableParams {
    uint capacity;
    uint mask;
    int empty_key;
    uint probe_count;
};

// CRC32 哈希 (ARM 兼容)
inline uint crc32_hash(int key) {
    uint h = as_type<uint>(key);
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// SEMI Join: 检查 probe 键是否存在于 build 表
kernel void semi_join_probe(
    device const int* hash_table_keys [[buffer(0)]],  // 哈希表键
    device const int* probe_keys [[buffer(1)]],       // 探测键
    device uint* out_indices [[buffer(2)]],           // 输出匹配索引
    device atomic_uint* counter [[buffer(3)]],        // 原子计数器
    constant HashTableParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.probe_count) return;

    int key = probe_keys[gid];
    uint h = crc32_hash(key);
    uint idx = h & params.mask;

    // 线性探测
    while (hash_table_keys[idx] != params.empty_key) {
        if (hash_table_keys[idx] == key) {
            // 找到匹配，原子写入
            uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            out_indices[pos] = gid;
            return;  // SEMI: 提前退出
        }
        idx = (idx + 1) & params.mask;
    }
}

// ANTI Join: 检查 probe 键是否不存在于 build 表
kernel void anti_join_probe(
    device const int* hash_table_keys [[buffer(0)]],
    device const int* probe_keys [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device atomic_uint* counter [[buffer(3)]],
    constant HashTableParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.probe_count) return;

    int key = probe_keys[gid];
    uint h = crc32_hash(key);
    uint idx = h & params.mask;

    // 线性探测
    while (hash_table_keys[idx] != params.empty_key) {
        if (hash_table_keys[idx] == key) {
            return;  // 找到匹配，ANTI 不输出
        }
        idx = (idx + 1) & params.mask;
    }

    // 未找到匹配，ANTI 输出
    uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    out_indices[pos] = gid;
}
)";

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            }

            library = [device newLibraryWithSource:shaderSource options:options error:&error];
            if (!library) {
                NSLog(@"SEMI Join shader compilation failed: %@", error);
                initialized = true;
                return false;
            }

            auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (!func) return nil;
                return [device newComputePipelineStateWithFunction:func error:&error];
            };

            semiJoinKernel = createPipeline("semi_join_probe");
            antiJoinKernel = createPipeline("anti_join_probe");

            initialized = true;
            return semiJoinKernel != nil;
        }
    }
};

SemiJoinMetalContext& getContext() {
    static SemiJoinMetalContext ctx;
    return ctx;
}

constexpr int32_t EMPTY_KEY = INT32_MIN;
constexpr size_t GPU_MIN_PROBE = 500000;  // 500K probe 以上使用 GPU

// CPU 哈希计算 - 必须与 GPU shader 完全一致
// 注意: 不使用硬件 CRC32，因为 GPU 无法使用
inline uint32_t gpu_hash(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

} // anonymous namespace

// ============================================================================
// GPU SEMI Join 实现
// ============================================================================

size_t semi_join_gpu(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result) {
    if (!result || build_count == 0 || probe_count == 0) {
        if (result) result->count = 0;
        return 0;
    }

    // 小数据量使用 CPU
    if (probe_count < GPU_MIN_PROBE) {
        return hash_join_i32_v10(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::SEMI, result);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.semiJoinKernel) {
        return hash_join_i32_v10(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::SEMI, result);
    }

    @autoreleasepool {
        // 1. CPU 构建哈希表
        size_t capacity = 16;
        while (capacity < build_count * 1.7) capacity *= 2;
        size_t mask = capacity - 1;

        std::vector<int32_t> ht_keys(capacity, EMPTY_KEY);
        for (size_t i = 0; i < build_count; ++i) {
            uint32_t h = gpu_hash(build_keys[i]);
            size_t idx = h & mask;
            while (ht_keys[idx] != EMPTY_KEY) idx = (idx + 1) & mask;
            ht_keys[idx] = build_keys[i];
        }

        // 2. 准备 Metal 缓冲区
        id<MTLBuffer> htBuffer = [ctx.device newBufferWithBytes:ht_keys.data()
                                                          length:capacity * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared];

        id<MTLBuffer> probeBuffer = nil;
        if (::thunderduck::uma::is_page_aligned(probe_keys)) {
            probeBuffer = [ctx.device newBufferWithBytesNoCopy:(void*)probe_keys
                                                         length:probe_count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        }
        if (!probeBuffer) {
            probeBuffer = [ctx.device newBufferWithBytes:probe_keys
                                                  length:probe_count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared];
        }

        // 确保输出缓冲区足够
        if (result->capacity < probe_count) {
            grow_join_result(result, probe_count);
        }

        id<MTLBuffer> outBuffer = [ctx.device newBufferWithBytesNoCopy:result->right_indices
                                                                 length:probe_count * sizeof(uint32_t)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!outBuffer) {
            outBuffer = [ctx.device newBufferWithLength:probe_count * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> counterBuffer = [ctx.device newBufferWithLength:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        *(uint32_t*)[counterBuffer contents] = 0;

        struct HashTableParams {
            uint32_t capacity;
            uint32_t mask;
            int32_t empty_key;
            uint32_t probe_count;
        };

        HashTableParams params = {
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(mask),
            EMPTY_KEY,
            static_cast<uint32_t>(probe_count)
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        // 3. 执行 GPU kernel
        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.semiJoinKernel];
        [encoder setBuffer:htBuffer offset:0 atIndex:0];
        [encoder setBuffer:probeBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuffer offset:0 atIndex:2];
        [encoder setBuffer:counterBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.semiJoinKernel.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(probe_count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // 4. 读取结果
        uint32_t match_count = *(uint32_t*)[counterBuffer contents];

        // 如果输出缓冲区不是直接映射，需要复制
        if ([outBuffer contents] != result->right_indices && match_count > 0) {
            std::memcpy(result->right_indices, [outBuffer contents],
                       match_count * sizeof(uint32_t));
        }

        // SEMI 只返回 probe 索引
        for (size_t i = 0; i < match_count; ++i) {
            result->left_indices[i] = NULL_INDEX;
        }

        result->count = match_count;
        return match_count;
    }
}

size_t anti_join_gpu(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result) {
    if (!result) return 0;

    if (build_count == 0) {
        // Build 为空，所有 probe 都是 ANTI 结果
        if (result->capacity < probe_count) {
            grow_join_result(result, probe_count);
        }
        for (size_t i = 0; i < probe_count; ++i) {
            result->right_indices[i] = static_cast<uint32_t>(i);
            result->left_indices[i] = NULL_INDEX;
        }
        result->count = probe_count;
        return probe_count;
    }

    // 小数据量使用 CPU
    if (probe_count < GPU_MIN_PROBE) {
        return hash_join_i32_v10(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::ANTI, result);
    }

    auto& ctx = getContext();
    if (!ctx.initialize() || !ctx.antiJoinKernel) {
        return hash_join_i32_v10(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::ANTI, result);
    }

    @autoreleasepool {
        // 类似 semi_join_gpu，但使用 antiJoinKernel
        size_t capacity = 16;
        while (capacity < build_count * 1.7) capacity *= 2;
        size_t mask = capacity - 1;

        std::vector<int32_t> ht_keys(capacity, EMPTY_KEY);
        for (size_t i = 0; i < build_count; ++i) {
            uint32_t h = gpu_hash(build_keys[i]);
            size_t idx = h & mask;
            while (ht_keys[idx] != EMPTY_KEY) idx = (idx + 1) & mask;
            ht_keys[idx] = build_keys[i];
        }

        id<MTLBuffer> htBuffer = [ctx.device newBufferWithBytes:ht_keys.data()
                                                          length:capacity * sizeof(int32_t)
                                                         options:MTLResourceStorageModeShared];

        id<MTLBuffer> probeBuffer = [ctx.device newBufferWithBytes:probe_keys
                                                            length:probe_count * sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];

        if (result->capacity < probe_count) {
            grow_join_result(result, probe_count);
        }

        id<MTLBuffer> outBuffer = [ctx.device newBufferWithLength:probe_count * sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];

        id<MTLBuffer> counterBuffer = [ctx.device newBufferWithLength:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        *(uint32_t*)[counterBuffer contents] = 0;

        struct HashTableParams {
            uint32_t capacity;
            uint32_t mask;
            int32_t empty_key;
            uint32_t probe_count;
        };

        HashTableParams params = {
            static_cast<uint32_t>(capacity),
            static_cast<uint32_t>(mask),
            EMPTY_KEY,
            static_cast<uint32_t>(probe_count)
        };

        id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params
                                                             length:sizeof(params)
                                                            options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:ctx.antiJoinKernel];
        [encoder setBuffer:htBuffer offset:0 atIndex:0];
        [encoder setBuffer:probeBuffer offset:0 atIndex:1];
        [encoder setBuffer:outBuffer offset:0 atIndex:2];
        [encoder setBuffer:counterBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        NSUInteger threadgroupSize = std::min((NSUInteger)256,
            ctx.antiJoinKernel.maxTotalThreadsPerThreadgroup);
        MTLSize gridSize = MTLSizeMake(probe_count, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        uint32_t match_count = *(uint32_t*)[counterBuffer contents];

        if (match_count > 0) {
            std::memcpy(result->right_indices, [outBuffer contents],
                       match_count * sizeof(uint32_t));
        }

        for (size_t i = 0; i < match_count; ++i) {
            result->left_indices[i] = NULL_INDEX;
        }

        result->count = match_count;
        return match_count;
    }
}

bool is_semi_join_gpu_available() {
    auto& ctx = getContext();
    return ctx.initialize() && ctx.semiJoinKernel != nil;
}

} // namespace join
} // namespace thunderduck
