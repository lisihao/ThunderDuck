/**
 * ThunderDuck - GPU INNER Join V19.1
 *
 * 核心优化: Metal GPU 两阶段并行
 *
 * 算法:
 * 1. CPU 构建哈希表 (存储 build 索引链表)
 * 2. Phase 1: GPU 并行计数每个 probe 键的匹配数
 * 3. CPU 计算前缀和，精确分配输出空间
 * 4. Phase 2: GPU 并行写入匹配对
 *
 * 目标: INNER Join 1.07x → 2.0x+
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/join.h"
#include <vector>
#include <cstring>
#include <numeric>

namespace thunderduck {
namespace join {

namespace {

// ============================================================================
// Metal 上下文
// ============================================================================

struct InnerJoinMetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> countKernel = nil;
    id<MTLComputePipelineState> writeKernel = nil;
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

// 哈希表槽位
struct HashSlot {
    int key;
    uint build_idx;   // 第一个 build 索引
    uint next_offset; // 链表下一个位置 (0 表示无)
};

// CRC32 哈希 (与 CPU 一致)
inline uint crc32_hash(int key) {
    uint h = as_type<uint>(key);
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Phase 1: 计数每个 probe 键的匹配数
kernel void inner_join_count(
    device const HashSlot* hash_table [[buffer(0)]],
    device const int* probe_keys [[buffer(1)]],
    device uint* match_counts [[buffer(2)]],          // 每个 probe 的匹配数
    constant HashTableParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.probe_count) return;

    int key = probe_keys[gid];
    uint h = crc32_hash(key);
    uint idx = h & params.mask;

    uint count = 0;

    // 线性探测找到键
    while (hash_table[idx].key != params.empty_key) {
        if (hash_table[idx].key == key) {
            // 遍历链表统计匹配数
            count = 1;
            uint next = hash_table[idx].next_offset;
            while (next != 0) {
                count++;
                next = hash_table[next].next_offset;
            }
            break;
        }
        idx = (idx + 1) & params.mask;
    }

    match_counts[gid] = count;
}

// Phase 2: 写入匹配对
kernel void inner_join_write(
    device const HashSlot* hash_table [[buffer(0)]],
    device const int* probe_keys [[buffer(1)]],
    device const uint* write_offsets [[buffer(2)]],   // 前缀和偏移
    device uint* out_build_indices [[buffer(3)]],
    device uint* out_probe_indices [[buffer(4)]],
    constant HashTableParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.probe_count) return;

    int key = probe_keys[gid];
    uint h = crc32_hash(key);
    uint idx = h & params.mask;
    uint write_pos = write_offsets[gid];

    // 线性探测找到键
    while (hash_table[idx].key != params.empty_key) {
        if (hash_table[idx].key == key) {
            // 写入第一个匹配
            out_build_indices[write_pos] = hash_table[idx].build_idx;
            out_probe_indices[write_pos] = gid;
            write_pos++;

            // 遍历链表写入其他匹配
            uint next = hash_table[idx].next_offset;
            while (next != 0) {
                out_build_indices[write_pos] = hash_table[next].build_idx;
                out_probe_indices[write_pos] = gid;
                write_pos++;
                next = hash_table[next].next_offset;
            }
            return;
        }
        idx = (idx + 1) & params.mask;
    }
}
)";

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            }

            library = [device newLibraryWithSource:shaderSource options:options error:&error];
            if (!library) {
                NSLog(@"INNER Join shader compilation failed: %@", error);
                initialized = true;
                return false;
            }

            auto createPipeline = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:
                    [NSString stringWithUTF8String:name]];
                if (!func) return nil;
                return [device newComputePipelineStateWithFunction:func error:&error];
            };

            countKernel = createPipeline("inner_join_count");
            writeKernel = createPipeline("inner_join_write");

            initialized = true;
            return countKernel != nil && writeKernel != nil;
        }
    }
};

InnerJoinMetalContext& getInnerJoinContext() {
    static InnerJoinMetalContext ctx;
    return ctx;
}

constexpr int32_t EMPTY_KEY = INT32_MIN;
constexpr size_t GPU_MIN_TOTAL = 500000;  // 总数据量 >= 500K 使用 GPU

// 哈希表槽位结构
struct HashSlot {
    int32_t key;
    uint32_t build_idx;
    uint32_t next_offset;  // 链表
};

// CPU 哈希函数 (与 GPU 一致)
inline uint32_t gpu_hash(int32_t key) {
    uint32_t h = static_cast<uint32_t>(key);
    h ^= h >> 16; h *= 0x85ebca6b;
    h ^= h >> 13; h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

} // anonymous namespace

// ============================================================================
// GPU INNER Join 实现
// ============================================================================

size_t inner_join_gpu(const int32_t* build_keys, size_t build_count,
                       const int32_t* probe_keys, size_t probe_count,
                       JoinResult* result) {
    if (!result || build_count == 0 || probe_count == 0) {
        if (result) result->count = 0;
        return 0;
    }

    // 小数据量使用 CPU V14
    if (build_count + probe_count < GPU_MIN_TOTAL) {
        return hash_join_i32_v14(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::INNER, result);
    }

    auto& ctx = getInnerJoinContext();
    if (!ctx.initialize() || !ctx.countKernel || !ctx.writeKernel) {
        return hash_join_i32_v14(build_keys, build_count, probe_keys, probe_count,
                                  JoinType::INNER, result);
    }

    @autoreleasepool {
        // ====================================================================
        // 1. CPU 构建哈希表 (支持链表处理重复键)
        // ====================================================================
        size_t capacity = 16;
        while (capacity < build_count * 2) capacity *= 2;  // 2x 容量支持链表
        size_t mask = capacity - 1;

        std::vector<HashSlot> hash_table(capacity, {EMPTY_KEY, 0, 0});
        std::vector<HashSlot> overflow;  // 溢出链表
        overflow.reserve(build_count / 4);

        for (size_t i = 0; i < build_count; ++i) {
            int32_t key = build_keys[i];
            uint32_t h = gpu_hash(key);
            size_t idx = h & mask;

            // 线性探测找空位或相同键
            while (hash_table[idx].key != EMPTY_KEY) {
                if (hash_table[idx].key == key) {
                    // 重复键: 添加到链表
                    HashSlot new_slot = {key, static_cast<uint32_t>(i), hash_table[idx].next_offset};
                    overflow.push_back(new_slot);
                    hash_table[idx].next_offset = capacity + overflow.size() - 1;
                    goto next_key;
                }
                idx = (idx + 1) & mask;
            }

            // 新键
            hash_table[idx] = {key, static_cast<uint32_t>(i), 0};
            next_key:;
        }

        // 合并哈希表和溢出链表
        size_t total_slots = capacity + overflow.size();
        std::vector<HashSlot> full_table(total_slots);
        std::memcpy(full_table.data(), hash_table.data(), capacity * sizeof(HashSlot));
        if (!overflow.empty()) {
            std::memcpy(full_table.data() + capacity, overflow.data(), overflow.size() * sizeof(HashSlot));
        }

        // ====================================================================
        // 2. 准备 Metal 缓冲区
        // ====================================================================
        id<MTLBuffer> htBuffer = [ctx.device newBufferWithBytes:full_table.data()
                                                          length:total_slots * sizeof(HashSlot)
                                                         options:MTLResourceStorageModeShared];

        id<MTLBuffer> probeBuffer = [ctx.device newBufferWithBytes:probe_keys
                                                            length:probe_count * sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];

        id<MTLBuffer> matchCountsBuffer = [ctx.device newBufferWithLength:probe_count * sizeof(uint32_t)
                                                                  options:MTLResourceStorageModeShared];

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

        // ====================================================================
        // Phase 1: GPU 计数
        // ====================================================================
        {
            id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

            [encoder setComputePipelineState:ctx.countKernel];
            [encoder setBuffer:htBuffer offset:0 atIndex:0];
            [encoder setBuffer:probeBuffer offset:0 atIndex:1];
            [encoder setBuffer:matchCountsBuffer offset:0 atIndex:2];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

            NSUInteger threadgroupSize = std::min((NSUInteger)256,
                ctx.countKernel.maxTotalThreadsPerThreadgroup);
            MTLSize gridSize = MTLSizeMake(probe_count, 1, 1);
            MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
            [encoder endEncoding];

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        // ====================================================================
        // CPU 计算前缀和
        // ====================================================================
        uint32_t* match_counts = (uint32_t*)[matchCountsBuffer contents];
        std::vector<uint32_t> write_offsets(probe_count + 1);
        write_offsets[0] = 0;
        for (size_t i = 0; i < probe_count; ++i) {
            write_offsets[i + 1] = write_offsets[i] + match_counts[i];
        }
        size_t total_matches = write_offsets[probe_count];

        if (total_matches == 0) {
            result->count = 0;
            return 0;
        }

        // 分配输出缓冲区
        if (result->capacity < total_matches) {
            grow_join_result(result, total_matches);
        }

        id<MTLBuffer> offsetsBuffer = [ctx.device newBufferWithBytes:write_offsets.data()
                                                               length:(probe_count + 1) * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];

        id<MTLBuffer> outBuildBuffer = [ctx.device newBufferWithLength:total_matches * sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> outProbeBuffer = [ctx.device newBufferWithLength:total_matches * sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];

        // ====================================================================
        // Phase 2: GPU 写入
        // ====================================================================
        {
            id<MTLCommandBuffer> cmdBuffer = [ctx.commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

            [encoder setComputePipelineState:ctx.writeKernel];
            [encoder setBuffer:htBuffer offset:0 atIndex:0];
            [encoder setBuffer:probeBuffer offset:0 atIndex:1];
            [encoder setBuffer:offsetsBuffer offset:0 atIndex:2];
            [encoder setBuffer:outBuildBuffer offset:0 atIndex:3];
            [encoder setBuffer:outProbeBuffer offset:0 atIndex:4];
            [encoder setBuffer:paramsBuffer offset:0 atIndex:5];

            NSUInteger threadgroupSize = std::min((NSUInteger)256,
                ctx.writeKernel.maxTotalThreadsPerThreadgroup);
            MTLSize gridSize = MTLSizeMake(probe_count, 1, 1);
            MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
            [encoder endEncoding];

            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        // ====================================================================
        // 复制结果
        // ====================================================================
        std::memcpy(result->left_indices, [outBuildBuffer contents], total_matches * sizeof(uint32_t));
        std::memcpy(result->right_indices, [outProbeBuffer contents], total_matches * sizeof(uint32_t));
        result->count = total_matches;

        return total_matches;
    }
}

bool is_inner_join_gpu_available() {
    auto& ctx = getInnerJoinContext();
    return ctx.initialize() && ctx.countKernel != nil && ctx.writeKernel != nil;
}

} // namespace join
} // namespace thunderduck
