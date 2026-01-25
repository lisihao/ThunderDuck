/**
 * ThunderDuck - GPU Hash Join v5 Implementation
 *
 * 基于 SIGMOD'25 GFTR 模式的优化:
 * - Radix 分区实现顺序内存访问
 * - Threadgroup memory 缓存
 * - SIMD prefix sum 批量结果收集
 *
 * 策略选择:
 * - < 100K: 回退到 CPU (GPU 启动开销)
 * - 100K - 1M: 简单 GPU join (simple_hash_join kernel)
 * - > 1M: 分区 GPU join (partitioned_hash_join kernel)
 */

#include "thunderduck/join.h"
#include "thunderduck/memory.h"

#include <vector>
#include <numeric>
#include <cstring>
#include <memory>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#define THUNDERDUCK_HAS_METAL 1
#endif

namespace thunderduck {
namespace join {
namespace v4 {

#ifdef THUNDERDUCK_HAS_METAL

// ============================================================================
// 常量
// ============================================================================

constexpr uint32_t NUM_PARTITIONS = 256;
constexpr size_t GPU_SIMPLE_THRESHOLD = 1000000;  // < 1M 用简单模式
constexpr size_t THREADGROUP_SIZE = 256;
constexpr size_t MAX_BUILD_PER_PARTITION = 8192;  // 共享内存限制

// ============================================================================
// Metal 上下文 (v5 增强版)
// ============================================================================

class MetalContextV5 {
public:
    static MetalContextV5& instance() {
        static MetalContextV5 ctx;
        return ctx;
    }

    bool is_available() const { return device_ != nil && partitioned_join_pipeline_ != nil; }

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

    // Pipelines
    id<MTLComputePipelineState> histogram_pipeline() const { return histogram_pipeline_; }
    id<MTLComputePipelineState> scatter_pipeline() const { return scatter_pipeline_; }
    id<MTLComputePipelineState> partitioned_join_pipeline() const { return partitioned_join_pipeline_; }
    id<MTLComputePipelineState> simple_join_pipeline() const { return simple_join_pipeline_; }
    id<MTLComputePipelineState> generate_indices_pipeline() const { return generate_indices_pipeline_; }

private:
    MetalContextV5() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) return;

            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                device_ = nil;
                return;
            }

            // 编译新 shader
            NSError* error = nil;
            NSString* shader_path = @"src/gpu/shaders/partitioned_join.metal";
            NSString* shader_source = [NSString stringWithContentsOfFile:shader_path
                                                                encoding:NSUTF8StringEncoding
                                                                   error:&error];

            id<MTLLibrary> library = nil;
            if (shader_source) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                if (@available(macOS 15.0, *)) {
                    options.mathMode = MTLMathModeFast;
                } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = YES;
#pragma clang diagnostic pop
                }
                library = [device_ newLibraryWithSource:shader_source options:options error:&error];
            }

            if (!library) {
                NSLog(@"Failed to compile partitioned_join.metal: %@", error);
                return;
            }

            // 创建所有 pipeline
            auto create_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> func = [library newFunctionWithName:name];
                if (!func) {
                    NSLog(@"Function %@ not found", name);
                    return nil;
                }
                id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:func error:&error];
                if (!pipeline) {
                    NSLog(@"Failed to create pipeline for %@: %@", name, error);
                }
                return pipeline;
            };

            histogram_pipeline_ = create_pipeline(@"radix_histogram");
            scatter_pipeline_ = create_pipeline(@"radix_scatter");
            partitioned_join_pipeline_ = create_pipeline(@"partitioned_hash_join");
            simple_join_pipeline_ = create_pipeline(@"simple_hash_join");
            generate_indices_pipeline_ = create_pipeline(@"generate_indices");
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> histogram_pipeline_ = nil;
    id<MTLComputePipelineState> scatter_pipeline_ = nil;
    id<MTLComputePipelineState> partitioned_join_pipeline_ = nil;
    id<MTLComputePipelineState> simple_join_pipeline_ = nil;
    id<MTLComputePipelineState> generate_indices_pipeline_ = nil;
};

// ============================================================================
// GPU 工具函数
// ============================================================================

// CPU prefix sum (256 个元素，超快)
void cpu_prefix_sum(const uint32_t* input, uint32_t* output, size_t count) {
    output[0] = 0;
    for (size_t i = 1; i < count; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// ============================================================================
// 简单 GPU Join (小数据量)
// ============================================================================

size_t gpu_simple_join(
    MetalContextV5& ctx,
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinResult* result) {

    @autoreleasepool {
        id<MTLDevice> device = ctx.device();

        // 构建 CPU 哈希表
        size_t capacity = 16;
        while (capacity < build_count * 1.7) capacity *= 2;
        uint32_t mask = static_cast<uint32_t>(capacity - 1);

        std::vector<int32_t> ht_keys(capacity, INT32_MIN);
        std::vector<uint32_t> ht_indices(capacity);

        for (size_t i = 0; i < build_count; i++) {
            int32_t key = build_keys[i];
            uint32_t h = static_cast<uint32_t>(key);
            h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13;
            h *= 0xc2b2ae35; h ^= h >> 16;
            size_t idx = h & mask;

            while (ht_keys[idx] != INT32_MIN) {
                idx = (idx + 1) & mask;
            }
            ht_keys[idx] = key;
            ht_indices[idx] = static_cast<uint32_t>(i);
        }

        // 创建 Metal 缓冲区
        id<MTLBuffer> probe_buffer = [device newBufferWithBytes:probe_keys
                                                         length:probe_count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> ht_keys_buffer = [device newBufferWithBytes:ht_keys.data()
                                                           length:capacity * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> ht_indices_buffer = [device newBufferWithBytes:ht_indices.data()
                                                              length:capacity * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];

        size_t max_matches = std::max(build_count, probe_count) * 4;
        id<MTLBuffer> out_build_buffer = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_probe_buffer = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> counter_buffer = [device newBufferWithLength:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        *((uint32_t*)[counter_buffer contents]) = 0;

        uint32_t probe_count_u32 = static_cast<uint32_t>(probe_count);
        uint32_t max_matches_u32 = static_cast<uint32_t>(max_matches);

        // 执行 kernel
        id<MTLCommandBuffer> cmd = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ctx.simple_join_pipeline()];
        [enc setBuffer:probe_buffer offset:0 atIndex:0];
        [enc setBuffer:ht_keys_buffer offset:0 atIndex:1];
        [enc setBuffer:ht_indices_buffer offset:0 atIndex:2];
        [enc setBytes:&mask length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:out_build_buffer offset:0 atIndex:4];
        [enc setBuffer:out_probe_buffer offset:0 atIndex:5];
        [enc setBuffer:counter_buffer offset:0 atIndex:6];
        [enc setBytes:&max_matches_u32 length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:8];

        MTLSize grid = MTLSizeMake(probe_count, 1, 1);
        MTLSize group = MTLSizeMake(std::min((size_t)256, probe_count), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        // 读取结果
        uint32_t match_count = *((uint32_t*)[counter_buffer contents]);
        if (match_count > max_matches) match_count = static_cast<uint32_t>(max_matches);

        if (result->capacity < match_count) {
            grow_join_result(result, match_count);
        }
        std::memcpy(result->left_indices, [out_build_buffer contents], match_count * sizeof(uint32_t));
        std::memcpy(result->right_indices, [out_probe_buffer contents], match_count * sizeof(uint32_t));
        result->count = match_count;

        return match_count;
    }
}

// ============================================================================
// 分区 GPU Join (大数据量)
// ============================================================================

size_t gpu_partitioned_join(
    MetalContextV5& ctx,
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinResult* result) {

    @autoreleasepool {
        id<MTLDevice> device = ctx.device();
        id<MTLCommandQueue> queue = ctx.queue();

        // ========== 步骤 1: 计算直方图 ==========

        // 创建输入缓冲区
        id<MTLBuffer> build_keys_buf = [device newBufferWithBytes:build_keys
                                                           length:build_count * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_keys_buf = [device newBufferWithBytes:probe_keys
                                                           length:probe_count * sizeof(int32_t)
                                                          options:MTLResourceStorageModeShared];

        // 直方图缓冲区
        id<MTLBuffer> build_histogram = [device newBufferWithLength:NUM_PARTITIONS * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_histogram = [device newBufferWithLength:NUM_PARTITIONS * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        std::memset([build_histogram contents], 0, NUM_PARTITIONS * sizeof(uint32_t));
        std::memset([probe_histogram contents], 0, NUM_PARTITIONS * sizeof(uint32_t));

        uint32_t build_count_u32 = static_cast<uint32_t>(build_count);
        uint32_t probe_count_u32 = static_cast<uint32_t>(probe_count);

        // 执行直方图 kernel
        id<MTLCommandBuffer> cmd1 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];

        [enc1 setComputePipelineState:ctx.histogram_pipeline()];

        // Build histogram
        [enc1 setBuffer:build_keys_buf offset:0 atIndex:0];
        [enc1 setBuffer:build_histogram offset:0 atIndex:1];
        [enc1 setBytes:&build_count_u32 length:sizeof(uint32_t) atIndex:2];
        MTLSize grid1 = MTLSizeMake(build_count, 1, 1);
        MTLSize group1 = MTLSizeMake(std::min((size_t)256, build_count), 1, 1);
        [enc1 dispatchThreads:grid1 threadsPerThreadgroup:group1];

        // Probe histogram
        [enc1 setBuffer:probe_keys_buf offset:0 atIndex:0];
        [enc1 setBuffer:probe_histogram offset:0 atIndex:1];
        [enc1 setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:2];
        MTLSize grid2 = MTLSizeMake(probe_count, 1, 1);
        MTLSize group2 = MTLSizeMake(std::min((size_t)256, probe_count), 1, 1);
        [enc1 dispatchThreads:grid2 threadsPerThreadgroup:group2];

        [enc1 endEncoding];
        [cmd1 commit];
        [cmd1 waitUntilCompleted];

        // ========== 步骤 2: CPU Prefix Sum ==========

        uint32_t* build_hist_ptr = (uint32_t*)[build_histogram contents];
        uint32_t* probe_hist_ptr = (uint32_t*)[probe_histogram contents];

        std::vector<uint32_t> build_offsets(NUM_PARTITIONS);
        std::vector<uint32_t> probe_offsets(NUM_PARTITIONS);
        std::vector<uint32_t> build_sizes(NUM_PARTITIONS);
        std::vector<uint32_t> probe_sizes(NUM_PARTITIONS);

        cpu_prefix_sum(build_hist_ptr, build_offsets.data(), NUM_PARTITIONS);
        cpu_prefix_sum(probe_hist_ptr, probe_offsets.data(), NUM_PARTITIONS);

        for (size_t i = 0; i < NUM_PARTITIONS; i++) {
            build_sizes[i] = build_hist_ptr[i];
            probe_sizes[i] = probe_hist_ptr[i];
        }

        // ========== 步骤 3: 分区数据 (Scatter) ==========

        // 创建索引数组
        id<MTLBuffer> build_indices_buf = [device newBufferWithLength:build_count * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_indices_buf = [device newBufferWithLength:probe_count * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];

        // 生成索引
        id<MTLCommandBuffer> cmd_idx = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_idx = [cmd_idx computeCommandEncoder];
        [enc_idx setComputePipelineState:ctx.generate_indices_pipeline()];

        [enc_idx setBuffer:build_indices_buf offset:0 atIndex:0];
        [enc_idx setBytes:&build_count_u32 length:sizeof(uint32_t) atIndex:1];
        [enc_idx dispatchThreads:MTLSizeMake(build_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc_idx setBuffer:probe_indices_buf offset:0 atIndex:0];
        [enc_idx setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:1];
        [enc_idx dispatchThreads:MTLSizeMake(probe_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc_idx endEncoding];
        [cmd_idx commit];
        [cmd_idx waitUntilCompleted];

        // 分区后的缓冲区
        id<MTLBuffer> build_keys_partitioned = [device newBufferWithLength:build_count * sizeof(int32_t)
                                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> build_indices_partitioned = [device newBufferWithLength:build_count * sizeof(uint32_t)
                                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_keys_partitioned = [device newBufferWithLength:probe_count * sizeof(int32_t)
                                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_indices_partitioned = [device newBufferWithLength:probe_count * sizeof(uint32_t)
                                                                      options:MTLResourceStorageModeShared];

        // 分区计数器 (用于 scatter)
        id<MTLBuffer> build_counters = [device newBufferWithLength:NUM_PARTITIONS * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_counters = [device newBufferWithLength:NUM_PARTITIONS * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        std::memset([build_counters contents], 0, NUM_PARTITIONS * sizeof(uint32_t));
        std::memset([probe_counters contents], 0, NUM_PARTITIONS * sizeof(uint32_t));

        // Prefix sum 缓冲区
        id<MTLBuffer> build_offsets_buf = [device newBufferWithBytes:build_offsets.data()
                                                              length:NUM_PARTITIONS * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_offsets_buf = [device newBufferWithBytes:probe_offsets.data()
                                                              length:NUM_PARTITIONS * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];

        // 执行 scatter
        id<MTLCommandBuffer> cmd2 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];
        [enc2 setComputePipelineState:ctx.scatter_pipeline()];

        // Build scatter
        [enc2 setBuffer:build_keys_buf offset:0 atIndex:0];
        [enc2 setBuffer:build_indices_buf offset:0 atIndex:1];
        [enc2 setBuffer:build_offsets_buf offset:0 atIndex:2];
        [enc2 setBuffer:build_counters offset:0 atIndex:3];
        [enc2 setBuffer:build_keys_partitioned offset:0 atIndex:4];
        [enc2 setBuffer:build_indices_partitioned offset:0 atIndex:5];
        [enc2 setBytes:&build_count_u32 length:sizeof(uint32_t) atIndex:6];
        [enc2 dispatchThreads:MTLSizeMake(build_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // Probe scatter
        [enc2 setBuffer:probe_keys_buf offset:0 atIndex:0];
        [enc2 setBuffer:probe_indices_buf offset:0 atIndex:1];
        [enc2 setBuffer:probe_offsets_buf offset:0 atIndex:2];
        [enc2 setBuffer:probe_counters offset:0 atIndex:3];
        [enc2 setBuffer:probe_keys_partitioned offset:0 atIndex:4];
        [enc2 setBuffer:probe_indices_partitioned offset:0 atIndex:5];
        [enc2 setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:6];
        [enc2 dispatchThreads:MTLSizeMake(probe_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc2 endEncoding];
        [cmd2 commit];
        [cmd2 waitUntilCompleted];

        // ========== 步骤 4: 分区 Join ==========

        // 分区大小缓冲区
        id<MTLBuffer> build_sizes_buf = [device newBufferWithBytes:build_sizes.data()
                                                            length:NUM_PARTITIONS * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> probe_sizes_buf = [device newBufferWithBytes:probe_sizes.data()
                                                            length:NUM_PARTITIONS * sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        // 输出缓冲区
        size_t max_matches = std::max(build_count, probe_count) * 4;
        id<MTLBuffer> out_build = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_probe = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> match_counter = [device newBufferWithLength:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
        *((uint32_t*)[match_counter contents]) = 0;

        uint32_t max_matches_u32 = static_cast<uint32_t>(max_matches);

        // 计算最大分区大小用于 threadgroup memory
        uint32_t max_build_partition = 0;
        for (size_t i = 0; i < NUM_PARTITIONS; i++) {
            if (build_sizes[i] > max_build_partition) {
                max_build_partition = build_sizes[i];
            }
        }
        // 限制共享内存大小
        if (max_build_partition > MAX_BUILD_PER_PARTITION) {
            max_build_partition = MAX_BUILD_PER_PARTITION;
        }

        // 执行分区 join
        id<MTLCommandBuffer> cmd3 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc3 = [cmd3 computeCommandEncoder];
        [enc3 setComputePipelineState:ctx.partitioned_join_pipeline()];

        [enc3 setBuffer:build_keys_partitioned offset:0 atIndex:0];
        [enc3 setBuffer:build_indices_partitioned offset:0 atIndex:1];
        [enc3 setBuffer:probe_keys_partitioned offset:0 atIndex:2];
        [enc3 setBuffer:probe_indices_partitioned offset:0 atIndex:3];
        [enc3 setBuffer:build_offsets_buf offset:0 atIndex:4];
        [enc3 setBuffer:build_sizes_buf offset:0 atIndex:5];
        [enc3 setBuffer:probe_offsets_buf offset:0 atIndex:6];
        [enc3 setBuffer:probe_sizes_buf offset:0 atIndex:7];
        [enc3 setBuffer:out_build offset:0 atIndex:8];
        [enc3 setBuffer:out_probe offset:0 atIndex:9];
        [enc3 setBuffer:match_counter offset:0 atIndex:10];
        [enc3 setBytes:&max_matches_u32 length:sizeof(uint32_t) atIndex:11];

        // Threadgroup memory
        size_t keys_mem = max_build_partition * sizeof(int32_t);
        size_t indices_mem = max_build_partition * sizeof(uint32_t);
        [enc3 setThreadgroupMemoryLength:keys_mem atIndex:0];
        [enc3 setThreadgroupMemoryLength:indices_mem atIndex:1];

        // 每个 threadgroup 处理一个分区
        MTLSize grid3 = MTLSizeMake(THREADGROUP_SIZE * NUM_PARTITIONS, 1, 1);
        MTLSize group3 = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
        [enc3 dispatchThreads:grid3 threadsPerThreadgroup:group3];

        [enc3 endEncoding];
        [cmd3 commit];
        [cmd3 waitUntilCompleted];

        // ========== 步骤 5: 读取结果 ==========

        uint32_t match_count = *((uint32_t*)[match_counter contents]);
        if (match_count > max_matches) match_count = static_cast<uint32_t>(max_matches);

        if (result->capacity < match_count) {
            grow_join_result(result, match_count);
        }
        std::memcpy(result->left_indices, [out_build contents], match_count * sizeof(uint32_t));
        std::memcpy(result->right_indices, [out_probe contents], match_count * sizeof(uint32_t));
        result->count = match_count;

        return match_count;
    }
}

#endif // THUNDERDUCK_HAS_METAL

// ============================================================================
// 公开接口
// ============================================================================

bool is_gpu_v5_ready() {
#ifdef THUNDERDUCK_HAS_METAL
    return MetalContextV5::instance().is_available();
#else
    return false;
#endif
}

size_t hash_join_gpu_v5(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

#ifdef THUNDERDUCK_HAS_METAL
    MetalContextV5& ctx = MetalContextV5::instance();

    if (!ctx.is_available()) {
        // 回退到 CPU
        return hash_join_i32_v3(build_keys, build_count,
                                 probe_keys, probe_count,
                                 join_type, result);
    }

    // 策略选择
    size_t total = build_count + probe_count;

    if (total < GPU_SIMPLE_THRESHOLD) {
        // 小数据量: 简单 GPU join
        return gpu_simple_join(ctx, build_keys, build_count,
                               probe_keys, probe_count, result);
    } else {
        // 大数据量: 分区 GPU join
        return gpu_partitioned_join(ctx, build_keys, build_count,
                                     probe_keys, probe_count, result);
    }

#else
    return hash_join_i32_v3(build_keys, build_count,
                             probe_keys, probe_count,
                             join_type, result);
#endif
}

} // namespace v4
} // namespace join
} // namespace thunderduck
