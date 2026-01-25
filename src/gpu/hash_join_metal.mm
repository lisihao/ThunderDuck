/**
 * ThunderDuck - Metal GPU Hash Join Implementation
 *
 * 使用 Metal 进行 GPU 并行 Hash Join
 * - 统一内存零拷贝
 * - 每线程处理一个 probe key
 * - 原子计数器收集匹配结果
 */

#include "thunderduck/join.h"
#include "thunderduck/bloom_filter.h"
#include "thunderduck/memory.h"

#include <vector>
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

// ============================================================================
// Metal 上下文管理
// ============================================================================

#ifdef THUNDERDUCK_HAS_METAL

namespace {

// Metal 单例管理器
class MetalContext {
public:
    static MetalContext& instance() {
        static MetalContext ctx;
        return ctx;
    }

    bool is_available() const { return device_ != nil; }

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }
    id<MTLComputePipelineState> hash_join_pipeline() const { return hash_join_pipeline_; }
    id<MTLComputePipelineState> bloom_join_pipeline() const { return bloom_join_pipeline_; }

private:
    MetalContext() {
        @autoreleasepool {
            // 获取默认 Metal 设备
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                return;
            }

            // 创建命令队列
            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                device_ = nil;
                return;
            }

            // 编译着色器
            NSError* error = nil;

            // 从源代码编译 (开发时使用)
            // 生产环境应该使用预编译的 .metallib
            NSString* shader_path = @"src/gpu/shaders/hash_join.metal";
            NSString* shader_source = [NSString stringWithContentsOfFile:shader_path
                                                                encoding:NSUTF8StringEncoding
                                                                   error:&error];

            id<MTLLibrary> library = nil;

            if (shader_source) {
                // 从源代码编译
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                // Use mathMode for macOS 15.0+ compatibility
                if (@available(macOS 15.0, *)) {
                    options.mathMode = MTLMathModeFast;
                } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = YES;
#pragma clang diagnostic pop
                }

                library = [device_ newLibraryWithSource:shader_source
                                                options:options
                                                  error:&error];
            }

            // 如果源代码加载失败，尝试默认库
            if (!library) {
                library = [device_ newDefaultLibrary];
            }

            if (!library) {
                NSLog(@"Failed to create Metal library: %@", error);
                // 继续运行，只是没有 GPU 加速
                return;
            }

            // 创建计算管线
            id<MTLFunction> hash_join_func = [library newFunctionWithName:@"hash_join_probe"];
            if (hash_join_func) {
                hash_join_pipeline_ = [device_ newComputePipelineStateWithFunction:hash_join_func
                                                                             error:&error];
                if (!hash_join_pipeline_) {
                    NSLog(@"Failed to create hash_join pipeline: %@", error);
                }
            }

            id<MTLFunction> bloom_join_func = [library newFunctionWithName:@"bloom_hash_join_probe"];
            if (bloom_join_func) {
                bloom_join_pipeline_ = [device_ newComputePipelineStateWithFunction:bloom_join_func
                                                                              error:&error];
                if (!bloom_join_pipeline_) {
                    NSLog(@"Failed to create bloom_join pipeline: %@", error);
                }
            }
        }
    }

    ~MetalContext() {
        // ARC 会自动释放
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> hash_join_pipeline_ = nil;
    id<MTLComputePipelineState> bloom_join_pipeline_ = nil;
};

// GPU 哈希表 (在 GPU 内存中)
class GPUHashTable {
public:
    GPUHashTable(id<MTLDevice> device) : device_(device) {}

    bool build(const int32_t* keys, size_t count) {
        if (count == 0) return false;

        // 计算容量
        capacity_ = 16;
        while (capacity_ < count * 1.7) {
            capacity_ *= 2;
        }
        mask_ = static_cast<uint32_t>(capacity_ - 1);

        // 初始化空表
        std::vector<int32_t> ht_keys(capacity_, INT32_MIN);
        std::vector<uint32_t> ht_indices(capacity_);

        // CPU 构建哈希表
        for (size_t i = 0; i < count; ++i) {
            int32_t key = keys[i];
            uint32_t hash = hash_key(key);
            size_t idx = hash & mask_;

            while (ht_keys[idx] != INT32_MIN) {
                idx = (idx + 1) & mask_;
            }

            ht_keys[idx] = key;
            ht_indices[idx] = static_cast<uint32_t>(i);
        }

        // 创建 Metal 缓冲区 (统一内存)
        @autoreleasepool {
            keys_buffer_ = [device_ newBufferWithBytes:ht_keys.data()
                                               length:capacity_ * sizeof(int32_t)
                                              options:MTLResourceStorageModeShared];

            indices_buffer_ = [device_ newBufferWithBytes:ht_indices.data()
                                                   length:capacity_ * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        }

        return keys_buffer_ != nil && indices_buffer_ != nil;
    }

    id<MTLBuffer> keys_buffer() const { return keys_buffer_; }
    id<MTLBuffer> indices_buffer() const { return indices_buffer_; }
    uint32_t mask() const { return mask_; }

private:
    uint32_t hash_key(int32_t key) {
        uint32_t h = static_cast<uint32_t>(key);
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    id<MTLDevice> device_;
    id<MTLBuffer> keys_buffer_ = nil;
    id<MTLBuffer> indices_buffer_ = nil;
    size_t capacity_ = 0;
    uint32_t mask_ = 0;
};

} // anonymous namespace

#endif // THUNDERDUCK_HAS_METAL

// ============================================================================
// GPU 可用性检测
// ============================================================================

bool is_gpu_ready() {
#ifdef THUNDERDUCK_HAS_METAL
    return MetalContext::instance().is_available() &&
           MetalContext::instance().hash_join_pipeline() != nil;
#else
    return false;
#endif
}

// ============================================================================
// GPU Hash Join 实现
// ============================================================================

size_t hash_join_gpu(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

#ifdef THUNDERDUCK_HAS_METAL
    @autoreleasepool {
        MetalContext& ctx = MetalContext::instance();

        if (!ctx.is_available() || !ctx.hash_join_pipeline()) {
            // GPU 不可用，回退到 CPU
            return hash_join_i32_v3(build_keys, build_count,
                                     probe_keys, probe_count,
                                     join_type, result);
        }

        id<MTLDevice> device = ctx.device();
        id<MTLCommandQueue> queue = ctx.queue();
        id<MTLComputePipelineState> pipeline = ctx.hash_join_pipeline();

        // 1. 构建 GPU 哈希表
        GPUHashTable ht(device);
        if (!ht.build(build_keys, build_count)) {
            return hash_join_i32_v3(build_keys, build_count,
                                     probe_keys, probe_count,
                                     join_type, result);
        }

        // 2. 创建 probe keys 缓冲区
        id<MTLBuffer> probe_buffer = [device newBufferWithBytes:probe_keys
                                                         length:probe_count * sizeof(int32_t)
                                                        options:MTLResourceStorageModeShared];

        // 3. 创建输出缓冲区 (预估匹配数量)
        size_t max_matches = std::max(build_count, probe_count) * 4;
        id<MTLBuffer> out_build_buffer = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_probe_buffer = [device newBufferWithLength:max_matches * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];

        // 4. 创建原子计数器
        id<MTLBuffer> counter_buffer = [device newBufferWithLength:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        *((uint32_t*)[counter_buffer contents]) = 0;

        // 5. 创建常量缓冲区
        uint32_t ht_mask = ht.mask();
        uint32_t max_matches_u32 = static_cast<uint32_t>(max_matches);
        uint32_t probe_count_u32 = static_cast<uint32_t>(probe_count);

        // 6. 创建命令缓冲区
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:probe_buffer offset:0 atIndex:0];
        [encoder setBuffer:ht.keys_buffer() offset:0 atIndex:1];
        [encoder setBuffer:ht.indices_buffer() offset:0 atIndex:2];
        [encoder setBytes:&ht_mask length:sizeof(uint32_t) atIndex:3];
        [encoder setBuffer:out_build_buffer offset:0 atIndex:4];
        [encoder setBuffer:out_probe_buffer offset:0 atIndex:5];
        [encoder setBuffer:counter_buffer offset:0 atIndex:6];
        [encoder setBytes:&max_matches_u32 length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:8];

        // 7. 计算线程配置
        NSUInteger threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
        if (threads_per_group > 256) threads_per_group = 256;

        MTLSize grid_size = MTLSizeMake(probe_count, 1, 1);
        MTLSize group_size = MTLSizeMake(threads_per_group, 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        // 8. 提交并等待
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        // 9. 读取结果
        uint32_t match_count = *((uint32_t*)[counter_buffer contents]);

        if (match_count > max_matches) {
            match_count = static_cast<uint32_t>(max_matches);  // 截断
        }

        // 10. 复制结果到输出
        if (result->capacity < match_count) {
            grow_join_result(result, match_count);
        }

        std::memcpy(result->left_indices,
                   [out_build_buffer contents],
                   match_count * sizeof(uint32_t));
        std::memcpy(result->right_indices,
                   [out_probe_buffer contents],
                   match_count * sizeof(uint32_t));
        result->count = match_count;

        return match_count;
    }
#else
    // 没有 Metal 支持，回退到 CPU
    return hash_join_i32_v3(build_keys, build_count,
                             probe_keys, probe_count,
                             join_type, result);
#endif
}

} // namespace v4
} // namespace join
} // namespace thunderduck
