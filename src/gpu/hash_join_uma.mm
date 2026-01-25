/**
 * ThunderDuck - UMA-Optimized GPU Hash Join
 *
 * 真正利用 Apple Silicon 统一内存架构:
 * - 零拷贝数据传输
 * - 缓冲区池复用
 * - 流水线化 kernel 执行
 * - 减少 CPU-GPU 同步
 */

#include "thunderduck/join.h"
#include "thunderduck/uma_memory.h"
#include "thunderduck/adaptive_strategy.h"

#include <vector>
#include <cstring>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace thunderduck {
namespace join {
namespace uma {

#ifdef __APPLE__

// ============================================================================
// 常量
// ============================================================================

constexpr uint32_t NUM_PARTITIONS = 256;
constexpr size_t THREADGROUP_SIZE = 256;
// 阈值现在由 adaptive_strategy 管理

// ============================================================================
// Metal 上下文 (UMA 优化版)
// ============================================================================

class MetalContextUMA {
public:
    static MetalContextUMA& instance() {
        static MetalContextUMA ctx;
        return ctx;
    }

    bool is_available() const {
        return device_ != nil && simple_join_pipeline_ != nil;
    }

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }
    id<MTLComputePipelineState> simple_join_pipeline() const { return simple_join_pipeline_; }
    id<MTLComputePipelineState> build_ht_pipeline() const { return build_ht_pipeline_; }

private:
    MetalContextUMA() {
        @autoreleasepool {
            // 从 UMA 管理器获取设备
            auto& mgr = thunderduck::uma::UMAMemoryManager::instance();
            device_ = (__bridge id<MTLDevice>)mgr.get_metal_device();

            if (!device_) return;

            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                device_ = nil;
                return;
            }

            // 编译优化版 shader
            NSError* error = nil;

            // 内联 shader 代码 - 优化版 (前缀和批量写入)
            NSString* shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

constant int32_t EMPTY_KEY = 0x80000000;

inline uint32_t hash_key(int32_t key) {
    uint32_t h = as_type<uint32_t>(key);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// 构建哈希表 kernel (GPU 并行构建)
kernel void build_hash_table(
    device const int32_t* keys [[buffer(0)]],
    device int32_t* ht_keys [[buffer(1)]],
    device uint32_t* ht_indices [[buffer(2)]],
    constant uint32_t& ht_mask [[buffer(3)]],
    constant uint32_t& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    int32_t key = keys[tid];
    uint32_t hash = hash_key(key);
    uint32_t idx = hash & ht_mask;

    // 线性探测 (使用原子操作)
    while (true) {
        int32_t expected = EMPTY_KEY;
        if (atomic_compare_exchange_weak_explicit(
                (device atomic_int*)&ht_keys[idx],
                &expected, key,
                memory_order_relaxed,
                memory_order_relaxed)) {
            ht_indices[idx] = tid;
            break;
        }
        idx = (idx + 1) & ht_mask;
    }
}

// ============================================================================
// 两阶段前缀和优化: 减少原子争用
// ============================================================================

// 阶段1: 计数每个线程的匹配数
kernel void probe_count_matches(
    device const int32_t* probe_keys [[buffer(0)]],
    device const int32_t* ht_keys [[buffer(1)]],
    constant uint32_t& ht_mask [[buffer(2)]],
    device uint32_t* match_counts [[buffer(3)]],  // 每线程匹配数
    constant uint32_t& probe_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= probe_count) {
        return;
    }

    int32_t key = probe_keys[tid];
    uint32_t hash = hash_key(key);
    uint32_t idx = hash & ht_mask;
    uint32_t count = 0;

    // 计数匹配
    for (uint i = 0; i < 64; i++) {
        int32_t ht_key = ht_keys[idx];
        if (ht_key == EMPTY_KEY) break;
        if (ht_key == key) count++;
        idx = (idx + 1) & ht_mask;
    }

    match_counts[tid] = count;
}

// 阶段2: Threadgroup 级别前缀和
kernel void prefix_sum_local(
    device uint32_t* data [[buffer(0)]],
    device uint32_t* block_sums [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]],
    threadgroup uint32_t* shared [[threadgroup(0)]]
) {
    uint global_id = tid;

    // 加载到共享内存
    shared[local_id] = (global_id < count) ? data[global_id] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch 扫描 - 上扫
    for (uint stride = 1; stride < group_size; stride *= 2) {
        uint idx = (local_id + 1) * stride * 2 - 1;
        if (idx < group_size) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 保存块总和并清零最后元素
    if (local_id == group_size - 1) {
        block_sums[group_id] = shared[local_id];
        shared[local_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 下扫
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        uint idx = (local_id + 1) * stride * 2 - 1;
        if (idx < group_size) {
            uint temp = shared[idx];
            shared[idx] += shared[idx - stride];
            shared[idx - stride] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写回
    if (global_id < count) {
        data[global_id] = shared[local_id];
    }
}

// 阶段3: 添加块偏移
kernel void prefix_sum_add_offset(
    device uint32_t* data [[buffer(0)]],
    device const uint32_t* block_offsets [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    if (tid < count && group_id > 0) {
        data[tid] += block_offsets[group_id - 1];
    }
}

// 阶段4: 使用预计算偏移写入结果
kernel void probe_write_results(
    device const int32_t* probe_keys [[buffer(0)]],
    device const int32_t* ht_keys [[buffer(1)]],
    device const uint32_t* ht_indices [[buffer(2)]],
    constant uint32_t& ht_mask [[buffer(3)]],
    device const uint32_t* write_offsets [[buffer(4)]],  // 前缀和结果
    device uint32_t* out_build [[buffer(5)]],
    device uint32_t* out_probe [[buffer(6)]],
    constant uint32_t& max_matches [[buffer(7)]],
    constant uint32_t& probe_count [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= probe_count) return;

    int32_t key = probe_keys[tid];
    uint32_t hash = hash_key(key);
    uint32_t idx = hash & ht_mask;
    uint32_t write_idx = write_offsets[tid];

    // 写入匹配结果
    for (uint i = 0; i < 64; i++) {
        int32_t ht_key = ht_keys[idx];
        if (ht_key == EMPTY_KEY) break;

        if (ht_key == key) {
            if (write_idx < max_matches) {
                out_build[write_idx] = ht_indices[idx];
                out_probe[write_idx] = tid;
                write_idx++;
            }
        }
        idx = (idx + 1) & ht_mask;
    }
}

// ============================================================================
// 兼容版: 优化的单 pass 探测 (使用 threadgroup 级别原子减少争用)
// ============================================================================
kernel void probe_hash_table(
    device const int32_t* probe_keys [[buffer(0)]],
    device const int32_t* ht_keys [[buffer(1)]],
    device const uint32_t* ht_indices [[buffer(2)]],
    constant uint32_t& ht_mask [[buffer(3)]],
    device uint32_t* out_build [[buffer(4)]],
    device uint32_t* out_probe [[buffer(5)]],
    device atomic_uint* match_counter [[buffer(6)]],
    constant uint32_t& max_matches [[buffer(7)]],
    constant uint32_t& probe_count [[buffer(8)]],
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    threadgroup uint32_t* tg_counts [[threadgroup(0)]]
) {
    // 本地缓存 (增加到16提高命中率)
    uint32_t local_build[16];
    uint32_t local_probe[16];
    uint local_count = 0;

    // 只有有效线程进行探测
    if (tid < probe_count) {
        int32_t key = probe_keys[tid];
        uint32_t hash = hash_key(key);
        uint32_t idx = hash & ht_mask;

        // 线性探测
        for (uint i = 0; i < 64; i++) {
            int32_t ht_key = ht_keys[idx];
            if (ht_key == EMPTY_KEY) break;

            if (ht_key == key) {
                if (local_count < 16) {
                    local_build[local_count] = ht_indices[idx];
                    local_probe[local_count] = tid;
                    local_count++;
                }
            }
            idx = (idx + 1) & ht_mask;
        }
    }

    // 所有线程都参与 threadgroup 前缀和 (关键修复!)
    tg_counts[local_id] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 快速前缀和 (适用于 256 线程)
    for (uint stride = 1; stride < group_size; stride *= 2) {
        uint val = 0;
        if (local_id >= stride) {
            val = tg_counts[local_id - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_counts[local_id] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 计算局部偏移
    uint local_offset = (local_id > 0) ? tg_counts[local_id - 1] : 0;
    uint tg_total = tg_counts[group_size - 1];

    // 只有最后一个线程执行全局原子操作
    threadgroup uint32_t tg_global_offset;
    if (local_id == group_size - 1) {
        tg_global_offset = atomic_fetch_add_explicit(
            match_counter, tg_total, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 写入结果 (只有有匹配的线程)
    if (local_count > 0) {
        uint global_offset = tg_global_offset + local_offset;
        for (uint i = 0; i < local_count; i++) {
            uint write_idx = global_offset + i;
            if (write_idx < max_matches) {
                out_build[write_idx] = local_build[i];
                out_probe[write_idx] = local_probe[i];
            }
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

            id<MTLLibrary> library = [device_ newLibraryWithSource:shader_source
                                                           options:options
                                                             error:&error];

            if (!library) {
                NSLog(@"UMA GPU: Failed to compile shader: %@", error);
                return;
            }

            // 创建 pipeline
            id<MTLFunction> build_func = [library newFunctionWithName:@"build_hash_table"];
            id<MTLFunction> probe_func = [library newFunctionWithName:@"probe_hash_table"];

            if (build_func) {
                build_ht_pipeline_ = [device_ newComputePipelineStateWithFunction:build_func
                                                                            error:&error];
            }
            if (probe_func) {
                simple_join_pipeline_ = [device_ newComputePipelineStateWithFunction:probe_func
                                                                               error:&error];
            }
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> simple_join_pipeline_ = nil;
    id<MTLComputePipelineState> build_ht_pipeline_ = nil;
};

// ============================================================================
// UMA 优化的 GPU Hash Join
// ============================================================================

size_t hash_join_gpu_uma(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {

    // 使用自适应策略选择器决定是否使用 GPU
    strategy::DataCharacteristics data_chars;
    data_chars.row_count = probe_count;
    data_chars.column_count = build_count;
    data_chars.element_size = sizeof(int32_t);
    data_chars.selectivity = -1.0f;
    data_chars.cardinality_ratio = -1.0f;
    data_chars.is_page_aligned = thunderduck::uma::is_page_aligned(probe_keys) &&
                                  thunderduck::uma::is_page_aligned(build_keys);

    auto executor = strategy::StrategySelector::instance().select(
        strategy::OperatorType::JOIN_HASH, data_chars);

    // 如果策略选择器建议使用 CPU，则回退到 v3
    if (executor == strategy::Executor::CPU_SIMD ||
        executor == strategy::Executor::CPU_SCALAR) {
        return hash_join_i32_v3(build_keys, build_count,
                                 probe_keys, probe_count,
                                 join_type, result);
    }

    MetalContextUMA& ctx = MetalContextUMA::instance();
    if (!ctx.is_available()) {
        return hash_join_i32_v3(build_keys, build_count,
                                 probe_keys, probe_count,
                                 join_type, result);
    }

    auto& mgr = thunderduck::uma::UMAMemoryManager::instance();

    @autoreleasepool {
        id<MTLDevice> device = ctx.device();
        id<MTLCommandQueue> queue = ctx.queue();

        // ========== 步骤 1: 尝试零拷贝包装输入数据 ==========
        // 如果用户数据页对齐，直接使用；否则拷贝一次

        thunderduck::uma::UMABuffer build_buf;
        thunderduck::uma::UMABuffer probe_buf;
        bool build_wrapped = false;
        bool probe_wrapped = false;

        // 尝试零拷贝包装 build_keys
        if (thunderduck::uma::is_page_aligned(build_keys)) {
            build_buf = mgr.wrap_external((void*)build_keys, build_count * sizeof(int32_t));
            build_wrapped = (build_buf.data != nullptr);
        }

        if (!build_wrapped) {
            // 需要拷贝 (但使用 UMA 缓冲区)
            build_buf = mgr.acquire_from_pool(build_count * sizeof(int32_t));
            std::memcpy(build_buf.data, build_keys, build_count * sizeof(int32_t));
        }

        // 尝试零拷贝包装 probe_keys
        if (thunderduck::uma::is_page_aligned(probe_keys)) {
            probe_buf = mgr.wrap_external((void*)probe_keys, probe_count * sizeof(int32_t));
            probe_wrapped = (probe_buf.data != nullptr);
        }

        if (!probe_wrapped) {
            probe_buf = mgr.acquire_from_pool(probe_count * sizeof(int32_t));
            std::memcpy(probe_buf.data, probe_keys, probe_count * sizeof(int32_t));
        }

        // ========== 步骤 2: 分配哈希表 (UMA 缓冲区) ==========
        size_t ht_capacity = 16;
        while (ht_capacity < build_count * 1.7) ht_capacity *= 2;
        uint32_t ht_mask = static_cast<uint32_t>(ht_capacity - 1);

        thunderduck::uma::UMABuffer ht_keys_buf = mgr.acquire_from_pool(ht_capacity * sizeof(int32_t));
        thunderduck::uma::UMABuffer ht_indices_buf = mgr.acquire_from_pool(ht_capacity * sizeof(uint32_t));

        // 初始化哈希表 (直接在 UMA 内存中)
        int32_t* ht_keys = ht_keys_buf.as<int32_t>();
        for (size_t i = 0; i < ht_capacity; i++) {
            ht_keys[i] = INT32_MIN;  // EMPTY_KEY
        }

        // ========== 步骤 3: 分配输出缓冲区 (UMA) ==========
        size_t max_matches = std::max(build_count, probe_count) * 4;
        thunderduck::uma::UMABuffer out_build_buf = mgr.acquire_from_pool(max_matches * sizeof(uint32_t));
        thunderduck::uma::UMABuffer out_probe_buf = mgr.acquire_from_pool(max_matches * sizeof(uint32_t));
        thunderduck::uma::UMABuffer counter_buf = mgr.acquire_from_pool(sizeof(uint32_t));

        *counter_buf.as<uint32_t>() = 0;

        // ========== 步骤 4: 获取 Metal 缓冲区视图 ==========
        id<MTLBuffer> mtl_build_keys = (__bridge id<MTLBuffer>)build_buf.metal_buffer;
        id<MTLBuffer> mtl_probe_keys = (__bridge id<MTLBuffer>)probe_buf.metal_buffer;
        id<MTLBuffer> mtl_ht_keys = (__bridge id<MTLBuffer>)ht_keys_buf.metal_buffer;
        id<MTLBuffer> mtl_ht_indices = (__bridge id<MTLBuffer>)ht_indices_buf.metal_buffer;
        id<MTLBuffer> mtl_out_build = (__bridge id<MTLBuffer>)out_build_buf.metal_buffer;
        id<MTLBuffer> mtl_out_probe = (__bridge id<MTLBuffer>)out_probe_buf.metal_buffer;
        id<MTLBuffer> mtl_counter = (__bridge id<MTLBuffer>)counter_buf.metal_buffer;

        // ========== 步骤 5: 使用 Shared Event 流水线执行 ==========
        id<MTLSharedEvent> event = [device newSharedEvent];
        uint64_t event_value = 0;

        uint32_t build_count_u32 = static_cast<uint32_t>(build_count);
        uint32_t probe_count_u32 = static_cast<uint32_t>(probe_count);
        uint32_t max_matches_u32 = static_cast<uint32_t>(max_matches);

        // Kernel 1: 构建哈希表 (GPU 并行)
        id<MTLCommandBuffer> cmd1 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];

        [enc1 setComputePipelineState:ctx.build_ht_pipeline()];
        [enc1 setBuffer:mtl_build_keys offset:0 atIndex:0];
        [enc1 setBuffer:mtl_ht_keys offset:0 atIndex:1];
        [enc1 setBuffer:mtl_ht_indices offset:0 atIndex:2];
        [enc1 setBytes:&ht_mask length:sizeof(uint32_t) atIndex:3];
        [enc1 setBytes:&build_count_u32 length:sizeof(uint32_t) atIndex:4];

        MTLSize grid1 = MTLSizeMake(build_count, 1, 1);
        MTLSize group1 = MTLSizeMake(std::min((size_t)THREADGROUP_SIZE, build_count), 1, 1);
        [enc1 dispatchThreads:grid1 threadsPerThreadgroup:group1];

        [enc1 endEncoding];
        [cmd1 encodeSignalEvent:event value:++event_value];
        [cmd1 commit];

        // Kernel 2: 探测哈希表 (等待 build 完成后开始)
        id<MTLCommandBuffer> cmd2 = [queue commandBuffer];
        [cmd2 encodeWaitForEvent:event value:event_value];

        id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];

        [enc2 setComputePipelineState:ctx.simple_join_pipeline()];
        [enc2 setBuffer:mtl_probe_keys offset:0 atIndex:0];
        [enc2 setBuffer:mtl_ht_keys offset:0 atIndex:1];
        [enc2 setBuffer:mtl_ht_indices offset:0 atIndex:2];
        [enc2 setBytes:&ht_mask length:sizeof(uint32_t) atIndex:3];
        [enc2 setBuffer:mtl_out_build offset:0 atIndex:4];
        [enc2 setBuffer:mtl_out_probe offset:0 atIndex:5];
        [enc2 setBuffer:mtl_counter offset:0 atIndex:6];
        [enc2 setBytes:&max_matches_u32 length:sizeof(uint32_t) atIndex:7];
        [enc2 setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:8];
        // 关键: 分配 threadgroup 内存用于前缀和优化
        [enc2 setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(uint32_t) atIndex:0];

        MTLSize grid2 = MTLSizeMake(probe_count, 1, 1);
        MTLSize group2 = MTLSizeMake(std::min((size_t)THREADGROUP_SIZE, probe_count), 1, 1);
        [enc2 dispatchThreads:grid2 threadsPerThreadgroup:group2];

        [enc2 endEncoding];
        [cmd2 commit];
        [cmd2 waitUntilCompleted];  // 只在最后等待

        // ========== 步骤 6: 直接读取结果 (零拷贝!) ==========
        uint32_t match_count = *counter_buf.as<uint32_t>();
        if (match_count > max_matches) match_count = static_cast<uint32_t>(max_matches);

        // 确保结果缓冲区足够大
        if (result->capacity < match_count) {
            grow_join_result(result, match_count);
        }

        // 这里仍需要拷贝到用户提供的 JoinResult
        // 如果用户使用 JoinResultUMA，则可完全零拷贝
        std::memcpy(result->left_indices, out_build_buf.data, match_count * sizeof(uint32_t));
        std::memcpy(result->right_indices, out_probe_buf.data, match_count * sizeof(uint32_t));
        result->count = match_count;

        // ========== 步骤 7: 归还缓冲区到池 ==========
        if (!build_wrapped) mgr.release_to_pool(build_buf);
        if (!probe_wrapped) mgr.release_to_pool(probe_buf);
        mgr.release_to_pool(ht_keys_buf);
        mgr.release_to_pool(ht_indices_buf);
        mgr.release_to_pool(out_build_buf);
        mgr.release_to_pool(out_probe_buf);
        mgr.release_to_pool(counter_buf);

        return match_count;
    }
}

// ============================================================================
// 完全零拷贝版本 (使用 JoinResultUMA)
// ============================================================================

size_t hash_join_gpu_uma_zerocopy(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type,
    thunderduck::uma::JoinResultUMA* result) {

    MetalContextUMA& ctx = MetalContextUMA::instance();
    if (!ctx.is_available() || !result) {
        return 0;
    }

    auto& mgr = thunderduck::uma::UMAMemoryManager::instance();

    @autoreleasepool {
        id<MTLDevice> device = ctx.device();
        id<MTLCommandQueue> queue = ctx.queue();

        // 尝试零拷贝包装
        thunderduck::uma::UMABuffer build_buf;
        thunderduck::uma::UMABuffer probe_buf;
        bool build_wrapped = false;
        bool probe_wrapped = false;

        if (thunderduck::uma::is_page_aligned(build_keys)) {
            build_buf = mgr.wrap_external((void*)build_keys, build_count * sizeof(int32_t));
            build_wrapped = (build_buf.data != nullptr);
        }
        if (!build_wrapped) {
            build_buf = mgr.acquire_from_pool(build_count * sizeof(int32_t));
            std::memcpy(build_buf.data, build_keys, build_count * sizeof(int32_t));
        }

        if (thunderduck::uma::is_page_aligned(probe_keys)) {
            probe_buf = mgr.wrap_external((void*)probe_keys, probe_count * sizeof(int32_t));
            probe_wrapped = (probe_buf.data != nullptr);
        }
        if (!probe_wrapped) {
            probe_buf = mgr.acquire_from_pool(probe_count * sizeof(int32_t));
            std::memcpy(probe_buf.data, probe_keys, probe_count * sizeof(int32_t));
        }

        // 哈希表
        size_t ht_capacity = 16;
        while (ht_capacity < build_count * 1.7) ht_capacity *= 2;
        uint32_t ht_mask = static_cast<uint32_t>(ht_capacity - 1);

        thunderduck::uma::UMABuffer ht_keys_buf = mgr.acquire_from_pool(ht_capacity * sizeof(int32_t));
        thunderduck::uma::UMABuffer ht_indices_buf = mgr.acquire_from_pool(ht_capacity * sizeof(uint32_t));

        int32_t* ht_keys = ht_keys_buf.as<int32_t>();
        for (size_t i = 0; i < ht_capacity; i++) {
            ht_keys[i] = INT32_MIN;
        }

        // 计数器
        thunderduck::uma::UMABuffer counter_buf = mgr.acquire_from_pool(sizeof(uint32_t));
        *counter_buf.as<uint32_t>() = 0;

        // 获取 Metal 缓冲区
        id<MTLBuffer> mtl_build_keys = (__bridge id<MTLBuffer>)build_buf.metal_buffer;
        id<MTLBuffer> mtl_probe_keys = (__bridge id<MTLBuffer>)probe_buf.metal_buffer;
        id<MTLBuffer> mtl_ht_keys = (__bridge id<MTLBuffer>)ht_keys_buf.metal_buffer;
        id<MTLBuffer> mtl_ht_indices = (__bridge id<MTLBuffer>)ht_indices_buf.metal_buffer;
        id<MTLBuffer> mtl_out_build = (__bridge id<MTLBuffer>)result->buffer.metal_buffer;
        id<MTLBuffer> mtl_out_probe = nil;  // 需要计算偏移
        id<MTLBuffer> mtl_counter = (__bridge id<MTLBuffer>)counter_buf.metal_buffer;

        uint32_t build_count_u32 = static_cast<uint32_t>(build_count);
        uint32_t probe_count_u32 = static_cast<uint32_t>(probe_count);
        uint32_t max_matches_u32 = static_cast<uint32_t>(result->capacity);

        // 流水线执行
        id<MTLSharedEvent> event = [device newSharedEvent];

        // Build
        id<MTLCommandBuffer> cmd1 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cmd1 computeCommandEncoder];

        [enc1 setComputePipelineState:ctx.build_ht_pipeline()];
        [enc1 setBuffer:mtl_build_keys offset:0 atIndex:0];
        [enc1 setBuffer:mtl_ht_keys offset:0 atIndex:1];
        [enc1 setBuffer:mtl_ht_indices offset:0 atIndex:2];
        [enc1 setBytes:&ht_mask length:sizeof(uint32_t) atIndex:3];
        [enc1 setBytes:&build_count_u32 length:sizeof(uint32_t) atIndex:4];

        [enc1 dispatchThreads:MTLSizeMake(build_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc1 endEncoding];
        [cmd1 encodeSignalEvent:event value:1];
        [cmd1 commit];

        // Probe (结果直接写入 JoinResultUMA 的 buffer)
        id<MTLCommandBuffer> cmd2 = [queue commandBuffer];
        [cmd2 encodeWaitForEvent:event value:1];

        id<MTLComputeCommandEncoder> enc2 = [cmd2 computeCommandEncoder];

        [enc2 setComputePipelineState:ctx.simple_join_pipeline()];
        [enc2 setBuffer:mtl_probe_keys offset:0 atIndex:0];
        [enc2 setBuffer:mtl_ht_keys offset:0 atIndex:1];
        [enc2 setBuffer:mtl_ht_indices offset:0 atIndex:2];
        [enc2 setBytes:&ht_mask length:sizeof(uint32_t) atIndex:3];
        [enc2 setBuffer:mtl_out_build offset:0 atIndex:4];
        [enc2 setBuffer:mtl_out_build offset:result->capacity * sizeof(uint32_t) atIndex:5];  // right_indices 偏移
        [enc2 setBuffer:mtl_counter offset:0 atIndex:6];
        [enc2 setBytes:&max_matches_u32 length:sizeof(uint32_t) atIndex:7];
        [enc2 setBytes:&probe_count_u32 length:sizeof(uint32_t) atIndex:8];
        // 关键: 分配 threadgroup 内存用于前缀和优化
        [enc2 setThreadgroupMemoryLength:THREADGROUP_SIZE * sizeof(uint32_t) atIndex:0];

        [enc2 dispatchThreads:MTLSizeMake(probe_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc2 endEncoding];
        [cmd2 commit];
        [cmd2 waitUntilCompleted];

        // 读取计数 (零拷贝 - 直接从 UMA 读取)
        result->count = *counter_buf.as<uint32_t>();
        if (result->count > result->capacity) {
            result->count = result->capacity;
        }

        // 释放临时缓冲区
        if (!build_wrapped) mgr.release_to_pool(build_buf);
        if (!probe_wrapped) mgr.release_to_pool(probe_buf);
        mgr.release_to_pool(ht_keys_buf);
        mgr.release_to_pool(ht_indices_buf);
        mgr.release_to_pool(counter_buf);

        return result->count;
    }
}

#else // !__APPLE__

size_t hash_join_gpu_uma(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config) {
    return hash_join_i32_v3(build_keys, build_count,
                             probe_keys, probe_count,
                             join_type, result);
}

#endif // __APPLE__

// ============================================================================
// 公开接口
// ============================================================================

bool is_uma_gpu_ready() {
#ifdef __APPLE__
    return MetalContextUMA::instance().is_available();
#else
    return false;
#endif
}

} // namespace uma
} // namespace join
} // namespace thunderduck
