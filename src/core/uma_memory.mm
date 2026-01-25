/**
 * ThunderDuck - UMA Memory Manager Implementation
 *
 * Apple Silicon 统一内存架构优化:
 * - 使用 MTLResourceStorageModeShared 实现零拷贝
 * - 缓冲区池复用减少分配开销
 * - 页对齐保证最佳性能
 */

#include "thunderduck/uma_memory.h"

#include <vector>
#include <map>
#include <mutex>
#include <algorithm>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

namespace thunderduck {
namespace uma {

// ============================================================================
// UMA 内存管理器实现
// ============================================================================

struct UMAMemoryManager::Impl {
#ifdef __APPLE__
    id<MTLDevice> device = nil;
#endif

    // 缓冲区池: 按大小排序
    std::multimap<size_t, UMABuffer> pool;
    std::mutex pool_mutex;

    // 统计
    Stats stats = {};

    Impl() {
#ifdef __APPLE__
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                NSLog(@"UMA: Failed to create Metal device");
            }
        }
#endif
    }

    ~Impl() {
        // 清空池
        for (auto& [size, buffer] : pool) {
            free_buffer(buffer);
        }
        pool.clear();

#ifdef __APPLE__
        device = nil;
#endif
    }

    bool is_available() const {
#ifdef __APPLE__
        return device != nil;
#else
        return false;
#endif
    }

    UMABuffer allocate_buffer(size_t size, size_t alignment) {
        UMABuffer buffer = {};
        buffer.size = size;
        buffer.owned = true;

#ifdef __APPLE__
        if (!device) return buffer;

        @autoreleasepool {
            // 对齐到页边界以获得最佳性能
            size_t aligned_size = align_to_page(size);

            // 创建共享模式缓冲区 (CPU/GPU 零拷贝)
            id<MTLBuffer> mtl_buffer = [device newBufferWithLength:aligned_size
                                                           options:MTLResourceStorageModeShared |
                                                                   MTLResourceCPUCacheModeWriteCombined];

            if (mtl_buffer) {
                buffer.data = [mtl_buffer contents];
                buffer.capacity = aligned_size;
                buffer.metal_buffer = (__bridge_retained void*)mtl_buffer;

                stats.total_allocated += aligned_size;
                stats.allocations++;
            }
        }
#endif

        return buffer;
    }

    UMABuffer wrap_external_memory(void* ptr, size_t size) {
        UMABuffer buffer = {};
        buffer.size = size;
        buffer.capacity = size;
        buffer.data = ptr;
        buffer.owned = false;

#ifdef __APPLE__
        if (!device || !ptr) return buffer;

        // 检查页对齐
        if (!is_page_aligned(ptr)) {
            NSLog(@"UMA: External memory is not page-aligned, cannot wrap");
            buffer.data = nullptr;
            return buffer;
        }

        @autoreleasepool {
            // 使用 newBufferWithBytesNoCopy 实现真正的零拷贝
            id<MTLBuffer> mtl_buffer = [device newBufferWithBytesNoCopy:ptr
                                                                 length:size
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];

            if (mtl_buffer) {
                buffer.metal_buffer = (__bridge_retained void*)mtl_buffer;
            } else {
                NSLog(@"UMA: Failed to wrap external memory");
                buffer.data = nullptr;
            }
        }
#endif

        return buffer;
    }

    void free_buffer(UMABuffer& buffer) {
        if (!buffer.data) return;

#ifdef __APPLE__
        if (buffer.metal_buffer) {
            @autoreleasepool {
                // 释放 Metal 缓冲区
                id<MTLBuffer> mtl_buffer = (__bridge_transfer id<MTLBuffer>)buffer.metal_buffer;
                mtl_buffer = nil;
            }

            if (buffer.owned) {
                stats.total_allocated -= buffer.capacity;
            }
        }
#endif

        buffer = {};
    }

    UMABuffer acquire_from_pool_impl(size_t min_size) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        // 查找足够大的缓冲区 (最小浪费)
        auto it = pool.lower_bound(min_size);

        // 允许最多 2x 浪费
        if (it != pool.end() && it->first <= min_size * 2) {
            UMABuffer buffer = it->second;
            buffer.size = min_size;
            pool.erase(it);

            stats.pool_hits++;
            stats.pool_size -= buffer.capacity;
            stats.pool_count--;

            return buffer;
        }

        // 池中没有合适的，分配新的
        return allocate_buffer(min_size, UMA_CACHE_LINE);
    }

    void release_to_pool_impl(UMABuffer& buffer) {
        if (!buffer.data || !buffer.owned) {
            free_buffer(buffer);
            return;
        }

        std::lock_guard<std::mutex> lock(pool_mutex);

        // 如果池太大，释放而不是缓存
        if (stats.pool_size > UMA_DEFAULT_POOL_SIZE) {
            free_buffer(buffer);
            return;
        }

        stats.pool_size += buffer.capacity;
        stats.pool_count++;
        pool.emplace(buffer.capacity, buffer);

        buffer = {};
    }

    void clear_pool_impl() {
        std::lock_guard<std::mutex> lock(pool_mutex);

        for (auto& [size, buffer] : pool) {
            free_buffer(buffer);
        }
        pool.clear();

        stats.pool_size = 0;
        stats.pool_count = 0;
    }
};

// ============================================================================
// 单例实现
// ============================================================================

UMAMemoryManager& UMAMemoryManager::instance() {
    static UMAMemoryManager mgr;
    return mgr;
}

UMAMemoryManager::UMAMemoryManager() : impl_(new Impl()) {}

UMAMemoryManager::~UMAMemoryManager() {
    delete impl_;
}

bool UMAMemoryManager::is_available() const {
    return impl_->is_available();
}

UMABuffer UMAMemoryManager::allocate(size_t size, size_t alignment) {
    return impl_->allocate_buffer(size, alignment);
}

UMABuffer UMAMemoryManager::wrap_external(void* ptr, size_t size) {
    return impl_->wrap_external_memory(ptr, size);
}

void UMAMemoryManager::free(UMABuffer& buffer) {
    impl_->free_buffer(buffer);
}

UMABuffer UMAMemoryManager::acquire_from_pool(size_t min_size) {
    return impl_->acquire_from_pool_impl(min_size);
}

void UMAMemoryManager::release_to_pool(UMABuffer& buffer) {
    impl_->release_to_pool_impl(buffer);
}

void UMAMemoryManager::clear_pool() {
    impl_->clear_pool_impl();
}

void* UMAMemoryManager::get_metal_device() const {
#ifdef __APPLE__
    return (__bridge void*)impl_->device;
#else
    return nullptr;
#endif
}

UMAMemoryManager::Stats UMAMemoryManager::get_stats() const {
    return impl_->stats;
}

// ============================================================================
// 零拷贝 JoinResult 实现
// ============================================================================

JoinResultUMA* JoinResultUMA::create(size_t capacity) {
    auto& mgr = UMAMemoryManager::instance();

    // 分配足够存储两个索引数组的空间
    size_t total_size = capacity * 2 * sizeof(uint32_t);
    UMABuffer buffer = mgr.acquire_from_pool(total_size);

    if (!buffer.data) {
        return nullptr;
    }

    auto* result = new JoinResultUMA();
    result->buffer = buffer;
    result->left_indices = buffer.as<uint32_t>();
    result->right_indices = result->left_indices + capacity;
    result->count = 0;
    result->capacity = capacity;

    return result;
}

void JoinResultUMA::destroy(JoinResultUMA* result) {
    if (result) {
        UMAMemoryManager::instance().release_to_pool(result->buffer);
        delete result;
    }
}

bool JoinResultUMA::grow(size_t new_capacity) {
    if (new_capacity <= capacity) {
        return true;
    }

    auto& mgr = UMAMemoryManager::instance();

    // 分配新缓冲区
    size_t new_size = new_capacity * 2 * sizeof(uint32_t);
    UMABuffer new_buffer = mgr.acquire_from_pool(new_size);

    if (!new_buffer.data) {
        return false;
    }

    // 复制现有数据
    uint32_t* new_left = new_buffer.as<uint32_t>();
    uint32_t* new_right = new_left + new_capacity;

    if (count > 0) {
        std::memcpy(new_left, left_indices, count * sizeof(uint32_t));
        std::memcpy(new_right, right_indices, count * sizeof(uint32_t));
    }

    // 释放旧缓冲区
    mgr.release_to_pool(buffer);

    // 更新指针
    buffer = new_buffer;
    left_indices = new_left;
    right_indices = new_right;
    capacity = new_capacity;

    return true;
}

} // namespace uma
} // namespace thunderduck
