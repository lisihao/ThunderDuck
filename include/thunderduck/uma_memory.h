/**
 * ThunderDuck - UMA (Unified Memory Architecture) Memory Manager
 *
 * 专为 Apple Silicon 统一内存架构设计:
 * - CPU/GPU 共享同一物理内存
 * - 零拷贝数据传输
 * - 缓冲区池复用
 * - 页对齐分配
 */

#ifndef THUNDERDUCK_UMA_MEMORY_H
#define THUNDERDUCK_UMA_MEMORY_H

#include <cstdint>
#include <cstddef>

#ifdef __APPLE__
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declaration for C++ code
typedef void* MTLBufferRef;
typedef void* MTLDeviceRef;
#endif
#endif

namespace thunderduck {
namespace uma {

// ============================================================================
// 常量
// ============================================================================

constexpr size_t UMA_PAGE_SIZE = 16384;        // 16KB 页大小 (Apple Silicon)
constexpr size_t UMA_CACHE_LINE = 128;         // M4 缓存行大小
constexpr size_t UMA_DEFAULT_POOL_SIZE = 256 * 1024 * 1024;  // 256MB 默认池大小

// ============================================================================
// UMA 缓冲区
// ============================================================================

/**
 * UMA 缓冲区 - CPU/GPU 共享内存
 *
 * 特点:
 * - 单一指针同时可被 CPU 和 GPU 访问
 * - 无需任何数据拷贝
 * - 自动页对齐
 */
struct UMABuffer {
    void* data;              // CPU 可访问指针
    size_t size;             // 缓冲区大小
    size_t capacity;         // 分配的容量
    void* metal_buffer;      // Metal 缓冲区句柄 (id<MTLBuffer>)
    bool owned;              // 是否拥有内存 (vs 包装外部内存)

    // 类型安全访问
    template<typename T>
    T* as() { return static_cast<T*>(data); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data); }
};

// ============================================================================
// UMA 内存管理器
// ============================================================================

/**
 * UMA 内存管理器
 *
 * 功能:
 * - 分配 CPU/GPU 共享内存
 * - 缓冲区池管理
 * - 零拷贝包装外部内存
 */
class UMAMemoryManager {
public:
    /**
     * 获取单例实例
     */
    static UMAMemoryManager& instance();

    /**
     * 检查 UMA 是否可用
     */
    bool is_available() const;

    /**
     * 分配 UMA 缓冲区
     *
     * @param size 请求大小
     * @param alignment 对齐要求 (默认 128 字节)
     * @return UMA 缓冲区，失败返回空缓冲区
     */
    UMABuffer allocate(size_t size, size_t alignment = UMA_CACHE_LINE);

    /**
     * 零拷贝包装外部内存
     *
     * 注意: 外部内存必须页对齐 (16KB)，且在使用期间不能释放
     *
     * @param ptr 外部内存指针
     * @param size 内存大小
     * @return UMA 缓冲区，失败返回空缓冲区
     */
    UMABuffer wrap_external(void* ptr, size_t size);

    /**
     * 释放 UMA 缓冲区
     */
    void free(UMABuffer& buffer);

    /**
     * 从缓冲区池获取 (优先复用)
     *
     * @param min_size 最小大小
     * @return UMA 缓冲区
     */
    UMABuffer acquire_from_pool(size_t min_size);

    /**
     * 归还到缓冲区池 (供复用)
     */
    void release_to_pool(UMABuffer& buffer);

    /**
     * 清空缓冲区池
     */
    void clear_pool();

    /**
     * 获取 Metal 设备句柄
     */
    void* get_metal_device() const;

    /**
     * 获取统计信息
     */
    struct Stats {
        size_t total_allocated;     // 总分配量
        size_t pool_size;           // 池中缓冲区总大小
        size_t pool_count;          // 池中缓冲区数量
        size_t allocations;         // 分配次数
        size_t pool_hits;           // 池命中次数
    };
    Stats get_stats() const;

private:
    UMAMemoryManager();
    ~UMAMemoryManager();

    // 禁止拷贝
    UMAMemoryManager(const UMAMemoryManager&) = delete;
    UMAMemoryManager& operator=(const UMAMemoryManager&) = delete;

    struct Impl;
    Impl* impl_;
};

// ============================================================================
// 便捷函数
// ============================================================================

/**
 * 分配 UMA 内存 (便捷函数)
 */
inline UMABuffer uma_alloc(size_t size) {
    return UMAMemoryManager::instance().allocate(size);
}

/**
 * 释放 UMA 内存 (便捷函数)
 */
inline void uma_free(UMABuffer& buffer) {
    UMAMemoryManager::instance().free(buffer);
}

/**
 * 检查指针是否页对齐
 */
inline bool is_page_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % UMA_PAGE_SIZE) == 0;
}

/**
 * 对齐大小到页边界
 */
inline size_t align_to_page(size_t size) {
    return (size + UMA_PAGE_SIZE - 1) & ~(UMA_PAGE_SIZE - 1);
}

// ============================================================================
// 零拷贝 Join 结果
// ============================================================================

/**
 * 零拷贝 Join 结果
 *
 * 与标准 JoinResult 不同，数据直接存储在 UMA 缓冲区中
 * CPU 和 GPU 可以直接访问，无需任何拷贝
 */
struct JoinResultUMA {
    UMABuffer buffer;           // 底层 UMA 缓冲区
    uint32_t* left_indices;     // 指向 buffer 内部
    uint32_t* right_indices;    // 指向 buffer 内部
    size_t count;               // 当前匹配数
    size_t capacity;            // 最大容量

    /**
     * 创建零拷贝 Join 结果
     */
    static JoinResultUMA* create(size_t capacity);

    /**
     * 销毁
     */
    static void destroy(JoinResultUMA* result);

    /**
     * 扩容
     */
    bool grow(size_t new_capacity);
};

} // namespace uma
} // namespace thunderduck

#endif // THUNDERDUCK_UMA_MEMORY_H
