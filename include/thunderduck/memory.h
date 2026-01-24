/**
 * ThunderDuck - Memory Management
 * 
 * 128 字节对齐内存分配器，优化 M4 缓存行访问
 */

#ifndef THUNDERDUCK_MEMORY_H
#define THUNDERDUCK_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <memory>

namespace thunderduck {

// M4 缓存行大小
constexpr size_t CACHE_LINE_SIZE = 128;

// SIMD 向量对齐（128-bit Neon）
constexpr size_t SIMD_ALIGNMENT = 16;

/**
 * 对齐内存分配
 * @param size 分配大小
 * @param alignment 对齐边界（默认 128 字节）
 * @return 对齐的内存指针，失败返回 nullptr
 */
void* aligned_alloc(size_t size, size_t alignment = CACHE_LINE_SIZE);

/**
 * 释放对齐内存
 */
void aligned_free(void* ptr);

/**
 * 检查指针是否对齐
 */
inline bool is_aligned(const void* ptr, size_t alignment = CACHE_LINE_SIZE) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * 计算对齐后的大小
 */
inline size_t align_size(size_t size, size_t alignment = CACHE_LINE_SIZE) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * 对齐分配器 - 用于 STL 容器
 */
template <typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        void* ptr = aligned_alloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        aligned_free(p);
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

/**
 * 对齐的 unique_ptr 删除器
 */
struct AlignedDeleter {
    void operator()(void* ptr) const {
        aligned_free(ptr);
    }
};

/**
 * 创建对齐的 unique_ptr
 */
template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter>;

template <typename T>
AlignedUniquePtr<T> make_aligned(size_t count) {
    return AlignedUniquePtr<T>(
        static_cast<T*>(aligned_alloc(count * sizeof(T), CACHE_LINE_SIZE))
    );
}

/**
 * 数据块 - 对齐的内存缓冲区
 */
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t size);
    ~AlignedBuffer();

    // 禁止拷贝
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // 允许移动
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

    template <typename T>
    T* as() { return static_cast<T*>(data_); }

    template <typename T>
    const T* as() const { return static_cast<const T*>(data_); }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace thunderduck

#endif // THUNDERDUCK_MEMORY_H
