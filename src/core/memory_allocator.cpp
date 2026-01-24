/**
 * ThunderDuck - Memory Allocator Implementation
 */

#include "thunderduck/memory.h"
#include <cstdlib>
#include <new>

#ifdef __APPLE__
#include <malloc/malloc.h>
#endif

namespace thunderduck {

void* aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }
    
    // 确保 alignment 是 2 的幂
    if ((alignment & (alignment - 1)) != 0) {
        return nullptr;
    }

    void* ptr = nullptr;
    
#ifdef __APPLE__
    // macOS 使用 posix_memalign
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
#else
    // 其他平台使用 std::aligned_alloc (C++17)
    // 注意：size 必须是 alignment 的倍数
    size_t aligned_size = align_size(size, alignment);
    ptr = std::aligned_alloc(alignment, aligned_size);
#endif

    return ptr;
}

void aligned_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// AlignedBuffer 实现

AlignedBuffer::AlignedBuffer(size_t size) : size_(size) {
    if (size > 0) {
        data_ = aligned_alloc(size, CACHE_LINE_SIZE);
        if (!data_) {
            throw std::bad_alloc();
        }
    }
}

AlignedBuffer::~AlignedBuffer() {
    aligned_free(data_);
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        aligned_free(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

} // namespace thunderduck
