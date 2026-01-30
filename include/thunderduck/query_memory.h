/**
 * ThunderDuck Query Memory Infrastructure
 *
 * 查询级内存管理基础设施:
 * - QueryArena: Bump allocator for query-scoped memory
 * - ChunkedDirectArray: 分块直接数组 (避免栈溢出)
 * - TypeLiftedColumn: 类型提升列 (scan-time cast)
 *
 * @version 53
 * @date 2026-01-29
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <memory>
#include <new>
#include <arm_neon.h>

namespace thunderduck {
namespace memory {

// ============================================================================
// QueryArena: Query-Scoped Bump Allocator
// ============================================================================

/**
 * QueryArena - 查询级 bump allocator
 *
 * 特点:
 * - 单次分配，查询结束统一释放
 * - Cache-line 对齐 (64 bytes)
 * - 支持 10MB-100MB 级结构
 * - 零碎片，O(1) 分配
 *
 * 使用模式:
 *   QueryArena arena(64 * 1024 * 1024);  // 64MB
 *   auto* data = arena.alloc<int32_t>(6000000);
 *   // ... use data ...
 *   // arena destructor releases all memory
 */
class QueryArena {
public:
    static constexpr size_t CACHE_LINE = 64;
    static constexpr size_t DEFAULT_SIZE = 64 * 1024 * 1024;  // 64MB

    explicit QueryArena(size_t capacity = DEFAULT_SIZE)
        : capacity_(capacity), used_(0) {
        // Allocate aligned memory
        base_ = static_cast<char*>(std::aligned_alloc(CACHE_LINE, capacity));
        if (!base_) {
            throw std::bad_alloc();
        }
    }

    ~QueryArena() {
        if (base_) {
            std::free(base_);
        }
    }

    // Non-copyable, movable
    QueryArena(const QueryArena&) = delete;
    QueryArena& operator=(const QueryArena&) = delete;

    QueryArena(QueryArena&& other) noexcept
        : base_(other.base_), capacity_(other.capacity_), used_(other.used_) {
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.used_ = 0;
    }

    QueryArena& operator=(QueryArena&& other) noexcept {
        if (this != &other) {
            if (base_) std::free(base_);
            base_ = other.base_;
            capacity_ = other.capacity_;
            used_ = other.used_;
            other.base_ = nullptr;
            other.capacity_ = 0;
            other.used_ = 0;
        }
        return *this;
    }

    /**
     * 分配内存 (cache-line aligned)
     */
    template<typename T>
    T* alloc(size_t count) {
        size_t bytes = count * sizeof(T);
        size_t aligned_bytes = (bytes + CACHE_LINE - 1) & ~(CACHE_LINE - 1);

        if (used_ + aligned_bytes > capacity_) {
            throw std::bad_alloc();
        }

        T* ptr = reinterpret_cast<T*>(base_ + used_);
        used_ += aligned_bytes;
        return ptr;
    }

    /**
     * 分配并初始化为零
     */
    template<typename T>
    T* calloc(size_t count) {
        T* ptr = alloc<T>(count);
        std::memset(ptr, 0, count * sizeof(T));
        return ptr;
    }

    /**
     * 分配并初始化为指定值
     */
    template<typename T>
    T* alloc_fill(size_t count, T value) {
        T* ptr = alloc<T>(count);
        for (size_t i = 0; i < count; ++i) {
            ptr[i] = value;
        }
        return ptr;
    }

    /**
     * 重置 arena (不释放内存，仅重置指针)
     */
    void reset() {
        used_ = 0;
    }

    /**
     * 统计信息
     */
    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }
    size_t available() const { return capacity_ - used_; }
    double usage_ratio() const { return static_cast<double>(used_) / capacity_; }

private:
    char* base_;
    size_t capacity_;
    size_t used_;
};

// ============================================================================
// ChunkedDirectArray: 分块直接数组
// ============================================================================

/**
 * ChunkedDirectArray - 分块版 DirectArrayJoin
 *
 * 解决问题:
 * - 避免 54MB 栈溢出
 * - TLB 友好 (每 chunk 256KB)
 * - Cache 友好 (64K entries per chunk)
 *
 * 关键设计:
 * - 只有 chunk 指针数组在栈上 (~8KB for 6M entries)
 * - 实际数据在 arena 或 heap
 * - O(1) 查找 (chunk index + offset)
 */
template<typename Value, size_t MaxSize = 6000001>
class ChunkedDirectArray {
public:
    static constexpr size_t CHUNK_SIZE = 65536;  // 64K entries per chunk
    static constexpr size_t NUM_CHUNKS = (MaxSize + CHUNK_SIZE - 1) / CHUNK_SIZE;
    static constexpr Value INVALID = static_cast<Value>(-1);

    struct Stats {
        size_t entries = 0;
        size_t chunks_used = 0;
        double build_time_ms = 0;
        size_t memory_bytes = 0;
    };

    /**
     * 构造函数 - 使用 arena 分配
     */
    explicit ChunkedDirectArray(QueryArena& arena)
        : arena_(&arena), owns_memory_(false) {
        // Allocate chunk pointers
        chunks_ = arena.alloc<Value*>(NUM_CHUNKS);
        valid_chunks_ = arena.calloc<bool>(NUM_CHUNKS);

        // Initialize all chunks to nullptr
        std::memset(chunks_, 0, NUM_CHUNKS * sizeof(Value*));
    }

    /**
     * 构造函数 - 自管理内存
     */
    ChunkedDirectArray()
        : arena_(nullptr), owns_memory_(true) {
        chunks_ = new Value*[NUM_CHUNKS]();
        valid_chunks_ = new bool[NUM_CHUNKS]();
    }

    ~ChunkedDirectArray() {
        if (owns_memory_) {
            for (size_t i = 0; i < NUM_CHUNKS; ++i) {
                delete[] chunks_[i];
            }
            delete[] chunks_;
            delete[] valid_chunks_;
        }
    }

    // Non-copyable
    ChunkedDirectArray(const ChunkedDirectArray&) = delete;
    ChunkedDirectArray& operator=(const ChunkedDirectArray&) = delete;

    /**
     * 设置单个值
     */
    void set(int32_t key, Value value) {
        if (key < 0 || static_cast<size_t>(key) >= MaxSize) return;

        size_t chunk_idx = key / CHUNK_SIZE;
        size_t offset = key % CHUNK_SIZE;

        ensure_chunk(chunk_idx);
        chunks_[chunk_idx][offset] = value;
    }

    /**
     * 批量构建
     */
    template<typename KeyIter, typename ValIter>
    Stats build(KeyIter keys_begin, KeyIter keys_end, ValIter vals_begin) {
        Stats stats;
        auto start = std::chrono::high_resolution_clock::now();

        auto kit = keys_begin;
        auto vit = vals_begin;

        while (kit != keys_end) {
            int32_t key = static_cast<int32_t>(*kit);
            if (key >= 0 && static_cast<size_t>(key) < MaxSize) {
                set(key, static_cast<Value>(*vit));
                stats.entries++;
            }
            ++kit;
            ++vit;
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.build_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Count used chunks
        for (size_t i = 0; i < NUM_CHUNKS; ++i) {
            if (valid_chunks_[i]) stats.chunks_used++;
        }

        stats.memory_bytes = stats.chunks_used * CHUNK_SIZE * sizeof(Value) +
                            NUM_CHUNKS * sizeof(Value*) +
                            NUM_CHUNKS * sizeof(bool);

        return stats;
    }

    /**
     * 查找 - O(1)
     */
    inline Value lookup(int32_t key) const {
        if (key < 0 || static_cast<size_t>(key) >= MaxSize) return INVALID;

        size_t chunk_idx = key / CHUNK_SIZE;
        if (!valid_chunks_[chunk_idx]) return INVALID;

        size_t offset = key % CHUNK_SIZE;
        return chunks_[chunk_idx][offset];
    }

    /**
     * 快速查找 (无边界检查)
     */
    inline Value lookup_fast(int32_t key) const {
        size_t chunk_idx = key / CHUNK_SIZE;
        size_t offset = key % CHUNK_SIZE;
        return valid_chunks_[chunk_idx] ? chunks_[chunk_idx][offset] : INVALID;
    }

    /**
     * 检查是否存在
     */
    inline bool contains(int32_t key) const {
        return lookup(key) != INVALID;
    }

private:
    void ensure_chunk(size_t chunk_idx) {
        if (valid_chunks_[chunk_idx]) return;

        if (arena_) {
            chunks_[chunk_idx] = arena_->alloc_fill<Value>(CHUNK_SIZE, INVALID);
        } else {
            chunks_[chunk_idx] = new Value[CHUNK_SIZE];
            std::fill_n(chunks_[chunk_idx], CHUNK_SIZE, INVALID);
        }
        valid_chunks_[chunk_idx] = true;
    }

    QueryArena* arena_;
    bool owns_memory_;
    Value** chunks_;
    bool* valid_chunks_;
};

// ============================================================================
// TypeLiftedColumn: 类型提升列
// ============================================================================

/**
 * TypeLiftedColumn - Scan-time 类型转换
 *
 * 解决问题:
 * - 避免 hot loop 内的 int64 → double 转换
 * - 一次转换，多次使用
 * - SIMD 友好的内存布局
 */
template<typename From, typename To>
class TypeLiftedColumn {
public:
    struct Stats {
        size_t rows = 0;
        double convert_time_ms = 0;
        size_t memory_bytes = 0;
    };

    /**
     * 从源列创建 (使用 arena)
     */
    static Stats lift(const From* src, size_t n, To* dst, To scale = To{1}) {
        Stats stats;
        stats.rows = n;
        stats.memory_bytes = n * sizeof(To);

        auto start = std::chrono::high_resolution_clock::now();

        // SIMD 批量转换
        size_t i = 0;

        // 处理主体 (4 元素一批)
        for (; i + 4 <= n; i += 4) {
            dst[i]   = static_cast<To>(src[i])   / scale;
            dst[i+1] = static_cast<To>(src[i+1]) / scale;
            dst[i+2] = static_cast<To>(src[i+2]) / scale;
            dst[i+3] = static_cast<To>(src[i+3]) / scale;
        }

        // 处理尾部
        for (; i < n; ++i) {
            dst[i] = static_cast<To>(src[i]) / scale;
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.convert_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return stats;
    }

    /**
     * 从源列创建 (分配新内存)
     */
    static std::pair<std::unique_ptr<To[]>, Stats> lift_alloc(
        const From* src, size_t n, To scale = To{1}
    ) {
        auto dst = std::make_unique<To[]>(n);
        auto stats = lift(src, n, dst.get(), scale);
        return {std::move(dst), stats};
    }

    /**
     * 从源列创建 (使用 arena)
     */
    static std::pair<To*, Stats> lift_arena(
        QueryArena& arena, const From* src, size_t n, To scale = To{1}
    ) {
        To* dst = arena.alloc<To>(n);
        auto stats = lift(src, n, dst, scale);
        return {dst, stats};
    }
};

// 常用类型别名
using Int64ToDouble = TypeLiftedColumn<int64_t, double>;
using Int32ToDouble = TypeLiftedColumn<int32_t, double>;

// ============================================================================
// QueryContext: 查询上下文 (组合以上组件)
// ============================================================================

/**
 * QueryContext - 单查询执行上下文
 *
 * 包含:
 * - Arena allocator
 * - Type-lifted columns cache
 * - 统计信息
 */
class QueryContext {
public:
    explicit QueryContext(size_t arena_size = QueryArena::DEFAULT_SIZE)
        : arena_(arena_size) {}

    QueryArena& arena() { return arena_; }
    const QueryArena& arena() const { return arena_; }

    /**
     * 类型提升 lineitem 数值列 (int64 x10000 → double)
     */
    struct LineitemLifted {
        double* l_extendedprice = nullptr;
        double* l_discount = nullptr;
        double* l_quantity = nullptr;
        size_t n = 0;
        double convert_time_ms = 0;
    };

    template<typename LineitemTable>
    LineitemLifted lift_lineitem(const LineitemTable& lineitem) {
        LineitemLifted result;
        result.n = lineitem.l_orderkey.size();

        auto start = std::chrono::high_resolution_clock::now();

        // 一次性分配所有内存
        result.l_extendedprice = arena_.alloc<double>(result.n);
        result.l_discount = arena_.alloc<double>(result.n);
        result.l_quantity = arena_.alloc<double>(result.n);

        // 批量转换 (SIMD 友好)
        constexpr double SCALE = 10000.0;
        size_t i = 0;

        for (; i + 4 <= result.n; i += 4) {
            result.l_extendedprice[i]   = lineitem.l_extendedprice[i]   / SCALE;
            result.l_extendedprice[i+1] = lineitem.l_extendedprice[i+1] / SCALE;
            result.l_extendedprice[i+2] = lineitem.l_extendedprice[i+2] / SCALE;
            result.l_extendedprice[i+3] = lineitem.l_extendedprice[i+3] / SCALE;

            result.l_discount[i]   = lineitem.l_discount[i]   / SCALE;
            result.l_discount[i+1] = lineitem.l_discount[i+1] / SCALE;
            result.l_discount[i+2] = lineitem.l_discount[i+2] / SCALE;
            result.l_discount[i+3] = lineitem.l_discount[i+3] / SCALE;

            result.l_quantity[i]   = lineitem.l_quantity[i]   / SCALE;
            result.l_quantity[i+1] = lineitem.l_quantity[i+1] / SCALE;
            result.l_quantity[i+2] = lineitem.l_quantity[i+2] / SCALE;
            result.l_quantity[i+3] = lineitem.l_quantity[i+3] / SCALE;
        }

        for (; i < result.n; ++i) {
            result.l_extendedprice[i] = lineitem.l_extendedprice[i] / SCALE;
            result.l_discount[i] = lineitem.l_discount[i] / SCALE;
            result.l_quantity[i] = lineitem.l_quantity[i] / SCALE;
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.convert_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return result;
    }

    void reset() {
        arena_.reset();
    }

private:
    QueryArena arena_;
};

} // namespace memory
} // namespace thunderduck
