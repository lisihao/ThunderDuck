/**
 * ThunderDuck - UMA Data Chunk
 *
 * 统一数据载体，实现:
 * - CPU/GPU 共享访问
 * - 算子间零拷贝传递
 * - 列式存储布局
 * - 选择向量延迟物化
 */

#ifndef THUNDERDUCK_UMA_DATA_CHUNK_H
#define THUNDERDUCK_UMA_DATA_CHUNK_H

#include "thunderduck/uma_memory.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace thunderduck {
namespace uma {

// ============================================================================
// 数据类型
// ============================================================================

enum class DataType {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
    BOOL,
    STRING,     // 变长字符串 (特殊处理)
    TIMESTAMP,  // 64-bit 时间戳
    DECIMAL     // 128-bit 定点数
};

/**
 * 获取类型大小
 */
inline size_t get_type_size(DataType type) {
    switch (type) {
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::FLOAT32:
            return 4;
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::FLOAT64:
        case DataType::TIMESTAMP:
            return 8;
        case DataType::DECIMAL:
            return 16;
        case DataType::STRING:
            return 0;  // 变长
        default:
            return 0;
    }
}

// ============================================================================
// 列定义
// ============================================================================

/**
 * UMA 列
 *
 * 特点:
 * - 数据存储在 UMA 缓冲区
 * - CPU/GPU 可直接访问
 * - 支持 NULL 位图
 */
struct UMAColumn {
    DataType type;              // 数据类型
    UMABuffer data;             // 数据缓冲区
    UMABuffer validity;         // NULL 位图 (1=有效, 0=NULL)
    size_t count;               // 元素数量
    std::string name;           // 列名 (可选)

    // 类型安全访问
    template<typename T>
    T* as() { return static_cast<T*>(data.data); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(data.data); }

    // 检查是否有 NULL
    bool has_nulls() const { return validity.data != nullptr; }

    // 检查特定位置是否为 NULL
    bool is_null(size_t idx) const {
        if (!validity.data) return false;
        const uint8_t* bits = static_cast<const uint8_t*>(validity.data);
        return (bits[idx / 8] & (1 << (idx % 8))) == 0;
    }

    // 设置 NULL
    void set_null(size_t idx, bool is_null) {
        if (!validity.data) return;
        uint8_t* bits = static_cast<uint8_t*>(validity.data);
        if (is_null) {
            bits[idx / 8] &= ~(1 << (idx % 8));
        } else {
            bits[idx / 8] |= (1 << (idx % 8));
        }
    }

    // 获取 Metal 缓冲区
    void* get_metal_buffer() const { return data.metal_buffer; }
};

// ============================================================================
// UMA 数据块
// ============================================================================

/**
 * UMA 数据块 (类似 DuckDB DataChunk)
 *
 * 特点:
 * - 列式存储
 * - 所有数据在 UMA 中
 * - 支持选择向量 (延迟物化)
 * - 支持流式追加
 */
class UMADataChunk {
public:
    /**
     * 创建空数据块
     */
    UMADataChunk();

    /**
     * 创建指定 schema 的数据块
     */
    UMADataChunk(const std::vector<DataType>& types, size_t capacity = 2048);

    /**
     * 析构
     */
    ~UMADataChunk();

    // 禁止拷贝
    UMADataChunk(const UMADataChunk&) = delete;
    UMADataChunk& operator=(const UMADataChunk&) = delete;

    // 允许移动
    UMADataChunk(UMADataChunk&& other) noexcept;
    UMADataChunk& operator=(UMADataChunk&& other) noexcept;

    // ========== 列操作 ==========

    /**
     * 获取列数
     */
    size_t column_count() const { return columns_.size(); }

    /**
     * 获取行数 (考虑选择向量)
     */
    size_t size() const {
        return selection_.data ? selected_count_ : count_;
    }

    /**
     * 获取原始行数 (忽略选择向量)
     */
    size_t raw_size() const { return count_; }

    /**
     * 获取容量
     */
    size_t capacity() const { return capacity_; }

    /**
     * 获取列
     */
    UMAColumn& get_column(size_t idx) { return columns_[idx]; }
    const UMAColumn& get_column(size_t idx) const { return columns_[idx]; }

    /**
     * 类型安全获取列数据
     */
    template<typename T>
    T* get_column_data(size_t idx) {
        return columns_[idx].as<T>();
    }

    template<typename T>
    const T* get_column_data(size_t idx) const {
        return columns_[idx].as<T>();
    }

    /**
     * 添加列
     */
    void add_column(DataType type, const std::string& name = "");

    // ========== 选择向量 ==========

    /**
     * 检查是否有选择向量
     */
    bool has_selection() const { return selection_.data != nullptr; }

    /**
     * 获取选择向量
     */
    const uint32_t* get_selection() const {
        return static_cast<const uint32_t*>(selection_.data);
    }

    /**
     * 获取选中行数
     */
    size_t selected_count() const { return selected_count_; }

    /**
     * 设置选择向量
     */
    void set_selection(UMABuffer&& selection, size_t selected_count);

    /**
     * 清除选择向量
     */
    void clear_selection();

    /**
     * 物化选择向量 (创建新的紧凑数据块)
     */
    UMADataChunk* materialize() const;

    // ========== 数据操作 ==========

    /**
     * 设置行数
     */
    void set_size(size_t count);

    /**
     * 追加数据
     */
    void append(const UMADataChunk& other);

    /**
     * 重置 (清空数据但保留 schema)
     */
    void reset();

    /**
     * 引用其他数据块 (零拷贝)
     */
    void reference(const UMADataChunk& other);

    // ========== GPU 支持 ==========

    /**
     * 获取列的 Metal 缓冲区
     */
    void* get_metal_buffer(size_t col_idx) const {
        return columns_[col_idx].get_metal_buffer();
    }

    /**
     * 获取选择向量的 Metal 缓冲区
     */
    void* get_selection_metal_buffer() const {
        return selection_.metal_buffer;
    }

    // ========== 工厂方法 ==========

    /**
     * 从原始数据创建 (零拷贝，需要页对齐)
     */
    static UMADataChunk* from_raw_data(
        const std::vector<DataType>& types,
        const std::vector<void*>& data_ptrs,
        size_t count
    );

    /**
     * 从原始数据创建 (拷贝，不需要页对齐)
     */
    static UMADataChunk* from_raw_data_copy(
        const std::vector<DataType>& types,
        const std::vector<const void*>& data_ptrs,
        size_t count
    );

private:
    std::vector<UMAColumn> columns_;     // 列数据
    UMABuffer selection_;                // 选择向量
    size_t count_ = 0;                   // 原始行数
    size_t selected_count_ = 0;          // 选中行数
    size_t capacity_ = 0;                // 容量
};

// ============================================================================
// 便捷函数
// ============================================================================

/**
 * 创建单列数据块
 */
template<typename T>
UMADataChunk* create_single_column_chunk(const T* data, size_t count, DataType type) {
    std::vector<DataType> types = {type};
    auto* chunk = new UMADataChunk(types, count);

    auto& mgr = UMAMemoryManager::instance();
    size_t size = count * sizeof(T);

    // 尝试零拷贝
    if (is_page_aligned(data)) {
        chunk->get_column(0).data = mgr.wrap_external((void*)data, size);
    } else {
        chunk->get_column(0).data = mgr.allocate(size);
        std::memcpy(chunk->get_column(0).data.data, data, size);
    }

    chunk->get_column(0).count = count;
    chunk->set_size(count);

    return chunk;
}

/**
 * 打印数据块信息
 */
void print_chunk_info(const UMADataChunk& chunk);

} // namespace uma
} // namespace thunderduck

#endif // THUNDERDUCK_UMA_DATA_CHUNK_H
