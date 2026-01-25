/**
 * ThunderDuck - UMA Data Chunk Implementation
 *
 * 统一数据载体实现:
 * - 列式存储管理
 * - 选择向量延迟物化
 * - GPU Metal 缓冲区访问
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "thunderduck/uma_data_chunk.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace thunderduck {
namespace uma {

// ============================================================================
// UMADataChunk 实现
// ============================================================================

UMADataChunk::UMADataChunk()
    : selection_{nullptr, 0, 0, nullptr, false}
    , count_(0)
    , selected_count_(0)
    , capacity_(0) {
}

UMADataChunk::UMADataChunk(const std::vector<DataType>& types, size_t capacity)
    : selection_{nullptr, 0, 0, nullptr, false}
    , count_(0)
    , selected_count_(0)
    , capacity_(capacity) {

    auto& mgr = UMAMemoryManager::instance();

    // 为每个列分配 UMA 缓冲区
    for (const auto& type : types) {
        UMAColumn col;
        col.type = type;
        col.count = 0;

        size_t elem_size = get_type_size(type);
        if (elem_size > 0) {
            size_t buffer_size = capacity * elem_size;
            col.data = mgr.allocate(buffer_size);
        } else {
            // STRING 类型需要特殊处理
            col.data = {nullptr, 0, 0, nullptr, false};
        }

        // 默认不分配 NULL 位图 (按需分配)
        col.validity = {nullptr, 0, 0, nullptr, false};

        columns_.push_back(std::move(col));
    }
}

UMADataChunk::~UMADataChunk() {
    auto& mgr = UMAMemoryManager::instance();

    // 释放所有列的缓冲区
    for (auto& col : columns_) {
        if (col.data.owned) {
            mgr.free(col.data);
        }
        if (col.validity.owned) {
            mgr.free(col.validity);
        }
    }

    // 释放选择向量
    if (selection_.owned) {
        mgr.free(selection_);
    }
}

UMADataChunk::UMADataChunk(UMADataChunk&& other) noexcept
    : columns_(std::move(other.columns_))
    , selection_(other.selection_)
    , count_(other.count_)
    , selected_count_(other.selected_count_)
    , capacity_(other.capacity_) {

    // 清空源对象的选择向量
    other.selection_ = {nullptr, 0, 0, nullptr, false};
    other.count_ = 0;
    other.selected_count_ = 0;
    other.capacity_ = 0;
}

UMADataChunk& UMADataChunk::operator=(UMADataChunk&& other) noexcept {
    if (this != &other) {
        // 释放当前资源
        auto& mgr = UMAMemoryManager::instance();
        for (auto& col : columns_) {
            if (col.data.owned) {
                mgr.free(col.data);
            }
            if (col.validity.owned) {
                mgr.free(col.validity);
            }
        }
        if (selection_.owned) {
            mgr.free(selection_);
        }

        // 移动资源
        columns_ = std::move(other.columns_);
        selection_ = other.selection_;
        count_ = other.count_;
        selected_count_ = other.selected_count_;
        capacity_ = other.capacity_;

        // 清空源对象
        other.selection_ = {nullptr, 0, 0, nullptr, false};
        other.count_ = 0;
        other.selected_count_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

void UMADataChunk::add_column(DataType type, const std::string& name) {
    auto& mgr = UMAMemoryManager::instance();

    UMAColumn col;
    col.type = type;
    col.name = name;
    col.count = count_;

    size_t elem_size = get_type_size(type);
    if (elem_size > 0 && capacity_ > 0) {
        size_t buffer_size = capacity_ * elem_size;
        col.data = mgr.allocate(buffer_size);
    } else {
        col.data = {nullptr, 0, 0, nullptr, false};
    }

    col.validity = {nullptr, 0, 0, nullptr, false};

    columns_.push_back(std::move(col));
}

void UMADataChunk::set_selection(UMABuffer&& selection, size_t selected_count) {
    // 释放旧的选择向量
    if (selection_.owned) {
        UMAMemoryManager::instance().free(selection_);
    }

    selection_ = std::move(selection);
    selected_count_ = selected_count;
}

void UMADataChunk::clear_selection() {
    if (selection_.owned) {
        UMAMemoryManager::instance().free(selection_);
    }
    selection_ = {nullptr, 0, 0, nullptr, false};
    selected_count_ = 0;
}

UMADataChunk* UMADataChunk::materialize() const {
    if (!has_selection()) {
        // 没有选择向量，直接复制
        return from_raw_data_copy(
            [this]() {
                std::vector<DataType> types;
                for (const auto& col : columns_) {
                    types.push_back(col.type);
                }
                return types;
            }(),
            [this]() {
                std::vector<const void*> ptrs;
                for (const auto& col : columns_) {
                    ptrs.push_back(col.data.data);
                }
                return ptrs;
            }(),
            count_
        );
    }

    // 根据选择向量创建紧凑数据块
    auto& mgr = UMAMemoryManager::instance();
    std::vector<DataType> types;
    for (const auto& col : columns_) {
        types.push_back(col.type);
    }

    auto* result = new UMADataChunk(types, selected_count_);
    const uint32_t* sel = get_selection();

    // 复制选中的行
    for (size_t col_idx = 0; col_idx < columns_.size(); col_idx++) {
        const auto& src_col = columns_[col_idx];
        auto& dst_col = result->columns_[col_idx];

        size_t elem_size = get_type_size(src_col.type);
        if (elem_size == 0) continue;  // 跳过变长类型

        const uint8_t* src = static_cast<const uint8_t*>(src_col.data.data);
        uint8_t* dst = static_cast<uint8_t*>(dst_col.data.data);

        for (size_t i = 0; i < selected_count_; i++) {
            size_t src_idx = sel[i];
            std::memcpy(dst + i * elem_size, src + src_idx * elem_size, elem_size);
        }

        dst_col.count = selected_count_;

        // 复制 NULL 位图 (如果有)
        if (src_col.has_nulls()) {
            size_t validity_size = (selected_count_ + 7) / 8;
            dst_col.validity = mgr.allocate(validity_size);
            uint8_t* dst_bits = static_cast<uint8_t*>(dst_col.validity.data);
            std::memset(dst_bits, 0xFF, validity_size);  // 默认全有效

            for (size_t i = 0; i < selected_count_; i++) {
                if (src_col.is_null(sel[i])) {
                    dst_bits[i / 8] &= ~(1 << (i % 8));
                }
            }
        }
    }

    result->count_ = selected_count_;
    return result;
}

void UMADataChunk::set_size(size_t count) {
    if (count > capacity_) {
        // 需要扩容
        auto& mgr = UMAMemoryManager::instance();
        size_t new_capacity = std::max(count, capacity_ * 2);

        for (auto& col : columns_) {
            size_t elem_size = get_type_size(col.type);
            if (elem_size == 0) continue;

            size_t new_size = new_capacity * elem_size;
            UMABuffer new_buffer = mgr.allocate(new_size);

            // 复制旧数据
            if (col.data.data && count_ > 0) {
                std::memcpy(new_buffer.data, col.data.data, count_ * elem_size);
            }

            // 释放旧缓冲区
            if (col.data.owned) {
                mgr.free(col.data);
            }

            col.data = new_buffer;
        }

        capacity_ = new_capacity;
    }

    count_ = count;

    // 更新每列的 count
    for (auto& col : columns_) {
        col.count = count;
    }
}

void UMADataChunk::append(const UMADataChunk& other) {
    if (other.size() == 0) return;
    if (columns_.size() != other.columns_.size()) return;

    size_t old_count = count_;
    size_t append_count = other.size();
    set_size(old_count + append_count);

    // 复制数据
    if (other.has_selection()) {
        const uint32_t* sel = other.get_selection();
        for (size_t col_idx = 0; col_idx < columns_.size(); col_idx++) {
            const auto& src_col = other.columns_[col_idx];
            auto& dst_col = columns_[col_idx];

            size_t elem_size = get_type_size(src_col.type);
            if (elem_size == 0) continue;

            const uint8_t* src = static_cast<const uint8_t*>(src_col.data.data);
            uint8_t* dst = static_cast<uint8_t*>(dst_col.data.data);

            for (size_t i = 0; i < append_count; i++) {
                std::memcpy(
                    dst + (old_count + i) * elem_size,
                    src + sel[i] * elem_size,
                    elem_size
                );
            }
        }
    } else {
        for (size_t col_idx = 0; col_idx < columns_.size(); col_idx++) {
            const auto& src_col = other.columns_[col_idx];
            auto& dst_col = columns_[col_idx];

            size_t elem_size = get_type_size(src_col.type);
            if (elem_size == 0) continue;

            std::memcpy(
                static_cast<uint8_t*>(dst_col.data.data) + old_count * elem_size,
                src_col.data.data,
                append_count * elem_size
            );
        }
    }
}

void UMADataChunk::reset() {
    count_ = 0;
    selected_count_ = 0;

    // 清除选择向量
    clear_selection();

    // 重置每列的 count
    for (auto& col : columns_) {
        col.count = 0;
    }
}

void UMADataChunk::reference(const UMADataChunk& other) {
    // 释放当前资源
    auto& mgr = UMAMemoryManager::instance();
    for (auto& col : columns_) {
        if (col.data.owned) {
            mgr.free(col.data);
        }
        if (col.validity.owned) {
            mgr.free(col.validity);
        }
    }
    if (selection_.owned) {
        mgr.free(selection_);
    }

    // 引用其他数据块 (不拥有内存)
    columns_.clear();
    for (const auto& src_col : other.columns_) {
        UMAColumn col;
        col.type = src_col.type;
        col.name = src_col.name;
        col.count = src_col.count;
        col.data = src_col.data;
        col.data.owned = false;  // 不拥有
        col.validity = src_col.validity;
        col.validity.owned = false;
        columns_.push_back(std::move(col));
    }

    selection_ = other.selection_;
    selection_.owned = false;
    count_ = other.count_;
    selected_count_ = other.selected_count_;
    capacity_ = other.capacity_;
}

// ============================================================================
// 工厂方法
// ============================================================================

UMADataChunk* UMADataChunk::from_raw_data(
    const std::vector<DataType>& types,
    const std::vector<void*>& data_ptrs,
    size_t count
) {
    if (types.size() != data_ptrs.size()) {
        return nullptr;
    }

    auto& mgr = UMAMemoryManager::instance();
    auto* chunk = new UMADataChunk();
    chunk->capacity_ = count;
    chunk->count_ = count;

    for (size_t i = 0; i < types.size(); i++) {
        UMAColumn col;
        col.type = types[i];
        col.count = count;

        size_t elem_size = get_type_size(types[i]);
        if (elem_size > 0 && data_ptrs[i] != nullptr) {
            size_t size = count * elem_size;

            // 尝试零拷贝
            if (is_page_aligned(data_ptrs[i])) {
                col.data = mgr.wrap_external(data_ptrs[i], size);
            } else {
                // 需要拷贝
                col.data = mgr.allocate(size);
                std::memcpy(col.data.data, data_ptrs[i], size);
            }
        } else {
            col.data = {nullptr, 0, 0, nullptr, false};
        }

        col.validity = {nullptr, 0, 0, nullptr, false};
        chunk->columns_.push_back(std::move(col));
    }

    return chunk;
}

UMADataChunk* UMADataChunk::from_raw_data_copy(
    const std::vector<DataType>& types,
    const std::vector<const void*>& data_ptrs,
    size_t count
) {
    if (types.size() != data_ptrs.size()) {
        return nullptr;
    }

    auto& mgr = UMAMemoryManager::instance();
    auto* chunk = new UMADataChunk();
    chunk->capacity_ = count;
    chunk->count_ = count;

    for (size_t i = 0; i < types.size(); i++) {
        UMAColumn col;
        col.type = types[i];
        col.count = count;

        size_t elem_size = get_type_size(types[i]);
        if (elem_size > 0 && data_ptrs[i] != nullptr) {
            size_t size = count * elem_size;
            col.data = mgr.allocate(size);
            std::memcpy(col.data.data, data_ptrs[i], size);
        } else {
            col.data = {nullptr, 0, 0, nullptr, false};
        }

        col.validity = {nullptr, 0, 0, nullptr, false};
        chunk->columns_.push_back(std::move(col));
    }

    return chunk;
}

// ============================================================================
// 调试函数
// ============================================================================

void print_chunk_info(const UMADataChunk& chunk) {
    std::cout << "UMADataChunk:" << std::endl;
    std::cout << "  Columns: " << chunk.column_count() << std::endl;
    std::cout << "  Rows: " << chunk.size() << " (raw: " << chunk.raw_size() << ")" << std::endl;
    std::cout << "  Capacity: " << chunk.capacity() << std::endl;
    std::cout << "  Has selection: " << (chunk.has_selection() ? "yes" : "no") << std::endl;

    for (size_t i = 0; i < chunk.column_count(); i++) {
        const auto& col = chunk.get_column(i);
        const char* type_name = "unknown";
        switch (col.type) {
            case DataType::INT8: type_name = "INT8"; break;
            case DataType::INT16: type_name = "INT16"; break;
            case DataType::INT32: type_name = "INT32"; break;
            case DataType::INT64: type_name = "INT64"; break;
            case DataType::UINT8: type_name = "UINT8"; break;
            case DataType::UINT16: type_name = "UINT16"; break;
            case DataType::UINT32: type_name = "UINT32"; break;
            case DataType::UINT64: type_name = "UINT64"; break;
            case DataType::FLOAT32: type_name = "FLOAT32"; break;
            case DataType::FLOAT64: type_name = "FLOAT64"; break;
            case DataType::BOOL: type_name = "BOOL"; break;
            case DataType::STRING: type_name = "STRING"; break;
            case DataType::TIMESTAMP: type_name = "TIMESTAMP"; break;
            case DataType::DECIMAL: type_name = "DECIMAL"; break;
        }

        std::cout << "  Column " << i << ": "
                  << (col.name.empty() ? "(unnamed)" : col.name.c_str())
                  << " [" << type_name << "]"
                  << " count=" << col.count
                  << " has_nulls=" << (col.has_nulls() ? "yes" : "no")
                  << " metal_buffer=" << (col.get_metal_buffer() ? "yes" : "no")
                  << std::endl;
    }
}

} // namespace uma
} // namespace thunderduck
