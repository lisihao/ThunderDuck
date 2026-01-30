/**
 * ThunderDuck NPU Abstraction Layer - Tensor Interface
 *
 * @file tensor.h
 * @version V20
 * @date 2026-01-27
 *
 * 统一的多维数组表示，支持 UMA 零拷贝
 */

#ifndef THUNDERDUCK_NPU_TENSOR_H
#define THUNDERDUCK_NPU_TENSOR_H

#include "device.h"
#include <cstddef>
#include <memory>
#include <vector>
#include <initializer_list>

namespace thunderduck {

// 前向声明
namespace uma {
    struct UMABuffer;
}

namespace npu {

using accelerator::DataType;
using accelerator::DeviceType;

//=============================================================================
// 张量描述
//=============================================================================

/**
 * 张量描述符
 */
struct TensorDesc {
    DataType dtype = DataType::FP32;    ///< 数据类型
    std::vector<size_t> shape;          ///< 形状 [dim0, dim1, ...]
    std::vector<size_t> strides;        ///< 步长 (可选，默认紧密排列)

    TensorDesc() = default;

    TensorDesc(DataType dt, std::initializer_list<size_t> dims)
        : dtype(dt), shape(dims) {}

    TensorDesc(DataType dt, const std::vector<size_t>& dims)
        : dtype(dt), shape(dims) {}

    /**
     * 获取维度数
     */
    size_t ndim() const { return shape.size(); }

    /**
     * 获取元素总数
     */
    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    /**
     * 获取字节大小
     */
    size_t size_bytes() const {
        return numel() * accelerator::dtype_size(dtype);
    }

    /**
     * 检查是否紧密排列
     */
    bool is_contiguous() const {
        if (strides.empty()) return true;
        size_t expected = accelerator::dtype_size(dtype);
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }
};

//=============================================================================
// 张量类
//=============================================================================

/**
 * 张量 - 统一的多维数组表示
 *
 * 特点:
 * - 支持 UMA 零拷贝
 * - 支持多种数据类型
 * - 引用计数内存管理
 * - 可从外部指针包装 (不拥有内存)
 */
class Tensor {
public:
    //=========================================================================
    // 构造与销毁
    //=========================================================================

    /**
     * 默认构造空张量
     */
    Tensor();

    /**
     * 从描述符构造 (分配内存)
     */
    explicit Tensor(const TensorDesc& desc);

    /**
     * 从类型和形状构造
     */
    Tensor(DataType dtype, std::initializer_list<size_t> shape);

    /**
     * 拷贝构造 (共享数据)
     */
    Tensor(const Tensor& other);

    /**
     * 移动构造
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * 拷贝赋值
     */
    Tensor& operator=(const Tensor& other);

    /**
     * 移动赋值
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * 析构
     */
    ~Tensor();

    //=========================================================================
    // 静态工厂方法
    //=========================================================================

    /**
     * 从现有内存创建 (零拷贝，不拥有内存)
     * @param data 数据指针
     * @param desc 张量描述
     * @return 包装的张量
     */
    static Tensor wrap(void* data, const TensorDesc& desc);

    /**
     * 从 const 内存创建只读张量
     */
    static Tensor wrap(const void* data, const TensorDesc& desc);

    /**
     * 从 UMA 缓冲区创建 (零拷贝，CPU/GPU 共享)
     * @param buffer UMA 缓冲区
     * @param desc 张量描述
     * @return UMA 张量
     */
    static Tensor wrap_uma(uma::UMABuffer& buffer, const TensorDesc& desc);

    /**
     * 创建全零张量
     */
    static Tensor zeros(DataType dtype, std::initializer_list<size_t> shape);

    /**
     * 创建全一张量
     */
    static Tensor ones(DataType dtype, std::initializer_list<size_t> shape);

    /**
     * 创建与另一张量相同形状的空张量
     */
    static Tensor empty_like(const Tensor& other);

    //=========================================================================
    // 数据访问
    //=========================================================================

    /**
     * 获取数据指针
     */
    void* data();
    const void* data() const;

    /**
     * 获取类型化数据指针
     */
    template <typename T>
    T* data_ptr() {
        return static_cast<T*>(data());
    }

    template <typename T>
    const T* data_ptr() const {
        return static_cast<const T*>(data());
    }

    /**
     * 获取张量描述
     */
    const TensorDesc& desc() const;

    /**
     * 获取数据类型
     */
    DataType dtype() const;

    /**
     * 获取维度数
     */
    size_t ndim() const;

    /**
     * 获取指定维度大小
     */
    size_t shape(size_t dim) const;

    /**
     * 获取形状向量
     */
    const std::vector<size_t>& shape() const;

    /**
     * 获取元素总数
     */
    size_t numel() const;

    /**
     * 获取字节大小
     */
    size_t size_bytes() const;

    /**
     * 检查是否为空
     */
    bool empty() const;

    /**
     * 检查是否为有效张量
     */
    bool valid() const;

    //=========================================================================
    // 内存与设备
    //=========================================================================

    /**
     * 检查是否是 UMA 张量
     */
    bool is_uma() const;

    /**
     * 获取 Metal buffer (仅 UMA 张量)
     * @return Metal buffer 指针，非 UMA 返回 nullptr
     */
    void* metal_buffer() const;

    /**
     * 检查张量是否拥有数据 (vs 包装外部数据)
     */
    bool owns_data() const;

    /**
     * 检查是否紧密排列
     */
    bool is_contiguous() const;

    /**
     * 获取数据所在设备
     */
    DeviceType device() const;

    //=========================================================================
    // 转换操作
    //=========================================================================

    /**
     * 转换到指定数据类型
     * @param dtype 目标类型
     * @return 新张量
     */
    Tensor to(DataType dtype) const;

    /**
     * 转换到指定设备
     * @param device 目标设备
     * @return 新张量
     */
    Tensor to_device(DeviceType device) const;

    /**
     * 转换为连续内存布局
     * @return 连续张量 (如果已连续则返回自身)
     */
    Tensor contiguous() const;

    /**
     * 克隆张量 (深拷贝)
     */
    Tensor clone() const;

    //=========================================================================
    // 视图操作
    //=========================================================================

    /**
     * 重塑 (不改变数据，必须元素数相同)
     */
    Tensor reshape(std::initializer_list<size_t> new_shape) const;

    /**
     * 转置 (2D 张量)
     */
    Tensor transpose() const;

    /**
     * 获取切片视图
     * @param dim 维度
     * @param start 起始索引
     * @param end 结束索引 (不包含)
     * @return 切片视图
     */
    Tensor slice(size_t dim, size_t start, size_t end) const;

    /**
     * 获取单个样本 (batch 维度)
     */
    Tensor operator[](size_t idx) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

//=============================================================================
// 张量工具函数
//=============================================================================

/**
 * 拼接张量
 * @param tensors 张量数组
 * @param dim 拼接维度
 */
Tensor concat(const std::vector<Tensor>& tensors, size_t dim = 0);

/**
 * 分割张量
 * @param tensor 输入张量
 * @param chunks 分块数
 * @param dim 分割维度
 */
std::vector<Tensor> split(const Tensor& tensor, size_t chunks, size_t dim = 0);

/**
 * 批量堆叠
 */
Tensor stack(const std::vector<Tensor>& tensors, size_t dim = 0);

} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_TENSOR_H
