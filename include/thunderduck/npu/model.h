/**
 * ThunderDuck NPU Abstraction Layer - Model Interface
 *
 * @file model.h
 * @version V20
 * @date 2026-01-27
 *
 * ML 模型加载和管理接口
 */

#ifndef THUNDERDUCK_NPU_MODEL_H
#define THUNDERDUCK_NPU_MODEL_H

#include "device.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <vector>
#include <future>
#include <functional>

namespace thunderduck {
namespace npu {

using accelerator::DeviceType;
using accelerator::DataType;

//=============================================================================
// 模型格式
//=============================================================================

/**
 * 模型格式
 */
enum class ModelFormat {
    COREML,         ///< Apple Core ML (.mlmodelc, .mlpackage)
    ONNX,           ///< ONNX format (需要转换)
    MLMODEL_BIN     ///< 预编译二进制
};

//=============================================================================
// 模型加载选项
//=============================================================================

/**
 * 模型加载选项
 */
struct ModelLoadOptions {
    DeviceType preferred_device = DeviceType::AUTO;  ///< 首选设备
    bool allow_low_precision = true;    ///< 允许 FP16 降精度
    bool compile_for_ane = true;        ///< 为 ANE 编译优化
    size_t max_batch_size = 64;         ///< 最大批处理大小
    bool enable_profiling = false;      ///< 启用性能分析
    bool async_load = false;            ///< 异步加载

    /**
     * 默认选项
     */
    static ModelLoadOptions defaults() {
        return ModelLoadOptions{};
    }

    /**
     * ANE 优化选项
     */
    static ModelLoadOptions for_ane() {
        ModelLoadOptions opts;
        opts.preferred_device = DeviceType::NPU;
        opts.allow_low_precision = true;
        opts.compile_for_ane = true;
        return opts;
    }

    /**
     * GPU 优化选项
     */
    static ModelLoadOptions for_gpu() {
        ModelLoadOptions opts;
        opts.preferred_device = DeviceType::GPU;
        opts.compile_for_ane = false;
        return opts;
    }

    /**
     * 低延迟选项
     */
    static ModelLoadOptions low_latency() {
        ModelLoadOptions opts;
        opts.max_batch_size = 1;
        return opts;
    }
};

//=============================================================================
// 模型接口
//=============================================================================

/**
 * ML 模型句柄
 *
 * 表示已加载的机器学习模型，可用于创建推理会话。
 * 线程安全：可在多个线程间共享。
 */
class Model {
public:
    using Ptr = std::shared_ptr<Model>;

    virtual ~Model() = default;

    //=========================================================================
    // 模型信息
    //=========================================================================

    /**
     * 获取模型名称
     */
    virtual const char* name() const = 0;

    /**
     * 获取模型路径
     */
    virtual const char* path() const = 0;

    /**
     * 获取模型格式
     */
    virtual ModelFormat format() const = 0;

    /**
     * 获取输入规格
     */
    virtual std::vector<TensorDesc> input_specs() const = 0;

    /**
     * 获取输出规格
     */
    virtual std::vector<TensorDesc> output_specs() const = 0;

    /**
     * 获取输入数量
     */
    virtual size_t num_inputs() const = 0;

    /**
     * 获取输出数量
     */
    virtual size_t num_outputs() const = 0;

    /**
     * 获取输入名称
     */
    virtual const char* input_name(size_t idx) const = 0;

    /**
     * 获取输出名称
     */
    virtual const char* output_name(size_t idx) const = 0;

    //=========================================================================
    // 设备能力
    //=========================================================================

    /**
     * 检查是否支持指定设备
     */
    virtual bool supports_device(DeviceType device) const = 0;

    /**
     * 获取首选设备
     */
    virtual DeviceType preferred_device() const = 0;

    /**
     * 获取支持的批处理大小范围
     */
    virtual std::pair<size_t, size_t> batch_size_range() const = 0;

    //=========================================================================
    // 模型元信息
    //=========================================================================

    /**
     * 获取模型描述
     */
    virtual const char* description() const = 0;

    /**
     * 获取模型作者
     */
    virtual const char* author() const = 0;

    /**
     * 获取模型版本
     */
    virtual const char* version() const = 0;

    /**
     * 获取估计的 FLOPS
     */
    virtual uint64_t estimated_flops() const = 0;

    /**
     * 获取模型参数量
     */
    virtual uint64_t num_parameters() const = 0;

protected:
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
};

//=============================================================================
// 模型加载器
//=============================================================================

/**
 * 模型加载回调
 */
using ModelLoadCallback = std::function<void(Model::Ptr, const char* error)>;

/**
 * 模型加载器 - 单例模式
 *
 * 负责加载、缓存和管理 ML 模型。
 */
class ModelLoader {
public:
    /**
     * 获取单例实例
     */
    static ModelLoader& instance();

    //=========================================================================
    // 同步加载
    //=========================================================================

    /**
     * 从 Core ML 模型文件加载
     * @param path .mlmodelc 或 .mlpackage 路径
     * @param options 加载选项
     * @return 模型指针，失败返回 nullptr
     */
    Model::Ptr load_coreml(const char* path,
                           const ModelLoadOptions& options = {});

    /**
     * 从内存加载预编译模型
     * @param data 模型数据
     * @param size 数据大小
     * @param format 模型格式
     * @return 模型指针
     */
    Model::Ptr load_from_buffer(const void* data, size_t size,
                                ModelFormat format);

    //=========================================================================
    // 异步加载
    //=========================================================================

    /**
     * 异步预加载模型
     * @param path 模型路径
     * @param options 加载选项
     * @return Future 模型指针
     */
    std::future<Model::Ptr> preload_async(const char* path,
                                          const ModelLoadOptions& options = {});

    /**
     * 异步加载（回调方式）
     * @param path 模型路径
     * @param callback 完成回调
     * @param options 加载选项
     */
    void load_async(const char* path,
                    ModelLoadCallback callback,
                    const ModelLoadOptions& options = {});

    //=========================================================================
    // 缓存管理
    //=========================================================================

    /**
     * 获取已缓存的模型
     * @param name 模型名称或路径
     * @return 缓存的模型，未找到返回 nullptr
     */
    Model::Ptr get_cached(const char* name);

    /**
     * 检查模型是否已缓存
     */
    bool is_cached(const char* name) const;

    /**
     * 添加模型到缓存
     */
    void cache_model(const char* name, Model::Ptr model);

    /**
     * 从缓存中移除模型
     */
    void uncache(const char* name);

    /**
     * 清空模型缓存
     */
    void clear_cache();

    /**
     * 获取缓存中的模型数量
     */
    size_t cache_size() const;

    /**
     * 获取缓存使用的内存大小
     */
    size_t cache_memory_usage() const;

    //=========================================================================
    // 模型编译
    //=========================================================================

    /**
     * 编译 ONNX 模型到 Core ML
     * @param onnx_path ONNX 模型路径
     * @param output_path 输出 Core ML 路径
     * @param options 编译选项
     * @return 是否成功
     */
    bool compile_onnx(const char* onnx_path,
                      const char* output_path,
                      const ModelLoadOptions& options = {});

    //=========================================================================
    // 错误处理
    //=========================================================================

    /**
     * 获取最后一次错误信息
     */
    const char* last_error() const;

    /**
     * 清除错误状态
     */
    void clear_error();

private:
    ModelLoader();
    ~ModelLoader();
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    struct Impl;
    Impl* impl_;
};

//=============================================================================
// 模型验证
//=============================================================================

/**
 * 验证模型输入
 * @param model 模型
 * @param inputs 输入张量数组
 * @param num_inputs 输入数量
 * @return 验证通过返回 true
 */
bool validate_inputs(const Model& model,
                     const Tensor* inputs,
                     size_t num_inputs);

/**
 * 验证模型输出
 */
bool validate_outputs(const Model& model,
                      const Tensor* outputs,
                      size_t num_outputs);

} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_MODEL_H
