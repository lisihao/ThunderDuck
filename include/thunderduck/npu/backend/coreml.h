/**
 * ThunderDuck NPU Backend - Core ML Implementation
 *
 * @file coreml.h
 * @version V20
 * @date 2026-01-27
 *
 * Apple Core ML 后端，访问 Neural Engine (ANE)
 */

#ifndef THUNDERDUCK_NPU_BACKEND_COREML_H
#define THUNDERDUCK_NPU_BACKEND_COREML_H

#include "../device.h"
#include "../tensor.h"
#include "../model.h"
#include "../inference.h"
#include <memory>

namespace thunderduck {
namespace npu {
namespace coreml {

using accelerator::DeviceType;

//=============================================================================
// Core ML 计算单元
//=============================================================================

/**
 * Core ML 计算单元选择
 */
enum class ComputeUnits {
    ALL,            ///< 所有可用 (ANE + GPU + CPU)
    CPU_AND_GPU,    ///< 仅 CPU 和 GPU
    CPU_AND_NE,     ///< CPU 和 Neural Engine
    CPU_ONLY,       ///< 仅 CPU
    NE_ONLY         ///< 仅 Neural Engine (如果支持)
};

//=============================================================================
// Core ML 模型
//=============================================================================

/**
 * Core ML 模型实现
 *
 * 支持 .mlmodelc (编译后) 和 .mlpackage 格式
 */
class CoreMLModel : public Model {
public:
    /**
     * 从文件加载模型
     * @param path 模型路径
     * @param options 加载选项
     */
    CoreMLModel(const char* path, const ModelLoadOptions& options = {});

    ~CoreMLModel() override;

    // 移动构造
    CoreMLModel(CoreMLModel&& other) noexcept;
    CoreMLModel& operator=(CoreMLModel&& other) noexcept;

    // 禁止拷贝
    CoreMLModel(const CoreMLModel&) = delete;
    CoreMLModel& operator=(const CoreMLModel&) = delete;

    //=========================================================================
    // Model 接口实现
    //=========================================================================

    const char* name() const override;
    const char* path() const override;
    ModelFormat format() const override;
    std::vector<TensorDesc> input_specs() const override;
    std::vector<TensorDesc> output_specs() const override;
    size_t num_inputs() const override;
    size_t num_outputs() const override;
    const char* input_name(size_t idx) const override;
    const char* output_name(size_t idx) const override;

    bool supports_device(DeviceType device) const override;
    DeviceType preferred_device() const override;
    std::pair<size_t, size_t> batch_size_range() const override;

    const char* description() const override;
    const char* author() const override;
    const char* version() const override;
    uint64_t estimated_flops() const override;
    uint64_t num_parameters() const override;

    //=========================================================================
    // Core ML 特有方法
    //=========================================================================

    /**
     * 获取 Core ML 计算单元
     */
    ComputeUnits compute_units() const;

    /**
     * 检查是否为 ANE 优化
     */
    bool is_ane_optimized() const;

    /**
     * 获取原生 Core ML 模型句柄
     * @return MLModel* (Objective-C 对象)
     */
    void* native_handle() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//=============================================================================
// Core ML 推理会话
//=============================================================================

/**
 * Core ML 推理会话实现
 */
class CoreMLSession : public InferenceSession {
public:
    /**
     * 创建推理会话
     * @param model Core ML 模型
     * @param config 会话配置
     */
    CoreMLSession(Model::Ptr model, const SessionConfig& config = {});

    ~CoreMLSession() override;

    // 移动语义
    CoreMLSession(CoreMLSession&& other) noexcept;
    CoreMLSession& operator=(CoreMLSession&& other) noexcept;

    //=========================================================================
    // InferenceSession 接口实现
    //=========================================================================

    // 同步推理
    bool run(const Tensor* inputs, size_t num_inputs,
             Tensor* outputs, size_t num_outputs) override;
    bool run(const Tensor& input, Tensor& output) override;
    bool run_batch(const Tensor* const* batch_inputs,
                   Tensor** batch_outputs,
                   size_t batch_size) override;

    // 异步推理
    InferenceRequest submit_async(const Tensor* inputs, size_t num_inputs,
                                  void* user_data = nullptr) override;
    InferenceRequest submit_async(const Tensor* inputs, size_t num_inputs,
                                  InferenceRequest::CompletionCallback callback,
                                  void* user_data = nullptr) override;
    void wait_all() override;
    size_t pending_requests() const override;

    // 流式推理
    StreamHandle begin_stream(size_t max_queue_depth = 16) override;
    void stream_submit(StreamHandle stream,
                       const Tensor* inputs, size_t num_inputs,
                       void* user_data = nullptr) override;
    bool stream_get_result(StreamHandle stream, StreamResult& result) override;
    bool stream_try_get_result(StreamHandle stream, StreamResult& result) override;
    size_t stream_pending(StreamHandle stream) const override;
    void end_stream(StreamHandle stream, bool wait_completion = true) override;

    // 设备信息
    DeviceType active_device() const override;
    const char* device_name() const override;
    Model::Ptr model() const override;
    const SessionConfig& config() const override;

    // 性能统计
    double avg_inference_time_us() const override;
    double throughput() const override;
    void reset_stats() override;

    const char* last_error() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//=============================================================================
// Core ML 工具函数
//=============================================================================

/**
 * 检查 Core ML 是否可用
 */
bool is_coreml_available();

/**
 * 获取 Core ML 版本
 */
const char* coreml_version();

/**
 * 编译 .mlmodel 到 .mlmodelc
 * @param input_path 输入路径 (.mlmodel)
 * @param output_path 输出路径 (.mlmodelc)
 * @return 是否成功
 */
bool compile_model(const char* input_path, const char* output_path);

/**
 * 验证模型是否 ANE 兼容
 * @param path 模型路径
 * @return 兼容性报告
 */
struct ANECompatibility {
    bool fully_compatible;
    int compatible_ops_percent;
    std::vector<std::string> unsupported_ops;
};
ANECompatibility check_ane_compatibility(const char* path);

} // namespace coreml
} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_BACKEND_COREML_H
