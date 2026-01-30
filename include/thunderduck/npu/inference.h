/**
 * ThunderDuck NPU Abstraction Layer - Inference Interface
 *
 * @file inference.h
 * @version V20
 * @date 2026-01-27
 *
 * ML 推理执行接口，支持同步、异步、流式推理
 */

#ifndef THUNDERDUCK_NPU_INFERENCE_H
#define THUNDERDUCK_NPU_INFERENCE_H

#include "device.h"
#include "tensor.h"
#include "model.h"
#include <memory>
#include <functional>
#include <chrono>

namespace thunderduck {
namespace npu {

using accelerator::DeviceType;

//=============================================================================
// 前向声明
//=============================================================================

class InferenceSession;
class InferenceRequest;

//=============================================================================
// 会话配置
//=============================================================================

/**
 * 推理会话配置
 */
struct SessionConfig {
    DeviceType device = DeviceType::AUTO;   ///< 推理设备
    size_t batch_size = 1;                  ///< 默认批处理大小
    bool enable_profiling = false;          ///< 启用性能分析
    bool use_shared_memory = true;          ///< 使用 UMA 共享内存
    size_t async_queue_depth = 8;           ///< 异步队列深度

    // 性能调优
    bool allow_fp16 = true;                 ///< 允许 FP16 计算
    bool optimize_latency = false;          ///< 优化延迟 (vs 吞吐量)
    size_t max_concurrent_requests = 1;     ///< 最大并发请求数

    /**
     * 默认配置
     */
    static SessionConfig defaults() {
        return SessionConfig{};
    }

    /**
     * 高吞吐量配置
     */
    static SessionConfig high_throughput() {
        SessionConfig cfg;
        cfg.batch_size = 32;
        cfg.async_queue_depth = 16;
        cfg.max_concurrent_requests = 4;
        return cfg;
    }

    /**
     * 低延迟配置
     */
    static SessionConfig low_latency() {
        SessionConfig cfg;
        cfg.batch_size = 1;
        cfg.optimize_latency = true;
        cfg.max_concurrent_requests = 1;
        return cfg;
    }
};

//=============================================================================
// 推理请求
//=============================================================================

/**
 * 推理请求状态
 */
enum class RequestStatus {
    PENDING,        ///< 等待执行
    RUNNING,        ///< 执行中
    COMPLETED,      ///< 已完成
    FAILED,         ///< 失败
    CANCELLED       ///< 已取消
};

/**
 * 推理请求 - 异步推理的句柄
 */
class InferenceRequest {
public:
    using CompletionCallback = std::function<void(const InferenceRequest&)>;

    InferenceRequest();
    ~InferenceRequest();

    InferenceRequest(InferenceRequest&& other) noexcept;
    InferenceRequest& operator=(InferenceRequest&& other) noexcept;

    // 禁止拷贝
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;

    /**
     * 获取请求状态
     */
    RequestStatus status() const;

    /**
     * 检查是否已完成
     */
    bool is_done() const;

    /**
     * 检查是否成功
     */
    bool is_success() const;

    /**
     * 等待完成
     * @param timeout_ns 超时 (纳秒)
     * @return 是否在超时前完成
     */
    bool wait(uint64_t timeout_ns = UINT64_MAX);

    /**
     * 等待完成 (chrono 版本)
     */
    template <typename Rep, typename Period>
    bool wait_for(std::chrono::duration<Rep, Period> timeout) {
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout);
        return wait(ns.count());
    }

    /**
     * 获取输出张量数量
     */
    size_t num_outputs() const;

    /**
     * 获取输出张量
     */
    const Tensor& output(size_t idx) const;

    /**
     * 获取所有输出
     */
    const std::vector<Tensor>& outputs() const;

    /**
     * 获取错误信息 (如果失败)
     */
    const char* error_message() const;

    /**
     * 获取执行时间 (纳秒)
     */
    uint64_t execution_time_ns() const;

    /**
     * 获取用户数据
     */
    void* user_data() const;

    /**
     * 取消请求
     */
    void cancel();

private:
    friend class InferenceSession;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//=============================================================================
// 流式推理句柄
//=============================================================================

/**
 * 流式推理句柄
 */
using StreamHandle = uint64_t;
constexpr StreamHandle INVALID_STREAM = 0;

/**
 * 流式推理结果
 */
struct StreamResult {
    std::vector<Tensor> outputs;
    void* user_data = nullptr;
    uint64_t sequence_id = 0;
    bool valid = false;
};

//=============================================================================
// 推理会话
//=============================================================================

/**
 * 推理会话 - 执行模型推理
 *
 * 支持三种推理模式:
 * 1. 同步推理: run() - 阻塞直到完成
 * 2. 异步推理: submit_async() + wait() - 提交并稍后等待
 * 3. 流式推理: begin_stream() + stream_submit() - 高吞吐流水线
 */
class InferenceSession {
public:
    using Ptr = std::shared_ptr<InferenceSession>;

    virtual ~InferenceSession() = default;

    //=========================================================================
    // 创建
    //=========================================================================

    /**
     * 创建推理会话
     * @param model 已加载的模型
     * @param config 会话配置
     * @return 会话指针
     */
    static Ptr create(Model::Ptr model,
                      const SessionConfig& config = {});

    //=========================================================================
    // 同步推理
    //=========================================================================

    /**
     * 单次推理
     * @param inputs 输入张量数组
     * @param num_inputs 输入数量
     * @param outputs 输出张量数组 (预分配或空)
     * @param num_outputs 输出数量
     * @return 成功返回 true
     */
    virtual bool run(const Tensor* inputs, size_t num_inputs,
                     Tensor* outputs, size_t num_outputs) = 0;

    /**
     * 单输入单输出推理 (便捷版本)
     */
    virtual bool run(const Tensor& input, Tensor& output) = 0;

    /**
     * 批量推理
     * @param batch_inputs 批量输入 [batch_size][num_inputs]
     * @param batch_outputs 批量输出 [batch_size][num_outputs]
     * @param batch_size 批次大小
     * @return 成功返回 true
     */
    virtual bool run_batch(const Tensor* const* batch_inputs,
                           Tensor** batch_outputs,
                           size_t batch_size) = 0;

    //=========================================================================
    // 异步推理
    //=========================================================================

    /**
     * 异步提交推理请求
     * @param inputs 输入张量数组
     * @param num_inputs 输入数量
     * @param user_data 用户数据 (可选)
     * @return 推理请求句柄
     */
    virtual InferenceRequest submit_async(const Tensor* inputs,
                                          size_t num_inputs,
                                          void* user_data = nullptr) = 0;

    /**
     * 异步提交 (带回调)
     */
    virtual InferenceRequest submit_async(
        const Tensor* inputs, size_t num_inputs,
        InferenceRequest::CompletionCallback callback,
        void* user_data = nullptr) = 0;

    /**
     * 等待所有异步请求完成
     */
    virtual void wait_all() = 0;

    /**
     * 获取挂起的请求数量
     */
    virtual size_t pending_requests() const = 0;

    //=========================================================================
    // 流式推理 (向量数据库优化)
    //=========================================================================

    /**
     * 开始流式推理
     * @param max_queue_depth 最大队列深度
     * @return 流句柄
     */
    virtual StreamHandle begin_stream(size_t max_queue_depth = 16) = 0;

    /**
     * 提交到流
     * @param stream 流句柄
     * @param inputs 输入张量
     * @param num_inputs 输入数量
     * @param user_data 用户数据
     */
    virtual void stream_submit(StreamHandle stream,
                               const Tensor* inputs, size_t num_inputs,
                               void* user_data = nullptr) = 0;

    /**
     * 从流获取结果 (阻塞)
     * @param stream 流句柄
     * @param result 输出结果
     * @return 成功返回 true，流已关闭返回 false
     */
    virtual bool stream_get_result(StreamHandle stream,
                                   StreamResult& result) = 0;

    /**
     * 尝试从流获取结果 (非阻塞)
     * @param stream 流句柄
     * @param result 输出结果
     * @return 有结果返回 true，无结果返回 false
     */
    virtual bool stream_try_get_result(StreamHandle stream,
                                       StreamResult& result) = 0;

    /**
     * 获取流中挂起的请求数
     */
    virtual size_t stream_pending(StreamHandle stream) const = 0;

    /**
     * 结束流式推理
     * @param stream 流句柄
     * @param wait_completion 是否等待所有请求完成
     */
    virtual void end_stream(StreamHandle stream,
                            bool wait_completion = true) = 0;

    //=========================================================================
    // 设备信息
    //=========================================================================

    /**
     * 获取当前活动设备
     */
    virtual DeviceType active_device() const = 0;

    /**
     * 获取设备名称
     */
    virtual const char* device_name() const = 0;

    /**
     * 获取关联的模型
     */
    virtual Model::Ptr model() const = 0;

    /**
     * 获取会话配置
     */
    virtual const SessionConfig& config() const = 0;

    //=========================================================================
    // 性能分析
    //=========================================================================

    /**
     * 获取平均推理时间 (微秒)
     */
    virtual double avg_inference_time_us() const = 0;

    /**
     * 获取吞吐量 (推理/秒)
     */
    virtual double throughput() const = 0;

    /**
     * 重置性能统计
     */
    virtual void reset_stats() = 0;

    //=========================================================================
    // 错误处理
    //=========================================================================

    /**
     * 获取最后一次错误
     */
    virtual const char* last_error() const = 0;

protected:
    InferenceSession() = default;
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
};

//=============================================================================
// 便捷函数
//=============================================================================

/**
 * 快速推理 (一次性加载+执行)
 * @param model_path 模型路径
 * @param inputs 输入张量
 * @param num_inputs 输入数量
 * @param outputs 输出张量
 * @param num_outputs 输出数量
 * @return 成功返回 true
 */
bool quick_inference(const char* model_path,
                     const Tensor* inputs, size_t num_inputs,
                     Tensor* outputs, size_t num_outputs);

/**
 * 批量快速推理
 */
bool quick_batch_inference(const char* model_path,
                           const Tensor* const* batch_inputs,
                           Tensor** batch_outputs,
                           size_t batch_size,
                           size_t num_inputs_per_batch,
                           size_t num_outputs_per_batch);

} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_INFERENCE_H
