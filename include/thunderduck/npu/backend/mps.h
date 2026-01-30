/**
 * ThunderDuck NPU Backend - Metal Performance Shaders
 *
 * @file mps.h
 * @version V20
 * @date 2026-01-27
 *
 * Metal Performance Shaders 后端，GPU 神经网络推理
 */

#ifndef THUNDERDUCK_NPU_BACKEND_MPS_H
#define THUNDERDUCK_NPU_BACKEND_MPS_H

#include "../device.h"
#include "../tensor.h"
#include <memory>
#include <functional>

namespace thunderduck {
namespace npu {
namespace mps {

//=============================================================================
// MPS 数据类型
//=============================================================================

/**
 * MPS 支持的数据类型
 */
enum class MPSDataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8
};

//=============================================================================
// MPS 神经网络层
//=============================================================================

/**
 * 激活函数类型
 */
enum class ActivationType {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    GELU,
    SILU,       // Swish
    SOFTMAX
};

/**
 * MPS 全连接层
 */
class MPSDenseLayer {
public:
    using Ptr = std::shared_ptr<MPSDenseLayer>;

    /**
     * 创建全连接层
     * @param input_features 输入特征数
     * @param output_features 输出特征数
     * @param activation 激活函数
     * @param use_bias 是否使用偏置
     */
    static Ptr create(size_t input_features,
                      size_t output_features,
                      ActivationType activation = ActivationType::NONE,
                      bool use_bias = true);

    /**
     * 设置权重
     * @param weights 权重数据 [output_features x input_features]
     * @param bias 偏置数据 [output_features] (可选)
     */
    virtual void set_weights(const float* weights,
                             const float* bias = nullptr) = 0;

    /**
     * 前向传播
     * @param input 输入张量 [batch x input_features]
     * @param output 输出张量 [batch x output_features]
     */
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    virtual size_t input_features() const = 0;
    virtual size_t output_features() const = 0;

protected:
    MPSDenseLayer() = default;
};

/**
 * MPS 卷积层 (1D, 用于序列处理)
 */
class MPSConv1D {
public:
    using Ptr = std::shared_ptr<MPSConv1D>;

    static Ptr create(size_t in_channels,
                      size_t out_channels,
                      size_t kernel_size,
                      size_t stride = 1,
                      size_t padding = 0);

    virtual void set_weights(const float* weights, const float* bias = nullptr) = 0;
    virtual void forward(const Tensor& input, Tensor& output) = 0;

protected:
    MPSConv1D() = default;
};

/**
 * MPS Layer Normalization
 */
class MPSLayerNorm {
public:
    using Ptr = std::shared_ptr<MPSLayerNorm>;

    static Ptr create(size_t normalized_shape, float eps = 1e-5f);

    virtual void set_parameters(const float* gamma, const float* beta) = 0;
    virtual void forward(const Tensor& input, Tensor& output) = 0;

protected:
    MPSLayerNorm() = default;
};

//=============================================================================
// MPS 神经网络图
//=============================================================================

/**
 * MPS 神经网络图 - 组合多个层
 */
class MPSNeuralNetwork {
public:
    using Ptr = std::shared_ptr<MPSNeuralNetwork>;

    /**
     * 创建空网络
     */
    static Ptr create();

    /**
     * 添加全连接层
     */
    virtual MPSNeuralNetwork& add_dense(size_t output_features,
                                        ActivationType activation = ActivationType::NONE,
                                        bool use_bias = true) = 0;

    /**
     * 添加 LayerNorm
     */
    virtual MPSNeuralNetwork& add_layer_norm(float eps = 1e-5f) = 0;

    /**
     * 添加 Dropout (仅训练时有效)
     */
    virtual MPSNeuralNetwork& add_dropout(float p) = 0;

    /**
     * 构建网络 (冻结结构)
     * @param input_features 输入特征数
     */
    virtual void build(size_t input_features) = 0;

    /**
     * 从文件加载权重
     */
    virtual bool load_weights(const char* path) = 0;

    /**
     * 前向传播
     */
    virtual void forward(const Tensor& input, Tensor& output) = 0;

    /**
     * 批量前向传播
     */
    virtual void forward_batch(const Tensor* inputs, Tensor* outputs,
                               size_t batch_size) = 0;

    /**
     * 获取输入特征数
     */
    virtual size_t input_features() const = 0;

    /**
     * 获取输出特征数
     */
    virtual size_t output_features() const = 0;

protected:
    MPSNeuralNetwork() = default;
};

//=============================================================================
// MPS 矩阵运算
//=============================================================================

/**
 * MPS 矩阵乘法
 */
class MPSMatrixMultiply {
public:
    using Ptr = std::shared_ptr<MPSMatrixMultiply>;

    /**
     * 创建矩阵乘法器
     * @param M 结果行数
     * @param N 结果列数
     * @param K 内部维度
     * @param transpose_a 是否转置 A
     * @param transpose_b 是否转置 B
     */
    static Ptr create(size_t M, size_t N, size_t K,
                      bool transpose_a = false,
                      bool transpose_b = false);

    /**
     * 执行 C = alpha * A @ B + beta * C
     */
    virtual void compute(const Tensor& A, const Tensor& B, Tensor& C,
                         float alpha = 1.0f, float beta = 0.0f) = 0;

protected:
    MPSMatrixMultiply() = default;
};

/**
 * MPS 批量矩阵乘法 (BMM)
 */
class MPSBatchMatrixMultiply {
public:
    using Ptr = std::shared_ptr<MPSBatchMatrixMultiply>;

    static Ptr create(size_t batch_size, size_t M, size_t N, size_t K);

    virtual void compute(const Tensor& A, const Tensor& B, Tensor& C) = 0;

protected:
    MPSBatchMatrixMultiply() = default;
};

//=============================================================================
// MPS 向量运算 (加速向量数据库)
//=============================================================================

/**
 * MPS 向量相似度计算
 */
class MPSVectorSimilarity {
public:
    using Ptr = std::shared_ptr<MPSVectorSimilarity>;

    enum class Metric {
        COSINE,
        DOT_PRODUCT,
        L2_DISTANCE
    };

    /**
     * 创建向量相似度计算器
     * @param dim 向量维度
     * @param metric 距离度量
     */
    static Ptr create(size_t dim, Metric metric = Metric::COSINE);

    /**
     * 批量计算相似度
     * @param queries 查询向量 [num_queries x dim]
     * @param candidates 候选向量 [num_candidates x dim]
     * @param scores 输出分数 [num_queries x num_candidates]
     */
    virtual void compute(const Tensor& queries,
                         const Tensor& candidates,
                         Tensor& scores) = 0;

    /**
     * Top-K 相似度搜索
     */
    virtual void topk(const Tensor& queries,
                      const Tensor& candidates,
                      size_t k,
                      Tensor& scores,
                      Tensor& indices) = 0;

protected:
    MPSVectorSimilarity() = default;
};

//=============================================================================
// MPS 工具函数
//=============================================================================

/**
 * 检查 MPS 是否可用
 */
bool is_mps_available();

/**
 * 获取 MPS 设备名称
 */
const char* mps_device_name();

/**
 * 获取 GPU 核心数
 */
size_t mps_gpu_cores();

/**
 * 同步 MPS 命令队列
 */
void mps_synchronize();

} // namespace mps
} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_BACKEND_MPS_H
