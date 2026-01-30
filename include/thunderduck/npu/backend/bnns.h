/**
 * ThunderDuck NPU Backend - BNNS (Basic Neural Network Subroutines)
 *
 * @file bnns.h
 * @version V20
 * @date 2026-01-27
 *
 * Apple Accelerate BNNS 后端，轻量级 CPU 神经网络操作
 */

#ifndef THUNDERDUCK_NPU_BACKEND_BNNS_H
#define THUNDERDUCK_NPU_BACKEND_BNNS_H

#include "../device.h"
#include "../tensor.h"
#include <memory>
#include <cstddef>

namespace thunderduck {
namespace npu {
namespace bnns {

//=============================================================================
// BNNS 激活函数
//=============================================================================

/**
 * BNNS 激活函数类型
 */
enum class BNNSActivation {
    IDENTITY,       ///< 无激活
    RELU,           ///< ReLU
    LEAKY_RELU,     ///< Leaky ReLU
    SIGMOID,        ///< Sigmoid
    TANH,           ///< Tanh
    SOFTMAX,        ///< Softmax
    GELU,           ///< GELU (近似)
    SILU            ///< SiLU / Swish
};

//=============================================================================
// BNNS 加速器
//=============================================================================

/**
 * BNNS 加速器 - 轻量级神经网络操作
 *
 * 使用 Apple Accelerate 框架，利用 AMX 和 SIMD 加速。
 * 适合不需要完整 Core ML 的轻量级计算。
 *
 * 特点:
 * - 低延迟 (无模型加载开销)
 * - CPU 优化 (AMX + NEON)
 * - 灵活的动态形状
 */
class BNNSAccelerator {
public:
    /**
     * 获取单例实例
     */
    static BNNSAccelerator& instance();

    //=========================================================================
    // 向量归一化
    //=========================================================================

    /**
     * L2 归一化单个向量
     * @param input 输入向量
     * @param output 输出向量 (可以与 input 相同)
     * @param dim 向量维度
     */
    void normalize_l2(const float* input, float* output, size_t dim);

    /**
     * 批量 L2 归一化
     */
    void normalize_l2_batch(const float* input, float* output,
                            size_t batch_size, size_t dim);

    /**
     * Layer Normalization
     * @param input 输入 [batch_size x dim]
     * @param output 输出
     * @param gamma 缩放参数 [dim]
     * @param beta 偏移参数 [dim]
     * @param batch_size 批次大小
     * @param dim 特征维度
     * @param eps epsilon
     */
    void layer_norm(const float* input, float* output,
                    const float* gamma, const float* beta,
                    size_t batch_size, size_t dim,
                    float eps = 1e-5f);

    //=========================================================================
    // 矩阵运算 (利用 BLAS/AMX)
    //=========================================================================

    /**
     * 矩阵乘法 C = A @ B
     * @param A 矩阵 A [M x K]
     * @param B 矩阵 B [K x N]
     * @param C 结果 C [M x N]
     */
    void matmul(const float* A, const float* B, float* C,
                size_t M, size_t K, size_t N);

    /**
     * 带转置的矩阵乘法
     * @param trans_a 是否转置 A
     * @param trans_b 是否转置 B
     */
    void matmul_ex(const float* A, const float* B, float* C,
                   size_t M, size_t K, size_t N,
                   bool trans_a, bool trans_b,
                   float alpha = 1.0f, float beta = 0.0f);

    /**
     * 批量矩阵乘法 (BMM)
     */
    void batch_matmul(const float* A, const float* B, float* C,
                      size_t batch_size, size_t M, size_t K, size_t N);

    //=========================================================================
    // 全连接层
    //=========================================================================

    /**
     * 全连接层前向传播
     * @param input 输入 [batch_size x input_dim]
     * @param weights 权重 [output_dim x input_dim]
     * @param bias 偏置 [output_dim] (可为 nullptr)
     * @param output 输出 [batch_size x output_dim]
     * @param batch_size 批次大小
     * @param input_dim 输入维度
     * @param output_dim 输出维度
     * @param activation 激活函数
     */
    void dense_layer(const float* input, const float* weights, const float* bias,
                     float* output,
                     size_t batch_size, size_t input_dim, size_t output_dim,
                     BNNSActivation activation = BNNSActivation::IDENTITY);

    //=========================================================================
    // 激活函数
    //=========================================================================

    /**
     * 应用激活函数 (就地)
     */
    void activation(float* data, size_t count, BNNSActivation type);

    /**
     * ReLU
     */
    void relu(float* data, size_t count);

    /**
     * GELU (近似)
     */
    void gelu(float* data, size_t count);

    /**
     * Softmax
     * @param input 输入 [batch_size x dim]
     * @param output 输出
     * @param batch_size 批次大小
     * @param dim 特征维度
     */
    void softmax(const float* input, float* output,
                 size_t batch_size, size_t dim);

    //=========================================================================
    // 向量运算 (vDSP)
    //=========================================================================

    /**
     * 向量点积
     */
    float dot_product(const float* a, const float* b, size_t count);

    /**
     * 批量点积
     */
    void batch_dot_product(const float* queries, const float* keys,
                           float* scores,
                           size_t num_queries, size_t num_keys, size_t dim);

    /**
     * 向量加法 c = a + b
     */
    void vector_add(const float* a, const float* b, float* c, size_t count);

    /**
     * 向量乘法 c = a * b (element-wise)
     */
    void vector_mul(const float* a, const float* b, float* c, size_t count);

    /**
     * 标量乘法 b = a * scalar
     */
    void scalar_mul(const float* a, float scalar, float* b, size_t count);

    /**
     * 向量求和
     */
    float vector_sum(const float* a, size_t count);

    /**
     * 向量均值
     */
    float vector_mean(const float* a, size_t count);

    /**
     * 向量 L2 范数
     */
    float vector_l2_norm(const float* a, size_t count);

    //=========================================================================
    // 距离计算
    //=========================================================================

    /**
     * 欧氏距离平方
     */
    float euclidean_distance_squared(const float* a, const float* b, size_t dim);

    /**
     * 批量欧氏距离
     */
    void batch_euclidean_distance(const float* query, const float* candidates,
                                  float* distances,
                                  size_t num_candidates, size_t dim);

    /**
     * 余弦相似度
     */
    float cosine_similarity(const float* a, const float* b, size_t dim);

    /**
     * 批量余弦相似度
     */
    void batch_cosine_similarity(const float* query, const float* candidates,
                                 float* similarities,
                                 size_t num_candidates, size_t dim);

private:
    BNNSAccelerator();
    ~BNNSAccelerator();
    BNNSAccelerator(const BNNSAccelerator&) = delete;
    BNNSAccelerator& operator=(const BNNSAccelerator&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

//=============================================================================
// BNNS 简单 MLP
//=============================================================================

/**
 * BNNS 简单 MLP (Multi-Layer Perceptron)
 *
 * 轻量级 MLP 实现，适合:
 * - 小型嵌入变换
 * - 学习型相似度
 * - 量化编解码器
 */
class BNNLSMLP {
public:
    using Ptr = std::shared_ptr<BNNLSMLP>;

    /**
     * 创建 MLP
     * @param layer_sizes 层大小 [input, hidden1, ..., output]
     * @param activation 隐藏层激活函数
     * @param output_activation 输出层激活函数
     */
    static Ptr create(const std::vector<size_t>& layer_sizes,
                      BNNSActivation activation = BNNSActivation::RELU,
                      BNNSActivation output_activation = BNNSActivation::IDENTITY);

    /**
     * 加载权重
     * @param weights 所有层权重 (连续存储)
     * @param biases 所有层偏置 (连续存储)
     */
    virtual void load_weights(const float* weights, const float* biases) = 0;

    /**
     * 从文件加载
     */
    virtual bool load_from_file(const char* path) = 0;

    /**
     * 前向传播
     */
    virtual void forward(const float* input, float* output, size_t batch_size) = 0;

    /**
     * 获取输入维度
     */
    virtual size_t input_dim() const = 0;

    /**
     * 获取输出维度
     */
    virtual size_t output_dim() const = 0;

protected:
    BNNLSMLP() = default;
};

//=============================================================================
// 工具函数
//=============================================================================

/**
 * 检查 BNNS 是否可用
 */
bool is_bnns_available();

/**
 * 获取 BNNS 版本
 */
const char* bnns_version();

/**
 * 检查是否支持 AMX
 */
bool bnns_has_amx();

} // namespace bnns
} // namespace npu
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_BACKEND_BNNS_H
