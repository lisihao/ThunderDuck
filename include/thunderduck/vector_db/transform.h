/**
 * ThunderDuck Vector Database - Embedding Transform Interface
 *
 * @file transform.h
 * @version V20
 * @date 2026-01-27
 *
 * 嵌入向量变换，通过神经网络调整向量表示
 */

#ifndef THUNDERDUCK_VECTOR_DB_TRANSFORM_H
#define THUNDERDUCK_VECTOR_DB_TRANSFORM_H

#include "../npu/model.h"
#include "../npu/tensor.h"
#include <memory>
#include <cstddef>

namespace thunderduck {
namespace vector_db {

using npu::Model;
using npu::Tensor;

//=============================================================================
// 变换类型
//=============================================================================

/**
 * 嵌入变换类型
 */
enum class TransformType {
    LINEAR,         ///< 线性变换 (矩阵乘法)
    MLP,            ///< 多层感知机
    ADAPTER,        ///< Adapter 模块 (低秩变换)
    PROJECTION      ///< 投影层 (降维)
};

//=============================================================================
// 变换配置
//=============================================================================

/**
 * 嵌入变换配置
 */
struct TransformConfig {
    size_t batch_size = 64;         ///< 批处理大小
    bool use_fp16 = true;           ///< 使用 FP16 (ANE 优化)
    bool normalize_output = false;  ///< 输出归一化
};

//=============================================================================
// 嵌入变换器
//=============================================================================

/**
 * 嵌入变换器 - 通过神经网络变换向量
 *
 * 用途:
 * - 领域适配: 调整通用嵌入到特定领域
 * - 降维: 减少向量维度以节省存储
 * - 对齐: 对齐不同模型的嵌入空间
 */
class EmbeddingTransformer {
public:
    using Ptr = std::shared_ptr<EmbeddingTransformer>;

    virtual ~EmbeddingTransformer() = default;

    //=========================================================================
    // 创建
    //=========================================================================

    /**
     * 从模型创建变换器
     * @param model 变换模型 (MLP, Linear 等)
     * @param config 配置
     * @return 变换器指针
     */
    static Ptr create(Model::Ptr model,
                      const TransformConfig& config = {});

    /**
     * 创建线性变换器 (无模型，使用权重矩阵)
     * @param weights 权重矩阵 [output_dim x input_dim]
     * @param bias 偏置向量 [output_dim] (可选)
     * @param input_dim 输入维度
     * @param output_dim 输出维度
     */
    static Ptr create_linear(const float* weights,
                             const float* bias,
                             size_t input_dim,
                             size_t output_dim);

    /**
     * 创建投影变换器 (PCA 风格降维)
     * @param projection_matrix 投影矩阵 [output_dim x input_dim]
     * @param mean 均值向量 [input_dim] (用于中心化)
     */
    static Ptr create_projection(const float* projection_matrix,
                                 const float* mean,
                                 size_t input_dim,
                                 size_t output_dim);

    //=========================================================================
    // 变换操作
    //=========================================================================

    /**
     * 变换嵌入向量
     * @param input 输入向量 [batch_size x input_dim]
     * @param batch_size 批次大小
     * @param output 输出向量 [batch_size x output_dim]
     */
    virtual void transform(const float* input, size_t batch_size,
                           float* output) = 0;

    /**
     * 变换 (Tensor 版本)
     * @param input 输入张量 [batch_size x input_dim]
     * @param output 输出张量 [batch_size x output_dim]
     */
    virtual void transform(const Tensor& input, Tensor& output) = 0;

    /**
     * 变换单个向量
     * @param input 输入向量 [input_dim]
     * @param output 输出向量 [output_dim]
     */
    virtual void transform_single(const float* input, float* output) = 0;

    //=========================================================================
    // 信息查询
    //=========================================================================

    /**
     * 获取输入维度
     */
    virtual size_t input_dim() const = 0;

    /**
     * 获取输出维度
     */
    virtual size_t output_dim() const = 0;

    /**
     * 获取变换类型
     */
    virtual TransformType type() const = 0;

    /**
     * 获取关联的模型
     */
    virtual Model::Ptr model() const = 0;

    /**
     * 获取配置
     */
    virtual const TransformConfig& config() const = 0;

protected:
    EmbeddingTransformer() = default;
};

//=============================================================================
// 多阶段变换流水线
//=============================================================================

/**
 * 变换流水线 - 串联多个变换器
 */
class TransformPipeline {
public:
    using Ptr = std::shared_ptr<TransformPipeline>;

    /**
     * 创建空流水线
     */
    static Ptr create();

    /**
     * 添加变换阶段
     */
    TransformPipeline& add(EmbeddingTransformer::Ptr transformer);

    /**
     * 执行完整流水线
     */
    void transform(const float* input, size_t batch_size, float* output);

    /**
     * 获取最终输入维度
     */
    size_t input_dim() const;

    /**
     * 获取最终输出维度
     */
    size_t output_dim() const;

    /**
     * 获取阶段数
     */
    size_t num_stages() const;

private:
    TransformPipeline() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vector_db
} // namespace thunderduck

#endif // THUNDERDUCK_VECTOR_DB_TRANSFORM_H
