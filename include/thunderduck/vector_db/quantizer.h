/**
 * ThunderDuck Vector Database - Neural Quantizer Interface
 *
 * @file quantizer.h
 * @version V20
 * @date 2026-01-27
 *
 * 神经网络量化编解码器，用于向量压缩
 */

#ifndef THUNDERDUCK_VECTOR_DB_QUANTIZER_H
#define THUNDERDUCK_VECTOR_DB_QUANTIZER_H

#include "../npu/model.h"
#include "../npu/tensor.h"
#include <memory>
#include <cstddef>
#include <cstdint>

namespace thunderduck {
namespace vector_db {

using npu::Model;
using npu::Tensor;

//=============================================================================
// 量化类型
//=============================================================================

/**
 * 量化方法类型
 */
enum class QuantizerType {
    // 传统方法
    SCALAR,         ///< 标量量化 (SQ8/SQ4)
    PRODUCT,        ///< 乘积量化 (PQ)
    OPQ,            ///< 优化乘积量化

    // 神经网络方法
    NEURAL_PQ,      ///< 神经乘积量化
    AUTOENCODER,    ///< 自编码器量化
    VQ_VAE          ///< VQ-VAE 量化
};

//=============================================================================
// 量化配置
//=============================================================================

/**
 * 量化配置
 */
struct QuantizerConfig {
    size_t code_size = 32;          ///< 压缩后码字大小 (字节)
    size_t batch_size = 256;        ///< 批处理大小
    bool use_fp16 = true;           ///< 使用 FP16 (ANE 优化)
    size_t num_subvectors = 8;      ///< 子向量数 (PQ)
    size_t codebook_size = 256;     ///< 码本大小 (PQ)
};

//=============================================================================
// 神经量化器
//=============================================================================

/**
 * 神经量化编解码器
 *
 * 使用神经网络进行向量压缩和解压。
 * 相比传统量化方法 (PQ, SQ)，神经方法通常能在相同压缩比下
 * 保持更高的检索精度。
 *
 * 典型压缩比:
 * - 768 dim float (3072 bytes) → 32 bytes = 96x 压缩
 * - 损失: ~5% recall@10 (vs 无压缩)
 */
class NeuralQuantizer {
public:
    using Ptr = std::shared_ptr<NeuralQuantizer>;

    virtual ~NeuralQuantizer() = default;

    //=========================================================================
    // 创建
    //=========================================================================

    /**
     * 从编码器/解码器模型创建
     * @param encoder 编码器模型 (向量 → 码字)
     * @param decoder 解码器模型 (码字 → 向量)
     * @param config 配置
     */
    static Ptr create(Model::Ptr encoder,
                      Model::Ptr decoder,
                      const QuantizerConfig& config = {});

    /**
     * 从单一 Autoencoder 模型创建
     * @param autoencoder 自编码器模型 (包含编码和解码)
     */
    static Ptr create_autoencoder(Model::Ptr autoencoder,
                                  const QuantizerConfig& config = {});

    /**
     * 创建传统标量量化器 (对比基线)
     * @param bits 量化位数 (4 或 8)
     */
    static Ptr create_scalar(size_t bits = 8);

    /**
     * 创建传统乘积量化器 (对比基线)
     * @param num_subvectors 子向量数
     * @param codebook_size 码本大小
     */
    static Ptr create_product_quantizer(size_t num_subvectors = 8,
                                        size_t codebook_size = 256);

    //=========================================================================
    // 编码 (量化)
    //=========================================================================

    /**
     * 量化向量
     * @param vectors 原始向量 [num_vectors x dim]
     * @param num_vectors 向量数量
     * @param codes 量化码 [num_vectors x code_size]
     */
    virtual void encode(const float* vectors, size_t num_vectors,
                        uint8_t* codes) = 0;

    /**
     * 量化 (Tensor 版本)
     * @param vectors 输入张量 [num_vectors x dim]
     * @param codes 输出张量 [num_vectors x code_size]
     */
    virtual void encode(const Tensor& vectors, Tensor& codes) = 0;

    /**
     * 量化单个向量
     */
    virtual void encode_single(const float* vector, uint8_t* code) = 0;

    //=========================================================================
    // 解码 (反量化)
    //=========================================================================

    /**
     * 反量化
     * @param codes 量化码 [num_vectors x code_size]
     * @param num_vectors 向量数量
     * @param vectors 重建向量 [num_vectors x dim]
     */
    virtual void decode(const uint8_t* codes, size_t num_vectors,
                        float* vectors) = 0;

    /**
     * 反量化 (Tensor 版本)
     */
    virtual void decode(const Tensor& codes, Tensor& vectors) = 0;

    /**
     * 反量化单个向量
     */
    virtual void decode_single(const uint8_t* code, float* vector) = 0;

    //=========================================================================
    // 压缩空间距离计算
    //=========================================================================

    /**
     * 在压缩空间计算距离 (无需解码)
     * @param query 查询向量 (原始)
     * @param codes 量化码
     * @param num_codes 码字数量
     * @param distances 输出距离
     *
     * @note 部分量化方法支持直接在压缩域计算近似距离
     */
    virtual void compute_distances(const float* query, size_t query_dim,
                                   const uint8_t* codes, size_t num_codes,
                                   float* distances) = 0;

    /**
     * 检查是否支持压缩空间距离计算
     */
    virtual bool supports_asymmetric_distance() const = 0;

    //=========================================================================
    // 信息查询
    //=========================================================================

    /**
     * 获取量化类型
     */
    virtual QuantizerType type() const = 0;

    /**
     * 获取输入向量维度
     */
    virtual size_t input_dim() const = 0;

    /**
     * 获取压缩后码字大小 (字节)
     */
    virtual size_t code_size() const = 0;

    /**
     * 获取压缩比 (原始大小 / 压缩大小)
     */
    virtual float compression_ratio() const = 0;

    /**
     * 获取配置
     */
    virtual const QuantizerConfig& config() const = 0;

    /**
     * 获取编码器模型
     */
    virtual Model::Ptr encoder_model() const = 0;

    /**
     * 获取解码器模型
     */
    virtual Model::Ptr decoder_model() const = 0;

    //=========================================================================
    // 质量评估
    //=========================================================================

    /**
     * 计算重建误差 (MSE)
     * @param original 原始向量
     * @param num_vectors 向量数量
     * @return 均方误差
     */
    virtual float compute_reconstruction_error(const float* original,
                                               size_t num_vectors) = 0;

protected:
    NeuralQuantizer() = default;
};

//=============================================================================
// 量化索引
//=============================================================================

/**
 * 量化向量索引
 *
 * 用于存储和检索量化后的向量。
 * 支持:
 * - 添加量化向量
 * - 使用原始查询检索
 * - 可选的两阶段检索 (粗排 + 精排)
 */
class QuantizedIndex {
public:
    using Ptr = std::shared_ptr<QuantizedIndex>;

    /**
     * 创建量化索引
     * @param quantizer 量化器
     * @param capacity 预期容量
     */
    static Ptr create(NeuralQuantizer::Ptr quantizer,
                      size_t capacity = 1000000);

    /**
     * 添加向量 (自动量化)
     * @param vectors 原始向量
     * @param num_vectors 向量数量
     * @param ids 向量 ID (可选)
     */
    virtual void add(const float* vectors, size_t num_vectors,
                     const uint64_t* ids = nullptr) = 0;

    /**
     * 搜索最近邻
     * @param query 查询向量 (原始)
     * @param k Top-K
     * @param out_ids 结果 ID
     * @param out_distances 结果距离
     */
    virtual void search(const float* query, size_t query_dim,
                        size_t k,
                        uint64_t* out_ids,
                        float* out_distances) = 0;

    /**
     * 获取索引大小
     */
    virtual size_t size() const = 0;

    /**
     * 获取内存使用 (字节)
     */
    virtual size_t memory_usage() const = 0;

protected:
    QuantizedIndex() = default;
};

} // namespace vector_db
} // namespace thunderduck

#endif // THUNDERDUCK_VECTOR_DB_QUANTIZER_H
