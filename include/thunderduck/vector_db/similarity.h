/**
 * ThunderDuck Vector Database - Similarity Interface
 *
 * @file similarity.h
 * @version V20
 * @date 2026-01-27
 *
 * 向量相似度计算，支持传统方法和学习型方法
 */

#ifndef THUNDERDUCK_VECTOR_DB_SIMILARITY_H
#define THUNDERDUCK_VECTOR_DB_SIMILARITY_H

#include "../npu/model.h"
#include "../npu/tensor.h"
#include <memory>
#include <cstddef>

namespace thunderduck {
namespace vector_db {

using npu::Model;
using npu::Tensor;

//=============================================================================
// 相似度类型
//=============================================================================

/**
 * 相似度计算类型
 */
enum class SimilarityType {
    // 传统方法 (CPU AMX / GPU 加速)
    COSINE,         ///< 余弦相似度
    EUCLIDEAN,      ///< 欧氏距离 (L2)
    DOT_PRODUCT,    ///< 点积
    MANHATTAN,      ///< 曼哈顿距离 (L1)

    // 学习型方法 (NPU 加速)
    LEARNED,        ///< 学习型相似度 (MLP)
    CROSS_ENCODER   ///< Cross-encoder (Transformer)
};

/**
 * 距离度量 (相似度的逆)
 */
enum class DistanceMetric {
    L2,             ///< 欧氏距离
    L2_SQUARED,     ///< 欧氏距离平方 (更快)
    L1,             ///< 曼哈顿距离
    COSINE,         ///< 余弦距离 (1 - cosine_sim)
    IP              ///< 内积距离 (负内积)
};

//=============================================================================
// 相似度计算配置
//=============================================================================

/**
 * 相似度计算配置
 */
struct SimilarityConfig {
    bool normalize_vectors = false;     ///< 是否预归一化向量
    bool use_fp16 = false;              ///< 使用 FP16 计算
    size_t batch_size = 64;             ///< 批处理大小
    size_t prefetch_distance = 8;       ///< 预取距离

    // 学习型相似度配置
    size_t model_max_batch = 64;        ///< 模型最大批次
};

//=============================================================================
// 相似度计算器
//=============================================================================

/**
 * 向量相似度计算器
 *
 * 统一接口支持:
 * - 传统相似度 (余弦、欧氏、点积) - 使用 AMX/GPU 加速
 * - 学习型相似度 (MLP) - 使用 NPU 加速
 * - Cross-encoder - 使用 NPU 加速
 */
class SimilarityComputer {
public:
    using Ptr = std::shared_ptr<SimilarityComputer>;

    virtual ~SimilarityComputer() = default;

    //=========================================================================
    // 创建
    //=========================================================================

    /**
     * 创建相似度计算器
     * @param type 相似度类型
     * @param model 学习模型 (学习型相似度需要)
     * @param config 配置
     * @return 计算器指针
     */
    static Ptr create(SimilarityType type,
                      Model::Ptr model = nullptr,
                      const SimilarityConfig& config = {});

    /**
     * 创建传统相似度计算器 (便捷方法)
     */
    static Ptr create_cosine(const SimilarityConfig& config = {});
    static Ptr create_euclidean(const SimilarityConfig& config = {});
    static Ptr create_dot_product(const SimilarityConfig& config = {});

    /**
     * 创建学习型相似度计算器
     * @param model 学习模型 (.mlmodelc)
     */
    static Ptr create_learned(Model::Ptr model,
                              const SimilarityConfig& config = {});

    /**
     * 创建 Cross-encoder 计算器
     * @param model Cross-encoder 模型
     */
    static Ptr create_cross_encoder(Model::Ptr model,
                                    const SimilarityConfig& config = {});

    //=========================================================================
    // 单查询批量计算
    //=========================================================================

    /**
     * 批量计算相似度
     * @param query 查询向量 [dim]
     * @param candidates 候选向量 [num_candidates x dim]
     * @param dim 向量维度
     * @param num_candidates 候选数量
     * @param out_scores 输出相似度分数 [num_candidates]
     */
    virtual void compute_batch(
        const float* query,
        const float* candidates,
        size_t dim,
        size_t num_candidates,
        float* out_scores) = 0;

    /**
     * 批量计算 (Tensor 版本)
     * @param query 查询张量 [dim] 或 [1 x dim]
     * @param candidates 候选张量 [num_candidates x dim]
     * @param out_scores 输出张量 [num_candidates]
     */
    virtual void compute_batch(
        const Tensor& query,
        const Tensor& candidates,
        Tensor& out_scores) = 0;

    //=========================================================================
    // 多查询批量计算
    //=========================================================================

    /**
     * 多查询批量计算
     * @param queries 查询向量 [num_queries x dim]
     * @param candidates 候选向量 [num_candidates x dim]
     * @param dim 向量维度
     * @param num_queries 查询数量
     * @param num_candidates 候选数量
     * @param out_scores 输出相似度矩阵 [num_queries x num_candidates]
     */
    virtual void compute_multi_query(
        const float* queries,
        const float* candidates,
        size_t dim,
        size_t num_queries,
        size_t num_candidates,
        float* out_scores) = 0;

    /**
     * 多查询批量计算 (Tensor 版本)
     */
    virtual void compute_multi_query(
        const Tensor& queries,
        const Tensor& candidates,
        Tensor& out_scores) = 0;

    //=========================================================================
    // Top-K 计算
    //=========================================================================

    /**
     * 计算并返回 Top-K 最相似的候选
     * @param query 查询向量
     * @param candidates 候选向量
     * @param dim 向量维度
     * @param num_candidates 候选数量
     * @param k Top-K 数量
     * @param out_scores Top-K 分数
     * @param out_indices Top-K 索引
     */
    virtual void compute_topk(
        const float* query,
        const float* candidates,
        size_t dim,
        size_t num_candidates,
        size_t k,
        float* out_scores,
        uint32_t* out_indices) = 0;

    //=========================================================================
    // 信息查询
    //=========================================================================

    /**
     * 获取相似度类型
     */
    virtual SimilarityType type() const = 0;

    /**
     * 是否是学习型相似度
     */
    virtual bool is_learned() const = 0;

    /**
     * 获取关联的模型 (学习型)
     */
    virtual Model::Ptr model() const = 0;

    /**
     * 获取配置
     */
    virtual const SimilarityConfig& config() const = 0;

protected:
    SimilarityComputer() = default;
};

//=============================================================================
// 便捷函数
//=============================================================================

/**
 * 快速计算余弦相似度
 */
void cosine_similarity(
    const float* query, const float* candidates,
    size_t dim, size_t num_candidates, float* scores);

/**
 * 快速计算欧氏距离
 */
void euclidean_distance(
    const float* query, const float* candidates,
    size_t dim, size_t num_candidates, float* distances);

/**
 * 快速计算点积
 */
void dot_product(
    const float* query, const float* candidates,
    size_t dim, size_t num_candidates, float* scores);

/**
 * 归一化向量 (就地)
 */
void normalize_vectors(float* vectors, size_t num_vectors, size_t dim);

} // namespace vector_db
} // namespace thunderduck

#endif // THUNDERDUCK_VECTOR_DB_SIMILARITY_H
