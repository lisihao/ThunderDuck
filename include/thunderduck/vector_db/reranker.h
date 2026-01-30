/**
 * ThunderDuck Vector Database - Reranker Interface
 *
 * @file reranker.h
 * @version V20
 * @date 2026-01-27
 *
 * 重排序器，使用 Cross-encoder 精确排序
 */

#ifndef THUNDERDUCK_VECTOR_DB_RERANKER_H
#define THUNDERDUCK_VECTOR_DB_RERANKER_H

#include "../npu/model.h"
#include "../npu/tensor.h"
#include <memory>
#include <cstddef>
#include <vector>
#include <string>

namespace thunderduck {
namespace vector_db {

using npu::Model;
using npu::Tensor;

//=============================================================================
// 重排序配置
//=============================================================================

/**
 * 重排序配置
 */
struct RerankerConfig {
    size_t batch_size = 32;         ///< 批处理大小
    bool use_fp16 = true;           ///< 使用 FP16 (ANE 优化)
    size_t max_candidates = 100;    ///< 最大候选数量
    bool normalize_scores = true;   ///< 归一化分数到 [0, 1]
};

//=============================================================================
// 重排序结果
//=============================================================================

/**
 * 单个重排序结果
 */
struct RerankResult {
    uint32_t original_index;    ///< 原始索引
    float score;                ///< 重排序分数
};

//=============================================================================
// 重排序器
//=============================================================================

/**
 * 重排序器 - 使用 Cross-encoder 精确排序
 *
 * Cross-encoder 比 Bi-encoder (向量相似度) 更精确，
 * 但计算成本更高，适合对 Top-K 候选进行精排。
 *
 * 工作流程:
 * 1. 粗排: 使用向量相似度快速检索 Top-K (e.g., 100)
 * 2. 精排: 使用 Cross-encoder 重排序 Top-K
 * 3. 返回: 最终 Top-N (e.g., 10)
 */
class Reranker {
public:
    using Ptr = std::shared_ptr<Reranker>;

    virtual ~Reranker() = default;

    //=========================================================================
    // 创建
    //=========================================================================

    /**
     * 从 Cross-encoder 模型创建
     * @param model Cross-encoder 模型
     * @param config 配置
     * @return 重排序器指针
     */
    static Ptr create(Model::Ptr model,
                      const RerankerConfig& config = {});

    //=========================================================================
    // 向量重排序 (嵌入输入)
    //=========================================================================

    /**
     * 重排序候选结果 (向量输入)
     * @param query 查询向量 [query_dim]
     * @param query_dim 查询维度
     * @param candidates 候选向量 [num_candidates x candidate_dim]
     * @param candidate_dim 候选维度
     * @param num_candidates 候选数量
     * @param out_scores 重排序分数 [num_candidates]
     * @param out_indices 排序后的索引 [num_candidates]
     */
    virtual void rerank(const float* query, size_t query_dim,
                        const float* candidates, size_t candidate_dim,
                        size_t num_candidates,
                        float* out_scores,
                        uint32_t* out_indices) = 0;

    /**
     * 重排序 (Tensor 版本)
     */
    virtual void rerank(const Tensor& query,
                        const Tensor& candidates,
                        Tensor& out_scores,
                        Tensor& out_indices) = 0;

    /**
     * 重排序并返回 Top-K
     * @param query 查询向量
     * @param candidates 候选向量
     * @param num_candidates 候选数量
     * @param k 返回数量
     * @param results Top-K 结果
     */
    virtual void rerank_topk(const float* query, size_t query_dim,
                             const float* candidates, size_t candidate_dim,
                             size_t num_candidates,
                             size_t k,
                             std::vector<RerankResult>& results) = 0;

    //=========================================================================
    // 文本重排序 (需要 tokenizer)
    //=========================================================================

    /**
     * 文本重排序 (适用于文本 Cross-encoder)
     * @param query_text 查询文本
     * @param candidate_texts 候选文本数组
     * @param num_candidates 候选数量
     * @param out_scores 分数
     * @param out_indices 排序索引
     *
     * @note 需要模型内置 tokenizer 或外部提供
     */
    virtual void rerank_text(const char* query_text,
                             const char* const* candidate_texts,
                             size_t num_candidates,
                             float* out_scores,
                             uint32_t* out_indices) = 0;

    //=========================================================================
    // 批量重排序
    //=========================================================================

    /**
     * 批量重排序 (多个查询)
     * @param queries 查询向量 [num_queries x query_dim]
     * @param candidates 候选向量 (所有查询共享) [num_candidates x candidate_dim]
     * @param num_queries 查询数量
     * @param num_candidates 候选数量
     * @param out_scores 分数矩阵 [num_queries x num_candidates]
     */
    virtual void rerank_batch(const float* queries, size_t query_dim,
                              const float* candidates, size_t candidate_dim,
                              size_t num_queries,
                              size_t num_candidates,
                              float* out_scores) = 0;

    //=========================================================================
    // 信息查询
    //=========================================================================

    /**
     * 获取关联的模型
     */
    virtual Model::Ptr model() const = 0;

    /**
     * 获取配置
     */
    virtual const RerankerConfig& config() const = 0;

    /**
     * 检查是否支持文本输入
     */
    virtual bool supports_text_input() const = 0;

    /**
     * 获取预期的查询维度
     */
    virtual size_t expected_query_dim() const = 0;

    /**
     * 获取预期的候选维度
     */
    virtual size_t expected_candidate_dim() const = 0;

protected:
    Reranker() = default;
};

//=============================================================================
// 两阶段检索器
//=============================================================================

/**
 * 两阶段检索配置
 */
struct TwoStageConfig {
    size_t first_stage_k = 100;     ///< 粗排 Top-K
    size_t final_k = 10;            ///< 精排返回数量
};

/**
 * 两阶段检索器 - 结合向量搜索和重排序
 *
 * 封装了常见的 "粗排 + 精排" 流程
 */
class TwoStageRetriever {
public:
    using Ptr = std::shared_ptr<TwoStageRetriever>;

    /**
     * 创建两阶段检索器
     * @param similarity 相似度计算器 (粗排)
     * @param reranker 重排序器 (精排)
     * @param config 配置
     */
    static Ptr create(std::shared_ptr<class SimilarityComputer> similarity,
                      Reranker::Ptr reranker,
                      const TwoStageConfig& config = {});

    /**
     * 执行两阶段检索
     * @param query 查询向量
     * @param query_dim 查询维度
     * @param candidates 候选向量
     * @param candidate_dim 候选维度
     * @param num_candidates 候选数量
     * @param results 最终结果
     */
    virtual void retrieve(const float* query, size_t query_dim,
                          const float* candidates, size_t candidate_dim,
                          size_t num_candidates,
                          std::vector<RerankResult>& results) = 0;

protected:
    TwoStageRetriever() = default;
};

} // namespace vector_db
} // namespace thunderduck

#endif // THUNDERDUCK_VECTOR_DB_RERANKER_H
