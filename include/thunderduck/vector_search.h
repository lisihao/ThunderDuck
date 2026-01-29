/**
 * ThunderDuck V20 - Vector Similarity Search
 *
 * 向量数据库引擎 @ 性能基线版
 *
 * 功能:
 * - 向量距离计算 (L2, Cosine, Inner Product)
 * - 暴力扫描搜索 (BLAS/AMX 加速)
 * - HNSW 索引 (ANN 近似搜索)
 * - FP16 量化向量支持
 * - Top-K 选择
 *
 * @version V20.1
 * @date 2026-01-28
 */

#ifndef THUNDERDUCK_VECTOR_SEARCH_H
#define THUNDERDUCK_VECTOR_SEARCH_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace thunderduck {
namespace vector {

// ============================================================================
// 距离度量类型
// ============================================================================

enum class DistanceMetric {
    L2_SQUARED,     // ||a-b||² 欧氏距离平方
    COSINE,         // 1 - cos(a,b) Cosine 距离
    INNER_PRODUCT   // -<a,b> 负内积 (用于最大内积搜索)
};

// ============================================================================
// 搜索配置
// ============================================================================

struct SearchConfig {
    DistanceMetric metric = DistanceMetric::L2_SQUARED;
    size_t k = 10;                     // Top-K 结果数
    bool use_blas = true;              // 使用 BLAS/AMX 加速
    size_t num_threads = 8;            // 并行线程数 (暴力扫描)
};

// ============================================================================
// 搜索结果
// ============================================================================

struct SearchResult {
    std::vector<uint32_t> indices;     // [k] 最近邻索引
    std::vector<float> distances;      // [k] 距离值
};

// ============================================================================
// 单向量距离函数 (SIMD 优化)
// ============================================================================

/**
 * L2 平方距离 (欧氏距离的平方)
 * ||a-b||² = Σ(a_i - b_i)²
 */
float distance_l2sq(const float* a, const float* b, size_t dim);

/**
 * Cosine 距离
 * 1 - <a,b>/(||a||·||b||)
 */
float distance_cosine(const float* a, const float* b, size_t dim);

/**
 * 负内积 (用于最大内积搜索)
 * -<a,b>
 */
float distance_inner_product(const float* a, const float* b, size_t dim);

/**
 * 通用距离函数
 */
float distance(const float* a, const float* b, size_t dim, DistanceMetric metric);

// ============================================================================
// 批量距离计算 (BLAS/AMX 加速)
// ============================================================================

/**
 * 批量计算查询向量与数据集的距离
 *
 * @param query     查询向量 [dim]
 * @param vectors   数据集矩阵 [n, dim] (行优先)
 * @param n         数据集向量数量
 * @param dim       向量维度
 * @param metric    距离度量
 * @param distances 输出距离数组 [n]
 */
void batch_distance(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
);

/**
 * BLAS 加速版批量距离计算 (利用 AMX)
 */
void batch_distance_blas(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
);

/**
 * SIMD 版批量距离计算
 */
void batch_distance_simd(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
);

/**
 * 多线程并行批量距离计算
 */
void batch_distance_parallel(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances,
    size_t num_threads
);

// ============================================================================
// 向量范数预计算 (用于 L2/Cosine 优化)
// ============================================================================

/**
 * 预计算向量范数 (||v||²)
 *
 * @param vectors   数据集矩阵 [n, dim]
 * @param n         向量数量
 * @param dim       向量维度
 * @param norms_sq  输出范数平方数组 [n]
 */
void precompute_norms_sq(
    const float* vectors,
    size_t n, size_t dim,
    float* norms_sq
);

/**
 * 使用预计算范数的批量 L2 距离
 * ||a-b||² = ||a||² + ||b||² - 2<a,b>
 */
void batch_distance_l2sq_with_norms(
    const float* query,
    float query_norm_sq,
    const float* vectors,
    const float* vector_norms_sq,
    size_t n, size_t dim,
    float* distances
);

// ============================================================================
// Top-K 选择
// ============================================================================

/**
 * 从距离数组中选择 Top-K 最小距离
 *
 * @param distances 距离数组 [n]
 * @param n         数组大小
 * @param k         选择数量
 * @param out_indices  输出索引 [k]
 * @param out_distances 输出距离 [k]
 */
void topk_smallest(
    const float* distances,
    size_t n, size_t k,
    uint32_t* out_indices,
    float* out_distances
);

// ============================================================================
// 统一搜索接口
// ============================================================================

/**
 * 暴力扫描向量搜索
 *
 * @param query     查询向量 [dim]
 * @param vectors   数据集矩阵 [n, dim]
 * @param n         数据集向量数量
 * @param dim       向量维度
 * @param config    搜索配置
 * @return 搜索结果 (Top-K 索引和距离)
 */
SearchResult brute_force_search(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    const SearchConfig& config
);

/**
 * 批量查询搜索
 *
 * @param queries   查询向量矩阵 [num_queries, dim]
 * @param num_queries 查询数量
 * @param vectors   数据集矩阵 [n, dim]
 * @param n         数据集向量数量
 * @param dim       向量维度
 * @param config    搜索配置
 * @param results   输出结果数组 [num_queries]
 */
void batch_search(
    const float* queries,
    size_t num_queries,
    const float* vectors,
    size_t n, size_t dim,
    const SearchConfig& config,
    SearchResult* results
);

// ============================================================================
// 版本信息
// ============================================================================

/**
 * 获取 V20 版本信息
 */
const char* get_vector_search_version();

/**
 * 距离度量名称
 */
const char* metric_name(DistanceMetric metric);

// ============================================================================
// P1: VectorIndex - 预计算范数优化的向量索引
// ============================================================================

/**
 * 向量索引 (带预计算范数)
 *
 * 优化特性:
 * - 预计算向量范数，避免重复计算
 * - BLAS/AMX 加速的批量距离计算
 * - 支持增量添加向量
 */
class VectorIndex {
public:
    /**
     * 创建空索引
     * @param dim 向量维度
     * @param metric 距离度量
     */
    VectorIndex(size_t dim, DistanceMetric metric = DistanceMetric::L2_SQUARED);

    /**
     * 从现有数据创建索引
     * @param vectors 向量数据 [n, dim]
     * @param n 向量数量
     * @param dim 向量维度
     * @param metric 距离度量
     */
    VectorIndex(const float* vectors, size_t n, size_t dim,
                DistanceMetric metric = DistanceMetric::L2_SQUARED);

    ~VectorIndex();

    // 禁止拷贝
    VectorIndex(const VectorIndex&) = delete;
    VectorIndex& operator=(const VectorIndex&) = delete;

    // 允许移动
    VectorIndex(VectorIndex&& other) noexcept;
    VectorIndex& operator=(VectorIndex&& other) noexcept;

    /**
     * 添加向量
     * @param vector 向量数据 [dim]
     * @return 向量 ID
     */
    uint32_t add(const float* vector);

    /**
     * 批量添加向量
     * @param vectors 向量数据 [n, dim]
     * @param n 向量数量
     */
    void add_batch(const float* vectors, size_t n);

    /**
     * 搜索最近邻
     * @param query 查询向量 [dim]
     * @param k Top-K
     * @return 搜索结果
     */
    SearchResult search(const float* query, size_t k) const;

    /**
     * 批量搜索
     * @param queries 查询向量 [num_queries, dim]
     * @param num_queries 查询数量
     * @param k Top-K
     * @param results 输出结果 [num_queries]
     */
    void search_batch(const float* queries, size_t num_queries,
                      size_t k, SearchResult* results) const;

    // 访问器
    size_t size() const { return size_; }
    size_t dim() const { return dim_; }
    DistanceMetric metric() const { return metric_; }
    const float* data() const { return vectors_.data(); }
    const float* norms() const { return norms_sq_.data(); }

private:
    size_t dim_;
    size_t size_;
    DistanceMetric metric_;
    std::vector<float> vectors_;      // [n, dim]
    std::vector<float> norms_sq_;     // [n] 预计算范数平方
};

// ============================================================================
// P2: HNSW 索引 (Hierarchical Navigable Small World)
// ============================================================================

/**
 * HNSW 索引配置
 */
struct HNSWConfig {
    size_t M = 16;                    // 每层最大连接数
    size_t ef_construction = 200;     // 构建时搜索宽度
    size_t ef_search = 64;            // 搜索时搜索宽度
    size_t max_elements = 0;          // 最大元素数 (0=自动)
    unsigned seed = 42;               // 随机种子
};

/**
 * HNSW 向量索引
 *
 * 特性:
 * - 近似最近邻搜索 (ANN)
 * - 对数复杂度查询
 * - 支持动态插入
 *
 * 基于 HNSW 算法 (Malkov & Yashunin, 2016)
 */
class HNSWIndex {
public:
    /**
     * 创建 HNSW 索引
     * @param dim 向量维度
     * @param metric 距离度量
     * @param config 配置参数
     */
    HNSWIndex(size_t dim, DistanceMetric metric = DistanceMetric::L2_SQUARED,
              const HNSWConfig& config = HNSWConfig{});

    /**
     * 从现有数据构建索引
     */
    HNSWIndex(const float* vectors, size_t n, size_t dim,
              DistanceMetric metric = DistanceMetric::L2_SQUARED,
              const HNSWConfig& config = HNSWConfig{});

    ~HNSWIndex();

    // 禁止拷贝
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    /**
     * 添加向量
     * @param vector 向量数据 [dim]
     * @return 向量 ID
     */
    uint32_t add(const float* vector);

    /**
     * 批量添加向量
     * @param vectors 向量数据 [n, dim]
     * @param n 向量数量
     */
    void add_batch(const float* vectors, size_t n);

    /**
     * 搜索最近邻 (ANN)
     * @param query 查询向量 [dim]
     * @param k Top-K
     * @param ef_search 搜索宽度 (0=使用默认)
     * @return 搜索结果
     */
    SearchResult search(const float* query, size_t k, size_t ef_search = 0) const;

    /**
     * 批量搜索
     */
    void search_batch(const float* queries, size_t num_queries,
                      size_t k, SearchResult* results,
                      size_t ef_search = 0) const;

    /**
     * 设置搜索宽度
     */
    void set_ef_search(size_t ef) { config_.ef_search = ef; }

    // 访问器
    size_t size() const { return size_; }
    size_t dim() const { return dim_; }
    DistanceMetric metric() const { return metric_; }
    const HNSWConfig& config() const { return config_; }

private:
    struct Node;
    struct Layer;

    size_t dim_;
    size_t size_;
    DistanceMetric metric_;
    HNSWConfig config_;

    std::vector<float> vectors_;           // [n, dim]
    std::vector<float> norms_sq_;          // [n]
    std::vector<std::vector<std::vector<uint32_t>>> layers_;  // [level][node][neighbors]
    std::vector<int> node_levels_;         // [n] 每个节点的层级
    int max_level_;                        // 当前最大层级
    uint32_t entry_point_;                 // 入口点

    int random_level();
    void insert_node(uint32_t id, const float* vector);
    std::vector<std::pair<float, uint32_t>> search_layer(
        const float* query, uint32_t entry, size_t ef, int level) const;
};

// ============================================================================
// P3: FP16 量化向量支持
// ============================================================================

/**
 * FP16 向量索引
 *
 * 优化特性:
 * - 内存减半 (FP32 → FP16)
 * - ANE 原生格式，可能获得硬件加速
 * - 适合大规模向量存储
 */
class VectorIndexFP16 {
public:
    /**
     * 创建 FP16 索引
     * @param dim 向量维度
     * @param metric 距离度量
     */
    VectorIndexFP16(size_t dim, DistanceMetric metric = DistanceMetric::L2_SQUARED);

    /**
     * 从 FP32 数据创建 FP16 索引 (自动量化)
     */
    VectorIndexFP16(const float* vectors, size_t n, size_t dim,
                    DistanceMetric metric = DistanceMetric::L2_SQUARED);

    ~VectorIndexFP16();

    // 禁止拷贝
    VectorIndexFP16(const VectorIndexFP16&) = delete;
    VectorIndexFP16& operator=(const VectorIndexFP16&) = delete;

    /**
     * 添加向量 (FP32 输入，自动量化)
     */
    uint32_t add(const float* vector);

    /**
     * 批量添加向量
     */
    void add_batch(const float* vectors, size_t n);

    /**
     * 搜索最近邻
     */
    SearchResult search(const float* query, size_t k) const;

    /**
     * 批量搜索
     */
    void search_batch(const float* queries, size_t num_queries,
                      size_t k, SearchResult* results) const;

    // 访问器
    size_t size() const { return size_; }
    size_t dim() const { return dim_; }
    DistanceMetric metric() const { return metric_; }

    /**
     * 获取内存使用量 (字节)
     */
    size_t memory_usage() const;

private:
    size_t dim_;
    size_t size_;
    DistanceMetric metric_;
    std::vector<uint16_t> vectors_fp16_;  // [n, dim] FP16 格式
    std::vector<float> norms_sq_;         // [n] FP32 范数 (精度保持)
};

// ============================================================================
// FP16 转换工具
// ============================================================================

/**
 * FP32 → FP16 转换
 */
uint16_t float_to_half(float f);

/**
 * FP16 → FP32 转换
 */
float half_to_float(uint16_t h);

/**
 * 批量 FP32 → FP16 转换
 */
void float_to_half_batch(const float* src, uint16_t* dst, size_t n);

/**
 * 批量 FP16 → FP32 转换
 */
void half_to_float_batch(const uint16_t* src, float* dst, size_t n);

/**
 * FP16 向量距离计算 (内部转换为 FP32)
 */
float distance_l2sq_fp16(const uint16_t* a, const uint16_t* b, size_t dim);
float distance_inner_product_fp16(const uint16_t* a, const uint16_t* b, size_t dim);

} // namespace vector
} // namespace thunderduck

#endif // THUNDERDUCK_VECTOR_SEARCH_H
