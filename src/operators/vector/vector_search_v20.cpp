/**
 * ThunderDuck V20 - Vector Similarity Search Implementation
 *
 * 向量数据库引擎 @ 性能基线版
 *
 * 优化特性:
 * - ARM Neon SIMD 距离计算
 * - BLAS/AMX 批量矩阵运算
 * - 多线程并行处理
 * - 向量范数预计算优化
 *
 * @version V20.0
 * @date 2026-01-28
 */

#include "thunderduck/vector_search.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <queue>
#include <random>
#include <functional>
#include <unordered_set>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define THUNDERDUCK_HAS_BLAS 1
#endif

namespace thunderduck {
namespace vector {

// ============================================================================
// 版本信息
// ============================================================================

const char* get_vector_search_version() {
    return "V20.0 Vector Database Engine (Performance Baseline)";
}

const char* metric_name(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::L2_SQUARED:    return "L2_SQUARED";
        case DistanceMetric::COSINE:        return "COSINE";
        case DistanceMetric::INNER_PRODUCT: return "INNER_PRODUCT";
    }
    return "UNKNOWN";
}

// ============================================================================
// 单向量距离函数 - ARM Neon SIMD 优化
// ============================================================================

float distance_l2sq(const float* a, const float* b, size_t dim) {
#ifdef __aarch64__
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;

    // 主循环: 16 元素/迭代
    for (; i + 15 < dim; i += 16) {
        // 加载
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        // 差值
        float32x4_t d0 = vsubq_f32(a0, b0);
        float32x4_t d1 = vsubq_f32(a1, b1);
        float32x4_t d2 = vsubq_f32(a2, b2);
        float32x4_t d3 = vsubq_f32(a3, b3);

        // 累加平方
        sum_vec = vfmaq_f32(sum_vec, d0, d0);
        sum_vec = vfmaq_f32(sum_vec, d1, d1);
        sum_vec = vfmaq_f32(sum_vec, d2, d2);
        sum_vec = vfmaq_f32(sum_vec, d3, d3);
    }

    // 4 元素/迭代
    for (; i + 3 < dim; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(av, bv);
        sum_vec = vfmaq_f32(sum_vec, diff, diff);
    }

    // 归约
    float sum = vaddvq_f32(sum_vec);

    // 处理剩余元素
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
#else
    // 标量回退
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

float distance_inner_product(const float* a, const float* b, size_t dim) {
#ifdef __aarch64__
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;

    // 主循环: 16 元素/迭代
    for (; i + 15 < dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum_vec = vfmaq_f32(sum_vec, a0, b0);
        sum_vec = vfmaq_f32(sum_vec, a1, b1);
        sum_vec = vfmaq_f32(sum_vec, a2, b2);
        sum_vec = vfmaq_f32(sum_vec, a3, b3);
    }

    // 4 元素/迭代
    for (; i + 3 < dim; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        sum_vec = vfmaq_f32(sum_vec, av, bv);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < dim; i++) {
        sum += a[i] * b[i];
    }

    // 返回负内积 (用于最小化搜索)
    return -sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return -sum;
#endif
}

float distance_cosine(const float* a, const float* b, size_t dim) {
#ifdef __aarch64__
    float32x4_t dot_vec = vdupq_n_f32(0.0f);
    float32x4_t norm_a_vec = vdupq_n_f32(0.0f);
    float32x4_t norm_b_vec = vdupq_n_f32(0.0f);

    size_t i = 0;

    // 主循环: 8 元素/迭代
    for (; i + 7 < dim; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        dot_vec = vfmaq_f32(dot_vec, a0, b0);
        dot_vec = vfmaq_f32(dot_vec, a1, b1);

        norm_a_vec = vfmaq_f32(norm_a_vec, a0, a0);
        norm_a_vec = vfmaq_f32(norm_a_vec, a1, a1);

        norm_b_vec = vfmaq_f32(norm_b_vec, b0, b0);
        norm_b_vec = vfmaq_f32(norm_b_vec, b1, b1);
    }

    // 4 元素/迭代
    for (; i + 3 < dim; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);

        dot_vec = vfmaq_f32(dot_vec, av, bv);
        norm_a_vec = vfmaq_f32(norm_a_vec, av, av);
        norm_b_vec = vfmaq_f32(norm_b_vec, bv, bv);
    }

    float dot = vaddvq_f32(dot_vec);
    float norm_a_sq = vaddvq_f32(norm_a_vec);
    float norm_b_sq = vaddvq_f32(norm_b_vec);

    for (; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    float norm_product = std::sqrt(norm_a_sq * norm_b_sq);
    if (norm_product < 1e-10f) return 1.0f;

    float cosine_sim = dot / norm_product;
    return 1.0f - cosine_sim;  // Cosine 距离
#else
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;

    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    float norm_product = std::sqrt(norm_a_sq * norm_b_sq);
    if (norm_product < 1e-10f) return 1.0f;

    return 1.0f - (dot / norm_product);
#endif
}

float distance(const float* a, const float* b, size_t dim, DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::L2_SQUARED:
            return distance_l2sq(a, b, dim);
        case DistanceMetric::COSINE:
            return distance_cosine(a, b, dim);
        case DistanceMetric::INNER_PRODUCT:
            return distance_inner_product(a, b, dim);
    }
    return 0.0f;
}

// ============================================================================
// 向量范数预计算
// ============================================================================

void precompute_norms_sq(const float* vectors, size_t n, size_t dim, float* norms_sq) {
#ifdef __aarch64__
    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t j = 0;
        for (; j + 15 < dim; j += 16) {
            float32x4_t v0 = vld1q_f32(vec + j);
            float32x4_t v1 = vld1q_f32(vec + j + 4);
            float32x4_t v2 = vld1q_f32(vec + j + 8);
            float32x4_t v3 = vld1q_f32(vec + j + 12);

            sum_vec = vfmaq_f32(sum_vec, v0, v0);
            sum_vec = vfmaq_f32(sum_vec, v1, v1);
            sum_vec = vfmaq_f32(sum_vec, v2, v2);
            sum_vec = vfmaq_f32(sum_vec, v3, v3);
        }

        for (; j + 3 < dim; j += 4) {
            float32x4_t v = vld1q_f32(vec + j);
            sum_vec = vfmaq_f32(sum_vec, v, v);
        }

        float sum = vaddvq_f32(sum_vec);
        for (; j < dim; j++) {
            sum += vec[j] * vec[j];
        }

        norms_sq[i] = sum;
    }
#else
    for (size_t i = 0; i < n; i++) {
        const float* vec = vectors + i * dim;
        float sum = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            sum += vec[j] * vec[j];
        }
        norms_sq[i] = sum;
    }
#endif
}

// ============================================================================
// 批量距离计算 - SIMD 版本
// ============================================================================

void batch_distance_simd(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
) {
    for (size_t i = 0; i < n; i++) {
        distances[i] = distance(query, vectors + i * dim, dim, metric);
    }
}

// ============================================================================
// 批量距离计算 - BLAS/AMX 加速版本
// ============================================================================

void batch_distance_blas(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
) {
#ifdef THUNDERDUCK_HAS_BLAS
    if (metric == DistanceMetric::INNER_PRODUCT) {
        // Inner Product: distances = -V @ q
        // 使用 cblas_sgemv: y = alpha * A * x + beta * y
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)n, (int)dim,
                    -1.0f,              // alpha = -1 (负内积)
                    vectors, (int)dim,  // A = vectors [n, dim]
                    query, 1,           // x = query [dim]
                    0.0f,               // beta = 0
                    distances, 1);      // y = distances [n]
        return;
    }

    if (metric == DistanceMetric::L2_SQUARED) {
        // L2²: ||a-b||² = ||a||² + ||b||² - 2<a,b>
        // 1. 计算 -2 * dot products: distances = -2 * V @ q
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)n, (int)dim,
                    -2.0f,              // alpha = -2
                    vectors, (int)dim,
                    query, 1,
                    0.0f,
                    distances, 1);

        // 2. 计算 query 范数
        float query_norm_sq = cblas_sdot((int)dim, query, 1, query, 1);

        // 3. 加上范数项
        for (size_t i = 0; i < n; i++) {
            const float* vec = vectors + i * dim;
            float vec_norm_sq = cblas_sdot((int)dim, vec, 1, vec, 1);
            distances[i] += query_norm_sq + vec_norm_sq;
        }
        return;
    }

    if (metric == DistanceMetric::COSINE) {
        // Cosine: 1 - <a,b>/(||a||·||b||)
        // 1. 计算 dot products: dots = V @ q
        std::vector<float> dots(n);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)n, (int)dim,
                    1.0f,
                    vectors, (int)dim,
                    query, 1,
                    0.0f,
                    dots.data(), 1);

        // 2. 计算 query 范数
        float query_norm = std::sqrt(cblas_sdot((int)dim, query, 1, query, 1));

        // 3. 计算 cosine 距离
        for (size_t i = 0; i < n; i++) {
            const float* vec = vectors + i * dim;
            float vec_norm = std::sqrt(cblas_sdot((int)dim, vec, 1, vec, 1));
            float norm_product = query_norm * vec_norm;
            if (norm_product < 1e-10f) {
                distances[i] = 1.0f;
            } else {
                distances[i] = 1.0f - dots[i] / norm_product;
            }
        }
        return;
    }
#endif

    // 回退到 SIMD 版本
    batch_distance_simd(query, vectors, n, dim, metric, distances);
}

// ============================================================================
// 批量距离计算 - 预计算范数版本
// ============================================================================

void batch_distance_l2sq_with_norms(
    const float* query,
    float query_norm_sq,
    const float* vectors,
    const float* vector_norms_sq,
    size_t n, size_t dim,
    float* distances
) {
#ifdef THUNDERDUCK_HAS_BLAS
    // L2²: ||a-b||² = ||a||² + ||b||² - 2<a,b>
    // distances = -2 * V @ q
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)n, (int)dim,
                -2.0f,
                vectors, (int)dim,
                query, 1,
                0.0f,
                distances, 1);

    // 加上范数项
    for (size_t i = 0; i < n; i++) {
        distances[i] += query_norm_sq + vector_norms_sq[i];
    }
#else
    batch_distance_simd(query, vectors, n, dim, DistanceMetric::L2_SQUARED, distances);
#endif
}

// ============================================================================
// 批量距离计算 - 多线程并行版本
// ============================================================================

void batch_distance_parallel(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances,
    size_t num_threads
) {
    if (n < 10000 || num_threads <= 1) {
        // 小数据量使用 BLAS
        batch_distance_blas(query, vectors, n, dim, metric, distances);
        return;
    }

    std::vector<std::thread> threads;
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);
        if (start >= n) break;

        threads.emplace_back([=]() {
            for (size_t i = start; i < end; i++) {
                distances[i] = distance(query, vectors + i * dim, dim, metric);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ============================================================================
// 统一批量距离接口
// ============================================================================

void batch_distance(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances
) {
    // 默认使用 BLAS 加速
    batch_distance_blas(query, vectors, n, dim, metric, distances);
}

// ============================================================================
// Top-K 选择 (最小堆)
// ============================================================================

void topk_smallest(
    const float* distances,
    size_t n, size_t k,
    uint32_t* out_indices,
    float* out_distances
) {
    if (k >= n) {
        // 返回所有元素
        std::vector<std::pair<float, uint32_t>> pairs(n);
        for (size_t i = 0; i < n; i++) {
            pairs[i] = {distances[i], static_cast<uint32_t>(i)};
        }
        std::sort(pairs.begin(), pairs.end());
        for (size_t i = 0; i < n; i++) {
            out_indices[i] = pairs[i].second;
            out_distances[i] = pairs[i].first;
        }
        return;
    }

    // 使用最大堆维护 Top-K 最小值
    std::priority_queue<std::pair<float, uint32_t>> max_heap;

    for (size_t i = 0; i < n; i++) {
        float dist = distances[i];

        if (max_heap.size() < k) {
            max_heap.push({dist, static_cast<uint32_t>(i)});
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({dist, static_cast<uint32_t>(i)});
        }
    }

    // 提取结果 (从大到小出堆，需要反转)
    size_t result_size = max_heap.size();
    for (size_t i = result_size; i > 0; i--) {
        out_indices[i - 1] = max_heap.top().second;
        out_distances[i - 1] = max_heap.top().first;
        max_heap.pop();
    }
}

// ============================================================================
// 暴力扫描搜索
// ============================================================================

SearchResult brute_force_search(
    const float* query,
    const float* vectors,
    size_t n, size_t dim,
    const SearchConfig& config
) {
    SearchResult result;
    result.indices.resize(config.k);
    result.distances.resize(config.k);

    // 计算所有距离
    std::vector<float> all_distances(n);

    if (config.use_blas) {
        batch_distance_blas(query, vectors, n, dim, config.metric, all_distances.data());
    } else if (config.num_threads > 1) {
        batch_distance_parallel(query, vectors, n, dim, config.metric,
                                all_distances.data(), config.num_threads);
    } else {
        batch_distance_simd(query, vectors, n, dim, config.metric, all_distances.data());
    }

    // 选择 Top-K
    topk_smallest(all_distances.data(), n, config.k,
                  result.indices.data(), result.distances.data());

    return result;
}

// ============================================================================
// 批量查询搜索
// ============================================================================

void batch_search(
    const float* queries,
    size_t num_queries,
    const float* vectors,
    size_t n, size_t dim,
    const SearchConfig& config,
    SearchResult* results
) {
    // 并行处理多个查询
    std::vector<std::thread> threads;
    size_t num_threads = std::min(config.num_threads, num_queries);

    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_queries);
        if (start >= num_queries) break;

        threads.emplace_back([=, &config]() {
            for (size_t q = start; q < end; q++) {
                results[q] = brute_force_search(
                    queries + q * dim, vectors, n, dim, config);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ============================================================================
// P1: VectorIndex 实现 (预计算范数优化)
// ============================================================================

VectorIndex::VectorIndex(size_t dim, DistanceMetric metric)
    : dim_(dim), size_(0), metric_(metric) {
}

VectorIndex::VectorIndex(const float* vectors, size_t n, size_t dim, DistanceMetric metric)
    : dim_(dim), size_(n), metric_(metric) {
    // 复制向量数据
    vectors_.resize(n * dim);
    std::memcpy(vectors_.data(), vectors, n * dim * sizeof(float));

    // 预计算范数
    norms_sq_.resize(n);
    precompute_norms_sq(vectors_.data(), n, dim, norms_sq_.data());
}

VectorIndex::~VectorIndex() = default;

VectorIndex::VectorIndex(VectorIndex&& other) noexcept
    : dim_(other.dim_), size_(other.size_), metric_(other.metric_),
      vectors_(std::move(other.vectors_)), norms_sq_(std::move(other.norms_sq_)) {
    other.size_ = 0;
}

VectorIndex& VectorIndex::operator=(VectorIndex&& other) noexcept {
    if (this != &other) {
        dim_ = other.dim_;
        size_ = other.size_;
        metric_ = other.metric_;
        vectors_ = std::move(other.vectors_);
        norms_sq_ = std::move(other.norms_sq_);
        other.size_ = 0;
    }
    return *this;
}

uint32_t VectorIndex::add(const float* vector) {
    uint32_t id = static_cast<uint32_t>(size_);

    // 添加向量
    vectors_.insert(vectors_.end(), vector, vector + dim_);

    // 计算并存储范数
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        norm_sq += vector[i] * vector[i];
    }
    norms_sq_.push_back(norm_sq);

    size_++;
    return id;
}

void VectorIndex::add_batch(const float* vectors, size_t n) {
    // 扩展容量
    vectors_.reserve(vectors_.size() + n * dim_);
    norms_sq_.reserve(norms_sq_.size() + n);

    // 添加向量
    vectors_.insert(vectors_.end(), vectors, vectors + n * dim_);

    // 计算范数
    size_t old_size = norms_sq_.size();
    norms_sq_.resize(old_size + n);
    precompute_norms_sq(vectors, n, dim_, norms_sq_.data() + old_size);

    size_ += n;
}

SearchResult VectorIndex::search(const float* query, size_t k) const {
    SearchResult result;
    result.indices.resize(k);
    result.distances.resize(k);

    if (size_ == 0) return result;

    std::vector<float> all_distances(size_);

    // 计算查询向量范数
    float query_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        query_norm_sq += query[i] * query[i];
    }

    if (metric_ == DistanceMetric::L2_SQUARED) {
        // 使用预计算范数优化的 L2 距离
        batch_distance_l2sq_with_norms(
            query, query_norm_sq,
            vectors_.data(), norms_sq_.data(),
            size_, dim_, all_distances.data());
    } else if (metric_ == DistanceMetric::INNER_PRODUCT) {
        // 内积直接使用 BLAS
        batch_distance_blas(query, vectors_.data(), size_, dim_, metric_, all_distances.data());
    } else {
        // Cosine 也可以用预计算范数优化
#ifdef THUNDERDUCK_HAS_BLAS
        // distances = V @ q
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)size_, (int)dim_,
                    1.0f,
                    vectors_.data(), (int)dim_,
                    query, 1,
                    0.0f,
                    all_distances.data(), 1);

        float query_norm = std::sqrt(query_norm_sq);
        for (size_t i = 0; i < size_; i++) {
            float vec_norm = std::sqrt(norms_sq_[i]);
            float norm_product = query_norm * vec_norm;
            if (norm_product < 1e-10f) {
                all_distances[i] = 1.0f;
            } else {
                all_distances[i] = 1.0f - all_distances[i] / norm_product;
            }
        }
#else
        batch_distance_simd(query, vectors_.data(), size_, dim_, metric_, all_distances.data());
#endif
    }

    // Top-K 选择
    topk_smallest(all_distances.data(), size_, k,
                  result.indices.data(), result.distances.data());

    return result;
}

void VectorIndex::search_batch(const float* queries, size_t num_queries,
                               size_t k, SearchResult* results) const {
    std::vector<std::thread> threads;
    size_t num_threads = std::min((size_t)8, num_queries);
    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_queries);
        if (start >= num_queries) break;

        threads.emplace_back([=, &queries, &results]() {
            for (size_t q = start; q < end; q++) {
                results[q] = search(queries + q * dim_, k);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ============================================================================
// P3: FP16 转换工具实现
// ============================================================================

// IEEE 754 半精度浮点转换
uint16_t float_to_half(float f) {
#ifdef __ARM_FP16_FORMAT_IEEE
    // ARM 硬件支持
    __fp16 h = (__fp16)f;
    uint16_t result;
    std::memcpy(&result, &h, sizeof(result));
    return result;
#else
    // 软件实现
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        // 下溢: 返回零
        return sign;
    } else if (exp >= 31) {
        // 上溢: 返回无穷大
        return sign | 0x7C00;
    }

    return sign | (exp << 10) | (mant >> 13);
#endif
}

float half_to_float(uint16_t h) {
#ifdef __ARM_FP16_FORMAT_IEEE
    // ARM 硬件支持
    __fp16 hf;
    std::memcpy(&hf, &h, sizeof(hf));
    return (float)hf;
#else
    // 软件实现
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        // 零或非规格化数
        if (mant == 0) {
            // 零
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }
        // 非规格化数 (简化处理)
        return 0.0f;
    } else if (exp == 31) {
        // 无穷大或 NaN
        float result;
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    exp = exp - 15 + 127;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
#endif
}

void float_to_half_batch(const float* src, uint16_t* dst, size_t n) {
#ifdef __aarch64__
    size_t i = 0;
    // ARM NEON 4 元素批量转换
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h);
    }
    // 剩余元素
    for (; i < n; i++) {
        dst[i] = float_to_half(src[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_half(src[i]);
    }
#endif
}

void half_to_float_batch(const uint16_t* src, float* dst, size_t n) {
#ifdef __aarch64__
    size_t i = 0;
    // ARM NEON 4 元素批量转换
    for (; i + 3 < n; i += 4) {
        float16x4_t h = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        float32x4_t v = vcvt_f32_f16(h);
        vst1q_f32(dst + i, v);
    }
    // 剩余元素
    for (; i < n; i++) {
        dst[i] = half_to_float(src[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        dst[i] = half_to_float(src[i]);
    }
#endif
}

float distance_l2sq_fp16(const uint16_t* a, const uint16_t* b, size_t dim) {
#ifdef __aarch64__
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float16x4_t a_h = vld1_f16(reinterpret_cast<const __fp16*>(a + i));
        float16x4_t b_h = vld1_f16(reinterpret_cast<const __fp16*>(b + i));
        float32x4_t a_f = vcvt_f32_f16(a_h);
        float32x4_t b_f = vcvt_f32_f16(b_h);
        float32x4_t diff = vsubq_f32(a_f, b_f);
        sum_vec = vfmaq_f32(sum_vec, diff, diff);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < dim; i++) {
        float af = half_to_float(a[i]);
        float bf = half_to_float(b[i]);
        float diff = af - bf;
        sum += diff * diff;
    }

    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float af = half_to_float(a[i]);
        float bf = half_to_float(b[i]);
        float diff = af - bf;
        sum += diff * diff;
    }
    return sum;
#endif
}

float distance_inner_product_fp16(const uint16_t* a, const uint16_t* b, size_t dim) {
#ifdef __aarch64__
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 3 < dim; i += 4) {
        float16x4_t a_h = vld1_f16(reinterpret_cast<const __fp16*>(a + i));
        float16x4_t b_h = vld1_f16(reinterpret_cast<const __fp16*>(b + i));
        float32x4_t a_f = vcvt_f32_f16(a_h);
        float32x4_t b_f = vcvt_f32_f16(b_h);
        sum_vec = vfmaq_f32(sum_vec, a_f, b_f);
    }

    float sum = vaddvq_f32(sum_vec);

    for (; i < dim; i++) {
        float af = half_to_float(a[i]);
        float bf = half_to_float(b[i]);
        sum += af * bf;
    }

    return -sum;  // 负内积
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float af = half_to_float(a[i]);
        float bf = half_to_float(b[i]);
        sum += af * bf;
    }
    return -sum;
#endif
}

// ============================================================================
// P3: VectorIndexFP16 实现
// ============================================================================

VectorIndexFP16::VectorIndexFP16(size_t dim, DistanceMetric metric)
    : dim_(dim), size_(0), metric_(metric) {
}

VectorIndexFP16::VectorIndexFP16(const float* vectors, size_t n, size_t dim, DistanceMetric metric)
    : dim_(dim), size_(n), metric_(metric) {
    // 量化为 FP16
    vectors_fp16_.resize(n * dim);
    float_to_half_batch(vectors, vectors_fp16_.data(), n * dim);

    // 预计算范数 (保持 FP32 精度)
    norms_sq_.resize(n);
    precompute_norms_sq(vectors, n, dim, norms_sq_.data());
}

VectorIndexFP16::~VectorIndexFP16() = default;

uint32_t VectorIndexFP16::add(const float* vector) {
    uint32_t id = static_cast<uint32_t>(size_);

    // 量化并添加向量
    size_t old_size = vectors_fp16_.size();
    vectors_fp16_.resize(old_size + dim_);
    float_to_half_batch(vector, vectors_fp16_.data() + old_size, dim_);

    // 计算并存储范数
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        norm_sq += vector[i] * vector[i];
    }
    norms_sq_.push_back(norm_sq);

    size_++;
    return id;
}

void VectorIndexFP16::add_batch(const float* vectors, size_t n) {
    // 扩展容量
    size_t old_vec_size = vectors_fp16_.size();
    vectors_fp16_.resize(old_vec_size + n * dim_);

    // 量化
    float_to_half_batch(vectors, vectors_fp16_.data() + old_vec_size, n * dim_);

    // 计算范数
    size_t old_norm_size = norms_sq_.size();
    norms_sq_.resize(old_norm_size + n);
    precompute_norms_sq(vectors, n, dim_, norms_sq_.data() + old_norm_size);

    size_ += n;
}

SearchResult VectorIndexFP16::search(const float* query, size_t k) const {
    SearchResult result;
    result.indices.resize(k);
    result.distances.resize(k);

    if (size_ == 0) return result;

    // 量化查询向量
    std::vector<uint16_t> query_fp16(dim_);
    float_to_half_batch(query, query_fp16.data(), dim_);

    // 计算查询范数
    float query_norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        query_norm_sq += query[i] * query[i];
    }

    // 计算距离
    std::vector<float> all_distances(size_);

    if (metric_ == DistanceMetric::L2_SQUARED) {
        // 使用预计算范数和 FP16 点积
        // ||a-b||² = ||a||² + ||b||² - 2<a,b>
        for (size_t i = 0; i < size_; i++) {
            const uint16_t* vec_fp16 = vectors_fp16_.data() + i * dim_;

            // 计算点积 (FP16)
            float dot = 0.0f;
#ifdef __aarch64__
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t j = 0;
            for (; j + 3 < dim_; j += 4) {
                float16x4_t q_h = vld1_f16(reinterpret_cast<const __fp16*>(query_fp16.data() + j));
                float16x4_t v_h = vld1_f16(reinterpret_cast<const __fp16*>(vec_fp16 + j));
                float32x4_t q_f = vcvt_f32_f16(q_h);
                float32x4_t v_f = vcvt_f32_f16(v_h);
                sum_vec = vfmaq_f32(sum_vec, q_f, v_f);
            }
            dot = vaddvq_f32(sum_vec);
            for (; j < dim_; j++) {
                dot += half_to_float(query_fp16[j]) * half_to_float(vec_fp16[j]);
            }
#else
            for (size_t j = 0; j < dim_; j++) {
                dot += half_to_float(query_fp16[j]) * half_to_float(vec_fp16[j]);
            }
#endif
            all_distances[i] = query_norm_sq + norms_sq_[i] - 2.0f * dot;
        }
    } else if (metric_ == DistanceMetric::INNER_PRODUCT) {
        for (size_t i = 0; i < size_; i++) {
            all_distances[i] = distance_inner_product_fp16(
                query_fp16.data(), vectors_fp16_.data() + i * dim_, dim_);
        }
    } else {
        // Cosine
        float query_norm = std::sqrt(query_norm_sq);
        for (size_t i = 0; i < size_; i++) {
            const uint16_t* vec_fp16 = vectors_fp16_.data() + i * dim_;
            float vec_norm = std::sqrt(norms_sq_[i]);

            // 计算点积
            float dot = 0.0f;
            for (size_t j = 0; j < dim_; j++) {
                dot += half_to_float(query_fp16[j]) * half_to_float(vec_fp16[j]);
            }

            float norm_product = query_norm * vec_norm;
            if (norm_product < 1e-10f) {
                all_distances[i] = 1.0f;
            } else {
                all_distances[i] = 1.0f - dot / norm_product;
            }
        }
    }

    // Top-K 选择
    topk_smallest(all_distances.data(), size_, k,
                  result.indices.data(), result.distances.data());

    return result;
}

void VectorIndexFP16::search_batch(const float* queries, size_t num_queries,
                                   size_t k, SearchResult* results) const {
    std::vector<std::thread> threads;
    size_t num_threads = std::min((size_t)8, num_queries);
    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_queries);
        if (start >= num_queries) break;

        threads.emplace_back([=, &queries, &results]() {
            for (size_t q = start; q < end; q++) {
                results[q] = search(queries + q * dim_, k);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

size_t VectorIndexFP16::memory_usage() const {
    return vectors_fp16_.size() * sizeof(uint16_t) +
           norms_sq_.size() * sizeof(float);
}

// ============================================================================
// P2: HNSW 索引实现
// ============================================================================

HNSWIndex::HNSWIndex(size_t dim, DistanceMetric metric, const HNSWConfig& config)
    : dim_(dim), size_(0), metric_(metric), config_(config),
      max_level_(-1), entry_point_(0) {
}

HNSWIndex::HNSWIndex(const float* vectors, size_t n, size_t dim,
                     DistanceMetric metric, const HNSWConfig& config)
    : dim_(dim), size_(0), metric_(metric), config_(config),
      max_level_(-1), entry_point_(0) {
    // 预分配空间
    vectors_.reserve(n * dim);
    norms_sq_.reserve(n);
    node_levels_.reserve(n);

    add_batch(vectors, n);
}

HNSWIndex::~HNSWIndex() = default;

int HNSWIndex::random_level() {
    // 使用几何分布生成层级
    // P(level = l) = (1/M)^l * (1 - 1/M)
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // ml = 1/ln(M), 标准 HNSW 参数
    double ml = 1.0 / std::log(static_cast<double>(config_.M));
    double r = dist(gen);
    if (r < 1e-10) r = 1e-10;  // 避免 log(0)
    int level = static_cast<int>(-std::log(r) * ml);

    // 限制最大层级
    const int max_allowed = 16;
    return std::min(level, max_allowed);
}

void HNSWIndex::insert_node(uint32_t id, const float* vector) {
    // 分配层级
    int level = random_level();
    node_levels_.push_back(level);

    // 扩展层级结构到需要的层数
    while (static_cast<int>(layers_.size()) <= level) {
        layers_.emplace_back();
    }

    // 确保所有层 0 到 level 都有足够空间
    for (int l = 0; l <= level; l++) {
        if (layers_[l].size() <= id) {
            layers_[l].resize(id + 1);
        }
    }

    // 确保层 0 总是有足够空间（所有节点都在层 0）
    if (layers_[0].size() <= id) {
        layers_[0].resize(id + 1);
    }

    // 如果是第一个节点
    if (id == 0) {
        entry_point_ = id;
        max_level_ = level;
        return;
    }

    // 从顶层开始搜索
    uint32_t curr = entry_point_;
    float curr_dist = distance(vector, vectors_.data() + curr * dim_, dim_, metric_);

    // 从最高层贪婪下降到 level+1
    for (int l = max_level_; l > level; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (l < static_cast<int>(layers_.size()) && curr < layers_[l].size()) {
                for (uint32_t neighbor : layers_[l][curr]) {
                    if (neighbor >= id) continue;  // 跳过尚未添加的节点
                    float dist = distance(vector, vectors_.data() + neighbor * dim_, dim_, metric_);
                    if (dist < curr_dist) {
                        curr = neighbor;
                        curr_dist = dist;
                        changed = true;
                    }
                }
            }
        }
    }

    // 在 level 层及以下建立连接
    int start_level = std::min(level, max_level_);
    for (int l = start_level; l >= 0; l--) {
        // 搜索当前层的候选邻居
        auto candidates = search_layer(vector, curr, config_.ef_construction, l);

        // 选择最近的 M 个作为邻居
        size_t M_max = (l == 0) ? config_.M * 2 : config_.M;
        size_t num_neighbors = std::min(M_max, candidates.size());

        // 添加邻居连接
        for (size_t i = 0; i < num_neighbors; i++) {
            uint32_t neighbor = candidates[i].second;
            if (neighbor == id) continue;  // 不连接到自己

            // 确保邻居节点在当前层有空间
            if (layers_[l].size() <= neighbor) {
                layers_[l].resize(neighbor + 1);
            }

            // 双向连接
            layers_[l][id].push_back(neighbor);
            layers_[l][neighbor].push_back(id);

            // 如果邻居的连接数超过限制，裁剪
            if (layers_[l][neighbor].size() > M_max) {
                // 计算邻居到所有连接节点的距离
                std::vector<std::pair<float, uint32_t>> neighbor_dists;
                neighbor_dists.reserve(layers_[l][neighbor].size());

                for (uint32_t nn : layers_[l][neighbor]) {
                    float d = distance(vectors_.data() + neighbor * dim_,
                                      vectors_.data() + nn * dim_, dim_, metric_);
                    neighbor_dists.push_back({d, nn});
                }

                // 按距离排序，保留最近的 M_max 个
                std::sort(neighbor_dists.begin(), neighbor_dists.end());

                layers_[l][neighbor].clear();
                layers_[l][neighbor].reserve(M_max);
                for (size_t j = 0; j < M_max && j < neighbor_dists.size(); j++) {
                    layers_[l][neighbor].push_back(neighbor_dists[j].second);
                }
            }
        }

        // 更新下一层的入口点
        if (!candidates.empty()) {
            curr = candidates[0].second;
        }
    }

    // 更新入口点（如果新节点的层级更高）
    if (level > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
}

std::vector<std::pair<float, uint32_t>> HNSWIndex::search_layer(
    const float* query, uint32_t entry, size_t ef, int level) const {

    if (size_ == 0) return {};

    // 使用 unordered_set 替代 vector<bool> 以支持动态大小
    std::unordered_set<uint32_t> visited;
    visited.reserve(std::min(ef * 10, size_));

    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<>> candidates;  // 最小堆：最小距离在顶部
    std::priority_queue<std::pair<float, uint32_t>> results;  // 最大堆：最大距离在顶部

    // 初始化：加入入口点
    if (entry >= size_) return {};

    float entry_dist = distance(query, vectors_.data() + entry * dim_, dim_, metric_);
    candidates.push({entry_dist, entry});
    results.push({entry_dist, entry});
    visited.insert(entry);

    // 搜索循环
    while (!candidates.empty()) {
        auto [c_dist, c_id] = candidates.top();
        candidates.pop();

        // 终止条件：最近的候选比结果集中最远的还要远
        if (results.size() >= ef && c_dist > results.top().first) {
            break;
        }

        // 检查当前节点的所有邻居
        if (level < static_cast<int>(layers_.size()) && c_id < layers_[level].size()) {
            for (uint32_t neighbor : layers_[level][c_id]) {
                // 跳过已访问的节点
                if (visited.count(neighbor) > 0) continue;
                if (neighbor >= size_) continue;  // 安全检查

                visited.insert(neighbor);
                float n_dist = distance(query, vectors_.data() + neighbor * dim_, dim_, metric_);

                // 如果结果集未满，或者新节点比最远结果更近，则加入
                bool should_add = (results.size() < ef) || (n_dist < results.top().first);

                if (should_add) {
                    candidates.push({n_dist, neighbor});
                    results.push({n_dist, neighbor});

                    // 保持结果集大小为 ef
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
    }

    // 将结果从最近到最远排序
    std::vector<std::pair<float, uint32_t>> result_vec;
    result_vec.reserve(results.size());

    while (!results.empty()) {
        result_vec.push_back(results.top());
        results.pop();
    }

    // 反转使最近的在前面
    std::reverse(result_vec.begin(), result_vec.end());

    return result_vec;
}

uint32_t HNSWIndex::add(const float* vector) {
    uint32_t id = static_cast<uint32_t>(size_);

    // 添加向量
    vectors_.insert(vectors_.end(), vector, vector + dim_);

    // 计算范数
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; i++) {
        norm_sq += vector[i] * vector[i];
    }
    norms_sq_.push_back(norm_sq);

    size_++;

    // 插入到 HNSW 图
    insert_node(id, vector);

    return id;
}

void HNSWIndex::add_batch(const float* vectors, size_t n) {
    for (size_t i = 0; i < n; i++) {
        add(vectors + i * dim_);
    }
}

SearchResult HNSWIndex::search(const float* query, size_t k, size_t ef_search) const {
    SearchResult result;
    result.indices.resize(k);
    result.distances.resize(k);

    if (size_ == 0) return result;

    size_t ef = (ef_search > 0) ? ef_search : config_.ef_search;
    ef = std::max(ef, k);

    // 从顶层开始贪婪下降
    uint32_t curr = entry_point_;
    float curr_dist = distance(query, vectors_.data() + curr * dim_, dim_, metric_);

    for (int l = max_level_; l > 0; l--) {
        bool changed = true;
        while (changed) {
            changed = false;
            if (l < static_cast<int>(layers_.size()) && curr < layers_[l].size()) {
                for (uint32_t neighbor : layers_[l][curr]) {
                    float dist = distance(query, vectors_.data() + neighbor * dim_, dim_, metric_);
                    if (dist < curr_dist) {
                        curr = neighbor;
                        curr_dist = dist;
                        changed = true;
                    }
                }
            }
        }
    }

    // 在第 0 层搜索
    auto candidates = search_layer(query, curr, ef, 0);

    // 取前 k 个
    size_t result_size = std::min(k, candidates.size());
    for (size_t i = 0; i < result_size; i++) {
        result.indices[i] = candidates[i].second;
        result.distances[i] = candidates[i].first;
    }

    return result;
}

void HNSWIndex::search_batch(const float* queries, size_t num_queries,
                             size_t k, SearchResult* results, size_t ef_search) const {
    std::vector<std::thread> threads;
    size_t num_threads = std::min((size_t)8, num_queries);
    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_queries);
        if (start >= num_queries) break;

        threads.emplace_back([=, &queries, &results]() {
            for (size_t q = start; q < end; q++) {
                results[q] = search(queries + q * dim_, k, ef_search);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

} // namespace vector
} // namespace thunderduck
