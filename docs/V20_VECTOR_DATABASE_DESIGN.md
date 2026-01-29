# ThunderDuck V20 - 向量数据库引擎 @ 性能基线版

> 版本: V20.0 | 日期: 2026-01-28 | 基线: V19.2

## 一、概述

### 1.1 目标

构建 ThunderDuck 向量数据库能力，对标 DuckDB VSS 扩展，实现：
- 向量相似度搜索（暴力扫描 + HNSW 索引）
- 三种距离度量：L2（欧氏距离）、Cosine、Inner Product
- Apple Silicon 硬件加速（AMX/BLAS + ANE）

### 1.2 对标：DuckDB VSS 扩展

DuckDB VSS 基于 [usearch](https://github.com/unum-cloud/usearch) 库：

| 特性 | DuckDB VSS | ThunderDuck V20 |
|------|------------|-----------------|
| 距离函数 | L2, Cosine, IP | L2, Cosine, IP |
| 索引类型 | HNSW | HNSW + 暴力扫描 |
| 数据类型 | FLOAT[N] | FLOAT[N], FP16 |
| 硬件加速 | 无 | AMX/BLAS, ANE |
| 持久化 | 实验性 | 内存优先 |

### 1.3 性能目标

| 场景 | DuckDB VSS | ThunderDuck V20 目标 |
|------|------------|---------------------|
| 暴力扫描 100K×768 | ~50ms | <10ms (5x) |
| 暴力扫描 1M×768 | ~500ms | <50ms (10x) |
| HNSW 查询 1M×768 | ~1ms | ~1ms (相当) |

---

## 二、技术设计

### 2.1 距离函数

```cpp
namespace thunderduck::vector {

// L2 平方距离 (欧氏距离的平方)
float distance_l2sq(const float* a, const float* b, size_t dim);

// Cosine 距离 (1 - cosine_similarity)
float distance_cosine(const float* a, const float* b, size_t dim);

// 负内积 (用于最大内积搜索)
float distance_inner_product(const float* a, const float* b, size_t dim);

}
```

### 2.2 暴力扫描实现

#### 2.2.1 CPU SIMD 版本

```cpp
// ARM Neon 优化的 L2 距离计算
void batch_distance_l2sq_neon(
    const float* query,       // [dim]
    const float* vectors,     // [n, dim]
    size_t n, size_t dim,
    float* distances          // [n]
);
```

#### 2.2.2 BLAS/AMX 加速版本

```cpp
// 利用 cblas_sgemv 计算内积/cosine
void batch_distance_blas(
    const float* query,       // [dim]
    const float* vectors,     // [n, dim]
    size_t n, size_t dim,
    DistanceMetric metric,
    float* distances          // [n]
);
```

对于 Inner Product 和 Cosine：
- `distances = V @ q^T` 是 GEMV 操作
- 自动利用 Apple AMX 加速

对于 L2：
- `||a-b||² = ||a||² + ||b||² - 2<a,b>`
- 预计算向量范数，转化为 GEMV

### 2.3 Top-K 选择

```cpp
// 从距离数组中选择 Top-K 最小距离
void topk_distances(
    const float* distances,   // [n]
    size_t n, size_t k,
    uint32_t* indices,        // [k] 输出索引
    float* values             // [k] 输出距离
);
```

### 2.4 向量搜索统一接口

```cpp
namespace thunderduck::vector {

enum class DistanceMetric {
    L2_SQUARED,    // ||a-b||²
    COSINE,        // 1 - <a,b>/(||a||·||b||)
    INNER_PRODUCT  // -<a,b> (负内积，用于最大化)
};

struct SearchConfig {
    DistanceMetric metric = DistanceMetric::L2_SQUARED;
    size_t k = 10;                    // Top-K
    bool use_blas = true;             // 使用 BLAS/AMX
    size_t num_threads = 8;           // 并行线程数
};

struct SearchResult {
    std::vector<uint32_t> indices;    // [k] 最近邻索引
    std::vector<float> distances;     // [k] 距离
};

// 暴力扫描搜索
SearchResult brute_force_search(
    const float* query,               // [dim]
    const float* vectors,             // [n, dim]
    size_t n, size_t dim,
    const SearchConfig& config
);

// 批量查询
void batch_search(
    const float* queries,             // [num_queries, dim]
    size_t num_queries,
    const float* vectors,             // [n, dim]
    size_t n, size_t dim,
    const SearchConfig& config,
    SearchResult* results             // [num_queries]
);

}
```

---

## 三、实现计划

### Phase 1: 基础距离函数 (P0)

1. `distance_l2sq_neon()` - ARM Neon 优化
2. `distance_cosine_neon()` - ARM Neon 优化
3. `distance_inner_product_neon()` - ARM Neon 优化
4. 单元测试验证正确性

### Phase 2: 批量距离计算 (P0)

1. `batch_distance_blas()` - BLAS/AMX 加速
2. L2 距离的 GEMV 转换
3. 向量范数预计算

### Phase 3: Top-K 集成 (P0)

1. 复用现有 `topk_min_f32_v5()` 实现
2. 与距离计算流水线集成

### Phase 4: 基准测试 (P0)

1. 对比 DuckDB VSS 暴力扫描性能
2. 多数据规模测试 (10K, 100K, 1M)
3. 多维度测试 (128, 384, 768, 1536)

---

## 四、基准测试设计

### 4.1 测试数据

```cpp
// 生成随机测试向量
void generate_random_vectors(
    float* vectors, size_t n, size_t dim,
    unsigned seed = 42
);

// 数据规模
const size_t SIZES[] = {10000, 100000, 1000000};
const size_t DIMS[] = {128, 384, 768, 1536};
```

### 4.2 测试场景

| 场景 | 描述 |
|------|------|
| T1 | 10K 向量, 768 维, K=10 |
| T2 | 100K 向量, 768 维, K=10 |
| T3 | 1M 向量, 768 维, K=10 |
| T4 | 1M 向量, 768 维, K=100 |
| T5 | 1M 向量, 1536 维, K=10 |

### 4.3 对比方法

```cpp
// DuckDB VSS 基准
double benchmark_duckdb_vss(
    const float* vectors, size_t n, size_t dim,
    const float* query, size_t k
);

// ThunderDuck V20 基准
double benchmark_thunderduck_v20(
    const float* vectors, size_t n, size_t dim,
    const float* query, size_t k,
    const SearchConfig& config
);
```

---

## 五、API 设计

### 5.1 C++ API

```cpp
#include "thunderduck/vector_search.h"

using namespace thunderduck::vector;

// 创建向量数据集
VectorDataset dataset(vectors, n, dim);

// 配置搜索
SearchConfig config;
config.metric = DistanceMetric::L2_SQUARED;
config.k = 10;

// 执行搜索
SearchResult result = dataset.search(query, config);

// 访问结果
for (size_t i = 0; i < result.indices.size(); i++) {
    printf("Rank %zu: idx=%u, dist=%.4f\n",
           i, result.indices[i], result.distances[i]);
}
```

### 5.2 索引类型 API

```cpp
// P1: VectorIndex - 预计算范数优化
VectorIndex index(vectors, n, dim, DistanceMetric::L2_SQUARED);
auto result = index.search(query, k);  // 2x faster than raw BLAS

// P2: HNSWIndex - 近似最近邻搜索
HNSWConfig config;
config.M = 32;
config.ef_construction = 400;
config.ef_search = 128;
HNSWIndex hnsw(vectors, n, dim, DistanceMetric::L2_SQUARED, config);
auto result = hnsw.search(query, k);  // ~1ms for 10K vectors, 90% recall

// P3: VectorIndexFP16 - 内存优化
VectorIndexFP16 fp16_index(vectors, n, dim);
auto result = fp16_index.search(query, k);  // 50% memory, ~same speed
size_t mem = fp16_index.memory_usage();     // bytes used
```

### 5.3 SQL 兼容接口 (未来)

```sql
-- 创建向量表
CREATE TABLE embeddings (
    id INTEGER,
    vec FLOAT[768]
);

-- 向量搜索
SELECT id, array_distance(vec, $query) AS dist
FROM embeddings
ORDER BY dist
LIMIT 10;
```

---

## 六、基准测试结果 (V20.1)

> 测试环境: Apple M4 Max, macOS 14.x, 10 次迭代中位数

### 6.1 暴力扫描性能对比

| 场景 | DuckDB list_distance | ThunderDuck SIMD | 加速比 |
|------|---------------------|------------------|--------|
| T1: 10K×768, K=10 | 9.57 ms | 0.36 ms | **27x** |
| T2: 100K×768, K=10 | ~95 ms | 3.31 ms | **29x** |
| T3: 1M×768, K=10 | ~950 ms | 32.35 ms | **29x** |

### 6.2 索引类型性能对比 (T1: 10K×768)

| 索引类型 | 查询延迟 | vs BLAS | 召回率 |
|----------|----------|---------|--------|
| BLAS/AMX | 1.05 ms | 1.0x | 100% |
| SIMD | 0.36 ms | **2.9x** | 100% |
| VectorIndex (P1) | 0.52 ms | **2.0x** | 100% |
| HNSW (ef=128) | 0.69 ms | **1.5x** | 90% |
| FP16 Index | 1.15 ms | 0.9x | ~99% |

### 6.3 关键发现

1. **SIMD 优于 BLAS**: 对于 L2 距离计算，直接 SIMD 实现比 BLAS 快 3x，因为避免了范数计算和内存复制开销

2. **预计算范数有效**: VectorIndex (P1) 通过预计算向量范数实现 2x 加速

3. **HNSW 高召回率**: 经过调优 (M=32, ef_construction=400)，HNSW 达到 90% Recall@10

4. **FP16 权衡**: FP16 节省 50% 内存但因转换开销略慢；适合内存受限场景

### 6.4 吞吐量指标

| 场景 | 吞吐量 | 计算能力 |
|------|--------|----------|
| 10K×768 | 28 M vectors/sec | 14.7 GFLOPS |
| 100K×768 | 30 M vectors/sec | 14.2 GFLOPS |
| 1M×768 | 31 M vectors/sec | 14.1 GFLOPS |
| 1M×1536 | 15 M vectors/sec | 14.1 GFLOPS |

---

## 七、参考资料

- [DuckDB VSS Extension](https://duckdb.org/docs/stable/core_extensions/vss)
- [USearch Library](https://github.com/unum-cloud/usearch)
- [Apple Accelerate BLAS](https://developer.apple.com/documentation/accelerate/blas)
