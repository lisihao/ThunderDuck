# ThunderDuck NPU SQL 算子加速研究分析

> 版本: 1.0.0 | 日期: 2026-01-28 | 基线: V19.2

## 一、研究背景

### 1.1 V19.5 实验总结

V19.5 尝试将标准聚合算子映射到 NPU，结论如下：

| 方法 | 问题 | 性能比 |
|------|------|--------|
| MPSGraph reductionSum | 启动开销 ~0.5-1ms | 0.07x-0.37x (1-10M) |
| One-Hot 矩阵乘法 | O(n×g) 矩阵构建 | 0.03x-0.16x |

**核心发现**：
- ANE/NPU 为 **ML 推理**设计，非通用计算
- 固定启动开销 (~0.5ms) 对小任务致命
- 稀疏操作（分组聚合）不适合密集矩阵硬件
- CPU SIMD + 多线程仍是通用聚合最优解

### 1.2 NPU 适用场景特征

NPU 适合的计算特征：
```
✓ 密集矩阵乘法 (GEMM)      - 核心优势
✓ 大规模向量点积           - 嵌入检索
✓ 批量数据处理 (>100K)     - 摊销启动开销
✓ 规则内存访问模式         - 连续/跨步访问
✓ FP16/BF16 精度可接受     - ANE 原生格式

✗ 稀疏/随机访问            - 哈希表、分组
✗ 分支密集逻辑             - 条件过滤
✗ 小数据量 (<100K)         - 启动开销主导
✗ 需要 FP64 精度           - 不支持
```

---

## 二、向量相似性搜索算子

### 2.1 问题定义

```sql
SELECT id, vec <-> $query_vec AS distance
FROM embeddings
ORDER BY distance
LIMIT k;
```

计算本质：
- 查询向量 $q \in \mathbb{R}^d$
- 数据矩阵 $V \in \mathbb{R}^{n \times d}$
- 相似度 $s = V \cdot q^T$ (矩阵-向量乘法)

### 2.2 NPU 适用性分析

| 指标 | 评估 | 说明 |
|------|------|------|
| 计算模式 | ✓ 密集 GEMV | $(n \times d) \cdot (d \times 1)$ |
| 数据规模 | ✓ 通常 100K-100M | 典型嵌入检索场景 |
| 内存访问 | ✓ 规则访问 | 连续读取向量 |
| 精度要求 | ✓ FP16 可接受 | 相似度排序不需高精度 |
| 启动开销 | ✓ 可摊销 | 大 n 情况下忽略不计 |

**结论**：**高度适合 NPU 加速** ⭐⭐⭐⭐⭐

### 2.3 性能预估

```
参数: n=1M 向量, d=768 (BERT), K=10

CPU (单线程):
  - 1M × 768 × 2 FLOP = 1.536 GFLOP
  - ~10 GFLOPS → 153 ms

CPU (8线程 SIMD):
  - ~80 GFLOPS → 19 ms

NPU (ANE):
  - ~38 TOPS (FP16) → 0.04 ms (理论)
  - 实际: ~1-2 ms (启动 + 数据传输)

预期加速: 10-20x vs CPU 多线程
```

### 2.4 实现方案

#### 方案 A: MPSGraph GEMV

```objc
// 创建计算图
MPSGraph* graph = [[MPSGraph alloc] init];
MPSGraphTensor* queryTensor = [graph placeholderWithShape:@[@1, @(dim)]
                                                 dataType:MPSDataTypeFloat16];
MPSGraphTensor* vectorsTensor = [graph placeholderWithShape:@[@(n), @(dim)]
                                                  dataType:MPSDataTypeFloat16];

// 矩阵乘法: [n, d] × [d, 1] = [n, 1]
MPSGraphTensor* result = [graph matrixMultiplicationWithPrimaryTensor:vectorsTensor
                                                      secondaryTensor:queryTensor
                                                                 name:nil];
```

#### 方案 B: Accelerate BLAS (AMX)

```cpp
// cblas_sgemv 利用 AMX 加速
cblas_sgemv(CblasRowMajor, CblasNoTrans,
            n, dim,           // 矩阵维度
            1.0f,             // alpha
            vectors, dim,     // A 矩阵
            query, 1,         // x 向量
            0.0f,             // beta
            distances, 1);    // y 结果
```

#### 方案 C: Core ML 模型

```python
# 预编译为 Core ML 模型
import coremltools as ct

# 定义计算图
@ct.function
def similarity_search(query, vectors):
    return np.dot(vectors, query.T)

# 编译为 ANE 优化格式
model = ct.convert(similarity_search,
                   compute_units=ct.ComputeUnit.ALL)  # CPU + GPU + ANE
```

### 2.5 建议实现路径

```
Phase 1: Accelerate BLAS (cblas_sgemv)
  - 利用 AMX，无框架开销
  - 预期: 5-10x vs 朴素实现

Phase 2: MPSGraph 批量查询
  - 多查询并行
  - 预期: 额外 2-3x

Phase 3: 量化向量 (INT8/FP16)
  - ANE 原生格式
  - 预期: 2x 吞吐 + 50% 内存
```

---

## 三、聚合与统计算子

### 3.1 V19.5 实验结论回顾

| 方法 | 结果 | 原因 |
|------|------|------|
| 简单 SUM (MPS) | 0.37x | 启动开销主导 |
| 分组 SUM (One-Hot) | 0.03x | 矩阵构建 O(n×g) |

### 3.2 替代方案分析

#### 3.2.1 直方图指令 (BNNS/vImage)

Apple BNNS 和 vImage 提供直方图计算：

```cpp
// vImage 直方图 (用于 COUNT GROUP BY)
vImagePixelCount histogram[256];
vImageHistogramCalculation_Planar8(&src, histogram, kvImageNoFlags);
```

**问题**：
- 仅支持 8-bit 整数
- 不支持带值聚合 (SUM)
- 限制太多，不实用

#### 3.2.2 Embedding Lookup 模拟

将分组聚合视为嵌入查找的逆操作：

```
正向 Embedding: group_id → vector
逆向 (聚合):    group_id[] + values[] → sums[]
```

Core ML 支持 `gather` 操作，但无对应的 "scatter-add" 原语。

#### 3.2.3 稀疏矩阵乘法

One-Hot 矩阵本质是稀疏矩阵，可用 CSR/CSC 格式：

```cpp
// Accelerate Sparse BLAS
sparse_matrix_float A;  // One-Hot 矩阵 (CSR 格式)
sparse_status = sparse_matrix_vector_product_dense_float(
    SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, A, values, result);
```

**问题**：
- 稀疏 BLAS 主要在 CPU 执行
- ANE 不支持稀疏运算
- 性能不如 CPU SIMD 直接累加

### 3.3 结论

| 聚合类型 | NPU 适用性 | 最优方案 |
|----------|------------|----------|
| 简单 SUM/AVG | ✗ 不适合 | CPU SIMD 多线程 |
| COUNT | ✗ 不适合 | CPU SIMD 多线程 |
| GROUP BY | ✗ 不适合 | CPU V15 (8线程) |
| MIN/MAX | ✗ 不适合 | CPU SIMD |

**聚合算子应继续使用 CPU 实现，V19.2 已是最优。**

---

## 四、连接算子 (Join)

### 4.1 Hash Join 的 NPU 可行性

Hash Join 核心操作：
1. 构建阶段: 对 build 表键计算哈希并插入
2. 探测阶段: 对 probe 表键查哈希表匹配

**NPU 困难点**：
- 哈希表访问是 **随机访问**
- 链表/开放寻址需 **分支和循环**
- ANE 完全不支持此类计算

### 4.2 Sort-Merge Join 的 NPU 可能性

```
步骤 1: 排序两表 → GPU Bitonic Sort
步骤 2: 合并匹配键 → ???
```

#### 4.2.1 排序阶段

GPU 排序已在 ThunderDuck 实现 (Metal Bitonic Sort)，性能良好。

#### 4.2.2 合并阶段的张量化尝试

将键匹配转化为外积比较：

```cpp
// 外积比较矩阵
// M[i,j] = 1 if build_keys[i] == probe_keys[j]
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
        M[i][j] = (build_keys[i] == probe_keys[j]) ? 1 : 0;
```

**问题**：
- 比较矩阵 $O(n \times m)$ 内存
- 10M × 10M = 100TB 内存！
- 完全不可行

#### 4.2.3 布隆过滤器 + NPU

```
思路:
1. 构建 Bloom Filter (CPU/GPU)
2. 用 NPU 做批量 membership test
```

Bloom Filter 测试可表示为：
```
hit = AND(hash1(key) in bitmap, hash2(key) in bitmap, ...)
```

**问题**：
- 位图访问是随机访问
- AND 操作不是 ANE 优势
- BNNS 可能有帮助，但增益有限

### 4.3 FFT 卷积连接 (理论探讨)

学术界有研究将连接转化为卷积：

```
// 集合交集通过多项式乘法
// A = {1,3,5} → P_A(x) = x + x³ + x⁵
// B = {2,3,4} → P_B(x) = x² + x³ + x⁴
// A ∩ B 对应 P_A × P_B 中的平方项
```

使用 FFT 可在 $O(n \log n)$ 完成多项式乘法。

**问题**：
- 仅适用于整数键
- 键域大时多项式度数爆炸
- 实际 Join 还需要返回索引，而非仅判断存在

### 4.4 结论

| Join 类型 | NPU 适用性 | 最优方案 |
|-----------|------------|----------|
| Inner/Left/Right | ✗ 不适合 | CPU Hash Join V19.2 |
| Semi/Anti | ✗ 不适合 | GPU Semi Join |
| Cross Join | △ 理论可行 | 罕见场景 |

**Join 算子应继续使用 CPU/GPU 实现。**

---

## 五、复杂算子融合 (ML + SQL)

### 5.1 核心机会

ANE 的真正优势在 **ML 推理**。如果将 ML 操作嵌入 SQL 查询：

```sql
-- 人脸识别 + 统计
SELECT date, COUNT(*)
FROM security_footage
WHERE face_match(frame, target_embedding) > 0.9  -- NPU 加速
GROUP BY date;

-- 时序异常检测
SELECT timestamp, value
FROM sensor_data
WHERE is_anomaly(value, prev_values) = true;  -- NPU 加速
```

### 5.2 实现模式

#### 模式 A: UDF 调用 Core ML

```cpp
// 注册 DuckDB UDF
connection.CreateScalarFunction("face_match", {
    LogicalType::BLOB,  // 图像
    LogicalType::BLOB   // 嵌入
}, LogicalType::DOUBLE, face_match_impl);

// 实现调用 Core ML
double face_match_impl(DataChunk& input) {
    // 1. 提取图像和目标嵌入
    // 2. 调用 Core ML 模型
    // 3. 返回相似度分数
    return score;
}
```

#### 模式 B: 向量化批处理

```cpp
// 批量处理多行
void face_match_vectorized(DataChunk& input, Vector& result) {
    // 1. 收集所有图像到批次
    // 2. 一次 Core ML 推理
    // 3. 分发结果
}
```

### 5.3 DAnA 启发的算子融合

参考 PVLDB'18 DAnA 论文的思想：

```
传统流程:
  SQL Parser → Logical Plan → Physical Plan → Executor
                                                  ↓
                                             CPU 执行
                                                  ↓
                                             Python UDF
                                                  ↓
                                             返回结果

融合流程:
  SQL Parser → Logical Plan → Physical Plan → NPU Compiler
                                                  ↓
                                        ML + SQL 融合图
                                                  ↓
                                        ANE 一次执行
```

### 5.4 实际可行方案

#### 方案 1: 嵌入检索加速

```sql
-- ThunderDuck 可优化此查询
SELECT id, content
FROM documents
WHERE embedding <-> $query < 0.5  -- NPU 计算距离
ORDER BY embedding <-> $query
LIMIT 10;
```

实现：
1. 拦截 `<->` 运算符
2. 提取所有嵌入到矩阵
3. NPU 执行 GEMV
4. 返回 Top-K

#### 方案 2: 图像/文本分类

```sql
-- 利用 Core ML 模型
SELECT classify(image) AS category, COUNT(*)
FROM photos
GROUP BY category;
```

实现：
1. 批量加载图像
2. Core ML 批量推理
3. 标准 GROUP BY

#### 方案 3: RAG 查询加速

```sql
-- Retrieval-Augmented Generation
WITH relevant_docs AS (
    SELECT content
    FROM knowledge_base
    ORDER BY embedding <-> embed($question)  -- NPU
    LIMIT 5
)
SELECT generate_answer($question, array_agg(content))  -- LLM
FROM relevant_docs;
```

### 5.5 性能预期

| 场景 | 数据规模 | CPU | NPU | 加速比 |
|------|----------|-----|-----|--------|
| 嵌入检索 | 1M × 768 | 150ms | 2ms | 75x |
| 图像分类 | 1K 图像 | 5s | 50ms | 100x |
| 文本嵌入 | 10K 文本 | 30s | 500ms | 60x |

---

## 六、推荐实现路径

### 6.1 优先级排序

| 优先级 | 功能 | ROI | 难度 |
|--------|------|-----|------|
| P0 | 向量相似度搜索 | ⭐⭐⭐⭐⭐ | 中 |
| P1 | 嵌入计算 UDF | ⭐⭐⭐⭐ | 中 |
| P2 | 图像分类 UDF | ⭐⭐⭐ | 高 |
| P3 | 量化向量支持 | ⭐⭐⭐ | 中 |
| -- | 聚合算子 NPU | ✗ 不建议 | -- |
| -- | Join 算子 NPU | ✗ 不建议 | -- |

### 6.2 V20 建议实现

```
ThunderDuck V20: 向量数据库能力

核心功能:
├── 向量相似度搜索 (HNSW + 暴力扫描)
│   ├── CPU 实现 (基线)
│   ├── AMX/BLAS 加速
│   └── NPU/ANE 加速 (>100K 向量)
├── 嵌入计算 UDF
│   ├── Core ML 模型调用
│   └── 批量推理优化
└── 量化向量支持
    ├── FP16 存储/计算
    └── INT8 量化检索
```

### 6.3 技术架构

```
                    ┌─────────────────────────────────────┐
                    │           DuckDB SQL Layer          │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │     ThunderDuck Operator Layer      │
                    │  ┌─────────┬─────────┬───────────┐  │
                    │  │ Filter  │  Join   │ Aggregate │  │
                    │  │ (CPU)   │ (CPU)   │  (CPU)    │  │
                    │  └─────────┴─────────┴───────────┘  │
                    │  ┌─────────────────────────────────┐│
                    │  │    Vector Similarity Search     ││
                    │  │  ┌───────┬───────┬───────────┐ ││
                    │  │  │ CPU   │ AMX   │ NPU/ANE   │ ││
                    │  │  │ SIMD  │ BLAS  │ (>100K)   │ ││
                    │  │  └───────┴───────┴───────────┘ ││
                    │  └─────────────────────────────────┘│
                    │  ┌─────────────────────────────────┐│
                    │  │      ML UDF Integration         ││
                    │  │      (Core ML / ANE)            ││
                    │  └─────────────────────────────────┘│
                    └─────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │    Hardware Acceleration Layer      │
                    │  ┌────────┬────────┬─────────────┐  │
                    │  │ Neon   │ Metal  │  ANE/NPU    │  │
                    │  │ SIMD   │ GPU    │  (ML only)  │  │
                    │  └────────┴────────┴─────────────┘  │
                    └─────────────────────────────────────┘
```

---

## 七、结论

### 7.1 关键洞察

1. **NPU 不是通用加速器**
   - 专为 ML 推理优化
   - 不适合传统 SQL 算子

2. **向量搜索是最佳突破口**
   - 天然矩阵乘法
   - 大数据量摊销开销
   - On-Device RAG 核心需求

3. **算子融合 > 单算子替换**
   - ML + SQL 一体化
   - 减少数据往返
   - 利用 NPU 真正优势

### 7.2 不建议的方向

- ✗ 聚合算子 NPU 加速 (V19.5 已验证)
- ✗ Join 算子 NPU 加速 (结构不匹配)
- ✗ 过滤算子 NPU 加速 (分支密集)

### 7.3 建议的方向

- ✓ 向量相似度搜索 (P0 优先)
- ✓ ML UDF 集成 (P1)
- ✓ 嵌入检索 RAG 优化 (P0)
- ✓ 量化向量支持 (P3)

---

## 参考文献

1. DAnA: Automatically Parallelizing Python Data Analysis to FPGA (PVLDB 2018)
2. DuckDB Vector Similarity Search Extension
3. Apple Neural Engine Programming Guide
4. Accelerate Framework BLAS Reference
5. Core ML Performance Optimization
