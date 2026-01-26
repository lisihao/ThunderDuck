# AMX 加速可行性分析

> **日期**: 2026-01-25 | **版本**: 1.0

## 一、AMX 概述

Apple AMX (Advanced Matrix Extensions) 是 Apple Silicon 的矩阵协处理器：

| 特性 | 说明 |
|------|------|
| 寄存器 | 8KB 专用寄存器文件 (64×128-bit) |
| 操作 | 外积累加 (Outer Product Accumulate) |
| 数据类型 | FP16, FP32, FP64, INT8, INT16 |
| 访问方式 | 无公开 API，通过 Accelerate 框架间接使用 |
| 吞吐量 | M4: ~2 TFLOPS (FP32), ~4 TOPS (INT8) |

### 1.1 Accelerate 框架中可能使用 AMX 的函数

```
vDSP:
├── vDSP_mmul()      - 矩阵乘法 ⭐ 最可能使用 AMX
├── vDSP_dotpr()     - 向量点积 ⭐
├── vDSP_sve()       - 向量求和 (可能)
└── vDSP_vdist()     - 向量距离

BLAS (cblas):
├── cblas_sgemm()    - 单精度矩阵乘 ⭐⭐ 确定使用 AMX
├── cblas_dgemm()    - 双精度矩阵乘 ⭐⭐
├── cblas_sdot()     - 单精度点积 ⭐
└── cblas_snrm2()    - 向量范数

BNNS:
├── BNNSFilterApplyBatch() - 神经网络推理 ⭐⭐
└── BNNSComputeNorm()      - 范数计算
```

---

## 二、数据库算子 → AMX 映射分析

### 2.1 向量相似度计算 (最适合 AMX)

**场景**: 向量数据库的相似度搜索

```
余弦相似度: cos(a,b) = (a·b) / (||a|| × ||b||)
欧氏距离:   ||a-b||² = ||a||² + ||b||² - 2(a·b)
点积:       a·b = Σ(aᵢ × bᵢ)
```

**AMX 优化方案**:
```
查询向量 Q (1×D)
候选矩阵 C (N×D)  // N 个候选，D 维

相似度 = Q × Cᵀ   // (1×D) × (D×N) = (1×N)
                  // 一次矩阵乘法计算所有相似度
```

**预期加速**: 10-50x (vs 逐个计算)

### 2.2 聚合运算 (部分适合)

**传统聚合**:
```cpp
// SUM: 线性遍历
for (i = 0; i < N; i++) sum += data[i];
```

**AMX 优化 - 转换为点积**:
```cpp
// SUM = data · 1⃗ (与全1向量点积)
float ones[N] = {1.0f, 1.0f, ...};
float sum = cblas_sdot(N, data, 1, ones, 1);
```

**问题**: 需要额外的全1向量，内存开销可能抵消收益

**更好的方案 - 分块矩阵乘**:
```cpp
// 将 N 个元素视为 (N/K × K) 矩阵
// 乘以 (K × 1) 全1向量，得到 (N/K × 1) 部分和
// 最后累加部分和

// 例: 100万元素 → 1000×1000 矩阵 × 1000×1 向量
float partial_sums[1000];
cblas_sgemv(CblasRowMajor, CblasNoTrans,
            1000, 1000,    // 1000×1000 矩阵
            1.0f, data, 1000,
            ones_1000, 1,  // 1000×1 全1向量
            0.0f, partial_sums, 1);
// 再对 partial_sums 求和
```

**预期加速**: 2-5x (vs vDSP_sve，取决于数据规模)

### 2.3 多列聚合 (适合 AMX)

**场景**: 同时计算多列的 SUM/AVG

```sql
SELECT SUM(a), SUM(b), SUM(c), SUM(d) FROM table;
```

**传统方式**: 4 次独立遍历

**AMX 优化 - 转换为矩阵乘**:
```
数据矩阵 D (N×4)  // N 行，4 列
全1向量  1⃗ (N×1)

结果 = Dᵀ × 1⃗    // (4×N) × (N×1) = (4×1)
                 // 一次乘法得到所有列的 SUM
```

**预期加速**: 4-8x (vs 分别计算)

### 2.4 Bloom Filter 向量化 (可能适合)

**传统 Bloom 检查**:
```cpp
for (key : probe_keys) {
    for (hash_func : hash_funcs) {
        bit_idx = hash_func(key);
        if (!bloom_bits[bit_idx]) {
            // not found
            break;
        }
    }
}
```

**AMX 优化 - 批量哈希矩阵**:
```
Keys 矩阵 K (B×1)      // B 个 probe keys
Hash 系数 H (K×1)      // K 个哈希函数的系数

Bit 索引 = K × Hᵀ      // (B×1) × (1×K) = (B×K)
                       // 一次计算所有 key 的所有哈希位置
```

**问题**:
- 哈希函数通常是非线性的 (CRC32)
- 位查找是随机访问，无法向量化

**结论**: Bloom Filter 本身不太适合 AMX，但可以用 SIMD 优化哈希计算

### 2.5 Hash Join (有限适用)

**位图匹配优化**:
```cpp
// 将 build keys 编码为位图
// 使用向量化与操作检查 probe keys

// 但问题是：
// 1. 哈希计算是非线性的
// 2. 哈希表访问是随机的
// 3. 线性探测无法向量化
```

**结论**: Hash Join 核心循环不太适合 AMX，保持当前 Neon SIMD 优化

---

## 三、适用场景评估

| 算子 | AMX 适用性 | 预期加速 | 实现复杂度 | 优先级 |
|------|-----------|---------|-----------|--------|
| 向量相似度 | ⭐⭐⭐⭐⭐ | 10-50x | 低 | P0 |
| 多列聚合 | ⭐⭐⭐⭐ | 4-8x | 中 | P1 |
| 单列 SUM | ⭐⭐ | 2-5x | 中 | P2 |
| Bloom Filter | ⭐ | 1-2x | 高 | P3 |
| Hash Join | ⭐ | <1.5x | 高 | 不推荐 |

---

## 四、实现方案

### 4.1 向量相似度 (优先实现)

```cpp
// include/thunderduck/vector_ops.h

namespace thunderduck {
namespace vector {

/**
 * 批量计算点积相似度
 *
 * @param query     查询向量 (D 维)
 * @param candidates 候选矩阵 (N×D)
 * @param dim       向量维度 D
 * @param num_candidates 候选数量 N
 * @param out_scores 输出相似度 (N 个)
 */
void batch_dot_product_amx(
    const float* query,
    const float* candidates,
    size_t dim,
    size_t num_candidates,
    float* out_scores);

/**
 * 批量计算余弦相似度
 */
void batch_cosine_similarity_amx(
    const float* query,
    const float* candidates,
    size_t dim,
    size_t num_candidates,
    float* out_scores);

/**
 * 批量计算 L2 距离
 */
void batch_l2_distance_amx(
    const float* query,
    const float* candidates,
    size_t dim,
    size_t num_candidates,
    float* out_distances);

} // namespace vector
} // namespace thunderduck
```

### 4.2 实现代码

```cpp
// src/operators/vector/amx_similarity.cpp

#include <Accelerate/Accelerate.h>

namespace thunderduck {
namespace vector {

void batch_dot_product_amx(
    const float* query,
    const float* candidates,
    size_t dim,
    size_t num_candidates,
    float* out_scores) {

    // 使用 cblas_sgemv: y = α·A·x + β·y
    // A = candidates (N×D)
    // x = query (D×1)
    // y = scores (N×1)

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                static_cast<int>(num_candidates),  // M = N
                static_cast<int>(dim),              // N = D
                1.0f,                               // α
                candidates, static_cast<int>(dim), // A, lda
                query, 1,                           // x, incx
                0.0f,                               // β
                out_scores, 1);                     // y, incy
}

void batch_cosine_similarity_amx(
    const float* query,
    const float* candidates,
    size_t dim,
    size_t num_candidates,
    float* out_scores) {

    // 1. 计算 query 范数
    float query_norm = cblas_snrm2(static_cast<int>(dim), query, 1);

    // 2. 计算所有候选的范数
    std::vector<float> candidate_norms(num_candidates);
    for (size_t i = 0; i < num_candidates; i++) {
        candidate_norms[i] = cblas_snrm2(
            static_cast<int>(dim),
            candidates + i * dim, 1);
    }

    // 3. 计算点积
    batch_dot_product_amx(query, candidates, dim, num_candidates, out_scores);

    // 4. 归一化: score[i] /= (query_norm * candidate_norms[i])
    vDSP_vsdiv(out_scores, 1, &query_norm, out_scores, 1, num_candidates);
    vDSP_vdiv(candidate_norms.data(), 1, out_scores, 1, out_scores, 1, num_candidates);
}

} // namespace vector
} // namespace thunderduck
```

### 4.3 多列聚合优化

```cpp
// src/operators/aggregate/amx_aggregate.cpp

/**
 * 多列同时聚合 (利用 AMX 矩阵乘)
 *
 * @param data       列式数据 (N×C 矩阵，C 列)
 * @param num_rows   行数 N
 * @param num_cols   列数 C
 * @param out_sums   输出各列 SUM (C 个)
 */
void multi_column_sum_amx(
    const float* data,
    size_t num_rows,
    size_t num_cols,
    float* out_sums) {

    // 创建全1向量
    std::vector<float> ones(num_rows, 1.0f);

    // 矩阵乘: sums = Dᵀ × ones
    // D: N×C, ones: N×1 → sums: C×1
    cblas_sgemv(CblasColMajor, CblasTrans,
                static_cast<int>(num_rows),   // M = N
                static_cast<int>(num_cols),   // N = C
                1.0f,                          // α
                data, static_cast<int>(num_rows), // A, lda (列主序)
                ones.data(), 1,                // x
                0.0f,                          // β
                out_sums, 1);                  // y
}
```

---

## 五、性能基准测试计划

### 5.1 测试场景

```cpp
// 向量相似度测试
// D = 128, 256, 512, 1024 维
// N = 1K, 10K, 100K, 1M 候选

// 多列聚合测试
// N = 1M, 10M 行
// C = 4, 8, 16, 32 列
```

### 5.2 对比基准

| 方法 | 说明 |
|------|------|
| Scalar | 标量循环 |
| Neon SIMD | ARM Neon 向量化 |
| vDSP | vDSP 函数 |
| BLAS/AMX | cblas_sgemv/sgemm (AMX) |

---

## 六、结论与建议

### 6.1 推荐实现

1. **P0: 向量相似度** (batch_dot_product_amx)
   - 加速比最高 (10-50x)
   - 实现简单 (直接调用 cblas_sgemv)
   - 应用广泛 (向量数据库、推荐系统)

2. **P1: 多列聚合** (multi_column_sum_amx)
   - 加速比可观 (4-8x)
   - 适合 OLAP 多列统计
   - 需要列式存储支持

3. **保持现有优化**
   - 单列聚合: vDSP (已实现，2.5-6.5x)
   - Hash Join: Neon SIMD (AMX 不适用)
   - Filter: Neon SIMD (AMX 不适用)

### 6.2 不推荐的方向

- **Bloom Filter**: 哈希计算非线性，位访问随机
- **Hash Join**: 哈希表访问模式不适合矩阵运算
- **Sort**: 比较操作无法矩阵化

### 6.3 架构建议

```
ThunderDuck 加速层次:

┌─────────────────────────────────────────────────────────┐
│                    算子层                               │
├─────────────────────────────────────────────────────────┤
│  向量相似度    │  多列聚合    │  单列聚合   │  Join/Filter │
├───────────────┼─────────────┼────────────┼─────────────┤
│     AMX       │    AMX      │   vDSP     │  Neon SIMD  │
│  (cblas_sgemv)│(cblas_sgemv)│ (vDSP_sve) │ (ARM Neon)  │
├───────────────┴─────────────┴────────────┴─────────────┤
│              Apple Accelerate Framework                 │
├─────────────────────────────────────────────────────────┤
│                 Apple Silicon (M4)                      │
│        ┌───────┬───────┬────────┬──────────┐           │
│        │  AMX  │ Neon  │  GPU   │   ANE    │           │
│        └───────┴───────┴────────┴──────────┘           │
└─────────────────────────────────────────────────────────┘
```

---

## 七、实测结果 (2026-01-25)

### 7.1 批量点积 (向量相似度)

| 维度 D | 数量 N | Scalar | Neon | AMX/BLAS | AMX vs Neon |
|--------|--------|--------|------|----------|-------------|
| 128 | 1,000 | 33.8 μs | 10.8 μs | 2.2 μs | **4.9x** |
| 128 | 10,000 | 340.6 μs | 108.8 μs | 20.8 μs | **5.2x** |
| 128 | 100,000 | 2636 μs | 833 μs | 697 μs | 1.2x |
| 256 | 10,000 | 705 μs | 228 μs | 26.4 μs | **8.6x** |
| 512 | 10,000 | 1699 μs | 627 μs | 259 μs | **2.4x** |
| 1024 | 10,000 | 4245 μs | 1317 μs | 557 μs | **2.4x** |

**关键发现**:
- AMX 在 D=256, N=10K 时达到最佳加速比 **8.6x**
- 适合中等维度 (128-512) 的大批量计算
- 超大批量 (N=100K+) 时内存带宽成为瓶颈

### 7.2 多列聚合

| 行数 N | 列数 C | 分别 vDSP | AMX 矩阵乘 | AMX 加速 |
|--------|--------|-----------|-----------|----------|
| 100K | 4 | 12.4 μs | 45.2 μs | 0.27x (更慢) |
| 100K | 8 | 25.0 μs | 51.4 μs | 0.49x |
| 100K | 16 | 50.2 μs | 7.8 μs | **6.44x** |
| 1M | 4 | 154.6 μs | 464.6 μs | 0.33x |
| 1M | 8 | 350.4 μs | 525.6 μs | 0.67x |
| 10M | 8 | 4557 μs | 5612 μs | 0.81x |

**关键发现**:
- 多列聚合时 AMX 不总是更快
- vDSP_sve 已经高度优化，单列聚合更适合 vDSP
- 只有在列数较多 (C≥16) 且行数较少时 AMX 有优势
- **结论**: 保持 vDSP 用于聚合，AMX 专注于向量相似度

### 7.3 最终建议

| 操作 | 推荐方法 | 加速比 |
|------|---------|--------|
| 向量相似度 (点积/余弦/L2) | **AMX (cblas_sgemv)** | 2-8x vs Neon |
| 单列聚合 (SUM/AVG) | vDSP_sve | 2.5-6.5x vs Neon |
| 多列聚合 (C<16) | 分别调用 vDSP | - |
| 多列聚合 (C≥16) | AMX 矩阵乘 | ~6x |
| Filter/Join | Neon SIMD | (AMX 不适用) |

---

## 八、已完成的实现

1. [x] 实现 `batch_dot_product_f32()` - AMX 加速
2. [x] 实现 `batch_cosine_similarity_f32()` - AMX 加速
3. [x] 实现 `batch_l2_distance_f32()` - AMX 加速
4. [x] 实现 `multi_column_sum_f32()` - AMX 矩阵乘
5. [x] 创建基准测试对比 AMX vs Neon vs vDSP

### 文件清单

```
include/thunderduck/
└── vector_ops.h           # 向量操作接口

src/operators/vector/
└── amx_vector_ops.cpp     # AMX 实现

benchmark/
└── test_amx_vector.cpp    # 基准测试
```

---

## 九、结论

基于实测数据，Apple AMX 的最佳应用场景是:

1. **向量数据库相似度搜索** ⭐⭐⭐⭐⭐
   - 批量点积: 8.6x 加速
   - 余弦相似度: 高效计算
   - L2 距离: 利用点积优化

2. **机器学习特征计算** ⭐⭐⭐⭐
   - 向量-矩阵乘法
   - 批量特征变换

3. **OLAP 多列统计** ⭐⭐ (有限)
   - 仅在列数 ≥16 时有优势
   - 大多数情况 vDSP 更优

**不推荐用于**:
- Hash Join (随机访问模式)
- Bloom Filter (非线性哈希)
- 单列聚合 (vDSP 已足够优化)

---

*ThunderDuck - 深度挖掘 Apple Silicon 潜力*
