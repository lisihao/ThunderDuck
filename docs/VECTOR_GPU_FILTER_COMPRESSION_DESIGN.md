# Vector GPU 并行 & Filter 结果压缩 - 设计文档

> 版本: 1.0.0 | 日期: 2026-01-26 | 作者: Claude

## 一、问题分析

### 1.1 Vector 大批量性能下降

**Benchmark 数据**:
| 向量数×维度 | AMX vs Neon | 带宽 |
|-------------|-------------|------|
| 10K×128 | 4.9x | 278 GB/s |
| 100K×128 | 1.1x | 65 GB/s |
| 1M×128 | 1.2x | 67 GB/s |

**根因分析**:
1. AMX/BLAS 使用 `cblas_sgemv` 做矩阵-向量乘法
2. 小数据量时，AMX 计算优势明显 (4.9x)
3. 大数据量时，内存带宽成为瓶颈 (65 GB/s << 理论 400 GB/s)
4. AMX 和 Neon 都受限于相同的 CPU 内存控制器
5. GPU 有独立的内存子系统，可并行处理

### 1.2 Filter 物化开销

**当前实现**:
```
输入数据 (10M int32) → SIMD比较 → 位图 (156KB) → 索引数组 (最大 40MB)
```

**问题**:
- 高选择率时，索引数组巨大
- 位图→索引转换有 O(n) 开销
- 索引数组的分配和复制开销

## 二、优化方案

### 方案 A: GPU Metal 向量相似度

**核心思想**: 利用 GPU 的高并行度和独立内存带宽

```metal
// Metal compute shader
kernel void batch_dot_product(
    constant float* query [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        sum += query[i] * candidates[idx * dim + i];
    }
    scores[idx] = sum;
}
```

**预期收益**:
- GPU 内存带宽: ~400 GB/s (M4 Pro)
- 并行线程: 数千个
- 100K+ 向量时预期 2-4x 加速

### 方案 B: Filter 结果压缩 (选择率自适应)

**策略选择**:
| 选择率 | 策略 | 表示方式 |
|--------|------|----------|
| <10% | 直接索引 | `uint32_t[]` |
| 10-50% | 位图 | `uint64_t[]` (1.5KB/100K) |
| >50% | 反向索引 | 记录不匹配的位置 |

**位图接口**:
```cpp
struct FilterResult {
    enum Type { INDICES, BITMAP, INVERTED };
    Type type;
    size_t count;           // 匹配数
    size_t total;           // 总数
    union {
        uint32_t* indices;  // Type::INDICES
        uint64_t* bitmap;   // Type::BITMAP
    } data;
};
```

**惰性评估**:
- 下游操作直接使用位图
- 仅在需要具体索引时才转换

## 三、实现计划

| 阶段 | 任务 | 文件 |
|------|------|------|
| 1 | GPU Metal 向量相似度 | `src/gpu/vector_similarity_metal.mm` |
| 2 | Filter 位图优先模式 | `src/operators/filter/simd_filter_v6.cpp` |
| 3 | FilterResult 压缩表示 | `include/thunderduck/filter.h` |
| 4 | 惰性评估接口 | `include/thunderduck/filter_result.h` |

## 四、关键代码位置

| 文件 | 说明 |
|------|------|
| `src/operators/vector/amx_vector_ops.cpp:163` | AMX batch_dot_product |
| `src/operators/filter/simd_filter_v5.cpp:161` | 位图过滤实现 |
| `src/gpu/filter_uma.mm` | 现有 GPU Filter |

## 五、实现结果

### 5.1 已实现文件

| 文件 | 说明 |
|------|------|
| `src/gpu/shaders/vector_similarity.metal` | GPU Metal compute shaders |
| `src/gpu/vector_similarity_metal.mm` | Metal 包装代码 |
| `include/thunderduck/filter_result.h` | CompressedFilterResult 类定义 |
| `src/operators/filter/filter_result.cpp` | 选择率自适应实现 |

### 5.2 Vector Similarity 性能数据

| 向量数×维度 | Scalar | Neon | AMX | AMX vs Neon |
|-------------|--------|------|-----|-------------|
| 10K×128 | 251.6μs | 82.4μs | 18.2μs | 4.5x |
| 100K×128 | 2532μs | 862μs | 721μs | 1.2x |
| 100K×512 | 16988μs | 6467μs | 2806μs | 2.3x |
| 1M×128 | 25481μs | 8738μs | 7072μs | 1.2x |

**关键洞察**:
- 小批量 (10K): AMX 优势明显 (4.5x vs Neon)
- 大批量 (100K+): AMX 优势下降到 1.2-2.3x
- 根因: 内存带宽瓶颈 (65-73 GB/s vs 理论 400 GB/s)

### 5.3 Filter 压缩策略

| 选择率 | 策略 | 内存占用 (10M 元素) |
|--------|------|---------------------|
| <10% | 索引数组 | 最大 4MB |
| 10-50% | 位图 | 1.25MB 固定 |
| >50% | 位图/反向 | 1.25MB 固定 |

**收益**:
- 中/高选择率时内存占用降低 50-90%
- 位图支持高效集合操作 (AND/OR/NOT)
- 惰性评估避免不必要的转换

### 5.4 后续优化方向

1. **GPU Vector Similarity 集成测试**: 添加 GPU 路径到 benchmark
2. **Filter GPU 并行**: 利用 GPU 处理位图过滤
3. **Query Optimizer 集成**: 自动选择最优执行路径
