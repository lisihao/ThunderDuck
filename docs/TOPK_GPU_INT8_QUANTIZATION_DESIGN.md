# TopK GPU 分区 & INT8 量化 - 设计文档

> 版本: 1.0.0 | 日期: 2026-01-26 | 作者: Claude

## 一、问题分析

### 1.1 TopK 超大数据集瓶颈

**当前实现** (topk_v4.cpp):
- 采样预过滤 + SIMD 批量跳过
- 大数据 (10M+) + 小 K 时使用采样策略

**性能数据**:
| 数据量 | v3 时间 | v4 时间 | vs DuckDB |
|--------|---------|---------|-----------|
| 100K | 11.6μs | 11.6μs | 40.9x |
| 1M | 46.4μs | 49.2μs | 26.9x |
| 10M | 478μs | 478μs | 5.6x |

**问题**:
- 10M 时 vs DuckDB 只有 5.6x，优势下降
- CPU 单线程处理，无法利用 GPU 并行
- 100M+ 数据时内存带宽成为瓶颈

### 1.2 Vector Similarity 带宽瓶颈

**当前状态**:
- 100K+ 向量时 AMX vs Neon 只有 1.2x
- 内存带宽 ~70 GB/s（理论 400 GB/s）
- Float32 精度浪费内存带宽

**INT8 量化收益分析**:
| 精度 | 内存占用 | 理论带宽提升 |
|------|----------|--------------|
| Float32 | 4 bytes | 1x (baseline) |
| INT8 | 1 byte | 4x |

## 二、优化方案

### 方案 A: GPU 分区 TopK

**核心思想**: 将数据分区，GPU 并行处理每个分区的 TopK，最后归并

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Data (100M ints)                   │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│ Partition 0│ Partition 1│ Partition 2│ ...        │ Part N-1│
│   10M      │   10M      │   10M      │            │   10M   │
├────────────┼────────────┼────────────┼────────────┼─────────┤
│  TopK_0    │  TopK_1    │  TopK_2    │    ...     │ TopK_N-1│
│  (K个)     │  (K个)     │  (K个)     │            │  (K个)  │
└────────────┴────────────┴────────────┴────────────┴─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Merge TopK (N×K → K)      │
              │     使用堆合并 O(NK log K)     │
              └───────────────────────────────┘
```

**GPU Kernel 设计**:
```metal
// 每个 threadgroup 处理一个分区
kernel void partition_topk(
    device const int32_t* input,
    device int32_t* partition_results,
    device uint* partition_indices,
    constant uint& partition_size,
    constant uint& k,
    uint tgid [[threadgroup_position_in_grid]],
    threadgroup int32_t* shared_heap
) {
    // 1. 协作加载分区数据到 shared memory
    // 2. 使用 bitonic sort 找 TopK
    // 3. 写入分区结果
}
```

**预期收益**:
- GPU 并行处理多个分区
- 减少数据传输 (只传输 N×K 个结果)
- 100M 数据预期 3-5x 加速

### 方案 B: INT8 量化向量相似度

**量化流程**:
```
Float32 向量 [v0, v1, ..., vn]
    │
    ▼ 量化参数计算
scale = (max - min) / 255
zero_point = round(-min / scale)
    │
    ▼ 量化
INT8 向量 [q0, q1, ..., qn]
qi = clamp(round(vi / scale) + zero_point, 0, 255)
```

**INT8 点积计算**:
```cpp
// 使用 ARM Neon SDOT 指令
int32x4_t dot_i8(int8x16_t a, int8x16_t b) {
    return vdotq_s32(vdupq_n_s32(0), a, b);
}
```

**精度分析**:
- 典型场景: 余弦相似度搜索
- 精度损失: < 0.1% (对 Top-10 召回率)
- 适用场景: 向量检索、推荐系统

## 三、实现计划

| 阶段 | 任务 | 文件 |
|------|------|------|
| 1 | GPU 分区 TopK shader | `src/gpu/shaders/topk_partition.metal` |
| 2 | GPU TopK 包装 | `src/gpu/topk_partition_metal.mm` |
| 3 | INT8 量化器 | `include/thunderduck/quantization.h` |
| 4 | INT8 向量运算 | `src/operators/vector/int8_vector_ops.cpp` |
| 5 | 集成测试 | `benchmark/` |

## 四、关键技术点

### 4.1 GPU TopK 分区策略

```cpp
// 分区大小选择
constexpr size_t GPU_PARTITION_SIZE = 1 << 20;  // 1M per partition
constexpr size_t GPU_TOPK_THRESHOLD = 10000000; // 10M+ 使用 GPU

// 分区数计算
size_t num_partitions = (count + GPU_PARTITION_SIZE - 1) / GPU_PARTITION_SIZE;
```

### 4.2 INT8 SDOT 加速

```cpp
// ARM Neon SDOT: 4x4 点积累加
// a: 16 个 int8, b: 16 个 int8
// result: 4 个 int32 (每个是 4 对 int8 的点积和)
int32x4_t result = vdotq_s32(acc, a, b);
```

**性能对比**:
| 实现 | 指令数/元素 | 吞吐量 |
|------|-------------|--------|
| Float32 FMLA | 1 | 8 FLOPS/cycle |
| INT8 SDOT | 0.25 | 32 OPS/cycle |

### 4.3 量化误差控制

```cpp
struct QuantizationParams {
    float scale;
    int32_t zero_point;

    // 反量化
    float dequantize(int8_t q) const {
        return (q - zero_point) * scale;
    }

    // 点积反量化
    float dequantize_dot(int32_t dot_q, size_t dim) const {
        return dot_q * scale * scale;
    }
};
```

## 五、实现结果

### 5.1 已实现文件

| 文件 | 说明 |
|------|------|
| `src/gpu/shaders/topk_partition.metal` | GPU TopK 分区 Metal shaders |
| `src/gpu/topk_partition_metal.mm` | GPU TopK 包装代码 |
| `include/thunderduck/quantization.h` | INT8 量化参数和批量转换 |
| `src/operators/vector/int8_vector_ops.cpp` | INT8 SDOT 向量运算 |

### 5.2 GPU TopK 分区实现

**关键组件**:

1. **Metal Shader** (`topk_partition.metal`):
   - `partition_topk_bitonic`: 每个 threadgroup 处理一个分区
   - `partition_topk_sampled`: 采样阈值版本
   - `merge_partition_topk`: GPU 合并 kernel

2. **Objective-C++ 包装** (`topk_partition_metal.mm`):
   - `TopKPartitionPipeline`: 单例管理 Metal 资源
   - `estimate_threshold_by_sampling`: CPU 采样估计阈值
   - `merge_partition_results`: CPU 归并分区结果
   - `partition_topk_gpu`: 主接口

**算法流程**:
```
Input (100M) → 分区 (100 × 1M) → GPU 并行 TopK → 归并 → Output (K)
```

### 5.3 INT8 量化实现

**量化参数** (`quantization.h`):
```cpp
struct QuantizationParams {
    float scale;
    int32_t zero_point;

    int8_t quantize(float v) const;
    float dequantize(int8_t q) const;
    float dequantize_dot(int32_t dot_q, const QuantizationParams& other) const;
};
```

**批量操作**:
- `quantize_batch`: Neon 加速 float→int8
- `dequantize_batch`: Neon 加速 int8→float
- `QuantizedVector`: 单向量量化表示
- `QuantizedMatrix`: 矩阵量化表示（每行独立参数）

**INT8 向量运算** (`int8_vector_ops.cpp`):
- `dot_product_i8_sdot`: SDOT 加速点积
- `dot_product_i8_neon`: Neon 回退实现
- `batch_dot_product_i8`: 批量点积
- `batch_cosine_similarity_i8`: INT8 余弦相似度
- `batch_l2_distance_squared_i8`: INT8 L2 距离

### 5.4 预期性能收益

| 优化 | 场景 | 预期收益 |
|------|------|----------|
| GPU TopK | 100M 数据 | 3-5x vs CPU |
| INT8 点积 | 大批量向量 | 2-4x 吞吐量 |
| INT8 带宽 | 内存受限场景 | 4x 带宽节省 |

## 六、验证计划

```bash
# 编译并测试
make clean && make lib benchmark full-report

# 期望结果
# TopK 100M: GPU vs CPU 3-5x
# Vector INT8: vs Float32 3-4x 吞吐量提升
```

## 七、后续优化方向

1. **GPU TopK 优化**:
   - 使用 Radix Select 替代 Bitonic Sort
   - 实现 GPU 端归并减少 CPU 开销

2. **INT8 优化**:
   - 矩阵级共享量化参数减少存储开销
   - 与 GPU Metal 结合实现混合精度
