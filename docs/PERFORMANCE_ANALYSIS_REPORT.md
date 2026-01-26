# ThunderDuck 全面性能分析报告

> **版本**: 1.0.0 | **日期**: 2026-01-26 | **平台**: Apple Silicon M4

## 一、测试环境

| 项目 | 配置 |
|------|------|
| **CPU** | Apple M4 (4 P-cores + 6 E-cores) |
| **内存** | 统一内存架构 (UMA) |
| **理论带宽** | ~400 GB/s |
| **L1 Cache** | 64 KB / core |
| **L2 Cache** | 4 MB shared |
| **Cache Line** | 128 bytes |
| **加速器** | Neon SIMD, AMX, GPU Metal, NPU BNNS |

## 二、算子性能详情

### 2.1 Filter 算子

**SQL**: `SELECT * FROM t WHERE value > threshold`

| 数据量 | 数据大小 | 硬件路径 | v3 时间 | v5 时间 | DuckDB | v5 vs v3 | vs DuckDB | 带宽 |
|--------|----------|----------|---------|---------|--------|----------|-----------|------|
| 100K | 400 KB | CPU Neon | 3.0 μs | 3.0 μs | 136.8 μs | 1.00x | **45.6x** | 133 GB/s |
| 1M | 4 MB | CPU Neon | 33.8 μs | 32.2 μs | 258.2 μs | 1.05x | **8.0x** | 124 GB/s |
| 10M | 40 MB | CPU Neon | 507 μs | 384.6 μs | 1427 μs | 1.32x | **3.7x** | 104 GB/s |

**分析**:
- 小数据量 (100K): 带宽 133 GB/s，接近缓存带宽
- 大数据量 (10M): 带宽 104 GB/s，受内存瓶颈影响
- vs DuckDB 加速比随数据量增大而下降 (45.6x → 3.7x)

**优化机会**:
- ⚠️ 10M 时带宽仅 104 GB/s (理论 400 GB/s)
- 建议: GPU Metal 并行过滤，预期提升 2-3x

---

### 2.2 Aggregate 算子

**SQL**: `SELECT SUM(v), AVG(v), MIN(v), MAX(v) FROM t`

| 数据量 | 数据大小 | 硬件路径 | SIMD 时间 | vDSP 时间 | DuckDB | vDSP vs SIMD | vs DuckDB | 带宽 |
|--------|----------|----------|-----------|-----------|--------|--------------|-----------|------|
| 100K | 400 KB | CPU vDSP | 51.2 μs | 9.6 μs | 674 μs | **5.33x** | **70.2x** | 42 GB/s |
| 1M | 4 MB | CPU vDSP | 516 μs | 97 μs | 1178 μs | **5.32x** | **12.1x** | 41 GB/s |
| 10M | 40 MB | CPU vDSP | 5113 μs | 1624 μs | 7982 μs | **3.15x** | **4.9x** | 25 GB/s |

**分析**:
- vDSP/AMX 提供 3-5x 加速（vs 纯 SIMD）
- 10M 时带宽仅 25 GB/s，明显低于 Filter

**优化机会**:
- ⚠️ 10M 时 vs DuckDB 仅 4.9x，vDSP vs SIMD 仅 3.15x
- ⚠️ 带宽 25 GB/s 远低于理论值
- 建议: 多线程并行 + 更激进预取

---

### 2.3 Hash Join 算子

**SQL**: `SELECT COUNT(*) FROM build b JOIN probe p ON b.key = p.key`

| Build×Probe | 数据大小 | 匹配数 | v3 时间 | v4 AUTO | DuckDB | v4 vs v3 | vs DuckDB |
|-------------|----------|--------|---------|---------|--------|----------|-----------|
| 10K×100K | 440 KB | 100K | 49 μs | 48.7 μs | 506 μs | 1.01x | **10.4x** |
| 100K×1M | 4 MB | 100K | 178 μs | 174 μs | 1336 μs | 1.03x | **7.7x** |
| 100K×1M | 4 MB | 1M | 598 μs | 514 μs | 1414 μs | 1.16x | **2.75x** |

**分析**:
- v4 vs v3 加速有限 (1.01-1.16x)
- 高匹配场景 (1M 匹配) 性能下降明显

**优化机会**:
- ⚠️ 高匹配场景 vs DuckDB 仅 2.75x
- ⚠️ v4 vs v3 提升不明显
- 建议:
  - GPU Metal 并行探测
  - 更好的结果缓冲区预分配

---

### 2.4 TopK 算子

**SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT K`

| 数据量 | K 值 | 硬件路径 | v3 时间 | v5 时间 | DuckDB | v5 vs v3 | vs DuckDB |
|--------|------|----------|---------|---------|--------|----------|-----------|
| 100K | 100 | CPU Neon | 11.6 μs | 10.8 μs | 488 μs | 1.07x | **45.2x** |
| 1M | 100 | CPU Neon | 47.2 μs | 51.4 μs | 1066 μs | 0.92x | **22.6x** |
| 10M | 100 | CPU Neon | 540 μs | 530.6 μs | 3180 μs | 1.02x | **6.2x** |

**分析**:
- 大数据量时 vs DuckDB 加速下降 (45x → 6x)
- v5 采样策略在 1M 时反而更慢

**优化机会**:
- ⚠️ 10M 时 vs DuckDB 仅 6.2x
- ⚠️ v5 vs v3 提升不明显，1M 时还有回退
- 建议:
  - GPU 分区 TopK (已实现，待集成)
  - 优化采样阈值策略

---

### 2.5 Vector Similarity 算子

**SQL**: `-- Batch Dot Product (向量相似度搜索)`

| 向量数×维度 | 数据大小 | Scalar | Neon SIMD | AMX/BLAS | AMX vs Neon | 带宽 |
|-------------|----------|--------|-----------|----------|-------------|------|
| 10K×128 | 5 MB | 249.6 μs | 88.8 μs | 18.4 μs | **4.8x** | 278 GB/s |
| 100K×128 | 51 MB | 2520 μs | 857.8 μs | 714.8 μs | **1.2x** | 72 GB/s |
| 100K×256 | 102 MB | 7138 μs | 2160 μs | 1495 μs | **1.4x** | 69 GB/s |
| 100K×512 | 204 MB | 17063 μs | 6228 μs | 2827 μs | **2.2x** | 72 GB/s |
| 1M×128 | 512 MB | 25944 μs | 9796 μs | 7166 μs | **1.4x** | 71 GB/s |

**分析**:
- 小批量 (10K): AMX 4.8x 加速，带宽 278 GB/s（接近缓存）
- 大批量 (100K+): AMX 优势下降到 1.2-2.2x
- 带宽稳定在 ~70 GB/s (理论 400 GB/s)

**优化机会**:
- ⚠️ 大批量时带宽仅 70 GB/s，利用率 ~18%
- ⚠️ AMX 优势在大批量时下降
- 建议:
  - INT8 量化 (已实现): 4x 带宽节省
  - GPU Metal 并行 (已实现框架)

---

## 三、优化优先级

### 高优先级 (预期收益 2x+)

| 优化项 | 当前性能 | 目标 | 实现方案 |
|--------|----------|------|----------|
| **Hash Join GPU** | 2.75x vs DuckDB | 4-5x | GPU Metal 并行探测 |
| **Filter GPU** | 3.7x @ 10M | 6-8x | GPU Metal 并行过滤 |
| **Vector INT8** | 70 GB/s | 150+ GB/s | INT8 量化 + SDOT |

### 中优先级 (预期收益 1.5x)

| 优化项 | 当前性能 | 目标 | 实现方案 |
|--------|----------|------|----------|
| **Aggregate 并行** | 25 GB/s @ 10M | 50+ GB/s | 多线程 + 预取优化 |
| **TopK GPU** | 6x @ 10M | 10-15x | GPU 分区 TopK |

### 低优先级 (已接近极限)

| 优化项 | 当前性能 | 状态 |
|--------|----------|------|
| **Filter 小数据** | 45x vs DuckDB | ✓ 已优化 |
| **Aggregate 小数据** | 70x vs DuckDB | ✓ 已优化 |
| **TopK 小数据** | 45x vs DuckDB | ✓ 已优化 |

---

## 四、已实现待集成

| 功能 | 文件 | 状态 |
|------|------|------|
| GPU TopK 分区 | `src/gpu/topk_partition_metal.mm` | ✓ 已实现 |
| INT8 量化 | `include/thunderduck/quantization.h` | ✓ 已实现 |
| INT8 向量运算 | `src/operators/vector/int8_vector_ops.cpp` | ✓ 已实现 |
| Filter 压缩结果 | `src/operators/filter/filter_result.cpp` | ✓ 已实现 |
| GPU 向量相似度 | `src/gpu/vector_similarity_metal.mm` | ✓ 已实现 |

---

## 五、带宽分析

### 理论 vs 实际

| 场景 | 理论带宽 | 实际带宽 | 利用率 |
|------|----------|----------|--------|
| L1 Cache | ~1 TB/s | 133 GB/s (Filter 100K) | ~13% |
| L2 Cache | ~400 GB/s | 104 GB/s (Filter 10M) | ~26% |
| 内存 | ~200 GB/s | 25-70 GB/s | 12-35% |

### 瓶颈分析

1. **Filter/TopK**: 内存带宽受限，可通过预取和 GPU 优化
2. **Aggregate**: 计算密集，vDSP 已接近极限
3. **Hash Join**: 随机访问模式导致缓存效率低
4. **Vector**: 大批量时内存带宽成为瓶颈

---

## 六、下一步行动

1. **集成 GPU TopK**: 将 `topk_partition_metal.mm` 集成到主路径
2. **集成 INT8 量化**: 在 Vector Similarity 中启用 INT8 路径
3. **GPU Hash Join**: 实现 GPU 并行探测
4. **Query Optimizer**: 自动选择最优硬件路径
