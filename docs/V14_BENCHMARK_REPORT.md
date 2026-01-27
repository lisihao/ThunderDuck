# ThunderDuck V14 全面性能基准报告

> **测试日期**: Jan 27 2026 20:22:10
> **平台**: Apple M4 Max
> **测试配置**: iterations=15, warmup=2

## 一、执行摘要

本报告测试了 ThunderDuck 各算子版本与 DuckDB 的性能对比。

### 最佳性能摘要

| 算子 | 数据量 | 最佳版本 | 设备 | vs DuckDB |
|------|--------|----------|------|-----------|
| Filter | 1M | V6 (prefetch) | CPU SIMD | 1.06x |
| Filter | 10M | V4 (AUTO) | GPU Auto | 1.30x |
| GROUP BY SUM | 10M | V15 (8T+展开) | CPU 8T | 2.87x |
| INNER JOIN | 1M | V14 (预分配) | CPU SIMD | 1.55x |
| SEMI JOIN | 1M | GPU (Metal) | Metal | 2.02x |
| TopK | 10M | V4 (sample) | CPU SIMD | 5.11x |

## 二、详细测试结果

### 2.1 Filter 算子

**等效 SQL**: `SELECT * FROM t WHERE value > 500000`

| 版本 | 设备 | 数据量 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 1M | 0.401 | 9.98 | 1.00x | - | PASS |
| V3 (bitmap) | CPU SIMD | 1M | 0.407 | 9.82 | 0.98x | 1.00x | PASS |
| V5 (LUT) | CPU SIMD | 1M | 0.393 | 10.19 | 1.02x | 1.03x | PASS |
| V6 (prefetch) | CPU SIMD | 1M | 0.380 | 10.54 | 1.06x | 1.07x | PASS |
| V15 (direct) | CPU SIMD | 1M | 0.440 | 9.09 | 0.91x | 0.92x | PASS |
| V4 (AUTO) | CPU Auto | 1M | 0.395 | 10.12 | 1.01x | 1.03x | PASS |
| Parallel (4T) | CPU 4T | 1M | 0.986 | 4.06 | 0.41x | 0.41x | PASS |
| DuckDB | CPU | 10M | 2.991 | 13.37 | 1.00x | - | PASS |
| V3 (bitmap) | CPU SIMD | 10M | 4.016 | 9.96 | 0.74x | 1.00x | PASS |
| V5 (LUT) | CPU SIMD | 10M | 4.009 | 9.98 | 0.75x | 1.00x | PASS |
| V6 (prefetch) | CPU SIMD | 10M | 4.012 | 9.97 | 0.75x | 1.00x | PASS |
| V15 (direct) | CPU SIMD | 10M | 3.262 | 12.26 | 0.92x | 1.23x | PASS |
| V4 (AUTO) | GPU Auto | 10M | 2.309 | 17.33 | 1.30x | 1.73x | PASS |
| Parallel (4T) | CPU 4T | 10M | 5.935 | 6.74 | 0.50x | 0.67x | PASS |

### 2.2 GROUP BY SUM 算子

**等效 SQL**: `SELECT group_id, SUM(value) FROM t GROUP BY group_id`

| 版本 | 设备 | 数据量 | 分组数 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V4 | 正确性 |
|------|------|--------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 1000 | 2.552 | 31.35 | 1.00x | - | PASS |
| V4 (single) | CPU SIMD | 10M | 1000 | 2.909 | 27.51 | 0.88x | 1.00x | PASS |
| V4 (parallel) | CPU 4T | 10M | 1000 | 1.209 | 66.18 | 2.11x | 2.40x | PASS |
| V6 (smart) | CPU/GPU Auto | 10M | 1000 | 1.908 | 41.93 | 1.34x | 1.52x | PASS |
| V14 (parallel) | CPU 8T | 10M | 1000 | 1.416 | 56.48 | 1.80x | 2.05x | PASS |
| V15 (8T+展开) | CPU 8T | 10M | 1000 | 0.890 | 89.88 | 2.87x | 3.26x | PASS |

### 2.3 INNER JOIN 算子

**等效 SQL**: `SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 100K | 1M | 1.839 | 2.39 | 1.00x | - | PASS |
| V3 | CPU SIMD | 100K | 1M | 1.297 | 3.39 | 1.42x | 1.00x | PASS |
| V6 (prefetch) | CPU SIMD | 100K | 1M | 1.394 | 3.16 | 1.32x | 0.93x | PASS |
| V10 | CPU SIMD | 100K | 1M | 1.306 | 3.37 | 1.41x | 0.99x | PASS |
| V11 (SIMD probe) | CPU SIMD | 100K | 1M | 3.528 | 1.25 | 0.52x | 0.36x | PASS |
| V13 (两阶段) | CPU SIMD | 100K | 1M | 19.100 | 0.23 | 0.10x | 0.06x | PASS |
| V14 (预分配) | CPU SIMD | 100K | 1M | 1.190 | 3.70 | 1.55x | 1.08x | PASS |

### 2.4 SEMI JOIN 算子

**等效 SQL**: `SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE ...)`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V10 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|--------|--------|
| DuckDB | CPU | 100K | 1M | 2.897 | 1.52 | 1.00x | - | PASS |
| V10 | CPU SIMD | 100K | 1M | 3.859 | 1.14 | 0.75x | 1.00x | PASS |
| GPU (Metal) | Metal | 100K | 1M | 1.431 | 3.07 | 2.02x | 2.69x | PASS |

### 2.5 TopK 算子

**等效 SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT 10`

| 版本 | 设备 | 数据量 | K | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|--------|---|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 10 | 2.736 | 14.62 | 1.00x | - | PASS |
| V3 (adaptive) | CPU | 10M | 10 | 4.881 | 8.20 | 0.56x | 1.00x | PASS |
| V4 (sample) | CPU SIMD | 10M | 10 | 0.535 | 74.76 | 5.11x | 9.12x | PASS |
| V5 (count) | CPU | 10M | 10 | 0.645 | 62.00 | 4.24x | 7.56x | PASS |
| V6 (UMA) | CPU/GPU Auto | 10M | 10 | 0.667 | 59.99 | 4.10x | 7.31x | PASS |

## 三、优化建议

### 潜在优化点

#### 需要优化 (vs DuckDB < 1.0x):

- **Filter V3 (bitmap)**: 0.98x
- **Filter V15 (direct)**: 0.91x
- **Filter Parallel (4T)**: 0.41x
- **Filter V3 (bitmap)**: 0.74x
- **Filter V5 (LUT)**: 0.75x
- **Filter V6 (prefetch)**: 0.75x
- **Filter V15 (direct)**: 0.92x
- **Filter Parallel (4T)**: 0.50x
- **GROUP BY SUM V4 (single)**: 0.88x
- **INNER JOIN V11 (SIMD probe)**: 0.52x
- **INNER JOIN V13 (两阶段)**: 0.10x
- **SEMI JOIN V10**: 0.75x
- **TopK V3 (adaptive)**: 0.56x

#### 表现优异 (vs DuckDB >= 2.0x):

- **GROUP BY SUM V4 (parallel)**: 2.11x
- **GROUP BY SUM V15 (8T+展开)**: 2.87x
- **SEMI JOIN GPU (Metal)**: 2.02x
- **TopK V4 (sample)**: 5.11x
- **TopK V5 (count)**: 4.24x
- **TopK V6 (UMA)**: 4.10x

## 四、版本历史对比

| 版本 | 主要优化 | 关键算子性能 |
|------|----------|-------------|
| V3 | 基础 SIMD | 基准版本 |
| V6 | 预取优化 | Join 1.3x |
| V10 | 完整语义 | SEMI/ANTI Join |
| V11 | SIMD 探测 | - |
| V13 | 两阶段算法 | Join 1.2x |
| V14 | GPU 加速 | SEMI Join 2.2x, Filter 1.5x |
