# ThunderDuck V14 完整性能基准报告

> **测试日期**: Jan 27 2026 20:45:52
> **平台**: Apple M4 Max
> **测试配置**: iterations=15, warmup=2, IQR 剔除异常值

## 一、执行摘要

本报告测试了 ThunderDuck **所有算子的所有版本**与 DuckDB 的性能对比。

### 最佳性能摘要

| 算子 | 数据量 | 最佳版本 | 设备 | vs DuckDB | 提升原因 |
|------|--------|----------|------|-----------|----------|
| Filter | 10M | **V4 (GPU AUTO)** | Metal | **1.10x** | GPU 并行扫描 |
| GROUP BY SUM | 10M | **V15 (8T+unroll)** | CPU 8T | **2.69x** | 8 线程 + 循环展开 |
| INNER JOIN | 100K×1M | **V14 (pre-alloc)** | CPU SIMD | **1.63x** | 预分配消除 realloc |
| SEMI JOIN | 100K×1M | **GPU (Metal)** | Metal | **2.47x** | GPU 并行哈希探测 |
| TopK | 10M | **V4 (sample)** | CPU SIMD | **4.71x** | 采样 + SIMD 筛选 |

---

## 二、详细测试结果

### 2.1 Filter 算子

**等效 SQL**: `SELECT rowid FROM t WHERE value > 500000`

| 版本 | 设备 | 数据量 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V1 | 正确性 |
|------|------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 2.679 | 14.93 | 1.00x | - | PASS |
| **V1 (base)** | CPU SIMD | 10M | 18.303 | 2.19 | 0.14x | 1.00x | PASS |
| **V2 (bitmap)** | CPU SIMD | 10M | 4.923 | 8.12 | 0.54x | 3.71x | PASS |
| **V3 (template)** | CPU SIMD | 10M | 3.839 | 10.42 | 0.69x | 4.76x | PASS |
| **V5 (LUT+cache)** | CPU SIMD | 10M | 3.848 | 10.40 | 0.69x | 4.75x | PASS |
| **V6 (prefetch)** | CPU SIMD | 10M | 3.848 | 10.39 | 0.69x | 4.75x | PASS |
| **V15 (direct)** | CPU SIMD | 10M | 3.239 | 12.35 | 0.82x | 5.65x | PASS |
| **V4 (GPU AUTO)** | Metal | 10M | 2.424 | 16.50 | **1.10x** | 7.55x | PASS |
| **Parallel (4T)** | CPU 4T | 10M | 6.016 | 6.65 | 0.44x | 3.04x | PASS |

**Filter 演进分析**:
```
V1 (0.14x) → V2 (0.54x) → V3 (0.69x) → V15 (0.82x) → V4 GPU (1.10x)
                 +3.7x       +1.3x         +1.2x          +1.3x
```

---

### 2.2 GROUP BY SUM 算子

**等效 SQL**: `SELECT group_id, SUM(value) FROM t GROUP BY group_id`

| 版本 | 设备 | 数据量 | 分组数 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V1 | 正确性 |
|------|------|--------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 1000 | 2.609 | 30.66 | 1.00x | - | PASS |
| **V1 (base)** | CPU | 10M | 1000 | 2.844 | 28.13 | 0.91x | 1.00x | PASS |
| **V4 (SIMD)** | CPU SIMD | 10M | 1000 | 2.734 | 29.26 | 0.95x | 1.04x | PASS |
| **V4 (parallel)** | CPU 4T | 10M | 1000 | 1.176 | 68.03 | **2.21x** | 2.41x | PASS |
| **V5 (GPU 2-phase)** | Metal | 10M | 1000 | 2.143 | 37.33 | 1.21x | 1.32x | PASS |
| **V6 (smart)** | CPU/GPU Auto | 10M | 1000 | 1.169 | 68.44 | **2.23x** | 2.43x | PASS |
| **V12.1 (GPU warp)** | Metal | 10M | 1000 | 2.317 | 34.53 | 1.12x | 1.22x | PASS |
| **V13 (GPU no-atomic)** | Metal | 10M | 1000 | 9.996 | 8.00 | 0.26x | 0.28x | PASS |
| **V14 (8T parallel)** | CPU 8T | 10M | 1000 | 1.079 | 74.17 | **2.41x** | 2.63x | PASS |
| **V15 (8T+unroll)** | CPU 8T | 10M | 1000 | 0.967 | 82.75 | **2.69x** | 2.94x | PASS |

**GROUP BY 演进分析**:
```
V1 (0.91x) → V4 SIMD (0.95x) → V4 parallel (2.21x) → V15 (2.69x)
                 +4%               +2.3x                +22%
```

---

### 2.3 INNER JOIN 算子

**等效 SQL**: `SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V1 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 100K | 1M | 1.933 | 2.28 | 1.00x | - | PASS |
| **V1 (base)** | CPU | 100K | 1M | 4.924 | 0.89 | 0.39x | 1.00x | PASS |
| **V2 (Robin Hood)** | CPU | 100K | 1M | 16.082 | 0.27 | 0.12x | 0.30x | PASS |
| **V3 (SOA+radix)** | CPU SIMD | 100K | 1M | 1.213 | 3.63 | **1.59x** | 4.05x | PASS |
| **V6 (prefetch)** | CPU SIMD | 100K | 1M | 1.402 | 3.14 | 1.37x | 3.51x | PASS |
| **V10 (full semantic)** | CPU SIMD | 100K | 1M | 1.285 | 3.42 | **1.50x** | 3.83x | PASS |
| **V10.1 (zero-copy)** | CPU SIMD | 100K | 1M | 4.802 | 0.92 | 0.40x | 1.02x | PASS |
| **V10.2 (single-hash)** | CPU SIMD | 100K | 1M | 4.730 | 0.93 | 0.40x | 1.04x | PASS |
| **V11 (SIMD probe)** | CPU SIMD | 100K | 1M | 3.319 | 1.33 | 0.58x | 1.48x | PASS |
| **V13 (two-phase)** | CPU SIMD | 100K | 1M | 18.903 | 0.23 | 0.10x | 0.26x | PASS |
| **V14 (pre-alloc)** | CPU SIMD | 100K | 1M | 1.182 | 3.72 | **1.63x** | 4.16x | PASS |

**INNER JOIN 演进分析**:
```
V1 (0.39x) → V3 (1.59x) → V10 (1.50x) → V14 (1.63x)
                +4.1x        -6%          +8.7%
```

**失败的优化尝试**:
- V2 Robin Hood: 哈希冲突处理开销太大
- V10.1/V10.2: 过度优化反而影响性能
- V11 SIMD probe: SIMD 探测收益不及开销
- V13 two-phase: 两阶段算法增加了遍历次数

---

### 2.4 SEMI JOIN 算子

**等效 SQL**: `SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t ...)`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V10 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|--------|--------|
| DuckDB | CPU | 100K | 1M | 3.250 | 1.35 | 1.00x | - | PASS |
| **V10 (CPU SIMD)** | CPU SIMD | 100K | 1M | 3.626 | 1.21 | 0.89x | 1.00x | PASS |
| **GPU (Metal)** | Metal | 100K | 1M | 1.313 | 3.35 | **2.47x** | 2.76x | PASS |

**SEMI JOIN 演进**:
```
V10 CPU (0.89x) → GPU Metal (2.47x)
                     +2.77x
```

---

### 2.5 TopK 算子

**等效 SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT 10`

| 版本 | 设备 | 数据量 | K | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V1 | 正确性 |
|------|------|--------|---|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 10 | 2.510 | 15.94 | 1.00x | - | PASS |
| **V1 (base)** | CPU | 10M | 10 | 4.577 | 8.74 | 0.54x | 1.00x | PASS |
| **V2 (heap)** | CPU | 10M | 10 | 4.527 | 8.84 | 0.55x | 1.01x | PASS |
| **V3 (adaptive)** | CPU | 10M | 10 | 4.520 | 8.85 | 0.55x | 1.01x | PASS |
| **V4 (sample)** | CPU SIMD | 10M | 10 | 0.532 | 75.15 | **4.71x** | 8.59x | PASS |
| **V5 (count-based)** | CPU | 10M | 10 | 0.615 | 65.06 | **4.08x** | 7.44x | PASS |
| **V13 (GPU select)** | Metal | 10M | 10 | 7.028 | 5.69 | 0.35x | 0.65x | PASS |

**TopK 演进分析**:
```
V1 (0.54x) → V2/V3 (0.55x) → V4 sample (4.71x) → V5 count (4.08x)
                                 +8.6x             优化策略不同
```

---

## 三、算子版本总览

### 3.1 各算子最佳版本

| 算子 | 推荐版本 | 设备 | 性能 | 适用场景 |
|------|----------|------|------|----------|
| Filter | V4 (GPU AUTO) | Metal | 1.10x | 大数据量 (>1M) |
| Filter | V6 (prefetch) | CPU SIMD | 0.69x | 小数据量 |
| GROUP BY | V15 (8T+unroll) | CPU 8T | 2.69x | 通用场景 |
| GROUP BY | V6 (smart) | Auto | 2.23x | 自动选择 |
| INNER JOIN | V14 (pre-alloc) | CPU SIMD | 1.63x | 通用场景 |
| SEMI JOIN | GPU (Metal) | Metal | 2.47x | 大数据量 (>500K) |
| SEMI JOIN | V10 (CPU) | CPU SIMD | 0.89x | 小数据量 |
| TopK | V4 (sample) | CPU SIMD | 4.71x | K 较小 |
| TopK | V5 (count-based) | CPU | 4.08x | 通用场景 |

### 3.2 版本功能对照表

| 版本 | Filter | GROUP BY | INNER JOIN | SEMI JOIN | TopK |
|------|--------|----------|------------|-----------|------|
| V1 | 基础 | 基础 | 基础 | - | 基础 |
| V2 | bitmap | - | Robin Hood | - | heap |
| V3 | template | - | SOA+radix | - | adaptive |
| V4 | GPU | SIMD/parallel | - | - | sample |
| V5 | LUT | GPU 2-phase | - | - | count |
| V6 | prefetch | smart | prefetch | - | UMA |
| V10 | - | - | full semantic | CPU SIMD | - |
| V10.1/2 | - | - | zero-copy | - | - |
| V11 | - | - | SIMD probe | - | - |
| V12.1 | - | GPU warp | - | - | - |
| V13 | - | GPU no-atomic | two-phase | - | GPU |
| V14 | - | 8T parallel | pre-alloc | - | - |
| V15 | direct | 8T+unroll | - | - | - |
| GPU | Metal | Metal | - | Metal | Metal |

---

## 四、优化建议

### 需要优化的版本 (vs DuckDB < 1.0x)

| 算子 | 版本 | 性能 | 问题分析 |
|------|------|------|----------|
| Filter | V1-V6, V15 | 0.14x-0.82x | CPU SIMD 无法超越 DuckDB 向量化 |
| Filter | Parallel 4T | 0.44x | 线程开销超过收益 |
| GROUP BY | V1, V4 SIMD | 0.91x-0.95x | 单线程瓶颈 |
| GROUP BY | V13 GPU | 0.26x | GPU 原子操作开销 |
| INNER JOIN | V2 Robin Hood | 0.12x | 算法不适合该场景 |
| INNER JOIN | V10.1/V10.2 | 0.40x | 过度优化 |
| INNER JOIN | V11 SIMD | 0.58x | SIMD 探测开销 |
| INNER JOIN | V13 two-phase | 0.10x | 两次遍历开销 |
| SEMI JOIN | V10 CPU | 0.89x | 需要 GPU 加速 |
| TopK | V1-V3 | 0.54x-0.55x | 需要采样优化 |
| TopK | V13 GPU | 0.35x | GPU 传输开销 |

### 表现优异的版本 (vs DuckDB >= 1.5x)

| 算子 | 版本 | 性能 | 成功因素 |
|------|------|------|----------|
| GROUP BY | V4 parallel | 2.21x | 4 线程并行 |
| GROUP BY | V6 smart | 2.23x | 智能策略选择 |
| GROUP BY | V14 8T | 2.41x | 8 线程并行 |
| GROUP BY | V15 8T+unroll | 2.69x | 循环展开优化 |
| INNER JOIN | V3 SOA | 1.59x | SOA 布局优化 |
| INNER JOIN | V14 pre-alloc | 1.63x | 预分配消除 realloc |
| SEMI JOIN | GPU Metal | 2.47x | GPU 并行探测 |
| TopK | V4 sample | 4.71x | 采样快速筛选 |
| TopK | V5 count | 4.08x | 计数排序优化 |

---

## 五、版本历史

| 版本 | 日期 | 主要优化 | 关键算子性能 |
|------|------|----------|-------------|
| V1 | - | 基础实现 | 基准版本 |
| V3 | - | SIMD + SOA | Join 1.6x |
| V4 | - | GPU Filter + 并行 GROUP BY | GROUP BY 2.2x |
| V5 | - | 采样 TopK | TopK 4.7x |
| V6 | - | 预取 + 智能选择 | 自动策略 |
| V10 | - | 完整语义 | SEMI/ANTI Join |
| V13 | - | 两阶段算法 | 尝试失败 |
| V14 | 2026-01-27 | GPU SEMI + 预分配 | SEMI 2.5x, Join 1.6x |
| V15 | 2026-01-27 | 8T + 循环展开 | GROUP BY 2.7x |

---

## 六、下一步优化方向

1. **Filter**: 探索更高效的 GPU kernel，或尝试 8 线程 CPU 并行
2. **GROUP BY**: V15 已接近理论极限 (82.75 GB/s vs 内存带宽 ~100 GB/s)
3. **INNER JOIN**: 尝试 GPU 加速，参考 SEMI JOIN 的成功经验
4. **SEMI JOIN**: V10 CPU 版本需要优化，可能需要更好的哈希表设计
5. **TopK**: V4/V5 已非常优秀，考虑特殊 K 值优化

---

*Generated by ThunderDuck V14 Benchmark Suite*
