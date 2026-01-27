# ThunderDuck V14 全面性能基准报告

> **测试日期**: 2026-01-27
> **平台**: Apple M4 Max (10核 CPU, 40核 GPU)
> **测试配置**: iterations=15, warmup=2, IQR异常值剔除
> **版本标签**: v14.0 - 新性能基线

## 一、执行摘要

本报告测试了 ThunderDuck V14 各算子版本与 DuckDB 的性能对比，覆盖 Filter、GROUP BY、JOIN、TopK 四大核心算子。

### 性能总览

| 算子 | 数据量 | 最佳版本 | 设备 | vs DuckDB | 状态 |
|------|--------|----------|------|-----------|------|
| **Filter** | 1M | V6 (prefetch) | CPU SIMD | 0.80x | **需优化** |
| **Filter** | 10M | V4 (AUTO) | GPU | 0.62x | **需优化** |
| **GROUP BY SUM** | 10M | V15 (8T+展开) | CPU 8T | **2.79x** | 优秀 |
| **INNER JOIN** | 1M probe | V10 | CPU SIMD | **1.52x** | 良好 |
| **SEMI JOIN** | 1M probe | GPU (Metal) | Metal | **2.28x** | 优秀 |
| **TopK** | 10M | V5 (count) | CPU | **4.65x** | 优秀 |

---

## 二、详细测试结果

### 2.1 Filter 算子

**等效 SQL**: `SELECT * FROM t WHERE value > 500000`
**选择率**: ~50%

#### 1M 数据量测试

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 0.304 | 13.18 | 1.00x | - | PASS |
| V3 (bitmap) | CPU SIMD | 0.409 | 9.77 | 0.74x | 1.00x | PASS |
| V5 (LUT) | CPU SIMD | 0.380 | 10.53 | **0.80x** | 1.07x | PASS |
| V6 (prefetch) | CPU SIMD | 0.380 | 10.54 | **0.80x** | 1.07x | PASS |
| V15 (direct) | CPU SIMD | 0.401 | 9.96 | 0.76x | 1.01x | PASS |
| V4 (AUTO) | CPU Auto | 0.385 | 10.39 | 0.79x | 1.06x | PASS |
| Parallel (4T) | CPU 4T | 1.007 | 3.97 | 0.30x | 0.40x | PASS |

#### 10M 数据量测试

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 1.391 | 28.76 | 1.00x | - | PASS |
| V3 (bitmap) | CPU SIMD | 3.923 | 10.20 | 0.35x | 1.00x | PASS |
| V5 (LUT) | CPU SIMD | 3.885 | 10.30 | 0.36x | 1.01x | PASS |
| V6 (prefetch) | CPU SIMD | 3.978 | 10.05 | 0.35x | 0.98x | PASS |
| V15 (direct) | CPU SIMD | 3.450 | 11.60 | 0.40x | 1.13x | PASS |
| **V4 (AUTO)** | **GPU** | **2.231** | **17.93** | **0.62x** | **1.75x** | PASS |
| Parallel (4T) | CPU 4T | 6.630 | 6.03 | 0.21x | 0.59x | PASS |

**分析**:
- DuckDB 在 Filter 上表现极强 (10M 达到 28.76 GB/s)
- 可能原因: DuckDB 使用向量化执行引擎 + 高效的选择向量
- ThunderDuck GPU 版本 (V4 AUTO) 相比 CPU 版本快 1.75x，但仍不及 DuckDB

---

### 2.2 GROUP BY SUM 算子

**等效 SQL**: `SELECT group_id, SUM(value) FROM t GROUP BY group_id`
**数据量**: 10M 行
**分组数**: 1000

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V4-single | 正确性 |
|------|------|----------|------------|-----------|--------------|--------|
| DuckDB | CPU | 2.675 | 29.91 | 1.00x | - | PASS |
| V4 (single) | CPU SIMD | 2.817 | 28.40 | 0.95x | 1.00x | PASS |
| V4 (parallel) | CPU 4T | 1.204 | 66.46 | **2.22x** | 2.34x | PASS |
| V6 (smart) | CPU/GPU Auto | 1.509 | 53.00 | 1.77x | 1.86x | PASS |
| V14 (parallel) | CPU 8T | 1.079 | 74.12 | **2.48x** | 2.61x | PASS |
| **V15 (8T+展开)** | **CPU 8T** | **0.958** | **83.50** | **2.79x** | **2.94x** | PASS |

**分析**:
- GROUP BY 是 ThunderDuck 的强项
- V15 达到 83.50 GB/s 带宽，相比 DuckDB 快 2.79x
- 多线程并行 (8T) 相比单线程快 2.94x

---

### 2.3 INNER JOIN 算子

**等效 SQL**: `SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key`
**Build**: 100K 唯一键
**Probe**: 1M 随机键 (~10% 匹配率)

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 1.953 | 2.25 | 1.00x | - | PASS |
| V3 | CPU SIMD | 1.374 | 3.20 | **1.42x** | 1.00x | PASS |
| V6 (prefetch) | CPU SIMD | 1.370 | 3.21 | **1.43x** | 1.00x | PASS |
| **V10** | **CPU SIMD** | **1.282** | **3.43** | **1.52x** | **1.07x** | PASS |
| V11 (SIMD probe) | CPU SIMD | 3.434 | 1.28 | 0.57x | 0.40x | PASS |
| V13 (两阶段) | CPU SIMD | 19.212 | 0.23 | 0.10x | 0.07x | PASS |
| V14 (预分配) | CPU SIMD | 1.338 | 3.29 | 1.46x | 1.02x | PASS |

**分析**:
- V10 是 INNER JOIN 最佳版本，达到 1.52x vs DuckDB
- V11 和 V13 有性能回退，需要进一步调查
- V3/V6/V10/V14 都超过 DuckDB

---

### 2.4 SEMI JOIN 算子

**等效 SQL**: `SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE build_t.key = probe_t.key)`
**Build**: 100K 唯一键
**Probe**: 1M 随机键

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V10 | 正确性 |
|------|------|----------|------------|-----------|--------|--------|
| DuckDB | CPU | 3.234 | 1.36 | 1.00x | - | PASS |
| V10 | CPU SIMD | 3.580 | 1.23 | 0.90x | 1.00x | PASS |
| **GPU (Metal)** | **Metal** | **1.421** | **3.10** | **2.28x** | **2.52x** | PASS |

**分析**:
- GPU SEMI Join 是本次优化的亮点
- Metal GPU 版本达到 2.28x vs DuckDB，2.52x vs V10 CPU
- CPU V10 稍慢于 DuckDB (0.90x)

---

### 2.5 TopK 算子

**等效 SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT 10`
**数据量**: 10M
**K**: 10

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 2.528 | 15.82 | 1.00x | - | PASS |
| V3 (adaptive) | CPU | 4.685 | 8.54 | 0.54x | 1.00x | PASS |
| V4 (sample) | CPU SIMD | 0.566 | 70.71 | **4.47x** | 8.28x | PASS |
| **V5 (count)** | **CPU** | **0.543** | **73.63** | **4.65x** | **8.62x** | PASS |
| V6 (UMA) | CPU/GPU Auto | 0.618 | 64.72 | 4.09x | 7.57x | PASS |

**分析**:
- TopK 是 ThunderDuck 最强算子
- V5 采样预过滤 + 计数方法达到 4.65x vs DuckDB
- 带宽达到 73.63 GB/s，接近内存带宽极限

---

## 三、综合性能矩阵

### 所有算子版本性能对比 (vs DuckDB)

| 算子 | V3 | V4 | V5 | V6 | V10 | V11 | V13 | V14 | V15 | GPU |
|------|-----|-----|-----|-----|------|------|------|------|------|-----|
| Filter 1M | 0.74x | 0.79x | 0.80x | 0.80x | - | - | - | - | 0.76x | - |
| Filter 10M | 0.35x | 0.62x | 0.36x | 0.35x | - | - | - | - | 0.40x | 0.62x |
| GROUP BY | - | 0.95x | - | 1.77x | - | - | - | 2.48x | **2.79x** | - |
| INNER JOIN | **1.42x** | - | - | **1.43x** | **1.52x** | 0.57x | 0.10x | **1.46x** | - | - |
| SEMI JOIN | - | - | - | - | 0.90x | - | - | - | - | **2.28x** |
| TopK | 0.54x | **4.47x** | **4.65x** | **4.09x** | - | - | - | - | - | - |

**图例**:
- **加粗** = 超过 DuckDB (> 1.0x)
- 普通 = 低于 DuckDB (< 1.0x)

---

## 四、优化建议

### 高优先级优化点

#### 1. Filter 算子 (急需优化)

**现状**: 所有版本都低于 DuckDB (最佳 0.80x)

**根因分析**:
- DuckDB 使用向量化执行引擎，选择向量避免了数据拷贝
- ThunderDuck 需要写出索引数组，有额外内存带宽开销
- DuckDB 可能使用了更高效的分支预测

**优化方向**:
1. **选择向量模式**: 返回位图而非索引数组
2. **批量写入优化**: 使用 NEON vst1q 批量存储
3. **GPU 优化**: 提升 GPU kernel 利用率
4. **预取距离调优**: 针对 M4 缓存特性调优

#### 2. SEMI JOIN CPU 版本 (V10)

**现状**: 0.90x vs DuckDB

**优化方向**:
1. 已有 GPU 版本 (2.28x)，考虑降低 GPU 启用阈值
2. CPU 版本尝试 Bloom Filter 预过滤

### 中优先级

#### 3. INNER JOIN V11/V13 性能回退

**现状**: V11 (0.57x), V13 (0.10x) 大幅低于 V3/V10

**根因分析**:
- V11 SIMD 并行槽位比较可能引入额外开销
- V13 两阶段算法可能在小数据集上开销过大

**建议**:
- 清理或禁用 V11/V13，保留 V10 作为默认版本
- 或添加数据量阈值选择策略

### 低优先级

#### 4. GROUP BY 进一步优化

**现状**: V15 已达到 2.79x，表现优异

**潜在优化**:
- GPU 加速 (适合超大数据集 > 100M)
- 低基数优化 (分组数 < 100)

---

## 五、版本演进历史

| 版本 | 发布日期 | 主要优化 | 关键性能提升 |
|------|----------|----------|-------------|
| V3 | - | 基础 SIMD | 基准版本 |
| V4 | - | GPU Filter | Filter GPU 1.75x vs CPU |
| V5 | - | TopK 采样 | TopK 4.65x vs DuckDB |
| V6 | - | 预取优化 | Join 1.43x vs DuckDB |
| V10 | - | 完整语义 | SEMI/ANTI Join 支持 |
| V14 | 2026-01-27 | GPU SEMI Join | **SEMI Join 2.28x vs DuckDB** |
| V15 | 2026-01-27 | 8T并行+展开 | **GROUP BY 2.79x vs DuckDB** |

---

## 六、结论

### 优势领域
1. **GROUP BY SUM**: V15 达到 2.79x vs DuckDB
2. **SEMI JOIN GPU**: 2.28x vs DuckDB
3. **TopK**: V5 达到 4.65x vs DuckDB
4. **INNER JOIN**: V10 达到 1.52x vs DuckDB

### 待改进领域
1. **Filter**: 所有版本都低于 DuckDB (最佳 0.80x)
2. **SEMI JOIN CPU**: 0.90x vs DuckDB
3. **JOIN V11/V13**: 性能回退严重

### 下一步计划
1. 优化 Filter 算子 (目标: 1.2x vs DuckDB)
2. 清理 JOIN V11/V13 冗余代码
3. 添加更多 GPU 加速算子
