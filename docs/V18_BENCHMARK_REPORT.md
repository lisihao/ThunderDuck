# ThunderDuck V18 性能基线 V2 基准报告

> **测试日期**: Jan 27 2026 21:00
> **平台**: Apple M4 Max (10 核心)
> **版本**: V18 = 最优算子组合 + 智能策略选择
> **测试配置**: iterations=15, warmup=2, IQR 剔除异常值

---

## 一、执行摘要

V18 整合了所有性能最优的算子版本，并添加智能策略选择机制。

### 最佳性能摘要

| 算子 | V18 选用 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|----------|------|----------|-----------|------|
| **Filter** | V4 GPU AUTO | Metal | 3.435 | **0.82x** | 需优化 |
| **GROUP BY SUM** | V15 8T+unroll | CPU 8T | 0.969 | **3.08x** | 优秀 |
| **INNER JOIN** | V14 pre-alloc | CPU SIMD | 2.764 | **0.96x** | 接近 |
| **SEMI JOIN** | GPU Metal | Metal | 1.410 | **2.95x** | 优秀 |
| **TopK** | V4 sample | CPU SIMD | 0.551 | **4.95x** | 优秀 |

---

## 二、V18 算子组合

| 算子 | 选用版本 | 策略 | 设备选择条件 |
|------|----------|------|--------------|
| Filter | V4 GPU / V15 CPU | 智能选择 | 数据量 >= 1M 用 GPU |
| GROUP BY | V15 8T+unroll | 8 线程并行 + 循环展开 | 始终 8T |
| INNER JOIN | V14 pre-alloc | 预分配消除 realloc | CPU SIMD |
| SEMI JOIN | GPU / V10 | 智能选择 | probe >= 500K 用 GPU |
| TopK | V4 sample | 采样 + SIMD 筛选 | CPU SIMD |

---

## 三、详细测试结果

### 3.1 Filter 算子

**等效 SQL**: `SELECT rowid FROM t WHERE value > 500000`

| 版本 | 设备 | 数据量 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 2.824 | 14.16 | 1.00x | - | PASS |
| V3 (template) | CPU SIMD | 10M | 4.059 | 9.85 | 0.69x | 1.00x | PASS |
| **V18 (V4 GPU)** | **Metal** | **10M** | **3.435** | **11.64** | **0.82x** | **1.18x** | PASS |

**分析**: GPU Filter 比 CPU V3 快 18%，但仍落后 DuckDB 18%。需要进一步优化 GPU kernel。

---

### 3.2 GROUP BY SUM 算子

**等效 SQL**: `SELECT group_id, SUM(value) FROM t GROUP BY group_id`

| 版本 | 设备 | 数据量 | 分组数 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V4 | 正确性 |
|------|------|--------|--------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 1000 | 2.992 | 26.73 | 1.00x | - | PASS |
| V4 (SIMD single) | CPU SIMD | 10M | 1000 | 2.812 | 28.45 | 1.06x | 1.00x | PASS |
| **V18 (V15 8T)** | **CPU 8T** | **10M** | **1000** | **0.969** | **82.54** | **3.08x** | **2.90x** | PASS |

**分析**: V18 达到 **82.54 GB/s** 带宽，接近 M4 Max 理论内存带宽，是 DuckDB 的 **3.08 倍**！

---

### 3.3 INNER JOIN 算子

**等效 SQL**: `SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 100K | 1M | 2.665 | 1.65 | 1.00x | - | PASS |
| V3 (SOA+radix) | CPU SIMD | 100K | 1M | 2.716 | 1.62 | 0.98x | 1.00x | PASS |
| **V18 (V14)** | **CPU SIMD** | **100K** | **1M** | **2.764** | **1.59** | **0.96x** | **0.98x** | PASS |

**分析**: INNER JOIN 与 DuckDB 接近 (0.96x)。建议尝试 GPU 加速。

---

### 3.4 SEMI JOIN 算子

**等效 SQL**: `SELECT * FROM probe_t WHERE EXISTS (...)`

| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V10 | 正确性 |
|------|------|-------|-------|----------|------------|-----------|--------|--------|
| DuckDB | CPU | 100K | 1M | 4.165 | 1.06 | 1.00x | - | PASS |
| V10 (CPU SIMD) | CPU SIMD | 100K | 1M | 4.022 | 1.09 | 1.03x | 1.00x | PASS |
| **V18 (GPU)** | **Metal** | **100K** | **1M** | **1.410** | **3.12** | **2.95x** | **2.85x** | PASS |

**分析**: GPU SEMI JOIN 达到 **2.95x** vs DuckDB，是 V18 的亮点功能！

---

### 3.5 TopK 算子

**等效 SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT 10`

| 版本 | 设备 | 数据量 | K | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |
|------|------|--------|---|----------|------------|-----------|-------|--------|
| DuckDB | CPU | 10M | 10 | 2.729 | 14.66 | 1.00x | - | PASS |
| V3 (adaptive) | CPU | 10M | 10 | 4.806 | 8.32 | 0.56x | 1.00x | PASS |
| **V18 (V4)** | **CPU SIMD** | **10M** | **10** | **0.551** | **72.59** | **4.95x** | **8.72x** | PASS |

**分析**: TopK 采样算法达到 **4.95x** vs DuckDB，是 V3 的 **8.72 倍**！

---

## 四、版本对比总览

| 算子 | DuckDB | V3 | V18 | V18 vs V3 |
|------|--------|-----|-----|-----------|
| Filter (10M) | 2.824ms | 4.059ms | 3.435ms | +18% |
| GROUP BY (10M) | 2.992ms | 2.812ms | 0.969ms | **+190%** |
| INNER JOIN (1M) | 2.665ms | 2.716ms | 2.764ms | -2% |
| SEMI JOIN (1M) | 4.165ms | 4.022ms | 1.410ms | **+185%** |
| TopK (10M) | 2.729ms | 4.806ms | 0.551ms | **+772%** |

---

## 五、全算子版本性能汇总表

### Filter 算子 (10M 数据)

| 版本 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|------|----------|-----------|------|
| DuckDB | CPU | 2.824 | 1.00x | 基准 |
| V3 (template) | CPU SIMD | 4.059 | 0.69x | 需优化 |
| **V18 (V4 GPU)** | **Metal** | **3.435** | **0.82x** | 需优化 |

### GROUP BY SUM 算子 (10M 数据)

| 版本 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|------|----------|-----------|------|
| DuckDB | CPU | 2.992 | 1.00x | 基准 |
| V4 (SIMD single) | CPU SIMD | 2.812 | 1.06x | 良好 |
| **V18 (V15 8T)** | **CPU 8T** | **0.969** | **3.08x** | 优秀 |

### INNER JOIN 算子 (100K x 1M)

| 版本 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|------|----------|-----------|------|
| DuckDB | CPU | 2.665 | 1.00x | 基准 |
| V3 (SOA+radix) | CPU SIMD | 2.716 | 0.98x | 接近 |
| **V18 (V14)** | **CPU SIMD** | **2.764** | **0.96x** | 接近 |

### SEMI JOIN 算子 (100K x 1M)

| 版本 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|------|----------|-----------|------|
| DuckDB | CPU | 4.165 | 1.00x | 基准 |
| V10 (CPU SIMD) | CPU SIMD | 4.022 | 1.03x | 良好 |
| **V18 (GPU)** | **Metal** | **1.410** | **2.95x** | 优秀 |

### TopK 算子 (10M 数据)

| 版本 | 设备 | 时间(ms) | vs DuckDB | 状态 |
|------|------|----------|-----------|------|
| DuckDB | CPU | 2.729 | 1.00x | 基准 |
| V3 (adaptive) | CPU | 4.806 | 0.56x | 需优化 |
| **V18 (V4)** | **CPU SIMD** | **0.551** | **4.95x** | 优秀 |

---

## 六、优化建议

### 已达标算子 (vs DuckDB >= 1.0x)

| 算子 | 性能 | 带宽利用率 |
|------|------|------------|
| GROUP BY V18 | 3.08x | 82.54 GB/s |
| SEMI JOIN V18 | 2.95x | 3.12 GB/s |
| TopK V18 | 4.95x | 72.59 GB/s |
| V4 SIMD single | 1.06x | 28.45 GB/s |
| V10 CPU SIMD | 1.03x | 1.09 GB/s |

### 需要优化算子 (vs DuckDB < 1.0x)

| 算子 | 性能 | 优化方向 |
|------|------|----------|
| Filter V18 | 0.82x | 优化 GPU kernel 或尝试 8T CPU 并行 |
| INNER JOIN V18 | 0.96x | 尝试 GPU 加速 (参考 SEMI JOIN) |

---

## 七、下一步优化方向

1. **Filter CPU 8T**: 尝试 8 线程并行 + SIMD，目标 1.5x+
2. **INNER JOIN GPU**: 参考 SEMI JOIN 的成功经验尝试 GPU 加速
3. **GROUP BY**: 已接近理论极限，可探索更多分组数场景
4. **更大数据量测试**: 测试 100M、1B 数据量下的表现

---

## 八、版本历史

| 版本 | 日期 | 主要内容 | 关键性能 |
|------|------|----------|----------|
| V3 | - | 基础 SIMD | 基准版本 |
| V10 | - | 完整语义 | SEMI/ANTI Join |
| V14 | 2026-01-27 | 新性能基线 | GPU SEMI 2.5x |
| **V18** | **2026-01-27** | **性能基线 V2** | **GROUP BY 3.1x, TopK 5.0x** |

---

*Generated by ThunderDuck V18 Benchmark Suite*
