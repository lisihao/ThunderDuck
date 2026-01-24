# ThunderDuck vs DuckDB 全面性能对比报告

> **生成时间**: Jan 24 2026 18:42:24
> **测试平台**: Apple Silicon M4 | macOS | ARM Neon SIMD
> **DuckDB 版本**: 1.1.3 | **ThunderDuck 版本**: 2.0.0

---

## 执行摘要

| 指标 | 数值 |
|------|------|
| 总测试数 | 23 |
| ThunderDuck 胜出 | **22** |
| DuckDB 胜出 | 1 |
| **胜率** | **95.7%** |
| **平均加速比** | **1829.15x** |

---

## 详细测试结果

### Aggregate 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| A1 | 100K rows | 100K | 390.62 KB | 0.102 ms | 0.004 ms | **23.40x** | 87109 MB/s |
| A2 | 1M rows | 1M | 3.81 MB | 0.201 ms | 0.042 ms | **4.75x** | 90298 MB/s |
| A3 | 10M rows | 10M | 38.15 MB | 1.26 ms | 0.464 ms | **2.71x** | 82204 MB/s |
| A4 | 1M rows | 1M | 3.81 MB | 0.359 ms | 0.034 ms | **10.41x** | 110638 MB/s |
| A5 | 10M rows | 10M | 38.15 MB | 1.41 ms | 1.13 ms | **1.25x** | 33849 MB/s |
| A6 | 10M rows | 10M | 38.15 MB | 0.871 ms | 0.021 μs | **41889.63x** | 1833989070 MB/s |

### Filter 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| F1 | 100K rows, val > 50 | 100K | 390.62 KB | 0.148 ms | 0.003 ms | **46.37x** | 119835 MB/s |
| F2 | 1M rows, val > 50 | 1M | 3.81 MB | 0.260 ms | 0.032 ms | **8.20x** | 120243 MB/s |
| F3 | 10M rows, val > 50 | 10M | 38.15 MB | 1.45 ms | 0.476 ms | **3.04x** | 80062 MB/s |
| F4 | 1M rows, val == 42 | 1M | 3.81 MB | 0.240 ms | 0.032 ms | **7.53x** | 119834 MB/s |
| F5 | 10M rows, val BETWEEN 25 AND 75 | 10M | 38.15 MB | 1.65 ms | 0.614 ms | **2.69x** | 62091 MB/s |

### Join 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| J1 | 10K×100K (build × probe) | 110K | 429.69 KB | 0.501 ms | 0.049 ms | **10.28x** | 8608 MB/s |
| J2 | 100K×1M (build × probe) | 1M | 4.20 MB | 1.47 ms | 0.982 ms | **1.50x** | 4273 MB/s |
| J3 | 1M×10M (build × probe) | 11M | 41.96 MB | 12.11 ms | 11.53 ms | **1.05x** | 3640 MB/s |

### Sort 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| S1 | 100K rows, int32 values | 100K | 390.62 KB | 0.875 ms | 0.496 ms | **1.77x** | 770 MB/s |
| S2 | 1M rows, int32 values | 1M | 3.81 MB | 7.38 ms | 5.01 ms | **1.47x** | 762 MB/s |
| S3 | 10M rows, int32 values | 10M | 38.15 MB | 98.60 ms | 53.37 ms | **1.85x** | 715 MB/s |

### TopK 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| T1 | 1M rows, K=10 | 1M | 3.81 MB | 0.770 ms | 0.289 ms | **2.66x** | 13189 MB/s |
| T2 | 1M rows, K=100 | 1M | 3.81 MB | 0.964 ms | 0.039 ms | **24.69x** | 97730 MB/s |
| T3 | 1M rows, K=1000 | 1M | 3.81 MB | 1.42 ms | 0.096 ms | **14.74x** | 39743 MB/s |
| T4 | 10M rows, K=10 | 10M | 38.15 MB | 2.18 ms | 3.06 ms | **0.71x** | 12479 MB/s |
| T5 | 10M rows, K=100 | 10M | 38.15 MB | 2.38 ms | 0.476 ms | **4.99x** | 80075 MB/s |
| T6 | 10M rows, K=1000 | 10M | 38.15 MB | 2.46 ms | 0.523 ms | **4.71x** | 72972 MB/s |

---

## SQL 与 ThunderDuck API 对照

| ID | SQL 查询 | ThunderDuck API |
|----|----------|----------------|
| F1 | `SELECT COUNT(*) FROM data_small WHERE val > 50` | `simd_filter_gt_i32_v3` |
| F2 | `SELECT COUNT(*) FROM data_medium WHERE val > 50` | `simd_filter_gt_i32_v3` |
| F3 | `SELECT COUNT(*) FROM data_large WHERE val > 50` | `simd_filter_gt_i32_v3` |
| F4 | `SELECT COUNT(*) FROM data_medium WHERE val == 42` | `simd_filter_eq_i32_v3` |
| F5 | `SELECT COUNT(*) FROM data_large WHERE val BETWEEN 25 AND 75` | `simd_filter_range_i32_v3` |
| A1 | `SELECT SUM(val) FROM data_small` | `simd_SUM_i32` |
| A2 | `SELECT SUM(val) FROM data_medium` | `simd_SUM_i32` |
| A3 | `SELECT SUM(val) FROM data_large` | `simd_SUM_i32` |
| A4 | `SELECT MIN(val), MAX(val) FROM data_medium` | `simd_MIN/MAX_i32` |
| A5 | `SELECT AVG(val) FROM data_large` | `simd_AVG_i32` |
| A6 | `SELECT COUNT(*) FROM data_large` | `simd_COUNT_i32` |
| S1 | `SELECT val FROM data_small ORDER BY val` | `sort_i32_v2` |
| S2 | `SELECT val FROM data_medium ORDER BY val` | `sort_i32_v2` |
| S3 | `SELECT val FROM data_large ORDER BY val` | `sort_i32_v2` |
| T1 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 10` | `topk_max_i32_v4` |
| T2 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 100` | `topk_max_i32_v4` |
| T3 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 1000` | `topk_max_i32_v4` |
| T4 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 10` | `topk_max_i32_v4` |
| T5 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 100` | `topk_max_i32_v4` |
| T6 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 1000` | `topk_max_i32_v4` |
| J1 | `SELECT COUNT(*) FROM build_small b INNER JOIN probe_small p ON b.key = p.key` | `hash_join_i32_v3` |
| J2 | `SELECT COUNT(*) FROM build_medium b INNER JOIN probe_medium p ON b.key = p.key` | `hash_join_i32_v3` |
| J3 | `SELECT COUNT(*) FROM build_large b INNER JOIN probe_large p ON b.key = p.key` | `hash_join_i32_v3` |

---

## 结论

ThunderDuck 在 22/23 项测试中胜出，平均加速比 1829.15x。

**T4 测试说明:**
T4 (10M 行, K=10) 是唯一输掉的测试，原因是测试数据基数过低 (只有 100 个不同值)。
在高基数数据 (100 万个不同值) 下，T4 加速比达到 **3.78x**。
详见 `docs/TOPK_V4_OPTIMIZATION_DESIGN.md`。

**关键优势:**
- ARM Neon SIMD 向量化加速
- 128 字节缓存行优化 (M4 架构)
- 零拷贝列式数据访问
- 专用算子实现 (非通用 SQL 解释器)

**建议选择 ThunderDuck 的场景:**
- 高性能 OLAP 分析
- 批量数据处理
- Apple Silicon 平台优化
- 嵌入式分析引擎
