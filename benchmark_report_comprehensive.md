# ThunderDuck vs DuckDB 全面性能对比报告

> **生成时间**: Jan 24 2026 18:16:17
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
| **平均加速比** | **1727.36x** |

---

## 详细测试结果

### Aggregate 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| A1 | 100K rows | 100K | 390.62 KB | 0.102 ms | 0.005 ms | **20.60x** | 76742 MB/s |
| A2 | 1M rows | 1M | 3.81 MB | 0.243 ms | 0.045 ms | **5.44x** | 85547 MB/s |
| A3 | 10M rows | 10M | 38.15 MB | 1.56 ms | 0.573 ms | **2.73x** | 66604 MB/s |
| A4 | 1M rows | 1M | 3.81 MB | 0.356 ms | 0.039 ms | **9.20x** | 98624 MB/s |
| A5 | 10M rows | 10M | 38.15 MB | 1.42 ms | 1.13 ms | **1.26x** | 33615 MB/s |
| A6 | 10M rows | 10M | 38.15 MB | 0.823 ms | 0.021 μs | **39563.70x** | 1833989070 MB/s |

### Filter 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| F1 | 100K rows, val > 50 | 100K | 390.62 KB | 0.136 ms | 0.003 ms | **39.89x** | 112197 MB/s |
| F2 | 1M rows, val > 50 | 1M | 3.81 MB | 0.256 ms | 0.032 ms | **8.06x** | 119975 MB/s |
| F3 | 10M rows, val > 50 | 10M | 38.15 MB | 1.56 ms | 0.517 ms | **3.01x** | 73811 MB/s |
| F4 | 1M rows, val == 42 | 1M | 3.81 MB | 0.297 ms | 0.046 ms | **6.52x** | 83747 MB/s |
| F5 | 10M rows, val BETWEEN 25 AND 75 | 10M | 38.15 MB | 2.63 ms | 0.783 ms | **3.36x** | 48734 MB/s |

### Join 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| J1 | 10K×100K (build × probe) | 110K | 429.69 KB | 0.485 ms | 0.055 ms | **8.88x** | 7679 MB/s |
| J2 | 100K×1M (build × probe) | 1M | 4.20 MB | 1.58 ms | 0.992 ms | **1.59x** | 4231 MB/s |
| J3 | 1M×10M (build × probe) | 11M | 41.96 MB | 13.18 ms | 11.51 ms | **1.15x** | 3645 MB/s |

### Sort 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| S1 | 100K rows, int32 values | 100K | 390.62 KB | 0.985 ms | 0.532 ms | **1.85x** | 717 MB/s |
| S2 | 1M rows, int32 values | 1M | 3.81 MB | 8.09 ms | 5.37 ms | **1.51x** | 710 MB/s |
| S3 | 10M rows, int32 values | 10M | 38.15 MB | 103.29 ms | 55.11 ms | **1.87x** | 692 MB/s |

### TopK 算子

| ID | 描述 | 数据量 | 数据大小 | DuckDB | ThunderDuck | 加速比 | 吞吐量 |
|-------|------|--------|----------|--------|-------------|--------|--------|
| T1 | 1M rows, K=10 | 1M | 3.81 MB | 0.929 ms | 0.473 ms | **1.96x** | 8057 MB/s |
| T2 | 1M rows, K=100 | 1M | 3.81 MB | 0.928 ms | 0.038 ms | **24.45x** | 100552 MB/s |
| T3 | 1M rows, K=1000 | 1M | 3.81 MB | 1.29 ms | 0.103 ms | **12.50x** | 36884 MB/s |
| T4 | 10M rows, K=10 | 10M | 38.15 MB | 2.02 ms | 4.87 ms | **0.41x** | 7835 MB/s |
| T5 | 10M rows, K=100 | 10M | 38.15 MB | 2.47 ms | 0.530 ms | **4.65x** | 71931 MB/s |
| T6 | 10M rows, K=1000 | 10M | 38.15 MB | 2.53 ms | 0.540 ms | **4.70x** | 70697 MB/s |

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
| T1 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 10` | `topk_max_i32_v3` |
| T2 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 100` | `topk_max_i32_v3` |
| T3 | `SELECT val FROM data_medium ORDER BY val DESC LIMIT 1000` | `topk_max_i32_v3` |
| T4 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 10` | `topk_max_i32_v3` |
| T5 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 100` | `topk_max_i32_v3` |
| T6 | `SELECT val FROM data_large ORDER BY val DESC LIMIT 1000` | `topk_max_i32_v3` |
| J1 | `SELECT COUNT(*) FROM build_small b INNER JOIN probe_small p ON b.key = p.key` | `hash_join_i32_v3` |
| J2 | `SELECT COUNT(*) FROM build_medium b INNER JOIN probe_medium p ON b.key = p.key` | `hash_join_i32_v3` |
| J3 | `SELECT COUNT(*) FROM build_large b INNER JOIN probe_large p ON b.key = p.key` | `hash_join_i32_v3` |

---

## 结论

ThunderDuck 在 22/23 项测试中胜出，平均加速比 1727.36x。

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
