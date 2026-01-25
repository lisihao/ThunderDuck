# ThunderDuck 文档索引

> **版本**: 2.0.0 | **更新日期**: 2026-01-24

---

## 一、项目概述

ThunderDuck 是针对 Apple Silicon M4 优化的高性能 OLAP 算子库，通过 SIMD 向量化和缓存优化实现对 DuckDB 的显著性能提升。

### 核心指标

| 指标 | 数值 |
|------|------|
| 测试胜率 | **95.7%** (22/23) |
| 平均加速比 | **1152.90x** |
| 支持算子 | Filter, Aggregate, Sort, TopK, Join |

---

## 二、架构设计文档

| 文档 | 描述 | 状态 |
|------|------|------|
| [THUNDERDUCK_DESIGN.md](THUNDERDUCK_DESIGN.md) | 整体架构设计 | 完成 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 详细架构说明 | 完成 |
| [REQUIREMENTS.md](REQUIREMENTS.md) | 需求规格说明 | 完成 |

---

## 三、性能优化设计

### 3.1 Filter 算子

| 文档 | 版本 | 核心优化 |
|------|------|----------|
| [FILTER_COUNT_OPTIMIZATION_DESIGN.md](FILTER_COUNT_OPTIMIZATION_DESIGN.md) | v3.0 | SIMD 批量处理、L2 流水线预取 |

### 3.2 TopK 算子

| 文档 | 版本 | 核心优化 |
|------|------|----------|
| [TOPK_OPTIMIZATION_DESIGN.md](TOPK_OPTIMIZATION_DESIGN.md) | v3.0 | 自适应策略选择 |
| [TOPK_V4_OPTIMIZATION_DESIGN.md](TOPK_V4_OPTIMIZATION_DESIGN.md) | v4.0 | 采样预过滤 + SIMD 批量跳过 |
| [TOPK_V5_RESEARCH.md](TOPK_V5_RESEARCH.md) | v5.0 | 低基数优化研究 |
| [TOPK_CARDINALITY_ANALYSIS.md](TOPK_CARDINALITY_ANALYSIS.md) | - | 基数敏感性分析 |

### 3.3 Join 算子

| 文档 | 版本 | 核心优化 |
|------|------|----------|
| [JOIN_V3_OPTIMIZATION_DESIGN.md](JOIN_V3_OPTIMIZATION_DESIGN.md) | v3.0 | Robin Hood 哈希 + SIMD 探测 |

### 3.4 内存优化

| 文档 | 描述 |
|------|------|
| [MEMORY_OPTIMIZATION_DESIGN.md](MEMORY_OPTIMIZATION_DESIGN.md) | 128 字节缓存行对齐、内存池设计 |

### 3.5 综合优化

| 文档 | 描述 |
|------|------|
| [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) | 优化计划总览 |
| [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) | 性能优化策略 |

---

## 四、测试设计文档

| 文档 | 描述 |
|------|------|
| [TESTING.md](TESTING.md) | 测试策略与用例设计 |

---

## 五、性能测试报告

| 文档 | 描述 | 日期 |
|------|------|------|
| [BENCHMARK_REPORT_COMPREHENSIVE.md](BENCHMARK_REPORT_COMPREHENSIVE.md) | 全面性能对比报告 | 2026-01-24 |
| [DETAILED_BENCHMARK_REPORT.md](DETAILED_BENCHMARK_REPORT.md) | 详细基准测试报告 | 2026-01-24 |
| [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) | 性能分析报告 | 2026-01-24 |

---

## 六、算子版本演进

### Filter 算子

| 版本 | 特性 | 加速比 |
|------|------|--------|
| v1.0 | 基础 SIMD | 3-5x |
| v2.0 | 循环展开 | 5-8x |
| v3.0 | L2 预取 + 批量处理 | 6-42x |

### TopK 算子

| 版本 | 特性 | 适用场景 |
|------|------|----------|
| v3.0 | 自适应策略 | 通用 |
| v4.0 | 采样预过滤 | 高基数 (>=500) |
| v5.0 | 研究版本 | 等同 v4 |

### Join 算子

| 版本 | 特性 | 加速比 |
|------|------|--------|
| v1.0 | 基础哈希 | 2-3x |
| v2.0 | 开放寻址 | 3-5x |
| v3.0 | Robin Hood + SIMD | 1.1-13x |

---

## 七、关键发现

### 7.1 TopK 基数敏感性

| 基数范围 | ThunderDuck vs DuckDB | 建议 |
|----------|----------------------|------|
| < 500 | 0.1-0.9x | 使用 DuckDB |
| >= 500 | 1.5-4.8x | 使用 ThunderDuck |

### 7.2 最佳应用场景

**推荐 ThunderDuck:**
- 用户ID、订单ID 等高基数 TopK
- 时间戳、价格等连续值 TopK
- 大规模 Filter/Aggregate 操作
- Apple Silicon 平台

**推荐 DuckDB:**
- 低基数 TopK (状态码、类别等)
- 通用 SQL 查询
- 跨平台部署

---

## 八、目录结构

```
docs/
├── INDEX.md                           # 本文档
├── ARCHITECTURE.md                    # 架构设计
├── THUNDERDUCK_DESIGN.md              # 整体设计
├── REQUIREMENTS.md                    # 需求规格
│
├── FILTER_COUNT_OPTIMIZATION_DESIGN.md  # Filter v3 优化
├── TOPK_OPTIMIZATION_DESIGN.md          # TopK v3 优化
├── TOPK_V4_OPTIMIZATION_DESIGN.md       # TopK v4 优化
├── TOPK_V5_RESEARCH.md                  # TopK v5 研究
├── TOPK_CARDINALITY_ANALYSIS.md         # 基数分析
├── JOIN_V3_OPTIMIZATION_DESIGN.md       # Join v3 优化
├── MEMORY_OPTIMIZATION_DESIGN.md        # 内存优化
│
├── OPTIMIZATION_PLAN.md               # 优化计划
├── PERFORMANCE_OPTIMIZATION.md        # 性能优化
├── PERFORMANCE_ANALYSIS.md            # 性能分析
│
├── TESTING.md                         # 测试设计
├── BENCHMARK_REPORT_COMPREHENSIVE.md  # 综合测试报告
└── DETAILED_BENCHMARK_REPORT.md       # 详细测试报告
```

---

*ThunderDuck - Apple Silicon 上的极致性能*
