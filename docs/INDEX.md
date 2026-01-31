# ThunderDuck 文档索引

> **版本**: V69 | **更新日期**: 2026-01-31 | **性能**: 3.18x (TPC-H SF=1)

---

## 项目概述

ThunderDuck 是针对 Apple Silicon M4 优化的高性能 SQL 算子后端，通过 ARM Neon SIMD、Metal GPU 和 UMA 统一内存架构实现对 DuckDB 的显著性能提升。

### 核心指标

| 指标 | 数值 |
|------|------|
| TPC-H 查询通过率 | **100%** (22/22) |
| 几何平均加速比 | **3.18x** |
| 最高加速比 | **23.49x** (Q21) |
| 算子版本 | V24-V69 (70+) |
| GPU 算子 | V66, V68, V69 |

---

## 核心文档

| 文档 | 描述 | 状态 |
|------|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 系统架构设计 | 最新 |
| [ALGORITHMS.md](ALGORITHMS.md) | 核心算法详解 | 最新 |
| [REQUIREMENTS.md](REQUIREMENTS.md) | 需求规格说明 | 最新 |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | 性能基准报告 | 最新 |
| [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) | 竞争力分析 | 最新 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   ThunderDuck V69                    │
├─────────────────────────────────────────────────────┤
│  TPC-H Benchmark  │  Query Optimizer  │  Executor   │
├───────────────────┴───────────────────┴─────────────┤
│                   Operator Layer                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Filter  │ │  Join   │ │Aggregate│ │  TopK   │   │
│  │ V19.1   │ │ V19.2   │ │  V15    │ │  V4     │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
├─────────────────────────────────────────────────────┤
│                    GPU Layer (Metal)                 │
│  ┌─────────────────┐ ┌─────────────────────────┐   │
│  │ V66 FusedFilter │ │ V68 FusedQ3 │ V69 GroupAgg│ │
│  └─────────────────┘ └─────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│                   Hardware Layer                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ARM Neon  │ │ Metal GPU│ │ UMA (400+ GB/s)  │    │
│  │  SIMD    │ │  Compute │ │  Zero-Copy       │    │
│  └──────────┘ └──────────┘ └──────────────────┘    │
├─────────────────────────────────────────────────────┤
│                   Apple M4 Silicon                   │
└─────────────────────────────────────────────────────┘
```

---

## 性能摘要 (TPC-H SF=1)

```
几何平均加速比: 3.18x
查询通过率: 22/22 (100%)

Top 5 性能:
├── Q21: 23.49x (Bitmap EXISTS/NOT EXISTS)
├── Q22:  9.04x (Bitmap Anti-Join)
├── Q1:   7.42x (直接数组聚合)
├── Q19:  5.24x (SIMD 多条件过滤)
└── Q14:  4.54x (CASE WHEN 融合)
```

---

## 核心技术

### Apple Silicon 专用优化

| 技术 | 描述 | 性能提升 |
|------|------|----------|
| ARM Neon SIMD | 4 累加器并行，消除依赖链 | 3-5x |
| CRC32 硬件哈希 | 单周期 hash 计算 | 3-5x (Join) |
| 128 字节对齐 | M4 缓存行感知 | 10-20% |
| UMA 零拷贝 | GPU 直接访问 CPU 内存 | 2-3x (GPU) |

### GPU 加速 (Metal)

| 算子 | 版本 | 技术 | 适用查询 |
|------|------|------|----------|
| FusedFilterAggregate | V66 | Filter + Aggregate 融合 | Q6 |
| FusedQ3 | V68 | 多表 Join + Aggregate 融合 | Q3 |
| GPUGroupByAggregate | V69 | Block-local hash + 两级归约 | (框架) |

### 专用算子

| 算子 | 技术 | 加速比 |
|------|------|--------|
| Bitmap Anti-Join | 位图替代 Hash 表 | 9x |
| 直接数组聚合 | L1 常驻，零 hash | 7x |
| 完美哈希 Join | key_range <= 2*count | 6.88x |
| Bloom 预过滤 | 减少无效 hash 查找 | 20-40% |

---

## 算子版本演进

### Filter 算子

| 版本 | 特性 | 加速比 |
|------|------|--------|
| V1 | 基础 SIMD | 3-5x |
| V3 | 4 累加器并行 | 5-8x |
| V19.1 | 无分支 + 预取 | 6-10x |

### Join 算子

| 版本 | 特性 | 加速比 |
|------|------|--------|
| V3 | Robin Hood Hash | 2-3x |
| V4 | 完美哈希 | 6.88x |
| V19.2 | SIMD 批量探测 | 3-5x |
| GPU | Metal 并行 | 4.6x |

### Aggregate 算子

| 版本 | 特性 | 加速比 |
|------|------|--------|
| V2 | SIMD + 预取 | 2-3x |
| V15 | 8 路并行 | 3-4x |
| V69 | GPU 分组聚合 | (框架) |

### TopK 算子

| 版本 | 特性 | 适用场景 |
|------|------|----------|
| V3 | 自适应策略 | 通用 |
| V4 | 采样预过滤 | 高基数 |
| V5 | 计数排序 | 低基数 |

---

## 开发指南

### 构建

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

### 运行基准测试

```bash
# 完整测试
./benchmarks/tpch_benchmark --sf 1 --iterations 10

# 单个查询
./benchmarks/tpch_benchmark --query Q1 --iterations 5

# 显示系统表
./benchmarks/tpch_benchmark --show-catalog
```

---

## 历史文档

### 架构设计

| 文档 | 描述 |
|------|------|
| [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) | 早期架构设计 |
| [THUNDERDUCK_DESIGN.md](THUNDERDUCK_DESIGN.md) | 整体设计 |
| [UMA_ARCHITECTURE_ANALYSIS.md](UMA_ARCHITECTURE_ANALYSIS.md) | UMA 架构分析 |

### 算子优化

| 文档 | 描述 |
|------|------|
| [FILTER_COUNT_OPTIMIZATION_DESIGN.md](FILTER_COUNT_OPTIMIZATION_DESIGN.md) | Filter 优化 |
| [TOPK_V4_OPTIMIZATION_DESIGN.md](TOPK_V4_OPTIMIZATION_DESIGN.md) | TopK V4 优化 |
| [JOIN_V3_OPTIMIZATION_DESIGN.md](JOIN_V3_OPTIMIZATION_DESIGN.md) | Join V3 优化 |
| [GPU_ACCELERATION_DEEP_ANALYSIS.md](GPU_ACCELERATION_DEEP_ANALYSIS.md) | GPU 加速分析 |

### 性能报告

| 文档 | 描述 |
|------|------|
| [V14_BENCHMARK_REPORT.md](V14_BENCHMARK_REPORT.md) | V14 基准报告 |
| [V25_COMPREHENSIVE_BENCHMARK_REPORT.md](V25_COMPREHENSIVE_BENCHMARK_REPORT.md) | V25 综合报告 |
| [TPCH_PERFORMANCE_ANALYSIS.md](TPCH_PERFORMANCE_ANALYSIS.md) | TPC-H 性能分析 |

---

*ThunderDuck - Apple Silicon 上的极致性能*
