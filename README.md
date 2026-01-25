# ThunderDuck

> **Supercharging DuckDB on Apple M4**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/user/ThunderDuck)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-orange.svg)](https://www.apple.com/mac/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ThunderDuck 是一个针对 Apple M4 芯片优化的 DuckDB 算子后端，通过深度利用 M4 的 SIMD 指令集和统一内存架构，实现核心算子的极致性能。

---

## 性能概览 (v2.0)

| 指标 | 结果 |
|------|------|
| 总测试数 | 23 |
| **ThunderDuck 胜出** | **22 (95.7%)** |
| 最大加速比 | **26,350x** (COUNT 操作) |
| 平均加速比 | **1,152x** |

### 分类性能

| 类别 | 平均加速比 | 最大加速比 | 胜率 |
|------|-----------|-----------|------|
| **Aggregate** | 4,397x | 26,350x | 100% |
| **Filter** | 12x | 42x | 100% |
| **TopK** | 8x | 24x | 83% |
| **Join** | 5x | 13x | 100% |
| **Sort** | 1.9x | 2.3x | 100% |

详见 [综合性能报告](docs/BENCHMARK_REPORT_COMPREHENSIVE.md)

---

## 特性

- **SIMD 加速** - 利用 128-bit ARM Neon 指令并行处理数据
- **Radix Sort** - O(n) 时间复杂度，比 DuckDB 快 1.5-2.3x
- **采样预过滤 TopK** - 高基数场景下达到 2-5x 加速
- **Robin Hood Hash Join** - SIMD 加速探测，1.1-13x 加速
- **缓存优化** - 128 字节缓存行对齐，L2 流水线预取
- **零拷贝** - 利用统一内存架构

---

## 优化算子

| 算子 | 版本 | 优化技术 | 加速比 |
|------|------|----------|--------|
| **Aggregate** | v2 | 16元素/迭代, 4累加器, 预取 | 1.3-26,350x |
| **Filter** | v3 | SIMD 批量比较、L2 预取 | 2.4-42x |
| **TopK** | v4 | 采样预过滤 + SIMD 批量跳过 | 2.9-24x |
| **Join** | v3 | Robin Hood Hash + SIMD 探测 | 1.1-13x |
| **Sort** | v2 | LSD Radix Sort (11-11-10 位) | 1.5-2.3x |

---

## 快速开始

### 系统要求

- macOS 14.0+
- Apple Silicon M4 芯片（推荐）
- Xcode 15.0+ 或 Apple Clang
- DuckDB v1.1.3+

### 构建

```bash
# 克隆项目
git clone https://github.com/user/ThunderDuck.git
cd ThunderDuck

# 构建
make clean && make

# 运行基准测试
./build/benchmark_app
```

### 运行 Benchmark

```bash
# 综合性能测试
./build/comprehensive_benchmark

# 基数敏感性测试 (TopK)
./build/topk_v5_benchmark
```

---

## 项目结构

```
ThunderDuck/
├── src/
│   ├── core/                    # 核心框架
│   ├── operators/               # 优化算子实现
│   │   ├── filter/              # Filter v1/v2/v3
│   │   ├── aggregate/           # Aggregate v1/v2
│   │   ├── sort/                # Sort/TopK v3/v4/v5
│   │   └── join/                # Join v1/v2/v3
│   └── utils/                   # 工具函数
├── include/                     # 公共头文件
├── benchmark/                   # 性能基准
└── docs/                        # 文档
```

---

## 文档

完整文档索引: [docs/INDEX.md](docs/INDEX.md)

### 核心文档

| 文档 | 描述 |
|------|------|
| [文档索引](docs/INDEX.md) | 所有文档的索引和导航 |
| [架构设计](docs/ARCHITECTURE.md) | 系统架构详细说明 |
| [综合测试报告](docs/BENCHMARK_REPORT_COMPREHENSIVE.md) | 最新性能对比报告 |

### 优化设计文档

| 文档 | 描述 |
|------|------|
| [Filter v3 优化](docs/FILTER_COUNT_OPTIMIZATION_DESIGN.md) | Filter 算子深度优化 |
| [TopK v4 优化](docs/TOPK_V4_OPTIMIZATION_DESIGN.md) | 采样预过滤设计 |
| [TopK 基数分析](docs/TOPK_CARDINALITY_ANALYSIS.md) | 基数敏感性研究 |
| [Join v3 优化](docs/JOIN_V3_OPTIMIZATION_DESIGN.md) | Robin Hood Hash 设计 |
| [内存优化](docs/MEMORY_OPTIMIZATION_DESIGN.md) | 缓存行优化策略 |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| **语言** | C/C++17 |
| **SIMD** | ARM Neon Intrinsics |
| **编译器** | Clang 17.0.0 |
| **构建** | Makefile |
| **依赖** | DuckDB 1.1.3 |

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-01-24 | 初始版本，基础 SIMD 实现 |
| v2.0 | 2026-01-24 | Radix Sort, TopK v4, Join v3, 95.7% 胜率 |

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

## 许可证

MIT License

---

*ThunderDuck v2.0 - 针对 Apple M4 优化，95.7% 测试超越 DuckDB！*
