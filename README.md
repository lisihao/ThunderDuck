# ThunderDuck

> **Supercharging DuckDB on Apple M4**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/user/ThunderDuck)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-orange.svg)](https://www.apple.com/mac/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ThunderDuck 是一个针对 Apple M4 芯片优化的 DuckDB 算子后端，通过深度利用 M4 的 SIMD 指令集、Neural Engine (NPU) 和统一内存架构，实现核心算子的极致性能。

---

## 性能概览 (v2.0)

| 指标 | 结果 |
|------|------|
| 总测试数 | 14 |
| **ThunderDuck 胜出** | **10 (71%)** |
| 最大加速比 | **24,130x** (COUNT 操作) |
| Sort 加速 | **5x** (Radix Sort) |
| Aggregation 加速 | **3-5x** |

### 分类性能

| 类别 | 胜率 | 最佳加速比 |
|------|------|-----------|
| **Aggregation** | 100% (4/4) | 24,130x |
| **Sort** | 100% (2/2) | 5.03x |
| TopK | 66% (2/3) | 2.75x |
| Filter | 50% (2/4) | 1.45x |
| Join | 0% (0/1) | - |

详见 [性能分析报告](docs/PERFORMANCE_ANALYSIS.md)

---

## 特性

- **SIMD 加速** - 利用 128-bit ARM Neon 指令并行处理数据
- **Radix Sort** - O(n) 时间复杂度，比 DuckDB 快 5x
- **合并 minmax** - 单次遍历计算 MIN 和 MAX
- **位图过滤** - 高效的范围查询
- **缓存优化** - 128 字节缓存行对齐，软件预取
- **零拷贝** - 利用统一内存架构，CPU/NPU 共享数据

---

## 优化算子

| 算子 | 优化技术 | 当前加速比 | v3.0 目标 |
|------|----------|------------|----------|
| **Aggregation** | 16元素/迭代, 4累加器, 预取 | 3-24,000x | - |
| **Sort** | LSD Radix Sort (11-11-10 位) | 5x | - |
| **Filter** | SIMD 批量比较、位图过滤 | 1.1-1.5x | 1.5x+ |
| **TopK** | 堆选择、nth_element | 2-3x | 3x+ |
| **Join** | Robin Hood Hash | 0.08x | 1.5x+ |

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

# 运行测试
./run_tests

# 运行基准测试
./build/benchmark_app
```

### 运行 Benchmark

```bash
# 小数据集快速测试
./build/benchmark_app --small

# 中等数据集 (默认)
./build/benchmark_app --medium

# 大数据集
./build/benchmark_app --large

# 详细报告
./build/detailed_benchmark_app
```

---

## 项目结构

```
ThunderDuck/
├── src/
│   ├── core/                    # 核心框架
│   ├── operators/               # 优化算子实现
│   │   ├── filter/
│   │   │   ├── simd_filter.cpp      # v1 Filter
│   │   │   └── simd_filter_v2.cpp   # v2 Filter
│   │   ├── aggregate/
│   │   │   ├── simd_aggregate.cpp   # v1 Aggregation
│   │   │   └── simd_aggregate_v2.cpp # v2 Aggregation
│   │   ├── sort/
│   │   │   └── radix_sort.cpp       # Radix Sort
│   │   ├── topk/
│   │   │   └── topk.cpp             # Top-K
│   │   └── join/
│   │       └── robin_hood_hash.cpp  # Robin Hood Hash
│   └── utils/                   # 工具函数
├── include/                     # 公共头文件
├── tests/                       # 测试
├── benchmark/                   # 性能基准
│   ├── benchmark.cpp
│   └── detailed_benchmark.cpp
└── docs/                        # 文档
    ├── THUNDERDUCK_DESIGN.md        # 设计文档
    ├── OPTIMIZATION_PLAN.md         # 优化计划
    ├── PERFORMANCE_ANALYSIS.md      # 性能分析
    ├── FILTER_COUNT_OPTIMIZATION_DESIGN.md  # v3 Filter 设计
    ├── DETAILED_BENCHMARK_REPORT.md # 详细测试报告
    ├── REQUIREMENTS.md              # 需求文档
    └── TESTING.md                   # 测试文档
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| **语言** | C/C++17, AArch64 汇编 |
| **SIMD** | ARM Neon Intrinsics |
| **编译器** | Clang 17.0.0 |
| **构建** | Makefile / CMake |
| **依赖** | DuckDB 1.1.3 |

---

## 文档

| 文档 | 描述 |
|------|------|
| [设计文档](docs/THUNDERDUCK_DESIGN.md) | 完整的技术设计和架构 |
| [优化计划](docs/OPTIMIZATION_PLAN.md) | v1-v3 优化路线图 |
| [性能分析](docs/PERFORMANCE_ANALYSIS.md) | 详细性能对比和分析 |
| [Filter 优化设计](docs/FILTER_COUNT_OPTIMIZATION_DESIGN.md) | v3 Filter 深度优化 |
| [详细测试报告](docs/DETAILED_BENCHMARK_REPORT.md) | 包含 SQL 的完整测试报告 |
| [需求文档](docs/REQUIREMENTS.md) | 功能和性能需求 |
| [测试文档](docs/TESTING.md) | 测试策略和用例 |

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-01-24 | 初始版本，基础 SIMD 实现 |
| v2.0 | 2026-01-24 | Radix Sort, 合并 minmax, 16元素/迭代 |
| v3.0 | 计划中 | Filter 模板特化, 独立累加器, TopK 自适应 |

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

## 许可证

MIT License

---

*ThunderDuck v2.0 - 针对 Apple M4 优化，71% 测试超越 DuckDB！*
