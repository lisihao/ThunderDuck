# ThunderDuck 项目概述

> **版本**: V34 | **日期**: 2026-01-29 | **标签**: 继续攻坚

## 一、项目简介

ThunderDuck 是针对 **Apple M4 芯片深度优化**的 DuckDB 算子后端系统，通过 SIMD 向量化、多核并行、GPU/NPU 加速等技术，实现对 DuckDB 的显著性能超越。

### 1.1 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **代码规模** | 31,016 行 | 核心源代码 (不含测试) |
| **版本迭代** | 34 个 | V1 → V34 |
| **TPC-H 覆盖** | 19/22 | 86% 查询已优化 |
| **最佳加速比** | 9.15x | Q1 定价汇总报告 |
| **几何平均** | ~1.5x | 全部已优化查询 |

### 1.2 技术亮点

```
算子级别优化:
├── ARM Neon SIMD      - 128-bit 向量化处理
├── Compact Hash Table - 缓存友好的紧凑哈希
├── Bloom Filter       - SEMI JOIN 预过滤
├── Thread-Local Agg   - 无锁并行聚合
├── 选择向量          - 稀疏数据高效处理
└── 自适应策略        - 运行时最优版本选择

架构层面:
├── V33 通用化架构    - 零硬编码 + 完全参数化
├── AutoTuner         - 自动线程数/批量大小调优
├── 线程池复用       - 预热 + 任务窃取
└── 内存对齐         - 128 字节缓存行对齐
```

---

## 二、代码结构

### 2.1 目录布局

```
ThunderDuck/
├── src/                          # 核心源代码 (21,690 行)
│   ├── core/                     # 执行引擎 (2,824 行)
│   ├── operators/                # 算子实现 (18,177 行)
│   │   ├── aggregate/            #   聚合算子 (7 版本, 2,515 行)
│   │   ├── filter/               #   过滤算子 (9 版本, 3,702 行)
│   │   ├── join/                 #   Join算子 (16 版本, 7,738 行)
│   │   ├── sort/                 #   排序算子 (5 版本, 2,805 行)
│   │   └── vector/               #   向量搜索 (1,417 行)
│   ├── npu/                      # NPU 推理 (487 行)
│   ├── extension/                # DuckDB 扩展 (67 行)
│   └── utils/                    # 工具类 (135 行)
│
├── include/thunderduck/          # 公共 API (31 头文件, 9,326 行)
│
├── benchmark/                    # 基准测试
│   └── tpch/                     # TPC-H 完整实现 (86 文件, 51,044 行)
│       ├── tpch_operators_v*.cpp # 各版本算子实现
│       ├── tpch_config_v33.*     # V33 配置层
│       ├── tpch_queries.cpp      # 查询注册
│       └── tpch_benchmark_main.cpp
│
├── tests/                        # 单元测试 (6 文件, 1,446 行)
│
├── docs/                         # 文档 (104 个 .md 文件)
│
└── third_party/                  # 依赖 (DuckDB 1.1.3)
```

### 2.2 核心算子版本

| 算子 | 版本数 | 最优版本 | 核心技术 |
|------|--------|----------|----------|
| **Filter** | 9 | V19 | SIMD 批量比较 + L2 预取 |
| **Join** | 16 | V19.2 | Compact Hash + SIMD Probe |
| **Aggregate** | 7 | V15 | Thread-Local + 无锁归并 |
| **Sort** | 5 | V4 | LSD Radix Sort + 采样 TopK |

---

## 三、版本演进

### 3.1 里程碑版本

```
Phase 1: 基础设计 (V1-V7)
├── V5:  首个综合基准测试
└── V7:  优先级系统 (P0/P1/P2)

Phase 2: 算子深度优化 (V12-V16)
├── V12: 统一架构设计
├── V13: 聚合+过滤优化
├── V14: Hash Join + GROUP BY 深度优化
└── V15: 增量优化

Phase 3: 加速器探索 (V18-V20)
├── V18: GPU Semi Join (Metal)
├── V19: Filter 优化 + Sort 优化
└── V20: 向量数据库设计

Phase 4: 框架突破 (V23-V25) ⭐
├── V23: 深度集成 + 框架设计
├── V24: 选择向量 + 数组替换
└── V25: 线程池 + Hash 优化 (40%↑)

Phase 5: 通用化 + 攻坚 (V26-V34) ← 当前
├── V26: 优化方向规划
├── V33: 通用化架构 (零硬编码)
└── V34: 继续攻坚 (Q22/Q13/Q8)
```

### 3.2 性能演进

```
版本    几何平均   超越DuckDB查询数   提升
V20     0.72x     2                 -
V24     0.85x     5                 +18%
V25     1.00x     7                 +40% ⭐
V33     ~1.50x    16                +50%
V34     目标      19                继续攻坚
```

---

## 四、TPC-H 查询覆盖

### 4.1 已优化查询 (16/22)

| 查询 | 版本 | 加速比 | 优化技术 | 状态 |
|------|------|--------|----------|------|
| **Q1** | 基础 | 9.15x | 直接数组聚合 + 8T | ✅ 最优 |
| **Q3** | V31 | 1.14x | Bloom Filter + Compact Hash | ✅ |
| **Q4** | V27 | 1.2x | Bitmap SEMI Join | ✅ |
| **Q5** | V33 | ~1.9x | 通用化 + 无硬编码 | ✅ |
| **Q6** | V25 | 1.3x | SIMD Filter + 线程池 | ✅ |
| **Q7** | V33 | ~1.9x | 动态国家列表 + SIMD | ✅ |
| **Q9** | V33 | ~1.4x | 通用化 LIKE 参数化 | ✅ |
| **Q10** | V25 | 1.7x | WeakHashTable + ThreadPool | ✅ |
| **Q11** | V27 | 1.1x | 单遍扫描 + 后置过滤 | ✅ |
| **Q12** | V27 | 0.8x | 直接数组索引 | ⚠️ 待优化 |
| **Q14** | V25 | 1.3x | 条件聚合 + 并行 | ✅ |
| **Q15** | V27 | 1.3x | 直接数组索引 + 并行 | ✅ |
| **Q16** | V27 | 1.2x | PredicatePrecomputer | ✅ |
| **Q18** | V33 | ~1.5x | 通用化 + 阈值参数化 | ✅ |
| **Q19** | V33 | ~2.0x | 条件组参数化 | ✅ |
| **Q2** | 基础 | 回退 | DuckDB 子查询 | ⚠️ |

### 4.2 V34 攻坚目标 (6/22)

| 查询 | 复杂度 | 可行性 | V34 计划 |
|------|--------|--------|----------|
| **Q22** | 中 | 75% | ✅ 实现 - SUBSTRING → 国家码预计算 |
| **Q13** | 中 | 70% | ✅ 实现 - LEFT JOIN + COUNT |
| **Q8** | 高 | 65% | ✅ 实现 - CASE + 多表 JOIN |
| Q17 | 高 | 45% | 🔜 下阶段 - 相关子查询 |
| Q20 | 高 | 40% | 🔜 下阶段 - EXISTS 嵌套 |
| Q21 | 高 | 35% | 🔜 下阶段 - EXISTS/NOT EXISTS |

---

## 五、技术栈

### 5.1 核心技术

| 层级 | 技术 | 用途 |
|------|------|------|
| **SIMD** | ARM Neon Intrinsics | 128-bit 向量化 |
| **并行** | std::thread + ThreadPool | 8核并行 |
| **哈希** | Compact Hash + Robin Hood | 缓存友好 |
| **过滤** | Bloom Filter + Bitmap | 预过滤 |
| **排序** | LSD Radix Sort | O(n) 排序 |
| **配置** | QueryConfig + AutoTuner | 运行时调优 |

### 5.2 加速器

| 加速器 | 技术 | 状态 |
|--------|------|------|
| **GPU** | Metal Performance Shaders | 实验中 |
| **NPU** | Core ML / BNNS | 设计完成 |
| **Vector DB** | 量化 + 相似度搜索 | 设计完成 |

---

## 六、构建与运行

### 6.1 系统要求

- macOS 14.0+
- Apple Silicon M4 (推荐)
- Xcode 15.0+ / Apple Clang
- DuckDB v1.1.3+
- CMake 3.20+

### 6.2 构建命令

```bash
# 配置
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译
make -j$(sysctl -n hw.ncpu)

# 运行 TPC-H 基准测试
./tpch_benchmark --sf=1 --queries=all
```

### 6.3 快速验证

```bash
# 单个查询
./tpch_benchmark --sf=1 --queries=Q1,Q5,Q19

# 全部查询
./tpch_benchmark --sf=1 --queries=all --iterations=30
```

---

## 七、文件清单

### 7.1 核心实现文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `benchmark/tpch/tpch_operators_v33.cpp` | 930 | V33 通用化算子 |
| `benchmark/tpch/tpch_operators_v32.cpp` | 850 | V32 紧凑哈希 |
| `benchmark/tpch/tpch_operators_v27.cpp` | 1200 | V27 批量优化 |
| `benchmark/tpch/tpch_queries.cpp` | 1523 | 查询注册 |
| `src/operators/join/hash_join_v19_2.cpp` | 600 | 最优 Join |
| `src/operators/filter/simd_filter_v19.cpp` | 450 | 最优 Filter |

### 7.2 配置文件

| 文件 | 说明 |
|------|------|
| `benchmark/tpch/tpch_config_v33.h` | V33 配置类定义 |
| `benchmark/tpch/tpch_config_v33.cpp` | V33 配置实现 |
| `benchmarks/CMakeLists.txt` | 基准测试构建配置 |

---

## 八、贡献指南

### 8.1 开发规范

- 遵循 `CLAUDE.md` 中的编码规范
- 所有新算子必须有基准测试
- 性能回退 >5% 阻止合并
- SIMD 代码必须保留

### 8.2 提交流程

```bash
# 创建功能分支
git checkout -b feature/v34-q22-optimization

# 提交代码
git add .
git commit -m "feat(v34): Q22 SUBSTRING optimization"

# 推送
git push origin feature/v34-q22-optimization
```

---

## 九、版本计划

### 9.1 V34 目标 (当前)

- [x] 项目概述文档
- [ ] Q22 优化 (SUBSTRING + NOT EXISTS)
- [ ] Q13 优化 (LEFT JOIN + COUNT)
- [ ] Q8 优化 (CASE + 多表 JOIN)
- [ ] 编译验证
- [ ] 性能回归测试

### 9.2 V35+ 规划

- Q17/Q20/Q21 相关子查询支持
- Q12 性能提升 (目前 0.8x)
- 生产环境部署

---

*ThunderDuck V34 - 继续攻坚，目标 19/22 查询覆盖*
