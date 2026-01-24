# ThunderDuck - 设计文档

> **版本**: 1.0.0 | **日期**: 2026-01-24

---

## 一、项目概述

ThunderDuck 项目旨在在 **Apple MacBook M4 芯片**上最大化 DuckDB 数据库的执行性能。

### 1.1 背景

DuckDB 作为一款嵌入式分析型数据库，默认采用通用的矢量化执行而避免使用特定硬件指令，以确保可移植性。这虽然带来了良好的跨平台支持，但也意味着未充分利用 M4 芯片的特殊硬件能力：

| 硬件特性 | 说明 |
|----------|------|
| **SIMD 矢量指令集** | 128-bit 宽度的 ARM Neon 指令 |
| **Neural Engine (NPU)** | M4 达到 38 TOPS（万亿次运算/秒） |
| **统一内存架构 (UMA)** | CPU/GPU/NPU 共享内存池 |
| **128 字节缓存行** | M4 特有的缓存行大小 |

### 1.2 目标

基于 DuckDB 最新主分支代码，替换其部分算子的后端实现：

- **Filter（筛选）** - WHERE 条件过滤
- **Aggregation（聚合）** - SUM/AVG/MIN/MAX/GROUP BY
- **Join（连接）** - Hash Join / Sort-Merge Join
- **Sorting（排序）** - ORDER BY 排序

### 1.3 技术路线

```
C/C++ 开发
    ↓
ARM Neon Intrinsics (SIMD)
    ↓
AArch64 汇编 (热点优化)
    ↓
Core ML / Metal (NPU 加速)
```

---

## 二、设计需求

### 2.1 筛选算子（Filter）

**优化目标**：加速 WHERE 条件过滤操作，减少判断和分支开销。

#### M4 特性映射

| 技术 | 应用方式 |
|------|----------|
| **128-bit SIMD (Neon)** | 一次比较多个数据值，批量过滤 |
| **位掩码操作** | SIMD 并行应用过滤谓词，结果存入位掩码 |
| **无分支判断** | 使用 SIMD 掩码压缩指令，避免条件跳转 |
| **缓存对齐** | 数据对齐到 128 字节边界 |

#### 实现策略

```cpp
// 示例：SIMD 批量过滤
// 一次处理 4 个 int32 值（128-bit = 4 × 32-bit）
int32x4_t data = vld1q_s32(input_ptr);
int32x4_t threshold = vdupq_n_s32(filter_value);
uint32x4_t mask = vcgtq_s32(data, threshold);  // 比较生成掩码
// 使用掩码压缩收集满足条件的元组
```

#### 优化要点

1. 对被过滤列的数据预对齐到 128 字节边界
2. 使用向量化按位操作组合多个布尔条件
3. 内置掩码压缩指令快速收集满足条件的索引
4. 循环展开减少分支跳转

---

### 2.2 聚合算子（Aggregation）

**优化目标**：加速聚合计算（SUM、AVG、MIN/MAX、GROUP BY），提升吞吐量。

#### 简单聚合（无分组）

| 操作 | SIMD 优化方式 |
|------|---------------|
| SUM | 4 路并行累加，最后归约 |
| MIN/MAX | 并行比较，水平归约 |
| AVG | 并行求和 + 计数 |

```cpp
// 示例：SIMD 并行求和
float32x4_t sum_vec = vdupq_n_f32(0.0f);
for (size_t i = 0; i < n; i += 4) {
    float32x4_t data = vld1q_f32(input + i);
    sum_vec = vaddq_f32(sum_vec, data);
}
float sum = vaddvq_f32(sum_vec);  // 水平归约
```

#### 分组聚合 + NPU 加速

**创新构想**：将大规模聚合视为张量运算，利用 NPU 的大规模并行计算能力。

| 转换思路 | 说明 |
|----------|------|
| GROUP BY → 稀疏矩阵乘法 | 分组键映射为稀疏矩阵行 |
| SUM → 向量归约 | 每组求和转为张量 reduce 操作 |
| Multi-group → Batch 处理 | 多组并行作为 batch 维度 |

**NPU 接入方式**：
- 使用 Core ML 或 Metal Performance Shaders
- 数据存储在统一内存区域，CPU/NPU 直接共享
- CPU 预处理 → NPU 计算 → CPU 后处理

**精度保证**：
- 浮点累加需保证顺序不敏感性
- 向量化算法结果与标量实现一致

---

### 2.3 连接算子（Join）

**优化目标**：加速哈希连接、排序合并连接，减少计算和内存访问瓶颈。

#### Hash Join 优化

**构建阶段（Build Phase）**：

```cpp
// SIMD 并行计算多个键的哈希值
// 使用 ARM CRC32 指令加速
uint32x4_t keys = vld1q_u32(key_ptr);
uint32x4_t hashes = vcrc32q_u32(seeds, keys);  // 伪代码
uint32x4_t buckets = vandq_u32(hashes, bucket_mask);
```

**探测阶段（Probe Phase）**：
- 批量加载候选键进行 SIMD 比较
- 并行匹配多个候选

#### 内存访问优化

| 策略 | 说明 |
|------|------|
| **分区预处理** | 对连接输入进行分区，使访问更顺序 |
| **缓存行对齐** | 哈希表桶和链表节点对齐到 128 字节 |
| **预取指令** | 使用 `__builtin_prefetch` 提前加载 |
| **批处理** | 一次处理多个待连接键 |

#### NPU 策略

- NPU 对连接算子的直接加速**暂不考虑**
- 原因：连接涉及大量指针操作和条件逻辑，不易映射为 NN 计算
- 架构保持弹性，未来可评估 GPU 加速

---

### 2.4 排序算子（Sorting）

**优化目标**：加速大规模数据集排序，降低比较和数据移动成本。

#### SIMD 加速策略

| 算法 | 说明 |
|------|------|
| **Bitonic Sort** | 矢量友好的排序网络 |
| **Odd-Even Merge Sort** | 并行比较交换 |
| **向量内排序** | 寄存器内元素排序 |

```cpp
// 示例：Bitonic 比较交换
int32x4_t a = vld1q_s32(ptr);
int32x4_t b = vld1q_s32(ptr + 4);
int32x4_t min_val = vminq_s32(a, b);
int32x4_t max_val = vmaxq_s32(a, b);
```

#### 分块排序策略

```
1. 数据分割为 L1/L2 缓存友好的小块
2. SIMD 对每块内部排序
3. 多路合并得到全局有序结果
4. 配合多核并行排序多个分区
```

#### NPU 探索方向

- Top-K 选择可能转化为 NN 筛选问题
- 常规全排序仍以 CPU + SIMD 为主

---

## 三、架构设计

### 3.1 DuckDB 集成机制

ThunderDuck 作为 DuckDB 执行引擎的**可选组件**：

```
┌─────────────────────────────────────────────────────────┐
│                      DuckDB Core                        │
├─────────────────────────────────────────────────────────┤
│                   Physical Plan                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Filter  │ │  Agg    │ │  Join   │ │  Sort   │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │           │             │
│       ▼           ▼           ▼           ▼             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           ThunderDuck Operator Backend          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │   │
│  │  │ SIMD Ops │ │ NPU Ops  │ │ ASM Ops  │        │   │
│  │  └──────────┘ └──────────┘ └──────────┘        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### 替换机制

1. **运行时检测**：检测是否为 Apple M4 环境
2. **函数指针替换**：物理计划生成阶段替换算子运算函数
3. **接口遵循**：遵循 DuckDB 的 `DataChunk` 和 `Vector` 接口规范
4. **回退机制**：非 M4 平台或优化禁用时，自动回退原生实现

#### 集成方式选项

| 方式 | 优点 | 缺点 |
|------|------|------|
| **Extension 插件** | 动态加载，低侵入 | 受限于扩展 API |
| **编译时替换** | 完全控制，性能最优 | 需维护 fork 分支 |
| **Hybrid** | 平衡灵活性和性能 | 复杂度较高 |

**推荐**：优先尝试 Extension 机制，必要时考虑编译时替换。

### 3.2 代码组织

```
ThunderDuck/
├── src/
│   ├── core/                    # 核心框架
│   │   ├── thunder_engine.cpp   # 主引擎入口
│   │   ├── operator_registry.cpp # 算子注册表
│   │   └── memory_allocator.cpp # 对齐内存分配器
│   │
│   ├── operators/               # 优化算子实现
│   │   ├── filter/
│   │   │   ├── simd_filter.cpp
│   │   │   └── filter_kernels.S  # ASM 热点
│   │   ├── aggregate/
│   │   │   ├── simd_aggregate.cpp
│   │   │   └── npu_aggregate.cpp
│   │   ├── join/
│   │   │   ├── simd_hash_join.cpp
│   │   │   └── hash_kernels.S
│   │   └── sort/
│   │       ├── simd_sort.cpp
│   │       └── bitonic_kernels.S
│   │
│   ├── npu/                     # NPU 加速层
│   │   ├── npu_backend.cpp      # NPU 接口封装
│   │   ├── tensor_converter.cpp # 数据 → 张量转换
│   │   └── models/              # Core ML 模型
│   │
│   └── utils/                   # 工具函数
│       ├── simd_utils.h         # Neon intrinsics 封装
│       ├── cache_utils.h        # 缓存/对齐工具
│       └── platform_detect.cpp  # 平台检测
│
├── include/                     # 公共头文件
│   └── thunderduck/
│       ├── thunderduck.h
│       └── operators.h
│
├── tests/                       # 测试
│   ├── test_filter.cpp
│   ├── test_aggregate.cpp
│   ├── test_join.cpp
│   └── test_sort.cpp
│
├── benchmarks/                  # 性能基准测试
│   ├── bench_filter.cpp
│   ├── bench_aggregate.cpp
│   └── tpc_h/                   # TPC-H 基准
│
└── docs/                        # 文档
    └── THUNDERDUCK_DESIGN.md
```

### 3.3 统一内存架构运用

#### 设计原则：**"数据一次到位"**

```
┌─────────────────────────────────────────────────────────┐
│                   Unified Memory Pool                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Aligned Data Buffer                │    │
│  │         (128-byte cache line aligned)           │    │
│  └─────────────────────────────────────────────────┘    │
│         ▲              ▲              ▲                 │
│         │              │              │                 │
│      ┌──┴──┐       ┌──┴──┐       ┌──┴──┐               │
│      │ CPU │       │ GPU │       │ NPU │               │
│      └─────┘       └─────┘       └─────┘               │
└─────────────────────────────────────────────────────────┘
```

#### 实现策略

1. **内存分配器**：自定义分配器，确保 128 字节对齐
2. **零拷贝传递**：CPU 写入后直接交由 NPU 访问
3. **内存屏障**：CPU 写入完成后执行必要的 cache flush
4. **共享容器**：统一管理 DataChunk 内存分配

```cpp
// 对齐内存分配示例
void* aligned_alloc_128(size_t size) {
    void* ptr;
    posix_memalign(&ptr, 128, size);  // 128 字节对齐
    return ptr;
}
```

---

## 四、实现计划

### Phase 1: 基础框架（2 周）

- [ ] 搭建项目结构和构建系统（CMake）
- [ ] 实现平台检测和 M4 特性探测
- [ ] 设计对齐内存分配器
- [ ] 建立 DuckDB 集成接口原型
- [ ] 编写基础测试框架

### Phase 2: Filter 算子（2 周）

- [ ] 实现 SIMD 比较内核（Neon intrinsics）
- [ ] 实现位掩码压缩和索引收集
- [ ] 热点循环的 ASM 优化
- [ ] 缓存对齐优化
- [ ] 性能基准测试

### Phase 3: Aggregation 算子（3 周）

- [ ] 实现 SIMD 简单聚合（SUM/MIN/MAX/AVG）
- [ ] 实现分组聚合的 SIMD 加速
- [ ] NPU 加速原型（Core ML 集成）
- [ ] 精度验证测试
- [ ] 性能基准测试

### Phase 4: Join 算子（3 周）

- [ ] 实现 SIMD 哈希计算
- [ ] 实现 SIMD 键比较
- [ ] 缓存友好的哈希表设计
- [ ] 分区和预取优化
- [ ] 性能基准测试

### Phase 5: Sort 算子（2 周）

- [ ] 实现 Bitonic Sort SIMD 内核
- [ ] 实现分块排序 + 多路合并
- [ ] 多线程并行排序
- [ ] 性能基准测试

### Phase 6: 集成与优化（2 周）

- [ ] DuckDB Extension 封装
- [ ] 端到端测试（TPC-H）
- [ ] 性能调优和回归测试
- [ ] 文档完善

---

## 五、性能目标

| 算子 | 基准（DuckDB 原生） | 目标加速比 |
|------|---------------------|------------|
| Filter | 1x | 2-4x |
| Aggregation (simple) | 1x | 2-3x |
| Aggregation (group by) | 1x | 1.5-3x (with NPU) |
| Hash Join | 1x | 1.5-2x |
| Sort | 1x | 1.5-2.5x |

**评估方法**：
- TPC-H 基准测试（SF=1, SF=10, SF=100）
- 单算子微基准测试
- 真实工作负载测试

---

## 六、风险与挑战

| 风险 | 缓解措施 |
|------|----------|
| NPU 加速收益不明确 | 先做 SIMD 优化，NPU 作为探索性功能 |
| DuckDB 版本升级兼容 | 模块化设计，最小化侵入 |
| ASM 代码维护成本 | 封装为独立函数，保持接口稳定 |
| 精度问题 | 严格的正确性测试，与标量实现对比 |

---

## 七、参考资料

- [DuckDB 源码](https://github.com/duckdb/duckdb)
- [ARM Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Apple M4 Chip Architecture](https://developer.apple.com/documentation/)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

---

> **维护者**: ThunderDuck Team  
> **最后更新**: 2026-01-24
