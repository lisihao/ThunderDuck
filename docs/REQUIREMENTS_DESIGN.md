# ThunderDuck 需求设计文档

> **版本**: 2.0.0 | **更新日期**: 2026-01-24
> **项目状态**: 生产就绪

---

## 一、项目背景

### 1.1 问题陈述

现代数据库在 Apple Silicon 平台上未能充分利用硬件特性：

| 问题 | 影响 |
|------|------|
| 未使用 UMA | CPU-GPU 数据传输开销大 |
| 单一执行路径 | 无法根据数据特征优化 |
| 通用 SIMD | 未针对 ARM NEON 优化 |
| 忽略 NPU | 未利用 Neural Engine 加速 |

### 1.2 解决方案

ThunderDuck 通过以下方式解决上述问题：

1. **统一内存架构 (UMA)**: 零拷贝 CPU/GPU 数据共享
2. **自适应策略选择**: 根据数据特征自动选择最优执行器
3. **ARM NEON 优化**: 针对 M4 的 128-bit SIMD 优化
4. **多执行器支持**: CPU SIMD / GPU Metal / NPU BNNS

---

## 二、功能需求

### 2.1 核心算子

#### FR-001: Filter 算子

| 属性 | 描述 |
|------|------|
| **功能** | 根据条件过滤数据行 |
| **SQL 语义** | `SELECT * FROM t WHERE col > value` |
| **输入** | 列数据数组、比较操作、比较值 |
| **输出** | 满足条件的行索引数组 |
| **支持类型** | int32, int64, float, double |
| **比较操作** | EQ, NE, LT, LE, GT, GE, RANGE |

**验收标准**:
- [ ] 结果与 DuckDB 一致
- [ ] 吞吐量 > 2000 M rows/s (CPU)
- [ ] 10M+ 数据量时 GPU 加速 > 1.5x

#### FR-002: Aggregate 算子

| 属性 | 描述 |
|------|------|
| **功能** | 计算聚合统计量 |
| **SQL 语义** | `SELECT SUM(col), MIN(col), MAX(col), COUNT(*) FROM t` |
| **输入** | 列数据数组 |
| **输出** | 聚合结果 (sum, min, max, count) |
| **支持类型** | int32, int64, float, double |
| **聚合函数** | SUM, MIN, MAX, COUNT, AVG |

**验收标准**:
- [ ] 结果精度与 DuckDB 一致
- [ ] 带宽利用率 > 60 GB/s
- [ ] 融合 kernel 减少内存访问次数

#### FR-003: Hash Join 算子

| 属性 | 描述 |
|------|------|
| **功能** | 连接两个表 |
| **SQL 语义** | `SELECT * FROM t1 JOIN t2 ON t1.key = t2.key` |
| **输入** | Build 表键、Probe 表键 |
| **输出** | 匹配的行索引对 |
| **支持类型** | int32 键 |
| **Join 类型** | INNER (主要), LEFT, RIGHT, FULL |

**验收标准**:
- [ ] 结果与 DuckDB 一致
- [ ] 500K-10M probe 时 GPU 加速 > 2x
- [ ] 零拷贝 GPU 执行

#### FR-004: Sort 算子

| 属性 | 描述 |
|------|------|
| **功能** | 排序数据 |
| **SQL 语义** | `SELECT * FROM t ORDER BY col [ASC|DESC]` |
| **输入** | 列数据数组 |
| **输出** | 排序后的索引数组 |
| **支持类型** | int32, int64, float, double |
| **排序方向** | ASC, DESC |

**验收标准**:
- [ ] 结果与 std::sort 一致
- [ ] 使用基数排序优化整数排序
- [ ] 吞吐量 > DuckDB 5x

#### FR-005: TopK 算子

| 属性 | 描述 |
|------|------|
| **功能** | 获取前 K 个最大/最小值 |
| **SQL 语义** | `SELECT * FROM t ORDER BY col DESC LIMIT K` |
| **输入** | 列数据数组、K 值 |
| **输出** | 前 K 个元素的索引数组 |
| **支持类型** | int32, int64, float, double |
| **采样优化** | 支持采样预过滤 |

**验收标准**:
- [ ] 结果正确 (包含正确的 K 个元素)
- [ ] K < 1000 时采样方法高效
- [ ] 吞吐量 > DuckDB 10x

### 2.2 内存管理

#### FR-006: UMA 内存分配

| 属性 | 描述 |
|------|------|
| **功能** | 分配 CPU/GPU 共享内存 |
| **对齐** | 16KB 页对齐 |
| **零拷贝** | 支持 Metal Buffer 包装 |
| **池化** | 支持缓冲区复用 |

**验收标准**:
- [ ] 页对齐内存可直接创建 Metal Buffer
- [ ] 缓冲区池减少分配次数 > 80%
- [ ] 线程安全

#### FR-007: 外部内存包装

| 属性 | 描述 |
|------|------|
| **功能** | 包装用户提供的内存 |
| **条件** | 必须页对齐 |
| **行为** | 不拥有内存所有权 |

**验收标准**:
- [ ] 页对齐内存零拷贝包装
- [ ] 非对齐内存自动复制
- [ ] 正确处理生命周期

### 2.3 策略选择

#### FR-008: 自适应策略选择

| 属性 | 描述 |
|------|------|
| **功能** | 根据数据特征选择执行器 |
| **输入** | 算子类型、数据特征 |
| **输出** | 最优执行器 (CPU/GPU/NPU) |
| **可配置** | 支持阈值配置 |

**验收标准**:
- [ ] 小数据量自动使用 CPU
- [ ] 大数据量自动使用 GPU
- [ ] 可手动覆盖策略

---

## 三、非功能需求

### 3.1 性能需求

#### NFR-001: 吞吐量目标

| 算子 | CPU 吞吐量 | GPU 吞吐量 | vs DuckDB |
|------|-----------|-----------|-----------|
| Filter | > 2500 M/s | > 6000 M/s | > 4x |
| Aggregate | > 17000 M/s | > 20000 M/s | > 5x |
| Join | > 300 M/s | > 900 M/s | > 2x |
| TopK | > 2000 M/s | - | > 10x |
| Sort | > 2000 M/s | - | > 5x |

#### NFR-002: 带宽利用率

| 指标 | 目标 | 当前 |
|------|------|------|
| Aggregate 带宽 | > 80 GB/s | 118 GB/s ✓ |
| Filter 带宽 | > 20 GB/s | 26.8 GB/s ✓ |
| 理论峰值利用率 | > 25% | 29.5% ✓ |

#### NFR-003: GPU 加速比

| 场景 | 目标加速比 | 当前 |
|------|-----------|------|
| Filter 50M | > 2x | 2.65x ✓ |
| Join 1M×10M | > 3x | 4.26x ✓ |
| Aggregate | ≈ 1x (带宽限制) | 1.05x ✓ |

### 3.2 延迟需求

#### NFR-004: 响应时间

| 操作 | 数据量 | 目标延迟 |
|------|--------|---------|
| Filter | 1M | < 1 ms |
| Aggregate | 1M | < 0.1 ms |
| Join | 100K×1M | < 5 ms |
| TopK-100 | 500K | < 0.5 ms |

#### NFR-005: GPU 启动开销

| 指标 | 目标 |
|------|------|
| GPU 启动时间 | < 0.5 ms |
| 最小 GPU 收益数据量 | > 500K rows |

### 3.3 可靠性需求

#### NFR-006: 正确性

| 指标 | 要求 |
|------|------|
| 结果一致性 | 与 DuckDB 100% 一致 |
| 浮点精度 | IEEE 754 标准 |
| 边界处理 | 正确处理空输入、极值 |

#### NFR-007: 稳定性

| 指标 | 要求 |
|------|------|
| 内存泄漏 | 无泄漏 |
| 崩溃率 | 0% |
| GPU 超时 | 自动回退到 CPU |

### 3.4 兼容性需求

#### NFR-008: 平台兼容

| 平台 | 支持状态 |
|------|---------|
| macOS 14.0+ (Sonoma) | 完全支持 |
| macOS 15.0+ (Sequoia) | 完全支持 |
| Apple M1/M2/M3 | 兼容 (未优化) |
| Apple M4 | 完全优化 |
| Intel Mac | 仅 CPU 路径 |

#### NFR-009: API 兼容

| 接口 | 要求 |
|------|------|
| C++ 标准 | C++17 |
| ABI 稳定 | 保持向后兼容 |
| DuckDB 集成 | 可作为后端替换 |

### 3.5 可维护性需求

#### NFR-010: 代码质量

| 指标 | 要求 |
|------|------|
| 代码覆盖率 | > 80% |
| 文档覆盖 | 所有公共 API |
| 编码规范 | 遵循项目规范 |

#### NFR-011: 可观测性

| 指标 | 要求 |
|------|------|
| 性能指标 | 吞吐量、延迟、带宽 |
| 策略日志 | 记录选择决策 |
| 错误日志 | 详细错误信息 |

---

## 四、约束条件

### 4.1 技术约束

| 约束 | 描述 |
|------|------|
| **语言** | C++17, Objective-C++ (.mm) |
| **GPU API** | Metal (无 CUDA/OpenCL) |
| **SIMD** | ARM NEON (无 AVX) |
| **内存模型** | 统一内存架构 (UMA) |

### 4.2 硬件约束

| 约束 | 描述 |
|------|------|
| **最低硬件** | Apple M1 |
| **推荐硬件** | Apple M4 |
| **最小内存** | 8 GB |
| **推荐内存** | 16 GB+ |

### 4.3 软件约束

| 约束 | 描述 |
|------|------|
| **操作系统** | macOS 14.0+ |
| **编译器** | Apple Clang 15+ |
| **构建系统** | Make / CMake |

---

## 五、用例设计

### 5.1 UC-001: 执行过滤查询

```
用例名称: 执行过滤查询
参与者: 数据库引擎
前置条件: 数据已加载到内存
触发条件: 收到 WHERE 子句

主流程:
1. 引擎调用 filter_i32_v4(input, count, op, value, output)
2. 系统检查数据特征 (大小、对齐)
3. 策略选择器选择执行器 (CPU/GPU)
4. 执行过滤操作
5. 返回匹配索引数组

替代流程:
3a. GPU 不可用
    - 回退到 CPU SIMD 执行

后置条件: 返回满足条件的行索引
```

### 5.2 UC-002: 执行 Hash Join

```
用例名称: 执行 Hash Join
参与者: 数据库引擎
前置条件: Build 和 Probe 表已加载
触发条件: 收到 JOIN 语句

主流程:
1. 引擎调用 hash_join_gpu_uma(build, probe, type, result)
2. 系统分配 UMA 缓冲区
3. GPU 并行构建哈希表
4. GPU 并行探测哈希表
5. 使用前缀和批量写入结果
6. 返回匹配索引对

替代流程:
2a. probe 数据量 < 500K
    - 直接使用 CPU v3 执行

后置条件: result 包含所有匹配的索引对
```

### 5.3 UC-003: 计算聚合

```
用例名称: 计算聚合统计
参与者: 数据库引擎
前置条件: 列数据已加载
触发条件: 收到聚合查询

主流程:
1. 引擎调用 aggregate_all_i32_v3(input, count)
2. 系统选择执行器 (CPU 优先，接近带宽上限)
3. 执行融合 kernel (SUM+MIN+MAX 单次遍历)
4. 返回 AggregateStats 结构

替代流程:
2a. 数据量 > 100M
    - 可选使用 GPU 获得额外并行度

后置条件: 返回 {sum, min, max, count}
```

---

## 六、数据需求

### 6.1 数据类型支持

| 类型 | Filter | Aggregate | Join | Sort | TopK |
|------|--------|-----------|------|------|------|
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | - | ✓ | ✓ |
| float | ✓ | ✓ | - | ✓ | ✓ |
| double | ✓ | ✓ | - | ✓ | ✓ |
| string | - | - | - | - | - |

### 6.2 数据规模

| 指标 | 最小 | 最大 | 典型 |
|------|------|------|------|
| 行数 | 1 | 2^32-1 | 1M-100M |
| 单列大小 | 4 B | 16 GB | 400 MB |
| Join Build | 1 | 10M | 100K-1M |
| Join Probe | 1 | 100M | 1M-50M |

### 6.3 数据对齐

| 类型 | 对齐要求 | 说明 |
|------|---------|------|
| 普通数据 | 4 字节 | 基本对齐 |
| SIMD 数据 | 16 字节 | NEON 优化 |
| UMA 数据 | 16384 字节 | 零拷贝要求 |

---

## 七、接口需求

### 7.1 C++ API

```cpp
// Filter
size_t filter_i32_v4(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// Aggregate
AggregateStats aggregate_all_i32_v3(const int32_t* input, size_t count);

// Join
size_t hash_join_gpu_uma(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result,
                          const JoinConfigV4& config = {});

// Sort
void sort_i32_v2(int32_t* data, size_t count, SortOrder order);

// TopK
size_t topk_i32_v5(const int32_t* input, size_t count,
                    size_t k, uint32_t* out_indices, TopKOrder order);
```

### 7.2 配置接口

```cpp
// Filter 配置
struct FilterConfigV4 {
    FilterStrategy strategy = FilterStrategy::AUTO;
    float selectivity_hint = -1.0f;  // 选择率提示
};

// Join 配置
struct JoinConfigV4 {
    JoinStrategy strategy = JoinStrategy::AUTO;
    size_t bloom_fpr = 0.01;  // Bloom 假阳性率
};

// 全局阈值配置
namespace thresholds {
    extern size_t FILTER_GPU_MIN;
    extern size_t JOIN_GPU_MIN_PROBE;
    // ...
}
```

---

## 八、测试需求

### 8.1 单元测试

| 测试类别 | 覆盖范围 |
|---------|---------|
| Filter 正确性 | 所有比较操作、边界值 |
| Aggregate 正确性 | 所有聚合函数、溢出处理 |
| Join 正确性 | 所有 Join 类型、空表处理 |
| UMA 内存 | 分配、释放、池化 |

### 8.2 性能测试

| 测试类别 | 指标 |
|---------|------|
| 吞吐量测试 | M rows/s |
| 延迟测试 | ms |
| 带宽测试 | GB/s |
| 加速比测试 | vs DuckDB |

### 8.3 压力测试

| 测试类别 | 场景 |
|---------|------|
| 大数据量 | 100M+ rows |
| 长时间运行 | 24 小时稳定性 |
| 内存压力 | 接近系统内存上限 |

---

## 九、文档需求

| 文档 | 状态 | 描述 |
|------|------|------|
| 架构设计 | ✓ 完成 | ARCHITECTURE_DESIGN.md |
| 需求设计 | ✓ 完成 | REQUIREMENTS_DESIGN.md |
| 性能设计 | ✓ 完成 | PERFORMANCE_DESIGN.md |
| API 参考 | 待完成 | API_REFERENCE.md |
| 基准测试报告 | ✓ 完成 | UMA_BENCHMARK_REPORT_V2.md |

---

## 十、验收标准

### 10.1 功能验收

- [ ] 所有算子结果与 DuckDB 一致
- [ ] 自适应策略正确选择执行器
- [ ] GPU 回退机制正常工作
- [ ] 无内存泄漏

### 10.2 性能验收

- [ ] Filter: CPU > 2500 M/s, GPU > 6000 M/s
- [ ] Aggregate: 带宽 > 80 GB/s
- [ ] Join: GPU 加速 > 2x (1M×10M)
- [ ] TopK: > 10x vs DuckDB

### 10.3 质量验收

- [ ] 单元测试通过率 100%
- [ ] 无 P0/P1 缺陷
- [ ] 文档完整
