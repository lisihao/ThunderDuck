# ThunderDuck V14 全面性能基准测试报告

> **版本**: 14.0 | **日期**: 2026-01-27 | **标签**: 新性能基线

## 一、测试环境

| 项目 | 配置 |
|------|------|
| 平台 | Apple Silicon M4 |
| 系统 | macOS 15.x |
| 编译器 | clang++ (Apple LLVM) |
| 优化 | -O3 -mcpu=native -march=armv8-a+crc |
| 对比版本 | DuckDB (原版), ThunderDuck V3-V14 |

## 二、测试数据规模

| 算子 | 数据量 | 说明 |
|------|--------|------|
| Filter | 10M 行 × 4 bytes = 38 MB | WHERE value > 500000 |
| Aggregate | 10M 行 × 4 bytes = 38 MB | SUM(value) |
| GROUP BY | 10M 行 × 8 bytes = 76 MB | 1000 分组 |
| Hash Join | 100K × 1M = 4.4 MB | 100% 匹配率 |
| TopK | 10M 行 × 4 bytes = 38 MB | Top 100 |

## 三、性能测试结果

### 3.1 Filter 算子

**等效 SQL**: `SELECT * FROM table WHERE value > 500000`

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 状态 |
|------|------|----------|------------|-----------|-------|------|
| DuckDB | CPU | 1.561 | 25.62 | 1.00x | - | **基准** |
| V3 SIMD | CPU | 18.022 | 2.22 | 0.09x | 1.00x | 慢 |
| V6 Prefetch | CPU | 3.974 | 10.07 | 0.39x | 4.53x | 慢 |
| V8 Parallel | CPU | 5.925 | 6.75 | 0.26x | 3.04x | 慢 |

**分析**: Filter 算子全面落后于 DuckDB，需要重点优化。V3 基础版本存在严重性能问题。

**优化方向**:
- 检查 V3 SIMD 实现是否有效启用 SIMD
- 分析内存访问模式，优化预取策略
- 考虑使用 DuckDB 类似的向量化执行模型

### 3.2 Aggregate 算子 (SUM)

**等效 SQL**: `SELECT SUM(value) FROM table`

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 状态 |
|------|------|----------|------------|-----------|-------|------|
| DuckDB | CPU | 1.664 | 24.03 | 1.00x | - | 基准 |
| V3 SIMD | CPU | 1.123 | 35.63 | 1.48x | 1.00x | 好 |
| **V7 Optimized** | CPU | **0.478** | **83.67** | **3.48x** | 2.35x | **最优** |
| V8 Parallel | CPU | 0.522 | 76.68 | 3.19x | 2.15x | 好 |

**分析**: V7 Optimized 达到 3.48x 加速比，性能优异。

**关键优化**:
- 8 路累加器消除依赖链
- 256 元素批次减少归约次数
- 自适应预取策略

### 3.3 GROUP BY 算子

**等效 SQL**: `SELECT group_id, SUM(value) FROM table GROUP BY group_id`

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 状态 |
|------|------|----------|------------|-----------|-------|------|
| DuckDB | CPU | 2.648 | 30.21 | 1.00x | - | 基准 |
| V3 Basic | CPU | 2.877 | 27.81 | 0.92x | 1.00x | 慢 |
| V7 SIMD | CPU | 2.801 | 28.56 | 0.95x | 1.03x | 慢 |
| V8 Parallel | CPU | 1.190 | 67.22 | 2.23x | 2.42x | 好 |
| V9 GPU | GPU | 3.131 | 25.55 | 0.85x | 0.92x | 慢 |
| **V14 SIMD合并** | CPU | **1.074** | **74.48** | **2.47x** | 2.68x | **最优** |

**分析**: V14 SIMD 合并版本是当前最优，达到 2.47x 加速比。

**V14 优化要点**:
- 4 线程并行分区累加
- SIMD 向量化合并线程结果
- 4 路循环展开 + 预取

**GPU 版本分析**: V9 GPU 出现反向加速 (0.85x)，原因是 1000 分组 × 40 threadgroups = 40K 次全局原子操作竞争。

### 3.4 Hash Join 算子

**等效 SQL**: `SELECT * FROM build JOIN probe ON build.key = probe.key`

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 状态 |
|------|------|----------|------------|-----------|-------|------|
| DuckDB | CPU | 1.727 | 2.55 | 1.00x | - | 基准 |
| V3 Radix16 | CPU | 0.962 | 4.57 | 1.80x | 1.00x | 好 |
| V10 Optimized | CPU | 0.982 | 4.48 | 1.76x | 0.98x | 好 |
| V11 SIMD | CPU | 3.370 | 1.31 | 0.51x | 0.29x | **问题** |
| V13 TwoPhase | CPU | 8.954 | 0.49 | 0.19x | 0.11x | **问题** |
| **V14 Best** | CPU | **0.956** | **4.60** | **1.81x** | 1.01x | **最优** |

**分析**: V14 委托 V3 实现，达到 1.81x 加速比。

**V11/V13 问题分析**:
- V11 SIMD: 负载因子 3.0x 导致每分区 18-20KB，超出 L1 缓存
- V13 TwoPhase: 两阶段计数遍历开销 > 预分配收益

**经验教训**:
1. 更大的 SIMD 向量并不总是更快
2. 两阶段算法需要仔细评估额外遍历开销

### 3.5 TopK 算子

**等效 SQL**: `SELECT * FROM table ORDER BY value DESC LIMIT 100`

| 版本 | 设备 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 状态 |
|------|------|----------|------------|-----------|-------|------|
| DuckDB | CPU | 3.041 | 13.15 | 1.00x | - | 基准 |
| V3 HeapSort | CPU | 4.733 | 8.45 | 0.64x | 1.00x | 慢 |
| **V5 Sampling** | CPU | **0.505** | **79.19** | **6.02x** | 9.37x | **最优** |
| V6 Adaptive | CPU | 0.528 | 75.70 | 5.76x | 8.96x | 好 |

**分析**: V5 Sampling 达到 6.02x 加速比，性能卓越。

**V5 采样优化原理**:
- 采样估计第 K 大元素阈值
- 单遍扫描过滤候选集
- 小数据集快速排序

## 四、性能总览

### 4.1 各算子最优实现

| 算子 | 最优版本 | vs DuckDB | 优先级 |
|------|----------|-----------|--------|
| Filter | V6 Prefetch | 0.39x | ★★★ **高** |
| Aggregate | V7 Optimized | 3.48x | ★ 低 |
| GROUP BY | V14 SIMD合并 | 2.47x | ★★ 中 |
| Hash Join | V14 Best (V3) | 1.81x | ★★ 中 |
| TopK | V5 Sampling | 6.02x | ★ 低 |

### 4.2 优化建议

#### 高优先级 (Filter)

Filter 是当前最大的性能瓶颈，所有版本都慢于 DuckDB。

建议:
1. **分析 DuckDB Filter 实现**: 研究其向量化执行模型
2. **批量索引生成**: 考虑使用位图中间表示
3. **分支预测优化**: 使用无分支 SIMD 选择

#### 中优先级 (GROUP BY, Hash Join)

GROUP BY 和 Hash Join 已有 1.8x-2.5x 加速，但仍有提升空间。

建议:
1. **Hash Join**: 探索 Robin-Hood 哈希减少最坏情况
2. **GROUP BY**: 针对低基数场景优化

#### 低优先级 (Aggregate, TopK)

已达到 3.5x-6x 加速，性能优异。

## 五、数据吞吐带宽分析

| 算子 | DuckDB | ThunderDuck | 提升 |
|------|--------|-------------|------|
| Filter | 25.62 GB/s | 10.07 GB/s | -61% |
| Aggregate | 24.03 GB/s | 83.67 GB/s | +248% |
| GROUP BY | 30.21 GB/s | 74.48 GB/s | +147% |
| Hash Join | 2.55 GB/s | 4.60 GB/s | +80% |
| TopK | 13.15 GB/s | 79.19 GB/s | +502% |

## 六、V14 架构决策

基于测试结果，V14 采用以下架构:

### 6.1 Hash Join V14

```cpp
// 委托给已验证最优的实现
if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
    return hash_join_i32_v10(...)  // SEMI/ANTI 提前退出优化
}
return hash_join_i32_v3(...)       // INNER JOIN: Radix16 分区
```

### 6.2 GROUP BY V14

```cpp
// 多线程并行 + SIMD 合并
1. 4 线程分区，每线程独立累加器
2. SIMD 向量化合并: vaddq_s64
3. 4 路循环展开 + 预取
```

## 七、下一步计划

1. **Filter 算子重写** (高优先级)
   - 分析 DuckDB 向量化执行模型
   - 实现批量位图过滤
   - 基准测试目标: 2x vs DuckDB

2. **Hash Join 优化** (中优先级)
   - Robin-Hood 哈希集成
   - 更小分区探索

3. **GPU 利用率提升** (低优先级)
   - GROUP BY GPU: 减少原子操作竞争
   - 考虑分层聚合

## 八、附录

### 测试命令

```bash
# 编译
make clean && make lib

# 运行完整测试
make v14-full

# 运行快速测试 (1M 数据)
./build/v14_comprehensive_benchmark --small
```

### 关键文件

| 文件 | 说明 |
|------|------|
| `src/operators/join/hash_join_v14.cpp` | V14 Hash Join |
| `src/operators/aggregate/group_aggregate_v14.cpp` | V14 GROUP BY |
| `include/thunderduck/v14.h` | V14 统一接口 |
| `benchmark/v14_comprehensive_benchmark.cpp` | 基准测试程序 |
