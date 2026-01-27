# V14 Hash Join & GROUP BY 深度优化 - 设计文档

> **版本**: 14.0 | **日期**: 2026-01-27

## 一、需求概述

### 性能目标
| 算子 | 当前最优 | 目标加速比 |
|------|----------|------------|
| Hash Join | V10 (4.28x) | 8x+ |
| GROUP BY | V8 Parallel (2.66x) | 4x+ |

### 问题分析

#### Hash Join 瓶颈
- V11 SIMD 版本反而更慢 (1.43x)：负载因子 3.0x 导致缓存 miss
- V3/V10 使用 1.7x 负载因子，缓存命中率 ~90%

#### GROUP BY 瓶颈
- GPU V9 版本出现反向加速 (0.63x)：全局原子操作竞争
- 40 threadgroups × 1000 groups = 40,000 次原子竞争

## 二、技术方案

### P0: Hash Join V14 优化

#### 方案 A: 分区参数调优
```cpp
// 当前 V3 配置
constexpr size_t NUM_PARTITIONS = 16;   // 4 bits
constexpr float LOAD_FACTOR = 1.7f;

// V14 优化配置
constexpr size_t NUM_PARTITIONS = 32;   // 5 bits
constexpr float LOAD_FACTOR = 1.5f;     // 更紧凑
```

预期收益: +15-20%

#### 方案 B: 两阶段预分配
```cpp
// Phase 1: 计数 (无写入)
size_t total_matches = count_all_matches(probe_keys, ht);

// Phase 2: 一次性分配
ensure_join_result_capacity(result, total_matches);

// Phase 3: 填充 (无动态扩容)
fill_matches(probe_keys, ht, result);
```

预期收益: 高匹配率场景 +20-30%

### P1: GROUP BY V14 优化

#### 方案 A: 寄存器缓冲累加
```cpp
constexpr size_t REG_CACHE_SIZE = 8;
int64_t reg_cache[REG_CACHE_SIZE];       // 8 个累加器
uint32_t reg_gids[REG_CACHE_SIZE];        // 对应的 group ID
uint8_t cache_valid = 0;                  // 位图标记有效性
```

预期收益: +30% (减少内存写入)

#### 方案 B: 多路分流
```cpp
// 按 group_id % 4 分成 4 路
// 每路独立 SIMD 累加，减少依赖
#pragma omp parallel for
for (int w = 0; w < 4; ++w) {
    accumulate_way(way_values[w], way_groups[w], out_sums);
}
```

预期收益: +15% (并行化)

## 三、详细设计

### 文件结构

```
include/thunderduck/
├── v14.h                    # V14 统一接口

src/operators/join/
├── hash_join_v14.cpp        # V14 两阶段 Hash Join

src/operators/aggregate/
├── group_aggregate_v14.cpp  # V14 GROUP BY 优化

src/core/
├── v14_unified.cpp          # V14 路由实现

benchmark/
├── v14_benchmark.cpp        # V14 基准测试
```

### 接口设计

```cpp
namespace thunderduck::v14 {

// Hash Join V14
size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type, JoinResult* result,
                     ExecutionStats* stats = nullptr);

// GROUP BY V14
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);
}
```

## 四、实现计划

| 阶段 | 任务 | 预期收益 |
|------|------|----------|
| 1 | Hash Join 分区参数优化 | +15% |
| 2 | Hash Join 两阶段预分配 | +25% |
| 3 | GROUP BY 寄存器缓冲 | +30% |
| 4 | GROUP BY 多路分组 | +15% |
| 5 | V14 统一接口 | - |
| 6 | 基准测试验证 | - |

## 五、验证计划

```bash
# 编译
make clean && make lib

# 运行 V14 基准测试
make v14-bench

# 期望结果
# Hash Join (100K×1M 100%): 4.28x → 8x+
# GROUP BY (10M, 1000 groups): 2.66x → 4x+
```

## 六、测试结论

### 6.1 原计划优化效果验证

| 优化方案 | 预期 | 实测 | 结论 |
|----------|------|------|------|
| Hash Join 两阶段预分配 | +25% | -94% (0.06x) | **无效** - 计数遍历开销 > 收益 |
| Hash Join 32 分区 | +15% | +20% | 有效 (通过委托 V3) |
| GROUP BY 寄存器缓冲 | +30% | -80% (低基数) | **无效** - O(n) 查找 + 频繁 flush |
| GROUP BY SIMD 合并 | +15% | +34% | **有效** |

### 6.2 最终 V14 策略

基于测试结果，V14 采用以下策略：

**Hash Join V14**:
- 委托给 V3 (INNER JOIN) - 已是最优
- 委托给 V10 (SEMI/ANTI JOIN) - 提前退出优化

**GROUP BY V14**:
- 多线程并行 + SIMD 合并
- 移除寄存器缓冲（测试证明无效）

### 6.3 最终性能结果

| 算子 | V14 vs 基准 | 说明 |
|------|-------------|------|
| Hash Join | 1.08-1.20x | V14 vs V10, 委托 V3 实现 |
| GROUP BY (1000 分组) | 1.34x | V14 vs V4 Parallel, SIMD 合并优化 |
| GROUP BY (8-64 分组) | ~1.0x | 持平，低基数无额外优化 |

### 6.4 经验教训

1. **两阶段算法慎用**：额外遍历开销可能超过预分配收益
2. **寄存器缓冲限制**：O(n) 查找对于低基数场景不如直接数组索引
3. **基准测试先行**：先测试再承诺，避免过度工程化

## 七、风险和备选

| 风险 | 备选方案 |
|------|----------|
| 两阶段 Join 内存开销过大 | ~~分批计数~~ **已验证无效** |
| 寄存器缓冲对高基数无效 | ~~回退 V8 Parallel~~ **已验证无效，直接移除** |
| 多路分流分流开销过大 | ~~使用固定分区~~ **已验证无效，直接移除** |
