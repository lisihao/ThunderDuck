# V9.3 智能策略选择分组聚合 - 设计文档

> 版本: 9.3 | 日期: 2026-01-26

## 一、需求概述

### 背景

V9.2 实现了 GPU 两阶段分组聚合优化，在高竞争场景（少分组）下相比 V8 GPU 原子版提升 105%。但基准测试显示，CPU 多线程 (v4-MT) 和 GPU 两阶段 (v5) 各有优势场景：

| 数据规模 | 分组数 | CPU v4-MT | GPU v5 | 最优 |
|---------|--------|-----------|--------|------|
| 10M | 10 | 2.54x | 2.53x | CPU |
| 10M | 100 | 2.29x | 1.35x | CPU |
| 10M | 1000 | 2.64x | 1.38x | CPU |
| **50M** | **10** | **2.55x** | **2.97x** | **GPU** |

### 目标

设计并实现智能策略选择器，根据数据特征自动选择最优实现：
- 小数据 → CPU 单线程（避免线程开销）
- 中等数据 → CPU 多线程（最佳通用性能）
- 大数据 + 少分组 → GPU 两阶段（带宽优势）

## 二、技术方案

### 策略选择规则

```
if count < 100K:
    return V4_SINGLE      // 线程启动开销 > 收益

elif count >= 50M AND num_groups <= 32 AND GPU可用:
    return V5_GPU         // GPU 带宽优势

else:
    return V4_PARALLEL    // 最佳通用性能
```

### 阈值来源

| 参数 | 值 | 来源 |
|------|-----|------|
| SINGLE_THREAD_MAX | 100K | M4 线程启动开销 ~50μs |
| GPU_MIN_COUNT | 50M | GPU 带宽利用率拐点 |
| GPU_MAX_GROUPS | 32 | Threadgroup 共享内存竞争阈值 |

## 三、详细设计

### 新增类型

```cpp
// 分组聚合策略版本
enum class GroupAggregateVersion {
    V4_SINGLE,    // CPU 单线程
    V4_PARALLEL,  // CPU 多线程
    V5_GPU,       // GPU 两阶段
    AUTO          // 自动选择
};

// 配置结构
struct GroupAggregateConfig {
    GroupAggregateVersion version = AUTO;
    bool debug_log = false;
};
```

### API 接口

```cpp
// 策略选择
GroupAggregateVersion select_group_aggregate_strategy(
    size_t count, size_t num_groups);

// 获取选择原因（调试用）
const char* get_group_aggregate_strategy_reason();

// v6 智能策略函数
void group_sum_i32_v6(const int32_t* values, const uint32_t* groups,
                      size_t count, size_t num_groups, int64_t* out_sums);

void group_sum_i32_v6_config(const int32_t* values, const uint32_t* groups,
                             size_t count, size_t num_groups, int64_t* out_sums,
                             const GroupAggregateConfig& config);
```

## 四、性能验证

### 基准测试结果

```
╔═══════════════════════════╦══════════════╦═══════════════╦═══════════════════╗
║ 场景                      ║ 策略选择     ║ V6 vs 最优    ║ 验证结果          ║
╠═══════════════════════════╬══════════════╬═══════════════╬═══════════════════╣
║ 50K, 10 groups            ║ V4_SINGLE    ║ ~最优         ║ PASS (optimal)    ║
║ 10M, 100 groups           ║ V4_PARALLEL  ║ ~最优         ║ PASS (optimal)    ║
║ 50M, 10 groups            ║ V5_GPU       ║ ~最优         ║ PASS (optimal)    ║
║ 50M, 1000 groups          ║ V4_PARALLEL  ║ ~最优         ║ PASS (close)      ║
╚═══════════════════════════╩══════════════╩═══════════════╩═══════════════════╝
```

### 关键发现

- **50M 元素 + 10 分组**: GPU v5 达到 **2.97x**，CPU v4-MT 为 **2.55x**
- GPU 在此场景下**快 16.5%**
- 智能策略正确识别并利用了 GPU 优势

## 五、文件清单

| 文件 | 说明 |
|------|------|
| `include/thunderduck/aggregate.h` | 新增 v6 API 声明 |
| `src/operators/aggregate/simd_aggregate_v6.cpp` | v6 智能策略实现 |
| `benchmark/test_group_aggregate_v93.cpp` | V9.3 对比测试 |
| `docs/V93_INTELLIGENT_STRATEGY_DESIGN.md` | 本设计文档 |

## 六、总结

V9.3 智能策略选择成功实现了：

1. **自动优化**: 根据数据规模和分组数自动选择最优实现
2. **全覆盖**: 小/中/大数据场景均有针对性优化
3. **零配置**: 用户无需手动选择，v6 函数自动处理
4. **可观测**: 提供 debug_log 选项查看策略决策
5. **向后兼容**: v4/v5 函数仍可独立使用
