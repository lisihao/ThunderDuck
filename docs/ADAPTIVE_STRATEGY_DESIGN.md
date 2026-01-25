# ThunderDuck 自适应策略选择器设计文档

> **版本**: 1.0.0 | **日期**: 2026-01-24

## 一、概述

自适应策略选择器是 ThunderDuck 的核心组件，根据数据特征自动选择最优执行策略（CPU SIMD vs GPU），避免 GPU 启动开销在小数据量时的负面影响，同时在大数据量时充分发挥 GPU 并行优势。

## 二、架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    StrategySelector                         │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
│  │ selectFilter │ selectAggreg │ selectJoin   │ selectTK │  │
│  └──────────────┴──────────────┴──────────────┴──────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────────┤
│  │ DataCharacteristics                                      │
│  │ - row_count        : 数据行数                            │
│  │ - column_count     : 列数 (Join 用)                      │
│  │ - element_size     : 元素大小                            │
│  │ - selectivity      : 选择率估计                          │
│  │ - cardinality_ratio: 基数比                              │
│  │ - is_page_aligned  : 页对齐 (零拷贝)                     │
│  └──────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────┴───────────────┐
          ▼                               ▼
   ┌──────────┐                    ┌──────────┐
   │ CPU SIMD │                    │   GPU    │
   └──────────┘                    └──────────┘
```

## 三、阈值设计

基于 M4 基准测试结果的最优阈值:

| 算子 | GPU 阈值 | 原因 |
|------|----------|------|
| Filter | 10M rows | CPU SIMD 已达 2500 M/s，GPU 启动开销主导 |
| Aggregate | 100M rows | CPU 已接近内存带宽上限 (118 GB/s) |
| Join | 500K-50M probe | 中等规模 GPU 最佳 (4x 加速) |
| TopK | 50M rows | CPU 采样方法高效，K < 1000 时 CPU 更优 |

### 3.1 Join 策略细节

```cpp
if (probe_count < 500K) {
    // GPU 启动开销主导
    return CPU_SIMD;
}
if (build_count < 10K) {
    // build 表太小，哈希表利用率低
    return CPU_SIMD;
}
if (probe_count > 50M) {
    // 仍用 GPU，但期望加速比降低 (1.3-2x)
    return GPU;  // bandwidth limited
}
// 最佳区间
return GPU;  // 期望 3-5x 加速
```

### 3.2 Filter 策略细节

```cpp
if (row_count < 10M) {
    return CPU_SIMD;  // 2500 M/s 已很快
}
if (selectivity > 0.5) {
    return CPU_SIMD;  // 高选择率时 GPU 原子争用严重
}
if (!is_page_aligned && row_count < 20M) {
    return CPU_SIMD;  // 非对齐需要拷贝，阈值加倍
}
return GPU;
```

## 四、API 设计

### 4.1 核心类

```cpp
namespace thunderduck::strategy {

enum class Executor {
    CPU_SCALAR,     // CPU 标量
    CPU_SIMD,       // CPU SIMD (NEON)
    GPU,            // GPU (Metal)
    NPU,            // NPU (BNNS)
    AUTO            // 自动选择
};

enum class OperatorType {
    FILTER,
    AGGREGATE_SUM,
    AGGREGATE_MINMAX,
    AGGREGATE_GROUP,
    JOIN_HASH,
    TOPK
};

struct DataCharacteristics {
    size_t row_count;
    size_t column_count;
    size_t element_size;
    float selectivity;
    float cardinality_ratio;
    bool is_page_aligned;
};

class StrategySelector {
public:
    static StrategySelector& instance();
    Executor select(OperatorType op, const DataCharacteristics& data,
                    Executor hint = Executor::AUTO);
    const char* get_decision_reason() const;  // 调试用
};

}
```

### 4.2 便捷函数

```cpp
// 快速判断
bool should_use_gpu(OperatorType op, size_t row_count);

// 带详细参数
Executor select_executor(OperatorType op, size_t row_count, float selectivity);
```

## 五、实现文件

| 文件 | 说明 |
|------|------|
| `include/thunderduck/adaptive_strategy.h` | 接口定义和阈值常量 |
| `src/core/adaptive_strategy.cpp` | 策略选择逻辑实现 |
| `src/gpu/hash_join_uma.mm` | Join GPU 策略集成 |
| `src/gpu/filter_uma.mm` | Filter GPU 策略集成 |
| `src/gpu/aggregate_uma.mm` | Aggregate GPU 策略集成 |
| `src/gpu/topk_uma.mm` | TopK GPU 策略集成 |

## 六、性能验证

### 6.1 Join 算子验证

| 测试场景 | 数据规模 | 策略 | 加速比 |
|----------|----------|------|--------|
| J1 | 10K × 100K | CPU (阈值回退) | ~1x |
| J2 | 100K × 1M | GPU | **2.36x** |
| J3 | 1M × 10M | GPU | **4.30x** |
| J4 | 5M × 50M | GPU (带宽限) | **1.33x** |

### 6.2 策略选择验证

- J1 正确回退到 CPU：probe 100K < 阈值 500K
- J2-J3 正确使用 GPU：probe 在最佳区间
- J4 继续使用 GPU：虽带宽受限仍比 CPU 快

## 七、调试支持

```cpp
auto& selector = StrategySelector::instance();
auto executor = selector.select(OperatorType::JOIN_HASH, data);

// 获取决策原因
printf("Strategy: %s\n", selector.get_decision_reason());
// 输出: "Optimal range for GPU join" 或 "Probe count below threshold (500K)"
```

## 八、扩展点

### 8.1 动态阈值调整

未来可基于运行时统计动态调整阈值:

```cpp
class AdaptiveThresholds {
    void record_execution(OperatorType op, size_t size,
                          Executor used, double time_ms);
    void update_thresholds();  // 基于历史数据调整
};
```

### 8.2 NPU 支持

当前 NPU 检测为 false，未来可添加:

```cpp
if (npu_available_ && op == OperatorType::BLOOM_FILTER) {
    return Executor::NPU;  // BNNS 加速
}
```

## 九、总结

自适应策略选择器实现了:

1. **基于数据特征的智能调度** - 根据规模、选择率、对齐等特征选择
2. **避免 GPU 启动开销陷阱** - 小数据量自动回退 CPU
3. **最大化 GPU 收益** - 在最佳区间获得 3-5x 加速
4. **调试友好** - 提供决策原因便于分析

核心收益: **Join 在 1M-10M 规模获得 4.3x 加速，同时避免小数据量的性能回退**
