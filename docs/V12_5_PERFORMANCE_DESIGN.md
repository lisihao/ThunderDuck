# ThunderDuck V12.5 - 性能之选

> **版本标签**: V12.5 - 性能之选
> **设计日期**: 2026-01-27
> **目标**: 集成各版本最优算子，消除路由开销，达到极致性能

---

## 一、版本定位

V12.5 是 ThunderDuck 的"性能之选"版本，目标是：

1. **零妥协**: 每个算子使用其历史最优实现
2. **零开销**: 消除不必要的路由和策略判断
3. **自适应**: 根据数据规模智能选择 CPU/GPU

---

## 二、优化策略矩阵

| 算子 | 小数据 (<1M) | 大数据 (≥10M) | 最优版本 |
|------|-------------|---------------|----------|
| **Filter** | GPU Metal (V12) | CPU SIMD (V3) | 自适应 |
| **Aggregate** | CPU SIMD+ (V9) | GPU Metal (V7) | 自适应 |
| **GROUP BY** | CPU 4核 (V8) | CPU 4核 (V8) | V8 固定 |
| **TopK** | CPU Count (V8) | CPU Sampling (V7) | 直调无路由 |
| **Hash Join** | 自适应 V7/V11 | 自适应 V7/V11 | 匹配率自适应 |

---

## 三、关键优化点

### 3.1 TopK 直调优化 (P0)

**问题**: V12 TopK 8.97x vs V8 最优 13.36x，差距 4.39x

**根因**: 路由层增加了函数调用、策略判断、结果拷贝开销

**解决方案**:
```cpp
// V12.5: 直接调用最优实现，零路由开销
inline size_t topk_v125(const int32_t* data, size_t count, size_t k,
                        int32_t* out_values, size_t* out_indices) {
    if (count <= 1000000) {
        // 1M 以下: V8 Count-Based (13.36x)
        return topk::topk_count_based_i32(data, count, k, out_values, out_indices);
    } else {
        // 1M 以上: V7 Sampling (5.12x)
        return topk::topk_sampling_i32(data, count, k, out_values, out_indices);
    }
}
```

**预期收益**: 8.97x → 13x+

### 3.2 Filter 自适应优化 (P1)

**问题**: V12 Filter 10M 2.70x vs V3 最优 3.02x

**根因**: 10M 数据 GPU 调度开销抵消了并行收益

**解决方案**:
```cpp
// V12.5: 根据数据规模选择最优路径
if (count < 5000000) {
    // 5M 以下: GPU Metal (7.54x on 1M)
    return gpu::filter_gt_i32_metal(data, count, threshold, output);
} else {
    // 5M 以上: CPU SIMD V3 (3.02x on 10M)
    return filter::simd_filter_gt_i32(data, count, threshold, output);
}
```

**预期收益**: 2.70x → 3.02x

### 3.3 Aggregate 自适应优化 (P2)

**问题**: 不同数据规模最优设备不同

**解决方案**:
```cpp
// V12.5: CPU/GPU 自适应
if (count < 5000000) {
    // 5M 以下: V9 CPU SIMD+ (5.83x on 1M)
    return aggregate::simd_sum_i32_v9(data, count);
} else {
    // 5M 以上: V7 GPU Metal (3.01x on 10M)
    return gpu::aggregate_sum_i32_metal(data, count);
}
```

### 3.4 Hash Join 自适应 (已实现)

基于匹配率估计选择算法:
- 低匹配率 (build < 200K, probe > build*3): V7 Adaptive
- 高匹配率: V11 SIMD

---

## 四、实现架构

```
┌─────────────────────────────────────────────────────────┐
│                    V12.5 统一入口                        │
│                 (v12_5_unified.cpp)                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Filter  │  │Aggregate│  │ GROUP BY│  │  TopK   │    │
│  │自适应   │  │自适应   │  │ V8固定  │  │ 直调    │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐    │
│  │<5M: GPU │  │<5M: V9  │  │V8 CPU   │  │<1M: V8  │    │
│  │≥5M: V3  │  │≥5M: V7  │  │4核并行  │  │≥1M: V7  │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │              Hash Join 自适应                    │    │
│  │  低匹配率: V7 Adaptive | 高匹配率: V11 SIMD     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 五、文件清单

| 文件 | 说明 |
|------|------|
| `src/core/v12_5_unified.cpp` | V12.5 统一路由实现 |
| `include/thunderduck/v12_5.h` | V12.5 公开接口 |
| `benchmark/v12_5_benchmark.cpp` | V12.5 基准测试 |
| `docs/V12_5_PERFORMANCE_DESIGN.md` | 本设计文档 |

---

## 六、预期性能目标

| 算子 | V12 当前 | V12.5 目标 | 提升 |
|------|----------|------------|------|
| Filter 1M | 7.54x | 7.54x | 保持 |
| Filter 10M | 2.70x | **3.02x** | +12% |
| Aggregate 1M | 5.77x | **5.83x** | +1% |
| Aggregate 10M | 2.85x | **3.01x** | +6% |
| GROUP BY 1M | 4.11x | **4.47x** | +9% |
| GROUP BY 10M | 2.07x | **2.32x** | +12% |
| TopK 1M | 8.97x | **13.36x** | +49% |
| TopK 10M | 4.96x | **5.12x** | +3% |
| Hash Join | 1.72x | 1.72x | 保持 |

---

## 七、验证计划

```bash
# 编译
make clean && make lib

# 运行 V12.5 基准测试
make v125-bench

# 对比 V12 vs V12.5 vs 各版本最优
```
