# ThunderDuck 性能优化方案

> **目标**: 全面超越 DuckDB 性能
> **版本**: 3.0
> **日期**: 2026-01-24

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0 | 2026-01-24 | 初始设计 |
| v2.0 | 2026-01-24 | Radix Sort, 合并 minmax, Robin Hood Hash |
| v3.0 | 2026-01-24 | Filter Count 深度优化，目标 100% 胜率 |

---

## 一、当前性能状态 (v2.0)

### 1.1 Benchmark 结果 (2026-01-24)

| 测试类别 | 测试数 | 胜出数 | 胜率 | 平均加速比 |
|----------|--------|--------|------|-----------|
| Aggregation | 4 | 4 | **100%** | 6,035x |
| Sort | 2 | 2 | **100%** | 4.9x |
| TopK | 3 | 2 | 66% | 1.96x |
| Filter | 4 | 2 | 50% | 1.06x |
| Join | 1 | 0 | 0% | 0.08x |
| **总计** | **14** | **10** | **71%** | - |

### 1.2 待优化项

| 测试 | 当前结果 | 差距 | 根因 |
|------|---------|------|------|
| F1: GT | 0.984 ms (DuckDB 0.854 ms) | 15% slower | 循环内 switch-case |
| F2: EQ | 0.975 ms (DuckDB 0.823 ms) | 18% slower | 累加器依赖链 |
| T3: Top-1000 | 3.154 ms (DuckDB 2.092 ms) | 51% slower | K 值过大时策略不优 |
| J1: Hash Join | 18.227 ms (DuckDB 1.408 ms) | 12.9x slower | 哈希表实现效率低 |

---

## 二、v3.0 Filter 优化方案

### 2.1 性能瓶颈深度分析

详见 [FILTER_COUNT_OPTIMIZATION_DESIGN.md](./FILTER_COUNT_OPTIMIZATION_DESIGN.md)

**瓶颈根因**:

| # | 瓶颈 | 影响 | 原因 |
|---|------|------|------|
| 1 | **循环内 Switch-Case** | 10-20% | 每次迭代执行分支判断，阻止编译器优化 |
| 2 | **累加器依赖链** | 20-30% | `count_vec = vaddq(count_vec, ...)` 串行等待 |
| 3 | **掩码计数方法** | 5-10% | `vshr + vadd` 两条指令 |
| 4 | **预取距离不当** | 5-10% | 当前预取 256 字节可能不足 |

### 2.2 优化策略

#### 策略 1: 模板特化消除 Switch-Case

```cpp
// Before (v2): switch 在循环内
for (...) {
    switch (op) {  // 每次迭代执行！
        case GT: ...
        case EQ: ...
    }
}

// After (v3): switch 在循环外，模板特化
template<CompareOp Op>
size_t count_i32_v3_core(...) {
    for (...) {
        // 编译期确定比较操作，无分支
        uint32x4_t m = simd_compare<Op>(data, threshold);
    }
}

size_t count_i32_v3(..., CompareOp op, ...) {
    switch (op) {  // 只执行一次
        case GT: return count_i32_v3_core<GT>(...);
        case EQ: return count_i32_v3_core<EQ>(...);
    }
}
```

**预期收益**: 10-15%

#### 策略 2: 独立累加器消除依赖链

```cpp
// Before (v2): 串行依赖
count_vec = vaddq(count_vec, ones0);  //
count_vec = vaddq(count_vec, ones1);  // 等待上一条
count_vec = vaddq(count_vec, ones2);  // 等待上一条
count_vec = vaddq(count_vec, ones3);  // 等待上一条

// After (v3): 4 个独立累加器
acc0 = vsubq(acc0, m0);  // 独立执行
acc1 = vsubq(acc1, m1);  // 独立执行
acc2 = vsubq(acc2, m2);  // 独立执行
acc3 = vsubq(acc3, m3);  // 独立执行
// 最后合并: total = acc0 + acc1 + acc2 + acc3
```

**预期收益**: 20-30%

#### 策略 3: vsub 替代 vshr+vadd

```cpp
// Before (v2): 两条指令
uint32x4_t ones = vshrq_n_u32(mask, 31);  // 指令 1
count_vec = vaddq_u32(count_vec, ones);   // 指令 2

// After (v3): 一条指令 (利用无符号溢出)
// mask = 0xFFFFFFFF 时: acc - mask = acc + 1
// mask = 0 时: acc - 0 = acc
acc = vsubq_u32(acc, mask);  // 单条指令！
```

**预期收益**: 5-10%

#### 策略 4: 256 元素批次处理

```cpp
// 外层: 256 元素批次
for (; i + 256 <= count; i += 256) {
    uint32x4_t acc0 = vdupq_n_u32(0);
    // ... acc1, acc2, acc3

    // 内层: 16 次迭代
    for (int j = 0; j < 16; ++j) {
        // 处理 16 元素
    }

    // 批次归约 (而非每次迭代归约)
    result += vaddvq_u32(total);
}
```

**预期收益**: 5-10%

#### 策略 5: 自适应预取

```cpp
// 根据数据大小调整预取距离
if (data_size <= L1_SIZE) {
    // 数据适合 L1，不需要预取
} else if (data_size <= L2_SIZE) {
    __builtin_prefetch(ptr + 256, 0, 1);  // L2 提示
} else {
    __builtin_prefetch(ptr + 512, 0, 0);  // 激进预取
    __builtin_prefetch(ptr + 1024, 0, 0);
}
```

**预期收益**: 5-10%

### 2.3 v3.0 预期性能

| 测试 | v2 当前 | v3 目标 | DuckDB | 目标差距 |
|------|---------|---------|--------|----------|
| F1 (GT) | 0.984 ms | <0.75 ms | 0.854 ms | **+14% 领先** |
| F2 (EQ) | 0.975 ms | <0.70 ms | 0.823 ms | **+17% 领先** |
| F3 (Range) | 0.699 ms | <0.60 ms | 1.014 ms | **+70% 领先** |
| F4 (High Sel) | 1.090 ms | <0.90 ms | 1.182 ms | **+31% 领先** |

**Filter 胜率目标**: 50% → **100%**

---

## 三、v2.0 已实现优化 (回顾)

### 3.1 Radix Sort (Sort 提升 5x)

```cpp
// LSD Radix Sort - O(n) 复杂度
void radix_sort_i32_v2(int32_t* data, size_t count, SortOrder order) {
    // 1. 翻转符号位处理有符号整数
    for (i = 0; i < count; ++i)
        data[i] = data[i] ^ 0x80000000;

    // 2. 11-11-10 位分组 (仅 3 趟)
    constexpr int BITS[3] = {11, 11, 10};
    for (pass = 0; pass < 3; ++pass) {
        // 计数排序
    }

    // 3. 翻转回有符号数
    for (i = 0; i < count; ++i)
        data[i] = data[i] ^ 0x80000000;
}
```

### 3.2 合并 minmax 函数 (MIN/MAX 提升 3.9x)

```cpp
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max) {
    int32x4_t min_vec = vdupq_n_s32(INT32_MAX);
    int32x4_t max_vec = vdupq_n_s32(INT32_MIN);

    for (; i + 16 <= count; i += 16) {
        // 单次遍历同时计算 MIN 和 MAX
        min_vec = vminq_s32(min_vec, batch_min);
        max_vec = vmaxq_s32(max_vec, batch_max);
    }
}
```

### 3.3 位图过滤 (Range Filter 提升 1.45x)

```cpp
size_t count_i32_range_v2(const int32_t* input, size_t count,
                           int32_t low, int32_t high) {
    // 单次比较完成范围检查
    uint32x4_t ge = vcgeq_s32(data, low_vec);
    uint32x4_t lt = vcltq_s32(data, high_vec);
    uint32x4_t mask = vandq_u32(ge, lt);
}
```

### 3.4 16 元素/迭代 + 预取 (Aggregation 提升 3-5x)

```cpp
for (; i + 16 <= count; i += 16) {
    __builtin_prefetch(input + i + 64, 0, 0);

    int32x4_t d0 = vld1q_s32(input + i);
    int32x4_t d1 = vld1q_s32(input + i + 4);
    int32x4_t d2 = vld1q_s32(input + i + 8);
    int32x4_t d3 = vld1q_s32(input + i + 12);

    // 处理 16 元素
}
```

---

## 四、后续优化方向

### 4.1 Join 优化 (优先级: 高)

**当前问题**: 12.9x slower

**优化方向**:
1. **分区 Hash Join** - 提升缓存局部性
2. **更激进的 SIMD 探测** - 批量键比较
3. **SOA 内存布局** - 键和值分离存储
4. **Perfect Hashing** - 针对小表优化

### 4.2 TopK 优化 (优先级: 中)

**当前问题**: T3 (Top-1000) 1.51x slower

**优化方向**:
1. **K 值自适应策略**
   - K <= 64: 最小堆
   - K <= 256: 堆 + partial_sort
   - K > 256: nth_element + partial_sort
2. **SIMD 堆操作** - 向量化堆维护

### 4.3 多核并行 (优先级: 低)

**潜力**: 4-10x (M4 有 10 核)

**方向**:
1. **并行 Radix Sort** - 分区并行
2. **并行聚合** - 分区 + 合并
3. **并行 Filter** - 分块处理

---

## 五、实现优先级

### Phase 3.1: Filter v3.0 (1-2 天)

| 任务 | 预期收益 | 复杂度 |
|-----|---------|--------|
| 模板特化消除 switch | +15% | 低 |
| 4 独立累加器 | +25% | 低 |
| vsub 替代 vshr+vadd | +8% | 低 |
| 256 元素批次 | +8% | 中 |
| 自适应预取 | +8% | 中 |

**综合预期**: Filter 1.2-1.4x 提升，超越 DuckDB

### Phase 3.2: TopK 优化 (1 天)

| 任务 | 预期收益 |
|-----|---------|
| K 值自适应策略 | T3 从 1.51x slower → 1.2x faster |

### Phase 3.3: Join 优化 (3-5 天)

| 任务 | 预期收益 |
|-----|---------|
| 分区 Hash Join | 2-3x |
| SIMD 批量探测 | 1.5x |
| SOA 布局 | 1.3x |

---

## 六、验收标准

### v3.0 目标

| 指标 | v2.0 当前 | v3.0 目标 |
|------|-----------|-----------|
| 总测试数 | 14 | 14 |
| ThunderDuck 胜出 | 10 (71%) | **12+ (86%+)** |
| Filter 胜率 | 2/4 (50%) | **4/4 (100%)** |
| TopK 胜率 | 2/3 (66%) | **3/3 (100%)** |
| Join 胜率 | 0/1 (0%) | 0/1 (暂不处理) |

### 长期目标

| 指标 | 目标 |
|------|------|
| 总体胜率 | **100%** |
| 平均加速比 | **5x+** |
| Join 胜率 | **100%** |

---

## 七、参考资料

1. [DuckDB Vector Execution](https://duckdb.org/docs/stable/internals/vector)
2. [Quickwit SIMD Filtering](https://quickwit.io/blog/simd-range)
3. [lemire/FastFlagStats](https://github.com/lemire/FastFlagStats)
4. [WojciechMula/sse-popcount](https://github.com/WojciechMula/sse-popcount)
5. [ARM Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

---

*ThunderDuck v3.0 - 目标: 全面超越 DuckDB！*
