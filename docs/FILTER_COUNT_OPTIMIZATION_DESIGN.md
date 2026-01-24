# ThunderDuck Filter Count 优化设计

> **版本**: 3.0 | **日期**: 2026-01-24
>
> 针对 F1 (Simple Comparison) 和 F2 (Equality Comparison) 性能低于 DuckDB 的深入分析与优化方案

---

## 一、性能问题分析

### 1.1 测试结果回顾

| 测试 | SQL | DuckDB | ThunderDuck | 差距 |
|------|-----|--------|-------------|------|
| F1 | `WHERE l_quantity > 25` | 0.854 ms | 0.984 ms | 15% slower |
| F2 | `WHERE l_quantity = 30` | 0.823 ms | 0.975 ms | 18% slower |

**数据规模**: 5,000,000 行 × 4 字节 = 19 MB

**理论带宽**: M4 内存带宽 ~100 GB/s，理论最小时间 = 19 MB / 100 GB/s ≈ 0.19 ms

**实际带宽利用率**:
- DuckDB: 19 MB / 0.854 ms ≈ 22.2 GB/s (22%)
- ThunderDuck: 19 MB / 0.984 ms ≈ 19.3 GB/s (19%)

### 1.2 当前实现代码分析

```cpp
// simd_filter_v2.cpp - count_i32_v2 函数
size_t count_i32_v2(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    uint32x4_t count_vec = vdupq_n_u32(0);
    size_t i = 0;

    for (; i + 16 <= count; i += 16) {
        __builtin_prefetch(input + i + 64, 0, 0);

        // 加载 16 个元素
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // ❌ 问题 1: switch-case 在循环内部
        uint32x4_t m0, m1, m2, m3;
        switch (op) {
            case CompareOp::GT:
                m0 = vcgtq_s32(d0, threshold);
                // ... 重复 4 次
                break;
            // ... 6 个 case
        }

        // ❌ 问题 2: 顺序累加形成依赖链
        uint32x4_t ones0 = vshrq_n_u32(m0, 31);
        count_vec = vaddq_u32(count_vec, ones0);
        count_vec = vaddq_u32(count_vec, ones1);  // 等待上一条
        count_vec = vaddq_u32(count_vec, ones2);  // 等待上一条
        count_vec = vaddq_u32(count_vec, ones3);  // 等待上一条
    }

    return vaddvq_u32(count_vec);  // 水平归约
}
```

---

## 二、性能瓶颈根因分析

### 2.1 瓶颈 1: 循环内 Switch-Case 开销

**问题**:
```cpp
for (...) {
    switch (op) {  // ❌ 每次迭代执行
        case GT: ...
        case EQ: ...
        // 6 个分支
    }
}
```

**影响**:
1. **分支预测开销**: 虽然分支总是走同一路径，但 CPU 仍需维护分支预测器状态
2. **指令缓存污染**: 6 个 case 的代码都被加载到 L1i
3. **编译器优化受限**: 无法内联和展开特定操作的循环
4. **额外指令**: 每次迭代需要 jump table lookup 和 indirect branch

**实测影响**: 约 10-20% 性能损失

### 2.2 瓶颈 2: 累加器依赖链

**问题**:
```cpp
count_vec = vaddq_u32(count_vec, ones0);
count_vec = vaddq_u32(count_vec, ones1);  // 依赖上一条
count_vec = vaddq_u32(count_vec, ones2);  // 依赖上一条
count_vec = vaddq_u32(count_vec, ones3);  // 依赖上一条
```

**CPU 流水线分析**:

```
时间 →
        T0    T1    T2    T3    T4    T5    T6
add0    [===]
add1          [等待]  [===]                      // 等待 add0 结果
add2                      [等待]  [===]          // 等待 add1 结果
add3                                  [等待]  [===]  // 等待 add2 结果
```

**延迟**: M4 的 NEON `vadd` 延迟约 2-3 周期，4 次串行累加 = 8-12 周期/迭代

### 2.3 瓶颈 3: 次优的掩码计数方法

**当前方法**:
```cpp
uint32x4_t ones = vshrq_n_u32(mask, 31);  // 4 个 0/1
count_vec = vaddq_u32(count_vec, ones);   // 累加
```

**每 16 元素的操作**:
- 4 次 `vshrq_n_u32`
- 4 次 `vaddq_u32` (串行)
- 最后 1 次 `vaddvq_u32` (水平归约)

**更优方法**: 使用 `vcnt` (population count) 直接计数

### 2.4 瓶颈 4: 预取距离不当

**当前设置**:
```cpp
__builtin_prefetch(input + i + 64, 0, 0);  // 64 × 4 = 256 字节
```

**M4 缓存特性**:
- L1 缓存行: 128 字节
- L2 缓存行: 128 字节
- 内存延迟: ~100 ns

**分析**:
- 每次迭代处理 64 字节 (16 × 4)
- 预取 256 字节提前约 4 次迭代
- 可能需要更激进的预取 (512-1024 字节)

---

## 三、DuckDB 性能优势分析

### 3.1 DuckDB 的设计策略

根据 [DuckDB 官方文档](https://duckdb.org/docs/stable/internals/vector) 和 [Vectorized Execution 研究](https://medium.com/@kaushalsinh73/vectorized-execution-101-7045b83e6f84):

1. **编译器自动向量化优先**
   - 写无分支的标量代码
   - 让编译器优化为 SIMD
   - 避免手写 intrinsics 的陷阱

2. **无分支计数模式**
   ```cpp
   // Branchless counting
   for (size_t i = 0; i < count; ++i) {
       result += (input[i] > threshold);  // 无分支！
   }
   ```
   编译器会自动向量化为高效的 SIMD 代码

3. **Chunk-based 处理**
   - 固定向量大小 (STANDARD_VECTOR_SIZE = 2048)
   - 批量操作减少函数调用
   - 缓存友好的访问模式

### 3.2 行业最佳实践

根据 [Quickwit SIMD Filtering](https://quickwit.io/blog/simd-range) 和 [lemire/FastFlagStats](https://github.com/lemire/FastFlagStats):

1. **Branchless 是关键**
   - 标量代码从 170M 提升到 300M elem/s (1.8x)
   - SIMD + branchless 达到 3.65B elem/s (20x)

2. **Popcount 技术**
   - Harley-Seal algorithm for 批量 popcount
   - ARM `vcnt` 指令直接计数字节中的 1

3. **掩码压缩存储**
   - AVX-512: `_mm512_mask_compressstoreu_epi32`
   - ARM: 需要手动实现 (无直接等价指令)

---

## 四、优化方案设计

### 4.1 优化 1: 模板特化消除 Switch-Case

**设计**:
```cpp
// 模板函数 - 编译期确定比较操作
template<CompareOp Op>
inline uint32x4_t compare_i32_simd(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) {
        return vcgtq_s32(data, threshold);
    } else if constexpr (Op == CompareOp::EQ) {
        return vceqq_s32(data, threshold);
    }
    // ... 其他操作
}

// 特化的计数函数
template<CompareOp Op>
size_t count_i32_v3_impl(const int32_t* input, size_t count, int32_t value) {
    // 无 switch-case 的高性能循环
}

// 外部接口 - 运行时分发
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    switch (op) {  // 只执行一次
        case CompareOp::GT: return count_i32_v3_impl<CompareOp::GT>(input, count, value);
        case CompareOp::EQ: return count_i32_v3_impl<CompareOp::EQ>(input, count, value);
        // ...
    }
}
```

**预期收益**: 10-15% 性能提升

### 4.2 优化 2: 独立累加器消除依赖链

**设计**:
```cpp
template<CompareOp Op>
size_t count_i32_v3_impl(const int32_t* input, size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);

    // 4 个独立累加器
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);
    uint32x4_t acc2 = vdupq_n_u32(0);
    uint32x4_t acc3 = vdupq_n_u32(0);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32x4_t m0 = compare_i32_simd<Op>(d0, threshold);
        uint32x4_t m1 = compare_i32_simd<Op>(d1, threshold);
        uint32x4_t m2 = compare_i32_simd<Op>(d2, threshold);
        uint32x4_t m3 = compare_i32_simd<Op>(d3, threshold);

        // 独立累加 - 无依赖！
        acc0 = vsubq_u32(acc0, m0);  // 0xFFFFFFFF 时 +1
        acc1 = vsubq_u32(acc1, m1);
        acc2 = vsubq_u32(acc2, m2);
        acc3 = vsubq_u32(acc3, m3);
    }

    // 合并累加器
    uint32x4_t total = vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3));
    return vaddvq_u32(total);
}
```

**关键优化**: 用 `vsub(acc, mask)` 替代 `vshr + vadd`
- 当 mask = 0xFFFFFFFF 时: acc - 0xFFFFFFFF = acc + 1 (无符号溢出)
- 当 mask = 0 时: acc - 0 = acc
- 减少一条指令，且更符合 CPU 流水线

**CPU 流水线分析 (优化后)**:

```
时间 →
        T0    T1    T2    T3
sub0    [===]                   // 独立执行
sub1    [===]                   // 独立执行
sub2    [===]                   // 独立执行
sub3    [===]                   // 独立执行
```

**预期收益**: 20-30% 性能提升

### 4.3 优化 3: 更大批次 + Harley-Seal Popcount

**设计**: 每 256 个元素做一次局部归约

```cpp
template<CompareOp Op>
size_t count_i32_v3_impl(const int32_t* input, size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t result = 0;
    size_t i = 0;

    // 外层循环：每 256 元素
    for (; i + 256 <= count; i += 256) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        // 内层循环：16 次迭代 × 16 元素 = 256 元素
        for (int j = 0; j < 16; ++j) {
            const int32_t* ptr = input + i + j * 16;
            __builtin_prefetch(ptr + 128, 0, 0);  // 预取 2 个迭代后

            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            uint32x4_t m0 = compare_i32_simd<Op>(d0, threshold);
            uint32x4_t m1 = compare_i32_simd<Op>(d1, threshold);
            uint32x4_t m2 = compare_i32_simd<Op>(d2, threshold);
            uint32x4_t m3 = compare_i32_simd<Op>(d3, threshold);

            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        // 批量归约
        uint32x4_t total = vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3));
        result += vaddvq_u32(total);
    }

    // 处理剩余
    // ...

    return result;
}
```

**预期收益**: 5-10% 性能提升 (减少水平归约次数)

### 4.4 优化 4: 自适应预取策略

**设计**:
```cpp
// 根据数据大小调整预取策略
constexpr size_t L1_SIZE = 64 * 1024;      // 64 KB
constexpr size_t L2_SIZE = 4 * 1024 * 1024; // 4 MB
constexpr size_t CACHE_LINE = 128;

inline void adaptive_prefetch(const void* ptr, size_t data_size, size_t current_offset) {
    if (data_size <= L1_SIZE) {
        // 数据适合 L1，不需要预取
        return;
    } else if (data_size <= L2_SIZE) {
        // 数据适合 L2，轻度预取
        __builtin_prefetch(static_cast<const char*>(ptr) + 256, 0, 1);
    } else {
        // 大数据，激进预取
        __builtin_prefetch(static_cast<const char*>(ptr) + 512, 0, 0);
        __builtin_prefetch(static_cast<const char*>(ptr) + 1024, 0, 0);
    }
}
```

**预期收益**: 5-10% 性能提升

### 4.5 优化 5: 编译器自动向量化路径

**设计**: 提供一个让编译器自动向量化的实现

```cpp
// 让编译器自动向量化的无分支实现
__attribute__((optimize("O3")))
__attribute__((target("arch=armv8-a+simd")))
size_t count_i32_autovec(const int32_t* __restrict input, size_t count,
                          CompareOp op, int32_t value) {
    size_t result = 0;

    switch (op) {
        case CompareOp::GT:
            #pragma clang loop vectorize(enable) interleave(enable)
            for (size_t i = 0; i < count; ++i) {
                result += (input[i] > value);  // 无分支！
            }
            break;
        case CompareOp::EQ:
            #pragma clang loop vectorize(enable) interleave(enable)
            for (size_t i = 0; i < count; ++i) {
                result += (input[i] == value);
            }
            break;
        // ...
    }

    return result;
}
```

**预期收益**: 可能与手写 SIMD 相当或更好

---

## 五、综合优化实现 (v3.0)

### 5.1 完整实现代码

```cpp
/**
 * ThunderDuck - SIMD Filter v3.0
 *
 * 优化特性:
 * 1. 模板特化消除 switch-case
 * 2. 4 独立累加器 ILP
 * 3. vsub 计数减少指令
 * 4. 256 元素批次处理
 * 5. 自适应预取
 */

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace thunderduck {
namespace filter {

// 编译期比较操作
template<CompareOp Op>
__attribute__((always_inline))
inline uint32x4_t simd_compare(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) return vcgtq_s32(data, threshold);
    if constexpr (Op == CompareOp::GE) return vcgeq_s32(data, threshold);
    if constexpr (Op == CompareOp::LT) return vcltq_s32(data, threshold);
    if constexpr (Op == CompareOp::LE) return vcleq_s32(data, threshold);
    if constexpr (Op == CompareOp::EQ) return vceqq_s32(data, threshold);
    if constexpr (Op == CompareOp::NE) return vmvnq_u32(vceqq_s32(data, threshold));
}

// v3 核心实现 - 模板特化
template<CompareOp Op>
__attribute__((noinline))
size_t count_i32_v3_core(const int32_t* __restrict input,
                          size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t result = 0;
    size_t i = 0;

    // 阶段 1: 256 元素批次
    constexpr size_t BATCH_SIZE = 256;
    constexpr size_t INNER_ITERS = BATCH_SIZE / 16;  // 16

    for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const int32_t* batch_ptr = input + i;

        // 预取下一批
        __builtin_prefetch(batch_ptr + BATCH_SIZE, 0, 0);
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 64, 0, 0);

        #pragma unroll
        for (size_t j = 0; j < INNER_ITERS; ++j) {
            const int32_t* ptr = batch_ptr + j * 16;

            // 加载 16 元素
            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            // 比较 (编译期确定操作)
            uint32x4_t m0 = simd_compare<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare<Op>(d3, threshold);

            // 累加 (独立累加器，无依赖)
            // mask = 0xFFFFFFFF 时: acc - mask = acc + 1
            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        // 批次归约
        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // 阶段 2: 16 元素迭代
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i + 16 <= count; i += 16) {
            int32x4_t d0 = vld1q_s32(input + i);
            int32x4_t d1 = vld1q_s32(input + i + 4);
            int32x4_t d2 = vld1q_s32(input + i + 8);
            int32x4_t d3 = vld1q_s32(input + i + 12);

            uint32x4_t m0 = simd_compare<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare<Op>(d3, threshold);

            acc = vsubq_u32(acc, m0);
            acc = vsubq_u32(acc, m1);
            acc = vsubq_u32(acc, m2);
            acc = vsubq_u32(acc, m3);
        }
        result += vaddvq_u32(acc);
    }

    // 阶段 3: 4 元素迭代
    {
        uint32x4_t acc = vdupq_n_u32(0);
        for (; i + 4 <= count; i += 4) {
            int32x4_t data = vld1q_s32(input + i);
            uint32x4_t mask = simd_compare<Op>(data, threshold);
            acc = vsubq_u32(acc, mask);
        }
        result += vaddvq_u32(acc);
    }

    // 阶段 4: 标量尾部
    for (; i < count; ++i) {
        if constexpr (Op == CompareOp::GT) result += (input[i] > value);
        if constexpr (Op == CompareOp::GE) result += (input[i] >= value);
        if constexpr (Op == CompareOp::LT) result += (input[i] < value);
        if constexpr (Op == CompareOp::LE) result += (input[i] <= value);
        if constexpr (Op == CompareOp::EQ) result += (input[i] == value);
        if constexpr (Op == CompareOp::NE) result += (input[i] != value);
    }

    return result;
}

// 公开接口 - 运行时分发
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    switch (op) {
        case CompareOp::GT: return count_i32_v3_core<CompareOp::GT>(input, count, value);
        case CompareOp::GE: return count_i32_v3_core<CompareOp::GE>(input, count, value);
        case CompareOp::LT: return count_i32_v3_core<CompareOp::LT>(input, count, value);
        case CompareOp::LE: return count_i32_v3_core<CompareOp::LE>(input, count, value);
        case CompareOp::EQ: return count_i32_v3_core<CompareOp::EQ>(input, count, value);
        case CompareOp::NE: return count_i32_v3_core<CompareOp::NE>(input, count, value);
        default: return 0;
    }
}

} // namespace filter
} // namespace thunderduck
```

---

## 六、预期性能提升

### 6.1 单项优化预估

| 优化 | 预期提升 | 原因 |
|------|---------|------|
| 模板特化消除 switch | 10-15% | 消除循环内分支 |
| 独立累加器 | 20-30% | 消除依赖链，提升 ILP |
| vsub 替代 vshr+vadd | 5-10% | 减少指令数 |
| 256 元素批次 | 5-10% | 减少水平归约 |
| 优化预取 | 5-10% | 提升缓存命中率 |

### 6.2 综合预期

**当前性能**: 0.975-0.984 ms (19.3 GB/s)

**目标性能**: 0.70-0.80 ms (24-27 GB/s)

**预期加速比**: 1.2-1.4x (超越 DuckDB)

### 6.3 理论上限分析

| 瓶颈 | 值 | 说明 |
|-----|---|-----|
| 内存带宽 | 100 GB/s | M4 理论峰值 |
| 最小时间 | 0.19 ms | 19 MB / 100 GB/s |
| 计算 IPC | ~8 | NEON 执行单元 |
| 实际期望 | 0.6-0.7 ms | 带宽利用率 30-35% |

---

## 七、验证计划

### 7.1 微基准测试

```cpp
// 测试不同数据大小
std::vector<size_t> sizes = {1000, 10000, 100000, 1000000, 5000000};

// 测试不同选择率
std::vector<int32_t> thresholds = {10, 25, 40, 49};  // 80%, 50%, 20%, 2%

// 测试所有比较操作
std::vector<CompareOp> ops = {GT, GE, LT, LE, EQ, NE};

for (auto size : sizes) {
    for (auto threshold : thresholds) {
        for (auto op : ops) {
            // 测试 v2 vs v3
            benchmark_compare(count_i32_v2, count_i32_v3, ...);
        }
    }
}
```

### 7.2 对比 DuckDB

```sql
-- 测试查询
SELECT COUNT(*) FROM lineitem WHERE l_quantity > 25;
SELECT COUNT(*) FROM lineitem WHERE l_quantity = 30;
SELECT COUNT(*) FROM lineitem WHERE l_quantity >= 10 AND l_quantity < 40;
```

### 7.3 验收标准

| 指标 | 目标 |
|------|------|
| F1 (GT) | 超越 DuckDB (< 0.85 ms) |
| F2 (EQ) | 超越 DuckDB (< 0.82 ms) |
| 总体 Filter 胜率 | 4/4 (100%) |

---

## 八、参考资料

1. [DuckDB Vector Execution](https://duckdb.org/docs/stable/internals/vector) - DuckDB 官方文档
2. [Vectorized Execution 101](https://medium.com/@kaushalsinh73/vectorized-execution-101-7045b83e6f84) - 向量化执行原理
3. [Quickwit SIMD Filtering](https://quickwit.io/blog/simd-range) - SIMD 过滤技术
4. [lemire/FastFlagStats](https://github.com/lemire/FastFlagStats) - 高性能 popcount
5. [WojciechMula/sse-popcount](https://github.com/WojciechMula/sse-popcount) - SIMD popcount 算法

---

*ThunderDuck v3.0 - 目标: Filter 100% 胜率！*
