# ThunderDuck TopK 自适应优化设计

> **版本**: 3.0 | **日期**: 2026-01-24
>
> 针对不同 K 值场景的深度分析与自适应优化方案

---

## 一、当前性能状态

### 1.1 测试结果 (Medium 数据集: 500K orders)

| 测试 | K 值 | DuckDB | ThunderDuck v2 | 加速比 | 胜者 |
|------|------|--------|----------------|--------|------|
| T1 | 10 | 0.696 ms | 0.236 ms | **2.95x** | ThunderDuck |
| T2 | 100 | 0.874 ms | 0.250 ms | **3.49x** | ThunderDuck |
| T3 | 1000 | 1.349 ms | 0.526 ms | **2.56x** | ThunderDuck |

**当前胜率**: 100% (3/3)

**观察**: K 值增大时，加速比下降 (3.49x → 2.56x)，说明大 K 值场景有优化空间。

### 1.2 当前实现分析

```cpp
// radix_sort.cpp - topk_max_i32_v2
void topk_max_i32_v2(const int32_t* data, size_t count, size_t k, ...) {
    if (k <= 100) {
        // 策略 1: 堆方法 - O(n log k)
        // 适合小 K，内存高效
        std::vector<std::pair<int32_t, uint32_t>> heap;
        for (size_t i = 0; i < count; ++i) {
            if (heap.size() < k) {
                heap.push_back({data[i], i});
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else if (data[i] > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                heap.back() = {data[i], i};
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
        std::sort(heap.begin(), heap.end(), ...);
    } else {
        // 策略 2: nth_element - O(n) + O(k log k)
        // 适合大 K，但需要复制全部数据
        std::vector<std::pair<int32_t, uint32_t>> pairs(count);
        for (size_t i = 0; i < count; ++i) {
            pairs[i] = {data[i], i};
        }
        std::nth_element(pairs.begin(), pairs.begin() + k, pairs.end(), ...);
        std::sort(pairs.begin(), pairs.begin() + k, ...);
    }
}
```

### 1.3 问题分析

| K 范围 | 当前策略 | 问题 |
|--------|----------|------|
| K ≤ 100 | 最小堆 | ✅ 高效 |
| K > 100 | nth_element | ⚠️ 需要复制全部数据 |
| K > 1000 | nth_element | ❌ 复制开销显著 |

**关键瓶颈**:
1. **数据复制开销**: `std::vector<std::pair<...>> pairs(count)` 需要 O(n) 内存和复制
2. **pair 结构开销**: 每个元素需要额外存储索引，内存带宽加倍
3. **策略阈值固定**: K=100 的阈值可能不是最优的

---

## 二、算法复杂度分析

### 2.1 各算法时间复杂度

| 算法 | 时间复杂度 | 空间复杂度 | 最适合场景 |
|------|------------|------------|------------|
| **堆方法** | O(n log k) | O(k) | 小 K (K << √n) |
| **快速选择** | O(n) 期望 | O(1) 原地 | 中等 K |
| **nth_element** | O(n) | O(1) 原地 | 大 K (只要值) |
| **Introselect** | O(n) 最坏 | O(log n) 栈 | 各种 K |
| **部分排序** | O(n + k log k) | O(1) | K 接近 n |

### 2.2 M4 平台特性考量

| 特性 | 影响 |
|------|------|
| L1 Cache: 64 KB | 堆大小 < 8K 元素时可常驻 L1 |
| L2 Cache: 4 MB | 数据 < 1M 元素时可常驻 L2 |
| 缓存行: 128 bytes | 连续访问比随机访问快 10x |
| 内存带宽: 100 GB/s | 数据复制瓶颈明显 |
| 分支预测 | 堆操作的条件分支可预测 |

### 2.3 最优 K 阈值理论分析

设 n 为数据量，k 为选择数量:

**堆方法**: T_heap = n × (log k) × C_compare
**nth_element**: T_nth = n × C_partition + k × (log k) × C_sort

交叉点: n × log k = n + k × log k
解得: k ≈ n / (log n - 1) (粗略估计)

对于 n = 500,000: k_crossover ≈ 25,000

**实际测试建议的阈值**:
- K ≤ 64: 堆方法 (可常驻 L1)
- 64 < K ≤ 1024: 优化堆 + 批量处理
- 1024 < K ≤ 10000: nth_element (无复制版本)
- K > 10000: 部分排序

---

## 三、v3.0 优化策略

### 3.1 策略 1: 无复制 nth_element

**问题**: 当前实现需要复制整个数组到 pairs 向量

**解决方案**: 使用索引数组而非完整复制

```cpp
void topk_max_i32_v3_nth(const int32_t* data, size_t count, size_t k,
                          int32_t* out_values, uint32_t* out_indices) {
    // 只分配索引数组 (4 bytes/元素 vs 12 bytes/元素)
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    // 基于间接比较的 nth_element
    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    // 只排序前 k 个索引
    std::sort(indices.begin(), indices.begin() + k,
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    // 输出结果
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}
```

**预期收益**:
- 内存使用: 从 12n bytes 降到 4n bytes (67% 减少)
- 内存带宽: 降低 ~50%

### 3.2 策略 2: SIMD 加速堆操作

**问题**: 堆的 push/pop 操作是标量的

**解决方案**: 批量处理 + SIMD 比较

```cpp
// SIMD 批量比较找到需要入堆的元素
void topk_max_i32_v3_simd_heap(const int32_t* data, size_t count, size_t k,
                                int32_t* out_values, uint32_t* out_indices) {
    // 初始化堆
    std::vector<std::pair<int32_t, uint32_t>> heap;
    heap.reserve(k);

    // 填充初始 k 个元素
    for (size_t i = 0; i < k && i < count; ++i) {
        heap.push_back({data[i], static_cast<uint32_t>(i)});
    }
    std::make_heap(heap.begin(), heap.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    int32_t threshold = heap.front().first;

    // SIMD 批量处理剩余元素
    size_t i = k;

#ifdef __aarch64__
    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    for (; i + 16 <= count; i += 16) {
        // 加载 16 个元素
        int32x4_t d0 = vld1q_s32(data + i);
        int32x4_t d1 = vld1q_s32(data + i + 4);
        int32x4_t d2 = vld1q_s32(data + i + 8);
        int32x4_t d3 = vld1q_s32(data + i + 12);

        // 比较是否大于阈值
        uint32x4_t m0 = vcgtq_s32(d0, thresh_vec);
        uint32x4_t m1 = vcgtq_s32(d1, thresh_vec);
        uint32x4_t m2 = vcgtq_s32(d2, thresh_vec);
        uint32x4_t m3 = vcgtq_s32(d3, thresh_vec);

        // 如果有任何元素大于阈值，进行详细处理
        uint32x4_t any = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
        if (vmaxvq_u32(any)) {
            // 标量处理需要入堆的元素
            for (size_t j = 0; j < 16; ++j) {
                if (data[i + j] > threshold) {
                    std::pop_heap(heap.begin(), heap.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                    heap.back() = {data[i + j], static_cast<uint32_t>(i + j)};
                    std::push_heap(heap.begin(), heap.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                    threshold = heap.front().first;
                }
            }
            thresh_vec = vdupq_n_s32(threshold);
        }
    }
#endif

    // 标量处理剩余
    for (; i < count; ++i) {
        if (data[i] > threshold) {
            std::pop_heap(heap.begin(), heap.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            heap.back() = {data[i], static_cast<uint32_t>(i)};
            std::push_heap(heap.begin(), heap.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            threshold = heap.front().first;
        }
    }

    // 排序输出
    std::sort(heap.begin(), heap.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t j = 0; j < k; ++j) {
        out_values[j] = heap[j].first;
        if (out_indices) out_indices[j] = heap[j].second;
    }
}
```

**预期收益**:
- SIMD 批量比较跳过大部分不需要入堆的元素
- 对于低选择率 (k << n)，大幅减少堆操作次数

### 3.3 策略 3: 分块 + 局部 TopK

**问题**: 大数据集单次遍历缓存不友好

**解决方案**: 分块处理，每块找 TopK，再合并

```cpp
void topk_max_i32_v3_blocked(const int32_t* data, size_t count, size_t k,
                              int32_t* out_values, uint32_t* out_indices) {
    constexpr size_t BLOCK_SIZE = 64 * 1024;  // 64K 元素 ~= 256KB，适合 L2

    if (count <= BLOCK_SIZE || k > count / 4) {
        // 小数据或大 K，直接处理
        topk_max_i32_v3_simd_heap(data, count, k, out_values, out_indices);
        return;
    }

    // 计算块数
    size_t num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t k_per_block = std::min(k * 2, BLOCK_SIZE);  // 每块保留 2k 个候选

    // 收集每块的候选
    std::vector<std::pair<int32_t, uint32_t>> candidates;
    candidates.reserve(num_blocks * k_per_block);

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * BLOCK_SIZE;
        size_t end = std::min(start + BLOCK_SIZE, count);
        size_t block_count = end - start;
        size_t block_k = std::min(k_per_block, block_count);

        // 块内 TopK
        std::vector<int32_t> block_values(block_k);
        std::vector<uint32_t> block_indices(block_k);
        topk_max_i32_v3_simd_heap(data + start, block_count, block_k,
                                   block_values.data(), block_indices.data());

        // 调整索引为全局索引
        for (size_t i = 0; i < block_k; ++i) {
            candidates.push_back({block_values[i],
                                  static_cast<uint32_t>(start + block_indices[i])});
        }
    }

    // 从候选中选择最终 TopK
    std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::sort(candidates.begin(), candidates.begin() + k,
        [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < k; ++i) {
        out_values[i] = candidates[i].first;
        if (out_indices) out_indices[i] = candidates[i].second;
    }
}
```

**预期收益**:
- 每块数据完全在 L2 缓存中
- 减少缓存 miss，提高内存带宽利用率

### 3.4 策略 4: 自适应 K 阈值

**完整的自适应策略**:

```cpp
void topk_max_i32_v3(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);

    // 根据 K 和 N 选择最优策略
    double k_ratio = static_cast<double>(k) / count;

    if (k <= 64) {
        // 策略 A: 纯堆方法 - K 很小，堆可常驻 L1
        topk_heap_small(data, count, k, out_values, out_indices);
    }
    else if (k <= 1024 && k_ratio < 0.01) {
        // 策略 B: SIMD 加速堆 - K 中等，但选择率低
        topk_simd_heap(data, count, k, out_values, out_indices);
    }
    else if (k <= 4096 || k_ratio < 0.1) {
        // 策略 C: 分块 + 局部 TopK - 大 K，缓存优化
        topk_blocked(data, count, k, out_values, out_indices);
    }
    else {
        // 策略 D: 无复制 nth_element - K 很大，直接选择
        topk_nth_element(data, count, k, out_values, out_indices);
    }
}
```

---

## 四、预期性能提升

### 4.1 各策略预期收益

| 策略 | 适用场景 | 预期收益 | 复杂度 |
|------|----------|----------|--------|
| 无复制 nth_element | K > 1000 | +20-30% | 低 |
| SIMD 加速堆 | K < 1000, 低选择率 | +30-50% | 中 |
| 分块处理 | 大数据集 | +20-40% | 中 |
| 自适应选择 | 所有场景 | +10-20% | 低 |

### 4.2 综合预期

| 测试 | v2 当前 | v3 目标 | 目标加速 |
|------|---------|---------|----------|
| Top-10 | 2.95x | **4.5x+** | +52% |
| Top-100 | 3.49x | **5.0x+** | +43% |
| Top-1000 | 2.56x | **4.0x+** | +56% |

**总体目标**: TopK 平均加速从 3.0x 提升到 **4.5x+**

---

## 五、实现计划

### 5.1 Phase 1: 无复制优化 (复杂度: 低)

1. 实现 `topk_max_i32_v3_nth` - 使用索引数组替代 pair 数组
2. 更新 K > 1000 的分支使用新实现

**预期收益**: Top-1000 从 2.56x → 3.2x

### 5.2 Phase 2: SIMD 加速堆 (复杂度: 中)

1. 实现 `topk_max_i32_v3_simd_heap`
2. SIMD 批量比较 + 阈值更新
3. 更新 K ≤ 1000 的分支

**预期收益**: Top-100 从 3.49x → 4.5x

### 5.3 Phase 3: 分块优化 (复杂度: 中)

1. 实现 `topk_max_i32_v3_blocked`
2. 确定最优块大小 (基于 L2 缓存)
3. 多块结果合并优化

**预期收益**: 大数据集 +20-40%

### 5.4 Phase 4: 自适应选择 (复杂度: 低)

1. 实现统一入口 `topk_max_i32_v3`
2. 根据 K/N 比例动态选择策略
3. 性能回归测试

---

## 六、验证计划

### 6.1 正确性测试

```cpp
// 测试不同 K 值
std::vector<size_t> test_k = {1, 10, 100, 1000, 10000};
std::vector<size_t> test_n = {10000, 100000, 1000000, 5000000};

for (auto n : test_n) {
    auto data = generate_random_data(n);
    for (auto k : test_k) {
        auto expected = reference_topk(data, k);
        auto actual = topk_max_i32_v3(data, k);
        ASSERT_EQ(expected, actual);
    }
}
```

### 6.2 性能基准

```cpp
// 测试矩阵
// N: 100K, 500K, 1M, 5M
// K: 10, 100, 1000, 10000

// 比较 v2 vs v3 vs DuckDB
benchmark_topk("v2", topk_max_i32_v2);
benchmark_topk("v3", topk_max_i32_v3);
benchmark_duckdb_topk();
```

### 6.3 验收标准

| 指标 | 目标 |
|------|------|
| 正确性 | 100% 通过 |
| Top-10 加速 | ≥ 4.5x vs DuckDB |
| Top-100 加速 | ≥ 5.0x vs DuckDB |
| Top-1000 加速 | ≥ 4.0x vs DuckDB |
| 内存使用 | ≤ v2 的 50% |

---

## 七、参考资料

1. [std::nth_element 实现分析](https://en.cppreference.com/w/cpp/algorithm/nth_element)
2. [Introselect 算法](https://en.wikipedia.org/wiki/Introselect)
3. [Heap vs QuickSelect 性能对比](https://www.geeksforgeeks.org/kth-smallest-largest-element-using-stl/)
4. [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
5. [ARM Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

---

*ThunderDuck v3.0 TopK - 目标: 4.5x+ 平均加速！*
