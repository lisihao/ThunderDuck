# TopK 基数敏感性分析报告

> **日期**: 2026-01-24
> **测试场景**: 10M 行, K=10 (T4 场景)
> **目标**: 找出 ThunderDuck 反超 DuckDB 的基数临界点

---

## 一、测试结果

### 1.1 性能对比表

| 基数 | DuckDB (ms) | ThunderDuck (ms) | 加速比 | 胜者 |
|------|-------------|------------------|--------|------|
| 10 | 1.94 | 14.16 | 0.14x | DuckDB |
| 20 | 1.81 | 12.34 | 0.15x | DuckDB |
| 50 | 1.83 | 4.94 | 0.37x | DuckDB |
| 100 | 1.73 | 3.24 | 0.54x | DuckDB |
| 200 | 1.66 | 2.04 | 0.82x | DuckDB |
| **500** | 1.74 | 1.11 | **1.58x** | **Thunder** ← 临界点 |
| 1,000 | 2.01 | 0.84 | 2.40x | Thunder |
| 2,000 | 2.26 | 0.66 | 3.42x | Thunder |
| 5,000 | 2.40 | 0.75 | 3.22x | Thunder |
| 10,000 | 2.33 | 0.64 | 3.63x | Thunder |
| 50,000 | 2.39 | 0.61 | 3.91x | Thunder |
| 100,000 | 2.90 | 0.61 | 4.76x | Thunder |
| 1,000,000 | 2.85 | 0.59 | 4.83x | Thunder |
| 10,000,000 | 2.78 | 0.58 | 4.83x | Thunder |

### 1.2 性能曲线

```
加速比 (ThunderDuck / DuckDB)
    ^
5.0 |                                    ████████████████
    |                               █████
4.0 |                          █████
    |                     █████
3.0 |                █████
    |           █████
2.0 |       ████
    |   ████
1.0 +---████--------------------------------------------> 基数
    |  █ ↑
0.5 | █  临界点: 500
    |█
0.0 └──────────────────────────────────────────────────
    10  50  200 500 1K  5K  10K 50K 100K 500K 1M  10M
```

---

## 二、临界点分析

### 2.1 关键发现

```
┌─────────────────────────────────────────────────────────────┐
│                    临界基数: 500                            │
├─────────────────────────────────────────────────────────────┤
│  唯一值比例: 0.005% (500 / 10,000,000)                      │
│  平均重复次数: 20,000 次                                     │
│  加速比: 1.58x (开始反超)                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 原因分析

#### 低基数时 ThunderDuck 较慢的原因

1. **采样预过滤失效**
   - 采样 8192 个元素，但只有 ~500 个不同值
   - 阈值估计不准确，过多元素通过过滤
   - 候选数量 > 5%，触发堆方法回退

2. **堆方法的局限**
   - 对于 10M 元素，堆方法需要 10M 次比较
   - 每次比较都可能触发堆调整
   - 内存访问模式不连续

3. **DuckDB 的优势**
   - 可能使用了基于哈希的 TopK 算法
   - 对低基数数据有特殊优化
   - 查询优化器选择最优执行计划

#### 高基数时 ThunderDuck 较快的原因

1. **采样预过滤有效**
   - 阈值估计准确，~99% 的元素被过滤
   - SIMD 批量跳过，大幅减少比较次数
   - 只有少量候选需要进一步处理

2. **内存效率**
   - 候选数量小，内存占用低
   - nth_element 在小数据集上高效
   - 缓存友好的访问模式

---

## 三、基数分布分析

### 3.1 性能分区

| 基数范围 | 平均加速比 | 策略效果 | 建议 |
|----------|-----------|---------|------|
| < 200 | 0.15-0.54x | 采样失效 | 使用 DuckDB |
| 200-500 | 0.82-1.58x | 临界区 | 可选 |
| 500-5000 | 1.58-3.42x | 采样部分有效 | 推荐 Thunder |
| > 5000 | 3.63-4.83x | 采样高效 | 强烈推荐 Thunder |

### 3.2 实际应用场景映射

| 场景 | 典型基数 | 预期加速比 | 建议 |
|------|---------|-----------|------|
| 性别字段 TopK | 2-3 | ~0.1x | DuckDB |
| 状态码 TopK | 10-50 | 0.1-0.4x | DuckDB |
| 地区 TopK | 50-500 | 0.4-1.6x | 临界 |
| 用户等级 TopK | 10-100 | 0.1-0.5x | DuckDB |
| 商品类别 TopK | 100-1000 | 0.5-2.4x | 临界/Thunder |
| 商品ID TopK | 10K-1M | 3.6-4.8x | **Thunder** |
| 用户ID TopK | 1M-100M | 4.5-4.8x | **Thunder** |
| 订单ID TopK | 1M-1B | 4.5-4.8x | **Thunder** |
| 时间戳 TopK | 连续值 | 4.5-4.8x | **Thunder** |
| 价格 TopK | 连续值 | 4.5-4.8x | **Thunder** |

---

## 四、优化策略

### 4.1 当前 v4 策略

```cpp
if (count >= 1M && k <= 64) {
    // 采样预过滤
    threshold = estimate_threshold(data, count, k);
    candidates = collect_candidates_simd(data, count, threshold);

    if (candidates.size() > count / 20) {  // > 5%
        // 低基数检测，回退到堆方法
        return topk_heap(data, count, k);
    }
    // 继续使用采样预过滤结果
}
```

### 4.2 可能的优化方向

#### 方案 A: 自适应基数检测

```cpp
// 在采样阶段检测基数
size_t unique_count = count_unique_in_sample(samples);
double estimated_cardinality = unique_count * count / sample_size;

if (estimated_cardinality < 500) {
    // 低基数，使用 DuckDB 风格的哈希 TopK
    return topk_hash_based(data, count, k);
}
```

#### 方案 B: 哈希 TopK 算法

```cpp
// 对低基数数据使用哈希计数
std::unordered_map<int32_t, size_t> value_counts;
for (size_t i = 0; i < count; ++i) {
    value_counts[data[i]]++;
}

// 在哈希表上找 TopK
std::vector<std::pair<int32_t, size_t>> items(value_counts.begin(), value_counts.end());
std::nth_element(items.begin(), items.begin() + k, items.end(),
    [](const auto& a, const auto& b) { return a.first > b.first; });
```

#### 方案 C: 混合策略

```cpp
// 快速基数估计
size_t cardinality_estimate = estimate_cardinality_hll(data, count);

if (cardinality_estimate < 500) {
    return topk_hash_based(data, count, k);
} else if (cardinality_estimate < 5000) {
    return topk_hybrid(data, count, k);  // 结合哈希和采样
} else {
    return topk_sampled_prefilter(data, count, k);
}
```

---

## 五、结论

### 5.1 核心发现

1. **临界基数: 500**
   - 当数据基数 >= 500 时，ThunderDuck 开始反超 DuckDB
   - 这意味着每个值平均重复 <= 20,000 次

2. **高基数优势明显**
   - 基数 >= 10,000 时，加速比稳定在 3.6-4.8x
   - 采样预过滤策略对高基数数据非常有效

3. **低基数需要改进**
   - 基数 < 500 时，当前策略效率较低
   - 需要实现专门的低基数优化算法

### 5.2 使用建议

| 数据特征 | 推荐引擎 |
|----------|---------|
| 基数 >= 500 | ThunderDuck |
| 基数 < 500 | DuckDB 或等待 v5 优化 |
| 用户/订单 ID | ThunderDuck |
| 时间戳/价格 | ThunderDuck |
| 状态码/类别 | DuckDB |

---

## 六、未来工作

### 6.1 短期 (v4.1)
- [ ] 添加基数检测启发式
- [ ] 实现哈希 TopK 作为低基数回退

### 6.2 中期 (v5.0)
- [ ] HyperLogLog 基数估计
- [ ] 自适应算法选择
- [ ] 并行 TopK 优化

---

*ThunderDuck - 基数 >= 500 时全面碾压！*
