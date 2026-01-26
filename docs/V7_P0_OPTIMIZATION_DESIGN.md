# V7.1 P0 优化设计文档

> **版本**: 1.0 | **日期**: 2026-01-26
> **目标**: 解决 V7 基线中识别的两个 P0 性能问题

---

## 一、问题概述

### 1.1 Hash Join `grow_join_result()` 瓶颈

| 属性 | 值 |
|------|-----|
| **位置** | `src/operators/join/simd_hash_join.cpp:392-414` |
| **当前性能** | 全匹配场景 2.7x vs DuckDB |
| **目标性能** | 4x vs DuckDB |
| **根因** | 动态扩容 + O(n) memcpy |

**问题代码分析**:

```cpp
void grow_join_result(JoinResult* result, size_t min_capacity) {
    // 每次扩容都要:
    // 1. 分配新内存 (2x 容量)
    // 2. memcpy 所有已有数据 → O(n)
    // 3. 释放旧内存

    size_t new_capacity = result->capacity * 2;
    uint32_t* new_left = aligned_alloc(...);
    uint32_t* new_right = aligned_alloc(...);

    std::memcpy(new_left, result->left_indices, result->count * sizeof(uint32_t));
    std::memcpy(new_right, result->right_indices, result->count * sizeof(uint32_t));
    // ...
}
```

**性能影响**:
- 假设匹配 1M 行，初始容量 1K
- 扩容序列: 1K → 2K → 4K → 8K → ... → 2M
- 共 11 次扩容，累计 memcpy: 1K + 2K + 4K + ... + 1M ≈ 2M 行数据
- 高匹配场景下，memcpy 成为主要开销

### 1.2 TopK 100K K=10 退化

| 属性 | 值 |
|------|-----|
| **位置** | `src/operators/sort/topk_v4.cpp:663-687` |
| **当前性能** | 0.79x vs partial_sort |
| **目标性能** | 1.2x vs partial_sort |
| **根因** | 堆+索引开销 > 直接 partial_sort |

**问题分析**:

```cpp
// 当前 v4 策略判断:
if (count >= LARGE_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
    // LARGE_N_THRESHOLD = 1M, 100K 不满足条件
    topk_sampled_prefilter_max(...);  // 不会执行
}

// 100K 走这个分支:
if (k <= K_SMALL_THRESHOLD) {  // K=10 <= 64
    topk_heap_small_max(...);   // 使用堆方法
}
```

**堆方法开销**:
1. 每个元素创建 `pair<int32_t, uint32_t>` (8 字节)
2. 100K 元素，每次堆比较涉及 pair 访问
3. 最终 sort 也是 pair 排序

**partial_sort 优势**:
1. 直接在 int32_t 数组上操作 (4 字节)
2. 缓存友好，连续访问
3. 无索引追踪开销

---

## 二、优化方案

### 2.1 Hash Join 两阶段算法

#### 2.1.1 核心思想

```
传统方案 (单遍历 + 动态扩容):
  for each probe_key:
    if match:
      grow_if_needed()  ← O(n) memcpy
      write_result()

两阶段方案:
  Phase 1: 计数遍历
    for each probe_key:
      if match: match_count++

  Phase 2: 一次性分配 + 填充
    allocate(match_count)  ← 精确容量
    for each probe_key:
      if match: write_result()
```

#### 2.1.2 实现设计

```cpp
// 新增函数: hash_join_i32_v5_twophase
size_t hash_join_i32_v5_twophase(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type,
    JoinResult* result)
{
    // 1. 构建哈希表 (复用现有实现)
    HashTable ht(build_count);
    ht.build_i32(build_keys, build_count);

    // 2. Phase 1: 计数遍历
    size_t match_count = ht.count_matches(probe_keys, probe_count);

    // 3. 一次性精确分配
    if (result->capacity < match_count) {
        // 直接分配精确容量，无需 grow
        reallocate_exact(result, match_count);
    }

    // 4. Phase 2: 填充结果
    match_count = ht.probe_i32(probe_keys, probe_count,
                                result->left_indices, result->right_indices);

    result->count = match_count;
    return match_count;
}

// HashTable 新增方法
class HashTable {
    // 只计数不写入
    size_t count_matches(const int32_t* probe_keys, size_t probe_count) {
        size_t count = 0;
        for (size_t i = 0; i < probe_count; ++i) {
            uint32_t slot = hash(probe_keys[i]) & mask_;
            while (keys_[slot] != EMPTY_KEY) {
                if (keys_[slot] == probe_keys[i]) {
                    count++;
                }
                slot = (slot + 1) & mask_;
            }
        }
        return count;
    }
};
```

#### 2.1.3 优化变体: SIMD 批量计数

```cpp
// 利用 SIMD 加速 Phase 1 计数
size_t count_matches_simd(const int32_t* probe_keys, size_t probe_count) {
    size_t count = 0;

    // 批量预取提高缓存命中
    for (size_t i = 0; i < probe_count; i += 4) {
        // 预取哈希表槽位
        __builtin_prefetch(&keys_[(hash(probe_keys[i+4]) & mask_)]);
        __builtin_prefetch(&keys_[(hash(probe_keys[i+5]) & mask_)]);
        __builtin_prefetch(&keys_[(hash(probe_keys[i+6]) & mask_)]);
        __builtin_prefetch(&keys_[(hash(probe_keys[i+7]) & mask_)]);

        // 处理当前批次
        count += count_single(probe_keys[i]);
        count += count_single(probe_keys[i+1]);
        count += count_single(probe_keys[i+2]);
        count += count_single(probe_keys[i+3]);
    }

    return count;
}
```

#### 2.1.4 性能预估

| 场景 | 当前 | 优化后 | 预估提升 |
|------|------|--------|---------|
| 100K×1M 全匹配 | 2.7x | 3.8x | **+41%** |
| 100K×1M 50%匹配 | 3.2x | 3.8x | **+19%** |
| 100K×1M 10%匹配 | 4.2x | 4.4x | +5% |

**分析**:
- 全匹配: 消除大量 grow + memcpy，收益最大
- 部分匹配: grow 次数减少，收益递减
- 低匹配: 原本扩容少，收益有限

**开销分析**:
- 新增一次哈希表遍历 (Phase 1)
- 但消除了 O(log(n)) 次 memcpy
- 当匹配数 > 初始容量 * 2 时，两阶段方案更优

---

### 2.2 TopK 小数据 partial_sort 回退

#### 2.2.1 核心思想

```
问题: 100K K=10，堆方法 + 索引追踪有开销

解决: 小数据量 (N < 阈值) 时，直接使用 partial_sort
      - 创建索引数组: indices[i] = i
      - partial_sort(indices, indices+k, indices+n, cmp)
      - cmp = data[a] > data[b]

优势:
- 比较函数简单
- partial_sort 对小 K 高度优化
- 无堆维护开销
```

#### 2.2.2 实现设计

```cpp
// 新增阈值
namespace {
constexpr size_t SMALL_N_THRESHOLD = 500000;  // N < 500K 使用 partial_sort
}

void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                      int32_t* out_values, uint32_t* out_indices) {
    if (k == 0 || count == 0) return;
    k = std::min(k, count);

    // 新增: 小数据量直接 partial_sort
    if (count < SMALL_N_THRESHOLD) {
        topk_partial_sort_max(data, count, k, out_values, out_indices);
        return;
    }

    // 大数据量 + 小 K → 采样预过滤
    if (count >= LARGE_N_THRESHOLD && k <= K_SMALL_THRESHOLD) {
        topk_sampled_prefilter_max(data, count, k, out_values, out_indices);
        return;
    }

    // 其他场景...
}

// partial_sort 实现
void topk_partial_sort_max(const int32_t* data, size_t count, size_t k,
                            int32_t* out_values, uint32_t* out_indices) {
    // 1. 创建索引数组
    std::vector<uint32_t> indices(count);
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }

    // 2. partial_sort (只排前 K 个)
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [data](uint32_t a, uint32_t b) { return data[a] > data[b]; });

    // 3. 输出结果
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = data[indices[i]];
        if (out_indices) out_indices[i] = indices[i];
    }
}
```

#### 2.2.3 进一步优化: 避免索引数组分配

```cpp
// 如果不需要索引输出，使用更快的路径
void topk_partial_sort_max_values_only(const int32_t* data, size_t count, size_t k,
                                        int32_t* out_values) {
    // 复制数据 (避免修改原数组)
    std::vector<int32_t> temp(data, data + count);

    // partial_sort 直接在值上操作
    std::partial_sort(temp.begin(), temp.begin() + k, temp.end(),
        std::greater<int32_t>());

    // 输出
    std::copy(temp.begin(), temp.begin() + k, out_values);
}

// 策略选择
void topk_max_i32_v4(...) {
    if (count < SMALL_N_THRESHOLD) {
        if (out_indices == nullptr) {
            // 只需要值，不需要索引
            topk_partial_sort_max_values_only(data, count, k, out_values);
        } else {
            topk_partial_sort_max(data, count, k, out_values, out_indices);
        }
        return;
    }
    // ...
}
```

#### 2.2.4 阈值调优

基于 benchmark 数据:

| N | K | v4 堆 (μs) | partial_sort (μs) | 推荐 |
|---|---|-----------|-------------------|------|
| 100K | 10 | 51.1 | 40.9 | partial_sort |
| 100K | 100 | 12.4 | 50.8 | v4 堆 ✓ |
| 500K | 10 | 229.7 | 306.7 | v4 堆 ✓ |
| 1M | 10 | 50.7 | 684.5 | v4 采样 ✓ |

**结论**:
- N < 500K 且 K 小 → partial_sort
- N >= 500K 且 K 小 → 堆方法
- N >= 1M → 采样预过滤

**阈值建议**:
```cpp
// 小数据阈值
constexpr size_t SMALL_N_THRESHOLD = 500000;

// 策略矩阵:
// N < 500K                → partial_sort
// 500K <= N < 1M          → 堆方法
// N >= 1M && K <= 64      → 采样预过滤
// N >= 1M && K > 64       → nth_element
```

#### 2.2.5 性能预估

| 场景 | 当前 | 优化后 | 预估提升 |
|------|------|--------|---------|
| 100K K=10 | 0.79x | 1.0x | **+27%** |
| 100K K=100 | 3.93x | 3.93x | 无变化 (仍用堆) |
| 200K K=10 | ~0.9x | 1.1x | **+22%** |
| 500K K=10 | 1.24x | 1.24x | 无变化 (仍用堆) |

---

## 三、实现计划

### 3.1 文件修改

| 文件 | 修改类型 | 描述 |
|------|---------|------|
| `src/operators/join/simd_hash_join.cpp` | 修改 | 添加两阶段算法 |
| `src/operators/sort/topk_v4.cpp` | 修改 | 添加 partial_sort 路径 |
| `include/thunderduck/join.h` | 修改 | 新增 v5 API |
| `benchmark/test_hash_join_twophase.cpp` | 新增 | 性能验证 |
| `benchmark/test_topk_smalln.cpp` | 新增 | 性能验证 |

### 3.2 实现步骤

#### Phase 1: Hash Join 两阶段 (预计 2-3 小时)

1. 在 `HashTable` 类添加 `count_matches()` 方法
2. 实现 `hash_join_i32_v5_twophase()` 函数
3. 添加 SIMD 批量计数优化
4. 编写 benchmark 验证

#### Phase 2: TopK partial_sort 回退 (预计 1-2 小时)

1. 添加 `topk_partial_sort_max/min()` 函数
2. 修改策略选择逻辑，添加 `SMALL_N_THRESHOLD`
3. 编写 benchmark 验证

### 3.3 测试计划

```bash
# Hash Join 验证
./build/test_hash_join_twophase

# TopK 验证
./build/test_topk_smalln

# 回归测试
./build/benchmark_app
```

---

## 四、风险评估

### 4.1 Hash Join 两阶段

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 两次哈希遍历增加开销 | 低匹配场景可能变慢 | 条件判断: 只在高匹配时使用 |
| 实现复杂度增加 | 维护成本 | 代码注释 + 单元测试 |

**保护措施**:
```cpp
// 只有预估高匹配率时才用两阶段
if (estimated_match_rate > 0.5 || probe_count > 100000) {
    return hash_join_v5_twophase(...);
} else {
    return hash_join_v4(...);
}
```

### 4.2 TopK partial_sort

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 阈值选择不当 | 某些场景变慢 | 基于 benchmark 精细调优 |
| partial_sort 不稳定 | 相同值顺序变化 | 文档说明 |

---

## 五、实验结果

> **更新**: 2026-01-26 实现并测试后的发现

### 5.1 Hash Join V5 两阶段算法

| 场景 | V3 时间 | V5 时间 | V5/V3 | 结论 |
|------|---------|---------|-------|------|
| 100K×1M 全匹配 | 0.92 ms | 3.44 ms | 3.7x 更慢 | ❌ 不采用 |
| 100K×1M 50%匹配 | 0.54 ms | 2.36 ms | 4.4x 更慢 | ❌ 不采用 |
| 100K×1M 10%匹配 | 0.29 ms | 1.87 ms | 6.5x 更慢 | ❌ 不采用 |

**分析**:
- 原假设: `grow_join_result()` 的 O(n) memcpy 是主要瓶颈
- 实际发现: 现代硬件 memcpy 极快 (20+ GB/s with SIMD)
- 两阶段算法的额外哈希表遍历开销 >> grow 开销
- **结论**: 保留 V5 代码作为参考，但不作为默认策略

### 5.2 TopK partial_sort 回退

| 场景 | partial_sort | ThunderDuck v4 | 比值 |
|------|-------------|----------------|------|
| 100K K=10 | 0.044 ms | 0.063 ms | 0.70x |
| 200K K=10 | 0.079 ms | 0.108 ms | 0.73x |

**分析**:
- ThunderDuck v4 追踪索引，std::partial_sort 只排值
- 间接比较 `data[a] > data[b]` 有额外内存访问开销
- **结论**: 保留 partial_sort 路径，提供索引追踪功能

---

## 六、总结

| 优化 | 预期收益 | 实际结果 | 状态 |
|------|---------|----------|------|
| Hash Join 两阶段 | +41% | -73% (更慢) | ❌ 不采用 |
| TopK partial_sort | +27% | +0% (持平) | ✓ 保留 |

**经验教训**:
1. 理论分析需要实测验证
2. 现代硬件的 memcpy 性能远超预期
3. 减少遍历次数比减少内存分配更重要

**V7.1 实际达成**:
- Hash Join: 维持 V3 性能 (已是最优)
- TopK: 小数据使用 partial_sort，大数据使用采样预过滤

---

*文档版本: 1.1 | 更新: 2026-01-26 | 添加实验结果*
