# ThunderDuck V13 - 极致优化版本

> **版本标签**: V13 - 极致优化
> **设计日期**: 2026-01-27
> **目标**: 解决 P0/P1/P3 性能瓶颈

---

## 一、优化目标

| 优先级 | 算子 | 当前性能 | 目标性能 | 优化策略 |
|--------|------|----------|----------|----------|
| **P0** | Hash Join | ~0.06x | **1.5x+** | 两阶段算法 + 预分配 |
| **P1** | GROUP BY GPU | 0.78x | **2.0x+** | 无原子累加 |
| **P3** | TopK GPU | 无 | **5x+** | GPU 并行 Bitonic Sort |

---

## 二、P0: Hash Join 两阶段优化

### 2.1 问题分析

当前实现瓶颈：
1. `grow_join_result()` 动态扩容导致 O(n) memcpy
2. 链表探测增加随机访问
3. 结果数组管理开销

### 2.2 解决方案: 两阶段算法

```
Phase 1: 计数阶段
  - 只统计匹配数量，不存储结果
  - 预分配精确容量的结果数组

Phase 2: 填充阶段
  - 直接写入预分配数组
  - 无动态扩容开销
```

### 2.3 核心代码设计

```cpp
// hash_join_v13_twophase.cpp

size_t hash_join_i32_v13(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result)
{
    // 1. 构建哈希表 (Robin Hood)
    RobinHoodHashTable ht(build_count);
    for (size_t i = 0; i < build_count; i++) {
        ht.insert(build_keys[i], i);
    }

    // 2. Phase 1: 计数
    size_t total_matches = 0;
    for (size_t i = 0; i < probe_count; i++) {
        total_matches += ht.count(probe_keys[i]);
    }

    // 3. 预分配结果数组
    ensure_capacity(result, total_matches);

    // 4. Phase 2: 填充
    size_t write_idx = 0;
    for (size_t i = 0; i < probe_count; i++) {
        auto matches = ht.find_all(probe_keys[i]);
        for (auto build_idx : matches) {
            result->left_indices[write_idx] = build_idx;
            result->right_indices[write_idx] = i;
            write_idx++;
        }
    }

    result->count = total_matches;
    return total_matches;
}
```

### 2.4 Robin Hood 哈希表

特点：
- 开放寻址，缓存友好
- PSL (Probe Sequence Length) 平衡
- 平均探测长度 < 2

---

## 三、P1: GROUP BY GPU 无原子优化

### 3.1 问题分析

当前 GPU GROUP BY 瓶颈：
- 全局原子操作争用
- Warp-level reduction 仍需原子合并

### 3.2 解决方案: 分区聚合

```
策略: 按 threadgroup 分区，每个分区独立累加，最后合并

Phase 1: 分区累加 (无原子)
  - 每个 threadgroup 维护本地 partial_sums[num_groups]
  - 线程内累加到本地数组

Phase 2: 全局合并 (原子，但次数少)
  - 每个 threadgroup 的 partial_sums 原子加到全局结果
  - 原子操作次数: num_threadgroups * num_groups
```

### 3.3 Metal Shader 设计

```metal
kernel void group_sum_v13_phase1(
    device const int32_t* values [[buffer(0)]],
    device const uint32_t* groups [[buffer(1)]],
    device int64_t* partial_sums [[buffer(2)]],  // [num_threadgroups][num_groups]
    constant uint32_t& count [[buffer(3)]],
    constant uint32_t& num_groups [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // 每个 threadgroup 的本地累加区
    uint offset = tg_id * num_groups;

    // 无原子: 直接累加到 partial_sums
    for (uint i = tg_id * tg_size + tid; i < count; i += gridDim * tg_size) {
        uint g = groups[i];
        // 使用 threadgroup 内存避免全局原子
        threadgroup_atomic_add(&local_sums[g], values[i]);
    }

    // 写回 partial_sums
    if (tid < num_groups) {
        partial_sums[offset + tid] = local_sums[tid];
    }
}

kernel void group_sum_v13_phase2(
    device int64_t* partial_sums [[buffer(0)]],
    device int64_t* final_sums [[buffer(1)]],
    constant uint32_t& num_groups [[buffer(2)]],
    constant uint32_t& num_threadgroups [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_groups) return;

    int64_t sum = 0;
    for (uint tg = 0; tg < num_threadgroups; tg++) {
        sum += partial_sums[tg * num_groups + tid];
    }
    final_sums[tid] = sum;
}
```

---

## 四、P3: TopK GPU 并行版本

### 4.1 算法选择

| 算法 | 复杂度 | GPU友好度 | 适用场景 |
|------|--------|-----------|----------|
| Bitonic Sort | O(n log²n) | ★★★★★ | K 较小 |
| Radix Select | O(n) | ★★★★ | K 较大 |
| Quick Select | O(n) 平均 | ★★ | 不适合 GPU |

选择: **Bitonic Partial Sort** (K 较小时最优)

### 4.2 Bitonic Partial Sort 设计

```
思路: 只对需要的 K 个元素进行完整排序

1. 初始化: 从 N 个元素中选取 K 个最大
2. Bitonic 网络: 对这 K 个元素排序
3. 增量更新: 逐步淘汰更小的元素
```

### 4.3 Metal Shader 设计

```metal
kernel void topk_bitonic_phase1(
    device const int32_t* data [[buffer(0)]],
    device int32_t* candidates [[buffer(1)]],
    device uint32_t* indices [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tg_id [[threadgroup_position_in_grid]])
{
    // 每个 threadgroup 找本地 TopK
    threadgroup int32_t local_topk[1024];
    threadgroup uint32_t local_idx[1024];

    // ... 本地堆选择 ...

    // 写入 candidates
}

kernel void topk_bitonic_merge(
    device int32_t* candidates [[buffer(0)]],
    device uint32_t* indices [[buffer(1)]],
    constant uint32_t& k [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    // Bitonic 比较交换
    uint partner = tid ^ (1 << stage);
    if (partner > tid) {
        if (candidates[tid] < candidates[partner]) {
            // 交换
            swap(candidates[tid], candidates[partner]);
            swap(indices[tid], indices[partner]);
        }
    }
}
```

---

## 五、V13 统一接口

```cpp
namespace thunderduck::v13 {

// P0: 两阶段 Hash Join
size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType type, JoinResult* result,
                     ExecutionStats* stats = nullptr);

// P1: GPU 无原子 GROUP BY
void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats = nullptr);

// P3: GPU TopK
size_t topk_max_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices,
                    ExecutionStats* stats = nullptr);

}
```

---

## 六、预期性能目标

| 算子 | V12.5 | V13 目标 | 提升 |
|------|-------|----------|------|
| Hash Join | 0.06x | **1.5x+** | 25x |
| GROUP BY GPU | 0.78x | **2.0x+** | 2.5x |
| TopK GPU | N/A | **5x+** | 新功能 |

---

## 七、实现计划

| 阶段 | 任务 | 文件 |
|------|------|------|
| 1 | P0 两阶段 Join | `hash_join_v13.cpp` |
| 2 | P1 GPU GROUP BY | `group_aggregate_v13.mm` |
| 3 | P3 GPU TopK | `topk_v13.mm` |
| 4 | V13 统一接口 | `v13_unified.cpp` |
| 5 | 基准测试 | `v13_benchmark.cpp` |
