# V15 深度优化设计文档

> **版本**: 15.0 | **日期**: 2026-01-27

## 一、问题分析

### P0: Filter 输出索引 (0.40x → 2x+)

**当前实现** (V3):
```
filter_to_bitmap_v3()  →  bitmap_to_indices()
      0.55ms                   3.5ms
      (SIMD)              (串行 CTZ 循环)
```

**瓶颈**: `bitmap_to_indices` 对 5M 匹配执行 5M 次 CTZ + 内存写入

**DuckDB 方案**: 使用 Selection Vector，直接在过滤时生成索引，无位图中间层

### P1: GROUP BY (2.12x → 3x+)

**当前实现** (V14):
- 4 线程并行累加
- SIMD 合并结果

**瓶颈**: 线程数固定为 4，合并阶段可进一步向量化

### P2: Hash Join (2.46x → 3x+)

**当前实现** (V3):
- 16 分区 Radix 哈希
- 1.7x 负载因子
- 线性探测

**瓶颈**: 线性探测在高负载时探测距离长

---

## 二、V15 优化方案

### P0: Filter V15 - 直接 SIMD 索引生成

**核心思想**: 跳过位图中间层，直接用 SIMD 生成索引

```cpp
// V15: 直接生成索引，无位图中间层
template<CompareOp Op>
size_t filter_i32_v15_direct(const int32_t* input, size_t count,
                              int32_t value, uint32_t* out_indices) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t out_count = 0;

    // 预生成索引基准向量
    uint32x4_t idx_base = {0, 1, 2, 3};
    uint32x4_t idx_inc = vdupq_n_u32(4);

    for (size_t i = 0; i + 4 <= count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        uint32x4_t mask = simd_compare_i32<Op>(data, threshold);

        // 提取 4-bit 掩码
        uint32_t bits = extract_mask_4(mask);

        if (bits) {
            // 使用 LUT 直接写入匹配的索引
            const uint8_t* lut = COMPRESS_LUT[bits];
            uint8_t cnt = POPCOUNT_4[bits];

            // SIMD 压缩存储
            uint32x4_t indices = vaddq_u32(idx_base, vdupq_n_u32(i));
            uint8x8_t perm = vld1_u8(lut);
            uint32x4_t compressed = vtbl1q_u32(indices, perm);

            // 只写入有效索引
            vst1q_u32(out_indices + out_count, compressed);
            out_count += cnt;
        }

        idx_base = vaddq_u32(idx_base, idx_inc);
    }

    return out_count;
}
```

**预期收益**: 消除 5M 次 CTZ 循环 → 3x+ 加速

### P1: GROUP BY V15 - 8 线程 + SIMD 归约

```cpp
void group_sum_i32_v15(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int64_t* out_sums) {
    constexpr size_t NUM_THREADS = 8;  // 使用全部 8 核

    // 每线程独立累加器 (cache line 对齐避免伪共享)
    alignas(128) int64_t local_sums[NUM_THREADS][num_groups];

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int t = 0; t < NUM_THREADS; ++t) {
        // 8 路循环展开 + 预取
        for (size_t i = start; i + 8 <= end; i += 8) {
            __builtin_prefetch(&groups[i + 64], 0, 2);
            __builtin_prefetch(&values[i + 64], 0, 2);

            // 8 路展开
            local_sums[t][groups[i]]   += values[i];
            local_sums[t][groups[i+1]] += values[i+1];
            // ... x8
        }
    }

    // SIMD 8 路归约
    for (size_t g = 0; g + 4 <= num_groups; g += 4) {
        int64x2_t sum0 = vdupq_n_s64(0);
        int64x2_t sum1 = vdupq_n_s64(0);

        for (int t = 0; t < NUM_THREADS; t += 2) {
            sum0 = vaddq_s64(sum0, vld1q_s64(&local_sums[t][g]));
            sum1 = vaddq_s64(sum1, vld1q_s64(&local_sums[t+1][g]));
        }

        vst1q_s64(&out_sums[g], vaddq_s64(sum0, sum1));
    }
}
```

**预期收益**: 8 线程 + 更好的 SIMD 归约 → 1.5x 加速

### P2: Hash Join V15 - Robin-Hood 哈希

```cpp
// Robin-Hood 哈希表：元素按探测距离排序
struct RobinHoodEntry {
    int32_t key;
    uint32_t value;
    uint8_t probe_distance;  // 当前探测距离
};

size_t hash_join_v15_robin_hood(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinResult* result) {
    // Robin-Hood 插入: 如果新元素探测距离 > 现有元素，交换
    for (size_t i = 0; i < build_count; ++i) {
        uint32_t h = hash(build_keys[i]);
        uint8_t dist = 0;

        while (true) {
            size_t pos = (h + dist) & mask;

            if (table[pos].empty()) {
                table[pos] = {build_keys[i], i, dist};
                break;
            }

            // Robin-Hood: 穷人让位给富人
            if (dist > table[pos].probe_distance) {
                std::swap(entry, table[pos]);
            }

            ++dist;
        }
    }

    // 探测: 探测距离超过当前位置的 probe_distance 即可停止
    for (size_t i = 0; i < probe_count; ++i) {
        uint32_t h = hash(probe_keys[i]);
        uint8_t dist = 0;

        while (dist <= table[(h + dist) & mask].probe_distance) {
            size_t pos = (h + dist) & mask;
            if (table[pos].key == probe_keys[i]) {
                // 匹配
                emit_match(i, table[pos].value);
            }
            ++dist;
        }
    }
}
```

**预期收益**: 更均匀的探测距离 → 1.2x 加速

---

## 三、实现计划

| 阶段 | 任务 | 文件 | 预期收益 |
|------|------|------|----------|
| 1 | Filter V15 直接索引 | `simd_filter_v15.cpp` | 3x+ |
| 2 | GROUP BY V15 8线程 | `group_aggregate_v15.cpp` | 1.5x |
| 3 | Hash Join V15 Robin-Hood | `hash_join_v15.cpp` | 1.2x |
| 4 | 基准测试验证 | `v15_benchmark.cpp` | - |

---

## 四、文件结构

```
src/operators/filter/
├── simd_filter_v15.cpp      # V15 直接索引生成

src/operators/aggregate/
├── group_aggregate_v15.cpp  # V15 8线程 + SIMD归约

src/operators/join/
├── hash_join_v15.cpp        # V15 Robin-Hood 哈希

include/thunderduck/
├── v15.h                    # V15 统一接口
```

---

## 五、验证目标

| 算子 | V14 | V15 目标 | 提升 |
|------|-----|----------|------|
| Filter 输出 | 0.40x | **2.0x+** | **5x** |
| GROUP BY | 2.12x | **3.0x+** | **1.5x** |
| Hash Join | 2.46x | **3.0x+** | **1.2x** |
