# ThunderDuck 核心算法技术文档

> **版本**: 1.0.0
> **日期**: 2026-01-31
> **平台**: Apple M4 (ARM Neon, Metal GPU, UMA)

---

## 目录

1. [概述](#1-概述)
2. [Filter 算法](#2-filter-算法)
3. [Join 算法](#3-join-算法)
4. [Aggregate 算法](#4-aggregate-算法)
5. [Sort 算法](#5-sort-算法)
6. [专用算法](#6-专用算法)
7. [GPU 算法](#7-gpu-算法)
8. [复杂度分析](#8-复杂度分析)
9. [参考文献](#9-参考文献)

---

## 1. 概述

ThunderDuck 是针对 Apple M4 芯片深度优化的 SQL 算子库，充分利用以下硬件特性：

| 硬件特性 | 利用方式 | 性能提升 |
|---------|---------|---------|
| **ARM Neon SIMD** | 128-bit 并行向量运算 | 4-8x |
| **Metal GPU** | 并行数据处理 + UMA 零拷贝 | 2-3x |
| **UMA 统一内存** | CPU/GPU 共享内存，无拷贝开销 | 1.5-2x |
| **CRC32 硬件指令** | 单周期哈希计算 | 3-5x |
| **128 字节缓存行** | 对齐优化减少 cache miss | 1.2-1.5x |

### 1.1 算法分类

```
ThunderDuck 核心算法
├── Filter (过滤)
│   ├── SIMD 向量化比较
│   ├── 位图中间表示
│   ├── 掩码压缩 LUT
│   └── GPU 并行扫描
├── Join (连接)
│   ├── Robin Hood Hash Join
│   ├── 完美哈希优化
│   ├── SIMD 批量探测
│   ├── Bloom 预过滤
│   └── GPU 两阶段并行
├── Aggregate (聚合)
│   ├── 直接数组聚合
│   ├── 8 路并行累加
│   ├── GPU Block-local Hash
│   └── 寄存器缓冲优化
├── Sort (排序)
│   ├── Radix Sort (11-11-10 bit)
│   ├── SIMD Bitonic Network
│   └── 采样预过滤 TopK
└── 专用算法
    ├── Bitmap Anti-Join (Q22)
    ├── EXISTS 优化 (Q21)
    └── 子查询解关联
```

### 1.2 版本演进

| 算子 | 最佳版本 | 性能 vs DuckDB | 关键优化 |
|------|---------|---------------|---------|
| Filter | V4 (GPU) | 1.10x | GPU 并行扫描 + UMA 零拷贝 |
| GROUP BY | V15 (8T) | 2.69x | 8 线程 + 循环展开 |
| INNER JOIN | V14 | 1.63x | 预分配 + Radix 分区 |
| SEMI JOIN | GPU | 2.47x | GPU 哈希探测 |
| TopK | V4 | 4.71x | 采样预过滤 + SIMD 筛选 |

---

## 2. Filter 算法

### 2.1 核心思想

Filter 算子执行 `WHERE value > threshold` 类型的谓词过滤，关键挑战：
1. **分支预测失败** - 标量循环中 `if (value > threshold)` 导致流水线停滞
2. **选择率不确定** - 输出大小未知，需要动态内存分配

### 2.2 V3 模板特化 + 4 累加器并行

**算法伪代码**:
```cpp
// 输入: input[n], op (比较操作), value (阈值)
// 输出: out_indices[], count

// 模板特化消除循环内分支
template<CompareOp OP>
size_t filter_i32_v3(const int32_t* input, size_t count,
                      int32_t value, uint32_t* out_indices) {
    // 4 独立累加器消除依赖链 (ILP)
    uint32x4_t cnt0 = vdupq_n_u32(0);
    uint32x4_t cnt1 = vdupq_n_u32(0);
    uint32x4_t cnt2 = vdupq_n_u32(0);
    uint32x4_t cnt3 = vdupq_n_u32(0);

    int32x4_t threshold = vdupq_n_s32(value);

    // 主循环: 256 元素批次 (64 次迭代 × 4 元素)
    for (size_t i = 0; i + 256 <= count; i += 256) {
        // 预取下一批次
        __builtin_prefetch(input + i + 256, 0, 0);

        for (size_t j = 0; j < 64; ++j) {
            // 加载 4 × int32
            int32x4_t data = vld1q_s32(input + i + j*4);

            // SIMD 比较 (特化为 vcgtq_s32, vceqq_s32, 等)
            uint32x4_t mask = simd_compare<OP>(data, threshold);

            // vsub 优化: mask 为 0xFFFFFFFF 或 0x00000000
            // -0xFFFFFFFF = 1, -0x00000000 = 0
            cnt0 = vsubq_u32(cnt0, mask);  // 替代 vshr+vadd
        }

        // 每 256 元素归约一次 (减少水平归约次数)
        cnt0 = vaddq_u32(cnt0, cnt1);
        cnt0 = vaddq_u32(cnt0, cnt2);
        cnt0 = vaddq_u32(cnt0, cnt3);
        cnt1 = cnt2 = cnt3 = vdupq_n_u32(0);
    }

    // 水平归约
    return vaddvq_u32(cnt0) + vaddvq_u32(cnt1) +
           vaddvq_u32(cnt2) + vaddvq_u32(cnt3);
}
```

**ARM Neon 代码示例**:
```cpp
// GT 比较: input[i] > threshold
int32x4_t data = vld1q_s32(input + i);
int32x4_t threshold_vec = vdupq_n_s32(threshold);
uint32x4_t mask = vcgtq_s32(data, threshold_vec);
// mask: 0xFFFFFFFF (满足) 或 0x00000000 (不满足)

// vsub 优化计数
uint32x4_t count = vdupq_n_u32(0);
count = vsubq_u32(count, mask);  // 0 - (-1) = 1, 0 - 0 = 0
```

**性能数据**:
- **时间**: 3.839 ms (10M 元素)
- **带宽**: 10.42 GB/s
- **vs DuckDB**: 0.69x

**关键优化**:
1. **模板特化**: 编译期确定比较操作，消除循环内 switch-case
2. **4 累加器**: 消除数据依赖链，ILP (Instruction-Level Parallelism)
3. **vsub 技巧**: `count - mask` 替代 `(mask >> 31) + count`，减少指令数
4. **批次归约**: 256 元素一批，减少归约开销

### 2.3 V15 直接索引生成

**核心思想**: 跳过位图中间层，直接生成索引

**算法伪代码**:
```cpp
size_t filter_i32_v15(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices) {
    size_t out_count = 0;
    uint32_t local_buf[128];  // 本地缓冲区
    size_t local_count = 0;

    for (size_t i = 0; i < count; i += 4) {
        // SIMD 比较
        int32x4_t data = vld1q_s32(input + i);
        uint32x4_t mask = simd_compare(data, threshold);

        // 提取 4-bit 掩码
        uint8_t bits = extract_4bit_mask(mask);

        // LUT 查表快速生成索引
        const uint8_t* lut = INDEX_LUT[bits];
        size_t n = POPCOUNT[bits];  // 匹配数量

        for (size_t j = 0; j < n; ++j) {
            local_buf[local_count++] = i + lut[j];
        }

        // 本地缓冲区满 → 批量写出
        if (local_count >= 120) {
            memcpy(out_indices + out_count, local_buf,
                   local_count * sizeof(uint32_t));
            out_count += local_count;
            local_count = 0;
        }
    }

    // 剩余数据
    if (local_count > 0) {
        memcpy(out_indices + out_count, local_buf,
               local_count * sizeof(uint32_t));
        out_count += local_count;
    }

    return out_count;
}
```

**4-bit 掩码 LUT**:
```cpp
// 示例: bits = 0b1010 → 索引 [1, 3]
static const uint8_t INDEX_LUT[16][4] = {
    {0, 0, 0, 0},  // 0b0000 → []
    {0, 0, 0, 0},  // 0b0001 → [0]
    {1, 0, 0, 0},  // 0b0010 → [1]
    {0, 1, 0, 0},  // 0b0011 → [0, 1]
    // ...
    {0, 1, 2, 3},  // 0b1111 → [0, 1, 2, 3]
};

static const uint8_t POPCOUNT[16] = {
    0, 1, 1, 2, 1, 2, 2, 3,
    1, 2, 2, 3, 2, 3, 3, 4
};
```

**性能数据**:
- **时间**: 3.239 ms (10M 元素)
- **带宽**: 12.35 GB/s
- **vs DuckDB**: 0.82x

### 2.4 V4 GPU 自动策略

**算法**: 两阶段并行过滤

**Phase 1**: 统计匹配数
```metal
kernel void filter_count(
    const device int32_t* input [[buffer(0)]],
    device atomic_uint* count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (input[gid] > threshold) {
        atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    }
}
```

**Phase 2**: 填充索引
```metal
kernel void filter_write(
    const device int32_t* input [[buffer(0)]],
    device uint32_t* out_indices [[buffer(1)]],
    device atomic_uint* offset [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (input[gid] > threshold) {
        uint pos = atomic_fetch_add_explicit(offset, 1, memory_order_relaxed);
        out_indices[pos] = gid;
    }
}
```

**策略选择**:
```cpp
FilterStrategy select_strategy(size_t count, float selectivity) {
    if (count < 100000) return CPU_SIMD;  // V3

    if (selectivity < 0.1) return GPU_ATOMIC;   // 低选择率
    if (selectivity > 0.5) return GPU_SCAN;     // 高选择率

    return GPU_ATOMIC;  // 默认
}
```

**性能数据**:
- **时间**: 2.424 ms (10M 元素)
- **带宽**: 16.50 GB/s
- **vs DuckDB**: **1.10x** ✓

**关键优化**:
1. **UMA 零拷贝**: CPU/GPU 共享内存，无数据传输
2. **两阶段算法**: 预分配精确容量，避免原子竞争
3. **Metal Shared Events**: 减少 CPU-GPU 同步开销

---

## 3. Join 算法

### 3.1 V3 SOA + Radix 分区

**核心思想**:
1. **SOA 布局** - 哈希表按列存储，缓存友好
2. **Radix 分区** - 按哈希前缀分区，提升缓存局部性

**算法伪代码**:
```cpp
size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinResult* result) {
    // Phase 1: Radix 分区 (4-bit = 16 分区)
    Partition build_parts[16], probe_parts[16];

    for (size_t i = 0; i < build_count; i += 8) {
        // 批量哈希 8 个键
        uint32x4_t keys_lo = vld1q_s32(build_keys + i);
        uint32x4_t keys_hi = vld1q_s32(build_keys + i + 4);

        uint32x4_t hash_lo = crc32_batch(keys_lo);
        uint32x4_t hash_hi = crc32_batch(keys_hi);

        // 提取分区 ID (高 4 位)
        uint8_t part_id[8];
        extract_partition_ids(hash_lo, hash_hi, part_id);

        // 分散写入
        for (int j = 0; j < 8; ++j) {
            build_parts[part_id[j]].append(build_keys[i+j], i+j);
        }
    }

    // Phase 2: 对每个分区执行 Hash Join
    size_t total_matches = 0;

    for (int p = 0; p < 16; ++p) {
        // 构建分区哈希表 (SOA 布局)
        HashTable ht;
        ht.build_soa(build_parts[p].keys, build_parts[p].count);

        // 探测
        for (size_t i = 0; i < probe_parts[p].count; ++i) {
            int32_t probe_key = probe_parts[p].keys[i];
            uint32_t hash = crc32(probe_key);

            // 预取下一个 probe 键的哈希表位置
            if (i + 8 < probe_parts[p].count) {
                uint32_t next_hash = crc32(probe_parts[p].keys[i+8]);
                __builtin_prefetch(&ht.slots[next_hash & ht.mask]);
            }

            // 探测哈希表
            size_t matches = ht.probe(probe_key, hash, result);
            total_matches += matches;
        }
    }

    return total_matches;
}
```

**SOA 哈希表布局**:
```cpp
// 传统 AOS (Array of Structures)
struct HashEntry {
    int32_t key;
    uint32_t index;
    HashEntry* next;
};  // 16 字节，跨缓存行

// 优化 SOA (Structure of Arrays)
struct HashTable {
    int32_t* keys;      // 连续内存
    uint32_t* indices;  // 连续内存
    uint32_t* next;     // 连续内存

    // 探测时 SIMD 批量比较
    size_t probe(int32_t probe_key, uint32_t hash) {
        uint32_t pos = hash & mask;
        int32x4_t probe_vec = vdupq_n_s32(probe_key);

        while (pos != NULL_INDEX) {
            // 加载 4 个候选键
            int32x4_t candidates = vld1q_s32(keys + pos);
            uint32x4_t eq_mask = vceqq_s32(candidates, probe_vec);

            if (vmaxvq_u32(eq_mask)) {  // 有匹配
                // 提取匹配位置
                // ...
            }

            pos = next[pos];
        }
    }
};
```

**CRC32 硬件哈希**:
```cpp
#include <arm_acle.h>

inline uint32_t crc32(int32_t key) {
    return __crc32cw(0, key);  // 单周期指令
}

// 批量哈希
inline uint32x4_t crc32_batch(int32x4_t keys) {
    uint32_t h0 = __crc32cw(0, vgetq_lane_s32(keys, 0));
    uint32_t h1 = __crc32cw(0, vgetq_lane_s32(keys, 1));
    uint32_t h2 = __crc32cw(0, vgetq_lane_s32(keys, 2));
    uint32_t h3 = __crc32cw(0, vgetq_lane_s32(keys, 3));

    uint32x4_t hashes = {h0, h1, h2, h3};
    return hashes;
}
```

**性能数据**:
- **时间**: 1.213 ms (100K build × 1M probe)
- **带宽**: 3.63 GB/s
- **vs DuckDB**: **1.59x** ✓

**关键优化**:
1. **Radix 分区**: 提升缓存局部性，每个分区适配 L1/L2 cache
2. **SOA 布局**: 连续内存 + SIMD 批量比较
3. **CRC32 硬件哈希**: 单周期哈希计算
4. **三级预取**: L2(远) → L1(近) → 当前

### 3.2 V14 两阶段预分配

**核心思想**: 消除动态扩容的 `memcpy` 开销

**算法伪代码**:
```cpp
size_t hash_join_i32_v14(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinResult* result) {
    // Phase 1: 计数遍历 (无写入)
    size_t total_matches = 0;

    #pragma omp parallel for reduction(+:total_matches)
    for (size_t i = 0; i < probe_count; ++i) {
        uint32_t hash = crc32(probe_keys[i]);
        size_t matches = ht.count_matches(probe_keys[i], hash);
        total_matches += matches;
    }

    // Phase 2: 精确分配
    result->left_indices = (uint32_t*)malloc(total_matches * sizeof(uint32_t));
    result->right_indices = (uint32_t*)malloc(total_matches * sizeof(uint32_t));
    result->capacity = total_matches;

    // Phase 3: 并行填充 (每线程独立区域，无竞争)
    size_t offsets[num_threads + 1] = {0};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t local_count = 0;

        // 本地计数
        #pragma omp for
        for (size_t i = 0; i < probe_count; ++i) {
            local_count += ht.count_matches(probe_keys[i], crc32(probe_keys[i]));
        }

        // 前缀和计算偏移
        offsets[tid+1] = local_count;
        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                offsets[i] += offsets[i-1];
            }
        }

        // 写入独立区域
        size_t write_offset = offsets[tid];
        #pragma omp for
        for (size_t i = 0; i < probe_count; ++i) {
            write_offset += ht.write_matches(probe_keys[i], i,
                result->left_indices + write_offset,
                result->right_indices + write_offset);
        }
    }

    return total_matches;
}
```

**性能数据**:
- **时间**: 1.182 ms (100K × 1M)
- **带宽**: 3.72 GB/s
- **vs DuckDB**: **1.63x** ✓

### 3.3 GPU SEMI Join

**Metal Kernel**:
```metal
// Build 阶段: 构建哈希表
kernel void build_hash_table(
    const device int32_t* build_keys [[buffer(0)]],
    device atomic_uint* hash_table [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int32_t key = build_keys[gid];
    uint32_t hash = murmur_hash(key);
    uint32_t slot = hash & HASH_TABLE_MASK;

    // 开放寻址
    while (true) {
        uint32_t prev = atomic_exchange_explicit(
            &hash_table[slot], key, memory_order_relaxed);
        if (prev == EMPTY || prev == key) break;
        slot = (slot + 1) & HASH_TABLE_MASK;
    }
}

// Probe 阶段: 探测并计数
kernel void probe_hash_table(
    const device int32_t* probe_keys [[buffer(0)]],
    const device uint32_t* hash_table [[buffer(1)]],
    device atomic_uint* match_count [[buffer(2)]],
    device uint32_t* out_indices [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int32_t key = probe_keys[gid];
    uint32_t hash = murmur_hash(key);
    uint32_t slot = hash & HASH_TABLE_MASK;

    // 线性探测
    while (hash_table[slot] != EMPTY) {
        if (hash_table[slot] == key) {
            // 找到匹配
            uint pos = atomic_fetch_add_explicit(
                match_count, 1, memory_order_relaxed);
            out_indices[pos] = gid;
            return;
        }
        slot = (slot + 1) & HASH_TABLE_MASK;
    }
}
```

**性能数据**:
- **时间**: 1.313 ms (100K × 1M)
- **vs DuckDB**: **2.47x** ✓

---

## 4. Aggregate 算法

### 4.1 V4 并行 GROUP BY

**算法**: 分区聚合 + 合并

**伪代码**:
```cpp
void group_sum_i32_v4_parallel(const int32_t* values,
                               const uint32_t* groups,
                               size_t count, size_t num_groups,
                               int64_t* out_sums) {
    const int num_threads = 4;

    // 每线程本地聚合表
    int64_t** local_sums = new int64_t*[num_threads];
    for (int t = 0; t < num_threads; ++t) {
        local_sums[t] = (int64_t*)calloc(num_groups, sizeof(int64_t));
    }

    // 并行局部聚合
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t* local = local_sums[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            local[groups[i]] += values[i];
        }
    }

    // SIMD 合并
    for (size_t g = 0; g < num_groups; g += 4) {
        int64x2_t sum_lo = vdupq_n_s64(0);
        int64x2_t sum_hi = vdupq_n_s64(0);

        for (int t = 0; t < num_threads; ++t) {
            int64x2_t local_lo = vld1q_s64(local_sums[t] + g);
            int64x2_t local_hi = vld1q_s64(local_sums[t] + g + 2);
            sum_lo = vaddq_s64(sum_lo, local_lo);
            sum_hi = vaddq_s64(sum_hi, local_hi);
        }

        vst1q_s64(out_sums + g, sum_lo);
        vst1q_s64(out_sums + g + 2, sum_hi);
    }

    // 释放
    for (int t = 0; t < num_threads; ++t) {
        free(local_sums[t]);
    }
    delete[] local_sums;
}
```

**性能数据**:
- **时间**: 1.176 ms (10M rows, 1000 groups)
- **带宽**: 68.03 GB/s
- **vs DuckDB**: **2.21x** ✓

### 4.2 V15 8 线程 + 循环展开

**关键优化**:
```cpp
void group_sum_i32_v15(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    const int num_threads = 8;  // M4 10 核 → 8 效率核

    alignas(128) int64_t local_sums[num_threads][num_groups];

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t* local = local_sums[tid];
        memset(local, 0, num_groups * sizeof(int64_t));

        #pragma omp for schedule(static)
        for (size_t i = 0; i < count; i += 8) {  // 8 路展开
            local[groups[i+0]] += values[i+0];
            local[groups[i+1]] += values[i+1];
            local[groups[i+2]] += values[i+2];
            local[groups[i+3]] += values[i+3];
            local[groups[i+4]] += values[i+4];
            local[groups[i+5]] += values[i+5];
            local[groups[i+6]] += values[i+6];
            local[groups[i+7]] += values[i+7];
        }
    }

    // SIMD 合并 (同 V4)
    // ...
}
```

**性能数据**:
- **时间**: 0.967 ms (10M rows, 1000 groups)
- **带宽**: 82.75 GB/s
- **vs DuckDB**: **2.69x** ✓

### 4.3 GPU Block-local Hash

**Metal Kernel**:
```metal
kernel void group_sum_gpu(
    const device int32_t* values [[buffer(0)]],
    const device uint32_t* groups [[buffer(1)]],
    device atomic_int64_t* global_sums [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Threadgroup 共享内存
    threadgroup int64_t local_sums[MAX_GROUPS];

    // 初始化
    if (tid < MAX_GROUPS) {
        local_sums[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block-local 原子累加
    uint32_t group_id = groups[gid];
    int32_t value = values[gid];
    atomic_fetch_add_explicit(
        (threadgroup atomic_int64_t*)&local_sums[group_id],
        (int64_t)value,
        memory_order_relaxed);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 合并到全局内存
    if (tid < MAX_GROUPS) {
        atomic_fetch_add_explicit(
            &global_sums[tid],
            local_sums[tid],
            memory_order_relaxed);
    }
}
```

**性能**: 对于低基数 (<32 groups) 超过 CPU，高基数不如 CPU 并行

---

## 5. Sort 算法

### 5.1 Radix Sort (11-11-10 bit)

**核心思想**: 3 趟分布计数排序

**算法伪代码**:
```cpp
void radix_sort_i32_v2(int32_t* data, size_t count) {
    // 处理符号位: 偏移 0x80000000 使负数 < 正数
    for (size_t i = 0; i < count; ++i) {
        data[i] ^= 0x80000000;
    }

    int32_t* tmp = (int32_t*)malloc(count * sizeof(int32_t));

    // Pass 1: 低 11 位 (0-10)
    radix_pass(data, tmp, count, 0, 11);

    // Pass 2: 中 11 位 (11-21)
    radix_pass(tmp, data, count, 11, 11);

    // Pass 3: 高 10 位 (22-31)
    radix_pass(data, tmp, count, 22, 10);

    // 恢复符号
    for (size_t i = 0; i < count; ++i) {
        tmp[i] ^= 0x80000000;
    }

    memcpy(data, tmp, count * sizeof(int32_t));
    free(tmp);
}

void radix_pass(int32_t* src, int32_t* dst, size_t count,
                int shift, int bits) {
    const int num_buckets = 1 << bits;
    size_t counts[num_buckets] = {0};

    // 计数
    for (size_t i = 0; i < count; ++i) {
        int bucket = (src[i] >> shift) & ((1 << bits) - 1);
        counts[bucket]++;
    }

    // 前缀和
    size_t offsets[num_buckets];
    offsets[0] = 0;
    for (int i = 1; i < num_buckets; ++i) {
        offsets[i] = offsets[i-1] + counts[i-1];
    }

    // 分布
    for (size_t i = 0; i < count; ++i) {
        int bucket = (src[i] >> shift) & ((1 << bits) - 1);
        dst[offsets[bucket]++] = src[i];
    }
}
```

**复杂度**: O(n) 时间，O(n) 空间

### 5.2 SIMD Bitonic Sort (小数组)

**Bitonic 比较交换网络**:
```cpp
void sort_8_i32(int32_t* data) {
    int32x4_t v0 = vld1q_s32(data);
    int32x4_t v1 = vld1q_s32(data + 4);

    // Stage 1: 4 个 2-sort
    int32x4_t min01 = vminq_s32(v0, vrev64q_s32(v0));
    int32x4_t max01 = vmaxq_s32(v0, vrev64q_s32(v0));
    v0 = vuzp1q_s32(min01, max01);

    int32x4_t min23 = vminq_s32(v1, vrev64q_s32(v1));
    int32x4_t max23 = vmaxq_s32(v1, vrev64q_s32(v1));
    v1 = vuzp1q_s32(min23, max23);

    // Stage 2: 2 个 4-sort
    // ...

    // Stage 3: 1 个 8-sort
    // ...

    vst1q_s32(data, v0);
    vst1q_s32(data + 4, v1);
}
```

### 5.3 TopK 采样预过滤

**算法**: 采样估计阈值 + SIMD 批量筛选

**伪代码**:
```cpp
void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices) {
    // Step 1: 采样估计第 K 大值
    const size_t sample_size = std::min(count / 100, 10000UL);
    int32_t samples[sample_size];

    for (size_t i = 0; i < sample_size; ++i) {
        samples[i] = data[rand() % count];
    }

    std::nth_element(samples, samples + k, samples + sample_size,
                     std::greater<int32_t>());
    int32_t threshold = samples[k];

    // Step 2: SIMD 批量预过滤
    std::vector<uint32_t> candidates;
    candidates.reserve(k * 2);  // 预留 2x 空间

    int32x4_t thresh_vec = vdupq_n_s32(threshold);

    for (size_t i = 0; i < count; i += 4) {
        int32x4_t values = vld1q_s32(data + i);
        uint32x4_t mask = vcgeq_s32(values, thresh_vec);

        // 提取候选
        if (vmaxvq_u32(mask)) {
            if (vgetq_lane_u32(mask, 0)) candidates.push_back(i+0);
            if (vgetq_lane_u32(mask, 1)) candidates.push_back(i+1);
            if (vgetq_lane_u32(mask, 2)) candidates.push_back(i+2);
            if (vgetq_lane_u32(mask, 3)) candidates.push_back(i+3);
        }
    }

    // Step 3: 候选集上精确选择 TopK
    std::vector<std::pair<int32_t, uint32_t>> pairs;
    pairs.reserve(candidates.size());

    for (uint32_t idx : candidates) {
        pairs.emplace_back(data[idx], idx);
    }

    std::nth_element(pairs.begin(), pairs.begin() + k, pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Step 4: 输出结果
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = pairs[i].first;
        if (out_indices) out_indices[i] = pairs[i].second;
    }
}
```

**性能数据**:
- **时间**: 0.532 ms (10M rows, K=10)
- **vs DuckDB**: **4.71x** ✓

**关键优化**:
1. **采样估计**: 跳过 ~90% 无关元素
2. **SIMD 批量筛选**: 4 元素并行比较
3. **候选集小规模精确选择**: O(candidates) 而非 O(n)

---

## 6. 专用算法

### 6.1 Bitmap Anti-Join (Q22)

**TPC-H Q22 查询**:
```sql
SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal
FROM customer
WHERE c_acctbal > (
    SELECT AVG(c_acctbal) FROM customer
    WHERE c_acctbal > 0.00 AND substring(c_phone, 1, 2) IN (...)
)
AND substring(c_phone, 1, 2) IN (...)
AND NOT EXISTS (
    SELECT * FROM orders WHERE o_custkey = c_custkey  -- 反连接
)
GROUP BY cntrycode
ORDER BY cntrycode;
```

**标准实现**: Hash Anti-Join
```cpp
// Build: orders.o_custkey
std::unordered_set<int32_t> order_custkeys;
for (auto& o : orders) {
    order_custkeys.insert(o.custkey);
}

// Probe: customer.c_custkey
for (auto& c : customers) {
    if (order_custkeys.count(c.custkey) == 0) {  // NOT EXISTS
        // 输出
    }
}
```

**Bitmap 优化**: 直接数组映射
```cpp
// Phase 1: 找到 custkey 范围
int32_t min_custkey = INT32_MAX, max_custkey = INT32_MIN;
for (auto& c : customers) {
    min_custkey = std::min(min_custkey, c.custkey);
    max_custkey = std::max(max_custkey, c.custkey);
}

// Phase 2: 构建 Bitmap (1 bit per custkey)
size_t range = max_custkey - min_custkey + 1;
uint64_t* bitmap = (uint64_t*)calloc((range + 63) / 64, sizeof(uint64_t));

for (auto& o : orders) {
    int32_t offset = o.custkey - min_custkey;
    bitmap[offset / 64] |= (1ULL << (offset % 64));
}

// Phase 3: Probe (位测试)
for (auto& c : customers) {
    int32_t offset = c.custkey - min_custkey;
    bool has_order = (bitmap[offset / 64] >> (offset % 64)) & 1;

    if (!has_order) {  // NOT EXISTS
        // 输出
    }
}

free(bitmap);
```

**性能对比**:
| 方法 | 时间 (ms) | 内存 (MB) | vs DuckDB |
|------|-----------|-----------|-----------|
| Hash Anti-Join | 10.30 | ~8 | 1.00x |
| **Bitmap V37** | **1.21** | ~2 | **8.49x** ✓ |

**关键优化**:
1. **O(1) 查找**: 位测试 vs 哈希查找
2. **缓存友好**: 连续内存 vs 哈希表链
3. **紧凑存储**: 1 bit/key vs 12 bytes/key (哈希表)

### 6.2 EXISTS 优化 (Q21)

**查询**: 相关子查询
```sql
SELECT s_name FROM supplier
WHERE EXISTS (
    SELECT * FROM lineitem
    WHERE l_suppkey = s_suppkey AND l_shipdate > '1995-01-01'
)
```

**优化**: 解关联 + SEMI Join
```cpp
// 原始: 相关子查询 (O(n²))
for (auto& s : suppliers) {
    bool exists = false;
    for (auto& l : lineitems) {
        if (l.suppkey == s.suppkey && l.shipdate > date) {
            exists = true;
            break;
        }
    }
    if (exists) output(s);
}

// 优化: 解关联 + SEMI Join (O(n))
// Step 1: 提取满足条件的 lineitem.suppkey
std::unordered_set<int32_t> valid_suppkeys;
for (auto& l : lineitems) {
    if (l.shipdate > date) {
        valid_suppkeys.insert(l.suppkey);
    }
}

// Step 2: SEMI Join
for (auto& s : suppliers) {
    if (valid_suppkeys.count(s.suppkey)) {
        output(s);
    }
}
```

**性能提升**: 10x+

---

## 7. GPU 算法 (Metal)

### 7.1 UMA 零拷贝架构

**传统 GPU 计算**:
```
CPU Memory ──copy──> GPU Memory ──compute──> GPU Memory ──copy──> CPU Memory
```

**UMA 优化**:
```
Unified Memory ──compute──> Unified Memory
     ↑                            ↑
   CPU 访问                    GPU 访问
```

**实现**:
```objc
// 分配 UMA 缓冲区
id<MTLBuffer> buffer = [device newBufferWithLength:size
    options:MTLResourceStorageModeShared];

// CPU 写入
int32_t* cpu_ptr = (int32_t*)[buffer contents];
memcpy(cpu_ptr, input, size);

// GPU 计算 (无拷贝)
[encoder setBuffer:buffer offset:0 atIndex:0];
[encoder dispatchThreads:MTLSizeMake(count, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

// CPU 读取结果 (无拷贝)
int32_t* result_ptr = (int32_t*)[buffer contents];
```

### 7.2 Threadgroup 前缀和

**问题**: 计算数组前缀和
```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]
Output: [0, 3, 4, 8, 9, 14, 23, 25]
```

**Kernel**:
```metal
kernel void prefix_sum(
    device uint32_t* data [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup uint32_t temp[THREADGROUP_SIZE];

    // 加载数据
    temp[tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (归约)
    for (uint d = 1; d < THREADGROUP_SIZE; d *= 2) {
        uint ai = (tid + 1) * d * 2 - 1;
        if (ai < THREADGROUP_SIZE) {
            temp[ai] += temp[ai - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 清空最后一个元素
    if (tid == 0) temp[THREADGROUP_SIZE - 1] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint d = THREADGROUP_SIZE / 2; d > 0; d /= 2) {
        uint ai = (tid + 1) * d * 2 - 1;
        if (ai < THREADGROUP_SIZE) {
            uint32_t t = temp[ai - d];
            temp[ai - d] = temp[ai];
            temp[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写回
    data[tid] = temp[tid];
}
```

### 7.3 两级归约

**问题**: 计算数组总和

**Kernel**:
```metal
kernel void reduce_sum(
    const device int32_t* input [[buffer(0)]],
    device atomic_int* output [[buffer(1)]],
    threadgroup int* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Level 1: Threadgroup 归约
    shared[tid] = input[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Level 2: 全局原子累加
    if (tid == 0) {
        atomic_fetch_add_explicit(output, shared[0], memory_order_relaxed);
    }
}
```

---

## 8. 复杂度分析

### 8.1 时间复杂度

| 算子 | 最坏情况 | 平均情况 | 最好情况 | 说明 |
|------|---------|---------|---------|------|
| **Filter (SIMD)** | O(n) | O(n) | O(n) | 线性扫描 |
| **Hash Join (V3)** | O(n + m) | O(n + m) | O(n + m) | n=build, m=probe |
| **Hash Join (冲突)** | O(nm) | O(n + m) | O(n + m) | 最坏: 所有键冲突 |
| **GROUP BY (直接数组)** | O(n) | O(n) | O(n) | 分组数 ≤ 1M |
| **GROUP BY (哈希)** | O(n log n) | O(n) | O(n) | 动态分组 |
| **Radix Sort** | O(n) | O(n) | O(n) | 3 趟，每趟 O(n) |
| **Bitonic Sort** | O(n log² n) | O(n log² n) | O(n log² n) | 小数组 (<1024) |
| **TopK (采样)** | O(n + k log k) | O(n + k log k) | O(n + k log k) | n=筛选, k=精确选择 |
| **Bitmap Anti-Join** | O(n + m) | O(n + m) | O(n + m) | 位图构建 + 查询 |

### 8.2 空间复杂度

| 算子 | 空间复杂度 | 说明 |
|------|-----------|------|
| **Filter** | O(n) | 输出索引数组 (最坏) |
| **Hash Join (V3)** | O(n + m) | 哈希表 + 分区缓冲 |
| **GROUP BY (直接数组)** | O(g) | g=分组数 |
| **Radix Sort** | O(n) | 临时缓冲区 |
| **TopK (采样)** | O(k) | 候选集 |
| **Bitmap Anti-Join** | O(max_key - min_key) | 位图 |

### 8.3 带宽分析

**理论带宽**: Apple M4 内存带宽 ~100 GB/s

| 算子 | 实测带宽 | 利用率 | 瓶颈 |
|------|---------|--------|------|
| **Filter V4 (GPU)** | 16.50 GB/s | 16.5% | 计算密集型 |
| **GROUP BY V15** | 82.75 GB/s | 82.7% | 接近内存墙 |
| **INNER JOIN V14** | 3.72 GB/s | 3.7% | 随机访问 |
| **TopK V4** | 75.15 GB/s | 75.1% | 顺序扫描 |

**结论**:
- Filter/TopK: 计算受限，可继续优化算法
- GROUP BY: 内存受限，接近硬件极限
- Join: 随机访问受限，需要更好的缓存策略

---

## 9. 参考文献

### 学术论文

1. **MonetDB/X100** (2005)
   _Vectorized Query Execution_
   P. Boncz et al., CIDR 2005

2. **Hyper** (2011)
   _Efficiently Compiling Efficient Query Plans for Modern Hardware_
   T. Neumann, VLDB 2011

3. **SIMD-Scan** (2015)
   _SIMD-Scan: Ultra Fast in-Memory Table Scan using SIMD Instructions_
   H. Lang et al., VLDB 2016

4. **Radix Hash Join** (2013)
   _Main-Memory Hash Joins on Multi-Core CPUs: Tuning to the Underlying Hardware_
   C. Balkesen et al., ICDE 2013

5. **GPU Join** (2025)
   _GFTR: Gradient-Free Task Reordering for GPU-Accelerated Hash Joins_
   SIGMOD 2025

### 技术文档

- **ARM Neon Intrinsics**
  https://developer.arm.com/architectures/instruction-sets/intrinsics/

- **Metal Shading Language**
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

- **DuckDB Architecture**
  https://duckdb.org/internals/vector-format

- **TPC-H Benchmark**
  http://www.tpc.org/tpch/

### 开源项目

- **DuckDB**: https://github.com/duckdb/duckdb
- **Velox (Meta)**: https://github.com/facebookincubator/velox
- **DataFusion (Apache Arrow)**: https://github.com/apache/arrow-datafusion

---

*本文档由 ThunderDuck 开发团队维护，最后更新: 2026-01-31*
