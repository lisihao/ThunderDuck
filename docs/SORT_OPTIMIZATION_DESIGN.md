# ThunderDuck 排序算子优化设计

> 版本: V20 | 日期: 2026-01-27

## 一、现状分析

### 1.1 已实现功能

| 功能 | 版本 | 实现方式 | 性能 |
|------|------|----------|------|
| SIMD 小数组排序 | v1 | Bitonic network (4/8/16 元素) | 优秀 |
| Radix Sort | v2 | 11-11-10 bit 分组, 3 趟 | O(n) |
| TopK | v1-v6, v13 | 多策略 (堆/采样/GPU) | **8x vs DuckDB** |
| GPU 排序 | v6, v13 | Metal bitonic | 大数据量有效 |

### 1.2 未实现功能

| 功能 | 优先级 | 预期收益 |
|------|--------|----------|
| 多线程并行排序 | P0 | 4-6x |
| SIMD 归并 | P1 | 1.5-2x (归并阶段) |
| Cache 感知分块 | P2 | 10-20% |
| 外部排序接口 | P3 | DuckDB 集成 |

## 二、P0: 多线程并行排序

### 2.1 设计目标

- 利用 M4 Max 的 10 核心 (8P + 2E)
- 内存带宽约 400 GB/s
- 目标: 排序 10M 整数 < 100ms

### 2.2 接口设计

```cpp
namespace thunderduck::sort {

/**
 * 并行排序配置
 */
struct ParallelSortConfig {
    size_t num_threads = 8;           // 线程数
    size_t block_size = 256 * 1024;   // 每块大小 (元素数)
    bool use_radix = true;            // 大数据使用 radix sort
};

/**
 * 多线程并行排序 - int32
 *
 * 算法:
 * 1. 数据分成 N 段，每段独立排序 (并行)
 * 2. 多路归并合并结果
 *
 * @param data 输入/输出数组
 * @param count 元素数量
 * @param order 排序顺序
 * @param config 配置参数
 */
void parallel_sort_i32(int32_t* data, size_t count,
                       SortOrder order = SortOrder::ASC,
                       const ParallelSortConfig& config = {});

/**
 * 并行排序 - 返回索引
 */
void parallel_argsort_i32(const int32_t* data, size_t count,
                          uint32_t* indices, SortOrder order = SortOrder::ASC,
                          const ParallelSortConfig& config = {});

} // namespace thunderduck::sort
```

### 2.3 算法设计

```
输入: 数组 A[0..N-1], 线程数 T

Phase 1: 并行分段排序
=========================================
将 A 分成 T 段: A0, A1, ..., A(T-1)
并行执行:
  for i in 0..T-1 (parallel):
    if |Ai| < 10000:
      std::sort(Ai)              // 小段用 std::sort
    else:
      radix_sort_i32(Ai)         // 大段用 radix sort

Phase 2: 多路归并
=========================================
方法 A: 顺序两两归并 (简单)
  while 段数 > 1:
    并行合并相邻段对

方法 B: K 路归并 (高效)
  使用最小堆进行 T 路归并
  SIMD 加速比较操作
```

### 2.4 伪代码

```cpp
void parallel_sort_i32(int32_t* data, size_t count, SortOrder order,
                       const ParallelSortConfig& config) {
    if (count < 100000) {
        // 小数据直接排序
        radix_sort_i32(data, count, order);
        return;
    }

    const size_t num_threads = config.num_threads;
    const size_t segment_size = (count + num_threads - 1) / num_threads;

    // Phase 1: 并行分段排序
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * segment_size;
        size_t end = std::min(start + segment_size, count);
        if (start >= count) break;

        threads.emplace_back([=]() {
            radix_sort_i32(data + start, end - start, order);
        });
    }
    for (auto& t : threads) t.join();

    // Phase 2: 多路归并
    std::vector<int32_t> buffer(count);
    merge_k_sorted_segments(data, count, buffer.data(),
                            num_threads, segment_size, order);
    std::memcpy(data, buffer.data(), count * sizeof(int32_t));
}
```

## 三、P1: SIMD 归并优化

### 3.1 当前归并实现

```cpp
// 标量双指针归并 - O(n) 但逐元素比较
while (left_ptr < left_end && right_ptr < right_end) {
    if (*left_ptr <= *right_ptr) {
        *out++ = *left_ptr++;
    } else {
        *out++ = *right_ptr++;
    }
}
```

### 3.2 SIMD 优化设计

```cpp
/**
 * SIMD 加速两路归并
 *
 * 核心思想:
 * - 从两个序列各取 4 元素
 * - 使用 SIMD 比较找出最小的 4 个
 * - 批量输出并移动指针
 */
void simd_merge_i32(const int32_t* left, size_t left_count,
                    const int32_t* right, size_t right_count,
                    int32_t* output) {
    // 批量处理
    while (left_count >= 4 && right_count >= 4) {
        int32x4_t a = vld1q_s32(left);
        int32x4_t b = vld1q_s32(right);

        // Bitonic merge: 将 8 元素合并，取最小 4 个
        int32x4_t min4 = bitonic_merge_min4(a, b);

        vst1q_s32(output, min4);
        output += 4;

        // 根据输出来源移动指针
        // ... (需要追踪哪些元素来自哪个序列)
    }

    // 标量处理剩余
    // ...
}
```

### 3.3 Bitonic Merge Network

```
输入: a[0..3] (已排序), b[0..3] (已排序)
输出: out[0..3] (8 个元素中最小的 4 个)

Step 1: 比较交换
  a[0] <-> b[3]  // 最小值到 a[0]
  a[1] <-> b[2]
  a[2] <-> b[1]
  a[3] <-> b[0]

Step 2: 继续比较直到前 4 个有序
  ...

ARM Neon 实现:
  vminq_s32(), vmaxq_s32() 实现比较交换
  vrev64q_s32() 实现元素重排
```

## 四、P2: Cache 感知分块

### 4.1 M4 缓存参数

| 缓存级别 | 大小 | 延迟 | 带宽 |
|----------|------|------|------|
| L1 (每核) | 128 KB | ~4 cycles | ~800 GB/s |
| L2 (共享) | 4 MB | ~12 cycles | ~400 GB/s |
| DRAM | 36-128 GB | ~100 cycles | ~400 GB/s |

### 4.2 最优分块大小

```cpp
// 排序分块策略
constexpr size_t ELEMENTS_PER_CACHELINE = 128 / sizeof(int32_t);  // 32 元素

// L1 优化: 小数据 (完全 L1 驻留)
constexpr size_t L1_SORT_THRESHOLD = 8 * 1024;  // 8K 元素 (32KB)

// L2 优化: 中等数据 (L2 驻留)
constexpr size_t L2_SORT_THRESHOLD = 512 * 1024;  // 512K 元素 (2MB)

// 超大数据: 流式处理
constexpr size_t STREAM_BLOCK_SIZE = 1024 * 1024;  // 1M 元素 (4MB)
```

### 4.3 自适应策略

```cpp
void adaptive_sort_i32(int32_t* data, size_t count, SortOrder order) {
    if (count <= 16) {
        // SIMD bitonic
        simd_sort_16(data, count, order);
    } else if (count <= L1_SORT_THRESHOLD) {
        // std::sort (L1 友好)
        std::sort(data, data + count);
    } else if (count <= L2_SORT_THRESHOLD) {
        // Radix sort (L2 友好)
        radix_sort_i32(data, count, order);
    } else {
        // 并行分块排序
        parallel_sort_i32(data, count, order);
    }
}
```

## 五、预期性能

### 5.1 基准测试场景

| 数据量 | 当前实现 | 优化后 | 提升 |
|--------|----------|--------|------|
| 1K | std::sort | SIMD bitonic | ~2x |
| 10K | std::sort | radix sort | ~1.5x |
| 1M | radix sort (单线程) | 并行 radix | ~4x |
| 10M | radix sort (单线程) | 并行 radix + SIMD merge | ~5x |
| 100M | 受内存带宽限制 | 并行 + 流水线 | ~6x |

### 5.2 与 DuckDB 对比目标

| 算子 | 当前 vs DuckDB | 目标 vs DuckDB |
|------|----------------|----------------|
| ORDER BY (1M) | ~1x | **2x** |
| ORDER BY (10M) | ~0.8x | **1.5x** |
| TopK (已优化) | **8x** | 保持 |

## 六、实现计划

| 阶段 | 任务 | 预计工作量 |
|------|------|------------|
| V20.1 | P0: 多线程并行排序框架 | 2-3 天 |
| V20.2 | P1: SIMD 归并优化 | 2 天 |
| V20.3 | P2: Cache 感知分块 | 1 天 |
| V20.4 | 基准测试 + 调优 | 1-2 天 |

## 七、不采纳的方案

| 方案 | 原因 |
|------|------|
| SIMD bitonic 扩展到 1024 元素 | 复杂度高，radix sort 更高效 |
| 字符串 SIMD 排序 | 变长类型复杂，收益有限 |
| 稳定排序 | 增加内存开销，多数场景不需要 |
| 外部排序 | DuckDB 有完善的外排序机制 |
