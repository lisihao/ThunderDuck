# ThunderDuck UMA 统一架构设计

> 版本: 2.0.0 | 日期: 2026-01-24
> 目标: 全面利用 Apple M4 统一内存架构，实现零拷贝数据处理

## 一、设计目标

### 1.1 核心目标

| 目标 | 描述 | 预期收益 |
|------|------|---------|
| **零拷贝数据流** | 所有算子共享统一内存池 | 消除 90%+ 数据拷贝 |
| **CPU/GPU 无缝协作** | 同一数据可被两者访问 | GPU 加速无传输开销 |
| **内存复用** | 缓冲区池管理 | 减少 80% 分配开销 |
| **流水线执行** | 算子间直接传递指针 | 消除中间物化 |

### 1.2 适用算子

| 算子 | 当前版本 | UMA 优化版本 | GPU 加速 |
|------|---------|-------------|---------|
| Filter | v3 | v4 UMA | ✅ 计划 |
| Aggregate | v2 | v3 UMA | ✅ 计划 |
| TopK | v5 | v6 UMA | ✅ 计划 |
| Sort | v2 | v3 UMA | ✅ 计划 |
| Join | v4 | **已完成** | ✅ 已实现 |

## 二、统一内存架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ThunderDuck UMA 统一内存架构                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      UMA 内存池 (UMAMemoryPool)                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │ Small   │ │ Medium  │ │ Large   │ │ Huge    │ │ GPU-Mapped  │   │   │
│  │  │ < 64KB  │ │ < 1MB   │ │ < 16MB  │ │ < 256MB │ │ (Shared)    │   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  │       │           │           │           │             │          │   │
│  │       └───────────┴───────────┴───────────┴─────────────┘          │   │
│  │                               │                                     │   │
│  │                    ┌──────────┴──────────┐                         │   │
│  │                    │    统一分配接口      │                         │   │
│  │                    │  uma_alloc(size)    │                         │   │
│  │                    └──────────┬──────────┘                         │   │
│  └───────────────────────────────┼─────────────────────────────────────┘   │
│                                  │                                         │
│  ┌───────────────────────────────┼─────────────────────────────────────┐   │
│  │                    算子执行层 │                                      │   │
│  │                               │                                      │   │
│  │  ┌─────────┐    ┌─────────┐  │  ┌─────────┐    ┌─────────┐        │   │
│  │  │ Filter  │───▶│  Join   │──┼─▶│Aggregate│───▶│  TopK   │        │   │
│  │  │  v4 UMA │    │  UMA    │  │  │  v3 UMA │    │  v6 UMA │        │   │
│  │  └─────────┘    └─────────┘  │  └─────────┘    └─────────┘        │   │
│  │       │              │       │       │              │              │   │
│  │       └──────────────┴───────┼───────┴──────────────┘              │   │
│  │                              │                                      │   │
│  │                    ┌─────────┴─────────┐                           │   │
│  │                    │   UMADataChunk    │  ← 零拷贝传递              │   │
│  │                    │ (统一数据载体)     │                           │   │
│  │                    └───────────────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        硬件抽象层                                    │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │  │   CPU       │  │    GPU      │  │    NPU      │                 │   │
│  │  │ (P+E cores) │  │  (Metal)    │  │   (BNNS)    │                 │   │
│  │  │             │  │             │  │             │                 │   │
│  │  │  SIMD Neon  │  │ Threadgroup │  │  Matrix Ops │                 │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  │         │                │                │                         │   │
│  │         └────────────────┴────────────────┘                         │   │
│  │                          │                                          │   │
│  │              ┌───────────┴───────────┐                              │   │
│  │              │   LPDDR5X 统一内存     │                              │   │
│  │              │   (400+ GB/s 带宽)     │                              │   │
│  │              └───────────────────────┘                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 UMA 数据载体 (UMADataChunk)

```cpp
/**
 * UMA 统一数据载体
 *
 * 特点:
 * - 所有数据存储在 UMA 缓冲区中
 * - CPU/GPU 可直接访问
 * - 算子间零拷贝传递
 */
struct UMADataChunk {
    // 元数据
    size_t count;                    // 行数
    size_t column_count;             // 列数
    DataType* types;                 // 列类型

    // 数据存储 (列式)
    UMABuffer* columns;              // 每列一个 UMA 缓冲区

    // 有效性位图 (NULL 处理)
    UMABuffer validity;              // 位图：1=有效, 0=NULL

    // 选择向量 (过滤后的有效索引)
    UMABuffer selection;             // 选中行的索引
    size_t selected_count;           // 选中行数

    // Metal 句柄 (GPU 访问)
    void** metal_buffers;            // 每列的 Metal 缓冲区

    // 生命周期
    bool owned;                      // 是否拥有内存

    // 方法
    template<typename T>
    T* get_column(size_t col_idx) {
        return columns[col_idx].as<T>();
    }

    id<MTLBuffer> get_metal_buffer(size_t col_idx) {
        return (__bridge id<MTLBuffer>)metal_buffers[col_idx];
    }
};
```

### 2.3 UMA 内存池

```cpp
/**
 * 分层内存池
 *
 * 根据大小分配到不同层级:
 * - Small: 频繁分配的小缓冲区 (< 64KB)
 * - Medium: 中等大小 (< 1MB)
 * - Large: 大数据块 (< 16MB)
 * - Huge: 超大数据 (< 256MB)
 * - GPU-Mapped: GPU 可访问 (MTLStorageModeShared)
 */
class UMAMemoryPool {
public:
    // 分配策略
    enum class AllocStrategy {
        CPU_ONLY,      // 纯 CPU 使用
        GPU_SHARED,    // CPU/GPU 共享
        GPU_PRIVATE    // GPU 专用 (高性能)
    };

    // 分配接口
    UMABuffer allocate(size_t size, AllocStrategy strategy = GPU_SHARED);

    // 从池获取 (优先复用)
    UMABuffer acquire(size_t min_size, AllocStrategy strategy = GPU_SHARED);

    // 归还到池
    void release(UMABuffer& buffer);

    // 预热 (预分配常用大小)
    void warmup(const std::vector<size_t>& sizes);

private:
    // 分层池
    struct Pool {
        std::multimap<size_t, UMABuffer> buffers;
        size_t total_size;
        size_t max_size;
    };

    Pool small_pool_;   // < 64KB
    Pool medium_pool_;  // < 1MB
    Pool large_pool_;   // < 16MB
    Pool huge_pool_;    // < 256MB

    id<MTLDevice> device_;
    std::mutex mutex_;
};
```

## 三、各算子 UMA 优化设计

### 3.1 Filter v4 UMA

#### 3.1.1 当前问题

```cpp
// v3 当前实现
size_t filter_result = simd_filter_gt_i32_v3(
    input,           // 用户内存
    count,
    threshold,
    out_indices      // 用户内存
);
// 问题: 输出缓冲区由用户分配，可能不是 UMA
```

#### 3.1.2 UMA 优化方案

```cpp
/**
 * Filter v4 UMA 设计
 *
 * 优化点:
 * 1. 输入输出都使用 UMADataChunk
 * 2. 位图中间结果直接在 UMA 中
 * 3. GPU 加速大规模过滤
 */

// 接口设计
struct FilterResultUMA {
    UMABuffer indices;           // 匹配行索引
    UMABuffer bitmap;            // 位图表示 (可选)
    size_t count;                // 匹配数量
    id<MTLBuffer> metal_indices; // GPU 可访问
};

FilterResultUMA* filter_uma(
    const UMADataChunk* input,   // UMA 输入
    FilterOp op,                 // 比较操作
    const void* value,           // 比较值
    size_t col_idx               // 过滤列
);

// GPU 加速版本 (大规模)
FilterResultUMA* filter_gpu(
    const UMADataChunk* input,
    FilterOp op,
    const void* value,
    size_t col_idx
);
```

#### 3.1.3 GPU Filter Kernel

```metal
// filter.metal - GPU 向量化过滤

kernel void filter_gt_i32(
    device const int32_t* input [[buffer(0)]],
    device uint32_t* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant int32_t& threshold [[buffer(3)]],
    constant uint32_t& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // 本地缓存
    uint32_t local_indices[8];
    uint local_count = 0;

    // 每线程处理多个元素
    uint base = tid * 8;
    for (uint i = 0; i < 8 && base + i < count; i++) {
        if (input[base + i] > threshold) {
            local_indices[local_count++] = base + i;
        }
    }

    // SIMD 批量写入
    if (local_count > 0) {
        uint offset = atomic_fetch_add_explicit(
            counter, local_count, memory_order_relaxed);
        for (uint i = 0; i < local_count; i++) {
            out_indices[offset + i] = local_indices[i];
        }
    }
}
```

#### 3.1.4 数据流

```
输入 (UMADataChunk)
    │
    ├─ count < 100K ──→ CPU SIMD Filter (v3)
    │                        │
    │                        ▼
    │                   FilterResultUMA
    │                   (直接写入 UMA)
    │
    └─ count >= 100K ──→ GPU Filter Kernel
                             │
                             ▼
                        FilterResultUMA
                        (GPU 直接写入)
```

---

### 3.2 Aggregate v3 UMA

#### 3.2.1 当前问题

```cpp
// v2 当前实现
// 1. 分组键需要哈希表，动态分配
// 2. 聚合结果存储在临时缓冲区
// 3. 最终结果需要拷贝到输出
```

#### 3.2.2 UMA 优化方案

```cpp
/**
 * Aggregate v3 UMA 设计
 *
 * 优化点:
 * 1. 哈希表直接在 UMA 中构建
 * 2. 聚合结果直接写入 UMA
 * 3. GPU 加速分组聚合
 */

// 聚合结果
struct AggregateResultUMA {
    UMADataChunk* groups;        // 分组键
    UMADataChunk* aggregates;    // 聚合结果
    size_t group_count;          // 分组数
};

// 接口
AggregateResultUMA* aggregate_uma(
    const UMADataChunk* input,
    const std::vector<size_t>& group_cols,   // 分组列
    const std::vector<AggFunc>& agg_funcs,   // 聚合函数
    const std::vector<size_t>& agg_cols      // 聚合列
);

// GPU 聚合
AggregateResultUMA* aggregate_gpu(
    const UMADataChunk* input,
    const std::vector<size_t>& group_cols,
    const std::vector<AggFunc>& agg_funcs,
    const std::vector<size_t>& agg_cols
);
```

#### 3.2.3 GPU Group-By Kernel

```metal
// aggregate.metal - GPU 分组聚合

// 阶段 1: 计算分组哈希
kernel void compute_group_hash(
    device const int32_t* group_keys [[buffer(0)]],
    device uint32_t* hashes [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hashes[tid] = hash_key(group_keys[tid]);
}

// 阶段 2: 分区聚合 (每个 threadgroup 处理一个分区)
kernel void partition_aggregate(
    device const int32_t* group_keys [[buffer(0)]],
    device const int64_t* values [[buffer(1)]],
    device const uint32_t* partition_offsets [[buffer(2)]],
    device int32_t* out_groups [[buffer(3)]],
    device int64_t* out_sums [[buffer(4)]],
    device uint32_t* out_counts [[buffer(5)]],
    device atomic_uint* group_counter [[buffer(6)]],
    // Threadgroup 哈希表
    threadgroup int32_t* ht_keys [[threadgroup(0)]],
    threadgroup int64_t* ht_sums [[threadgroup(1)]],
    threadgroup uint32_t* ht_counts [[threadgroup(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // 初始化本地哈希表
    // ...

    // 聚合到本地
    // ...

    // 合并到全局结果
    // ...
}
```

#### 3.2.4 数据流

```
输入 (UMADataChunk)
    │
    ├─ 无分组 ──→ CPU SIMD Aggregate
    │                  │
    │                  ▼
    │             标量结果 (直接返回)
    │
    └─ 有分组 ──→ 基数估计
                      │
         ┌────────────┼────────────┐
         │            │            │
    基数 < 1K    1K-100K      > 100K
         │            │            │
    CPU 哈希表   GPU 分区    GPU 多级
         │       聚合         聚合
         │            │            │
         └────────────┴────────────┘
                      │
                      ▼
              AggregateResultUMA
              (直接在 UMA 中)
```

---

### 3.3 TopK v6 UMA

#### 3.3.1 当前问题

```cpp
// v5 Count-Based TopK
// 1. HashMap 在 CPU 堆上
// 2. 结果需要拷贝
// 3. 大基数时仍然较慢
```

#### 3.3.2 UMA 优化方案

```cpp
/**
 * TopK v6 UMA 设计
 *
 * 优化点:
 * 1. Count 数组直接在 UMA 中
 * 2. GPU 加速计数 (大数据)
 * 3. GPU Radix TopK (大基数)
 */

// TopK 结果
struct TopKResultUMA {
    UMABuffer values;           // TopK 值
    UMABuffer indices;          // 原始索引 (可选)
    size_t k;                   // K 值
};

// 接口
TopKResultUMA* topk_uma(
    const UMADataChunk* input,
    size_t k,
    size_t col_idx,
    bool descending = true
);

// GPU TopK
TopKResultUMA* topk_gpu(
    const UMADataChunk* input,
    size_t k,
    size_t col_idx,
    bool descending = true
);
```

#### 3.3.3 GPU TopK 策略

```cpp
/**
 * GPU TopK 策略选择
 */
TopKStrategy select_topk_strategy(size_t n, size_t k, size_t cardinality) {
    // 小数据: CPU 更快
    if (n < 100000) {
        return TopKStrategy::CPU_HEAP;
    }

    // 低基数: GPU 计数排序
    if (cardinality < 10000) {
        return TopKStrategy::GPU_COUNT_SORT;
    }

    // 小 K: GPU 采样 + 并行选择
    if (k <= 64) {
        return TopKStrategy::GPU_SAMPLE_SELECT;
    }

    // 大 K: GPU Radix Select
    return TopKStrategy::GPU_RADIX_SELECT;
}
```

#### 3.3.4 GPU Count-Based TopK Kernel

```metal
// topk.metal - GPU Count-Based TopK

// 阶段 1: 并行计数
kernel void count_values(
    device const int32_t* values [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],     // 每个值的计数
    constant int32_t& min_value [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    int32_t v = values[tid];
    uint32_t idx = v - min_value;  // 假设值已映射到 [0, cardinality)
    atomic_fetch_add_explicit(&counts[idx], 1, memory_order_relaxed);
}

// 阶段 2: 前缀和 (确定每个值的起始位置)
// 使用 GPU scan 算法

// 阶段 3: 收集 TopK
kernel void collect_topk(
    device const uint32_t* counts [[buffer(0)]],
    device const uint32_t* prefix_sum [[buffer(1)]],
    device int32_t* out_values [[buffer(2)]],
    constant int32_t& min_value [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    constant uint32_t& cardinality [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= k) return;

    // 二分查找 tid 属于哪个值
    // ...
}
```

---

### 3.4 Sort v3 UMA

#### 3.4.1 当前问题

```cpp
// v2 Radix Sort
// 1. 临时缓冲区在 CPU 堆上
// 2. 大数据排序内存密集
// 3. 没有 GPU 加速
```

#### 3.4.2 UMA 优化方案

```cpp
/**
 * Sort v3 UMA 设计
 *
 * 优化点:
 * 1. 所有临时缓冲区在 UMA 中
 * 2. GPU Radix Sort (大数据)
 * 3. 原地排序减少内存使用
 */

// 排序结果
struct SortResultUMA {
    UMABuffer sorted_data;      // 排序后数据 (或原地)
    UMABuffer indices;          // 排序索引 (argsort)
    size_t count;
};

// 接口
SortResultUMA* sort_uma(
    UMADataChunk* input,        // 可能原地修改
    size_t col_idx,
    bool ascending = true,
    bool in_place = false
);

// GPU 排序
SortResultUMA* sort_gpu(
    const UMADataChunk* input,
    size_t col_idx,
    bool ascending = true
);
```

#### 3.4.3 GPU Radix Sort

```metal
// sort.metal - GPU Radix Sort

// 阶段 1: 计算直方图 (每个 digit)
kernel void radix_histogram(
    device const uint32_t* keys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],  // 256 bins
    constant uint32_t& count [[buffer(2)]],
    constant uint32_t& digit [[buffer(3)]],       // 当前处理的 byte (0-3)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    uint32_t key = keys[tid];
    uint32_t bin = (key >> (digit * 8)) & 0xFF;
    atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
}

// 阶段 2: 前缀和 (计算每个 bin 的起始位置)
// ...

// 阶段 3: Scatter (重排数据)
kernel void radix_scatter(
    device const uint32_t* keys_in [[buffer(0)]],
    device const uint32_t* indices_in [[buffer(1)]],
    device const uint32_t* prefix_sum [[buffer(2)]],
    device atomic_uint* counters [[buffer(3)]],
    device uint32_t* keys_out [[buffer(4)]],
    device uint32_t* indices_out [[buffer(5)]],
    constant uint32_t& count [[buffer(6)]],
    constant uint32_t& digit [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint32_t key = keys_in[tid];
    uint32_t idx = indices_in[tid];
    uint32_t bin = (key >> (digit * 8)) & 0xFF;

    uint32_t offset = prefix_sum[bin] + atomic_fetch_add_explicit(
        &counters[bin], 1, memory_order_relaxed);

    keys_out[offset] = key;
    indices_out[offset] = idx;
}
```

---

### 3.5 Join (已实现)

Join 算子的 UMA 优化已经完成，详见 `hash_join_uma.mm`。

核心特点:
1. 零拷贝输入 (`newBufferWithBytesNoCopy`)
2. 缓冲区池复用
3. GPU 并行构建哈希表
4. Shared Events 流水线执行
5. 完全零拷贝输出 (`JoinResultUMA`)

## 四、算子流水线设计

### 4.1 流水线执行模型

```cpp
/**
 * UMA 流水线
 *
 * 特点:
 * - 算子间零拷贝传递 UMADataChunk
 * - 惰性执行 (只在需要时物化)
 * - 支持 CPU/GPU 混合执行
 */

class UMAPipeline {
public:
    // 添加算子
    UMAPipeline& filter(FilterOp op, const void* value, size_t col);
    UMAPipeline& join(const UMADataChunk* other, size_t left_col, size_t right_col);
    UMAPipeline& aggregate(const std::vector<AggFunc>& funcs);
    UMAPipeline& topk(size_t k, size_t col);
    UMAPipeline& sort(size_t col, bool ascending);

    // 执行
    UMADataChunk* execute();

    // 执行策略
    enum class Strategy {
        CPU_ONLY,          // 纯 CPU
        GPU_PREFERRED,     // 优先 GPU
        ADAPTIVE           // 自适应选择
    };

    void set_strategy(Strategy s);

private:
    struct Stage {
        OperatorType type;
        std::function<UMADataChunk*(UMADataChunk*)> execute;
    };

    std::vector<Stage> stages_;
    Strategy strategy_ = Strategy::ADAPTIVE;
};
```

### 4.2 示例流水线

```cpp
// TPC-H Query 6 简化版
// SELECT SUM(l_extendedprice * l_discount) FROM lineitem
// WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
// AND l_discount BETWEEN 0.05 AND 0.07
// AND l_quantity < 24

UMADataChunk* lineitem = load_table_uma("lineitem");

auto result = UMAPipeline(lineitem)
    .filter(FilterOp::GE, &date_1994, col_shipdate)
    .filter(FilterOp::LT, &date_1995, col_shipdate)
    .filter(FilterOp::GE, &discount_005, col_discount)
    .filter(FilterOp::LE, &discount_007, col_discount)
    .filter(FilterOp::LT, &quantity_24, col_quantity)
    .aggregate({AggFunc::SUM}, {col_revenue})  // revenue = price * discount
    .execute();

// 整个流水线:
// 1. 数据始终在 UMA 中
// 2. Filter 间共享选择向量
// 3. 最终聚合直接输出结果
// 无任何数据拷贝!
```

### 4.3 数据流可视化

```
┌─────────────────────────────────────────────────────────────────────┐
│ UMA 流水线执行                                                       │
└─────────────────────────────────────────────────────────────────────┘

lineitem (UMADataChunk)
    │
    │ selection: NULL
    │ count: 6M
    ▼
┌─────────────────┐
│ Filter (date)   │ ← GPU 加速 (6M 行)
└────────┬────────┘
         │ selection: [0, 3, 7, ...]
         │ selected: 1M
         ▼
┌─────────────────┐
│ Filter (disc)   │ ← CPU (已减少到 1M)
└────────┬────────┘
         │ selection: [0, 7, 15, ...]
         │ selected: 100K
         ▼
┌─────────────────┐
│ Filter (qty)    │ ← CPU (100K)
└────────┬────────┘
         │ selection: [7, 15, ...]
         │ selected: 50K
         ▼
┌─────────────────┐
│ Aggregate (SUM) │ ← CPU SIMD (50K 行，无分组)
└────────┬────────┘
         │
         ▼
    结果: 123456.78

内存使用: 仅 lineitem 原始数据
拷贝次数: 0
```

## 五、策略选择框架

### 5.1 自适应策略选择器

```cpp
/**
 * 自适应策略选择
 *
 * 根据:
 * - 数据规模
 * - 数据特征 (基数、选择率)
 * - 硬件可用性
 * - 历史性能
 */

class AdaptiveScheduler {
public:
    // 选择执行设备
    enum class Device {
        CPU,
        GPU,
        NPU
    };

    Device select_device(
        OperatorType op,
        size_t data_size,
        const DataStats& stats
    ) {
        // Filter
        if (op == FILTER) {
            if (data_size < 100000) return CPU;
            if (gpu_available_) return GPU;
            return CPU;
        }

        // Join
        if (op == JOIN) {
            if (data_size < 500000) return CPU;
            if (gpu_available_ && data_size > 1000000) return GPU;
            return CPU;
        }

        // Aggregate
        if (op == AGGREGATE) {
            if (stats.group_count < 1000) return CPU;
            if (gpu_available_ && stats.group_count > 10000) return GPU;
            return CPU;
        }

        // TopK
        if (op == TOPK) {
            if (data_size < 100000) return CPU;
            if (stats.cardinality < 10000) return GPU;  // Count-based
            return CPU;
        }

        // Sort
        if (op == SORT) {
            if (data_size < 500000) return CPU;
            if (gpu_available_) return GPU;
            return CPU;
        }

        return CPU;
    }

private:
    bool gpu_available_;
    bool npu_available_;
    PerformanceHistory history_;
};
```

### 5.2 阈值配置

```cpp
/**
 * 各算子 CPU/GPU 切换阈值
 */
struct OperatorThresholds {
    // Filter
    static constexpr size_t FILTER_GPU_MIN = 100000;

    // Join
    static constexpr size_t JOIN_GPU_MIN_TOTAL = 1000000;
    static constexpr size_t JOIN_GPU_MIN_BUILD = 100000;

    // Aggregate
    static constexpr size_t AGG_GPU_MIN_ROWS = 500000;
    static constexpr size_t AGG_GPU_MIN_GROUPS = 10000;

    // TopK
    static constexpr size_t TOPK_GPU_MIN_N = 100000;
    static constexpr size_t TOPK_GPU_MAX_CARDINALITY = 10000;

    // Sort
    static constexpr size_t SORT_GPU_MIN = 500000;
};
```

## 六、性能预期

### 6.1 各算子预期加速

| 算子 | 数据规模 | CPU (当前) | GPU UMA (预期) | 加速比 |
|------|---------|-----------|---------------|-------|
| Filter | 10M | 5 ms | 1.5 ms | **3.3x** |
| Aggregate (分组) | 10M, 10K 组 | 50 ms | 15 ms | **3.3x** |
| TopK (低基数) | 10M, 基数 100 | 3 ms | 0.8 ms | **3.8x** |
| Sort | 10M | 200 ms | 60 ms | **3.3x** |
| Join | 1M × 10M | 50 ms | 11 ms | **4.6x** (已实现) |

### 6.2 内存节省

| 场景 | 旧实现 | UMA 实现 | 节省 |
|------|--------|---------|------|
| Filter 链 (3 个) | 3 × N 中间结果 | 1 × 选择向量 | **>90%** |
| Join + Aggregate | 结果拷贝 2 次 | 0 次拷贝 | **100%** |
| 流水线 (5 算子) | 5 次分配 | 池复用 | **80%** |

## 七、实现计划

### 7.1 阶段划分

| 阶段 | 内容 | 预计工作量 |
|------|------|-----------|
| **Phase 1** | UMA 内存池优化 (分层、预热) | 2 天 |
| **Phase 2** | Filter v4 UMA + GPU | 3 天 |
| **Phase 3** | Aggregate v3 UMA + GPU | 4 天 |
| **Phase 4** | TopK v6 UMA + GPU | 3 天 |
| **Phase 5** | Sort v3 UMA + GPU | 3 天 |
| **Phase 6** | 流水线执行器 | 3 天 |
| **Phase 7** | 集成测试 + 调优 | 2 天 |

### 7.2 文件结构

```
include/thunderduck/
├── uma_memory.h           # UMA 内存管理 (已完成)
├── uma_data_chunk.h       # 统一数据载体 (新增)
├── uma_pipeline.h         # 流水线执行器 (新增)
├── filter_uma.h           # Filter UMA 接口 (新增)
├── aggregate_uma.h        # Aggregate UMA 接口 (新增)
├── topk_uma.h             # TopK UMA 接口 (新增)
├── sort_uma.h             # Sort UMA 接口 (新增)

src/core/
├── uma_memory.mm          # UMA 内存管理 (已完成)
├── uma_data_chunk.mm      # 数据载体实现 (新增)
├── uma_pipeline.mm        # 流水线实现 (新增)

src/operators/
├── filter/
│   └── filter_uma.mm      # Filter UMA + GPU (新增)
├── aggregate/
│   └── aggregate_uma.mm   # Aggregate UMA + GPU (新增)
├── sort/
│   ├── topk_uma.mm        # TopK UMA + GPU (新增)
│   └── sort_uma.mm        # Sort UMA + GPU (新增)

src/gpu/shaders/
├── filter.metal           # Filter GPU kernel (新增)
├── aggregate.metal        # Aggregate GPU kernel (新增)
├── topk.metal             # TopK GPU kernel (新增)
├── sort.metal             # Sort GPU kernel (新增)
```

## 八、总结

本设计文档提出了一个全面的 UMA 优化架构，核心特点:

1. **统一内存模型**: 所有算子共享 UMA 内存池
2. **零拷贝数据流**: 算子间直接传递指针
3. **CPU/GPU 协同**: 根据数据规模自适应选择
4. **流水线执行**: 最小化中间物化
5. **缓冲区池复用**: 减少分配开销

通过 Join 算子的 UMA 优化已验证了该架构的有效性 (4.6x 加速)，预计其他算子也能获得类似的性能提升。
