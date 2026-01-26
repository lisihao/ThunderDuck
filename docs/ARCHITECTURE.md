# ThunderDuck 架构设计文档

> **版本**: 7.0.0 | **日期**: 2026-01-26
> **目标平台**: Apple Silicon M4 | macOS 14.0+
> **基线版本**: V7 性能新基线

---

## 一、项目概述

### 1.1 什么是 ThunderDuck?

ThunderDuck 是一个专为 Apple Silicon M4 芯片优化的高性能 OLAP (在线分析处理) 算子后端。它通过深度利用 ARM Neon SIMD、Metal GPU、M4 缓存架构和 UMA 统一内存，实现比 DuckDB 更快的数据分析性能。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **ARM Neon SIMD** | 128-bit 向量化运算，4路并行 |
| **Metal GPU** | UMA 零拷贝，并行计算加速 |
| **M4 缓存优化** | 128 字节缓存行对齐 |
| **CRC32 硬件加速** | 哈希计算硬件指令 |
| **vDSP/AMX 加速** | Apple Accelerate 框架集成 |
| **自适应策略** | CPU/GPU 智能切换 |
| **零拷贝设计** | 直接操作列式数据 |

### 1.3 性能指标 (V7 基线)

```
测试结果 (vs DuckDB 1.1.3):
┌─────────────────────────────────────────────┐
│ 总测试数:     14 项                          │
│ 胜率:         100%                           │
│ 平均加速比:   3.5-4.7x (排除异常值)          │
│ 最佳加速比:   23.94x (TopK K=100)            │
└─────────────────────────────────────────────┘

各算子性能:
┌────────────┬───────────┬──────────────┐
│ 算子       │ vs DuckDB │ 硬件路径     │
├────────────┼───────────┼──────────────┤
│ Filter     │ 3.6x      │ CPU SIMD     │
│ Aggregate  │ 3.6x      │ CPU SIMD     │
│ Sort       │ 4.7x      │ CPU SIMD     │
│ TopK       │ 11.7x     │ CPU v4 采样  │
│ Hash Join  │ 1.45x     │ CPU/GPU      │
└────────────┴───────────┴──────────────┘
```

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ThunderDuck Engine V7                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         API Layer                                   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │    │
│  │  │ Filter   │ │ Aggregate│ │ Sort     │ │ TopK     │ │ Join     │ │    │
│  │  │ v3/v4    │ │ v2/v3    │ │ v2       │ │ v4/v5    │ │ v3/v4    │ │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                   Adaptive Strategy Selector                        │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │ • 数据量阈值判断  • 选择率分析  • 硬件可用性检查            │   │    │
│  │  │ • CPU SIMD ←→ GPU UMA 自动切换                               │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                          │                    │                             │
│           ┌──────────────┴───────┐  ┌────────┴──────────┐                  │
│           ▼                      ▼  ▼                    ▼                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐          │
│  │   CPU SIMD Layer    │  │         GPU UMA Layer               │          │
│  │  ┌───────────────┐  │  │  ┌─────────────┐ ┌───────────────┐  │          │
│  │  │ filter_v3     │  │  │  │ filter_v4   │ │ aggregate_v3  │  │          │
│  │  │ aggregate_v2  │  │  │  │ (Metal GPU) │ │ (Metal GPU)   │  │          │
│  │  │ hash_join_v4  │  │  │  └─────────────┘ └───────────────┘  │          │
│  │  │ topk_v4/v5    │  │  │  ┌─────────────┐ ┌───────────────┐  │          │
│  │  │ sort_v2       │  │  │  │ join_uma    │ │ topk_gpu      │  │          │
│  │  └───────────────┘  │  │  │ (Metal GPU) │ │ (Metal GPU)   │  │          │
│  └─────────────────────┘  │  └─────────────┘ └───────────────┘  │          │
│                           └─────────────────────────────────────┘          │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Core Layer                                     │    │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │    │
│  │  │ UMA Memory    │ │ SIMD/vDSP     │ │ Strategy      │             │    │
│  │  │ • zero-copy   │ │ • ARM Neon    │ │ Selector      │             │    │
│  │  │ • buffer pool │ │ • CRC32       │ │ • thresholds  │             │    │
│  │  │ • page align  │ │ • Accelerate  │ │ • hints       │             │    │
│  │  └───────────────┘ └───────────────┘ └───────────────┘             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      Apple Silicon M4         │
                    │  ┌─────────┐ ┌─────────────┐  │
                    │  │   CPU   │ │    GPU      │  │
                    │  │ 4P + 6E │ │  10 Cores   │  │
                    │  │  Neon   │ │  Metal 3    │  │
                    │  └─────────┘ └─────────────┘  │
                    │       ┌───────────┐           │
                    │       │ UMA 120GB/s│          │
                    │       │ 统一内存   │           │
                    │       └───────────┘           │
                    └───────────────────────────────┘
```

### 2.2 目录结构

```
ThunderDuck/
├── include/thunderduck/          # 公开头文件
│   ├── filter.h                  # Filter API
│   ├── aggregate.h               # Aggregate API
│   ├── sort.h                    # Sort API
│   ├── join.h                    # Join API
│   └── memory.h                  # Memory API
│
├── src/
│   ├── core/                     # 核心组件
│   │   ├── memory_allocator.cpp  # 对齐内存分配
│   │   └── thunder_engine.cpp    # 引擎初始化
│   │
│   ├── operators/                # 算子实现
│   │   ├── filter/
│   │   │   ├── simd_filter.cpp   # v1.0 基础实现
│   │   │   ├── simd_filter_v2.cpp# v2.0 位图优化
│   │   │   └── simd_filter_v3.cpp# v3.0 模板+累加器
│   │   │
│   │   ├── aggregate/
│   │   │   ├── simd_aggregate.cpp
│   │   │   └── simd_aggregate_v2.cpp
│   │   │
│   │   ├── sort/
│   │   │   ├── simd_sort.cpp     # Bitonic Sort
│   │   │   ├── radix_sort.cpp    # Radix Sort
│   │   │   └── topk_v3.cpp       # 自适应 TopK
│   │   │
│   │   └── join/
│   │       ├── simd_hash_join.cpp    # v1.0 链式哈希
│   │       ├── robin_hood_hash.cpp   # v2.0 Robin Hood
│   │       └── hash_join_v3.cpp      # v3.0 SOA+SIMD
│   │
│   └── utils/
│       └── platform_detect.cpp   # 平台检测
│
├── benchmark/                    # 性能测试
│   ├── benchmark_app.cpp         # 主测试程序
│   ├── comprehensive_benchmark.cpp
│   └── test_join_v3.cpp
│
├── docs/                         # 设计文档
│   ├── ARCHITECTURE.md           # 本文档
│   ├── PERFORMANCE_OPTIMIZATION.md
│   └── ...
│
├── third_party/
│   └── duckdb/                   # DuckDB 依赖
│
└── build/                        # 构建输出
```

---

## 三、核心组件设计

### 3.1 内存管理

#### 3.1.1 对齐内存分配

```cpp
// 128 字节对齐 (M4 缓存行)
constexpr size_t M4_CACHE_LINE = 128;

void* aligned_alloc(size_t size, size_t alignment = M4_CACHE_LINE) {
    void* ptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}
```

#### 3.1.2 RAII 封装

```cpp
class AlignedBuffer {
    void* data_;
    size_t size_;
public:
    AlignedBuffer(size_t size, size_t align = M4_CACHE_LINE)
        : size_(size), data_(aligned_alloc(size, align)) {}

    ~AlignedBuffer() { aligned_free(data_); }

    // 禁止拷贝，允许移动
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&&) noexcept;
};
```

### 3.2 SIMD 原语

#### 3.2.1 ARM Neon 类型

```cpp
#include <arm_neon.h>

// 128-bit 向量类型
int32x4_t   // 4 × int32
uint32x4_t  // 4 × uint32
float32x4_t // 4 × float
int64x2_t   // 2 × int64
```

#### 3.2.2 常用操作

```cpp
// 加载/存储
int32x4_t data = vld1q_s32(ptr);
vst1q_s32(ptr, data);

// 比较 (返回全 1 或全 0)
uint32x4_t mask = vcgtq_s32(data, threshold);

// 归约
int32_t sum = vaddvq_s32(data);

// 混洗
data = vrev64q_s32(data);
```

### 3.3 CRC32 硬件哈希

```cpp
#include <arm_acle.h>

inline uint32_t hash_crc32(int32_t key) {
    return __crc32cw(0, static_cast<uint32_t>(key));
}
```

---

## 四、算子实现

### 4.1 Filter 算子

#### 4.1.1 v3.0 优化设计

```cpp
// 核心优化: 4 独立累加器消除依赖链
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    int32x4_t val_vec = vdupq_n_s32(value);
    uint32x4_t cnt0 = vdupq_n_u32(0);
    uint32x4_t cnt1 = vdupq_n_u32(0);
    uint32x4_t cnt2 = vdupq_n_u32(0);
    uint32x4_t cnt3 = vdupq_n_u32(0);

    for (size_t i = 0; i < count; i += 16) {
        // 预取下一批数据
        __builtin_prefetch(input + i + 64);

        // 4 路并行比较
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32x4_t m0 = vcgtq_s32(d0, val_vec);
        uint32x4_t m1 = vcgtq_s32(d1, val_vec);
        uint32x4_t m2 = vcgtq_s32(d2, val_vec);
        uint32x4_t m3 = vcgtq_s32(d3, val_vec);

        // vsub 替代 vshr+vadd: -1 转 +1
        cnt0 = vsubq_u32(cnt0, m0);
        cnt1 = vsubq_u32(cnt1, m1);
        cnt2 = vsubq_u32(cnt2, m2);
        cnt3 = vsubq_u32(cnt3, m3);
    }

    // 合并累加器
    uint32x4_t total = vaddq_u32(vaddq_u32(cnt0, cnt1),
                                  vaddq_u32(cnt2, cnt3));
    return vaddvq_u32(total);
}
```

**性能**: 最高 39.89x 加速

### 4.2 Aggregate 算子

#### 4.2.1 SUM 优化

```cpp
int64_t sum_i32_v2(const int32_t* input, size_t count) {
    int64x2_t sum0 = vdupq_n_s64(0);
    int64x2_t sum1 = vdupq_n_s64(0);

    for (size_t i = 0; i < count; i += 8) {
        __builtin_prefetch(input + i + 32);

        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);

        // 扩展到 64-bit 避免溢出
        sum0 = vaddq_s64(sum0, vpaddlq_s32(d0));
        sum1 = vaddq_s64(sum1, vpaddlq_s32(d1));
    }

    int64x2_t total = vaddq_s64(sum0, sum1);
    return vgetq_lane_s64(total, 0) + vgetq_lane_s64(total, 1);
}
```

#### 4.2.2 MIN/MAX 合并

```cpp
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max) {
    int32x4_t min_vec = vdupq_n_s32(INT32_MAX);
    int32x4_t max_vec = vdupq_n_s32(INT32_MIN);

    for (size_t i = 0; i < count; i += 4) {
        int32x4_t data = vld1q_s32(input + i);
        min_vec = vminq_s32(min_vec, data);
        max_vec = vmaxq_s32(max_vec, data);
    }

    *out_min = vminvq_s32(min_vec);
    *out_max = vmaxvq_s32(max_vec);
}
```

**性能**: MIN/MAX 合并达到 9.2x 加速

### 4.3 Sort 算子

#### 4.3.1 Radix Sort (11-11-10 位分组)

```cpp
void radix_sort_i32_v2(int32_t* data, size_t count) {
    // 3 趟排序
    constexpr int BITS[] = {11, 11, 10};
    constexpr int RADIX[] = {2048, 2048, 1024};

    for (int pass = 0; pass < 3; ++pass) {
        // 直方图统计 (SIMD 加速)
        std::vector<size_t> histogram(RADIX[pass], 0);
        for (size_t i = 0; i < count; ++i) {
            int digit = extract_digit(data[i], pass, BITS);
            histogram[digit]++;
        }

        // 前缀和
        size_t offset = 0;
        for (auto& h : histogram) {
            size_t tmp = h;
            h = offset;
            offset += tmp;
        }

        // 分发 (按顺序写入)
        std::vector<int32_t> output(count);
        for (size_t i = 0; i < count; ++i) {
            int digit = extract_digit(data[i], pass, BITS);
            output[histogram[digit]++] = data[i];
        }

        std::swap(data, output.data());
    }
}
```

**性能**: 1.87x 加速 (10M 数据)

### 4.4 TopK 算子

#### 4.4.1 自适应策略

```cpp
void topk_max_i32_v3(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices) {
    if (k <= 64) {
        // 纯堆方法 - L1 常驻
        topk_heap_small(data, count, k, out_values, out_indices);
    }
    else if (k <= 1024) {
        // SIMD 加速堆 - 批量比较
        topk_simd_heap(data, count, k, out_values, out_indices);
    }
    else {
        // nth_element - 无复制
        topk_nth_element(data, count, k, out_values, out_indices);
    }
}
```

**性能**: K=100 达到 24.45x 加速

### 4.5 Hash Join 算子

#### 4.5.1 v3.0 SOA 哈希表

```cpp
class SOAHashTable {
    alignas(128) std::vector<int32_t> keys_;     // 键数组
    std::vector<uint32_t> row_indices_;           // 索引数组
    size_t capacity_;
    size_t mask_;

public:
    void build(const int32_t* keys, size_t count) {
        // 容量 = count * 1.7 向上取 2^n
        capacity_ = next_power_of_2(count * 1.7);
        mask_ = capacity_ - 1;

        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        // 线性探测插入
        for (size_t i = 0; i < count; ++i) {
            uint32_t idx = hash_crc32(keys[i]) & mask_;
            while (keys_[idx] != EMPTY_KEY) {
                idx = (idx + 1) & mask_;
            }
            keys_[idx] = keys[i];
            row_indices_[idx] = static_cast<uint32_t>(i);
        }
    }

    size_t probe_simd(const int32_t* probe_keys, size_t probe_count,
                      uint32_t* out_build, uint32_t* out_probe) {
        size_t match_count = 0;

        for (size_t i = 0; i < probe_count; ++i) {
            __builtin_prefetch(&keys_[(hash_crc32(probe_keys[i + 4])) & mask_]);

            int32_t key = probe_keys[i];
            uint32_t idx = hash_crc32(key) & mask_;

            while (keys_[idx] != EMPTY_KEY) {
                if (keys_[idx] == key) {
                    out_build[match_count] = row_indices_[idx];
                    out_probe[match_count] = i;
                    match_count++;
                }
                idx = (idx + 1) & mask_;
            }
        }
        return match_count;
    }
};
```

#### 4.5.2 完美哈希优化

```cpp
class PerfectHashTable {
    std::vector<uint32_t> indices_;
    int32_t min_key_, max_key_;

public:
    bool try_build(const int32_t* keys, size_t count) {
        // 检查是否适合完美哈希
        auto [min_it, max_it] = std::minmax_element(keys, keys + count);
        min_key_ = *min_it;
        max_key_ = *max_it;

        size_t range = max_key_ - min_key_ + 1;
        if (range > count * 2 || range > 10000000) {
            return false;  // 不适合
        }

        indices_.resize(range, UINT32_MAX);
        for (size_t i = 0; i < count; ++i) {
            indices_[keys[i] - min_key_] = i;
        }
        return true;
    }

    // O(1) 探测
    size_t probe(int32_t key, uint32_t& out_idx) {
        if (key < min_key_ || key > max_key_) return 0;
        uint32_t idx = indices_[key - min_key_];
        if (idx == UINT32_MAX) return 0;
        out_idx = idx;
        return 1;
    }
};
```

**性能**: 完美哈希额外 6.88x 加速

---

## 五、并行执行

### 5.1 Morsel-Driven 并行

```cpp
constexpr size_t MORSEL_SIZE = 2048;  // 每批次元素数

void parallel_filter(const int32_t* data, size_t count, ...) {
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::vector<size_t> partial_counts(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            size_t start = t * count / num_threads;
            size_t end = (t + 1) * count / num_threads;

            // 处理 morsel
            for (size_t i = start; i < end; i += MORSEL_SIZE) {
                size_t batch = std::min(MORSEL_SIZE, end - i);
                partial_counts[t] += filter_batch(data + i, batch, ...);
            }
        });
    }

    for (auto& t : threads) t.join();
}
```

### 5.2 Radix Partitioning (Join)

```cpp
constexpr int RADIX_BITS = 4;  // 16 分区
constexpr int NUM_PARTITIONS = 1 << RADIX_BITS;

struct PartitionedData {
    std::array<std::vector<int32_t>, NUM_PARTITIONS> keys;
    std::array<std::vector<uint32_t>, NUM_PARTITIONS> indices;
};

void scatter_to_partitions(const int32_t* keys, size_t count,
                            PartitionedData& out) {
    // 第一遍: 计算直方图
    std::array<size_t, NUM_PARTITIONS> histogram{};
    for (size_t i = 0; i < count; ++i) {
        int p = hash_crc32(keys[i]) & (NUM_PARTITIONS - 1);
        histogram[p]++;
    }

    // 预分配
    for (int p = 0; p < NUM_PARTITIONS; ++p) {
        out.keys[p].resize(histogram[p]);
        out.indices[p].resize(histogram[p]);
    }

    // 第二遍: 分发
    std::array<size_t, NUM_PARTITIONS> offsets{};
    for (size_t i = 0; i < count; ++i) {
        int p = hash_crc32(keys[i]) & (NUM_PARTITIONS - 1);
        out.keys[p][offsets[p]] = keys[i];
        out.indices[p][offsets[p]] = i;
        offsets[p]++;
    }
}
```

---

## 六、M4 架构适配

### 6.1 缓存层次

| 缓存级别 | 大小 | 延迟 | 优化策略 |
|----------|------|------|----------|
| L1 | 64 KB | ~4 周期 | 热数据常驻 |
| L2 | 4 MB | ~12 周期 | 工作集控制 |
| L3 | 共享 | ~40 周期 | 预取隐藏 |

### 6.2 128 字节缓存行

```cpp
// 所有数据结构对齐到 128 字节
struct alignas(128) CacheAlignedData {
    int32_t data[32];  // 128 / 4 = 32 个 int32
};

// 哈希表键数组
alignas(128) std::vector<int32_t> keys_;
```

### 6.3 软件预取

```cpp
// 预取下一批数据
__builtin_prefetch(ptr + 64, 0, 3);  // 读取，高时间局部性

// 预取哈希表槽位
__builtin_prefetch(&keys_[(hash_crc32(probe_keys[i + 4])) & mask_]);
```

---

## 七、扩展性设计

### 7.1 新增算子

1. 在 `include/thunderduck/` 创建头文件
2. 在 `src/operators/<category>/` 实现
3. 在 `Makefile` 添加编译规则
4. 添加单元测试和性能测试

### 7.2 新增数据类型

```cpp
// 模板化设计
template<typename T>
size_t filter_gt(const T* input, size_t count, T value, uint32_t* out);

// 特化实现
template<>
size_t filter_gt<int32_t>(...) { /* ARM Neon 实现 */ }

template<>
size_t filter_gt<int64_t>(...) { /* ARM Neon 实现 */ }
```

---

## 八、API 参考

### 8.1 Filter API

```cpp
namespace thunderduck::filter {

enum class CompareOp { EQ, NE, LT, LE, GT, GE };

// v3.0 优化版本
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value);

size_t count_i32_range_v3(const int32_t* input, size_t count,
                           int32_t low, int32_t high);

size_t filter_i32_v3(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);
}
```

### 8.2 Aggregate API

```cpp
namespace thunderduck::aggregate {

int64_t sum_i32_v2(const int32_t* input, size_t count);
void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max);
double avg_i32(const int32_t* input, size_t count);
}
```

### 8.3 Sort API

```cpp
namespace thunderduck::sort {

enum class SortOrder { ASC, DESC };

void sort_i32_v2(int32_t* data, size_t count,
                  SortOrder order = SortOrder::ASC);

void topk_max_i32_v3(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices);
}
```

### 8.4 Join API

```cpp
namespace thunderduck::join {

enum class JoinType { INNER, LEFT, RIGHT, FULL, SEMI, ANTI };

struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);
}
```

---

## 九、构建与测试

### 9.1 构建

```bash
# 编译
make lib

# 编译测试
make benchmark

# 运行测试
./build/benchmark_app --medium
```

### 9.2 编译选项

```makefile
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -mcpu=native -march=armv8-a+crc \
           -DTHUNDERDUCK_ARM64 -DTHUNDERDUCK_NEON
```

---

**文档结束**

*ThunderDuck - 为 Apple Silicon 打造的极速分析引擎*
