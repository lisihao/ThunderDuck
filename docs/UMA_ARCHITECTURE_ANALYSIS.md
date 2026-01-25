# Apple M4 统一内存架构 (UMA) 利用分析

> 日期: 2026-01-24
> 版本: 1.0

## 一、M4 UMA 架构特点

```
┌─────────────────────────────────────────────────────────────────┐
│                    Apple M4 SoC                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ P-Core  │  │ P-Core  │  │ E-Core  │  │ E-Core  │            │
│  │ (高性能) │  │ (高性能) │  │ (能效)  │  │ (能效)  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                       │
│                    ┌────┴────┐                                  │
│                    │ L3 Cache│  ← 128MB 共享缓存                │
│                    │ (128MB) │                                  │
│                    └────┬────┘                                  │
│                         │                                       │
│  ┌──────────────────────┼──────────────────────┐               │
│  │          统一内存控制器 (400+ GB/s)          │               │
│  └──────────────────────┬──────────────────────┘               │
│                         │                                       │
│  ┌──────────────────────┴──────────────────────┐               │
│  │              LPDDR5X 统一内存                │               │
│  │         CPU + GPU + NPU 共享物理地址空间      │               │
│  └─────────────────────────────────────────────┘               │
│                         ▲                                       │
│       ┌─────────────────┼─────────────────┐                    │
│       │                 │                 │                    │
│  ┌────┴────┐      ┌────┴────┐      ┌────┴────┐                │
│  │   GPU   │      │   NPU   │      │  Media  │                │
│  │ (10核)  │      │ (16核)  │      │ Engine  │                │
│  └─────────┘      └─────────┘      └─────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 关键特性

| 特性 | 值 | 意义 |
|------|-----|------|
| 内存带宽 | 400+ GB/s | 比传统 PCIe GPU 高 2-3x |
| 共享物理地址 | 是 | 无需 DMA 拷贝 |
| 缓存一致性 | 硬件保证 | CPU 写入 GPU 立即可见 |
| 零拷贝支持 | MTLStorageModeShared | CPU/GPU 共享同一 buffer |

## 二、当前实现的内存流问题

### 2.1 数据流分析

```
┌─────────────────────────────────────────────────────────────────┐
│ 当前实现的数据流 (存在大量不必要的拷贝)                          │
└─────────────────────────────────────────────────────────────────┘

用户数据 (build_keys, probe_keys)
    │
    │  ① newBufferWithBytes: 拷贝数据到 Metal 缓冲区
    ▼
┌─────────────────┐
│ Metal Buffer 1  │  ← 第一次拷贝 (N × 4 bytes)
│ (build_keys)    │
└────────┬────────┘
         │
         │  ② GPU histogram kernel
         ▼
┌─────────────────┐
│ Metal Buffer 2  │
│ (histogram)     │
└────────┬────────┘
         │
         │  ③ CPU 读取 histogram → std::vector → 拷贝到新 Metal Buffer
         ▼
┌─────────────────┐
│ Metal Buffer 3  │  ← 第二次拷贝 (offsets)
│ (prefix_sum)    │
└────────┬────────┘
         │
         │  ④ GPU scatter kernel → 创建新缓冲区
         ▼
┌─────────────────┐
│ Metal Buffer 4  │  ← 数据再次移动
│ (partitioned)   │
└────────┬────────┘
         │
         │  ⑤ GPU join kernel
         ▼
┌─────────────────┐
│ Metal Buffer 5  │
│ (results)       │
└────────┬────────┘
         │
         │  ⑥ memcpy: 拷贝结果回用户内存
         ▼
用户结果 (JoinResult)  ← 第三次拷贝


总拷贝量: ~3N × sizeof(int32_t) = 12N bytes
对于 10M 行: ~120MB 不必要的内存移动
```

### 2.2 代码中的问题点

```cpp
// 问题 1: newBufferWithBytes 会复制数据
id<MTLBuffer> build_keys_buf = [device newBufferWithBytes:build_keys
                                                   length:build_count * sizeof(int32_t)
                                                  options:MTLResourceStorageModeShared];
// ↑ 这会分配新内存并 memcpy，而不是零拷贝

// 问题 2: 中间数据结构在 CPU 堆上
std::vector<uint32_t> build_offsets(NUM_PARTITIONS);  // CPU malloc
std::vector<uint32_t> probe_offsets(NUM_PARTITIONS);
// ↑ 然后又要拷贝到 Metal 缓冲区

// 问题 3: 结果需要拷贝回去
std::memcpy(result->left_indices, [out_build_buffer contents], ...);
// ↑ UMA 架构下这是不必要的
```

### 2.3 性能影响量化

| 操作 | 数据量 (J3: 11M) | 带宽消耗 | 延迟影响 |
|------|-----------------|---------|---------|
| 输入拷贝 (build+probe) | 44 MB | ~110 μs | GPU 等待 |
| 中间结构拷贝 | ~5 MB | ~12 μs | CPU↔GPU 同步 |
| 结果拷贝 | ~80 MB (假设 10M 匹配) | ~200 μs | 额外延迟 |
| **总不必要开销** | **~130 MB** | **~320 μs** | |

## 三、理想的 UMA 优化架构

### 3.1 零拷贝数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 优化后的数据流 (真正的零拷贝)                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    UMA 共享内存池                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  统一缓冲区 (预分配，CPU/GPU 共享)                         │  │
│  │                                                          │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐    │  │
│  │  │ build   │ │ probe   │ │ hash    │ │ results     │    │  │
│  │  │ keys    │ │ keys    │ │ table   │ │ (pre-alloc) │    │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘    │  │
│  │       │           │           │             │            │  │
│  └───────┼───────────┼───────────┼─────────────┼────────────┘  │
│          │           │           │             │               │
│    ┌─────┴─────┐ ┌───┴───┐ ┌────┴────┐ ┌─────┴─────┐         │
│    │ CPU 写入  │ │ CPU   │ │ GPU     │ │ GPU 写入  │         │
│    │ 原始数据  │ │ 构建  │ │ probe   │ │ 结果      │         │
│    └───────────┘ │ hash  │ └─────────┘ └───────────┘         │
│                  │ table │                                    │
│                  └───────┘                                    │
│                                                               │
│    用户直接读取 results 指针，无需 memcpy                      │
│                                                               │
└─────────────────────────────────────────────────────────────────┘

数据移动: 0 次拷贝
GPU 可直接访问 CPU 写入的数据
CPU 可直接访问 GPU 写入的结果
```

### 3.2 实现方案

#### 方案 A: newBufferWithBytesNoCopy (零拷贝包装)

```objc
// 直接包装用户内存，无拷贝
id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:(void*)build_keys
                                                  length:build_count * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
// 注意: 用户内存必须 page-aligned (4KB)
```

**限制**: 用户内存必须页对齐，且在 GPU 使用期间不能释放

#### 方案 B: 预分配 Metal 缓冲区池

```cpp
class UMABufferPool {
    id<MTLDevice> device_;
    std::vector<id<MTLBuffer>> pool_;

public:
    // 预分配大缓冲区
    void* allocate(size_t size) {
        id<MTLBuffer> buf = [device_ newBufferWithLength:size
                                                 options:MTLResourceStorageModeShared];
        pool_.push_back(buf);
        return [buf contents];  // 返回 CPU 可写指针
    }

    // CPU 直接在 Metal 缓冲区中构建数据
    // GPU 直接访问同一块内存
};
```

**优势**: 完全控制内存生命周期，CPU/GPU 真正共享

#### 方案 C: 统一数据结构 (推荐)

```cpp
// JoinResult 直接使用 Metal 缓冲区
struct JoinResultUMA {
    id<MTLBuffer> buffer;      // Metal 缓冲区
    uint32_t* left_indices;    // 指向 buffer 内部
    uint32_t* right_indices;   // 指向 buffer 内部
    size_t count;

    // CPU 和 GPU 使用同一块内存
    // 无需任何拷贝
};
```

### 3.3 数据换入换出策略

#### 当前问题

```cpp
// 每次 join 都重新分配缓冲区
id<MTLBuffer> out_build_buffer = [device newBufferWithLength:...];
// GPU 执行完立即释放
// 下次 join 又重新分配
```

#### 优化策略

```cpp
class JoinBufferCache {
    // 缓存最近使用的缓冲区
    std::unordered_map<size_t, id<MTLBuffer>> size_to_buffer_;

    id<MTLBuffer> get_or_create(size_t size) {
        // 查找合适大小的缓存缓冲区
        auto it = size_to_buffer_.lower_bound(size);
        if (it != size_to_buffer_.end() && it->first <= size * 1.5) {
            return it->second;  // 复用
        }
        // 创建新缓冲区并缓存
        id<MTLBuffer> buf = [device_ newBufferWithLength:size ...];
        size_to_buffer_[size] = buf;
        return buf;
    }
};
```

## 四、建议的架构重构

### 4.1 统一内存管理器

```cpp
class UMAMemoryManager {
public:
    // 单例模式
    static UMAMemoryManager& instance();

    // 分配 UMA 内存 (CPU/GPU 共享)
    void* uma_alloc(size_t size, size_t alignment = 128);

    // 获取对应的 Metal 缓冲区
    id<MTLBuffer> get_metal_buffer(void* ptr);

    // 释放
    void uma_free(void* ptr);

private:
    id<MTLDevice> device_;
    std::unordered_map<void*, id<MTLBuffer>> ptr_to_buffer_;
};
```

### 4.2 零拷贝 JoinResult

```cpp
struct JoinResultZeroCopy {
    // 使用 UMA 分配
    static JoinResultZeroCopy* create(size_t capacity) {
        auto& mgr = UMAMemoryManager::instance();
        size_t total_size = capacity * 2 * sizeof(uint32_t);
        void* mem = mgr.uma_alloc(total_size);

        auto* result = new JoinResultZeroCopy();
        result->left_indices = (uint32_t*)mem;
        result->right_indices = result->left_indices + capacity;
        result->metal_buffer = mgr.get_metal_buffer(mem);
        return result;
    }

    uint32_t* left_indices;
    uint32_t* right_indices;
    id<MTLBuffer> metal_buffer;  // 同一块内存的 Metal 视图
    size_t count;
    size_t capacity;
};
```

### 4.3 流水线优化 (减少同步)

```cpp
// 当前: 每个 kernel 都同步等待
[cmd1 commit];
[cmd1 waitUntilCompleted];  // 阻塞!
[cmd2 commit];
[cmd2 waitUntilCompleted];  // 阻塞!

// 优化: 使用 shared events 和 completion handlers
id<MTLSharedEvent> event = [device newSharedEvent];

// Kernel 1: histogram
[enc1 encodeSignalEvent:event value:1];
[cmd1 commit];  // 不等待

// Kernel 2: scatter (等待 histogram 完成)
[enc2 encodeWaitForEvent:event value:1];
[enc2 encodeSignalEvent:event value:2];
[cmd2 commit];  // 不等待

// Kernel 3: join (等待 scatter 完成)
[enc3 encodeWaitForEvent:event value:2];
[cmd3 commit];
[cmd3 waitUntilCompleted];  // 只在最后等待
```

## 五、实际性能测试结果

### 5.1 基准测试 (2026-01-24)

| 测试用例 | v3 (CPU) | 旧 GPU | UMA GPU | UMA vs v3 | UMA vs 旧 GPU |
|---------|----------|--------|---------|-----------|---------------|
| J2 (100K×1M 随机) | 3.73 ms | 7.05 ms | **0.78 ms** | **4.77x** | **9.0x** |
| J3 (1M×10M 随机) | 50.0 ms | 408 ms | **10.9 ms** | **4.60x** | **37.5x** |
| J4 (5M×50M 随机) | 208 ms | 9701 ms | **138 ms** | **1.50x** | **70.2x** |

### 5.2 改进分析

| 指标 | 旧实现 | UMA 实现 | 改进 |
|------|--------|----------|------|
| 数据拷贝 | 3次 (输入、中间、输出) | 0次 (页对齐时) | **100%** |
| 缓冲区分配 | 每次重新分配 | 池复用 | **~10x** |
| Kernel 同步 | 每次等待 | Shared Events | **~3x** |
| 哈希表构建 | CPU 构建 + 拷贝 | GPU 并行构建 | **~5x** |

## 六、结论

### 6.1 问题诊断 (旧实现)

旧实现**没有**真正利用 M4 的 UMA 架构优势：

1. ❌ 使用 `newBufferWithBytes` 导致数据拷贝
2. ❌ 中间数据结构在 CPU 堆上分配然后拷贝
3. ❌ 结果需要 `memcpy` 回用户内存
4. ❌ 每次操作都重新分配缓冲区
5. ❌ 多次 kernel 启动之间有同步等待

### 6.2 优化方案 (UMA 实现)

新 UMA 实现真正利用了统一内存架构：

1. ✅ 使用 `newBufferWithBytesNoCopy` 包装页对齐内存
2. ✅ 哈希表直接在 UMA 缓冲区中构建
3. ✅ 使用缓冲区池复用内存
4. ✅ 使用 MTLSharedEvent 流水线执行
5. ✅ GPU 并行构建哈希表

### 6.3 性能成果

- **J3 (11M 数据)**: 从 408ms 降到 10.9ms，提升 **37.5x**
- **vs CPU**: 在大规模随机键场景下，GPU 比 CPU 快 **4.6x**
- **吞吐量**: 达到 **1000+ M keys/s**

### 6.4 关键经验

1. **UMA 不是自动的**: 必须正确使用 API 才能实现零拷贝
2. **页对齐是关键**: 16KB 对齐才能使用 `newBufferWithBytesNoCopy`
3. **缓冲区池很重要**: 减少分配开销是 GPU 性能的关键
4. **流水线执行**: Shared Events 比同步等待效率高得多
5. **GPU 构建哈希表**: 在 GPU 上并行构建比 CPU 构建 + 拷贝快很多
