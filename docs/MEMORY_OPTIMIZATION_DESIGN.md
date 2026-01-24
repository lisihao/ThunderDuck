# ThunderDuck 内存优化设计文档

> **版本**: 1.0.0 | **日期**: 2026-01-24
> **作者**: Claude Code Assistant
> **状态**: 设计阶段

## 一、执行摘要

本文档深入分析 ThunderDuck 和 DuckDB 的内存使用模式，识别 ThunderDuck 中的内存过度使用问题，并提出针对 Apple M4 处理器架构的深度优化方案。

### 关键发现

| 指标 | ThunderDuck | DuckDB | 分析 |
|------|-------------|--------|------|
| 数据装载开销 | 1.0x | 2.5x | ThunderDuck 更高效 |
| 哈希表开销 | 3.4x | ~2.0x | ThunderDuck 有优化空间 |
| 临时缓冲区 | 最高 99%+ 浪费 | 流式处理 | 严重问题 |
| 内存池 | 无 | 统一 Buffer Manager | 需要实现 |

### 优化目标

- 减少哈希表内存开销 30%
- 消除临时缓冲区浪费
- 实现内存池复用
- 支持大于内存的数据集（溢出到磁盘）

---

## 二、当前内存使用分析

### 2.1 ThunderDuck 内存分配模式

#### 2.1.1 数据存储层

```
当前实现: 直接使用 std::vector
┌─────────────────────────────────────────────────┐
│ std::vector<int32_t> column_data               │
│ - 连续内存分配                                  │
│ - 容量可能 > 实际大小（vector growth）         │
│ - 无压缩、无编码                               │
└─────────────────────────────────────────────────┘
内存效率: ~100% (几乎无开销)
```

**优点**: 简单直接，访问速度快
**缺点**: 无压缩、无列式编码优化

#### 2.1.2 哈希表层 (SOAHashTable)

```cpp
// 当前实现 (hash_join_v3.cpp:125-259)
class SOAHashTable {
    alignas(128) std::vector<int32_t> keys_;      // 4B per slot
    std::vector<uint32_t> row_indices_;            // 4B per slot
    size_t capacity_;  // = count * 1.7 (向上取 2^n)
};

容量计算:
  capacity = 16
  while (capacity < count * 1.7)
      capacity *= 2

示例内存计算:
  ┌──────────┬──────────┬──────────┬──────────┬─────────┐
  │ 元素数量 │ 原始数据 │ 容量     │ 哈希表   │ 开销    │
  ├──────────┼──────────┼──────────┼──────────┼─────────┤
  │ 1,000    │ 4 KB     │ 2,048    │ 16 KB    │ 4.0x    │
  │ 10,000   │ 40 KB    │ 32,768   │ 256 KB   │ 6.4x    │
  │ 100,000  │ 400 KB   │ 262,144  │ 2 MB     │ 5.2x    │
  │ 1,000,000│ 4 MB     │ 2,097,152│ 16 MB    │ 4.2x    │
  └──────────┴──────────┴──────────┴──────────┴─────────┘
```

**问题**:
1. 负载因子 1.7 导致容量膨胀
2. 向上取 2^n 进一步放大
3. 每个条目 8 字节 (key + index)，比原始数据 2x

#### 2.1.3 JoinResult 缓冲区

```cpp
// 当前实现 (simd_hash_join.cpp:373-414)
struct JoinResult {
    uint32_t* left_indices;   // aligned_alloc
    uint32_t* right_indices;  // aligned_alloc
    size_t count;
    size_t capacity;
};

// 预分配策略 (hash_join_v3.cpp:702-706)
size_t estimated_matches = std::max(build_count, probe_count) * 4;

// 增长策略 (2x exponential)
void grow_join_result(JoinResult* result, size_t min_capacity) {
    size_t new_capacity = result->capacity * 2;
    while (new_capacity < min_capacity)
        new_capacity *= 2;
    // memcpy 全量拷贝
}
```

**严重问题 - 稀疏 Join 场景**:

| 场景 | probe_count | 实际匹配 | 预分配 | 浪费率 |
|------|-------------|----------|--------|--------|
| 极稀疏 | 1M | 100 | 32 MB | 99.999% |
| 稀疏 | 1M | 10K | 32 MB | 99.9% |
| 中等 | 1M | 100K | 32 MB | 96.9% |
| 密集 | 1M | 1M | 32 MB | 0% |

#### 2.1.4 临时缓冲区 (分区 Join)

```cpp
// 每个分区每个线程分配 (hash_join_v3.cpp:576-577, 630-631)
std::vector<uint32_t> temp_build(probe_count);
std::vector<uint32_t> temp_probe(probe_count);

// 多线程场景
16 分区 × 4 线程 × (probe_count/16) × 8 bytes = probe_count × 8 bytes
```

**问题**: 临时缓冲区按 probe_count 分配，不考虑实际匹配数

---

### 2.2 DuckDB 内存管理架构

基于 [DuckDB Memory Management](https://duckdb.org/2024/07/09/memory-management) 官方文档分析：

#### 2.2.1 统一 Buffer Manager

```
┌─────────────────────────────────────────────────────────────┐
│                    DuckDB Buffer Manager                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 持久化页面  │  │ 临时数据   │  │ 中间结果   │         │
│  │ (数据表)    │  │ (排序/聚合) │  │ (Hash表)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                          │                                   │
│                    ┌─────┴─────┐                            │
│                    │ 统一内存池 │ ← 默认 80% 系统 RAM       │
│                    └─────┬─────┘                            │
│                          │                                   │
│                    ┌─────┴─────┐                            │
│                    │ 磁盘溢出  │ ← 超过限制时自动触发       │
│                    └───────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

**关键设计**:
1. **轻量级去中心化**: 类似 LeanStore，无全局锁
2. **大页面**: 256KB (vs 传统 4-16KB)
3. **智能驱逐**: 针对分析查询优化（非 LRU）
4. **透明溢出**: 超限时自动写入临时文件

#### 2.2.2 流式执行模型

```
DuckDB 流式处理:
┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
│ Scan  │───>│ Filter│───>│ Hash  │───>│ Agg   │
│ Chunk │    │ Chunk │    │ Build │    │ Chunk │
└───────┘    └───────┘    └───────┘    └───────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                  向量化批处理
                  (2048 rows/batch)
```

**优点**:
- 数据不完整加载到内存
- 增量处理，内存占用稳定
- 自动并行化

---

### 2.3 内存使用对比测试结果

```
测试环境: Apple M4, 16GB RAM
数据集: 1M 行 × 3 列 (int32, int32, double)

┌─────────────────────┬──────────────┬──────────────┬─────────┐
│ 测试项目            │ ThunderDuck  │ DuckDB       │ 差异    │
├─────────────────────┼──────────────┼──────────────┼─────────┤
│ 数据装载            │ 15.30 MB     │ 38.31 MB     │ -60.1%  │
│ 原始数据大小        │ 15.26 MB     │ 15.26 MB     │ -       │
│ 装载开销            │ 1.00x        │ 2.51x        │ -       │
└─────────────────────┴──────────────┴──────────────┴─────────┘

分析:
- ThunderDuck 数据装载近乎零开销（直接 vector）
- DuckDB 有 2.5x 开销（列式存储、元数据、Buffer Manager）
```

---

## 三、问题总结与优化目标

### 3.1 识别的内存问题

| 优先级 | 问题 | 影响范围 | 浪费估算 |
|--------|------|----------|----------|
| P0 | JoinResult 预分配过大 | 所有 Join | 最高 99%+ |
| P0 | 临时缓冲区按 probe_count | 分区 Join | 最高 99%+ |
| P1 | 哈希表负载因子 1.7 | Hash Join | 30% |
| P1 | 指数增长策略 (2x) | 动态分配 | 平均 50% |
| P2 | 过度对齐 (128B) | 小对象 | 变化大 |
| P2 | 缺少内存池 | 频繁分配 | 碎片化 |

### 3.2 优化目标

| 目标 | 当前 | 目标 | 改善 |
|------|------|------|------|
| 哈希表开销 | 3.4x | 2.0x | -41% |
| JoinResult 浪费 | 最高 99% | <20% | 显著 |
| 临时缓冲区 | probe_count | 实际匹配数 | 消除浪费 |
| 内存碎片 | 无管理 | 内存池 | 减少碎片 |

---

## 四、优化方案设计

### 4.1 方案 1: 智能 JoinResult 分配

#### 4.1.1 选择率估算

```cpp
// 新增: 基于直方图的选择率估算
struct JoinSelectivityEstimator {
    // 方法 1: 采样估算
    static double estimate_by_sampling(
        const int32_t* build_keys, size_t build_count,
        const int32_t* probe_keys, size_t probe_count,
        size_t sample_size = 1000) {

        // 随机采样 probe keys
        std::vector<int32_t> sample(sample_size);
        // ... 采样逻辑

        // 在 build 中查找匹配数
        size_t sample_matches = count_matches(sample, build_keys);

        // 外推到全量
        return static_cast<double>(sample_matches) / sample_size;
    }

    // 方法 2: 基于 NDV (Number of Distinct Values) 估算
    static size_t estimate_matches(
        size_t build_count, size_t build_ndv,
        size_t probe_count, size_t probe_ndv) {

        // 假设均匀分布
        double overlap = std::min(build_ndv, probe_ndv);
        double avg_build_per_key = static_cast<double>(build_count) / build_ndv;
        double avg_probe_per_key = static_cast<double>(probe_count) / probe_ndv;

        return static_cast<size_t>(overlap * avg_build_per_key * avg_probe_per_key);
    }
};
```

#### 4.1.2 增量分配策略

```cpp
// 新增: 分段增量分配
class IncrementalJoinResult {
    static constexpr size_t CHUNK_SIZE = 65536;  // 64K entries per chunk

    struct Chunk {
        std::unique_ptr<uint32_t[]> left_indices;
        std::unique_ptr<uint32_t[]> right_indices;
        size_t count = 0;
    };

    std::vector<Chunk> chunks_;
    size_t total_count_ = 0;

public:
    void add_match(uint32_t left_idx, uint32_t right_idx) {
        if (chunks_.empty() || chunks_.back().count >= CHUNK_SIZE) {
            // 分配新 chunk
            chunks_.emplace_back();
            chunks_.back().left_indices.reset(new uint32_t[CHUNK_SIZE]);
            chunks_.back().right_indices.reset(new uint32_t[CHUNK_SIZE]);
        }

        auto& chunk = chunks_.back();
        chunk.left_indices[chunk.count] = left_idx;
        chunk.right_indices[chunk.count] = right_idx;
        chunk.count++;
        total_count_++;
    }

    // 最终合并（可选）
    JoinResult* finalize();
};

// 内存使用: 按需分配，每次 64K × 8B = 512KB
// 稀疏 Join: 1000 匹配 → 1 chunk = 512KB (vs 当前 32MB)
```

### 4.2 方案 2: 优化哈希表设计

#### 4.2.1 更紧凑的哈希表

```cpp
// 优化: Compact Hash Table
class CompactHashTable {
    // 方案 A: 更高负载因子 + Robin Hood
    static constexpr double LOAD_FACTOR = 0.8;  // vs 当前 0.59 (1/1.7)

    // 方案 B: 分离链表 (Separate Chaining with Pooled Nodes)
    struct alignas(16) Entry {  // 16 字节对齐
        int32_t key;
        uint32_t row_idx;
        uint32_t next;  // 链表指针
        uint32_t padding;
    };

    // 方案 C: Cuckoo Hashing (更高效的空间利用)
    // 使用两个哈希函数，保证 O(1) 查找
};

// 内存节省计算:
// 当前: capacity = count * 1.7, 8B/entry → 13.6B/key
// 优化: capacity = count * 1.25, 8B/entry → 10B/key
// 节省: 26%
```

#### 4.2.2 内联小值优化

```cpp
// 小表优化: 内联存储
class InlineHashTable {
    // 对于 < 64 个条目，使用线性扫描
    static constexpr size_t INLINE_THRESHOLD = 64;

    union {
        struct {
            alignas(64) int32_t keys[INLINE_THRESHOLD];
            uint32_t indices[INLINE_THRESHOLD];
            uint8_t count;
        } inline_data;

        struct {
            int32_t* keys;
            uint32_t* indices;
            size_t capacity;
        } heap_data;
    };

    bool is_inline_;

public:
    // 小表: 无额外开销，缓存友好
    // 大表: 正常哈希表
};
```

### 4.3 方案 3: 内存池架构

```cpp
// 新增: 线程本地内存池
class ThreadLocalMemoryPool {
    static constexpr size_t BLOCK_SIZE = 1 << 20;  // 1MB blocks
    static constexpr size_t MAX_CACHED_BLOCKS = 4;

    struct Block {
        alignas(128) char data[BLOCK_SIZE];
        size_t used = 0;
    };

    thread_local static std::vector<std::unique_ptr<Block>> free_blocks_;
    thread_local static Block* current_block_;

public:
    static void* allocate(size_t size, size_t alignment = 16) {
        // 对齐计算
        size_t aligned_offset = (current_block_->used + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + size > BLOCK_SIZE) {
            // 当前 block 不足，获取新 block
            if (!free_blocks_.empty()) {
                current_block_ = free_blocks_.back().release();
                free_blocks_.pop_back();
            } else {
                current_block_ = new Block();
            }
            aligned_offset = 0;
        }

        void* ptr = current_block_->data + aligned_offset;
        current_block_->used = aligned_offset + size;
        return ptr;
    }

    static void reset() {
        // 回收当前 block 到 free list
        if (current_block_ && free_blocks_.size() < MAX_CACHED_BLOCKS) {
            current_block_->used = 0;
            free_blocks_.push_back(std::unique_ptr<Block>(current_block_));
        } else {
            delete current_block_;
        }
        current_block_ = nullptr;
    }
};

// 使用示例:
// 每个 Join 操作开始时分配，结束时 reset()
// 避免频繁 malloc/free
```

### 4.4 方案 4: 直接输出优化

```cpp
// 消除临时缓冲区: 直接写入最终结果
size_t join_partition_direct(
    const Partition& build_part,
    const Partition& probe_part,
    JoinResult* result,  // 直接输出
    std::atomic<size_t>& write_offset) {  // 原子写入位置

    SOAHashTable ht;
    ht.build(build_part.keys.data(), build_part.keys.size());

    size_t local_count = 0;

    for (size_t i = 0; i < probe_part.keys.size(); ++i) {
        int32_t probe_key = probe_part.keys[i];
        uint32_t probe_idx = probe_part.indices[i];

        // 探测并直接写入结果
        auto matches = ht.probe_one(probe_key);
        for (uint32_t build_idx : matches) {
            // 原子获取写入位置
            size_t pos = write_offset.fetch_add(1, std::memory_order_relaxed);

            // 检查容量（可能需要扩展）
            if (pos >= result->capacity) {
                // 扩展逻辑（需要同步）
            }

            result->left_indices[pos] = build_idx;
            result->right_indices[pos] = probe_idx;
            local_count++;
        }
    }

    return local_count;
}

// 优势:
// - 消除 temp_build, temp_probe 临时缓冲区
// - 内存使用 = 实际匹配数 × 8B
// - 对稀疏 Join 特别有效
```

### 4.5 方案 5: 溢出到磁盘机制

```cpp
// 新增: 大于内存数据集支持
class SpillableHashTable {
    static constexpr size_t MEMORY_LIMIT = 1ULL << 30;  // 1GB

    struct SpillFile {
        int fd;
        size_t offset;
        size_t count;
    };

    std::vector<SpillFile> spill_files_;
    size_t memory_used_ = 0;

public:
    void build(const int32_t* keys, size_t count) {
        // 分批处理
        size_t batch_size = MEMORY_LIMIT / (sizeof(int32_t) + sizeof(uint32_t));

        for (size_t i = 0; i < count; i += batch_size) {
            size_t batch_count = std::min(batch_size, count - i);

            // 构建内存哈希表
            build_batch(keys + i, batch_count);

            if (memory_used_ > MEMORY_LIMIT) {
                // 溢出到磁盘
                spill_to_disk();
            }
        }
    }

private:
    void spill_to_disk() {
        // 创建临时文件
        char path[] = "/tmp/thunderduck_spill_XXXXXX";
        int fd = mkstemp(path);
        unlink(path);  // 进程结束时自动删除

        // 写入分区数据
        // ...

        spill_files_.push_back({fd, 0, /*count*/});
        memory_used_ = 0;
    }
};
```

---

## 五、架构设计

### 5.1 新内存管理架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ThunderDuck Memory Manager v2                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Thread Pool  │    │ Memory Pools │    │ Spill Manager│          │
│  │              │    │              │    │              │          │
│  │ ┌──────────┐ │    │ ┌──────────┐ │    │ ┌──────────┐ │          │
│  │ │Thread 0  │ │    │ │ Small    │ │    │ │ Temp Dir │ │          │
│  │ │Local Pool│◄├────┤►│ Objects  │ │    │ │ /tmp     │ │          │
│  │ └──────────┘ │    │ │ <1KB     │ │    │ └──────────┘ │          │
│  │ ┌──────────┐ │    │ └──────────┘ │    │ ┌──────────┐ │          │
│  │ │Thread 1  │ │    │ ┌──────────┐ │    │ │ Memory   │ │          │
│  │ │Local Pool│◄├────┤►│ Medium   │ │    │ │ Limit    │ │          │
│  │ └──────────┘ │    │ │ 1KB-1MB  │ │    │ │ 80% RAM  │ │          │
│  │     ...      │    │ └──────────┘ │    │ └──────────┘ │          │
│  └──────────────┘    │ ┌──────────┐ │    └──────────────┘          │
│                      │ │ Large    │ │                               │
│                      │ │ >1MB     │ │                               │
│                      │ │ mmap     │ │                               │
│                      │ └──────────┘ │                               │
│                      └──────────────┘                               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Unified Memory Accounting                    │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │ │
│  │  │ Hash Tables│  │ Join Result│  │ Temp Buffer│               │ │
│  │  │ Memory     │  │ Memory     │  │ Memory     │               │ │
│  │  └────────────┘  └────────────┘  └────────────┘               │ │
│  │                          │                                      │ │
│  │                    ┌─────┴─────┐                               │ │
│  │                    │ Total Used│ → 超限触发 Spill             │ │
│  │                    └───────────┘                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 组件设计

#### 5.2.1 MemoryManager 接口

```cpp
class MemoryManager {
public:
    // 配置
    struct Config {
        size_t memory_limit = 0;        // 0 = 自动 (80% RAM)
        std::string temp_directory = "/tmp";
        size_t small_object_threshold = 1024;
        size_t large_object_threshold = 1 << 20;
    };

    static MemoryManager& instance();

    // 分配接口
    void* allocate(size_t size, size_t alignment = 16);
    void deallocate(void* ptr, size_t size);

    // 内存统计
    size_t used_memory() const;
    size_t available_memory() const;
    bool should_spill() const;

    // 操作级别生命周期
    class OperationScope {
    public:
        OperationScope();
        ~OperationScope();  // 自动回收临时内存
    };
};
```

#### 5.2.2 优化后的 Hash Join

```cpp
size_t hash_join_i32_v4(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type,
    JoinResult* result) {

    MemoryManager::OperationScope scope;  // RAII 内存管理

    // 1. 选择率估算
    double selectivity = JoinSelectivityEstimator::estimate_by_sampling(
        build_keys, build_count, probe_keys, probe_count);
    size_t estimated_matches = static_cast<size_t>(
        probe_count * selectivity * 1.2);  // 20% 余量

    // 2. 智能预分配
    ensure_capacity(result, estimated_matches);

    // 3. 检查内存限制
    if (MemoryManager::instance().should_spill()) {
        return hash_join_with_spill(build_keys, build_count,
                                     probe_keys, probe_count,
                                     join_type, result);
    }

    // 4. 内存充足，使用优化哈希表
    CompactHashTable ht(build_count);  // 使用优化负载因子
    ht.build(build_keys, build_count);

    // 5. 直接输出探测（无临时缓冲区）
    return ht.probe_direct(probe_keys, probe_count, result);
}
```

---

## 六、实施计划

### 6.1 阶段划分

| 阶段 | 任务 | 优先级 | 预期效果 |
|------|------|--------|----------|
| Phase 1 | 智能 JoinResult 分配 | P0 | 消除 99% 浪费 |
| Phase 2 | 直接输出优化 | P0 | 消除临时缓冲区 |
| Phase 3 | 优化哈希表负载因子 | P1 | 减少 30% 内存 |
| Phase 4 | 内存池实现 | P1 | 减少碎片 |
| Phase 5 | 溢出机制 | P2 | 支持大数据集 |

### 6.2 Phase 1 详细设计

```cpp
// 文件: src/core/join_result_allocator.h

class SmartJoinResultAllocator {
public:
    // 基于采样的选择率估算
    static size_t estimate_result_size(
        const int32_t* build_keys, size_t build_count,
        const int32_t* probe_keys, size_t probe_count) {

        constexpr size_t SAMPLE_SIZE = 1000;
        constexpr double SAFETY_FACTOR = 1.5;

        // 构建 build 侧采样集合
        std::unordered_set<int32_t> build_sample;
        size_t sample_step = std::max(1UL, build_count / SAMPLE_SIZE);
        for (size_t i = 0; i < build_count; i += sample_step) {
            build_sample.insert(build_keys[i]);
        }

        // 估算 probe 侧匹配率
        size_t probe_sample_step = std::max(1UL, probe_count / SAMPLE_SIZE);
        size_t matches = 0;
        size_t probed = 0;
        for (size_t i = 0; i < probe_count; i += probe_sample_step) {
            if (build_sample.count(probe_keys[i])) {
                matches++;
            }
            probed++;
        }

        double selectivity = static_cast<double>(matches) / probed;

        // 估算重复率 (build_count / build_sample.size())
        double duplication = static_cast<double>(build_count) / build_sample.size();

        return static_cast<size_t>(
            probe_count * selectivity * duplication * SAFETY_FACTOR);
    }
};
```

---

## 七、性能预期

### 7.1 内存使用改善

| 场景 | 当前 | 优化后 | 改善 |
|------|------|--------|------|
| 稀疏 Join (0.1% 匹配) | 32 MB | 50 KB | **99.8%** |
| 中等 Join (10% 匹配) | 32 MB | 4 MB | **87.5%** |
| 密集 Join (100% 匹配) | 32 MB | 32 MB | 0% |
| 哈希表 (100K keys) | 2 MB | 1.4 MB | **30%** |

### 7.2 端到端内存对比

```
预期优化后对比 (1M 行数据):

┌─────────────────────┬──────────────┬──────────────┬─────────┐
│ 操作                │ ThunderDuck  │ DuckDB       │ 对比    │
│                     │ (优化后)     │              │         │
├─────────────────────┼──────────────┼──────────────┼─────────┤
│ 数据装载            │ 15 MB        │ 38 MB        │ -60%    │
│ Hash Join (稀疏)    │ 16 MB        │ 45 MB        │ -64%    │
│ Hash Join (密集)    │ 48 MB        │ 55 MB        │ -13%    │
│ Filter              │ 0.25 MB      │ 1 MB         │ -75%    │
│ Aggregation         │ 4 MB         │ 8 MB         │ -50%    │
└─────────────────────┴──────────────┴──────────────┴─────────┘
```

---

## 八、参考资料

1. [DuckDB Memory Management](https://duckdb.org/2024/07/09/memory-management) - DuckDB 官方内存管理文档
2. [LeanStore: In-Memory Data Management Beyond Main Memory](https://db.in.tum.de/~leis/papers/leanstore.pdf) - 轻量级 Buffer Manager 设计
3. [Robin Hood Hashing](https://programming.guide/robin-hood-hashing.html) - Robin Hood 哈希原理
4. [Cuckoo Hashing](https://en.wikipedia.org/wiki/Cuckoo_hashing) - Cuckoo 哈希空间效率

---

## 九、附录: 测试代码

见 `/Users/sihaoli/ThunderDuck/benchmark/memory_benchmark.cpp`

---

**文档结束**
