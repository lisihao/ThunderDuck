# Hash Join v3.0 优化设计文档

> **版本**: 3.0.0 | **日期**: 2026-01-24
>
> 针对 Apple M4 处理器深度优化的 Hash Join 实现

---

## 一、现状分析

### 1.1 性能对比

| 指标 | DuckDB | ThunderDuck v2 | 差距 |
|------|--------|----------------|------|
| Hash Join (1M rows) | 1.365 ms | 18.382 ms | **13.5x 更慢** |
| 内存访问模式 | 向量化批量 | 逐行处理 | 低效 |
| 缓存利用率 | 高 | 低 | 大量缓存未命中 |

### 1.2 当前实现问题分析

#### 问题 1: 哈希表数据结构 (AoS 布局)

```cpp
// 当前 Robin Hood 哈希表 Entry 结构
struct Entry {
    int32_t key;      // 4 bytes
    uint32_t row_idx; // 4 bytes
    uint8_t psl;      // 1 byte
    // padding:       // 3 bytes (对齐)
};  // Total: 12 bytes
```

**问题**:
- M4 缓存行 128 bytes，每行仅容纳 10 个 Entry
- 探测时只需要 key 进行比较，却加载了整个 Entry
- 内存带宽浪费 66% (只用 4/12 bytes)

#### 问题 2: 探测阶段逐行处理

```cpp
// 当前探测逻辑
for (size_t i = 0; i < probe_count; ++i) {
    int32_t probe_key = probe_keys[i];
    uint32_t hash = hash_key(probe_key);
    size_t idx = hash & mask_;

    while (entries_[idx].key != EMPTY_KEY) {  // 逐个比较
        if (entries_[idx].key == probe_key) {
            // 匹配
        }
        idx = (idx + 1) & mask_;
    }
}
```

**问题**:
- 每个 probe key 单独处理，无法利用 SIMD 批量比较
- 随机访问哈希表，L1 缓存命中率极低
- 预取效果有限，因为下一个位置取决于当前比较结果

#### 问题 3: 哈希冲突处理开销

- Robin Hood 策略虽然减少探测方差，但插入时的"抢夺"操作增加开销
- 链表/开放寻址都存在大量缓存未命中

#### 问题 4: 未利用 M4 硬件加速

- 未使用 AMX (Apple Matrix Extensions) 进行批量操作
- 未利用 GPU 进行大规模并行计算
- 未考虑 M4 的 128-byte 缓存行优化

---

## 二、M4 处理器架构分析

### 2.1 硬件规格

| 组件 | 规格 | 优化机会 |
|------|------|----------|
| CPU | 10 核 (4P + 6E) | 多线程并行分区 |
| L1 缓存 | 64 KB/核 | 小分区适配 L1 |
| L2 缓存 | 4 MB 共享 | 中等分区适配 L2 |
| 缓存行 | **128 bytes** | 数据布局优化 |
| 内存带宽 | 120 GB/s | 流式处理 |
| Neural Engine | 38 TOPS | 批量矩阵运算 |
| AMX | BF16/FP16/INT8 | 批量比较加速 |
| GPU | 10 核 | 大规模并行哈希 |

### 2.2 M4 特殊优化点

#### 2.2.1 128-byte 缓存行
- 是 x86 的 2 倍 (64 bytes)
- 需要更大的预取粒度
- SOA 布局可在一个缓存行放 32 个 int32 键

#### 2.2.2 AMX (Apple Matrix Extensions)
- 专用矩阵运算单元
- 支持 INT8/FP16/BF16
- 可用于批量键比较（将比较视为矩阵运算）

#### 2.2.3 高内存带宽
- 120 GB/s 统一内存
- 适合流式扫描和批量操作
- 顺序访问远快于随机访问

---

## 三、优化策略设计

### 3.1 策略总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Hash Join v3.0 架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │   Build     │    │   Radix      │    │   SOA Hash Table        │ │
│  │   Table     │───▶│   Partition  │───▶│   (L1/L2 Friendly)      │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │   Probe     │    │   Batch      │    │   SIMD Batch Probe      │ │
│  │   Table     │───▶│   Streaming  │───▶│   + Software Prefetch   │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Multi-core Parallelism                       │ │
│  │  - Partition-level: 每核处理独立分区                             │ │
│  │  - Morsel-driven: 细粒度任务调度                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心优化策略

#### 策略 A: SOA 哈希表布局

**从 AoS 到 SOA**:

```cpp
// 当前 AoS (Array of Structures)
struct Entry { int32_t key; uint32_t row_idx; uint8_t psl; };
Entry entries_[N];

// 优化后 SOA (Structure of Arrays)
int32_t* keys_;           // 连续存储所有键
uint32_t* row_indices_;   // 连续存储所有行索引
uint8_t* psls_;           // 连续存储所有 PSL

// 或者更紧凑的分组布局
struct KeyGroup {
    int32_t keys[32];     // 32 个键 = 128 bytes = 1 缓存行
};
```

**优势**:
- 键比较只加载 keys_ 数组，缓存利用率提升 3x
- 一个 128-byte 缓存行可存 32 个 int32 键
- SIMD 可一次比较 32 个键

#### 策略 B: Radix Partitioning

**分区策略**:

```cpp
// 使用哈希值的高位进行分区
constexpr int RADIX_BITS = 4;          // 16 个分区
constexpr int NUM_PARTITIONS = 1 << RADIX_BITS;

// 分区大小目标
// - 每分区 < 32KB (适配 L1 64KB，留空间给探测侧)
// - 或每分区 < 256KB (适配 L2 4MB / 16 分区)
```

**分区流程**:

```
Build Table (100K rows)
    │
    ├── 直方图统计 (计算每个分区大小)
    │
    ├── 分配分区缓冲区
    │
    └── 分散写入各分区
            │
            ├── Partition 0: ~6.25K rows (hash & 0xF == 0)
            ├── Partition 1: ~6.25K rows (hash & 0xF == 1)
            │   ...
            └── Partition 15: ~6.25K rows

Probe Table (1M rows)
    │
    └── 同样分区，然后每分区独立 Join
```

**优势**:
- 每分区数据适配 L1/L2 缓存
- 分区间无数据依赖，可完全并行
- 减少缓存冲突

#### 策略 C: SIMD 批量探测

**批量比较 32 个键**:

```cpp
// 使用 ARM Neon 一次比较 32 个键
void probe_batch_32(const int32_t* probe_keys, size_t probe_count,
                    const int32_t* ht_keys, size_t ht_size,
                    uint32_t* matches) {

    // 每次处理 32 个 probe keys
    for (size_t i = 0; i < probe_count; i += 32) {
        // 预取下一批
        __builtin_prefetch(probe_keys + i + 64, 0, 3);

        // 加载 32 个 probe keys (8 个 int32x4_t)
        int32x4_t p0 = vld1q_s32(probe_keys + i);
        int32x4_t p1 = vld1q_s32(probe_keys + i + 4);
        // ... p2-p7

        // 计算哈希值
        uint32x4_t h0 = vcrc32cq_u32(vdupq_n_u32(0xFFFFFFFF),
                                      vreinterpretq_u32_s32(p0));
        // ...

        // 查找哈希表 (使用 SIMD gather 模拟)
        // ...
    }
}
```

#### 策略 D: 完美哈希优化

**对于小整数键 (如 customer_id)**:

```cpp
// 如果键范围较小且密集，使用直接索引
if (key_max - key_min <= 2 * build_count) {
    // 使用 Perfect Hash: 键直接作为索引
    std::vector<uint32_t> direct_index(key_max - key_min + 1, EMPTY);

    // Build: O(n)
    for (size_t i = 0; i < build_count; ++i) {
        direct_index[build_keys[i] - key_min] = i;
    }

    // Probe: O(1) 每个键，无冲突！
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t key = probe_keys[i];
        if (key >= key_min && key <= key_max) {
            uint32_t idx = direct_index[key - key_min];
            if (idx != EMPTY) {
                // 匹配！
            }
        }
    }
}
```

#### 策略 E: 多线程并行

**Morsel-Driven 并行**:

```cpp
// 细粒度任务调度
constexpr size_t MORSEL_SIZE = 2048;  // DuckDB 标准

void parallel_probe(const int32_t* probe_keys, size_t probe_count,
                    const HashTable& ht, JoinResult* result) {

    std::atomic<size_t> next_morsel{0};

    // 每个线程独立处理 morsel
    auto worker = [&]() {
        std::vector<uint32_t> local_matches_build;
        std::vector<uint32_t> local_matches_probe;

        while (true) {
            size_t morsel_start = next_morsel.fetch_add(MORSEL_SIZE);
            if (morsel_start >= probe_count) break;

            size_t morsel_end = std::min(morsel_start + MORSEL_SIZE, probe_count);

            // 处理当前 morsel
            for (size_t i = morsel_start; i < morsel_end; ++i) {
                // 探测...
            }
        }

        // 合并结果
        merge_results(result, local_matches_build, local_matches_probe);
    };

    // 启动 P-core 数量的线程
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {  // 4 个 P-cores
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();
}
```

---

## 四、数据结构设计

### 4.1 SOA 哈希表

```cpp
class SOAHashTable {
public:
    // 128-byte 对齐的键数组 (适配 M4 缓存行)
    alignas(128) std::vector<int32_t> keys_;

    // 行索引数组
    std::vector<uint32_t> row_indices_;

    // 状态位图 (空/占用)
    std::vector<uint64_t> occupied_bitmap_;

    // 元数据
    size_t capacity_;
    size_t mask_;
    size_t size_;

    // 分区信息
    struct Partition {
        size_t start;
        size_t count;
    };
    std::vector<Partition> partitions_;
};
```

### 4.2 分区缓冲区

```cpp
struct PartitionedData {
    // 每个分区的数据
    struct Partition {
        alignas(128) std::vector<int32_t> keys;
        std::vector<uint32_t> indices;
    };

    std::array<Partition, NUM_PARTITIONS> partitions;

    // 直方图 (用于预分配)
    std::array<size_t, NUM_PARTITIONS> histogram;
};
```

### 4.3 批量探测结果

```cpp
struct BatchProbeResult {
    // 预分配的结果缓冲区 (每线程独立)
    alignas(128) std::vector<uint32_t> build_indices;
    alignas(128) std::vector<uint32_t> probe_indices;

    size_t count;
    size_t capacity;
};
```

---

## 五、实现计划

### 5.1 Phase 1: SOA 哈希表 + SIMD 探测

**目标**: 单线程性能提升 3-5x

**实现步骤**:
1. 实现 SOA 布局的哈希表
2. 实现 SIMD 批量键比较 (32 个键/批次)
3. 优化预取策略 (128-byte 粒度)

**预期效果**:
- 缓存利用率: +200%
- 内存带宽利用率: +150%

### 5.2 Phase 2: Radix Partitioning

**目标**: L1/L2 缓存友好

**实现步骤**:
1. 实现直方图统计
2. 实现分散写入
3. 实现分区级并行

**预期效果**:
- 缓存命中率: +300%
- 适配 L1: <32KB/分区

### 5.3 Phase 3: 完美哈希优化

**目标**: 小整数键场景极速

**实现步骤**:
1. 检测键范围和密度
2. 实现直接索引表
3. 自动选择策略

**预期效果**:
- 小整数键: O(1) 探测
- 消除哈希冲突

### 5.4 Phase 4: 多线程并行

**目标**: 利用多核

**实现步骤**:
1. 实现 morsel-driven 调度
2. 实现线程本地结果缓冲
3. 实现无锁结果合并

**预期效果**:
- 4 P-cores: 理论 4x 加速
- 实际: 3-3.5x (考虑调度开销)

### 5.5 Phase 5: 高级硬件加速 (可选)

**目标**: 探索 M4 特殊硬件

**可选方向**:
1. **AMX 批量比较**: 将键比较映射为矩阵运算
2. **GPU 并行哈希**: 使用 Metal 进行大规模并行
3. **Neural Engine**: 探索 NPU 加速可能性

---

## 六、性能预测

### 6.1 理论分析

| 优化阶段 | 主要瓶颈 | 优化后 | 预期加速 |
|----------|----------|--------|----------|
| 当前 | 随机内存访问 | - | 1x |
| Phase 1 | 缓存利用率低 | SOA + SIMD | 3-5x |
| Phase 2 | L1/L2 未命中 | Radix 分区 | 2-3x |
| Phase 3 | 哈希冲突 | 完美哈希 | 1.5-2x |
| Phase 4 | 单线程 | 多核并行 | 3-4x |
| **总计** | - | - | **18-60x** |

### 6.2 目标性能

| 测试 | DuckDB | 当前 v2 | 目标 v3 | 预期加速 |
|------|--------|---------|---------|----------|
| Hash Join (1M rows) | 1.365 ms | 18.382 ms | **< 1.0 ms** | **>18x** |

### 6.3 阶段性目标

| 里程碑 | 目标时间 | 性能目标 |
|--------|----------|----------|
| Phase 1 完成 | - | < 6 ms (3x 提升) |
| Phase 2 完成 | - | < 2 ms (10x 提升) |
| Phase 3 完成 | - | < 1.2 ms (15x 提升) |
| Phase 4 完成 | - | < 0.8 ms (23x 提升) |

---

## 七、API 设计

### 7.1 新接口

```cpp
namespace thunderduck::join {

// v3.0 优化版 Hash Join
size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

// 配置选项
struct JoinConfig {
    size_t num_threads = 4;           // 并行线程数
    size_t morsel_size = 2048;        // morsel 大小
    int radix_bits = 4;               // 分区位数
    bool enable_perfect_hash = true;  // 启用完美哈希检测
    bool enable_prefetch = true;      // 启用软件预取
};

size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result,
                         const JoinConfig& config);

} // namespace thunderduck::join
```

---

## 八、关键代码设计

### 8.1 SOA 哈希表实现

```cpp
class SOAHashTableV3 {
public:
    static constexpr int32_t EMPTY_KEY = INT32_MIN;
    static constexpr size_t KEYS_PER_CACHELINE = 128 / sizeof(int32_t);  // 32

    void build(const int32_t* keys, size_t count) {
        // 1. 计算容量 (2的幂，负载因子0.7)
        capacity_ = next_power_of_2(count * 1.5);
        mask_ = capacity_ - 1;

        // 2. 分配 128-byte 对齐的数组
        keys_.resize(capacity_, EMPTY_KEY);
        row_indices_.resize(capacity_);

        // 3. 插入 (线性探测)
        for (size_t i = 0; i < count; ++i) {
            insert(keys[i], static_cast<uint32_t>(i));
        }
    }

    void insert(int32_t key, uint32_t row_idx) {
        uint32_t hash = crc32_hash(key);
        size_t idx = hash & mask_;

        while (keys_[idx] != EMPTY_KEY) {
            idx = (idx + 1) & mask_;
        }

        keys_[idx] = key;
        row_indices_[idx] = row_idx;
    }

private:
    alignas(128) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    size_t capacity_;
    size_t mask_;
};
```

### 8.2 SIMD 批量探测

```cpp
#ifdef __aarch64__

// 一次探测 4 个键 (使用单个 Neon 向量)
inline void probe_4_keys(const int32_t* probe_keys,
                         const int32_t* ht_keys,
                         const uint32_t* ht_row_indices,
                         size_t ht_mask,
                         uint32_t probe_base_idx,
                         std::vector<uint32_t>& out_build,
                         std::vector<uint32_t>& out_probe) {

    // 加载 4 个 probe keys
    int32x4_t pkeys = vld1q_s32(probe_keys);

    // 计算哈希值
    uint32x4_t hashes;
    hashes = vsetq_lane_u32(__crc32cw(0xFFFFFFFF, vgetq_lane_s32(pkeys, 0)), hashes, 0);
    hashes = vsetq_lane_u32(__crc32cw(0xFFFFFFFF, vgetq_lane_s32(pkeys, 1)), hashes, 1);
    hashes = vsetq_lane_u32(__crc32cw(0xFFFFFFFF, vgetq_lane_s32(pkeys, 2)), hashes, 2);
    hashes = vsetq_lane_u32(__crc32cw(0xFFFFFFFF, vgetq_lane_s32(pkeys, 3)), hashes, 3);

    // 计算桶索引
    uint32x4_t indices = vandq_u32(hashes, vdupq_n_u32(ht_mask));

    // 线性探测每个键
    for (int i = 0; i < 4; ++i) {
        int32_t probe_key = vgetq_lane_s32(pkeys, i);
        size_t idx = vgetq_lane_u32(indices, i);

        // 预取哈希表位置
        __builtin_prefetch(&ht_keys[idx], 0, 3);

        while (ht_keys[idx] != EMPTY_KEY) {
            if (ht_keys[idx] == probe_key) {
                out_build.push_back(ht_row_indices[idx]);
                out_probe.push_back(probe_base_idx + i);
            }
            idx = (idx + 1) & ht_mask;
        }
    }
}

#endif
```

### 8.3 Radix 分区

```cpp
void radix_partition(const int32_t* keys, const uint32_t* indices, size_t count,
                     PartitionedData& out) {
    constexpr int RADIX_BITS = 4;
    constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;
    constexpr size_t PARTITION_MASK = NUM_PARTITIONS - 1;

    // Phase 1: 直方图统计
    std::array<size_t, NUM_PARTITIONS> histogram{};
    for (size_t i = 0; i < count; ++i) {
        uint32_t hash = crc32_hash(keys[i]);
        size_t partition = (hash >> (32 - RADIX_BITS)) & PARTITION_MASK;
        histogram[partition]++;
    }

    // Phase 2: 计算偏移并预分配
    std::array<size_t, NUM_PARTITIONS> offsets{};
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        out.partitions[p].keys.resize(histogram[p]);
        out.partitions[p].indices.resize(histogram[p]);
    }

    // Phase 3: 分散写入
    std::array<size_t, NUM_PARTITIONS> write_pos{};
    for (size_t i = 0; i < count; ++i) {
        uint32_t hash = crc32_hash(keys[i]);
        size_t partition = (hash >> (32 - RADIX_BITS)) & PARTITION_MASK;

        size_t pos = write_pos[partition]++;
        out.partitions[partition].keys[pos] = keys[i];
        out.partitions[partition].indices[pos] = indices[i];
    }
}
```

---

## 九、测试计划

### 9.1 正确性测试

| 测试 | 描述 | 预期结果 |
|------|------|----------|
| 小表 Join | 100 x 1000 | 匹配数正确 |
| 大表 Join | 100K x 1M | 匹配数正确 |
| 重复键 Join | 大量重复键 | 正确处理 |
| 无匹配 Join | 无任何匹配 | 返回 0 |
| 完全匹配 | 1:1 完全匹配 | 匹配数 = build_count |

### 9.2 性能测试

| 测试 | 数据规模 | 目标 |
|------|----------|------|
| Build 性能 | 100K rows | < 1 ms |
| Probe 性能 | 1M rows | < 1 ms |
| 总体性能 | 100K x 1M | < 1.5 ms |
| 多线程扩展性 | 1-4 线程 | 接近线性 |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 哈希冲突严重 | 性能下降 | 完美哈希 + 动态调整负载因子 |
| 分区不均匀 | 并行效率低 | 动态负载均衡 |
| 内存分配开销 | 延迟增加 | 预分配 + 内存池 |
| SIMD 效果有限 | 加速不明显 | 批量处理 + 循环展开 |

---

## 参考资料

- [Apple M4 Wikipedia](https://en.wikipedia.org/wiki/Apple_M4)
- [DuckDB Join Processing](https://deepwiki.com/duckdb/duckdb/8.1-join-processing)
- [Saving Private Hash Join - VLDB 2024](https://www.vldb.org/pvldb/vol18/p2748-kuiper.pdf)
- [DuckDB Performance Guide](https://duckdb.org/docs/stable/guides/performance/join_operations)

---

*ThunderDuck - 针对 Apple M4 优化的高性能数据库算子库*
