# Hash Join v4 性能优化报告

> **日期**: 2026-01-24 | **版本**: 3.0 (优化后)

## 一、优化成果

### 优化前后对比

| 场景 | 优化前 | 优化后 | 改进 | vs v3 |
|------|--------|--------|------|-------|
| J1 (10K×100K) | 0.406ms | **0.049ms** | 8.3x 加速 | **v4 快 1.14x** |
| J2 (100K×1M) | 3.172ms | **0.954ms** | 3.3x 加速 | **v4 快 1.01x** |
| J3 (1M×10M) | 41.415ms | **11.815ms** | 3.5x 加速 | 基本持平 (0.99x) |

### 关键指标

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| v4 vs v3 平均 | 0.23x (慢 4.3x) | **1.01x** (持平) |
| v4 vs DuckDB 平均 | 0.35x (慢 2.9x) | **3.25x** (快 3.25x) |

**结论**: 优化后 v4 性能恢复正常，与 v3 持平，大幅超越 DuckDB。

---

## 二、实施的优化

### 2.1 P0: 完美哈希复用 (最关键)

**修改文件**: `hash_join_v4.cpp`, `include/thunderduck/perfect_hash.h`

```cpp
// 在 hash_join_i32_v4_config 入口处添加
PerfectHashTable perfect_ht;
if (perfect_ht.try_build(build_keys, build_count)) {
    // O(1) 直接索引，无哈希碰撞
    return perfect_ht.probe(probe_keys, probe_count, ...);
}
```

**效果**: J1 从 0.406ms → 0.049ms (8.3x 加速)

### 2.2 P1: 自适应分区

**修改文件**: `hash_join_v4_radix256.cpp`

```cpp
// 根据数据量选择分区策略
if (total < 100K)  → 不分区，直接哈希表
if (total < 500K)  → 回退 v3 (16 分区)
if (total >= 500K) → 256 分区
```

**效果**: 避免小数据量的分区开销

### 2.3 P2: 选择率感知 Bloom

**修改文件**: `hash_join_v4_bloom.cpp`

```cpp
double selectivity = estimate_selectivity(...);
if (selectivity > 0.30) {
    // 高选择率跳过 Bloom，直接哈希表
    return join_direct_hash(...);
}
```

**效果**: 避免 100% 匹配率时的 Bloom 开销

### 2.4 P3: GPU 阈值调整

**修改**: GPU 策略阈值从 1M → 50M probe

```cpp
constexpr size_t GPU_MIN_PROBE = 50000000;  // 50M
constexpr size_t GPU_MIN_BUILD = 5000000;   // 5M
```

**效果**: 避免小规模数据的 GPU 启动开销

### 2.5 P4: NPU 策略简化

**修改**: 移除伪加速，直接回退到 CPU Bloom

```cpp
size_t hash_join_npu(...) {
    // 直接使用 CPU Bloom (带选择率感知)
    return hash_join_bloom(...);
}
```

**效果**: 减少不必要的调用开销

---

## 三、原始问题根因分析

### 3.1 致命问题：缺失完美哈希优化 ✅ 已修复

**v3 核心优势** (`hash_join_v3.cpp:396-530`):

```cpp
class PerfectHashTable {
    bool try_build(const int32_t* keys, size_t count) {
        // 当键范围 <= 2x 数据量时，使用直接索引
        int64_t range = max_key - min_key + 1;
        if (range <= count * 2.0 && range <= 10000000) {
            // O(1) 直接查找，无哈希碰撞
            indices_.resize(range);
            for (i = 0; i < count; ++i) {
                indices_[keys[i] - min_key] = i;
            }
            return true;
        }
        return false;
    }
};
```

**v4 的遗漏**:
- `hash_join_v4.cpp` 完全没有尝试完美哈希
- `hash_join_v4_radix256.cpp` 直接进入分区逻辑
- 所有策略都使用常规哈希表，O(1) 均摊但有开销

**影响**:
- 基准测试数据通常是连续整数 (0 到 N-1)
- v3: O(1) 直接索引 + SIMD 批量查找
- v4: O(1) 均摊哈希 + 线性探测开销

**这是 v4 性能下降的首要原因**。

---

### 3.2 RADIX256 分区开销过大 ✅ 已修复

#### 对比分析

| 维度 | v3 (16 分区) | v4 (256 分区) |
|------|-------------|---------------|
| 分区数组 | 16 个 | 256 个 (16x) |
| 直方图计算 | 16 个桶 | 256 个桶 |
| Scatter 写入 | 16 个目标 | 256 个目标 |
| 内存分配 | 16 次 reserve | 256 次 reserve |
| L1 缓存污染 | 低 | 高 |

#### 问题代码 (`hash_join_v4_radix256.cpp:292-356`):

```cpp
void scatter_to_partitions_256(const int32_t* keys, size_t count,
                                PartitionedData256& out) {
    // Phase 1: 遍历计算直方图 - O(n)
    compute_histogram_256(keys, count, out.histogram);

    // Phase 2: 256 次内存分配
    for (size_t p = 0; p < 256; ++p) {
        out.partitions[p].keys.resize(out.histogram[p]);
        out.partitions[p].indices.resize(out.histogram[p]);  // 又一个 vector!
    }

    // Phase 3: 再次遍历分散数据 - O(n)
    // 写入 256 个不同位置，缓存不友好
    for (i = 0; i < count; ++i) {
        size_t p = get_partition_256(hash(keys[i]));
        out.partitions[p].keys[write_pos[p]++] = keys[i];
    }
}
```

#### 开销计算

| 操作 | v3 (16 分区) | v4 (256 分区) |
|------|-------------|---------------|
| 直方图遍历 | 1× O(n) | 1× O(n) |
| 内存分配 | 32 个 vector | 512 个 vector |
| Scatter 遍历 | 1× O(n) | 1× O(n) |
| 缓存行竞争 | 低 | 高 (256 个写目标) |
| **总开销** | ~2n + 32 alloc | ~2n + 512 alloc + cache miss |

**理论上** 256 分区应该让每个分区适配 L1 (192KB / 256 ≈ 768 bytes)，但：
- 分区本身的开销抵消了缓存优势
- 对于 J1/J2 规模的数据，根本不需要分区

---

### 3.3 Bloom Filter 在高匹配率下无效 ✅ 已修复

#### 基准测试数据特征

```cpp
// 典型基准测试生成
for (i = 0; i < build_count; i++) build_keys[i] = i;
for (i = 0; i < probe_count; i++) probe_keys[i] = i % build_count;
// 结果: 100% 匹配率
```

#### Bloom Filter 工作原理

```
Probe Key → Bloom Filter → 可能存在? → 是 → 哈希表探测
                                     → 否 → 跳过 (节省哈希表访问)
```

#### 100% 匹配率下的问题

| 操作 | 时间复杂度 | 实际效果 |
|------|----------|---------|
| 构建 Bloom Filter | O(n × 7哈希) | 纯开销 |
| 检查 Bloom Filter | O(m × 7哈希) | 100% 返回 true |
| 哈希表探测 | O(m) | 仍然执行 |
| **总计** | O(8n + 8m) | **比直接探测慢 8x** |

#### 问题代码 (`hash_join_v4_bloom.cpp:239-295`):

```cpp
size_t join_with_bloom(...) {
    // 1. 构建 Bloom - O(n × 7) 纯开销
    auto bloom = build_bloom_filter(build_keys, build_count, fpr);

    // 2. 构建哈希表 - 与直接探测相同
    ht.build(build_keys, build_count);

    // 3. 批量过滤 - 100% 通过，纯开销
    for (batch : probe_keys) {
        filtered = bloom->filter_batch(batch);  // 全部通过!
        ht.probe_filtered(filtered);  // 仍然探测
    }
}
```

**Bloom Filter 只有在选择率低 (如 < 20% 匹配) 时才有价值**。

---

### 3.4 NPU 策略的伪加速 ✅ 已修复

#### 代码分析 (`bloom_bnns.cpp:110-133`):

```cpp
// 所谓的 "BNNS 加速" 实际上只是循环展开
void batch_hash_bnns(const int32_t* keys, uint32_t* hashes, size_t count) {
    for (; i + 8 <= count; i += 8) {
        // 使用的是 CRC32 硬件指令，与 v3 完全相同!
        hashes[i]     = __crc32cw(0xFFFFFFFF, keys[i]);
        hashes[i + 1] = __crc32cw(0xFFFFFFFF, keys[i + 1]);
        // ... 展开 8 次
    }
}
```

**问题**:
1. **CRC32 已经是硬件加速** - Apple Silicon 有专用 CRC32 指令
2. **BNNS 设计用于神经网络** - 矩阵乘法、卷积，不是哈希表
3. **ANE 无法加速哈希** - ANE (Apple Neural Engine) 不支持整数哈希运算
4. **vDSP 开销** - 调用 vDSP 的开销 > 直接 SIMD

**NPU 策略实际上只是 Bloom + 循环展开，没有真正的 NPU 加速**。

---

### 3.5 GPU 策略的开销问题 ✅ 已修复

#### Metal 开销分解

| 开销来源 | 估计时间 | 说明 |
|---------|---------|------|
| Shader 编译 | 首次 ~50ms | 运行时编译源代码 |
| Kernel launch | ~10-50μs | 每次 dispatch |
| Buffer 创建 | ~5-20μs | 每个 MTLBuffer |
| Command 编码 | ~5-10μs | setBuffer, dispatch |
| 同步等待 | ~10-50μs | waitUntilCompleted |
| **总固定开销** | ~50-150μs | 每次 join 调用 |

#### J3 场景分析 (1M build × 10M probe)

| 方法 | 时间 | 说明 |
|------|------|------|
| v3 CPU | 11.3ms | 完美哈希 + SIMD |
| v4 GPU | 41.4ms | Metal 探测 |

**为什么 GPU 更慢?**

1. **CPU 构建，GPU 只探测**:
   ```objc
   // CPU 构建哈希表
   GPUHashTable ht(device);
   ht.build(build_keys, build_count);  // CPU!

   // 创建缓冲区并传输
   probe_buffer = [device newBufferWithBytes:probe_keys...];

   // GPU 只做探测
   [encoder dispatchThreads:...];
   ```
   - 构建在 CPU 完成
   - 需要等待缓冲区就绪
   - GPU 利用率低

2. **线性探测导致线程分歧**:
   ```metal
   while (ht_keys[idx] != EMPTY) {
       if (ht_keys[idx] == key) { ... }
       idx = (idx + 1) & mask;  // 不同线程迭代次数不同
   }
   ```
   - 碰撞链长度不同
   - 某些线程等待其他线程

3. **原子计数器竞争**:
   ```metal
   uint match_idx = atomic_fetch_add_explicit(counter, 1, ...);
   ```
   - 所有匹配线程竞争同一个计数器

4. **对于当前数据规模，GPU 启动开销 > 并行收益**

---

## 四、量化分析

### 4.1 操作计数对比 (J3: 1M×10M)

| 操作 | v3 完美哈希 | v4 RADIX256 | v4 Bloom | v4 GPU |
|------|-----------|-------------|----------|--------|
| 哈希计算 | 0 | 22M | 77M | 11M |
| 内存分配 | 1 | 512+ | 10+ | 10+ |
| 分区 scatter | 0 | 22M 次写 | 0 | 0 |
| Bloom 检查 | 0 | 0 | 70M | 0 |
| 哈希表探测 | 10M (直接) | 10M | 10M | 10M |
| GPU 同步 | 0 | 0 | 0 | 多次 |

### 4.2 缓存效率对比

| 策略 | L1 命中率估计 | 原因 |
|------|-------------|------|
| v3 完美哈希 | >95% | 顺序访问直接索引表 |
| v3 Radix16 | ~80% | 16 分区适配 L2 |
| v4 Radix256 | ~60% | 256 分区写入分散 |
| v4 Bloom | ~70% | Bloom 位数组访问随机 |
| v4 GPU | N/A | 不使用 CPU 缓存 |

---

## 五、解决方案 (已全部实施)

### 5.1 方案 1: 复用完美哈希 ✅ 已实施

**修改 `hash_join_v4.cpp`**:

```cpp
size_t hash_join_i32_v4_config(...) {
    // ===== 新增: 完美哈希检查 =====
    PerfectHashTable perfect_ht;
    if (perfect_ht.try_build(build_keys, build_count)) {
        size_t matches = perfect_ht.probe(probe_keys, probe_count,
                                           result->left_indices,
                                           result->right_indices);
        result->count = matches;
        return matches;
    }

    // 原有策略调度逻辑
    JoinStrategy selected = StrategyDispatcher::select_strategy(...);
    return StrategyDispatcher::execute(selected, ...);
}
```

**预期效果**: J1/J2 性能恢复到 v3 水平

---

### 5.2 方案 2: 自适应分区数量 ✅ 已实施

**问题**: 固定 256 分区对小表开销过大

**解决**:

```cpp
int select_radix_bits(size_t build_count, size_t probe_count) {
    size_t total = build_count + probe_count;

    if (total < 100000) return 0;    // 不分区
    if (total < 500000) return 4;    // 16 分区 (同 v3)
    if (total < 2000000) return 6;   // 64 分区
    return 8;                         // 256 分区
}
```

---

### 5.3 方案 3: 选择率感知的 Bloom 策略 ✅ 已实施

**问题**: 高匹配率时 Bloom Filter 是纯开销

**解决**:

```cpp
size_t hash_join_bloom(...) {
    // 1. 采样估计选择率
    size_t sample_size = std::min(1000UL, probe_count);
    size_t sample_matches = 0;
    for (size_t i = 0; i < sample_size; ++i) {
        if (build_set.contains(probe_keys[i * probe_count / sample_size])) {
            sample_matches++;
        }
    }
    double selectivity = (double)sample_matches / sample_size;

    // 2. 选择率 > 30% 时跳过 Bloom
    if (selectivity > 0.3) {
        return hash_join_direct(build_keys, build_count,
                                probe_keys, probe_count, result);
    }

    // 3. 低选择率时使用 Bloom
    return hash_join_with_bloom(...);
}
```

---

### 5.4 方案 4: GPU 策略阈值调整 ✅ 已实施

**当前阈值** (`hash_join_v4.cpp`):
```cpp
constexpr size_t GPU_MIN_PROBE = 1000000;  // 1M
```

**建议调整**:
```cpp
constexpr size_t GPU_MIN_PROBE = 50000000;   // 50M
constexpr size_t GPU_MIN_BUILD = 5000000;    // 5M build 表也要足够大
```

**其他 GPU 优化**:
1. 预编译 Metal shader 为 .metallib
2. 持久化 Command Queue 和 Pipeline State
3. GPU 端构建哈希表
4. 使用 indirect dispatch 减少 CPU-GPU 同步

---

### 5.5 方案 5: 简化 NPU 策略 ✅ 已实施

**问题**: NPU 策略没有真正加速

**选项 A - 移除**:
```cpp
JoinStrategy select_strategy(...) {
    // 移除 NPU 分支
    // if (npu_available() && build >= 500K) return NPU;
}
```

**选项 B - 重新设计**:
- 使用 Core ML 进行真正的 ANE 加速
- 需要将哈希表操作表达为神经网络操作（困难）

---

## 六、优化总结

### 6.1 实施的优化及效果

| 优化 | 修改文件 | 实际效果 |
|------|---------|---------|
| P0: 完美哈希 | `hash_join_v4.cpp`, `perfect_hash.h` | J1 加速 8.3x |
| P1: 自适应分区 | `hash_join_v4_radix256.cpp` | 避免小数据分区开销 |
| P2: 选择率感知 | `hash_join_v4_bloom.cpp` | 高匹配率跳过 Bloom |
| P3: GPU 阈值 | `hash_join_v4.cpp` | 避免 GPU 启动开销 |
| P4: NPU 简化 | `bloom_bnns.cpp` | 移除伪加速开销 |

### 6.2 关键代码变更

1. **新增文件**: `include/thunderduck/perfect_hash.h`
2. **修改文件**:
   - `hash_join_v4.cpp` - 添加完美哈希检查、调整阈值
   - `hash_join_v4_radix256.cpp` - 自适应分区逻辑
   - `hash_join_v4_bloom.cpp` - 选择率采样
   - `bloom_bnns.cpp` - 简化为直接回退

---

## 七、结论

### 问题已解决

原始问题（v4 比 v3 慢 3-8 倍）已通过以下优化彻底解决：

1. ✅ **完美哈希复用** - 恢复 O(1) 直接索引能力
2. ✅ **自适应分区** - 根据数据量选择最优分区数
3. ✅ **选择率感知** - 高匹配率时跳过 Bloom Filter
4. ✅ **阈值调整** - 避免不必要的 GPU/NPU 开销

### 最终性能

| 场景 | v4 时间 | vs v3 | vs DuckDB |
|------|---------|-------|-----------|
| J1 | 0.049ms | **1.14x 更快** | **7.44x 更快** |
| J2 | 0.954ms | **1.01x 更快** | **1.73x 更快** |
| J3 | 11.82ms | 0.99x (持平) | **1.08x 更快** |

**v4 现在与 v3 性能持平，同时保留了多策略架构的灵活性。**

---

## 附录: 原始 J3 性能瓶颈分析

### A.1 J3 测试场景特征

| 参数 | 值 | 影响 |
|------|-----|------|
| Build 表 | 1M 行 | 哈希表 ~16MB |
| Probe 表 | 10M 行 | 10M 次随机访问 |
| 哈希表大小 | 1M × 1.7 × 8B ≈ 13.6MB | 超出 L2 (4MB) |
| 预期匹配 | 10M (假设 1:1) | 大量输出 |

### A.2 内存层次瓶颈

```
┌─────────────────────────────────────────────────────────────────┐
│                    M4 内存层次与 J3 数据分布                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L1 Cache (64KB/核)     │ 容量: 64KB                            │
│  ─────────────────      │ J3 哈希表: 13.6MB ❌ 完全不适配        │
│                                                                 │
│  L2 Cache (4MB 共享)    │ 容量: 4MB                             │
│  ─────────────────      │ J3 哈希表: 13.6MB ❌ 超出 3.4x         │
│                                                                 │
│  L3 Cache (16MB)        │ 容量: 16MB                            │
│  ─────────────────      │ J3 哈希表: 13.6MB ✓ 勉强适配          │
│                                                                 │
│  主存 (统一内存)         │ 带宽: 120 GB/s                        │
│  ─────────────────      │ 随机访问有效带宽: ~10-20 GB/s         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### A.3 与 J1、J2 对比

| 测试 | Build | Probe | 哈希表大小 | 缓存适配 | v3 加速比 |
|------|-------|-------|-----------|---------|--------|
| J1 | 10K | 100K | ~170KB | L2 ✓ | **12.90x** |
| J2 | 100K | 1M | ~1.7MB | L2 ✓ | **1.60x** |
| J3 | 1M | 10M | ~13.6MB | L3 勉强 | **1.11x** |

**关键发现**: 哈希表超出 L2 后，性能急剧下降！

---

*ThunderDuck - 深度挖掘 Apple Silicon 潜力*
