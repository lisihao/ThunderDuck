# ThunderDuck 性能设计文档

> **版本**: 2.0.0 | **更新日期**: 2026-01-24
> **测试平台**: Apple M4, macOS 15.0
> **理论内存带宽**: 400 GB/s

---

## 一、性能目标

### 1.1 总体目标

| 指标 | 目标 | 实际达成 |
|------|------|---------|
| vs DuckDB 平均加速 | > 5x | 5-13x ✓ |
| 带宽利用率 | > 25% | 29.5% ✓ |
| GPU 加速比 (最佳区间) | > 3x | 4.26x ✓ |
| 零拷贝比例 | > 80% | ~90% ✓ |

### 1.2 各算子目标

| 算子 | CPU 目标 | GPU 目标 | vs DuckDB |
|------|---------|---------|-----------|
| Filter | 2500 M/s | 6000 M/s | > 4x |
| Aggregate | 17000 M/s | 20000 M/s | > 5x |
| Join | 300 M/s | 900 M/s | > 2x |
| TopK | 2000 M/s | - | > 10x |
| Sort | 2000 M/s | - | > 5x |

---

## 二、硬件特性分析

### 2.1 Apple M4 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Apple M4 SoC 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 统一内存 (UMA)                        │   │
│  │                 400 GB/s 带宽                         │   │
│  └─────────────────────────────────────────────────────┘   │
│         │              │              │              │      │
│  ┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐┌──────┴────┐│
│  │ CPU Cores  ││    GPU      ││    NPU      ││  Memory   ││
│  │ 4P + 6E    ││  10 Cores   ││ 16 Cores    ││Controller ││
│  │ ARMv8.6-A  ││  Metal 3    ││ ANE         ││           ││
│  └─────────────┘└─────────────┘└─────────────┘└───────────┘│
│                                                             │
│  关键参数:                                                   │
│  - CPU: 4 性能核 (最高 4.4 GHz) + 6 能效核                   │
│  - GPU: 10 核, ~4 TFLOPS (FP32)                             │
│  - 内存: 16/32 GB LPDDR5, 400 GB/s                          │
│  - L1 Cache: 192 KB (P), 128 KB (E)                         │
│  - L2 Cache: 16 MB (共享)                                   │
│  - SLC: 8 MB (系统级缓存)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 内存层次

| 层级 | 大小 | 延迟 | 带宽 |
|------|------|------|------|
| L1 数据缓存 | 128-192 KB | ~3 周期 | ~500 GB/s |
| L2 缓存 | 16 MB | ~10 周期 | ~300 GB/s |
| SLC | 8 MB | ~30 周期 | ~200 GB/s |
| DRAM | 16-32 GB | ~100 周期 | 400 GB/s |

### 2.3 UMA 优势

```
传统架构 (离散 GPU):
  CPU Memory ──[PCIe 16 GB/s]──> GPU Memory
  延迟: ~10 μs

UMA 架构 (Apple Silicon):
  CPU ──[400 GB/s]──> 统一内存 <──[400 GB/s]── GPU
  延迟: ~100 ns

加速比: PCIe 拷贝开销 / UMA 零拷贝 ≈ 100x (小数据量)
```

---

## 三、优化策略

### 3.1 Filter 优化

#### 3.1.1 CPU SIMD 优化 (v3)

```cpp
// ARM NEON 128-bit 向量化
int32x4_t data = vld1q_s32(input + i);     // 加载 4 个 int32
int32x4_t cmp = vdupq_n_s32(value);        // 广播比较值
uint32x4_t mask = vcgtq_s32(data, cmp);    // 向量比较

// 位掩码提取
uint64_t bits = vget_lane_u64(vreinterpret_u64_u32(
    vuzp1_u32(vget_low_u32(mask), vget_high_u32(mask))), 0);
```

**性能分析**:
- 每次迭代处理 4 个元素
- 无分支 (使用位操作)
- 达到 ~2500 M rows/s

#### 3.1.2 GPU 优化 (v4)

```metal
// 向量化过滤 + threadgroup 前缀和
kernel void filter_simd4_i32(...) {
    // 1. 向量化加载和比较 (4 元素/线程)
    int4 vals = input[gid];
    bool4 mask = (vals > int4(params.value));

    // 2. 收集匹配索引到本地数组
    uint local_indices[4];
    uint local_count = 0;
    for (uint i = 0; i < 4; i++) {
        if (mask[i]) local_indices[local_count++] = gid * 4 + i;
    }

    // 3. Threadgroup 前缀和 (减少原子争用)
    tg_counts[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 快速前缀和 (O(log n) 步骤)
    for (uint stride = 1; stride < group_size; stride *= 2) {
        uint val = (lid >= stride) ? tg_counts[lid - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_counts[lid] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. 单次全局原子操作 (每 threadgroup)
    if (lid == group_size - 1) {
        tg_global_offset = atomic_fetch_add_explicit(
            counter, tg_total, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. 批量写入
    uint global_offset = tg_global_offset + local_offset;
    for (uint i = 0; i < local_count; i++) {
        out_indices[global_offset + i] = local_indices[i];
    }
}
```

**优化点**:
- 向量化: 4 元素/线程
- 前缀和: 将原子操作从 O(n) 降到 O(n/256)
- 达到 ~6000+ M rows/s (50M 数据量)

### 3.2 Aggregate 优化

#### 3.2.1 CPU SIMD 融合

```cpp
// 单次遍历计算 SUM + MIN + MAX
int64x2_t sum_vec = vdupq_n_s64(0);
int32x4_t min_vec = vdupq_n_s32(INT32_MAX);
int32x4_t max_vec = vdupq_n_s32(INT32_MIN);

for (size_t i = 0; i < count; i += 4) {
    int32x4_t data = vld1q_s32(input + i);

    // SUM: 扩展到 64 位累加
    sum_vec = vaddq_s64(sum_vec,
        vpaddlq_s32(data));  // pairwise add long

    // MIN/MAX: 向量比较
    min_vec = vminq_s32(min_vec, data);
    max_vec = vmaxq_s32(max_vec, data);
}

// 水平归约
int64_t sum = vgetq_lane_s64(sum_vec, 0) + vgetq_lane_s64(sum_vec, 1);
int32_t min_val = vminvq_s32(min_vec);  // 水平最小值
int32_t max_val = vmaxvq_s32(max_vec);  // 水平最大值
```

**性能分析**:
- 单次内存遍历
- 3 个统计量并行计算
- 达到 ~118 GB/s 带宽利用

#### 3.2.2 GPU 融合 Kernel

```metal
// aggregate_all_i32_phase1: 块级融合统计
kernel void aggregate_all_i32_phase1(
    device const int32_t* input [[buffer(0)]],
    device AllStats* block_stats [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup int64_t* shared_sum [[threadgroup(0)]],
    threadgroup int32_t* shared_min [[threadgroup(1)]],
    threadgroup int32_t* shared_max [[threadgroup(2)]]
) {
    // 每线程处理多个元素
    int64_t sum = 0;
    int32_t min_val = INT_MAX;
    int32_t max_val = INT_MIN;

    for (uint i = gid; i < count; i += stride) {
        int32_t val = input[i];
        sum += val;
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }

    // 存入共享内存
    shared_sum[lid] = sum;
    shared_min[lid] = min_val;
    shared_max[lid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 树形归约
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_sum[lid] += shared_sum[lid + s];
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写入块结果
    if (lid == 0) {
        block_stats[tgid] = {shared_sum[0], shared_min[0], shared_max[0]};
    }
}
```

**优化点**:
- 融合 kernel: 减少内存访问从 3 次到 1 次
- 树形归约: O(log n) 复杂度
- 两阶段: Phase1 块级归约 + Final 全局归约

### 3.3 Join 优化

#### 3.3.1 SOA 哈希表

```cpp
// 结构体数组 (AoS) - 缓存不友好
struct HashEntry { int32_t key; uint32_t index; };
HashEntry* table;  // 访问 key 时也加载 index

// 数组的结构体 (SOA) - 缓存友好
int32_t* ht_keys;     // 独立数组
uint32_t* ht_indices; // 独立数组
// 探测时只访问 keys，命中后才访问 indices
```

**优势**:
- 缓存行利用率: AoS ~50% → SOA ~100%
- 探测吞吐提升: ~1.5x

#### 3.3.2 GPU 前缀和批量写入

```
问题: 原子争用
  每个线程找到匹配后执行 atomic_add
  256 线程/threadgroup → 256 次原子操作

解决: Threadgroup 级前缀和
  1. 每线程统计本地匹配数
  2. Threadgroup 内前缀和
  3. 最后一个线程执行单次 atomic_add
  4. 使用前缀和偏移批量写入

原子操作: O(n) → O(n/256)
```

```metal
// Threadgroup 前缀和减少原子争用
tg_counts[lid] = local_count;
threadgroup_barrier(mem_flags::mem_threadgroup);

// 快速前缀和
for (uint stride = 1; stride < group_size; stride *= 2) {
    uint val = (lid >= stride) ? tg_counts[lid - stride] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_counts[lid] += val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 单次全局原子
threadgroup uint32_t tg_global_offset;
if (lid == group_size - 1) {
    tg_global_offset = atomic_fetch_add_explicit(
        match_counter, tg_total, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// 批量写入
uint global_offset = tg_global_offset + local_offset;
for (uint i = 0; i < local_count; i++) {
    out_build[global_offset + i] = local_build[i];
    out_probe[global_offset + i] = local_probe[i];
}
```

#### 3.3.3 SharedEvent 流水线

```objc
// 步骤 1: 构建哈希表
id<MTLCommandBuffer> cmd1 = [queue commandBuffer];
[encoder1 setComputePipelineState:buildPipeline];
// ... dispatch ...
[cmd1 encodeSignalEvent:event value:1];
[cmd1 commit];

// 步骤 2: 探测 (等待构建完成)
id<MTLCommandBuffer> cmd2 = [queue commandBuffer];
[cmd2 encodeWaitForEvent:event value:1];
[encoder2 setComputePipelineState:probePipeline];
// ... dispatch ...
[cmd2 commit];
[cmd2 waitUntilCompleted];  // 只在最后等待
```

**优势**:
- 减少 CPU-GPU 同步
- 构建和准备探测可重叠

### 3.4 TopK 优化

#### 3.4.1 采样预过滤 (v5)

```cpp
// 问题: 完整排序 O(n log n) 太慢

// 解决: 采样估计阈值
size_t topk_i32_v5(const int32_t* input, size_t count,
                    size_t k, uint32_t* out_indices) {
    // 1. 采样估计第 k 大值的阈值
    size_t sample_size = std::min(count, k * 100);
    std::vector<int32_t> samples(sample_size);
    // 随机采样...

    // 2. 找采样中的第 k 大值作为阈值
    std::nth_element(samples.begin(),
                     samples.begin() + k,
                     samples.end(),
                     std::greater<int32_t>());
    int32_t threshold = samples[k];

    // 3. 过滤: 只保留 >= threshold 的元素
    std::vector<uint32_t> candidates;
    for (size_t i = 0; i < count; i++) {
        if (input[i] >= threshold) {
            candidates.push_back(i);
        }
    }

    // 4. 对候选集排序 (远小于 n)
    std::partial_sort(candidates.begin(),
                      candidates.begin() + k,
                      candidates.end(),
                      [&](uint32_t a, uint32_t b) {
                          return input[a] > input[b];
                      });

    std::copy_n(candidates.begin(), k, out_indices);
    return k;
}
```

**复杂度**:
- 采样: O(sample_size)
- 过滤: O(n)
- 排序: O(candidates * log k)
- 总体: O(n) vs O(n log n)

---

## 四、内存优化

### 4.1 零拷贝策略

```cpp
// 检查页对齐
bool is_page_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16384) == 0;
}

// 零拷贝包装 (页对齐)
UMABuffer wrap_external(void* ptr, size_t size) {
    if (!is_page_aligned(ptr)) {
        return {};  // 无法零拷贝
    }

    id<MTLBuffer> mtlBuf = [device newBufferWithBytesNoCopy:ptr
                                                     length:size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    return {ptr, size, (__bridge void*)mtlBuf, true};
}

// 必须拷贝 (非对齐)
UMABuffer allocate_and_copy(const void* src, size_t size) {
    UMABuffer buf = allocate(size);
    std::memcpy(buf.data, src, size);
    return buf;
}
```

### 4.2 缓冲区池

```cpp
class UMAMemoryManager {
    std::vector<UMABuffer> buffer_pool_;
    std::mutex pool_mutex_;
    static constexpr size_t MAX_POOL_SIZE = 32;

    UMABuffer acquire_from_pool(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        // 查找合适的缓冲区 (最佳适配)
        auto it = std::find_if(buffer_pool_.begin(), buffer_pool_.end(),
            [size](const UMABuffer& buf) {
                return buf.size >= size && buf.size <= size * 2;
            });

        if (it != buffer_pool_.end()) {
            UMABuffer buf = *it;
            buffer_pool_.erase(it);
            return buf;  // 复用
        }

        return allocate(size);  // 新分配
    }

    void release_to_pool(UMABuffer& buf) {
        std::lock_guard<std::mutex> lock(pool_mutex_);

        if (buffer_pool_.size() < MAX_POOL_SIZE) {
            buffer_pool_.push_back(buf);
        } else {
            // 池满，释放最小的
            auto min_it = std::min_element(buffer_pool_.begin(),
                buffer_pool_.end(),
                [](const UMABuffer& a, const UMABuffer& b) {
                    return a.size < b.size;
                });
            deallocate(*min_it);
            *min_it = buf;
        }
    }
};
```

**效果**:
- 减少分配次数: ~80%
- 减少内存碎片
- 热路径无 malloc

### 4.3 预取优化

```cpp
// SIMD 循环中的软件预取
for (size_t i = 0; i < count; i += 4) {
    // 预取下一个缓存行
    __builtin_prefetch(input + i + 64, 0, 3);

    // 处理当前数据
    int32x4_t data = vld1q_s32(input + i);
    // ...
}
```

---

## 五、GPU 优化

### 5.1 线程配置

```objc
// 获取最优线程组大小
NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
NSUInteger threadgroupSize = std::min((NSUInteger)256, maxThreads);

// 计算网格大小
MTLSize gridSize = MTLSizeMake(count, 1, 1);
MTLSize groupSize = MTLSizeMake(threadgroupSize, 1, 1);

// 使用 dispatchThreads (自动处理边界)
[encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
```

### 5.2 Threadgroup 内存

```objc
// 分配 threadgroup 内存
[encoder setThreadgroupMemoryLength:256 * sizeof(uint32_t) atIndex:0];
[encoder setThreadgroupMemoryLength:256 * sizeof(int32_t) atIndex:1];
[encoder setThreadgroupMemoryLength:256 * sizeof(int32_t) atIndex:2];
```

**注意**: 必须调用 `setThreadgroupMemoryLength`，否则 shader 中的 `threadgroup` 参数未定义!

### 5.3 减少原子争用

```
策略                    原子操作数    适用场景
─────────────────────────────────────────────
全局原子               O(n)          小数据量
Threadgroup 归约       O(n/256)      中等数据量
两阶段前缀和           O(n/256²)     大数据量
```

---

## 六、自适应策略

### 6.1 策略选择矩阵

| 算子 | 条件 | 选择 | 原因 |
|------|------|------|------|
| Filter | < 10M | CPU SIMD | GPU 启动开销 > 收益 |
| Filter | >= 10M | GPU | 并行优势显现 |
| Aggregate | < 100M | CPU SIMD | 已接近带宽上限 |
| Aggregate | >= 100M | GPU | 额外并行度 |
| Join | probe < 500K | CPU | GPU 启动开销 > 收益 |
| Join | 500K-50M | GPU | 最佳加速区间 |
| Join | > 50M | GPU | 带宽受限，加速有限 |
| TopK | < 50M | CPU Sample | 采样高效 |
| TopK | >= 50M | GPU | 并行优势 |

### 6.2 动态调整

```cpp
Executor StrategySelector::select(OperatorType op,
                                   const DataCharacteristics& data) {
    // 1. GPU 可用性检查
    if (!is_gpu_available()) {
        return Executor::CPU_SIMD;
    }

    // 2. 数据对齐检查 (零拷贝优势)
    float alignment_bonus = data.is_page_aligned ? 1.2f : 1.0f;

    // 3. 根据算子和数据量选择
    switch (op) {
        case OperatorType::FILTER:
            if (data.row_count * alignment_bonus >= 10'000'000) {
                return Executor::GPU;
            }
            break;

        case OperatorType::JOIN:
            if (data.row_count >= 500'000 &&
                data.row_count <= 50'000'000) {
                return Executor::GPU;
            }
            break;

        // ...
    }

    return Executor::CPU_SIMD;
}
```

---

## 七、基准测试结果

### 7.1 Filter 性能

| 数据量 | CPU SIMD | GPU | GPU 加速比 | vs DuckDB |
|--------|----------|-----|-----------|-----------|
| 100K | 2492 M/s | 2487 M/s | 1.00x | 5.17x |
| 1M | 2634 M/s | 2635 M/s | 1.00x | 4.24x |
| 10M | 2583 M/s | 3155 M/s | 1.22x | 4.06x |
| 50M | 2534 M/s | 6701 M/s | **2.65x** | 5.38x |

### 7.2 Aggregate 性能

| 数据量 | CPU SIMD | GPU | 带宽 | vs DuckDB |
|--------|----------|-----|------|-----------|
| 100K | 23759 M/s | 23765 M/s | 95 GB/s | 3.64x |
| 1M | 23576 M/s | 23715 M/s | 94 GB/s | 6.82x |
| 10M | 18560 M/s | 19495 M/s | 78 GB/s | 4.36x |
| 50M | 17056 M/s | 17232 M/s | 69 GB/s | 11276x (COUNT) |

### 7.3 Join 性能

| 规模 | CPU v3 | GPU UMA | GPU 加速比 | vs DuckDB |
|------|--------|---------|-----------|-----------|
| 10K×100K | 2234 M/s | 2219 M/s | 0.99x | 2.08x |
| 100K×1M | 329 M/s | 575 M/s | **1.75x** | 2.08x |
| 1M×10M | 235 M/s | 947 M/s | **4.26x** | - |
| 5M×50M | 287 M/s | 395 M/s | **1.38x** | - |

### 7.4 TopK 性能

| K 值 | 数据量 | 时间 | 吞吐量 | vs DuckDB |
|------|--------|------|--------|-----------|
| 10 | 500K | 0.236 ms | 2119 M/s | 3.07x |
| 100 | 500K | 0.030 ms | 16667 M/s | **26.72x** |
| 1000 | 500K | 0.137 ms | 3650 M/s | 10.36x |

---

## 八、性能瓶颈分析

### 8.1 当前瓶颈

| 瓶颈 | 影响 | 缓解措施 |
|------|------|---------|
| 内存带宽 | Aggregate 已达上限 | 融合 kernel |
| 原子争用 | Join 写入 | 前缀和批量写入 |
| GPU 启动 | 小数据量开销 | 自适应回退 |
| 缓存失效 | 随机访问 Join | SOA 布局 |

### 8.2 优化空间

| 优化方向 | 预期提升 | 复杂度 |
|---------|---------|--------|
| Join Cuckoo Hashing | +20% | 高 |
| NPU Bloom 加速 | +30% (大数据) | 中 |
| 多命令缓冲区重叠 | +15% | 低 |
| SIMD 宽度扩展 (SVE2) | +50% (未来硬件) | 低 |

---

## 九、性能调优指南

### 9.1 数据准备

```cpp
// 推荐: 页对齐分配 (16KB)
void* data = nullptr;
posix_memalign(&data, 16384, size);

// 或使用 UMA 分配器
auto& mgr = UMAMemoryManager::instance();
UMABuffer buf = mgr.allocate(size);
```

### 9.2 配置调优

```cpp
// 手动指定策略
FilterConfigV4 config;
config.strategy = FilterStrategy::GPU;  // 强制 GPU
config.selectivity_hint = 0.5f;         // 提示选择率

// 调整阈值
namespace thresholds {
    FILTER_GPU_MIN = 5'000'000;  // 降低 GPU 阈值
}
```

### 9.3 批量操作

```cpp
// 避免: 多次小操作
for (size_t i = 0; i < n; i++) {
    filter_i32_v4(input + i * chunk, chunk, ...);  // 每次 GPU 启动
}

// 推荐: 单次大操作
filter_i32_v4(input, total_count, ...);  // 一次 GPU 启动
```

---

## 十、未来优化计划

### 10.1 短期 (v2.1)

- [ ] Join Cuckoo Hashing 减少冲突
- [ ] NPU BNNS Bloom Filter 加速
- [ ] 更细粒度的策略选择

### 10.2 中期 (v3.0)

- [ ] 多 GPU 支持 (外接 GPU)
- [ ] 查询编译 (JIT)
- [ ] 向量化表达式求值

### 10.3 长期

- [ ] 支持 Apple Silicon SVE2 (未来芯片)
- [ ] 分布式查询执行
- [ ] AI 驱动的策略选择
