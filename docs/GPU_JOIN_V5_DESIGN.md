# GPU Hash Join v5 优化设计

> 版本: 5.0.0 | 日期: 2026-01-24
> 基于 SIGMOD'25 / VLDB 2025 最新论文

## 一、问题分析

### 当前 v4 GPU 实现的瓶颈

| 问题 | 原因 | 影响 |
|------|------|------|
| 随机内存访问 | 线性探测哈希表 | GPU 带宽利用率 < 10% |
| 原子竞争 | 全局计数器 | 10M 次原子操作造成串行化 |
| 低占用率 | 无共享内存优化 | GPU 核心空闲等待内存 |
| 启动开销 | 单次 dispatch | kernel 启动延迟占比高 |

### 性能目标

- J3 (1M × 10M): **v4 当前 ~12ms → 目标 < 5ms (2.4x 加速)**
- vs DuckDB: **当前 1.08x → 目标 > 2x**

## 二、核心优化技术

### 2.1 Radix Partitioned Join (SIGMOD'25 GFTR Pattern)

**核心思想**: 将随机访问转换为顺序访问

```
传统方法:
  Probe Key → Hash → 随机位置 → 冲突链遍历 (随机跳转)

GFTR 方法:
  Probe Key → Radix 前缀 → 确定性分区 → 顺序扫描分区
```

**实现步骤**:
1. **分区阶段**: 使用 8-bit radix (256 分区) 同时分区 build 和 probe
2. **排序阶段**: 对每个分区内的 build keys 进行 radix sort
3. **探测阶段**: 对每个分区使用 merge-join 或 binary search

**优势**:
- 顺序内存访问 → 100% 带宽利用
- 无哈希冲突 → 确定性执行时间
- 分区独立 → 完美并行

### 2.2 Threadgroup Memory 两阶段处理

**利用 Metal 的 threadgroup memory (类似 CUDA shared memory)**:

```metal
// 阶段 1: 全局内存 → 共享内存
threadgroup int32_t shared_keys[PARTITION_SIZE];
shared_keys[local_id] = build_keys[global_offset + local_id];
threadgroup_barrier(mem_flags::mem_threadgroup);

// 阶段 2: 在共享内存中探测 (超快)
for (int i = 0; i < local_probe_count; i++) {
    // 访问 shared_keys 而非全局内存
}
```

**共享内存容量**: Apple M4 GPU 每个 threadgroup 有 32KB

### 2.3 Warp-Level 批量结果收集

**问题**: 每次匹配都用 atomic_fetch_add → 严重竞争

**解决**: Warp 内先本地累积，再批量写入

```metal
// 每个线程本地缓存
thread uint32_t local_build[16];
thread uint32_t local_probe[16];
thread uint local_count = 0;

// 探测时累积到本地
if (found_match) {
    local_build[local_count] = build_idx;
    local_probe[local_count] = probe_idx;
    local_count++;
}

// Warp 内 prefix sum 计算全局偏移
uint warp_offset = simd_prefix_exclusive_sum(local_count);
uint warp_total = simd_sum(local_count);
uint global_offset = atomic_fetch_add(counter, warp_total);

// 批量写入
for (uint i = 0; i < local_count; i++) {
    out_build[global_offset + warp_offset + i] = local_build[i];
    out_probe[global_offset + warp_offset + i] = local_probe[i];
}
```

**效果**: 原子操作从 10M 次减少到 ~300K 次 (32 线程/warp)

### 2.4 Apple Silicon 统一内存优化

**特有优势**:
- **零拷贝**: CPU 和 GPU 共享物理内存，无需 DMA
- **带宽**: M4 有 400+ GB/s 统一内存带宽
- **延迟**: 无 PCIe 传输延迟

**优化策略**:
```objc
// 使用 Shared 存储模式启用零拷贝
MTLResourceStorageModeShared

// 对于只读数据，可以避免同步开销
MTLResourceCPUCacheModeWriteCombined
```

## 三、详细设计

### 3.1 新文件结构

```
src/gpu/
├── hash_join_metal.mm          # 主入口 (修改)
├── radix_partition_metal.mm    # 新增: GPU radix 分区
├── shaders/
│   ├── hash_join.metal         # 修改: 添加新 kernel
│   ├── radix_partition.metal   # 新增: 分区 shader
│   └── merge_join.metal        # 新增: 合并连接 shader
```

### 3.2 新增 Metal Kernels

#### 3.2.1 Radix Histogram Kernel
```metal
kernel void radix_histogram(
    device const int32_t* keys,
    device atomic_uint* histogram,  // 256 buckets
    constant uint& count,
    uint thread_id [[thread_position_in_grid]]
);
```

#### 3.2.2 Radix Scatter Kernel
```metal
kernel void radix_scatter(
    device const int32_t* keys,
    device const uint32_t* indices,
    device const uint* prefix_sum,  // 分区起始位置
    device int32_t* out_keys,
    device uint32_t* out_indices,
    constant uint& count,
    uint thread_id [[thread_position_in_grid]]
);
```

#### 3.2.3 Partitioned Join Kernel (核心)
```metal
kernel void partitioned_join(
    device const int32_t* build_keys,
    device const uint32_t* build_indices,
    device const int32_t* probe_keys,
    device const uint32_t* probe_indices,
    device const uint* partition_offsets,
    device const uint* partition_sizes,
    device uint32_t* out_build,
    device uint32_t* out_probe,
    device atomic_uint* counter,
    constant uint& partition_id,
    // 使用 threadgroup memory
    threadgroup int32_t* shared_build_keys [[threadgroup(0)]],
    threadgroup uint32_t* shared_build_indices [[threadgroup(1)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
);
```

### 3.3 执行流程

```
输入: build_keys[N], probe_keys[M]
         │
         ▼
┌─────────────────────────────────┐
│ 1. GPU Radix Histogram          │  O(N+M) 顺序读
│    计算 256 个分区的大小         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 2. CPU Prefix Sum               │  O(256) 超快
│    计算每个分区的起始偏移        │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 3. GPU Radix Scatter            │  O(N+M) 顺序写
│    重新排列数据到分区内          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 4. GPU Partitioned Join         │  256 个并行分区
│    每个分区在 threadgroup 内处理 │
│    使用 SIMD prefix sum 收集结果 │
└─────────────────────────────────┘
         │
         ▼
输出: matched pairs
```

### 3.4 Fallback 策略

当数据规模不适合 GPU 时自动降级:

```cpp
JoinStrategy select_gpu_strategy(size_t build, size_t probe) {
    // < 100K: CPU 更快 (GPU 启动开销)
    if (build + probe < 100000) return V3_FALLBACK;

    // 100K - 1M: 简单 GPU join (避免分区开销)
    if (build + probe < 1000000) return GPU_SIMPLE;

    // 1M - 50M: 单次 radix 分区
    if (build + probe < 50000000) return GPU_RADIX_SINGLE;

    // > 50M: 多轮分区 (避免超出 GPU 内存)
    return GPU_RADIX_MULTI;
}
```

## 四、预期收益

| 优化 | J1 提升 | J2 提升 | J3 提升 |
|------|--------|--------|--------|
| Radix 分区 (顺序访问) | 1.1x | 1.3x | 1.5x |
| Threadgroup 缓存 | 1.0x | 1.2x | 1.4x |
| 批量原子操作 | 1.0x | 1.1x | 1.3x |
| **总计** | ~1.1x | ~1.7x | **~2.7x** |

## 五、实现计划

| 阶段 | 任务 | 预计复杂度 |
|------|------|-----------|
| 1 | 实现 radix histogram/scatter kernels | 中 |
| 2 | 实现 partitioned join kernel with threadgroup | 高 |
| 3 | 实现 SIMD prefix sum 批量收集 | 中 |
| 4 | 集成到 v4 调度器 | 低 |
| 5 | 性能调优和基准测试 | 中 |

## 六、参考文献

1. "Efficiently Processing Joins and Grouped Aggregations on GPUs" - SIGMOD'25
2. "Efficiently Joining Large Relations on Multi-GPU Systems" - VLDB 2025
3. "Scaling GPU-Accelerated Databases Beyond GPU Memory Size" - VLDB 2025
4. Metal Performance Shaders Programming Guide - Apple Developer
