# ThunderDuck 架构设计文档

> **版本**: 8.0.0 | **日期**: 2026-01-31
> **目标平台**: Apple Silicon M4 | macOS 15.0+
> **当前性能**: 几何平均 3.18x vs DuckDB (22/22 查询全胜)

---

## 一、项目定位

### 1.1 什么是 ThunderDuck?

ThunderDuck 是专为 **Apple M4 芯片深度优化**的 SQL 算子后端系统，作为 DuckDB 的高性能替代方案，通过硬件感知的算子实现实现数倍性能提升。

**核心设计理念**:
- **硬件第一**: 针对 M4 的 ARM Neon SIMD、128字节缓存行、UMA统一内存架构深度优化
- **算子专精**: 70+ 专用算子版本，针对不同数据规模和特征自适应选择
- **零拷贝加速**: 利用 UMA 实现 CPU/GPU 零拷贝数据共享
- **完全兼容**: 与 DuckDB 接口兼容，可作为 TPC-H 基准测试后端

### 1.2 核心指标

| 类别 | 指标 | 数值 |
|------|------|------|
| **代码规模** | 源代码 | 31,016 行 |
| | 算子版本 | 70+ 专用实现 |
| **TPC-H 性能** | SF=1 覆盖率 | 22/22 查询 |
| | 几何平均加速比 | **3.18x** |
| | 最佳加速比 | **22.44x** (Q2) |
| | 胜率 | **100%** (22/22) |
| **硬件加速** | ARM Neon SIMD | ✅ |
| | Metal GPU | ✅ |
| | CRC32 硬件哈希 | ✅ |
| | 多核并行 (8T) | ✅ |

### 1.3 性能矩阵

```
TPC-H SF=1 性能分布 (vs DuckDB):

22.44x ┤ ▇ Q2 (最小成本供应商)
       │
 7.12x ┤ ▇ Q1 (定价汇总)
       │
 3.54x ┤ ▇▇ Q4, Q21 (SEMI JOIN)
       │
 2.57x ┤ ▇▇▇ Q6, Q11, Q19 (Filter + Aggregate)
       │
 1.46x ┤ ▇▇▇▇▇▇▇ Q3, Q5, Q7, Q8, Q10, Q12, Q14, Q15, Q16, Q18, Q20
       │
 1.00x ┼─────────────────────────────────────────────────
       └─ DuckDB 基线

几何平均: 3.18x  |  算术平均: 5.02x
```

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ThunderDuck Engine V8.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        SQL 查询层                                   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │  DuckDB API 兼容层 (TPC-H Extension)                        │  │    │
│  │  │  - 22 条 TPC-H 查询注册                                       │  │    │
│  │  │  - 数据加载接口 (Parquet/CSV)                                │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        算子注册层                                   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │  OperatorRegistry - 算子版本管理                             │  │    │
│  │  │  - Filter: v1→v19.1 (9个版本)                                │  │    │
│  │  │  - Join: v1→v19.2 (16个版本)                                 │  │    │
│  │  │  - Aggregate: v1→v21 (7个版本)                               │  │    │
│  │  │  - Sort/TopK: v1→v5 (5个版本)                                │  │    │
│  │  │  - GPU: v66/v68/v69 (融合算子)                               │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                   自适应策略选择器                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  AdaptiveOptimizer - 运行时算子选择                         │   │    │
│  │  │                                                             │   │    │
│  │  │  输入: 数据规模 + 选择率 + 基数 + 系统负载                  │   │    │
│  │  │  输出: 最优算子版本 + 执行路径 (CPU/GPU)                    │   │    │
│  │  │                                                             │   │    │
│  │  │  策略示例:                                                   │   │    │
│  │  │  - Filter: count<1M → v3, 1M-10M → v15, >10M → v19 (8T)    │   │    │
│  │  │  - Join: perfect_hash → v4, bloom → v32, GPU → v19.2       │   │    │
│  │  │  - Aggregate: groups<=64 → v14 (寄存器), >64 → v15 (并行)  │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                          │                    │                             │
│           ┌──────────────┴───────┐  ┌────────┴──────────┐                  │
│           ▼                      ▼  ▼                    ▼                  │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐          │
│  │   CPU SIMD 层       │  │          GPU 加速层                 │          │
│  │                     │  │                                     │          │
│  │  ARM Neon SIMD:     │  │  Metal Compute Shader:              │          │
│  │  ┌───────────────┐  │  │  ┌─────────────────────────────┐   │          │
│  │  │ Filter v19    │  │  │  │ GPUFusedFilterAggregate v66 │   │          │
│  │  │ - 4累加器并行 │  │  │  │ - Threadgroup 前缀和        │   │          │
│  │  │ - vsub优化    │  │  │  │ - 零拷贝 UMA                │   │          │
│  │  │ - 8线程       │  │  │  └─────────────────────────────┘   │          │
│  │  └───────────────┘  │  │  ┌─────────────────────────────┐   │          │
│  │  ┌───────────────┐  │  │  │ GPUFusedQ3 v68              │   │          │
│  │  │ Join v19.2    │  │  │  │ - Filter+Join+Agg 融合      │   │          │
│  │  │ - Compact Hash│  │  │  │ - Block-local hash          │   │          │
│  │  │ - 完美哈希    │  │  │  └─────────────────────────────┘   │          │
│  │  │ - Bitmap      │  │  │  ┌─────────────────────────────┐   │          │
│  │  └───────────────┘  │  │  │ GPUGroupByAggregate v69     │   │          │
│  │  ┌───────────────┐  │  │  │ - 分组聚合框架              │   │          │
│  │  │ Aggregate v21 │  │  │  │ - Atomic 优化               │   │          │
│  │  │ - 直接数组    │  │  │  └─────────────────────────────┘   │          │
│  │  │ - 8路展开     │  │  │                                     │          │
│  │  └───────────────┘  │  │                                     │          │
│  │  ┌───────────────┐  │  │                                     │          │
│  │  │ Sort/TopK v4  │  │  │                                     │          │
│  │  │ - Radix Sort  │  │  │                                     │          │
│  │  │ - 采样 TopK   │  │  │                                     │          │
│  │  └───────────────┘  │  │                                     │          │
│  └─────────────────────┘  └─────────────────────────────────────┘          │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      核心组件层                                     │    │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │    │
│  │  │ UMA Memory    │ │ SIMD/vDSP     │ │ Thread Pool   │             │    │
│  │  │ - zero-copy   │ │ - ARM Neon    │ │ - 8线程预热   │             │    │
│  │  │ - 16KB 对齐   │ │ - CRC32 hash  │ │ - 任务队列    │             │    │
│  │  │ - Metal Buffer│ │ - Accelerate  │ │ - 负载均衡    │             │    │
│  │  └───────────────┘ └───────────────┘ └───────────────┘             │    │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │    │
│  │  │ Cache Align   │ │ Perfect Hash  │ │ Bloom Filter  │             │    │
│  │  │ - 128 字节    │ │ - O(1) lookup │ │ - SIMD probe  │             │    │
│  │  │ - M4 L1 优化  │ │ - 小整数键    │ │ - 预过滤      │             │    │
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
                    │  │ CRC32   │ │  UMA 共享   │  │
                    │  └─────────┘ └─────────────┘  │
                    │       ┌───────────┐           │
                    │       │ UMA 120GB/s│          │
                    │       │ 统一内存   │           │
                    │       │ 零拷贝     │           │
                    │       └───────────┘           │
                    └───────────────────────────────┘
```

### 2.2 目录结构

```
ThunderDuck/
├── include/thunderduck/          # 公共 API (31 头文件, 9,326 行)
│   ├── filter.h                  # Filter 算子接口 (v1-v19.1)
│   ├── aggregate.h               # Aggregate 算子接口 (v1-v21)
│   ├── join.h                    # Join 算子接口 (v1-v19.2)
│   ├── sort.h                    # Sort/TopK 接口 (v1-v5)
│   ├── uma_memory.h              # UMA 内存管理
│   ├── operator_registry.h       # 算子注册系统
│   ├── perfect_hash.h            # 完美哈希表
│   └── generic_operators_v54.h   # 通用算子框架
│
├── src/                          # 核心源代码 (21,690 行)
│   ├── core/                     # 执行引擎 (2,824 行)
│   │   ├── uma_memory.cpp        # UMA 内存管理实现
│   │   ├── adaptive_strategy.cpp # 策略选择器
│   │   └── thread_pool.cpp       # 线程池
│   │
│   ├── operators/                # 算子实现 (18,177 行)
│   │   ├── filter/               # 过滤算子 (9 版本, 3,702 行)
│   │   │   ├── simd_filter.cpp           # v1 基础 SIMD
│   │   │   ├── simd_filter_v2.cpp        # v2 位图优化
│   │   │   ├── simd_filter_v3.cpp        # v3 4累加器
│   │   │   ├── simd_filter_v15.cpp       # v15 直接索引生成
│   │   │   └── simd_filter_v19.cpp       # v19 8线程并行
│   │   │
│   │   ├── aggregate/            # 聚合算子 (7 版本, 2,515 行)
│   │   │   ├── simd_aggregate.cpp        # v1 基础
│   │   │   ├── simd_aggregate_v2.cpp     # v2 融合 MIN/MAX
│   │   │   ├── simd_aggregate_v14.cpp    # v14 寄存器缓冲
│   │   │   └── simd_aggregate_v21.cpp    # v21 8路展开
│   │   │
│   │   ├── join/                 # Join 算子 (16 版本, 7,738 行)
│   │   │   ├── hash_join_v3.cpp          # v3 SOA 哈希表
│   │   │   ├── hash_join_v4.cpp          # v4 完美哈希
│   │   │   ├── hash_join_v13.cpp         # v13 两阶段
│   │   │   ├── hash_join_v19_2.cpp       # v19.2 SIMD 探测
│   │   │   └── perfect_hash.cpp          # 完美哈希实现
│   │   │
│   │   └── sort/                 # 排序算子 (5 版本, 2,805 行)
│   │       ├── radix_sort.cpp            # Radix Sort (11-11-10)
│   │       ├── topk_v3.cpp               # v3 自适应堆
│   │       ├── topk_v4.cpp               # v4 采样算法
│   │       └── topk_v5.cpp               # v5 混合策略
│   │
│   ├── gpu/                      # Metal GPU 实现
│   │   ├── filter_uma.mm                 # Filter GPU
│   │   ├── aggregate_uma.mm              # Aggregate GPU
│   │   ├── hash_join_uma.mm              # Join GPU (UMA)
│   │   ├── inner_join_gpu.mm             # Inner Join GPU
│   │   └── semi_join_gpu.mm              # SEMI Join GPU
│   │
│   └── utils/                    # 工具类 (135 行)
│       └── platform_detect.cpp   # 平台检测
│
├── benchmark/                    # 基准测试
│   ├── tpch/                     # TPC-H 完整实现 (86 文件, 51,044 行)
│   │   ├── tpch_benchmark_main.cpp       # 主测试程序
│   │   ├── tpch_queries.cpp              # 22 条查询注册
│   │   ├── tpch_data_loader.cpp          # Parquet 数据加载
│   │   ├── tpch_executor.cpp             # 查询执行器
│   │   ├── tpch_config_v33.*             # V33 通用配置
│   │   ├── tpch_operators_v24.cpp        # V24 算子实现
│   │   ├── tpch_operators_v25.cpp        # V25 优化算子
│   │   ├── tpch_operators_v32.cpp        # V32 Bloom Filter
│   │   ├── tpch_operators_v47.cpp        # V47 最新优化
│   │   └── tpch_report.cpp               # 性能报告生成
│   │
│   └── operator_benchmark.cpp    # 单算子基准测试
│
├── tests/                        # 单元测试 (6 文件, 1,446 行)
│   ├── framework_test.cpp        # 框架测试
│   └── ...
│
├── docs/                         # 文档 (104 个 .md 文件)
│   ├── ARCHITECTURE.md           # 本文档
│   ├── V47_TPCH_COMPREHENSIVE_BENCHMARK.md
│   ├── PROJECT_OVERVIEW.md
│   └── ...
│
└── third_party/                  # 依赖
    └── duckdb/                   # DuckDB 1.1.3
```

---

## 三、核心技术

### 3.1 ARM Neon SIMD 优化

#### 3.1.1 4 累加器并行 (Filter v3)

**核心问题**: CPU 流水线中的数据依赖链限制 ILP (指令级并行)

**解决方案**: 4 独立累加器 + vsub 优化

```cpp
// Filter v3 - 4 独立累加器消除依赖链
size_t count_i32_v3(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
    int32x4_t val_vec = vdupq_n_s32(value);

    // 4 独立累加器 (消除依赖链)
    uint32x4_t cnt0 = vdupq_n_u32(0);
    uint32x4_t cnt1 = vdupq_n_u32(0);
    uint32x4_t cnt2 = vdupq_n_u32(0);
    uint32x4_t cnt3 = vdupq_n_u32(0);

    for (size_t i = 0; i < count; i += 16) {
        // L2 预取下一批
        __builtin_prefetch(input + i + 64);

        // 4 路并行加载
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 向量化比较
        uint32x4_t m0 = vcgtq_s32(d0, val_vec);
        uint32x4_t m1 = vcgtq_s32(d1, val_vec);
        uint32x4_t m2 = vcgtq_s32(d2, val_vec);
        uint32x4_t m3 = vcgtq_s32(d3, val_vec);

        // vsub 优化: -1 转 +1 (比 vshr+vadd 快)
        cnt0 = vsubq_u32(cnt0, m0);
        cnt1 = vsubq_u32(cnt1, m1);
        cnt2 = vsubq_u32(cnt2, m2);
        cnt3 = vsubq_u32(cnt3, m3);
    }

    // 合并累加器
    uint32x4_t total = vaddq_u32(vaddq_u32(cnt0, cnt1),
                                  vaddq_u32(cnt2, cnt3));
    return vaddvq_u32(total);  // 横向归约
}
```

**性能**:
- vs DuckDB: 3.6x (10M 数据)
- vs v1 (单累加器): 1.8x

#### 3.1.2 SIMD 并行槽位比较 (Join v19.2)

**核心问题**: 哈希表探测时逐个比较槽位，分支预测失败导致停顿

**解决方案**: 一次比较 4 个连续槽位

```cpp
// Join v19.2 - SIMD 并行槽位比较
while (true) {
    // 加载 4 个连续槽位的键
    int32x4_t keys_vec = vld1q_s32(&keys[idx]);
    int32x4_t key_broadcast = vdupq_n_s32(probe_key);

    // SIMD 比较 4 个槽位
    uint32x4_t eq_mask = vceqq_s32(keys_vec, key_broadcast);

    // 转换为 bitmask
    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(
        vshrn_n_u16(vreinterpretq_u16_u32(eq_mask), 4)), 0);

    // 处理匹配
    if (mask) {
        for (int i = 0; i < 4; i++) {
            if ((mask >> (i * 16)) & 0x1) {
                // 找到匹配
                out_build[match_count] = row_indices[idx + i];
                out_probe[match_count] = probe_idx;
                match_count++;
            }
        }
        break;
    }

    // 检查空槽 (SIMD)
    uint32x4_t empty_mask = vceqq_s32(keys_vec, empty_vec);
    if (vmaxvq_u32(empty_mask)) break;  // 有空槽

    idx = (idx + 4) & mask_;  // 线性探测 +4
}
```

**性能**:
- 槽位比较: 4x 并行
- 分支减少: 75% (4 次检查 → 1 次)

#### 3.1.3 8 路展开循环 (Aggregate v21)

**核心问题**: 循环开销 (条件判断、指针递增) 占比高

**解决方案**: 8 路循环展开 + 编译器自动向量化

```cpp
// Aggregate v21 - 8 路展开 SUM
int64_t sum_i32_v21(const int32_t* input, size_t count) {
    const int32_t* __restrict p = input;
    size_t i = 0;

    // 8 独立累加器
    int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    // 8 路展开主循环
    for (; i + 8 <= count; i += 8) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
        s4 += p[i + 4];
        s5 += p[i + 5];
        s6 += p[i + 6];
        s7 += p[i + 7];
    }

    // 归约
    int64_t result = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);

    // 尾部处理
    for (; i < count; ++i) {
        result += p[i];
    }

    return result;
}
```

**优化效果**:
- 循环开销: 8x 减少
- 编译器向量化: 自动转换为 SIMD
- ILP: 8 路并行

### 3.2 Metal GPU 加速

#### 3.2.1 UMA 零拷贝

**核心优势**: Apple M4 的 CPU 和 GPU 共享统一内存，无需拷贝数据

```cpp
// UMA 内存包装
struct UMABuffer {
    void* data;              // CPU 可访问指针
    size_t size;
    id<MTLBuffer> metal_buffer;  // GPU 可访问 Metal Buffer
    bool is_external;        // 是否为外部包装 (零拷贝)
};

// 零拷贝包装外部数据
UMABuffer wrap_external(void* ptr, size_t size) {
    // 检查页对齐 (16KB)
    if ((uintptr_t)ptr % 16384 != 0) {
        // 回退: 分配新缓冲区并拷贝
        return allocate_and_copy(ptr, size);
    }

    // 零拷贝: 直接创建 Metal Buffer 视图
    id<MTLBuffer> mtlBuf = [device
        newBufferWithBytesNoCopy:ptr
        length:size
        options:MTLResourceStorageModeShared
        deallocator:nil];

    return {ptr, size, mtlBuf, true};
}
```

**性能影响**:
- 数据传输时间: **0 ms** (vs 常规 GPU 的拷贝开销)
- 适用条件: 输入数据页对齐 (16KB)

#### 3.2.2 Threadgroup 前缀和 (Filter GPU)

**核心问题**: 全局原子操作争用严重

**解决方案**: Threadgroup 级别前缀和 + 批量写入

```metal
// Filter GPU Kernel - Threadgroup 前缀和
kernel void filter_simd4_i32(
    device const int4* input [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device atomic_uint* counter [[buffer(2)]],
    constant FilterParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    threadgroup uint* tg_counts [[threadgroup(0)]]
) {
    // 1. 向量化比较 (4 元素/线程)
    int4 vals = input[gid];
    bool4 mask = (vals > int4(params.value));

    // 2. 收集本地匹配
    uint local_count = 0;
    uint local_indices[4];
    for (uint i = 0; i < 4; i++) {
        if (mask[i]) local_indices[local_count++] = gid * 4 + i;
    }

    // 3. Threadgroup 前缀和
    tg_counts[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch scan (O(log n) 并行前缀和)
    for (uint d = 1; d < group_size; d *= 2) {
        uint ai = (lid + 1) * d * 2 - 1;
        if (ai < group_size) {
            tg_counts[ai] += tg_counts[ai - d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. 单次全局原子分配
    threadgroup uint tg_global_offset;
    if (lid == group_size - 1) {
        tg_global_offset = atomic_fetch_add_explicit(
            counter, tg_counts[lid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5. 批量写入结果
    uint local_offset = (lid == 0) ? 0 : tg_counts[lid - 1];
    uint global_offset = tg_global_offset + local_offset;
    for (uint i = 0; i < local_count; i++) {
        out_indices[global_offset + i] = local_indices[i];
    }
}
```

**优化效果**:
- 全局原子操作: 从 N 次 → N/group_size 次
- 典型配置: 256 线程/组，减少 256x

#### 3.2.3 Block-local Hash (GPUFusedQ3 v68)

**核心问题**: 全局哈希表争用 + 内存带宽瓶颈

**解决方案**: Threadgroup 级别独立哈希表 + 最终归并

```metal
// Q3 GPU Fusion - Block-local Hash
kernel void fused_q3_kernel(
    device const Customer* customers [[buffer(0)]],
    device const Order* orders [[buffer(1)]],
    device const LineItem* lineitems [[buffer(2)]],
    device GroupResult* final_results [[buffer(3)]],
    device atomic_uint* result_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup HashEntry* local_hash [[threadgroup(0)]],  // 本地哈希表
    threadgroup uint* local_count [[threadgroup(1)]]
) {
    // 1. 初始化 threadgroup 哈希表
    if (lid < LOCAL_HASH_SIZE) {
        local_hash[lid].key = EMPTY_KEY;
        local_hash[lid].revenue = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 处理本批数据 (Filter + Join + Aggregate)
    for (uint i = gid; i < count; i += grid_size) {
        // Filter
        if (!filter_condition(customers, orders, lineitems, i)) continue;

        // Join
        uint order_key = join_lookup(orders, lineitems, i);
        if (order_key == NULL_KEY) continue;

        // Aggregate (本地哈希表)
        uint hash = order_key % LOCAL_HASH_SIZE;
        while (true) {
            uint old_key = atomic_load_explicit(
                &local_hash[hash].key, memory_order_relaxed);

            if (old_key == EMPTY_KEY) {
                // 插入新键
                if (atomic_compare_exchange_weak_explicit(
                    &local_hash[hash].key, &old_key, order_key,
                    memory_order_relaxed, memory_order_relaxed)) {
                    local_hash[hash].revenue = revenue;
                    break;
                }
            } else if (old_key == order_key) {
                // 原子累加
                atomic_fetch_add_explicit(
                    &local_hash[hash].revenue, revenue,
                    memory_order_relaxed);
                break;
            }
            hash = (hash + 1) % LOCAL_HASH_SIZE;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. 合并本地结果到全局 (单线程)
    if (lid == 0) {
        for (uint i = 0; i < LOCAL_HASH_SIZE; i++) {
            if (local_hash[i].key != EMPTY_KEY) {
                uint idx = atomic_fetch_add_explicit(
                    result_count, 1, memory_order_relaxed);
                final_results[idx] = {
                    local_hash[i].key,
                    local_hash[i].revenue
                };
            }
        }
    }
}
```

**优化效果**:
- 内存争用: Threadgroup 内无争用 (共享内存原子)
- 带宽: 减少全局内存访问
- 融合收益: 3 个算子 → 1 次 GPU 调用

### 3.3 完美哈希优化

#### 3.3.1 适用条件检测

```cpp
class PerfectHashTable {
    std::vector<uint32_t> indices_;
    int32_t min_key_, max_key_;

public:
    // 检查是否适合完美哈希
    bool try_build(const int32_t* keys, size_t count) {
        auto [min_it, max_it] = std::minmax_element(keys, keys + count);
        min_key_ = *min_it;
        max_key_ = *max_it;

        size_t range = max_key_ - min_key_ + 1;

        // 条件 1: 范围 <= 2 * count (避免空间浪费)
        if (range > count * 2) return false;

        // 条件 2: 绝对范围 <= 10M (内存限制)
        if (range > 10000000) return false;

        // 分配直接索引数组
        indices_.resize(range, UINT32_MAX);
        for (size_t i = 0; i < count; ++i) {
            indices_[keys[i] - min_key_] = i;
        }

        return true;
    }

    // O(1) 探测
    inline size_t probe(int32_t key, uint32_t& out_idx) {
        if (key < min_key_ || key > max_key_) return 0;
        uint32_t idx = indices_[key - min_key_];
        if (idx == UINT32_MAX) return 0;
        out_idx = idx;
        return 1;
    }
};
```

**性能**:
- 典型场景: Q4/Q21 (订单键 1-150万)
- vs 哈希表: 6.88x (消除哈希计算 + 冲突解决)

### 3.4 Bitmap SEMI Join

#### 3.4.1 位图构建 + SIMD 探测

```cpp
// SEMI Join - Bitmap 优化
size_t semi_join_bitmap(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    uint32_t* out_probe_indices
) {
    // 1. 构建位图 (适用于小整数键)
    auto [min_key, max_key] = std::minmax_element(
        build_keys, build_keys + build_count);
    size_t range = *max_key - *min_key + 1;

    std::vector<uint64_t> bitmap((range + 63) / 64, 0);
    for (size_t i = 0; i < build_count; ++i) {
        size_t bit_idx = build_keys[i] - *min_key;
        bitmap[bit_idx / 64] |= (1ULL << (bit_idx % 64));
    }

    // 2. SIMD 批量探测
    size_t match_count = 0;
    for (size_t i = 0; i < probe_count; i += 4) {
        // 加载 4 个 probe 键
        int32x4_t keys_vec = vld1q_s32(probe_keys + i);
        int32x4_t min_vec = vdupq_n_s32(*min_key);
        int32x4_t max_vec = vdupq_n_s32(*max_key);

        // 范围检查 (SIMD)
        uint32x4_t in_range = vandq_u32(
            vcgeq_s32(keys_vec, min_vec),
            vcleq_s32(keys_vec, max_vec)
        );

        // 标量处理匹配
        uint32_t mask = vget_lane_u32(vreinterpret_u32_u8(
            vshrn_n_u16(vreinterpretq_u16_u32(in_range), 4)), 0);

        for (int j = 0; j < 4 && i + j < probe_count; j++) {
            if (mask & (1 << (j * 8))) {
                int32_t key = probe_keys[i + j];
                size_t bit_idx = key - *min_key;
                if (bitmap[bit_idx / 64] & (1ULL << (bit_idx % 64))) {
                    out_probe_indices[match_count++] = i + j;
                }
            }
        }
    }

    return match_count;
}
```

**性能**:
- 典型场景: Q4 (订单键 SEMI JOIN)
- vs 哈希表: 2.5x (位图访问 vs 哈希探测)

### 3.5 Radix Sort (11-11-10 位分组)

#### 3.5.1 三趟排序优化

```cpp
// Radix Sort - 11-11-10 位分组
void radix_sort_i32_v2(int32_t* data, size_t count) {
    constexpr int BITS[] = {11, 11, 10};      // 3 趟排序
    constexpr int RADIX[] = {2048, 2048, 1024};

    std::vector<int32_t> buffer(count);
    int32_t* src = data;
    int32_t* dst = buffer.data();

    for (int pass = 0; pass < 3; ++pass) {
        // 直方图统计 (SIMD 加速)
        alignas(128) size_t histogram[RADIX[pass]] = {};

        for (size_t i = 0; i < count; i += 4) {
            int32x4_t vals = vld1q_s32(src + i);
            // 提取对应位的数字
            for (int j = 0; j < 4; ++j) {
                int digit = extract_digit(src[i + j], pass, BITS);
                histogram[digit]++;
            }
        }

        // 前缀和
        size_t offset = 0;
        for (auto& h : histogram) {
            size_t tmp = h;
            h = offset;
            offset += tmp;
        }

        // 分发 (缓存友好的顺序写入)
        for (size_t i = 0; i < count; ++i) {
            int digit = extract_digit(src[i], pass, BITS);
            dst[histogram[digit]++] = src[i];
        }

        std::swap(src, dst);
    }

    // 确保结果在 data 中
    if (src != data) {
        memcpy(data, src, count * sizeof(int32_t));
    }
}
```

**性能**:
- vs 快速排序: 1.87x (10M 数据)
- 趟数优化: 3 趟 vs 传统 8-bit 的 4 趟

### 3.6 TopK 采样算法

#### 3.6.1 自适应策略

```cpp
// TopK v4 - 采样算法
void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices) {
    if (k <= 64) {
        // 小 K: 纯堆方法 (L1 缓存常驻)
        topk_heap_small(data, count, k, out_values, out_indices);
    }
    else if (k <= 1024) {
        // 中 K: SIMD 加速堆
        topk_simd_heap(data, count, k, out_values, out_indices);
    }
    else if (k >= count / 10) {
        // 大 K (>10%): 部分排序
        topk_partial_sort(data, count, k, out_values, out_indices);
    }
    else {
        // 超大数据 + 中等 K: 采样 + 精确阈值
        // 1. 采样 1% 数据
        size_t sample_size = std::max<size_t>(1000, count / 100);
        std::vector<int32_t> sample;
        for (size_t i = 0; i < sample_size; ++i) {
            sample.push_back(data[rand() % count]);
        }

        // 2. 在样本中找 TopK
        std::nth_element(sample.begin(), sample.begin() + k, sample.end(),
                         std::greater<int32_t>());
        int32_t threshold = sample[k - 1];

        // 3. 完整扫描收集候选
        std::vector<std::pair<int32_t, uint32_t>> candidates;
        for (size_t i = 0; i < count; ++i) {
            if (data[i] >= threshold) {
                candidates.push_back({data[i], i});
            }
        }

        // 4. 精确选择
        std::nth_element(candidates.begin(), candidates.begin() + k,
                         candidates.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < k; ++i) {
            out_values[i] = candidates[i].first;
            out_indices[i] = candidates[i].second;
        }
    }
}
```

**性能**:
- K=100, N=100M: 24.45x vs DuckDB
- 采样准确率: >99.9% (阈值通常非常接近)

---

## 四、算子版本谱系

### 4.1 Filter 算子演进

| 版本 | 核心技术 | 性能 | 适用场景 |
|------|----------|------|----------|
| **v1** | 基础 SIMD | 基线 | 小数据 |
| **v2** | 位图中间层 | 1.2x | 低选择率 |
| **v3** | 4 累加器 + vsub | 1.8x | 通用 |
| **v15** | 直接索引生成 + LUT | 3.0x | 高选择率 |
| **v19** | 8 线程并行 | 4.5x | 大数据 (>1M) |
| **v19.1** | 线程池优化 | 5.0x | 多次调用 |

**选择策略**:
```cpp
if (count < 100000) {
    return filter_i32_v3(input, count, op, value, out_indices);
} else if (count < 1000000) {
    return filter_i32_v15(input, count, op, value, out_indices);
} else {
    return filter_i32_v19(input, count, op, value, out_indices);
}
```

### 4.2 Join 算子演进

| 版本 | 核心技术 | 性能 | 适用场景 |
|------|----------|------|----------|
| **v3** | SOA 哈希表 + 缓存对齐 | 基线 | 通用 |
| **v4** | 完美哈希检测 | 6.88x | 小整数键 |
| **v5** | 两阶段预分配 | 1.4x | 高匹配率 |
| **v13** | 两阶段 + 紧凑哈希 | 1.5x | 中等数据 |
| **v19.2** | SIMD 并行槽位比较 | 2.0x | 大数据 |
| **GPU v68** | Fused Q3 (Filter+Join+Agg) | 1.46x | Q3 专用 |

**选择策略**:
```cpp
// 1. 检查完美哈希
if (try_perfect_hash(build_keys, build_count)) {
    return hash_join_perfect(build_keys, build_count, ...);
}

// 2. 检查 GPU 条件
size_t total = build_count + probe_count;
if (is_gpu_available() && total >= 500000 && total <= 50000000) {
    return hash_join_gpu(build_keys, build_count, ...);
}

// 3. CPU 路径选择
if (probe_count < 100000) {
    return hash_join_i32_v13(build_keys, build_count, ...);
} else {
    return hash_join_i32_v19_2(build_keys, build_count, ...);
}
```

### 4.3 Aggregate 算子演进

| 版本 | 核心技术 | 性能 | 适用场景 |
|------|----------|------|----------|
| **v2** | 融合 MIN/MAX | 基线 | 简单聚合 |
| **v14** | 寄存器缓冲 (groups<=64) | 4.0x | 低基数分组 |
| **v15** | 8 线程 + 缓存对齐 | 5.0x | 高基数分组 |
| **v21** | 8 路展开 (SUM) | 6.0x | 简单 SUM |

**选择策略**:
```cpp
if (num_groups <= 64) {
    // 低基数: 寄存器缓冲
    group_sum_i32_v14(values, groups, count, num_groups, out_sums);
} else if (count < 100000) {
    // 小数据: 单线程
    group_sum_i32_v4(values, groups, count, num_groups, out_sums);
} else {
    // 大数据: 多线程
    group_sum_i32_v15(values, groups, count, num_groups, out_sums);
}
```

### 4.4 Sort/TopK 算子演进

| 版本 | 核心技术 | 性能 | 适用场景 |
|------|----------|------|----------|
| **v2** | Radix Sort (11-11-10) | 1.87x | 完整排序 |
| **v3** | 自适应堆 | 基线 | 小 K |
| **v4** | 采样算法 | 24.45x | 大数据小 K |
| **v5** | 混合策略 | 最优 | 全场景 |

---

## 五、M4 硬件适配

### 5.1 缓存层次优化

| 缓存级别 | 大小 | 延迟 | 优化策略 |
|----------|------|------|----------|
| **L1 Data** | 64 KB | 4 周期 | 热数据常驻 (堆、寄存器缓冲) |
| **L2** | 4 MB | 12 周期 | 工作集控制 (分块算法) |
| **L3** | 共享 | 40 周期 | 预取隐藏延迟 (激进预取) |
| **主内存** | 128 GB | 120 GB/s | UMA 零拷贝 |

#### 5.1.1 128 字节缓存行对齐

```cpp
// M4 L1 缓存行 = 128 字节
constexpr size_t M4_CACHE_LINE = 128;

// 所有数据结构对齐
struct alignas(128) CacheAlignedData {
    int32_t data[32];  // 128 / 4 = 32 个 int32
};

// 哈希表键数组对齐
alignas(128) std::vector<int32_t> keys_;
```

#### 5.1.2 多级预取

```cpp
// 3 级预取策略
for (size_t i = 0; i < count; i += 4) {
    // L1 预取 (近距离)
    __builtin_prefetch(input + i + 16, 0, 3);

    // L2 预取 (中距离)
    __builtin_prefetch(input + i + 64, 0, 2);

    // L3 预取 (远距离)
    __builtin_prefetch(input + i + 128, 0, 1);

    // 处理当前批次
    process_batch(input + i);
}
```

### 5.2 CRC32 硬件哈希

```cpp
#include <arm_acle.h>

// 单周期 CRC32 哈希
inline uint32_t hash_crc32(int32_t key) {
    return __crc32cw(0, static_cast<uint32_t>(key));
}

// 批量哈希 (8 路并行)
void hash_batch_crc32(const int32_t* keys, uint32_t* hashes, size_t count) {
    for (size_t i = 0; i < count; i += 8) {
        // 8 独立哈希 (无依赖链)
        hashes[i + 0] = __crc32cw(0, keys[i + 0]);
        hashes[i + 1] = __crc32cw(0, keys[i + 1]);
        hashes[i + 2] = __crc32cw(0, keys[i + 2]);
        hashes[i + 3] = __crc32cw(0, keys[i + 3]);
        hashes[i + 4] = __crc32cw(0, keys[i + 4]);
        hashes[i + 5] = __crc32cw(0, keys[i + 5]);
        hashes[i + 6] = __crc32cw(0, keys[i + 6]);
        hashes[i + 7] = __crc32cw(0, keys[i + 7]);
    }
}
```

**性能**:
- vs 普通哈希: 5-10x (单周期 vs 多次乘法)
- 质量: CRC32 分布优于 MurmurHash

### 5.3 多核并行策略

#### 5.3.1 线程池设计

```cpp
class ThreadPool {
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;

public:
    ThreadPool(size_t num_threads = 8) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::forward<F>(f));
        }
        condition_.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }
};

// 全局线程池 (预热)
static ThreadPool& get_thread_pool() {
    static ThreadPool pool(8);  // M4 Max: 4P + 4E (预留 2 核)
    return pool;
}
```

#### 5.3.2 Thread-Local 聚合

```cpp
// 无锁并行聚合
void group_sum_parallel(const int32_t* values, const uint32_t* groups,
                        size_t count, size_t num_groups, int64_t* out_sums) {
    constexpr size_t NUM_THREADS = 8;
    alignas(128) int64_t thread_sums[NUM_THREADS][MAX_GROUPS];

    // 1. 并行阶段 (无锁)
    auto& pool = get_thread_pool();
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        pool.enqueue([&, t] {
            // 初始化本地累加器
            std::memset(thread_sums[t], 0, num_groups * sizeof(int64_t));

            // 处理本线程负责的数据
            size_t start = t * count / NUM_THREADS;
            size_t end = (t + 1) * count / NUM_THREADS;

            for (size_t i = start; i < end; ++i) {
                thread_sums[t][groups[i]] += values[i];
            }
        });
    }

    // 2. 等待完成
    pool.wait_all();

    // 3. SIMD 合并
    for (size_t g = 0; g < num_groups; g += 2) {
        int64x2_t sum = vdupq_n_s64(0);
        for (size_t t = 0; t < NUM_THREADS; ++t) {
            int64x2_t vals = vld1q_s64(&thread_sums[t][g]);
            sum = vaddq_s64(sum, vals);
        }
        vst1q_s64(&out_sums[g], sum);
    }
}
```

---

## 六、API 参考

### 6.1 Filter API

```cpp
namespace thunderduck::filter {

enum class CompareOp { EQ, NE, LT, LE, GT, GE };

// 推荐: v19 (自动选择版本)
size_t filter_i32_v19(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices);

// 高级: 手动选择
size_t filter_i32_v3(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);  // 小数据

size_t filter_i32_v15(const int32_t* input, size_t count,
                       CompareOp op, int32_t value,
                       uint32_t* out_indices);  // 中等数据

size_t filter_i32_v19_1(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         uint32_t* out_indices);  // 大数据 + 线程池

} // namespace thunderduck::filter
```

### 6.2 Join API

```cpp
namespace thunderduck::join {

enum class JoinType { INNER, LEFT, RIGHT, FULL, SEMI, ANTI };

struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

// 推荐: v19.2 (自适应)
size_t hash_join_i32_v19_2(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinType join_type,
                            JoinResult* result);

// GPU 路径
size_t semi_join_gpu(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result);

size_t inner_join_gpu(const int32_t* build_keys, size_t build_count,
                       const int32_t* probe_keys, size_t probe_count,
                       JoinResult* result);

} // namespace thunderduck::join
```

### 6.3 Aggregate API

```cpp
namespace thunderduck::aggregate {

// 简单聚合 (推荐 v21)
int64_t sum_i32_v21(const int32_t* input, size_t count);

void minmax_i32_v4(const int32_t* input, size_t count,
                   int32_t* out_min, int32_t* out_max);

// 分组聚合 (推荐 v15)
void group_sum_i32_v15(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums);

void group_count_v15(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts);

} // namespace thunderduck::aggregate
```

### 6.4 Sort/TopK API

```cpp
namespace thunderduck::sort {

enum class SortOrder { ASC, DESC };

// Radix Sort
void radix_sort_i32_v2(int32_t* data, size_t count,
                        SortOrder order = SortOrder::ASC);

// TopK (推荐 v4)
void topk_max_i32_v4(const int32_t* data, size_t count, size_t k,
                     int32_t* out_values, uint32_t* out_indices);

} // namespace thunderduck::sort
```

---

## 七、构建与测试

### 7.1 构建

```bash
# 进入构建目录
cd /Users/sihaoli/ThunderDuck/build

# CMake 配置 (Release 模式)
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译 (使用全部核心)
make -j$(sysctl -n hw.ncpu)

# 运行 TPC-H 基准测试
./benchmark/tpch/tpch_benchmark_main --sf 1
```

### 7.2 编译选项

```cmake
# CMakeLists.txt 关键配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a+crc -mcpu=native -flto")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DTHUNDERDUCK_ARM64")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DTHUNDERDUCK_NEON")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DTHUNDERDUCK_HAS_METAL")

# Metal 编译
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
target_link_libraries(thunderduck ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
```

### 7.3 基准测试规范

```cpp
// 使用中位数而非平均值
std::vector<double> times;
for (int i = 0; i < 30; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    run_query();
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
}

// IQR 剔除异常值
std::sort(times.begin(), times.end());
double q1 = times[times.size() / 4];
double q3 = times[times.size() * 3 / 4];
double iqr = q3 - q1;
double lower = q1 - 1.5 * iqr;
double upper = q3 + 1.5 * iqr;

times.erase(std::remove_if(times.begin(), times.end(),
    [&](double t) { return t < lower || t > upper; }), times.end());

// 取中位数
double median = times[times.size() / 2];

// 计算标准差
double sq_sum = 0;
for (double t : times) sq_sum += (t - median) * (t - median);
double stddev = sqrt(sq_sum / times.size());

std::cout << "Median: " << median << " ms (σ=" << stddev << ")\n";
```

---

## 八、扩展性设计

### 8.1 新增算子

**步骤**:
1. 在 `include/thunderduck/` 创建头文件定义接口
2. 在 `src/operators/<category>/` 实现算子
3. 在 `OperatorRegistry` 注册版本
4. 添加单元测试和性能基准

**示例**:
```cpp
// 1. 头文件: include/thunderduck/window.h
namespace thunderduck::window {
    size_t window_sum_v1(const int32_t* input, size_t count,
                         size_t window_size, int64_t* out_sums);
}

// 2. 实现: src/operators/window/simd_window.cpp
size_t window_sum_v1(const int32_t* input, size_t count,
                     size_t window_size, int64_t* out_sums) {
    // SIMD 滑动窗口实现
}

// 3. 注册: src/core/operator_registry.cpp
registry.register_operator("window_sum", {
    {"v1", &window_sum_v1}
});
```

### 8.2 新增数据类型

```cpp
// 模板化设计支持多类型
template<typename T>
size_t filter_gt(const T* input, size_t count, T value, uint32_t* out);

// 特化实现
template<>
size_t filter_gt<int32_t>(...) { /* ARM Neon int32 */ }

template<>
size_t filter_gt<int64_t>(...) { /* ARM Neon int64 */ }

template<>
size_t filter_gt<float>(...) { /* ARM Neon float32 */ }
```

---

## 九、性能调优指南

### 9.1 数据对齐

```cpp
// 分配 128 字节对齐内存
void* aligned_alloc_128(size_t size) {
    void* ptr = nullptr;
    posix_memalign(&ptr, 128, size);
    return ptr;
}

// 检查对齐
bool is_aligned_128(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 128) == 0;
}
```

### 9.2 预取优化

```cpp
// 预取参数
// rw: 0=读, 1=写
// locality: 0=不保留, 1=L3, 2=L2, 3=L1
__builtin_prefetch(ptr, rw, locality);

// 典型配置
__builtin_prefetch(input + i + 64, 0, 3);  // L1 读取
__builtin_prefetch(output + i + 32, 1, 2); // L2 写入
```

### 9.3 分支消除

```cpp
// 低效: 分支
for (size_t i = 0; i < count; ++i) {
    if (input[i] > value) {
        out[match_count++] = i;
    }
}

// 高效: 无分支 SIMD
uint32x4_t mask = vcgtq_s32(data_vec, value_vec);
count = vsubq_u32(count, mask);  // -1 转 +1
```

---

## 十、已知限制

### 10.1 硬件依赖

- **平台**: 仅支持 Apple Silicon (M1/M2/M3/M4)
- **系统**: macOS 14.0+ (Metal 3 需求)
- **编译器**: Clang 15+ (ARM Neon intrinsics)

### 10.2 内存限制

- **完美哈希**: 键范围 <= 10M
- **Bitmap SEMI Join**: 键范围 <= 100M
- **GPU**: 输入数据建议页对齐 (16KB) 以启用零拷贝

### 10.3 功能限制

- **字符串**: 有限支持 (定长字符串 SIMD 优化)
- **复杂表达式**: 不支持 UDF (用户自定义函数)
- **并发**: 单查询串行执行 (多查询可并行)

---

## 十一、未来规划

### 11.1 短期目标 (V48-V50)

- [ ] Q6 Filter-Aggregate 融合优化 (目标 3.0x)
- [ ] Q8/Q13/Q22 深度优化 (目标 1.5x+)
- [ ] GPU Operator 框架完善 (v69 推广)
- [ ] 字符串算子 SIMD 加速

### 11.2 中期目标 (V51-V60)

- [ ] 自适应查询优化器 (基于历史统计)
- [ ] 代码生成框架 (运行时编译)
- [ ] 分布式执行引擎 (多机并行)
- [ ] 持久化存储引擎集成

### 11.3 长期愿景

- [ ] 完整 SQL 支持 (超越 TPC-H)
- [ ] OLAP/OLTP 混合负载
- [ ] 实时分析平台
- [ ] 云原生部署

---

## 十二、参考文献

### 12.1 学术论文

1. **Morsel-Driven Parallelism** - SIGMOD 2014
   并行执行模型，Morsel 批处理

2. **Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask** - VLDB 2018
   向量化 vs 代码生成对比

3. **Processing Analytical Workloads on Apple Silicon GPUs** - SIGMOD 2025
   UMA 架构优化 (GFTR 模式)

### 12.2 技术文档

- [ARM Neon Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [DuckDB Architecture](https://duckdb.org/docs/internals/overview)

### 12.3 项目文档

- `docs/V47_TPCH_COMPREHENSIVE_BENCHMARK.md` - 最新性能报告
- `docs/PROJECT_OVERVIEW.md` - 项目总览
- `docs/V35_GENERIC_ARCHITECTURE.md` - 通用架构设计

---

**文档结束**

*ThunderDuck - 为 Apple Silicon 打造的极速 SQL 引擎*
*版本 8.0.0 | 2026-01-31*
