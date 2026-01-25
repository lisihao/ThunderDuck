# ThunderDuck 架构设计文档

> **版本**: 2.0.0 | **更新日期**: 2026-01-24
> **目标平台**: Apple M4, macOS 15.0+

---

## 一、系统概述

ThunderDuck 是针对 Apple Silicon (M4) 优化的高性能数据库算子后端，利用统一内存架构 (UMA) 实现 CPU/GPU 零拷贝数据共享，显著提升查询性能。

### 1.1 设计目标

| 目标 | 描述 |
|------|------|
| **零拷贝** | 利用 UMA 消除 CPU-GPU 数据传输开销 |
| **自适应** | 根据数据特征自动选择最优执行器 (CPU/GPU/NPU) |
| **高吞吐** | 目标 >100 GB/s 内存带宽利用率 |
| **兼容性** | 与 DuckDB API 兼容，可作为后端替换 |

### 1.2 核心优势

```
┌─────────────────────────────────────────────────────────────┐
│                    ThunderDuck 核心优势                      │
├─────────────────────────────────────────────────────────────┤
│  1. UMA 零拷贝      - CPU/GPU 共享内存，无需数据传输         │
│  2. 自适应策略      - 自动选择 CPU SIMD / GPU / NPU          │
│  3. SIMD 向量化     - ARM NEON 128-bit 并行处理              │
│  4. GPU 并行        - Metal Compute Shader 大规模并行        │
│  5. 缓冲区池复用    - 减少内存分配开销                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、系统架构

### 2.1 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         应用层 (Application)                      │
│                    DuckDB / SQL 查询接口                          │
├──────────────────────────────────────────────────────────────────┤
│                       算子层 (Operators)                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Filter  │  │Aggregate│  │  Join   │  │  Sort   │  │  TopK   │ │
│  │  v3/v4  │  │  v2/v3  │  │ v3/v4   │  │  Radix  │  │   v5    │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │            │            │            │            │       │
├───────┴────────────┴────────────┴────────────┴────────────┴──────┤
│                    策略选择层 (Strategy Selector)                 │
│              AdaptiveStrategySelector - 自动选择执行路径          │
├──────────────────────────────────────────────────────────────────┤
│                      执行层 (Executors)                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
│  │ CPU SIMD   │  │ GPU Metal  │  │ NPU BNNS   │                  │
│  │ ARM NEON   │  │ Compute    │  │ Accelerate │                  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                  │
│        │               │               │                          │
├────────┴───────────────┴───────────────┴─────────────────────────┤
│                     内存管理层 (Memory)                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              UMA Memory Manager                           │    │
│  │  - 页对齐分配 (16KB)                                      │    │
│  │  - Metal Buffer 包装 (零拷贝)                             │    │
│  │  - 缓冲区池复用                                           │    │
│  └──────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│                      硬件层 (Hardware)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ CPU Core │  │   GPU    │  │   NPU    │  │ Unified  │          │
│  │ ARM v8.6 │  │  Metal   │  │ ANE/BNNS │  │  Memory  │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖

```
thunderduck/
├── include/thunderduck/
│   ├── filter.h           # Filter 算子接口
│   ├── aggregate.h        # Aggregate 算子接口
│   ├── join.h             # Join 算子接口
│   ├── sort.h             # Sort/TopK 算子接口
│   ├── uma_memory.h       # UMA 内存管理接口
│   └── adaptive_strategy.h # 策略选择器接口
│
├── src/
│   ├── core/
│   │   ├── uma_memory.cpp          # UMA 内存管理实现
│   │   ├── adaptive_strategy.cpp   # 策略选择器实现
│   │   └── memory_allocator.cpp    # 通用内存分配器
│   │
│   ├── operators/
│   │   ├── filter/
│   │   │   ├── simd_filter.cpp     # Filter v1 (基础 SIMD)
│   │   │   ├── simd_filter_v2.cpp  # Filter v2 (优化 SIMD)
│   │   │   └── simd_filter_v3.cpp  # Filter v3 (批量处理)
│   │   │
│   │   ├── aggregate/
│   │   │   ├── simd_aggregate.cpp  # Aggregate v1
│   │   │   └── simd_aggregate_v2.cpp # Aggregate v2 (融合)
│   │   │
│   │   ├── join/
│   │   │   ├── hash_join_v3.cpp    # Join v3 (CPU 优化)
│   │   │   └── hash_join_v4.cpp    # Join v4 (多策略)
│   │   │
│   │   └── sort/
│   │       ├── radix_sort.cpp      # 基数排序
│   │       └── topk_v5.cpp         # TopK v5 (采样)
│   │
│   ├── gpu/
│   │   ├── filter_uma.mm           # Filter GPU (Metal)
│   │   ├── aggregate_uma.mm        # Aggregate GPU (Metal)
│   │   ├── hash_join_uma.mm        # Join GPU (Metal + UMA)
│   │   └── topk_uma.mm             # TopK GPU (Metal)
│   │
│   └── npu/
│       └── bloom_bnns.cpp          # Bloom Filter NPU 加速
│
└── benchmark/
    ├── benchmark_app.cpp           # 基准测试主程序
    └── comprehensive_uma_benchmark.cpp # 全面性能测试
```

---

## 三、核心组件设计

### 3.1 UMA 内存管理器

**文件**: `src/core/uma_memory.cpp`, `include/thunderduck/uma_memory.h`

```cpp
namespace thunderduck::uma {

// UMA 缓冲区 - CPU/GPU 共享
struct UMABuffer {
    void* data;              // CPU 可访问指针
    size_t size;             // 缓冲区大小
    void* metal_buffer;      // Metal Buffer (id<MTLBuffer>)
    bool is_external;        // 是否为外部包装
};

class UMAMemoryManager {
public:
    static UMAMemoryManager& instance();

    // 分配 UMA 内存 (16KB 页对齐)
    UMABuffer allocate(size_t size);

    // 包装外部指针 (零拷贝)
    UMABuffer wrap_external(void* ptr, size_t size);

    // 缓冲区池操作
    UMABuffer acquire_from_pool(size_t size);
    void release_to_pool(UMABuffer& buffer);

    // Metal 设备访问
    void* get_metal_device();

private:
    id<MTLDevice> device_;
    std::vector<UMABuffer> buffer_pool_;
    std::mutex pool_mutex_;
};

// 页对齐检查
inline bool is_page_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16384) == 0;
}

} // namespace thunderduck::uma
```

**设计要点**:
1. **16KB 页对齐**: 匹配 macOS 虚拟内存页大小
2. **零拷贝包装**: 页对齐内存可直接创建 Metal Buffer 视图
3. **缓冲区池**: 复用已分配的 Metal Buffer，减少分配开销
4. **线程安全**: 池操作使用互斥锁保护

### 3.2 自适应策略选择器

**文件**: `src/core/adaptive_strategy.cpp`, `include/thunderduck/adaptive_strategy.h`

```cpp
namespace thunderduck::strategy {

// 执行器类型
enum class Executor {
    CPU_SCALAR,    // 标量 CPU
    CPU_SIMD,      // SIMD 向量化
    GPU,           // Metal GPU
    NPU            // Neural Engine (BNNS)
};

// 算子类型
enum class OperatorType {
    FILTER,
    AGGREGATE_SUM,
    AGGREGATE_MINMAX,
    JOIN,
    TOPK,
    SORT
};

// 数据特征
struct DataCharacteristics {
    size_t row_count;
    size_t column_count;
    size_t element_size;
    float selectivity;        // 选择率 (0-1)
    float cardinality_ratio;  // 基数比
    bool is_page_aligned;     // 是否页对齐
};

// 阈值配置
namespace thresholds {
    constexpr size_t FILTER_GPU_MIN     = 10'000'000;   // 10M
    constexpr size_t AGGREGATE_GPU_MIN  = 100'000'000;  // 100M
    constexpr size_t JOIN_GPU_MIN_PROBE = 500'000;      // 500K
    constexpr size_t JOIN_GPU_MAX_PROBE = 50'000'000;   // 50M
    constexpr size_t TOPK_GPU_MIN       = 50'000'000;   // 50M
}

class StrategySelector {
public:
    static StrategySelector& instance();

    // 选择最优执行器
    Executor select(OperatorType op, const DataCharacteristics& data);

    // 更新性能统计 (用于学习)
    void update_stats(OperatorType op, Executor exec,
                      double throughput, double latency);

private:
    // 性能历史记录
    std::unordered_map<std::pair<OperatorType, Executor>,
                       PerformanceStats> stats_;
};

} // namespace thunderduck::strategy
```

**选择逻辑**:

```
┌─────────────────────────────────────────────────────────────┐
│                    策略选择决策树                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Filter:                                                    │
│    rows < 10M         → CPU SIMD                            │
│    rows >= 10M        → GPU (if available) else CPU SIMD    │
│                                                             │
│  Aggregate:                                                 │
│    rows < 100M        → CPU SIMD (已接近带宽上限)            │
│    rows >= 100M       → GPU (可获得额外并行度)               │
│                                                             │
│  Join:                                                      │
│    probe < 500K       → CPU (GPU 启动开销 > 收益)            │
│    500K <= probe <= 50M → GPU (最佳加速区间)                 │
│    probe > 50M        → GPU (带宽受限，加速有限)             │
│                                                             │
│  TopK:                                                      │
│    rows < 50M         → CPU Sample (采样高效)                │
│    rows >= 50M        → GPU (并行优势)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Filter 算子

**文件**: `src/operators/filter/simd_filter_v3.cpp`, `src/gpu/filter_uma.mm`

```cpp
namespace thunderduck::filter {

// 比较操作
enum class CompareOp { EQ, NE, LT, LE, GT, GE };

// 过滤策略
enum class FilterStrategy {
    AUTO,        // 自动选择
    CPU_SIMD,    // CPU NEON SIMD
    GPU_ATOMIC,  // GPU 原子版
    GPU_SCAN     // GPU 前缀和版
};

// CPU SIMD 实现 (v3)
size_t filter_i32_v3(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

// GPU UMA 实现 (v4)
size_t filter_i32_v4(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices);

} // namespace thunderduck::filter
```

**GPU Shader (内联)**:

```metal
// filter_simd4_i32 - 向量化过滤 + threadgroup 前缀和
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

    // 2. 收集匹配索引
    uint local_count = 0;
    uint local_indices[4];
    for (uint i = 0; i < 4; i++) {
        if (mask[i]) local_indices[local_count++] = gid * 4 + i;
    }

    // 3. Threadgroup 前缀和 (减少原子争用)
    tg_counts[lid] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch scan...

    // 4. 批量写入结果
    uint global_offset = tg_global_offset + local_offset;
    for (uint i = 0; i < local_count; i++) {
        out_indices[global_offset + i] = local_indices[i];
    }
}
```

### 3.4 Join 算子

**文件**: `src/operators/join/hash_join_v3.cpp`, `src/gpu/hash_join_uma.mm`

```cpp
namespace thunderduck::join {

// Join 类型
enum class JoinType { INNER, LEFT, RIGHT, FULL };

// Join 结果
struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

// CPU v3 实现 (SOA 哈希表 + SIMD)
size_t hash_join_i32_v3(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result);

// GPU UMA 实现 (前缀和批量写入)
size_t hash_join_gpu_uma(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

} // namespace thunderduck::join
```

**GPU 执行流程**:

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Hash Join 流水线                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  步骤 1: 分配 UMA 缓冲区                                    │
│    - 哈希表 keys/indices                                    │
│    - 输出 build/probe indices                               │
│    - 计数器                                                  │
│                                                             │
│  步骤 2: GPU 构建哈希表 (build_hash_table kernel)           │
│    - 并行插入，原子 CAS 解决冲突                             │
│    - 线性探测                                                │
│                                                             │
│  步骤 3: SharedEvent 同步                                   │
│    - 确保构建完成后再探测                                    │
│                                                             │
│  步骤 4: GPU 探测 (probe_hash_table kernel)                 │
│    - Threadgroup 级别前缀和                                  │
│    - 批量写入 (减少全局原子操作)                             │
│                                                             │
│  步骤 5: 读取结果 (零拷贝)                                  │
│    - 直接从 UMA 内存读取                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、Metal Shader 设计

### 4.1 内联 Shader 策略

为避免文件路径问题，所有 Metal shader 代码均内联到 .mm 文件中：

```objc
NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void my_kernel(...) {
    // shader 实现
}
)";

MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
options.mathMode = MTLMathModeFast;  // macOS 15+

id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                               options:options
                                                 error:&error];
```

### 4.2 Threadgroup 内存使用

```metal
// 声明 threadgroup 内存
kernel void my_kernel(
    ...
    threadgroup uint32_t* shared_data [[threadgroup(0)]]
) {
    // 使用共享内存
}

// 主机端分配
[encoder setThreadgroupMemoryLength:256 * sizeof(uint32_t) atIndex:0];
```

### 4.3 原子操作优化

```metal
// 低效: 每线程一次全局原子
atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);

// 高效: threadgroup 级别归约后单次全局原子
tg_counts[lid] = local_count;
threadgroup_barrier(mem_flags::mem_threadgroup);
// prefix sum...
if (lid == group_size - 1) {
    tg_global_offset = atomic_fetch_add_explicit(
        counter, tg_total, memory_order_relaxed);
}
```

---

## 五、错误处理

### 5.1 GPU 不可用回退

```cpp
size_t filter_i32_v4(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices) {
    auto& ctx = getContext();

    // GPU 不可用，回退到 CPU
    if (!ctx.initialize() || !ctx.filterAtomicI32) {
        return filter_i32_v3(input, count, op, value, out_indices);
    }

    // GPU 路径...
}
```

### 5.2 内存分配失败

```cpp
UMABuffer mgr.allocate(size_t size) {
    // 尝试 UMA 分配
    void* ptr = nullptr;
    posix_memalign(&ptr, 16384, size);

    if (!ptr) {
        // 回退到普通分配
        ptr = malloc(size);
        return {ptr, size, nullptr, false};
    }

    // 创建 Metal Buffer 视图
    id<MTLBuffer> mtlBuf = [device newBufferWithBytesNoCopy:ptr
                                                     length:size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    return {ptr, size, (__bridge void*)mtlBuf, false};
}
```

---

## 六、线程安全

### 6.1 单例模式

```cpp
class MetalContextUMA {
public:
    static MetalContextUMA& instance() {
        static MetalContextUMA ctx;  // C++11 线程安全
        return ctx;
    }

private:
    MetalContextUMA() { /* 初始化 */ }
};
```

### 6.2 缓冲区池锁

```cpp
class UMAMemoryManager {
    std::mutex pool_mutex_;

    UMABuffer acquire_from_pool(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        // 查找合适缓冲区...
    }

    void release_to_pool(UMABuffer& buf) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        // 归还缓冲区...
    }
};
```

---

## 七、扩展性设计

### 7.1 新算子接入

```cpp
// 1. 定义接口 (include/thunderduck/new_op.h)
namespace thunderduck::new_op {
    size_t new_operation_v1(const int32_t* input, size_t count, ...);
}

// 2. 实现 CPU 版本 (src/operators/new_op/simd_new_op.cpp)
size_t new_operation_v1(...) {
    // SIMD 实现
}

// 3. 实现 GPU 版本 (src/gpu/new_op_uma.mm)
size_t new_operation_gpu(...) {
    // Metal 实现
}

// 4. 集成策略选择器
enum class OperatorType {
    // ...
    NEW_OP  // 新增
};
```

### 7.2 新执行器接入

```cpp
enum class Executor {
    CPU_SCALAR,
    CPU_SIMD,
    GPU,
    NPU,
    // 新执行器
    FPGA,
    REMOTE_GPU
};
```

---

## 八、构建配置

### 8.1 Makefile 关键配置

```makefile
CXX = clang++
CXXFLAGS = -std=c++17 -O3 -mcpu=native -march=armv8-a+crc
CXXFLAGS += -DTHUNDERDUCK_ARM64 -DTHUNDERDUCK_NEON
CXXFLAGS += -DTHUNDERDUCK_HAS_METAL -DTHUNDERDUCK_HAS_BNNS

LDFLAGS = -framework Metal -framework Foundation -framework Accelerate
```

### 8.2 条件编译

```cpp
#ifdef THUNDERDUCK_HAS_METAL
    // Metal GPU 代码
#endif

#ifdef THUNDERDUCK_HAS_BNNS
    // BNNS NPU 代码
#endif

#ifdef THUNDERDUCK_NEON
    // ARM NEON SIMD 代码
#endif
```
