# UMA + Arrow 零拷贝架构设计

> **版本**: 1.0.0 | **日期**: 2026-01-25

## 一、需求概述

### 1.1 目标

利用 Apple Silicon 统一内存架构 (UMA)，实现 CPU-GPU-NPU 之间的 Apache Arrow 零拷贝数据流，消除处理单元间的数据传输开销。

### 1.2 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| UMAMemoryManager | ✅ 已实现 | MTLResourceStorageModeShared 零拷贝 |
| Buffer Pool | ✅ 已实现 | 缓冲区复用 |
| JoinResultUMA | ✅ 已实现 | Join 结果零拷贝 |
| Arrow 集成 | ❌ 未实现 | 目标 |
| NPU 协调 | ❌ 未实现 | 目标 |

### 1.3 预期收益

```
数据流对比:

[传统架构]
CPU → memcpy → GPU显存 → compute → memcpy → CPU内存
延迟: ~2ms (10M rows)

[UMA 零拷贝]
CPU ───────────────────────────────────────> GPU/NPU
        共享物理内存，延迟 < 1μs
```

---

## 二、技术方案

### 2.1 核心设计原则

1. **单一内存地址空间**: CPU/GPU/NPU 使用相同物理地址
2. **Arrow 原生格式**: 直接操作 Arrow 列式内存布局
3. **延迟绑定**: 按需创建 Metal/BNNS 资源句柄
4. **零额外拷贝**: 从分配到计算完成，数据不移动

### 2.2 内存布局

```
┌──────────────────────────────────────────────────────────────┐
│                    UMA 物理内存 (共享)                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           ArrowUMAColumn                             │   │
│   │  ┌─────────────────────────────────────────────┐    │   │
│   │  │ Validity Bitmap (64-bit aligned)            │    │   │
│   │  │ [11111111][11111110][...] (bit-packed)      │    │   │
│   │  └─────────────────────────────────────────────┘    │   │
│   │  ┌─────────────────────────────────────────────┐    │   │
│   │  │ Data Buffer (type-specific alignment)       │    │   │
│   │  │ int32: [1, 2, 3, 4, ...]                    │    │   │
│   │  │ float: [1.0, 2.0, 3.0, ...]                 │    │   │
│   │  └─────────────────────────────────────────────┘    │   │
│   │  ┌─────────────────────────────────────────────┐    │   │
│   │  │ Offsets (for variable-length types)         │    │   │
│   │  │ [0, 5, 12, 18, ...]                         │    │   │
│   │  └─────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   CPU: 直接指针访问                                          │
│   GPU: MTLBuffer (StorageModeShared)                        │
│   NPU: BNNS 直接读取 (via Accelerate)                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 处理单元协调

```
                    ┌────────────────┐
                    │  Coordinator   │
                    │  (CPU Main)    │
                    └───────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │     CPU       │ │     GPU       │ │     NPU       │
    │  (Neon SIMD)  │ │   (Metal)     │ │   (BNNS)      │
    └───────────────┘ └───────────────┘ └───────────────┘
            │               │               │
            └───────────────┴───────────────┘
                            │
                    ┌───────▼────────┐
                    │  ArrowUMABatch │
                    │  (零拷贝共享)   │
                    └────────────────┘
```

---

## 三、详细设计

### 3.1 Arrow 列适配器

```cpp
// include/thunderduck/arrow_uma.h

namespace thunderduck {
namespace arrow {

/**
 * Arrow 数据类型
 */
enum class ArrowType {
    INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FLOAT32, FLOAT64,
    BOOL,
    STRING,      // 变长字符串
    BINARY,      // 变长二进制
    FIXED_BINARY // 定长二进制
};

/**
 * Arrow UMA 列 - 零拷贝列式数据
 *
 * 内存布局符合 Arrow 规范:
 * - Validity Bitmap: 64-bit aligned, bit-packed
 * - Data Buffer: type-specific alignment
 * - Offsets: int32/int64 for variable-length types
 */
class ArrowUMAColumn {
public:
    ArrowType type;
    size_t length;           // 行数
    size_t null_count;       // NULL 数量

    // 底层 UMA 缓冲区 (GPU 可直接访问)
    uma::UMABuffer validity_buffer;   // 位图
    uma::UMABuffer data_buffer;       // 数据
    uma::UMABuffer offsets_buffer;    // 偏移 (变长类型)

    // 创建固定长度列
    static ArrowUMAColumn* create_fixed(ArrowType type, size_t length);

    // 创建变长列
    static ArrowUMAColumn* create_variable(ArrowType type, size_t length,
                                           size_t total_bytes);

    // 包装外部 Arrow 内存 (零拷贝)
    static ArrowUMAColumn* wrap_external(
        ArrowType type, size_t length,
        void* validity, void* data, void* offsets,
        size_t data_size, size_t offsets_size);

    // 销毁
    static void destroy(ArrowUMAColumn* col);

    // 类型安全访问
    template<typename T> T* data() {
        return data_buffer.as<T>();
    }

    template<typename T> const T* data() const {
        return data_buffer.as<const T>();
    }

    // 位图操作
    bool is_valid(size_t idx) const;
    void set_valid(size_t idx, bool valid);

    // 字节大小
    size_t data_size() const;
    size_t total_size() const;
};

/**
 * Arrow UMA 批次 - 多列的零拷贝容器
 */
class ArrowUMABatch {
public:
    std::vector<ArrowUMAColumn*> columns;
    size_t num_rows;

    // 创建批次
    static ArrowUMABatch* create(size_t num_rows);

    // 添加列
    void add_column(ArrowUMAColumn* col);

    // 获取列
    ArrowUMAColumn* column(size_t idx);
    const ArrowUMAColumn* column(size_t idx) const;

    // GPU 同步点 (确保 CPU 写入对 GPU 可见)
    void sync_for_gpu();

    // CPU 同步点 (确保 GPU 写入对 CPU 可见)
    void sync_for_cpu();

    // 销毁
    static void destroy(ArrowUMABatch* batch);
};

} // namespace arrow
} // namespace thunderduck
```

### 3.2 处理单元协调器

```cpp
// include/thunderduck/uma_coordinator.h

namespace thunderduck {
namespace uma {

/**
 * 处理单元类型
 */
enum class ProcessingUnit {
    CPU,       // Neon SIMD
    GPU,       // Metal Compute
    NPU,       // BNNS/ANE
    AUTO       // 自动选择
};

/**
 * 计算任务
 */
struct ComputeTask {
    enum class Type {
        FILTER,
        AGGREGATE,
        JOIN,
        SORT,
        VECTOR_SIMILARITY
    };

    Type type;
    arrow::ArrowUMABatch* input;
    arrow::ArrowUMABatch* output;
    void* params;

    // 推荐的处理单元
    ProcessingUnit recommended_unit;

    // 估算成本
    size_t estimated_cpu_cycles;
    size_t estimated_gpu_threads;
};

/**
 * UMA 协调器 - 管理 CPU/GPU/NPU 计算分发
 */
class UMACoordinator {
public:
    static UMACoordinator& instance();

    /**
     * 提交计算任务
     *
     * @param task 计算任务
     * @param unit 目标处理单元 (AUTO = 自动选择)
     * @return 任务句柄
     */
    uint64_t submit(ComputeTask& task, ProcessingUnit unit = ProcessingUnit::AUTO);

    /**
     * 等待任务完成
     */
    void wait(uint64_t task_handle);

    /**
     * 等待所有任务完成
     */
    void wait_all();

    /**
     * 选择最优处理单元
     */
    ProcessingUnit select_best_unit(const ComputeTask& task);

    /**
     * 获取处理单元状态
     */
    struct UnitStatus {
        bool available;
        size_t pending_tasks;
        double utilization;  // 0.0 - 1.0
    };
    UnitStatus get_unit_status(ProcessingUnit unit);

private:
    UMACoordinator();
    ~UMACoordinator();

    struct Impl;
    Impl* impl_;
};

/**
 * 选择策略
 */
class UnitSelector {
public:
    /**
     * 基于数据规模选择
     */
    static ProcessingUnit select_by_size(size_t num_rows, size_t num_cols) {
        // NPU: 不适合通用数据库操作，跳过
        // GPU: > 100K rows 时更高效
        // CPU: 小数据量更快 (避免 GPU 调度开销)

        if (num_rows >= 100000 && num_cols >= 2) {
            return ProcessingUnit::GPU;
        }
        return ProcessingUnit::CPU;
    }

    /**
     * 基于操作类型选择
     */
    static ProcessingUnit select_by_operation(ComputeTask::Type type, size_t rows) {
        switch (type) {
            case ComputeTask::Type::VECTOR_SIMILARITY:
                // 向量相似度: 优先 GPU (高度并行)
                return rows >= 1000 ? ProcessingUnit::GPU : ProcessingUnit::CPU;

            case ComputeTask::Type::AGGREGATE:
                // 聚合: CPU (vDSP 已优化)
                return ProcessingUnit::CPU;

            case ComputeTask::Type::JOIN:
                // Join: GPU 处理大探测表
                return rows >= 500000 ? ProcessingUnit::GPU : ProcessingUnit::CPU;

            case ComputeTask::Type::FILTER:
            case ComputeTask::Type::SORT:
            default:
                return select_by_size(rows, 1);
        }
    }
};

} // namespace uma
} // namespace thunderduck
```

### 3.3 Metal 着色器 Arrow 适配

```metal
// src/gpu/shaders/arrow_ops.metal

#include <metal_stdlib>
using namespace metal;

// Arrow 列元数据 (在 GPU 端)
struct ArrowColumnMeta {
    uint64_t length;        // 行数
    uint64_t null_count;    // NULL 数量
    uint64_t data_offset;   // 数据偏移
    uint64_t validity_offset; // 位图偏移
};

// 位图检查 (Arrow validity bitmap)
inline bool is_valid(device const uint8_t* validity, uint64_t idx) {
    if (validity == nullptr) return true;  // 无位图 = 全部有效
    uint64_t byte_idx = idx / 8;
    uint8_t bit_idx = idx % 8;
    return (validity[byte_idx] >> bit_idx) & 1;
}

// 向量点积内核 - 直接操作 Arrow 列
kernel void arrow_batch_dot_product(
    device const float* query       [[buffer(0)]],  // D 维查询向量
    device const float* candidates  [[buffer(1)]],  // N×D 候选矩阵 (Arrow 数据)
    device const uint8_t* validity  [[buffer(2)]],  // Arrow validity bitmap
    device float* scores            [[buffer(3)]],  // 输出分数
    constant uint& dim              [[buffer(4)]],
    constant uint& num_candidates   [[buffer(5)]],
    uint idx                        [[thread_position_in_grid]])
{
    if (idx >= num_candidates) return;

    // 检查 NULL
    if (!is_valid(validity, idx)) {
        scores[idx] = NAN;  // NULL 输出
        return;
    }

    // 计算点积
    float sum = 0.0f;
    device const float* cand = candidates + idx * dim;

    for (uint i = 0; i < dim; i++) {
        sum += query[i] * cand[i];
    }

    scores[idx] = sum;
}

// 过滤内核 - 直接操作 Arrow 列
kernel void arrow_filter_gt_i32(
    device const int32_t* data      [[buffer(0)]],  // Arrow 数据列
    device const uint8_t* validity  [[buffer(1)]],  // Arrow validity bitmap
    constant int32_t& threshold     [[buffer(2)]],
    device uint8_t* result_validity [[buffer(3)]],  // 输出 validity
    device atomic_uint* match_count [[buffer(4)]],
    uint idx                        [[thread_position_in_grid]],
    uint total                      [[threads_per_grid]])
{
    if (idx >= total) return;

    bool valid = is_valid(validity, idx);
    bool passes = valid && (data[idx] > threshold);

    // 设置结果位图
    if (passes) {
        uint byte_idx = idx / 8;
        uint8_t bit_mask = 1 << (idx % 8);
        atomic_fetch_or_explicit(
            (device atomic_uint*)(result_validity + byte_idx),
            bit_mask, memory_order_relaxed);
        atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
    }
}

// 聚合内核 - Arrow 列求和
kernel void arrow_sum_f32(
    device const float* data        [[buffer(0)]],  // Arrow 数据列
    device const uint8_t* validity  [[buffer(1)]],  // Arrow validity bitmap
    device atomic_float* partial    [[buffer(2)]],  // 部分和
    device atomic_uint* valid_count [[buffer(3)]],  // 有效行计数
    uint idx                        [[thread_position_in_grid]],
    uint total                      [[threads_per_grid]])
{
    if (idx >= total) return;

    if (is_valid(validity, idx)) {
        float val = data[idx];
        // 使用原子加法累积
        float old = atomic_load_explicit(partial, memory_order_relaxed);
        while (!atomic_compare_exchange_weak_explicit(
            partial, &old, old + val,
            memory_order_relaxed, memory_order_relaxed)) {}

        atomic_fetch_add_explicit(valid_count, 1, memory_order_relaxed);
    }
}
```

### 3.4 CPU-GPU 流水线

```cpp
// src/core/uma_pipeline.cpp

/**
 * 流水线执行引擎
 *
 * 支持 CPU-GPU 重叠执行:
 * 1. CPU 准备下一批数据
 * 2. GPU 计算当前批
 * 3. CPU 消费上一批结果
 */
class UMAPipeline {
public:
    struct Stage {
        arrow::ArrowUMABatch* batch;
        uint64_t task_handle;
        enum State { IDLE, CPU_PREP, GPU_EXEC, CPU_CONSUME } state;
    };

    // 三缓冲流水线
    Stage stages_[3];

    void execute_pipeline(
        std::function<void(arrow::ArrowUMABatch*)> prepare,   // CPU 准备
        std::function<void(arrow::ArrowUMABatch*)> compute,   // GPU 计算
        std::function<void(arrow::ArrowUMABatch*)> consume    // CPU 消费
    ) {
        auto& coordinator = UMACoordinator::instance();

        // 初始化: 准备前两批
        prepare(stages_[0].batch);
        stages_[0].state = Stage::GPU_EXEC;
        stages_[0].task_handle = submit_gpu(stages_[0].batch);

        prepare(stages_[1].batch);
        stages_[1].state = Stage::CPU_PREP;

        // 流水线循环
        while (has_more_data()) {
            // Stage 0: GPU 执行中 → 等待完成 → CPU 消费
            coordinator.wait(stages_[0].task_handle);
            stages_[0].batch->sync_for_cpu();
            consume(stages_[0].batch);
            stages_[0].state = Stage::IDLE;

            // Stage 1: CPU 准备完成 → 提交 GPU
            stages_[1].batch->sync_for_gpu();
            stages_[1].task_handle = submit_gpu(stages_[1].batch);
            stages_[1].state = Stage::GPU_EXEC;

            // Stage 0: IDLE → CPU 准备下一批
            prepare(stages_[0].batch);
            stages_[0].state = Stage::CPU_PREP;

            // 轮转
            std::swap(stages_[0], stages_[1]);
        }
    }

private:
    uint64_t submit_gpu(arrow::ArrowUMABatch* batch);
    bool has_more_data();
};
```

---

## 四、数据流示例

### 4.1 向量相似度搜索

```
输入: 查询向量 Q (256维), 候选表 T (1M×256)

[零拷贝数据流]

1. CPU 分配 Arrow 列 (UMA 共享内存)
   ┌──────────────────────────────┐
   │ ArrowUMAColumn (candidates)  │
   │ - data: float[1M × 256]      │
   │ - validity: uint8[125000]    │
   └──────────────────────────────┘
           │
           │ 零拷贝 (共享地址)
           ▼
2. GPU Metal 着色器直接读取
   ┌──────────────────────────────┐
   │ arrow_batch_dot_product      │
   │ - 1M 线程并行计算            │
   │ - 直接读取 Arrow 数据        │
   └──────────────────────────────┘
           │
           │ 零拷贝 (共享地址)
           ▼
3. CPU 直接消费结果
   ┌──────────────────────────────┐
   │ scores[1M] 已就绪            │
   │ - 无需任何内存拷贝           │
   └──────────────────────────────┘

数据移动量: 0 字节 (vs 传统架构 ~2GB)
```

### 4.2 Join 分阶段处理

```
输入: Build 表 B (1M), Probe 表 P (10M)

[CPU+GPU 协同]

Phase 1: CPU Build 哈希表
   - CPU 读取 Arrow 列 B
   - 构建哈希表 (CPU 优化)
   - 哈希表存入 UMA 共享内存

Phase 2: GPU Probe
   - GPU 读取 Arrow 列 P (零拷贝)
   - GPU 读取哈希表 (零拷贝)
   - 并行探测，写入匹配索引

Phase 3: CPU 物化
   - CPU 读取 GPU 结果 (零拷贝)
   - 构建最终 Arrow 批次

总数据传输: 0 字节
```

---

## 五、实现计划

### 5.1 文件结构

```
include/thunderduck/
├── arrow_uma.h              # Arrow UMA 列/批次
├── uma_coordinator.h        # 处理单元协调器

src/core/
├── arrow_uma.cpp            # Arrow 适配实现
├── uma_coordinator.mm       # 协调器实现 (ObjC++)
├── uma_pipeline.cpp         # 流水线引擎

src/gpu/shaders/
├── arrow_ops.metal          # Arrow 操作着色器

benchmark/
├── test_arrow_uma.cpp       # 基准测试
```

### 5.2 实现顺序

| 阶段 | 任务 | 优先级 |
|------|------|-------|
| 1 | Arrow 列/批次数据结构 | P0 |
| 2 | CPU 路径验证 | P0 |
| 3 | Metal 着色器 (点积、过滤) | P1 |
| 4 | 协调器框架 | P1 |
| 5 | 流水线引擎 | P2 |
| 6 | 基准测试 | P1 |

### 5.3 验证计划

```bash
# 正确性验证
./build/test_arrow_uma --validate

# 性能验证
./build/test_arrow_uma --benchmark

# 预期结果:
# - 向量相似度: GPU 8-15x vs CPU (1M 候选)
# - Filter: GPU 3-5x vs CPU (10M 行)
# - 数据传输开销: < 1μs (零拷贝)
```

---

## 六、与现有组件集成

### 6.1 与 Hash Join v4 集成

```cpp
// 使用 Arrow UMA 列作为 Join 输入
void hash_join_with_arrow(
    arrow::ArrowUMAColumn* build_keys,
    arrow::ArrowUMAColumn* probe_keys,
    JoinResultUMA* result)
{
    // 零拷贝访问 Arrow 数据
    const int32_t* build = build_keys->data<int32_t>();
    const int32_t* probe = probe_keys->data<int32_t>();

    // 使用现有 v4 实现
    hash_join_i32_v4(build, build_keys->length,
                     probe, probe_keys->length,
                     JoinType::INNER, result);
}
```

### 6.2 与 AMX 向量运算集成

```cpp
// AMX 直接操作 Arrow UMA 列
void vector_similarity_amx(
    const float* query,
    arrow::ArrowUMAColumn* candidates,
    float* scores)
{
    // 零拷贝访问
    const float* data = candidates->data<float>();
    size_t dim = /* from schema */;
    size_t num = candidates->length;

    // AMX 批量点积
    vector::batch_dot_product_f32(query, data, dim, num, scores);
}
```

---

## 七、性能预估

| 场景 | 传统架构 | UMA 零拷贝 | 提升 |
|------|---------|-----------|------|
| 1M 向量相似度 | 50ms + 2ms copy | 6ms | ~8x |
| 10M 行过滤 | 20ms + 8ms copy | 5ms | ~5x |
| 1M×10M Join | 1200ms + 10ms copy | 900ms | ~1.3x |

**注**: Join 提升较小因为计算占主导，但消除了内存拷贝的延迟抖动。

---

## 八、限制与风险

### 8.1 内存对齐要求

- 外部 Arrow 内存必须 16KB 页对齐
- 如不满足，需要一次性拷贝对齐

### 8.2 同步点

- CPU → GPU: 需调用 `sync_for_gpu()` 确保可见性
- GPU → CPU: 需等待 command buffer 完成

### 8.3 NPU 限制

- BNNS 主要用于神经网络，通用数据库操作受限
- 当前设计优先 CPU + GPU 协同

---

## 九、总结

本设计通过 UMA + Arrow 零拷贝架构，实现了:

1. **消除数据传输**: CPU/GPU 共享同一物理内存
2. **Arrow 原生支持**: 直接操作 Arrow 列式格式
3. **智能调度**: 协调器根据数据规模选择最优处理单元
4. **流水线执行**: CPU 准备 + GPU 计算 + CPU 消费重叠

预期在向量相似度、过滤等场景实现 5-15x 性能提升。
