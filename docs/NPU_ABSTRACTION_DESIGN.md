# ThunderDuck NPU 抽象接口设计

> 版本: V20 | 日期: 2026-01-27 | 目标: 向量数据库 ML 增强

## 一、设计背景与目标

### 1.1 向量数据库的 ML 增强需求

| 功能 | 描述 | NPU 适用性 |
|------|------|-----------|
| **学习型相似度** | 使用神经网络计算语义相似度 | ✅ 高度适合 |
| **嵌入变换** | 通过 MLP 调整向量表示 | ✅ 高度适合 |
| **重排序模型** | Cross-encoder 精确排序 | ✅ 高度适合 |
| **量化编解码** | 神经网络压缩/解压 | ✅ 适合 |
| **多模态融合** | 文本+图像联合嵌入 | ✅ 高度适合 |
| **原始向量运算** | 余弦/欧氏距离 | ❌ 用 AMX/GPU |

### 1.2 设计目标

1. **统一硬件抽象**: NPU/GPU/CPU 统一接口
2. **ML 推理优先**: 专注 ANE 擅长的推理任务
3. **向量数据库适配**: 支持批量处理和流水线
4. **跨平台可移植**: M4 → A17 → 未来芯片
5. **零开销抽象**: 内联优化，无运行时多态

### 1.3 Apple 硬件加速器特点

| 加速器 | 适用场景 | 峰值性能 | 数据类型 |
|--------|----------|----------|----------|
| **ANE** | ML 推理 | 38 TOPS | FP16 |
| **AMX** | 矩阵运算 | ~2 TFLOPS | FP32/FP64 |
| **GPU** | 大规模并行 | ~10 TFLOPS | FP16/FP32 |
| **NEON** | SIMD 向量 | ~200 GFLOPS | 全类型 |

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ThunderDuck Vector DB                    │
├─────────────────────────────────────────────────────────────┤
│                   Vector Operations Layer                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
│  │ Similarity│  │ Transform │  │ Reranker  │  │ Quantize │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────┬─────┘ │
├────────┼──────────────┼──────────────┼─────────────┼────────┤
│        │              │              │             │        │
│  ┌─────▼──────────────▼──────────────▼─────────────▼─────┐  │
│  │              Hardware Abstraction Layer               │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │              InferenceEngine (统一接口)          │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │        │              │              │                │  │
│  │  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐         │  │
│  │  │NPUBackend │  │GPUBackend │  │CPUBackend │         │  │
│  │  │(Core ML)  │  │(Metal/MPS)│  │(AMX/SIMD) │         │  │
│  │  └───────────┘  └───────────┘  └───────────┘         │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Memory Management (UMA)                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

```cpp
namespace thunderduck {
namespace accelerator {

// 1. 设备类型
enum class DeviceType {
    AUTO,       // 自动选择
    NPU,        // Apple Neural Engine
    GPU,        // Metal GPU
    CPU_AMX,    // CPU with AMX (BLAS)
    CPU_SIMD    // CPU with NEON SIMD
};

// 2. 数据类型
enum class DataType {
    FP32,       // 32-bit float
    FP16,       // 16-bit float (ANE native)
    INT8,       // 8-bit quantized
    INT4        // 4-bit quantized
};

// 3. 张量描述
struct TensorDesc {
    DataType dtype;
    std::vector<size_t> shape;
    size_t size_bytes() const;
};

}}
```

## 三、NPU 抽象接口设计

### 3.1 模型管理接口

```cpp
namespace thunderduck {
namespace npu {

/**
 * ML 模型句柄
 */
class Model {
public:
    using Ptr = std::shared_ptr<Model>;

    virtual ~Model() = default;

    // 模型信息
    virtual const char* name() const = 0;
    virtual std::vector<TensorDesc> input_specs() const = 0;
    virtual std::vector<TensorDesc> output_specs() const = 0;

    // 设备能力
    virtual bool supports_device(DeviceType device) const = 0;
    virtual DeviceType preferred_device() const = 0;
};

/**
 * 模型加载器
 */
class ModelLoader {
public:
    static ModelLoader& instance();

    /**
     * 从 Core ML 模型文件加载
     * @param path .mlmodelc 或 .mlpackage 路径
     * @param options 加载选项
     */
    Model::Ptr load_coreml(const char* path,
                           const ModelLoadOptions& options = {});

    /**
     * 从内存加载预编译模型
     */
    Model::Ptr load_from_buffer(const void* data, size_t size,
                                 ModelFormat format);

    /**
     * 预加载模型（后台加载，提前编译）
     */
    std::future<Model::Ptr> preload_async(const char* path);

    /**
     * 获取已缓存的模型
     */
    Model::Ptr get_cached(const char* name);

    /**
     * 清理模型缓存
     */
    void clear_cache();
};

/**
 * 模型加载选项
 */
struct ModelLoadOptions {
    DeviceType preferred_device = DeviceType::AUTO;
    bool allow_low_precision = true;   // 允许 FP16 降精度
    bool compile_for_ane = true;       // 为 ANE 编译优化
    size_t max_batch_size = 64;        // 最大批处理大小
};

}}
```

### 3.2 推理执行接口

```cpp
namespace thunderduck {
namespace npu {

/**
 * 推理会话 - 执行模型推理
 */
class InferenceSession {
public:
    using Ptr = std::shared_ptr<InferenceSession>;

    /**
     * 创建推理会话
     * @param model 已加载的模型
     * @param config 会话配置
     */
    static Ptr create(Model::Ptr model,
                      const SessionConfig& config = {});

    // ===== 同步推理 =====

    /**
     * 单次推理
     * @param inputs 输入张量数组
     * @param outputs 输出张量数组（预分配）
     * @return 成功返回 true
     */
    virtual bool run(const Tensor* inputs, size_t num_inputs,
                     Tensor* outputs, size_t num_outputs) = 0;

    /**
     * 批量推理
     * @param batch_inputs 批量输入 [batch_size x num_inputs]
     * @param batch_outputs 批量输出
     * @param batch_size 批次大小
     */
    virtual bool run_batch(const Tensor* const* batch_inputs,
                           Tensor** batch_outputs,
                           size_t batch_size) = 0;

    // ===== 异步推理 =====

    /**
     * 异步提交推理请求
     */
    virtual InferenceRequest submit_async(const Tensor* inputs,
                                          size_t num_inputs) = 0;

    /**
     * 等待异步请求完成
     */
    virtual bool wait(InferenceRequest& request,
                      Tensor* outputs, size_t num_outputs,
                      uint64_t timeout_ns = UINT64_MAX) = 0;

    // ===== 流式推理 (向量数据库优化) =====

    /**
     * 开始流式推理
     * 适用于向量数据库的连续查询场景
     */
    virtual StreamHandle begin_stream(size_t max_queue_depth = 16) = 0;

    /**
     * 提交到流
     */
    virtual void stream_submit(StreamHandle stream,
                               const Tensor* inputs, size_t num_inputs,
                               void* user_data = nullptr) = 0;

    /**
     * 从流获取结果
     */
    virtual bool stream_get_result(StreamHandle stream,
                                   Tensor* outputs, size_t num_outputs,
                                   void** user_data = nullptr) = 0;

    /**
     * 结束流式推理
     */
    virtual void end_stream(StreamHandle stream) = 0;

    // ===== 设备信息 =====

    virtual DeviceType active_device() const = 0;
    virtual const char* device_name() const = 0;
};

/**
 * 会话配置
 */
struct SessionConfig {
    DeviceType device = DeviceType::AUTO;
    size_t batch_size = 1;
    bool enable_profiling = false;
    bool use_shared_memory = true;  // UMA 共享内存
};

}}
```

### 3.3 张量接口

```cpp
namespace thunderduck {
namespace npu {

/**
 * 张量 - 统一的多维数组表示
 */
class Tensor {
public:
    // 构造
    Tensor();
    Tensor(const TensorDesc& desc);
    Tensor(DataType dtype, std::initializer_list<size_t> shape);

    // 从现有内存创建（零拷贝）
    static Tensor wrap(void* data, const TensorDesc& desc);
    static Tensor wrap_uma(uma::UMABuffer& buffer, const TensorDesc& desc);

    // 访问
    void* data();
    const void* data() const;
    const TensorDesc& desc() const;

    size_t ndim() const;
    size_t shape(size_t dim) const;
    size_t numel() const;  // 元素总数
    size_t size_bytes() const;

    // 类型转换
    Tensor to(DataType dtype) const;
    Tensor to_device(DeviceType device) const;

    // UMA 支持
    bool is_uma() const;
    void* metal_buffer() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}}
```

### 3.4 向量数据库专用接口

```cpp
namespace thunderduck {
namespace vector_db {

/**
 * 向量相似度计算器 - 支持学习型相似度
 */
class SimilarityComputer {
public:
    using Ptr = std::shared_ptr<SimilarityComputer>;

    /**
     * 创建相似度计算器
     * @param type 相似度类型
     * @param model 可选的学习模型（用于学习型相似度）
     */
    static Ptr create(SimilarityType type,
                      npu::Model::Ptr model = nullptr);

    /**
     * 批量计算相似度
     * @param query 查询向量 [dim]
     * @param candidates 候选向量 [num_candidates x dim]
     * @param dim 向量维度
     * @param num_candidates 候选数量
     * @param out_scores 输出相似度分数
     */
    virtual void compute_batch(
        const float* query,
        const float* candidates,
        size_t dim,
        size_t num_candidates,
        float* out_scores) = 0;

    /**
     * 多查询批量计算
     */
    virtual void compute_multi_query(
        const float* queries,      // [num_queries x dim]
        const float* candidates,   // [num_candidates x dim]
        size_t dim,
        size_t num_queries,
        size_t num_candidates,
        float* out_scores) = 0;    // [num_queries x num_candidates]
};

enum class SimilarityType {
    COSINE,         // 余弦相似度 (AMX/GPU)
    EUCLIDEAN,      // 欧氏距离 (AMX/GPU)
    DOT_PRODUCT,    // 点积 (AMX/GPU)
    LEARNED,        // 学习型相似度 (NPU)
    CROSS_ENCODER   // Cross-encoder (NPU)
};

/**
 * 嵌入变换器 - 通过神经网络变换向量
 */
class EmbeddingTransformer {
public:
    using Ptr = std::shared_ptr<EmbeddingTransformer>;

    /**
     * 创建嵌入变换器
     * @param model 变换模型 (MLP, Transformer 等)
     */
    static Ptr create(npu::Model::Ptr model);

    /**
     * 变换嵌入向量
     * @param input 输入向量 [batch_size x input_dim]
     * @param batch_size 批次大小
     * @param output 输出向量 [batch_size x output_dim]
     */
    virtual void transform(const float* input, size_t batch_size,
                           float* output) = 0;

    /**
     * 获取输出维度
     */
    virtual size_t output_dim() const = 0;
};

/**
 * 重排序器 - 使用 Cross-encoder 精确排序
 */
class Reranker {
public:
    using Ptr = std::shared_ptr<Reranker>;

    /**
     * 创建重排序器
     * @param model Cross-encoder 模型
     */
    static Ptr create(npu::Model::Ptr model);

    /**
     * 重排序候选结果
     * @param query 查询（可能是文本嵌入）
     * @param candidates 候选结果
     * @param num_candidates 候选数量
     * @param out_scores 重排序分数
     * @param out_indices 排序后的索引
     */
    virtual void rerank(const float* query, size_t query_dim,
                        const float* candidates, size_t candidate_dim,
                        size_t num_candidates,
                        float* out_scores,
                        uint32_t* out_indices) = 0;
};

/**
 * 神经量化编解码器
 */
class NeuralQuantizer {
public:
    using Ptr = std::shared_ptr<NeuralQuantizer>;

    /**
     * 创建量化器
     * @param encoder 编码器模型
     * @param decoder 解码器模型
     */
    static Ptr create(npu::Model::Ptr encoder,
                      npu::Model::Ptr decoder);

    /**
     * 量化向量
     * @param vectors 原始向量 [num_vectors x dim]
     * @param num_vectors 向量数量
     * @param codes 量化码 [num_vectors x code_size]
     */
    virtual void encode(const float* vectors, size_t num_vectors,
                        uint8_t* codes) = 0;

    /**
     * 反量化
     */
    virtual void decode(const uint8_t* codes, size_t num_vectors,
                        float* vectors) = 0;

    /**
     * 获取压缩比
     */
    virtual float compression_ratio() const = 0;
};

}}
```

## 四、设备调度策略

### 4.1 自动设备选择

```cpp
namespace thunderduck {
namespace accelerator {

/**
 * 设备选择器 - 自动选择最优设备
 */
class DeviceSelector {
public:
    static DeviceSelector& instance();

    /**
     * 为操作选择最优设备
     */
    DeviceType select_device(OperationType op,
                             size_t data_size,
                             DataType dtype);

    /**
     * 获取设备能力
     */
    DeviceCapabilities get_capabilities(DeviceType device);

    /**
     * 设置设备偏好
     */
    void set_preference(DeviceType device, int priority);

    /**
     * 禁用特定设备
     */
    void disable_device(DeviceType device);
};

/**
 * 操作类型 - 用于设备选择决策
 */
enum class OperationType {
    // 原始向量运算 - 优先 AMX/GPU
    VECTOR_SIMILARITY,
    VECTOR_DISTANCE,
    VECTOR_NORMALIZE,

    // ML 推理 - 优先 NPU
    ML_INFERENCE,
    EMBEDDING_TRANSFORM,
    RERANKING,
    QUANTIZATION,

    // 大规模并行 - 优先 GPU
    BATCH_SEARCH,
    INDEX_BUILD,
    CLUSTERING
};

/**
 * 设备选择策略表
 */
struct DeviceSelectionPolicy {
    // 操作 → 设备优先级列表
    std::unordered_map<OperationType, std::vector<DeviceType>> priorities;

    // 数据量阈值
    size_t npu_min_batch = 16;      // NPU 最小批次
    size_t gpu_min_elements = 10000; // GPU 最小元素数

    // 默认策略
    static DeviceSelectionPolicy default_policy();
};

}}
```

### 4.2 设备调度决策矩阵

```
设备选择决策矩阵:

┌────────────────────────┬─────────────────────────────────────────────┐
│     Operation          │              Device Priority                │
│                        ├─────────┬─────────┬─────────┬──────────────┤
│                        │ Small   │ Medium  │ Large   │ ML Inference │
│                        │ (<1K)   │ (1K-1M) │ (>1M)   │              │
├────────────────────────┼─────────┼─────────┼─────────┼──────────────┤
│ Cosine Similarity      │ SIMD    │ AMX     │ GPU     │ -            │
│ Euclidean Distance     │ SIMD    │ AMX     │ GPU     │ -            │
│ Dot Product            │ SIMD    │ AMX     │ GPU     │ -            │
├────────────────────────┼─────────┼─────────┼─────────┼──────────────┤
│ Learned Similarity     │ CPU*    │ NPU     │ NPU     │ ✓            │
│ Embedding Transform    │ CPU*    │ NPU     │ NPU     │ ✓            │
│ Cross-encoder Rerank   │ NPU     │ NPU     │ NPU     │ ✓            │
│ Neural Quantization    │ NPU     │ NPU     │ NPU     │ ✓            │
├────────────────────────┼─────────┼─────────┼─────────┼──────────────┤
│ KNN Search             │ SIMD    │ GPU     │ GPU     │ -            │
│ Index Building         │ CPU     │ GPU     │ GPU     │ -            │
└────────────────────────┴─────────┴─────────┴─────────┴──────────────┘

* CPU fallback when batch size < NPU minimum
```

## 五、实现方案

### 5.1 Core ML 后端

```cpp
namespace thunderduck {
namespace npu {
namespace coreml {

/**
 * Core ML 模型实现
 */
class CoreMLModel : public Model {
public:
    CoreMLModel(const char* path, const ModelLoadOptions& options);
    ~CoreMLModel();

    // Model 接口实现
    const char* name() const override;
    std::vector<TensorDesc> input_specs() const override;
    std::vector<TensorDesc> output_specs() const override;
    bool supports_device(DeviceType device) const override;
    DeviceType preferred_device() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Core ML 推理会话
 */
class CoreMLSession : public InferenceSession {
public:
    CoreMLSession(Model::Ptr model, const SessionConfig& config);
    ~CoreMLSession();

    bool run(const Tensor* inputs, size_t num_inputs,
             Tensor* outputs, size_t num_outputs) override;

    bool run_batch(const Tensor* const* batch_inputs,
                   Tensor** batch_outputs,
                   size_t batch_size) override;

    // ... 其他接口实现

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}}}
```

### 5.2 Metal Performance Shaders 后端

```cpp
namespace thunderduck {
namespace npu {
namespace mps {

/**
 * MPS 神经网络推理
 * 用于 GPU 上的 ML 推理（作为 NPU 的备选）
 */
class MPSNeuralNetwork {
public:
    MPSNeuralNetwork(const char* model_path);

    void forward(const Tensor& input, Tensor& output);
    void forward_batch(const Tensor* inputs, Tensor* outputs, size_t batch_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}}}
```

### 5.3 BNNS 后端 (轻量级操作)

```cpp
namespace thunderduck {
namespace npu {
namespace bnns {

/**
 * BNNS 加速的基础操作
 * 用于不需要完整 Core ML 的轻量级计算
 */
class BNNSAccelerator {
public:
    static BNNSAccelerator& instance();

    // 向量归一化 (利用 vDSP)
    void normalize_f32(const float* input, float* output, size_t dim);
    void normalize_batch_f32(const float* input, float* output,
                             size_t batch_size, size_t dim);

    // 矩阵乘法 (利用 BLAS/AMX)
    void matmul_f32(const float* A, const float* B, float* C,
                    size_t M, size_t K, size_t N);

    // 全连接层 (BNNS 原生)
    void dense_layer_f32(const float* input, const float* weights,
                         const float* bias, float* output,
                         size_t batch_size, size_t input_dim, size_t output_dim,
                         ActivationType activation);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}}}
```

## 六、使用示例

### 6.1 学习型相似度计算

```cpp
#include <thunderduck/npu/model.h>
#include <thunderduck/vector_db/similarity.h>

// 加载学习型相似度模型
auto model = npu::ModelLoader::instance().load_coreml(
    "models/learned_similarity.mlmodelc",
    {.preferred_device = DeviceType::NPU}
);

// 创建相似度计算器
auto sim = vector_db::SimilarityComputer::create(
    vector_db::SimilarityType::LEARNED, model
);

// 批量计算相似度
std::vector<float> query(128);        // 128 维查询向量
std::vector<float> candidates(1000 * 128);  // 1000 个候选
std::vector<float> scores(1000);

sim->compute_batch(query.data(), candidates.data(),
                   128, 1000, scores.data());
```

### 6.2 Cross-encoder 重排序

```cpp
#include <thunderduck/npu/model.h>
#include <thunderduck/vector_db/reranker.h>

// 加载 Cross-encoder 模型
auto model = npu::ModelLoader::instance().load_coreml(
    "models/cross_encoder.mlmodelc"
);

// 创建重排序器
auto reranker = vector_db::Reranker::create(model);

// 重排序 Top-100 候选
std::vector<float> query(768);           // 查询嵌入
std::vector<float> candidates(100 * 768); // 100 个候选
std::vector<float> scores(100);
std::vector<uint32_t> indices(100);

reranker->rerank(query.data(), 768,
                 candidates.data(), 768,
                 100, scores.data(), indices.data());

// indices 现在包含按相关性排序的候选索引
```

### 6.3 流式推理 (高吞吐场景)

```cpp
#include <thunderduck/npu/inference.h>

auto model = npu::ModelLoader::instance().load_coreml("models/encoder.mlmodelc");
auto session = npu::InferenceSession::create(model);

// 开始流式推理
auto stream = session->begin_stream(/*queue_depth=*/32);

// 持续提交查询
for (size_t i = 0; i < num_queries; ++i) {
    npu::Tensor input = prepare_query(i);
    session->stream_submit(stream, &input, 1, (void*)i);
}

// 获取结果
for (size_t i = 0; i < num_queries; ++i) {
    npu::Tensor output;
    void* user_data;
    session->stream_get_result(stream, &output, 1, &user_data);
    process_result(output, (size_t)user_data);
}

session->end_stream(stream);
```

## 七、性能预期

### 7.1 ANE 推理性能预估

| 模型类型 | 批次大小 | 延迟 | 吞吐量 |
|----------|----------|------|--------|
| MLP (768→256) | 1 | ~0.1ms | 10K/s |
| MLP (768→256) | 64 | ~0.5ms | 128K/s |
| Cross-encoder | 1 | ~5ms | 200/s |
| Cross-encoder | 32 | ~20ms | 1.6K/s |

### 7.2 对比传统方法

| 操作 | CPU SIMD | GPU Metal | NPU (ANE) |
|------|----------|-----------|-----------|
| 余弦相似度 1M | 0.5ms | 0.3ms | N/A |
| 学习相似度 1K | 50ms | 5ms | **0.5ms** |
| Cross-encoder 100 | 500ms | 50ms | **20ms** |

## 八、实现计划

### Phase 1: 基础框架 (V20.1)

| 任务 | 工作量 |
|------|--------|
| 设计张量和模型接口 | 1 天 |
| 实现 Core ML 模型加载 | 2 天 |
| 实现基础推理会话 | 2 天 |
| 单元测试 | 1 天 |

### Phase 2: 向量数据库集成 (V20.2)

| 任务 | 工作量 |
|------|--------|
| 实现 SimilarityComputer | 2 天 |
| 实现 EmbeddingTransformer | 1 天 |
| 实现 Reranker | 2 天 |
| 集成测试 | 1 天 |

### Phase 3: 优化与生产化 (V20.3)

| 任务 | 工作量 |
|------|--------|
| 流式推理优化 | 2 天 |
| 模型缓存与预热 | 1 天 |
| 基准测试与调优 | 2 天 |
| 文档完善 | 1 天 |

## 九、文件结构

```
include/thunderduck/
├── npu/
│   ├── device.h         // 设备检测与选择
│   ├── tensor.h         // 张量类型
│   ├── model.h          // 模型加载接口
│   ├── inference.h      // 推理执行接口
│   └── backend/
│       ├── coreml.h     // Core ML 后端
│       ├── mps.h        // MPS 后端
│       └── bnns.h       // BNNS 后端
└── vector_db/
    ├── similarity.h     // 相似度计算
    ├── transform.h      // 嵌入变换
    ├── reranker.h       // 重排序
    └── quantizer.h      // 神经量化

src/npu/
├── device.cpp
├── tensor.cpp
├── model_loader.cpp
├── inference_session.cpp
└── backend/
    ├── coreml_model.mm      // Objective-C++ 实现
    ├── coreml_session.mm
    ├── mps_network.mm
    └── bnns_ops.cpp

src/vector_db/
├── similarity_computer.cpp
├── embedding_transformer.cpp
├── reranker.cpp
└── neural_quantizer.cpp
```

## 十、结论

本设计将 NPU 抽象聚焦于 **ML 推理任务**，这是 Apple Neural Engine 真正擅长的领域。对于向量数据库：

1. **原始向量运算** → 继续使用 AMX/GPU (已有优秀实现)
2. **ML 增强功能** → 使用新的 NPU 抽象接口

这种分层设计既发挥了各硬件的优势，又为未来的向量数据库 ML 功能提供了扩展基础。
