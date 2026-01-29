/**
 * ThunderDuck Operator Registry
 *
 * 算子元数据注册系统，支持：
 * 1. 算子注册与发现
 * 2. 成本模型估算
 * 3. DuckDB 优化器集成
 * 4. GPU/NPU/向量算子统一管理
 *
 * @version 2.0
 * @date 2026-01-29
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cstdint>

namespace thunderduck {

// ============================================================================
// 设备类型
// ============================================================================

enum class DeviceType {
    CPU_SIMD,       // CPU with ARM NEON SIMD
    CPU_AMX,        // CPU with Apple AMX (via Accelerate/BLAS)
    GPU_METAL,      // Apple Metal GPU
    GPU_MPS,        // Metal Performance Shaders
    NPU_ANE,        // Apple Neural Engine (via Core ML)
    NPU_BNNS,       // BNNS (CPU Neural Network)
    AUTO            // 自动选择最优设备
};

// ============================================================================
// 数据类型
// ============================================================================

enum class DataType {
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    FLOAT16,        // 半精度浮点
    BFLOAT16,       // Brain Float 16
    FLOAT32,
    FLOAT64,
    STRING,
    DATE,
    TIMESTAMP,
    BOOLEAN,
    VECTOR_F32,     // 向量 (float32 数组)
    VECTOR_F16,     // 向量 (float16 数组)
    TENSOR,         // 张量 (多维数组)
    ANY             // 泛型
};

// ============================================================================
// 算子类型
// ============================================================================

enum class OperatorType {
    // ========== 传统数据库算子 ==========
    // 扫描
    TABLE_SCAN,
    INDEX_SCAN,

    // 过滤
    FILTER,
    BLOOM_FILTER,

    // 连接
    HASH_JOIN,
    MERGE_JOIN,
    NESTED_LOOP_JOIN,
    SEMI_JOIN,
    ANTI_JOIN,

    // 聚合
    HASH_AGGREGATE,
    SORT_AGGREGATE,
    STREAMING_AGGREGATE,

    // 排序
    SORT,
    TOP_N,

    // 其他
    PROJECTION,
    LIMIT,
    UNION,
    DISTINCT,

    // ========== GPU 算子 ==========
    GPU_FILTER,         // GPU 过滤
    GPU_HASH_JOIN,      // GPU Hash Join
    GPU_SEMI_JOIN,      // GPU Semi Join
    GPU_ANTI_JOIN,      // GPU Anti Join
    GPU_AGGREGATE,      // GPU 聚合
    GPU_TOP_K,          // GPU Top-K 选择
    GPU_SORT,           // GPU 排序

    // ========== NPU/ML 算子 ==========
    NPU_INFERENCE,      // NPU 推理
    NPU_EMBEDDING,      // NPU 嵌入变换
    NPU_MATMUL,         // NPU 矩阵乘法
    NPU_NORMALIZE,      // NPU 向量归一化
    NPU_ATTENTION,      // NPU 注意力计算

    // ========== 向量数据库算子 ==========
    VECTOR_SEARCH,      // 向量搜索 (Brute-force)
    VECTOR_ANN,         // 近似最近邻 (HNSW/IVF)
    VECTOR_SIMILARITY,  // 相似度计算
    VECTOR_QUANTIZE,    // 向量量化
    VECTOR_RERANK,      // 重排序
    VECTOR_INDEX_BUILD  // 索引构建
};

// ============================================================================
// 启动代价分解
// ============================================================================

struct StartupCost {
    double initialization = 0.0;    // 初始化开销 (ms) - 内存分配、结构初始化
    double compilation = 0.0;       // 编译开销 (ms) - JIT、shader 编译
    double data_transfer = 0.0;     // 数据传输开销 (ms) - CPU↔GPU/NPU
    double model_load = 0.0;        // 模型加载开销 (ms) - NPU 模型加载
    double warmup = 0.0;            // 预热开销 (ms) - 缓存预热、分支预测

    double total() const {
        return initialization + compilation + data_transfer + model_load + warmup;
    }

    // 判断是否为冷启动场景
    bool is_cold_start_heavy() const {
        return total() > 5.0;  // > 5ms 认为启动代价高
    }
};

// ============================================================================
// 并行度建议
// ============================================================================

struct ParallelismHint {
    // 建议并行度
    uint32_t min_threads = 1;       // 最小线程数
    uint32_t optimal_threads = 4;   // 最优线程数 (基于算子特性)
    uint32_t max_threads = 16;      // 最大有效线程数 (超过后边际效益递减)

    // 并行效率
    double scaling_factor = 1.0;    // 并行扩展因子 (0.0-1.0, 1.0=完美线性扩展)
    double contention_factor = 0.0; // 竞争因子 (0.0=无竞争, 1.0=严重竞争)

    // 批大小建议 (用于 GPU/NPU)
    size_t min_batch = 1;
    size_t optimal_batch = 64;
    size_t max_batch = 1024;

    // 根据可用线程数估算实际并行度
    double effective_parallelism(uint32_t available_threads) const {
        uint32_t threads = std::min(available_threads, max_threads);
        threads = std::max(threads, min_threads);
        // Amdahl's Law 调整
        double parallel_fraction = scaling_factor;
        double serial_fraction = 1.0 - parallel_fraction;
        return 1.0 / (serial_fraction + parallel_fraction / threads);
    }
};

// ============================================================================
// 数据量区间 (直方图桶)
// ============================================================================

struct DataRangeBucket {
    size_t min_rows;                // 最小行数
    size_t max_rows;                // 最大行数
    double cost_per_row;            // 该区间的每行成本 (ns)
    double efficiency;              // 该区间的效率 (0.0-1.0, 1.0=最高效)
    bool is_optimal;                // 是否为最优区间
};

struct DataRangeHistogram {
    // 数据量区间桶
    std::vector<DataRangeBucket> buckets;

    // 预定义区间边界
    static constexpr size_t ROWS_TINY = 1000;           // < 1K
    static constexpr size_t ROWS_SMALL = 10000;         // 1K - 10K
    static constexpr size_t ROWS_MEDIUM = 100000;       // 10K - 100K
    static constexpr size_t ROWS_LARGE = 1000000;       // 100K - 1M
    static constexpr size_t ROWS_XLARGE = 10000000;     // 1M - 10M
    static constexpr size_t ROWS_MASSIVE = 100000000;   // > 10M

    // 初始化默认直方图
    void init_default() {
        buckets = {
            {0, ROWS_TINY, 10.0, 0.3, false},              // 太小，启动开销占主导
            {ROWS_TINY, ROWS_SMALL, 5.0, 0.6, false},      // 小数据
            {ROWS_SMALL, ROWS_MEDIUM, 2.0, 0.9, true},     // 中等数据，通常最优
            {ROWS_MEDIUM, ROWS_LARGE, 1.5, 1.0, true},     // 大数据，高效
            {ROWS_LARGE, ROWS_XLARGE, 1.2, 0.95, true},    // 超大数据
            {ROWS_XLARGE, ROWS_MASSIVE, 1.0, 0.85, false}, // 巨量数据，可能内存受限
            {ROWS_MASSIVE, SIZE_MAX, 1.0, 0.7, false}      // 超出常规处理范围
        };
    }

    // 获取指定行数的桶
    const DataRangeBucket* get_bucket(size_t rows) const {
        for (const auto& bucket : buckets) {
            if (rows >= bucket.min_rows && rows < bucket.max_rows) {
                return &bucket;
            }
        }
        return nullptr;
    }

    // 获取最优区间
    std::pair<size_t, size_t> optimal_range() const {
        size_t min_opt = SIZE_MAX, max_opt = 0;
        for (const auto& bucket : buckets) {
            if (bucket.is_optimal) {
                min_opt = std::min(min_opt, bucket.min_rows);
                max_opt = std::max(max_opt, bucket.max_rows);
            }
        }
        return {min_opt, max_opt};
    }

    // 判断数据量是否在最优区间
    bool is_in_optimal_range(size_t rows) const {
        auto [min_opt, max_opt] = optimal_range();
        return rows >= min_opt && rows < max_opt;
    }
};

// ============================================================================
// 成本模型 (增强版)
// ============================================================================

struct CostModel {
    // ========== 启动代价 ==========
    StartupCost startup;            // 分解的启动代价
    double startup_cost = 0.0;      // 总启动开销 (ms) - 向后兼容

    // ========== 运行代价 ==========
    double per_row_cost = 0.0;      // 每行开销 (ns)
    double memory_factor = 1.0;     // 内存因子 (相对于基准)

    // ========== 并行度 ==========
    ParallelismHint parallelism_hint;  // 并行度建议
    double parallelism = 1.0;          // 实际并行度 - 向后兼容

    // ========== 数据量区间 ==========
    DataRangeHistogram histogram;   // 数据量-效率直方图

    // ========== GPU/NPU 特有成本 ==========
    double transfer_cost = 0.0;     // CPU↔GPU 数据传输开销 (ms/MB)
    double kernel_launch_cost = 0.0;// GPU 内核启动开销 (ms)
    double model_load_cost = 0.0;   // NPU 模型加载开销 (ms)

    // 初始化 (设置向后兼容值)
    void finalize() {
        startup_cost = startup.total();
        parallelism = parallelism_hint.optimal_threads;
        if (histogram.buckets.empty()) {
            histogram.init_default();
        }
    }

    // ========== 成本估算方法 ==========

    // 基础成本估算
    double estimate(size_t rows) const {
        return startup_cost + (per_row_cost * rows / 1e6) / parallelism;
    }

    // 考虑数据量区间的精确估算
    double estimate_with_histogram(size_t rows) const {
        const auto* bucket = histogram.get_bucket(rows);
        double effective_per_row = bucket ? bucket->cost_per_row : per_row_cost;
        double efficiency = bucket ? bucket->efficiency : 1.0;
        return startup.total() + (effective_per_row * rows / 1e6) / (parallelism * efficiency);
    }

    // 考虑系统负载的估算
    double estimate_with_load(size_t rows, double system_load, uint32_t available_threads) const {
        double effective_par = parallelism_hint.effective_parallelism(available_threads);
        // 系统负载影响: 负载越高，性能越差
        double load_penalty = 1.0 + system_load * 0.5;  // 50% 负载时性能降 25%
        return (startup.total() + (per_row_cost * rows / 1e6) / effective_par) * load_penalty;
    }

    // GPU 成本估算 (考虑传输开销)
    double estimate_gpu(size_t rows, size_t bytes_per_row) const {
        double data_mb = (rows * bytes_per_row) / (1024.0 * 1024.0);
        return startup.total() + kernel_launch_cost + transfer_cost * data_mb +
               (per_row_cost * rows / 1e6) / parallelism;
    }

    // NPU 成本估算 (考虑模型加载和批大小)
    double estimate_npu(size_t batch_size, bool model_cached = true) const {
        double load = model_cached ? 0.0 : startup.model_load;
        // 批大小效率调整
        double batch_efficiency = 1.0;
        if (batch_size < parallelism_hint.optimal_batch) {
            batch_efficiency = 0.5 + 0.5 * batch_size / parallelism_hint.optimal_batch;
        }
        return load + startup.initialization + (per_row_cost * batch_size / 1e6) / batch_efficiency;
    }
};

// ============================================================================
// 算子能力描述
// ============================================================================

struct OperatorCapabilities {
    // CPU 能力
    bool supports_simd = false;         // 支持 SIMD
    bool supports_parallel = false;     // 支持并行
    bool supports_streaming = false;    // 支持流式处理
    bool supports_predicate_pushdown = false;  // 支持谓词下推

    // GPU 能力
    bool supports_gpu = false;          // 支持 GPU 加速
    bool supports_uma = false;          // 支持 UMA 零拷贝
    bool requires_metal = false;        // 需要 Metal 支持

    // NPU 能力
    bool supports_npu = false;          // 支持 NPU 加速
    bool supports_ane = false;          // 支持 Apple Neural Engine
    bool supports_fp16 = false;         // 支持 FP16 精度
    bool supports_int8 = false;         // 支持 INT8 量化

    // 向量能力
    bool supports_batch = false;        // 支持批量处理
    bool supports_incremental = false;  // 支持增量更新

    // 约束
    size_t min_rows_for_benefit = 1000; // 最小受益行数
    size_t max_cardinality = SIZE_MAX;  // 最大基数支持
    size_t min_batch_size = 1;          // 最小批大小 (NPU)
    size_t max_batch_size = SIZE_MAX;   // 最大批大小 (NPU)
    size_t max_dimension = SIZE_MAX;    // 最大向量维度
};

// ============================================================================
// 距离度量类型 (向量数据库)
// ============================================================================

enum class DistanceMetric {
    L2,                 // 欧氏距离
    L2_SQUARED,         // 欧氏距离平方
    COSINE,             // 余弦距离
    INNER_PRODUCT,      // 内积 (负数)
    MANHATTAN           // 曼哈顿距离
};

// ============================================================================
// 算子元数据
// ============================================================================

struct OperatorMeta {
    std::string name;                   // 算子名称
    std::string description;            // 描述
    OperatorType type;                  // 算子类型
    DeviceType device = DeviceType::CPU_SIMD;  // 目标设备
    std::vector<DataType> input_types;  // 支持的输入类型
    std::vector<DataType> output_types; // 输出类型
    CostModel cost;                     // 成本模型
    OperatorCapabilities capabilities;  // 能力

    // 向量算子专用参数
    DistanceMetric metric = DistanceMetric::L2;  // 默认距离度量

    // 创建算子实例
    std::function<void*()> create;

    // 执行函数 (通用接口)
    std::function<void(void* op, void* input, void* output, size_t count)> execute;

    // 销毁算子
    std::function<void(void* op)> destroy;

    // 检查设备类型
    bool is_gpu() const {
        return device == DeviceType::GPU_METAL || device == DeviceType::GPU_MPS;
    }

    bool is_npu() const {
        return device == DeviceType::NPU_ANE || device == DeviceType::NPU_BNNS;
    }

    bool is_cpu() const {
        return device == DeviceType::CPU_SIMD || device == DeviceType::CPU_AMX;
    }
};

// ============================================================================
// 算子注册表
// ============================================================================

class OperatorRegistry {
public:
    // 单例
    static OperatorRegistry& instance() {
        static OperatorRegistry reg;
        return reg;
    }

    // 注册算子
    void register_operator(const OperatorMeta& meta) {
        operators_[meta.name] = meta;
        by_type_[meta.type].push_back(meta.name);
    }

    // 查找算子
    const OperatorMeta* find(const std::string& name) const {
        auto it = operators_.find(name);
        return it != operators_.end() ? &it->second : nullptr;
    }

    // 按类型查找
    std::vector<const OperatorMeta*> find_by_type(OperatorType type) const {
        std::vector<const OperatorMeta*> result;
        auto it = by_type_.find(type);
        if (it != by_type_.end()) {
            for (const auto& name : it->second) {
                result.push_back(&operators_.at(name));
            }
        }
        return result;
    }

    // 选择最优算子 (基于成本)
    const OperatorMeta* select_best(
        OperatorType type,
        const std::vector<DataType>& input_types,
        size_t estimated_rows
    ) const {
        auto candidates = find_by_type(type);
        const OperatorMeta* best = nullptr;
        double best_cost = std::numeric_limits<double>::max();

        for (const auto* meta : candidates) {
            // 检查类型兼容
            if (!is_type_compatible(meta->input_types, input_types)) continue;

            // 检查行数要求
            if (estimated_rows < meta->capabilities.min_rows_for_benefit) continue;

            double cost = meta->cost.estimate(estimated_rows);
            if (cost < best_cost) {
                best_cost = cost;
                best = meta;
            }
        }

        return best;
    }

    // 列出所有算子
    std::vector<std::string> list_all() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : operators_) {
            names.push_back(name);
        }
        return names;
    }

    // 打印注册信息
    void print_registry() const;

private:
    OperatorRegistry() = default;

    bool is_type_compatible(
        const std::vector<DataType>& supported,
        const std::vector<DataType>& requested
    ) const {
        for (DataType req : requested) {
            bool found = false;
            for (DataType sup : supported) {
                if (sup == req || sup == DataType::ANY) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    std::unordered_map<std::string, OperatorMeta> operators_;
    std::unordered_map<OperatorType, std::vector<std::string>> by_type_;
};

// ============================================================================
// 系统负载监控
// ============================================================================

struct SystemLoad {
    double cpu_utilization = 0.0;   // CPU 利用率 (0.0-1.0)
    double memory_pressure = 0.0;   // 内存压力 (0.0-1.0)
    double gpu_utilization = 0.0;   // GPU 利用率 (0.0-1.0)
    double npu_utilization = 0.0;   // NPU 利用率 (0.0-1.0)
    uint32_t available_threads = 8; // 可用线程数
    size_t available_memory_mb = 8192;  // 可用内存 (MB)

    // 综合负载评分
    double overall_load() const {
        return 0.4 * cpu_utilization + 0.3 * memory_pressure +
               0.2 * gpu_utilization + 0.1 * npu_utilization;
    }

    // 判断是否为低负载状态
    bool is_low_load() const { return overall_load() < 0.3; }
    bool is_medium_load() const { return overall_load() >= 0.3 && overall_load() < 0.7; }
    bool is_high_load() const { return overall_load() >= 0.7; }
};

// ============================================================================
// 优化器选择策略
// ============================================================================

enum class SelectionStrategy {
    LOWEST_COST,        // 最低成本优先
    LOWEST_LATENCY,     // 最低延迟优先 (考虑启动代价)
    HIGHEST_THROUGHPUT, // 最高吞吐量优先
    MEMORY_EFFICIENT,   // 内存效率优先
    ENERGY_EFFICIENT,   // 能耗效率优先 (优先 NPU/SIMD)
    ADAPTIVE            // 自适应 (根据系统负载动态选择)
};

// ============================================================================
// 优化器选择上下文
// ============================================================================

struct SelectionContext {
    // 查询特征
    size_t estimated_rows = 0;          // 估计行数
    size_t cardinality = 0;             // 基数 (去重后)
    double selectivity = 1.0;           // 选择率
    std::vector<DataType> input_types;  // 输入类型

    // 系统状态
    SystemLoad system_load;             // 当前系统负载

    // 选择策略
    SelectionStrategy strategy = SelectionStrategy::ADAPTIVE;

    // 约束条件
    double max_latency_ms = 0.0;        // 最大延迟约束 (0=无约束)
    size_t max_memory_mb = 0;           // 最大内存约束 (0=无约束)
    bool prefer_gpu = false;            // 优先 GPU
    bool prefer_npu = false;            // 优先 NPU
    bool avoid_cold_start = false;      // 避免冷启动

    // 缓存状态
    bool model_cached = true;           // NPU 模型是否已缓存
    bool data_in_gpu = false;           // 数据是否已在 GPU
};

// ============================================================================
// 优化器选择结果
// ============================================================================

struct SelectionResult {
    const OperatorMeta* selected = nullptr;  // 选中的算子
    double estimated_cost = 0.0;             // 估计成本
    double estimated_latency_ms = 0.0;       // 估计延迟
    uint32_t recommended_threads = 1;        // 建议线程数
    size_t recommended_batch = 1;            // 建议批大小
    std::string selection_reason;            // 选择原因

    bool is_valid() const { return selected != nullptr; }
};

// ============================================================================
// 智能优化器
// ============================================================================

class AdaptiveOptimizer {
public:
    // 单例
    static AdaptiveOptimizer& instance() {
        static AdaptiveOptimizer opt;
        return opt;
    }

    // ========== 主选择接口 ==========

    /**
     * 根据上下文选择最佳算子
     */
    SelectionResult select_best(
        OperatorType type,
        const SelectionContext& ctx
    ) const {
        auto& reg = OperatorRegistry::instance();
        auto candidates = reg.find_by_type(type);

        SelectionResult result;
        double best_score = std::numeric_limits<double>::max();

        for (const auto* meta : candidates) {
            // 检查类型兼容
            if (!is_type_compatible(meta, ctx)) continue;

            // 检查约束条件
            if (!check_constraints(meta, ctx)) continue;

            // 计算综合得分
            double score = calculate_score(meta, ctx);

            if (score < best_score) {
                best_score = score;
                result.selected = meta;
                result.estimated_cost = score;
                result.estimated_latency_ms = estimate_latency(meta, ctx);
                result.recommended_threads = recommend_threads(meta, ctx);
                result.recommended_batch = recommend_batch(meta, ctx);
                result.selection_reason = explain_selection(meta, ctx);
            }
        }

        return result;
    }

    /**
     * 按设备类型选择最佳算子
     */
    SelectionResult select_for_device(
        OperatorType type,
        DeviceType device,
        const SelectionContext& ctx
    ) const {
        SelectionContext device_ctx = ctx;
        switch (device) {
            case DeviceType::GPU_METAL:
            case DeviceType::GPU_MPS:
                device_ctx.prefer_gpu = true;
                break;
            case DeviceType::NPU_ANE:
            case DeviceType::NPU_BNNS:
                device_ctx.prefer_npu = true;
                break;
            default:
                break;
        }
        return select_best(type, device_ctx);
    }

    /**
     * 批量选择 (用于查询计划优化)
     */
    std::vector<SelectionResult> select_pipeline(
        const std::vector<std::pair<OperatorType, size_t>>& pipeline,
        const SelectionContext& base_ctx
    ) const {
        std::vector<SelectionResult> results;
        SelectionContext ctx = base_ctx;

        for (const auto& [type, rows] : pipeline) {
            ctx.estimated_rows = rows;
            results.push_back(select_best(type, ctx));

            // 更新上下文 (流水线传递)
            if (results.back().selected) {
                // 如果选了 GPU 算子，后续数据可能在 GPU
                ctx.data_in_gpu = results.back().selected->is_gpu();
            }
        }

        return results;
    }

    // ========== 系统负载更新 ==========

    void update_system_load(const SystemLoad& load) {
        current_load_ = load;
    }

    const SystemLoad& current_load() const {
        return current_load_;
    }

private:
    AdaptiveOptimizer() = default;

    SystemLoad current_load_;

    // ========== 内部方法 ==========

    bool is_type_compatible(const OperatorMeta* meta, const SelectionContext& ctx) const {
        for (DataType req : ctx.input_types) {
            bool found = false;
            for (DataType sup : meta->input_types) {
                if (sup == req || sup == DataType::ANY) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    bool check_constraints(const OperatorMeta* meta, const SelectionContext& ctx) const {
        // 行数约束
        if (ctx.estimated_rows < meta->capabilities.min_rows_for_benefit) {
            // 允许小数据量使用，但会影响得分
        }

        // 基数约束
        if (ctx.cardinality > meta->capabilities.max_cardinality) {
            return false;
        }

        // 设备偏好约束
        if (ctx.prefer_gpu && !meta->is_gpu()) {
            // 不强制排除，但会影响得分
        }
        if (ctx.prefer_npu && !meta->is_npu()) {
            // 不强制排除，但会影响得分
        }

        // 冷启动约束
        if (ctx.avoid_cold_start && meta->cost.startup.is_cold_start_heavy()) {
            return false;
        }

        // 延迟约束
        if (ctx.max_latency_ms > 0) {
            double est = estimate_latency(meta, ctx);
            if (est > ctx.max_latency_ms) {
                return false;
            }
        }

        return true;
    }

    double calculate_score(const OperatorMeta* meta, const SelectionContext& ctx) const {
        double score = 0.0;

        switch (ctx.strategy) {
            case SelectionStrategy::LOWEST_COST:
                score = meta->cost.estimate_with_histogram(ctx.estimated_rows);
                break;

            case SelectionStrategy::LOWEST_LATENCY:
                score = meta->cost.startup.total() +
                        meta->cost.per_row_cost * ctx.estimated_rows / 1e6;
                break;

            case SelectionStrategy::HIGHEST_THROUGHPUT:
                // 吞吐量 = 行数 / 时间，取倒数作为分数
                score = meta->cost.estimate_with_load(
                    ctx.estimated_rows,
                    ctx.system_load.overall_load(),
                    ctx.system_load.available_threads
                );
                break;

            case SelectionStrategy::MEMORY_EFFICIENT:
                score = meta->cost.estimate(ctx.estimated_rows) *
                        meta->cost.memory_factor;
                break;

            case SelectionStrategy::ENERGY_EFFICIENT:
                score = meta->cost.estimate(ctx.estimated_rows);
                // NPU 最节能，其次是 SIMD，GPU 最耗能
                if (meta->is_npu()) score *= 0.7;
                else if (meta->is_cpu()) score *= 0.9;
                else if (meta->is_gpu()) score *= 1.2;
                break;

            case SelectionStrategy::ADAPTIVE:
            default:
                score = calculate_adaptive_score(meta, ctx);
                break;
        }

        // 应用设备偏好调整
        if (ctx.prefer_gpu && meta->is_gpu()) score *= 0.8;
        if (ctx.prefer_npu && meta->is_npu()) score *= 0.8;
        if (ctx.data_in_gpu && meta->is_gpu()) score *= 0.7;  // 数据已在 GPU，避免传输

        return score;
    }

    double calculate_adaptive_score(const OperatorMeta* meta, const SelectionContext& ctx) const {
        double base_score = meta->cost.estimate_with_load(
            ctx.estimated_rows,
            ctx.system_load.overall_load(),
            ctx.system_load.available_threads
        );

        // 根据系统负载动态调整
        if (ctx.system_load.is_high_load()) {
            // 高负载: 优先选择低资源消耗的算子
            if (meta->is_gpu() && ctx.system_load.gpu_utilization > 0.8) {
                base_score *= 1.5;  // 惩罚 GPU 算子
            }
            if (meta->cost.memory_factor > 1.0) {
                base_score *= meta->cost.memory_factor;  // 惩罚高内存算子
            }
        } else if (ctx.system_load.is_low_load()) {
            // 低负载: 可以使用更激进的算子
            if (meta->is_gpu()) {
                base_score *= 0.9;  // 奖励 GPU 算子
            }
        }

        // 根据数据量区间调整
        if (!meta->cost.histogram.is_in_optimal_range(ctx.estimated_rows)) {
            base_score *= 1.2;  // 非最优区间惩罚
        }

        return base_score;
    }

    double estimate_latency(const OperatorMeta* meta, const SelectionContext& ctx) const {
        return meta->cost.estimate_with_load(
            ctx.estimated_rows,
            ctx.system_load.overall_load(),
            ctx.system_load.available_threads
        );
    }

    uint32_t recommend_threads(const OperatorMeta* meta, const SelectionContext& ctx) const {
        uint32_t available = ctx.system_load.available_threads;
        const auto& hint = meta->cost.parallelism_hint;

        // 考虑系统负载
        if (ctx.system_load.is_high_load()) {
            return std::min(available, hint.min_threads);
        } else if (ctx.system_load.is_low_load()) {
            return std::min(available, hint.max_threads);
        } else {
            return std::min(available, hint.optimal_threads);
        }
    }

    size_t recommend_batch(const OperatorMeta* meta, const SelectionContext& ctx) const {
        const auto& hint = meta->cost.parallelism_hint;

        // NPU 批大小建议
        if (meta->is_npu()) {
            if (ctx.estimated_rows < hint.min_batch) {
                return hint.min_batch;
            } else if (ctx.estimated_rows > hint.max_batch) {
                return hint.optimal_batch;
            }
            return std::min(ctx.estimated_rows, hint.optimal_batch);
        }

        return 1;  // 非 NPU 算子不需要批大小
    }

    std::string explain_selection(const OperatorMeta* meta, const SelectionContext& ctx) const {
        std::string reason = meta->name + ": ";

        if (ctx.strategy == SelectionStrategy::ADAPTIVE) {
            if (ctx.system_load.is_high_load()) {
                reason += "high load, ";
            } else if (ctx.system_load.is_low_load()) {
                reason += "low load, ";
            }
        }

        if (meta->is_gpu()) reason += "GPU accelerated, ";
        if (meta->is_npu()) reason += "NPU accelerated, ";
        if (meta->capabilities.supports_simd) reason += "SIMD optimized, ";

        if (meta->cost.histogram.is_in_optimal_range(ctx.estimated_rows)) {
            reason += "optimal data range";
        } else {
            reason += "suboptimal data range";
        }

        return reason;
    }
};

// ============================================================================
// 算子注册宏
// ============================================================================

#define REGISTER_OPERATOR(meta) \
    static bool _reg_##__LINE__ = []() { \
        OperatorRegistry::instance().register_operator(meta); \
        return true; \
    }()

// ============================================================================
// 预定义算子注册
// ============================================================================

namespace operators {

// 注册所有 ThunderDuck 算子
void register_all_operators();

// ========== CPU 算子 (SIMD) ==========
void register_filter_operators();
void register_join_operators();
void register_aggregate_operators();
void register_sort_operators();

// ========== GPU 算子 (Metal/MPS) ==========
void register_gpu_operators();
void register_gpu_join_operators();
void register_gpu_aggregate_operators();
void register_gpu_filter_operators();
void register_gpu_topk_operators();

// ========== NPU 算子 (ANE/BNNS/CoreML/MPS) ==========
void register_npu_operators();
void register_npu_inference_operators();
void register_npu_matmul_operators();
void register_npu_embedding_operators();

// ========== 向量数据库算子 ==========
void register_vector_operators();
void register_vector_search_operators();
void register_vector_index_operators();
void register_vector_similarity_operators();
void register_vector_quantizer_operators();
void register_vector_reranker_operators();

} // namespace operators

} // namespace thunderduck
