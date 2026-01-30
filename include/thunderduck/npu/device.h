/**
 * ThunderDuck NPU Abstraction Layer - Device Interface
 *
 * @file device.h
 * @version V20
 * @date 2026-01-27
 *
 * 设备检测、能力查询和自动选择接口
 */

#ifndef THUNDERDUCK_NPU_DEVICE_H
#define THUNDERDUCK_NPU_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

namespace thunderduck {
namespace accelerator {

//=============================================================================
// 设备类型枚举
//=============================================================================

/**
 * 硬件加速器类型
 */
enum class DeviceType {
    AUTO,       ///< 自动选择最优设备
    NPU,        ///< Apple Neural Engine (ANE)
    GPU,        ///< Metal GPU
    CPU_AMX,    ///< CPU with AMX (通过 Accelerate/BLAS)
    CPU_SIMD    ///< CPU with NEON SIMD
};

/**
 * 数据类型
 */
enum class DataType {
    FP32,       ///< 32-bit float
    FP16,       ///< 16-bit float (ANE native)
    BF16,       ///< bfloat16
    INT8,       ///< 8-bit quantized
    INT4,       ///< 4-bit quantized
    UINT8,      ///< 8-bit unsigned
    INT32,      ///< 32-bit integer
    INT64       ///< 64-bit integer
};

/**
 * 操作类型 - 用于设备选择决策
 */
enum class OperationType {
    // 原始向量运算 - 优先 AMX/GPU
    VECTOR_SIMILARITY,
    VECTOR_DISTANCE,
    VECTOR_NORMALIZE,
    MATRIX_MULTIPLY,

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

//=============================================================================
// 设备能力描述
//=============================================================================

/**
 * 设备能力信息
 */
struct DeviceCapabilities {
    DeviceType type;
    std::string name;               ///< 设备名称 (e.g., "Apple M4 Max GPU")

    // 计算能力
    uint64_t peak_flops_fp32;       ///< FP32 峰值 FLOPS
    uint64_t peak_flops_fp16;       ///< FP16 峰值 FLOPS
    uint64_t peak_tops_int8;        ///< INT8 峰值 TOPS

    // 内存
    uint64_t memory_size;           ///< 可用内存 (字节)
    uint64_t memory_bandwidth;      ///< 内存带宽 (字节/秒)
    bool supports_uma;              ///< 支持统一内存

    // 功能
    bool supports_fp16;
    bool supports_int8;
    bool supports_int4;
    bool supports_async;            ///< 支持异步执行
    bool supports_batch;            ///< 支持批处理

    // ANE 特有
    uint32_t ane_cores;             ///< ANE 核心数 (仅 NPU)

    // GPU 特有
    uint32_t gpu_cores;             ///< GPU 核心数
    uint32_t max_threadgroup_size;  ///< 最大线程组大小
};

//=============================================================================
// 设备选择策略
//=============================================================================

/**
 * 设备选择策略配置
 */
struct DeviceSelectionPolicy {
    /// 操作 → 设备优先级列表
    std::unordered_map<OperationType, std::vector<DeviceType>> priorities;

    /// 数据量阈值
    size_t npu_min_batch = 16;          ///< NPU 最小批次大小
    size_t gpu_min_elements = 10000;    ///< GPU 最小元素数
    size_t amx_min_elements = 256;      ///< AMX 最小元素数

    /// 延迟敏感模式 (优先低延迟设备)
    bool latency_sensitive = false;

    /**
     * 获取默认策略
     */
    static DeviceSelectionPolicy default_policy() {
        DeviceSelectionPolicy policy;

        // 向量运算: SIMD < AMX < GPU
        policy.priorities[OperationType::VECTOR_SIMILARITY] =
            {DeviceType::CPU_SIMD, DeviceType::CPU_AMX, DeviceType::GPU};
        policy.priorities[OperationType::VECTOR_DISTANCE] =
            {DeviceType::CPU_SIMD, DeviceType::CPU_AMX, DeviceType::GPU};
        policy.priorities[OperationType::VECTOR_NORMALIZE] =
            {DeviceType::CPU_SIMD, DeviceType::CPU_AMX, DeviceType::GPU};
        policy.priorities[OperationType::MATRIX_MULTIPLY] =
            {DeviceType::CPU_AMX, DeviceType::GPU};

        // ML 推理: NPU 优先
        policy.priorities[OperationType::ML_INFERENCE] =
            {DeviceType::NPU, DeviceType::GPU, DeviceType::CPU_AMX};
        policy.priorities[OperationType::EMBEDDING_TRANSFORM] =
            {DeviceType::NPU, DeviceType::GPU, DeviceType::CPU_AMX};
        policy.priorities[OperationType::RERANKING] =
            {DeviceType::NPU, DeviceType::GPU};
        policy.priorities[OperationType::QUANTIZATION] =
            {DeviceType::NPU, DeviceType::GPU};

        // 大规模并行: GPU 优先
        policy.priorities[OperationType::BATCH_SEARCH] =
            {DeviceType::GPU, DeviceType::CPU_AMX};
        policy.priorities[OperationType::INDEX_BUILD] =
            {DeviceType::GPU, DeviceType::CPU_AMX};
        policy.priorities[OperationType::CLUSTERING] =
            {DeviceType::GPU, DeviceType::CPU_AMX};

        return policy;
    }
};

//=============================================================================
// 设备选择器
//=============================================================================

/**
 * 设备选择器 - 自动选择最优设备
 */
class DeviceSelector {
public:
    /**
     * 获取单例实例
     */
    static DeviceSelector& instance();

    /**
     * 为操作选择最优设备
     * @param op 操作类型
     * @param data_size 数据大小 (元素数量)
     * @param dtype 数据类型
     * @return 推荐的设备类型
     */
    DeviceType select_device(OperationType op,
                             size_t data_size,
                             DataType dtype = DataType::FP32) const;

    /**
     * 检查设备是否可用
     */
    bool is_available(DeviceType device) const;

    /**
     * 获取设备能力
     */
    DeviceCapabilities get_capabilities(DeviceType device) const;

    /**
     * 获取所有可用设备
     */
    std::vector<DeviceType> available_devices() const;

    /**
     * 设置设备选择策略
     */
    void set_policy(const DeviceSelectionPolicy& policy);

    /**
     * 获取当前策略
     */
    const DeviceSelectionPolicy& policy() const;

    /**
     * 设置设备偏好优先级 (覆盖策略)
     * @param device 设备类型
     * @param priority 优先级 (越高越优先)
     */
    void set_preference(DeviceType device, int priority);

    /**
     * 禁用特定设备
     */
    void disable_device(DeviceType device);

    /**
     * 启用特定设备
     */
    void enable_device(DeviceType device);

private:
    DeviceSelector();
    ~DeviceSelector();
    DeviceSelector(const DeviceSelector&) = delete;
    DeviceSelector& operator=(const DeviceSelector&) = delete;

    struct Impl;
    Impl* impl_;
};

//=============================================================================
// 平台信息
//=============================================================================

/**
 * 平台信息查询
 */
namespace platform {

/**
 * 获取 CPU 核心数
 */
size_t get_cpu_cores();

/**
 * 获取性能核心数
 */
size_t get_performance_cores();

/**
 * 获取效率核心数
 */
size_t get_efficiency_cores();

/**
 * 获取 GPU 核心数
 */
size_t get_gpu_cores();

/**
 * 获取系统内存大小 (字节)
 */
size_t get_memory_size();

/**
 * 检查是否有 Neural Engine
 */
bool has_neural_engine();

/**
 * 获取 ANE 核心数
 */
size_t get_ane_cores();

/**
 * 检查是否支持 AMX
 */
bool has_amx();

/**
 * 获取芯片名称 (e.g., "Apple M4 Max")
 */
const char* get_chip_name();

/**
 * 是否运行在 Apple Silicon 上
 */
bool is_apple_silicon();

} // namespace platform

//=============================================================================
// 辅助函数
//=============================================================================

/**
 * 获取数据类型大小 (字节)
 */
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return 4;
        case DataType::FP16:  return 2;
        case DataType::BF16:  return 2;
        case DataType::INT8:  return 1;
        case DataType::INT4:  return 1;  // packed, effective 0.5
        case DataType::UINT8: return 1;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
    }
    return 0;
}

/**
 * 获取设备类型名称
 */
inline const char* device_type_name(DeviceType type) {
    switch (type) {
        case DeviceType::AUTO:     return "AUTO";
        case DeviceType::NPU:      return "NPU";
        case DeviceType::GPU:      return "GPU";
        case DeviceType::CPU_AMX:  return "CPU_AMX";
        case DeviceType::CPU_SIMD: return "CPU_SIMD";
    }
    return "UNKNOWN";
}

/**
 * 获取数据类型名称
 */
inline const char* dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:  return "FP32";
        case DataType::FP16:  return "FP16";
        case DataType::BF16:  return "BF16";
        case DataType::INT8:  return "INT8";
        case DataType::INT4:  return "INT4";
        case DataType::UINT8: return "UINT8";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
    }
    return "UNKNOWN";
}

} // namespace accelerator
} // namespace thunderduck

#endif // THUNDERDUCK_NPU_DEVICE_H
