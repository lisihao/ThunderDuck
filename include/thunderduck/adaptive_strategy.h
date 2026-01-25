/**
 * ThunderDuck - 自适应策略选择器
 *
 * 根据数据特征自动选择最优执行策略:
 * - 数据规模
 * - 选择率估计
 * - 基数估计
 * - 硬件能力
 */

#ifndef THUNDERDUCK_ADAPTIVE_STRATEGY_H
#define THUNDERDUCK_ADAPTIVE_STRATEGY_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace strategy {

// ============================================================================
// 执行器类型
// ============================================================================

enum class Executor {
    CPU_SCALAR,     // CPU 标量
    CPU_SIMD,       // CPU SIMD (NEON)
    GPU,            // GPU (Metal)
    NPU,            // NPU (BNNS)
    AUTO            // 自动选择
};

// ============================================================================
// 算子类型
// ============================================================================

enum class OperatorType {
    FILTER,
    AGGREGATE_SUM,
    AGGREGATE_MINMAX,
    AGGREGATE_GROUP,
    JOIN_HASH,
    TOPK
};

// ============================================================================
// 数据特征
// ============================================================================

struct DataCharacteristics {
    size_t row_count;           // 行数
    size_t column_count;        // 列数 (用于 join)
    size_t element_size;        // 元素大小 (bytes)
    float selectivity;          // 选择率估计 (0-1, -1=未知)
    float cardinality_ratio;    // 基数比 (unique/total, -1=未知)
    bool is_page_aligned;       // 数据是否页对齐
};

// ============================================================================
// 阈值常量 (基于 M4 基准测试结果)
// ============================================================================

namespace thresholds {

// Filter: CPU SIMD 足够快，只有超大数据才用 GPU
constexpr size_t FILTER_GPU_MIN = 10000000;      // 10M rows

// Aggregate: CPU 接近内存带宽上限，GPU 价值有限
constexpr size_t AGGREGATE_GPU_MIN = 100000000;  // 100M rows

// Join: GPU 在中等规模最有效
constexpr size_t JOIN_GPU_MIN_PROBE = 500000;    // 500K probe rows
constexpr size_t JOIN_GPU_MAX_PROBE = 50000000;  // 50M probe rows (超大时带宽受限)
constexpr size_t JOIN_GPU_MIN_BUILD = 10000;     // 10K build rows

// TopK: CPU 采样/count 方法已很高效
constexpr size_t TOPK_GPU_MIN = 50000000;        // 50M rows
constexpr size_t TOPK_K_THRESHOLD = 1000;        // K > 1000 考虑 GPU 排序

}  // namespace thresholds

// ============================================================================
// 策略选择器
// ============================================================================

class StrategySelector {
public:
    /**
     * 获取单例实例
     */
    static StrategySelector& instance();

    /**
     * 选择最优执行器
     */
    Executor select(OperatorType op, const DataCharacteristics& data,
                    Executor hint = Executor::AUTO);

    /**
     * 检查 GPU 是否可用
     */
    bool is_gpu_available() const;

    /**
     * 检查 NPU 是否可用
     */
    bool is_npu_available() const;

    /**
     * 设置强制执行器 (用于测试)
     */
    void set_forced_executor(Executor exec);
    void clear_forced_executor();

    /**
     * 获取决策说明 (调试用)
     */
    const char* get_decision_reason() const;

private:
    StrategySelector();

    Executor selectFilterStrategy(const DataCharacteristics& data);
    Executor selectAggregateStrategy(OperatorType op, const DataCharacteristics& data);
    Executor selectJoinStrategy(const DataCharacteristics& data);
    Executor selectTopKStrategy(const DataCharacteristics& data);

    bool gpu_available_ = false;
    bool npu_available_ = false;
    Executor forced_executor_ = Executor::AUTO;
    mutable const char* last_reason_ = nullptr;
};

// ============================================================================
// 便捷函数
// ============================================================================

/**
 * 快速选择执行器
 */
inline Executor select_executor(OperatorType op, size_t row_count,
                                 float selectivity = -1.0f) {
    DataCharacteristics data;
    data.row_count = row_count;
    data.column_count = 1;
    data.element_size = sizeof(int32_t);
    data.selectivity = selectivity;
    data.cardinality_ratio = -1.0f;
    data.is_page_aligned = false;

    return StrategySelector::instance().select(op, data);
}

/**
 * 判断是否应该使用 GPU
 */
inline bool should_use_gpu(OperatorType op, size_t row_count) {
    return select_executor(op, row_count) == Executor::GPU;
}

}  // namespace strategy
}  // namespace thunderduck

#endif // THUNDERDUCK_ADAPTIVE_STRATEGY_H
