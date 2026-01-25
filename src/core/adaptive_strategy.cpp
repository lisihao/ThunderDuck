/**
 * ThunderDuck - 自适应策略选择器实现
 */

#include "thunderduck/adaptive_strategy.h"
#include "thunderduck/uma_memory.h"

namespace thunderduck {
namespace strategy {

// ============================================================================
// StrategySelector 实现
// ============================================================================

StrategySelector& StrategySelector::instance() {
    static StrategySelector selector;
    return selector;
}

StrategySelector::StrategySelector() {
    // 检查 GPU 可用性
    gpu_available_ = uma::UMAMemoryManager::instance().is_available();

    // TODO: 检查 NPU 可用性
    npu_available_ = false;
}

bool StrategySelector::is_gpu_available() const {
    return gpu_available_;
}

bool StrategySelector::is_npu_available() const {
    return npu_available_;
}

void StrategySelector::set_forced_executor(Executor exec) {
    forced_executor_ = exec;
}

void StrategySelector::clear_forced_executor() {
    forced_executor_ = Executor::AUTO;
}

const char* StrategySelector::get_decision_reason() const {
    return last_reason_;
}

Executor StrategySelector::select(OperatorType op, const DataCharacteristics& data,
                                   Executor hint) {
    // 强制执行器优先
    if (forced_executor_ != Executor::AUTO) {
        last_reason_ = "Forced executor";
        return forced_executor_;
    }

    // 用户提示优先 (非 AUTO)
    if (hint != Executor::AUTO) {
        // 验证请求的执行器可用
        if (hint == Executor::GPU && !gpu_available_) {
            last_reason_ = "GPU requested but unavailable, fallback to CPU";
            return Executor::CPU_SIMD;
        }
        if (hint == Executor::NPU && !npu_available_) {
            last_reason_ = "NPU requested but unavailable, fallback to CPU";
            return Executor::CPU_SIMD;
        }
        last_reason_ = "User hint";
        return hint;
    }

    // 根据算子类型选择
    switch (op) {
        case OperatorType::FILTER:
            return selectFilterStrategy(data);

        case OperatorType::AGGREGATE_SUM:
        case OperatorType::AGGREGATE_MINMAX:
        case OperatorType::AGGREGATE_GROUP:
            return selectAggregateStrategy(op, data);

        case OperatorType::JOIN_HASH:
            return selectJoinStrategy(data);

        case OperatorType::TOPK:
            return selectTopKStrategy(data);

        default:
            last_reason_ = "Unknown operator, default to CPU SIMD";
            return Executor::CPU_SIMD;
    }
}

Executor StrategySelector::selectFilterStrategy(const DataCharacteristics& data) {
    // Filter: CPU SIMD 非常高效 (~2500 M rows/s)
    // GPU 只有在超大数据量且低选择率时才有优势

    if (!gpu_available_) {
        last_reason_ = "GPU unavailable";
        return Executor::CPU_SIMD;
    }

    // 数据量阈值检查
    if (data.row_count < thresholds::FILTER_GPU_MIN) {
        last_reason_ = "Row count below GPU threshold (10M)";
        return Executor::CPU_SIMD;
    }

    // 选择率检查: 高选择率时 CPU 更优 (输出多，原子争用高)
    if (data.selectivity > 0 && data.selectivity > 0.5f) {
        last_reason_ = "High selectivity favors CPU";
        return Executor::CPU_SIMD;
    }

    // 页对齐数据更适合 GPU (零拷贝)
    if (!data.is_page_aligned) {
        // 非对齐数据需要拷贝，只有更大数据才值得
        if (data.row_count < thresholds::FILTER_GPU_MIN * 2) {
            last_reason_ = "Non-aligned data, threshold doubled";
            return Executor::CPU_SIMD;
        }
    }

    last_reason_ = "Large data with low selectivity, use GPU";
    return Executor::GPU;
}

Executor StrategySelector::selectAggregateStrategy(OperatorType op,
                                                     const DataCharacteristics& data) {
    // Aggregate: CPU 已接近内存带宽上限 (~118 GB/s)
    // GPU 在带宽受限场景无优势，只有超大数据才考虑

    if (!gpu_available_) {
        last_reason_ = "GPU unavailable";
        return Executor::CPU_SIMD;
    }

    // 分组聚合: GPU 原子操作争用，不推荐
    if (op == OperatorType::AGGREGATE_GROUP) {
        // 只有分组数很少时才考虑 GPU
        // 暂时全部用 CPU
        last_reason_ = "Group aggregate uses CPU (atomic contention)";
        return Executor::CPU_SIMD;
    }

    // 简单聚合: 超大数据才用 GPU
    if (data.row_count < thresholds::AGGREGATE_GPU_MIN) {
        last_reason_ = "Row count below GPU threshold (100M)";
        return Executor::CPU_SIMD;
    }

    last_reason_ = "Very large data, use GPU";
    return Executor::GPU;
}

Executor StrategySelector::selectJoinStrategy(const DataCharacteristics& data) {
    // Join: GPU 在中等规模最有效 (4x 加速)
    // 太小: GPU 启动开销主导
    // 太大: 内存带宽主导

    if (!gpu_available_) {
        last_reason_ = "GPU unavailable";
        return Executor::CPU_SIMD;
    }

    // probe_count 是主要指标 (data.row_count 存储 probe 数量)
    // column_count 存储 build 数量
    size_t probe_count = data.row_count;
    size_t build_count = data.column_count;

    // 太小: GPU 启动开销主导
    if (probe_count < thresholds::JOIN_GPU_MIN_PROBE) {
        last_reason_ = "Probe count below threshold (500K)";
        return Executor::CPU_SIMD;
    }

    if (build_count < thresholds::JOIN_GPU_MIN_BUILD) {
        last_reason_ = "Build count below threshold (10K)";
        return Executor::CPU_SIMD;
    }

    // 太大: 内存带宽主导，GPU 优势减弱
    // 但仍比 CPU 快，所以还是用 GPU
    if (probe_count > thresholds::JOIN_GPU_MAX_PROBE) {
        last_reason_ = "Very large probe, GPU but bandwidth limited";
        return Executor::GPU;  // 仍用 GPU，但期望加速比降低
    }

    // 最佳区间: 500K - 50M
    last_reason_ = "Optimal range for GPU join";
    return Executor::GPU;
}

Executor StrategySelector::selectTopKStrategy(const DataCharacteristics& data) {
    // TopK: CPU 采样/count 方法已很高效
    // GPU 只有在 N 很大且 K 较大时才有优势

    if (!gpu_available_) {
        last_reason_ = "GPU unavailable";
        return Executor::CPU_SIMD;
    }

    // 数据量阈值
    if (data.row_count < thresholds::TOPK_GPU_MIN) {
        last_reason_ = "Row count below GPU threshold (50M)";
        return Executor::CPU_SIMD;
    }

    // K 值通过 selectivity 传递 (hack: selectivity = K / row_count)
    // 如果 K 很小，CPU 采样预过滤更高效
    if (data.selectivity > 0 && data.selectivity < 0.001f) {
        // K 很小 (< 0.1% of N)
        last_reason_ = "Small K, CPU sampling more efficient";
        return Executor::CPU_SIMD;
    }

    last_reason_ = "Large N with medium K, use GPU";
    return Executor::GPU;
}

}  // namespace strategy
}  // namespace thunderduck
