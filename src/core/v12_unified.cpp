/**
 * ThunderDuck V12 - 统一最优版本实现
 *
 * 集成各算子历史最优实现，根据数据规模自动选择最佳策略
 */

#include "thunderduck/v12_unified.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"

#include <chrono>
#include <cstring>

namespace thunderduck {
namespace v12 {

// ============================================================================
// 内部工具函数
// ============================================================================

namespace {

using Clock = std::chrono::high_resolution_clock;

inline double calculate_throughput(size_t count, size_t element_size, double elapsed_ms) {
    if (elapsed_ms <= 0) return 0;
    double bytes = count * element_size;
    double seconds = elapsed_ms / 1000.0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;  // GB/s
}

inline void fill_stats(ExecutionStats* stats, const char* op_name,
                       const char* version, const char* device,
                       size_t count, double elapsed_ms, size_t element_size = 4) {
    if (!stats) return;
    stats->operator_name = op_name;
    stats->version_used = version;
    stats->device_used = device;
    stats->data_count = count;
    stats->elapsed_ms = elapsed_ms;
    stats->throughput_gbps = calculate_throughput(count, element_size, elapsed_ms);
}

// 将 v12::CompareOp 转换为 filter::CompareOp
inline filter::CompareOp convert_compare_op(CompareOp op) {
    switch (op) {
        case CompareOp::EQ: return filter::CompareOp::EQ;
        case CompareOp::NE: return filter::CompareOp::NE;
        case CompareOp::LT: return filter::CompareOp::LT;
        case CompareOp::LE: return filter::CompareOp::LE;
        case CompareOp::GT: return filter::CompareOp::GT;
        case CompareOp::GE: return filter::CompareOp::GE;
    }
    return filter::CompareOp::GT;
}

// 将 v12::JoinType 转换为 join::JoinType
inline join::JoinType convert_join_type(JoinType type) {
    switch (type) {
        case JoinType::INNER: return join::JoinType::INNER;
        case JoinType::LEFT:  return join::JoinType::LEFT;
        case JoinType::RIGHT: return join::JoinType::RIGHT;
        case JoinType::FULL:  return join::JoinType::FULL;
        case JoinType::SEMI:  return join::JoinType::SEMI;
        case JoinType::ANTI:  return join::JoinType::ANTI;
    }
    return join::JoinType::INNER;
}

}  // anonymous namespace

// ============================================================================
// V12 Filter 实现
// ============================================================================

size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats) {
    auto start = Clock::now();
    size_t result;
    const char* version;
    const char* device;

    auto filter_op = convert_compare_op(op);

    // V12.1 基准测试结果:
    // 1M: V12 GPU = 7.11x (最优), V3 = 6.88x
    // 10M: V3 CPU = 3.23x (最优), V12 GPU = 2.85x
    // 策略: 小数据用 GPU，大数据用 CPU SIMD
    if (count < SMALL_DATA_THRESHOLD && filter::is_filter_gpu_available()) {
        result = filter::filter_i32_v4(input, count, filter_op, value, out_indices);
        version = "V12.1";
        device = "GPU Metal";
    } else {
        // 大数据: V3 CPU SIMD 更快
        result = filter::filter_i32_v3(input, count, filter_op, value, out_indices);
        version = "V3";
        device = "CPU SIMD";
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Filter", version, device, count, elapsed);

    return result;
}

size_t count_i32(const int32_t* input, size_t count,
                 CompareOp op, int32_t value,
                 ExecutionStats* stats) {
    auto start = Clock::now();
    size_t result;
    const char* version;
    const char* device;

    auto filter_op = convert_compare_op(op);

    if (count < SMALL_DATA_THRESHOLD) {
        // 小数据: V9 CPU SIMD
        result = filter::count_i32_v3(input, count, filter_op, value);
        version = "V9";
        device = "CPU SIMD";
    } else {
        // 大数据: V7 GPU
        if (filter::is_filter_gpu_available()) {
            result = filter::count_i32_v4(input, count, filter_op, value);
            version = "V7";
            device = "GPU Metal";
        } else {
            result = filter::count_i32_v3(input, count, filter_op, value);
            version = "V3";
            device = "CPU SIMD";
        }
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Count", version, device, count, elapsed);

    return result;
}

size_t filter_i32_config(const int32_t* input, size_t count,
                         CompareOp op, int32_t value,
                         uint32_t* out_indices,
                         const V12Config& config,
                         ExecutionStats* stats) {
    auto start = Clock::now();
    size_t result;
    const char* version;
    const char* device;

    auto filter_op = convert_compare_op(op);

    if (config.force_version) {
        switch (config.force_version_number) {
            case 3:
                result = filter::filter_i32_v3(input, count, filter_op, value, out_indices);
                version = "V3 (forced)";
                device = "CPU SIMD";
                break;
            case 4:
            case 7:
                result = filter::filter_i32_v4(input, count, filter_op, value, out_indices);
                version = "V7 (forced)";
                device = "GPU Metal";
                break;
            case 5:
                result = filter::filter_i32_v5(input, count, filter_op, value, out_indices);
                version = "V5 (forced)";
                device = "CPU SIMD";
                break;
            case 6:
            case 9:
                result = filter::filter_i32_v6(input, count, filter_op, value, out_indices);
                version = "V9 (forced)";
                device = "CPU SIMD";
                break;
            default:
                result = filter::filter_i32_v3(input, count, filter_op, value, out_indices);
                version = "V3 (default)";
                device = "CPU SIMD";
        }
    } else {
        // 自动选择
        return filter_i32(input, count, op, value, out_indices, stats);
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Filter", version, device, count, elapsed);

    return result;
}

// ============================================================================
// V12 Aggregate 实现
// ============================================================================

int64_t sum_i32(const int32_t* input, size_t count, ExecutionStats* stats) {
    auto start = Clock::now();
    int64_t result;
    const char* version;
    const char* device;

    if (count < SMALL_DATA_THRESHOLD) {
        // 小数据: V9 CPU SIMD (5.1x)
        result = aggregate::sum_i32_v4(input, count);
        version = "V9";
        device = "CPU SIMD";
    } else {
        // 大数据: V7 GPU Metal (3.5x)
        if (aggregate::is_aggregate_gpu_available()) {
            result = aggregate::sum_i32_v3(input, count);
            version = "V7";
            device = "GPU Metal";
        } else {
            result = aggregate::sum_i32_v4(input, count);
            version = "V4";
            device = "CPU SIMD";
        }
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "SUM", version, device, count, elapsed);

    return result;
}

void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max,
                ExecutionStats* stats) {
    auto start = Clock::now();
    const char* version;
    const char* device;

    if (count < SMALL_DATA_THRESHOLD) {
        // 小数据: V9 CPU SIMD
        aggregate::minmax_i32_v4(input, count, out_min, out_max);
        version = "V9";
        device = "CPU SIMD";
    } else {
        // 大数据: V7 GPU Metal
        if (aggregate::is_aggregate_gpu_available()) {
            aggregate::minmax_i32_v3(input, count, out_min, out_max);
            version = "V7";
            device = "GPU Metal";
        } else {
            aggregate::minmax_i32_v4(input, count, out_min, out_max);
            version = "V4";
            device = "CPU SIMD";
        }
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "MIN/MAX", version, device, count, elapsed);
}

AggregateResult aggregate_all_i32(const int32_t* input, size_t count,
                                   ExecutionStats* stats) {
    auto start = Clock::now();
    const char* version;
    const char* device;

    aggregate::AggregateStats agg_stats;

    if (count < SMALL_DATA_THRESHOLD) {
        agg_stats = aggregate::aggregate_all_i32_v4(input, count);
        version = "V9";
        device = "CPU SIMD";
    } else {
        if (aggregate::is_aggregate_gpu_available()) {
            agg_stats = aggregate::aggregate_all_i32_v3(input, count);
            version = "V7";
            device = "GPU Metal";
        } else {
            agg_stats = aggregate::aggregate_all_i32_v4(input, count);
            version = "V4";
            device = "CPU SIMD";
        }
    }

    AggregateResult result;
    result.sum = agg_stats.sum;
    result.count = agg_stats.count;
    result.min_val = agg_stats.min_val;
    result.max_val = agg_stats.max_val;
    result.avg = (count > 0) ? (double)agg_stats.sum / count : 0;

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Aggregate All", version, device, count, elapsed);

    return result;
}

int64_t sum_i32_config(const int32_t* input, size_t count,
                       const V12Config& config,
                       ExecutionStats* stats) {
    if (!config.force_version) {
        return sum_i32(input, count, stats);
    }

    auto start = Clock::now();
    int64_t result;
    const char* version;
    const char* device = "CPU SIMD";

    switch (config.force_version_number) {
        case 2:
            result = aggregate::sum_i32_v2(input, count);
            version = "V2 (forced)";
            break;
        case 3:
        case 7:
            result = aggregate::sum_i32_v3(input, count);
            version = "V7 (forced)";
            device = "GPU Metal";
            break;
        case 4:
        case 9:
            result = aggregate::sum_i32_v4(input, count);
            version = "V9 (forced)";
            break;
        default:
            result = aggregate::sum_i32(input, count);
            version = "V1 (default)";
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "SUM", version, device, count, elapsed);

    return result;
}

// ============================================================================
// V12.1 GROUP BY 实现
// ============================================================================

void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats) {
    auto start = Clock::now();

    // V12.1 基准测试结果:
    // V8 CPU 4核: 1M = 3.36x, 10M = 2.26x (最优)
    // V12.1 GPU: 1M = 0.97x, 10M = 1.11x (改进但仍不如 CPU)
    // 策略: 始终使用 V8 CPU 多线程并行
    aggregate::group_sum_i32_v4_parallel(values, groups, count, num_groups, out_sums);

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "GROUP BY SUM", "V8", "CPU Parallel (4核)", count, elapsed, 8);
}

void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats) {
    auto start = Clock::now();

    // V8: 始终使用多线程并行
    aggregate::group_count_v4_parallel(groups, count, num_groups, out_counts);

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "GROUP BY COUNT", "V8", "CPU Parallel (4核)", count, elapsed, 4);
}

void group_min_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_mins,
                   ExecutionStats* stats) {
    auto start = Clock::now();

    // V8: CPU 多线程
    aggregate::group_min_i32_v4(values, groups, count, num_groups, out_mins);

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "GROUP BY MIN", "V8", "CPU SIMD", count, elapsed, 8);
}

void group_max_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups, int32_t* out_maxs,
                   ExecutionStats* stats) {
    auto start = Clock::now();

    // V8: CPU 多线程
    aggregate::group_max_i32_v4(values, groups, count, num_groups, out_maxs);

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "GROUP BY MAX", "V8", "CPU SIMD", count, elapsed, 8);
}

// ============================================================================
// V12 TopK 实现
// ============================================================================

void topk_max_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices,
                  ExecutionStats* stats) {
    auto start = Clock::now();
    const char* version;
    const char* device = "CPU";

    // 基准测试显示 V8 (Count-Based) 在所有场景都最快
    // 1M: V8 = 13.22x vs V7 = 12.28x
    // 10M: V8 = 4.89x vs V7 = 4.61x
    sort::topk_max_i32_v5(data, count, k, out_values, out_indices);
    version = "V8";
    device = "CPU Count-Based";

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "TopK Max", version, device, count, elapsed);
}

void topk_min_i32(const int32_t* data, size_t count, size_t k,
                  int32_t* out_values, uint32_t* out_indices,
                  ExecutionStats* stats) {
    auto start = Clock::now();
    const char* version;
    const char* device = "CPU";

    // V8 (Count-Based) 在所有场景都最快
    sort::topk_min_i32_v5(data, count, k, out_values, out_indices);
    version = "V8";
    device = "CPU Count-Based";

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "TopK Min", version, device, count, elapsed);
}

void topk_max_i32_config(const int32_t* data, size_t count, size_t k,
                         int32_t* out_values, uint32_t* out_indices,
                         const V12Config& config,
                         ExecutionStats* stats) {
    if (!config.force_version) {
        topk_max_i32(data, count, k, out_values, out_indices, stats);
        return;
    }

    auto start = Clock::now();
    const char* version;
    const char* device = "CPU";

    switch (config.force_version_number) {
        case 3:
            sort::topk_max_i32_v3(data, count, k, out_values, out_indices);
            version = "V3 (forced)";
            device = "CPU Heap";
            break;
        case 4:
        case 7:
            sort::topk_max_i32_v4(data, count, k, out_values, out_indices);
            version = "V7 (forced)";
            device = "CPU Sampling";
            break;
        case 5:
        case 8:
            sort::topk_max_i32_v5(data, count, k, out_values, out_indices);
            version = "V8 (forced)";
            device = "CPU Count-Based";
            break;
        case 6:
            if (sort::is_topk_gpu_available()) {
                sort::topk_max_i32_v6(data, count, k, out_values, out_indices);
                version = "V6 (forced)";
                device = "GPU Metal";
            } else {
                sort::topk_max_i32_v5(data, count, k, out_values, out_indices);
                version = "V5 (fallback)";
            }
            break;
        default:
            sort::topk_max_i32_v3(data, count, k, out_values, out_indices);
            version = "V3 (default)";
            device = "CPU Heap";
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "TopK Max", version, device, count, elapsed);
}

// ============================================================================
// V12 Hash Join 实现
// ============================================================================

JoinResult* create_join_result(size_t initial_capacity) {
    // v12::JoinResult 和 join::JoinResult 布局相同，可以安全转换
    return reinterpret_cast<JoinResult*>(join::create_join_result(initial_capacity));
}

void free_join_result(JoinResult* result) {
    if (result) {
        join::free_join_result(reinterpret_cast<join::JoinResult*>(result));
    }
}

size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type, JoinResult* result,
                     ExecutionStats* stats) {
    auto start = Clock::now();
    size_t count;
    const char* version;
    const char* device;

    auto jt = convert_join_type(join_type);
    auto* jr = reinterpret_cast<join::JoinResult*>(result);

    // V12.1 自适应策略：根据预估匹配率选择算法
    //
    // 基准测试结果：
    // - 高匹配率 (>30%): V11 SIMD 最优 (预取+展开有效)
    // - 低匹配率 (<20%): V7 Adaptive 最优 (开销小)
    //
    // 启发式判断匹配率：
    // - 如果 probe_count >> build_count，且 build_count 较小，
    //   则可能是低匹配率场景（probe 范围远大于 build）
    //
    // 策略选择：
    // - build_count < 200K 且 probe_count > build_count * 3: 低匹配率 -> V7
    // - 否则: 高匹配率 -> V11 SIMD

    bool likely_low_match_rate = (build_count < 200000) &&
                                  (probe_count > build_count * 3);

    if (likely_low_match_rate) {
        // 低匹配率场景: 使用 V7 Adaptive (更简单，开销小)
        count = join::hash_join_i32_v4(build_keys, build_count,
                                        probe_keys, probe_count, jt, jr);
        version = "V12.1";
        device = "CPU Adaptive (低匹配率)";
    } else {
        // 高匹配率场景: 使用 V11 SIMD (预取+展开有效)
        count = join::hash_join_i32_v11(build_keys, build_count,
                                         probe_keys, probe_count, jt, jr);
        version = "V12.1";
        device = "CPU SIMD (高匹配率)";
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Hash Join", version, device, build_count + probe_count, elapsed);

    return count;
}

size_t hash_join_i32_config(const int32_t* build_keys, size_t build_count,
                            const int32_t* probe_keys, size_t probe_count,
                            JoinType join_type, JoinResult* result,
                            const V12Config& config,
                            ExecutionStats* stats) {
    if (!config.force_version) {
        return hash_join_i32(build_keys, build_count, probe_keys, probe_count,
                            join_type, result, stats);
    }

    auto start = Clock::now();
    size_t count;
    const char* version;
    const char* device = "CPU";

    auto jt = convert_join_type(join_type);
    auto* jr = reinterpret_cast<join::JoinResult*>(result);

    switch (config.force_version_number) {
        case 3:
            count = join::hash_join_i32_v3(build_keys, build_count,
                                           probe_keys, probe_count, jt, jr);
            version = "V3 (forced)";
            device = "CPU Radix";
            break;
        case 4:
        case 7:
            count = join::hash_join_i32_v4(build_keys, build_count,
                                           probe_keys, probe_count, jt, jr);
            version = "V7 (forced)";
            device = "CPU/GPU Adaptive";
            break;
        case 6:
            count = join::hash_join_i32_v6(build_keys, build_count,
                                           probe_keys, probe_count, jt, jr);
            version = "V6 (forced)";
            device = "CPU Prefetch";
            break;
        case 10:
            count = join::hash_join_i32_v10(build_keys, build_count,
                                            probe_keys, probe_count, jt, jr);
            version = "V10 (forced)";
            device = "CPU Full Semantic";
            break;
        case 11:
            count = join::hash_join_i32_v11(build_keys, build_count,
                                            probe_keys, probe_count, jt, jr);
            version = "V11 (forced)";
            device = "CPU SIMD";
            break;
        default:
            count = join::hash_join_i32_v11(build_keys, build_count,
                                            probe_keys, probe_count, jt, jr);
            version = "V11 (default)";
            device = "CPU SIMD";
    }

    auto end = Clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    fill_stats(stats, "Hash Join", version, device, build_count + probe_count, elapsed);

    return count;
}

// ============================================================================
// V12 版本信息
// ============================================================================

const char* get_version_info() {
    return "ThunderDuck V12.1 - Unified Optimal Version\n"
           "Build: 2026-01-27\n"
           "Target: Apple M4 Max\n"
           "Features: Auto-routing to best implementation per operator\n"
           "V12.1: Optimized strategy selection based on benchmark results";
}

const char* get_optimal_versions() {
    return "Optimal Version Matrix (V12.1):\n"
           "| Operator       | Small (<1M)  | Large (>=1M)  |\n"
           "|----------------|--------------|---------------|\n"
           "| Filter         | GPU (7.11x)  | CPU V3 (3.23x)|\n"
           "| Aggregate      | V9 (5.39x)   | V7 (3.57x)    |\n"
           "| TopK           | V8 (14.26x)  | V8 (4.51x)    |\n"
           "| GROUP BY       | V8 (3.36x)   | V8 (2.26x)    |\n"
           "| Hash Join      | V11 (~1.0x)  | V11 (~1.0x)   |";
}

bool is_gpu_available() {
    return filter::is_filter_gpu_available();
}

bool is_npu_available() {
    // NPU 检查 - 暂时返回 false
    return false;
}

} // namespace v12
} // namespace thunderduck
