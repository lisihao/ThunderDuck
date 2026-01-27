/**
 * ThunderDuck V14 - 统一路由实现
 *
 * 将 V14 优化版本集成到统一接口
 */

#include "thunderduck/v14.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/sort.h"

#include <chrono>

namespace thunderduck {
namespace v14 {

namespace {

const char* V14_VERSION_INFO = "ThunderDuck V14.0 - 深度优化版本";
const char* V14_OPTIMAL_VERSIONS =
    "Filter: V8 SIMD Parallel (CPU) / V12.1 GPU\n"
    "Aggregate: V8 SIMD Parallel (CPU)\n"
    "GROUP BY: V14 寄存器缓冲 + 多路分流 (NEW)\n"
    "TopK: V8 Count-Based / V7 Sampling / V13 GPU\n"
    "Hash Join: V14 两阶段预分配 (NEW)\n";

} // anonymous namespace

// ============================================================================
// Filter - 委托给 V13 策略
// ============================================================================

size_t filter_i32(const int32_t* input, size_t count,
                  CompareOp op, int32_t value,
                  uint32_t* out_indices,
                  ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    // 转换操作符
    filter::CompareOp filter_op;
    switch (op) {
        case CompareOp::EQ: filter_op = filter::CompareOp::EQ; break;
        case CompareOp::NE: filter_op = filter::CompareOp::NE; break;
        case CompareOp::LT: filter_op = filter::CompareOp::LT; break;
        case CompareOp::LE: filter_op = filter::CompareOp::LE; break;
        case CompareOp::GT: filter_op = filter::CompareOp::GT; break;
        case CompareOp::GE: filter_op = filter::CompareOp::GE; break;
        default: filter_op = filter::CompareOp::EQ;
    }

    // 使用 V8 Parallel 策略
    size_t result = filter::filter_i32_parallel(input, count, filter_op, value, out_indices);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "Filter";
        stats->version_used = "V8 SIMD Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(int32_t)) / (elapsed_ms * 1e6);
    }

    return result;
}

// ============================================================================
// Aggregate - 委托给最优实现
// ============================================================================

int64_t sum_i32(const int32_t* input, size_t count, ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    int64_t result = aggregate::sum_i32_parallel(input, count);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "SUM";
        stats->version_used = "V8 Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(int32_t)) / (elapsed_ms * 1e6);
    }

    return result;
}

void minmax_i32(const int32_t* input, size_t count,
                int32_t* out_min, int32_t* out_max,
                ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    aggregate::minmax_i32_parallel(input, count, out_min, out_max);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "MIN/MAX";
        stats->version_used = "V8 Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(int32_t)) / (elapsed_ms * 1e6);
    }
}

// ============================================================================
// GROUP BY - V14 优化实现
// ============================================================================

void group_sum_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int64_t* out_sums,
                   ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    // 使用 V14 实现
    aggregate::group_sum_i32_v14(values, groups, count, num_groups, out_sums);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "GROUP BY SUM";
        stats->version_used = "V14 RegBuf+Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * (sizeof(int32_t) + sizeof(uint32_t))) / (elapsed_ms * 1e6);
    }
}

void group_count(const uint32_t* groups, size_t count,
                 size_t num_groups, size_t* out_counts,
                 ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    aggregate::group_count_v14(groups, count, num_groups, out_counts);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "GROUP BY COUNT";
        stats->version_used = "V14 Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(uint32_t)) / (elapsed_ms * 1e6);
    }
}

void group_min_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int32_t* out_mins,
                   ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    aggregate::group_min_i32_v14(values, groups, count, num_groups, out_mins);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "GROUP BY MIN";
        stats->version_used = "V14 Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * (sizeof(int32_t) + sizeof(uint32_t))) / (elapsed_ms * 1e6);
    }
}

void group_max_i32(const int32_t* values, const uint32_t* groups,
                   size_t count, size_t num_groups,
                   int32_t* out_maxs,
                   ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    aggregate::group_max_i32_v14(values, groups, count, num_groups, out_maxs);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "GROUP BY MAX";
        stats->version_used = "V14 Parallel";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * (sizeof(int32_t) + sizeof(uint32_t))) / (elapsed_ms * 1e6);
    }
}

// ============================================================================
// TopK - 委托给最优实现
// ============================================================================

size_t topk_max_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices,
                    ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    sort::topk_max_i32_v6(data, count, k, out_values, out_indices);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "TopK MAX";
        stats->version_used = "V6 Sampling";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(int32_t)) / (elapsed_ms * 1e6);
    }

    return k;
}

size_t topk_min_i32(const int32_t* data, size_t count, size_t k,
                    int32_t* out_values, uint32_t* out_indices,
                    ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    sort::topk_min_i32_v6(data, count, k, out_values, out_indices);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "TopK MIN";
        stats->version_used = "V6 Sampling";
        stats->device_used = "CPU";
        stats->data_count = count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = (count * sizeof(int32_t)) / (elapsed_ms * 1e6);
    }

    return k;
}

// ============================================================================
// Hash Join - V14 两阶段预分配
// ============================================================================

JoinResult* create_join_result(size_t initial_capacity) {
    auto* result = new JoinResult();
    result->left_indices = new uint32_t[initial_capacity];
    result->right_indices = new uint32_t[initial_capacity];
    result->count = 0;
    result->capacity = initial_capacity;
    return result;
}

void free_join_result(JoinResult* result) {
    if (result) {
        delete[] result->left_indices;
        delete[] result->right_indices;
        delete result;
    }
}

size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type, JoinResult* result,
                     ExecutionStats* stats) {
    auto start = std::chrono::high_resolution_clock::now();

    // 转换 JoinType
    join::JoinType join_type_internal;
    switch (join_type) {
        case JoinType::INNER: join_type_internal = join::JoinType::INNER; break;
        case JoinType::LEFT: join_type_internal = join::JoinType::LEFT; break;
        case JoinType::RIGHT: join_type_internal = join::JoinType::RIGHT; break;
        case JoinType::FULL: join_type_internal = join::JoinType::FULL; break;
        case JoinType::SEMI: join_type_internal = join::JoinType::SEMI; break;
        case JoinType::ANTI: join_type_internal = join::JoinType::ANTI; break;
        default: join_type_internal = join::JoinType::INNER;
    }

    // 创建内部结果
    join::JoinResult* internal_result = join::create_join_result(
        std::max(probe_count, build_count));

    // 使用 V14 实现
    size_t match_count = join::hash_join_i32_v14(
        build_keys, build_count,
        probe_keys, probe_count,
        join_type_internal,
        internal_result);

    // 复制结果
    if (result->capacity < match_count) {
        delete[] result->left_indices;
        delete[] result->right_indices;
        result->left_indices = new uint32_t[match_count];
        result->right_indices = new uint32_t[match_count];
        result->capacity = match_count;
    }

    std::memcpy(result->left_indices, internal_result->left_indices,
                match_count * sizeof(uint32_t));
    std::memcpy(result->right_indices, internal_result->right_indices,
                match_count * sizeof(uint32_t));
    result->count = match_count;

    join::free_join_result(internal_result);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (stats) {
        stats->operator_name = "Hash Join";
        stats->version_used = "V14 Two-Phase";
        stats->device_used = "CPU";
        stats->data_count = build_count + probe_count;
        stats->elapsed_ms = elapsed_ms;
        stats->throughput_gbps = ((build_count + probe_count) * sizeof(int32_t)) /
                                  (elapsed_ms * 1e6);
    }

    return match_count;
}

// ============================================================================
// 版本信息
// ============================================================================

const char* get_version_info() {
    return V14_VERSION_INFO;
}

const char* get_optimal_versions() {
    return V14_OPTIMAL_VERSIONS;
}

bool is_gpu_available() {
#ifdef __APPLE__
    return true;  // Apple Silicon 支持 Metal
#else
    return false;
#endif
}

// ============================================================================
// detail 命名空间 - 底层接口
// ============================================================================

namespace detail {

size_t hash_join_i32_v14(const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result) {
    join::JoinType internal_type;
    switch (join_type) {
        case JoinType::INNER: internal_type = join::JoinType::INNER; break;
        case JoinType::LEFT: internal_type = join::JoinType::LEFT; break;
        case JoinType::RIGHT: internal_type = join::JoinType::RIGHT; break;
        case JoinType::FULL: internal_type = join::JoinType::FULL; break;
        case JoinType::SEMI: internal_type = join::JoinType::SEMI; break;
        case JoinType::ANTI: internal_type = join::JoinType::ANTI; break;
        default: internal_type = join::JoinType::INNER;
    }

    join::JoinResult* internal_result = join::create_join_result(
        std::max(probe_count, build_count));

    size_t match_count = join::hash_join_i32_v14(
        build_keys, build_count,
        probe_keys, probe_count,
        internal_type,
        internal_result);

    // 复制结果
    if (result->capacity < match_count) {
        delete[] result->left_indices;
        delete[] result->right_indices;
        result->left_indices = new uint32_t[match_count];
        result->right_indices = new uint32_t[match_count];
        result->capacity = match_count;
    }

    std::memcpy(result->left_indices, internal_result->left_indices,
                match_count * sizeof(uint32_t));
    std::memcpy(result->right_indices, internal_result->right_indices,
                match_count * sizeof(uint32_t));
    result->count = match_count;

    join::free_join_result(internal_result);
    return match_count;
}

void group_sum_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    aggregate::group_sum_i32_v14(values, groups, count, num_groups, out_sums);
}

void group_count_v14(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts) {
    aggregate::group_count_v14(groups, count, num_groups, out_counts);
}

void group_min_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins) {
    aggregate::group_min_i32_v14(values, groups, count, num_groups, out_mins);
}

void group_max_i32_v14(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs) {
    aggregate::group_max_i32_v14(values, groups, count, num_groups, out_maxs);
}

void group_sum_i32_v14_parallel(const int32_t* values, const uint32_t* groups,
                                 size_t count, size_t num_groups, int64_t* out_sums) {
    aggregate::group_sum_i32_v14_parallel(values, groups, count, num_groups, out_sums);
}

} // namespace detail

} // namespace v14
} // namespace thunderduck
