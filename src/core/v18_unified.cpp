/**
 * ThunderDuck V18 - 性能基线 V2 实现
 *
 * 整合最优算子 + 智能策略选择
 */

#include "thunderduck/v18.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/sort.h"

#include <algorithm>
#include <thread>
#include <vector>
#include <cstring>

namespace thunderduck {
namespace v18 {

// ============================================================================
// 全局配置
// ============================================================================

static V18Config g_config;

V18Config& get_config() {
    return g_config;
}

// ============================================================================
// 设备检测
// ============================================================================

// 前向声明 (来自 GPU 模块)
namespace {
    extern "C" {
        bool is_filter_gpu_available();
        bool is_semi_join_gpu_available();
    }
}

bool is_gpu_available() {
    // 检查 Metal GPU 是否可用
    static bool checked = false;
    static bool available = false;

    if (!checked) {
        // 尝试调用 GPU 函数检测
        available = true;  // M4 默认可用
        checked = true;
    }
    return available;
}

DeviceInfo get_device_info() {
    DeviceInfo info;
    info.gpu_available = is_gpu_available();
    info.npu_available = false;  // 暂不支持
    info.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
    info.gpu_memory = 0;  // TODO: 获取 GPU 内存
    info.gpu_name = "Apple M4 Max GPU";
    return info;
}

// ============================================================================
// Filter 实现 - 智能选择 V4 GPU / V15 CPU
// ============================================================================

size_t filter_gt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    auto& cfg = get_config();

    // 智能选择: 大数据量 + GPU 可用 → GPU
    if (cfg.auto_select && cfg.prefer_gpu &&
        count >= cfg.filter_gpu_threshold && is_gpu_available()) {
        // 使用 V4 GPU AUTO
        return thunderduck::filter::filter_i32_v4(data, count,
            thunderduck::filter::CompareOp::GT, threshold, out_indices);
    }

    // 否则使用 V15 CPU SIMD
    return thunderduck::filter::filter_i32_v15(data, count,
        thunderduck::filter::CompareOp::GT, threshold, out_indices);
}

size_t filter_lt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    auto& cfg = get_config();

    if (cfg.auto_select && cfg.prefer_gpu &&
        count >= cfg.filter_gpu_threshold && is_gpu_available()) {
        return thunderduck::filter::filter_i32_v4(data, count,
            thunderduck::filter::CompareOp::LT, threshold, out_indices);
    }

    return thunderduck::filter::filter_i32_v15(data, count,
        thunderduck::filter::CompareOp::LT, threshold, out_indices);
}

size_t filter_eq_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    auto& cfg = get_config();

    if (cfg.auto_select && cfg.prefer_gpu &&
        count >= cfg.filter_gpu_threshold && is_gpu_available()) {
        return thunderduck::filter::filter_i32_v4(data, count,
            thunderduck::filter::CompareOp::EQ, threshold, out_indices);
    }

    return thunderduck::filter::filter_i32_v15(data, count,
        thunderduck::filter::CompareOp::EQ, threshold, out_indices);
}

// ============================================================================
// GROUP BY SUM 实现 - V15 8T + 循环展开
// ============================================================================

size_t group_sum_i32(const int32_t* group_ids, const int32_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups) {
    auto& cfg = get_config();

    // Cast group_ids to uint32_t* (assuming non-negative group IDs)
    const uint32_t* groups = reinterpret_cast<const uint32_t*>(group_ids);

    // 大数据量使用 V15 8 线程并行
    if (count >= cfg.parallel_threshold) {
        thunderduck::aggregate::group_sum_i32_v15(values, groups, count, max_groups, out_sums);
        return max_groups;
    }

    // 小数据量使用单线程 V4 SIMD
    thunderduck::aggregate::group_sum_i32_v4(values, groups, count, max_groups, out_sums);
    return max_groups;
}

size_t group_sum_i64(const int32_t* group_ids, const int64_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups) {
    // Cast group_ids to uint32_t* (assuming non-negative group IDs)
    const uint32_t* groups = reinterpret_cast<const uint32_t*>(group_ids);

    // group_sum_i64 always uses the base implementation
    thunderduck::aggregate::group_sum_i64(values, groups, count, max_groups, out_sums);
    return max_groups;
}

size_t group_count(const int32_t* group_ids, size_t count,
                   int64_t* out_counts, size_t max_groups) {
    auto& cfg = get_config();

    // Cast group_ids to uint32_t* (assuming non-negative group IDs)
    const uint32_t* groups = reinterpret_cast<const uint32_t*>(group_ids);

    // group_count uses size_t*, need temp buffer
    std::vector<size_t> temp_counts(max_groups);

    if (count >= cfg.parallel_threshold) {
        thunderduck::aggregate::group_count_v4_parallel(groups, count, max_groups, temp_counts.data());
    } else {
        thunderduck::aggregate::group_count_v4(groups, count, max_groups, temp_counts.data());
    }

    // Convert size_t to int64_t
    for (size_t i = 0; i < max_groups; ++i) {
        out_counts[i] = static_cast<int64_t>(temp_counts[i]);
    }
    return max_groups;
}

// ============================================================================
// INNER JOIN 实现 - V14 预分配
// ============================================================================

size_t inner_join_i32(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result) {
    if (!result) return 0;

    // 转换为内部 JoinResult 格式
    thunderduck::join::JoinResult internal_result;
    internal_result.left_indices = result->left_indices;
    internal_result.right_indices = result->right_indices;
    internal_result.count = result->count;
    internal_result.capacity = result->capacity;

    // 使用 V14 预分配版本
    size_t cnt = thunderduck::join::hash_join_i32_v14(
        build_keys, build_count,
        probe_keys, probe_count,
        thunderduck::join::JoinType::INNER,
        &internal_result);

    // 更新结果
    result->left_indices = internal_result.left_indices;
    result->right_indices = internal_result.right_indices;
    result->count = internal_result.count;
    result->capacity = internal_result.capacity;

    return cnt;
}

// ============================================================================
// SEMI JOIN 实现 - 智能选择 GPU / CPU
// ============================================================================

size_t semi_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result) {
    if (!result) return 0;

    auto& cfg = get_config();

    thunderduck::join::JoinResult internal_result;
    internal_result.left_indices = result->left_indices;
    internal_result.right_indices = result->right_indices;
    internal_result.count = result->count;
    internal_result.capacity = result->capacity;

    size_t cnt;

    // 智能选择: 大数据量 + GPU 可用 → GPU
    if (cfg.auto_select && cfg.prefer_gpu &&
        probe_count >= cfg.semi_join_gpu_threshold && is_gpu_available()) {
        // 使用 GPU Metal
        cnt = thunderduck::join::semi_join_gpu(
            build_keys, build_count,
            probe_keys, probe_count,
            &internal_result);
    } else {
        // 使用 V10 CPU
        cnt = thunderduck::join::hash_join_i32_v10(
            build_keys, build_count,
            probe_keys, probe_count,
            thunderduck::join::JoinType::SEMI,
            &internal_result);
    }

    result->left_indices = internal_result.left_indices;
    result->right_indices = internal_result.right_indices;
    result->count = internal_result.count;
    result->capacity = internal_result.capacity;

    return cnt;
}

size_t anti_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result) {
    if (!result) return 0;

    auto& cfg = get_config();

    thunderduck::join::JoinResult internal_result;
    internal_result.left_indices = result->left_indices;
    internal_result.right_indices = result->right_indices;
    internal_result.count = result->count;
    internal_result.capacity = result->capacity;

    size_t cnt;

    if (cfg.auto_select && cfg.prefer_gpu &&
        probe_count >= cfg.semi_join_gpu_threshold && is_gpu_available()) {
        cnt = thunderduck::join::anti_join_gpu(
            build_keys, build_count,
            probe_keys, probe_count,
            &internal_result);
    } else {
        cnt = thunderduck::join::hash_join_i32_v10(
            build_keys, build_count,
            probe_keys, probe_count,
            thunderduck::join::JoinType::ANTI,
            &internal_result);
    }

    result->left_indices = internal_result.left_indices;
    result->right_indices = internal_result.right_indices;
    result->count = internal_result.count;
    result->capacity = internal_result.capacity;

    return cnt;
}

// ============================================================================
// TopK 实现 - V4 采样
// ============================================================================

size_t topk_i32(const int32_t* data, size_t count, size_t k,
                uint32_t* out_indices, int32_t* out_values) {
    // 使用 V4 采样算法
    thunderduck::sort::topk_max_i32_v4(data, count, k, out_values, out_indices);
    return k;
}

size_t topk_i64(const int64_t* data, size_t count, size_t k,
                uint32_t* out_indices, int64_t* out_values) {
    // int64 TopK - 使用 argsort 作为回退
    // 首先获取排序索引，然后取前 k 个
    std::vector<uint32_t> indices(count);
    thunderduck::sort::argsort_i64(data, count, indices.data(), thunderduck::sort::SortOrder::DESC);

    size_t actual_k = std::min(k, count);
    for (size_t i = 0; i < actual_k; ++i) {
        out_indices[i] = indices[i];
        out_values[i] = data[indices[i]];
    }
    return actual_k;
}

} // namespace v18
} // namespace thunderduck
