/**
 * ThunderDuck V19 - 统一实现
 *
 * 整合所有最优算子
 */

#include "thunderduck/v19.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/sort.h"

#include <vector>
#include <algorithm>

namespace thunderduck {
namespace v19 {

// ============================================================================
// 全局配置
// ============================================================================

static V19Config g_config;

V19Config& get_config() {
    return g_config;
}

// ============================================================================
// Filter 实现 - V19 两阶段 8T
// ============================================================================

size_t filter_gt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::GT, threshold, out_indices);
}

size_t filter_lt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::LT, threshold, out_indices);
}

size_t filter_eq_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::EQ, threshold, out_indices);
}

size_t filter_ge_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::GE, threshold, out_indices);
}

size_t filter_le_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::LE, threshold, out_indices);
}

size_t filter_ne_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices) {
    return filter::filter_i32_v19(data, count, filter::CompareOp::NE, threshold, out_indices);
}

// ============================================================================
// GROUP BY 实现 - V15 8T + 展开
// ============================================================================

size_t group_sum_i32(const int32_t* group_ids, const int32_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups) {
    // 转换 group_ids 到 uint32_t*
    aggregate::group_sum_i32_v15(values, reinterpret_cast<const uint32_t*>(group_ids),
                                  count, max_groups, out_sums);
    return max_groups;
}

size_t group_sum_i64(const int32_t* group_ids, const int64_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups) {
    aggregate::group_sum_i64(values, reinterpret_cast<const uint32_t*>(group_ids),
                              count, max_groups, out_sums);
    return max_groups;
}

size_t group_count(const int32_t* group_ids, size_t count,
                   int64_t* out_counts, size_t max_groups) {
    std::vector<size_t> counts(max_groups, 0);
    aggregate::group_count(reinterpret_cast<const uint32_t*>(group_ids),
                            count, max_groups, counts.data());
    for (size_t i = 0; i < max_groups; ++i) {
        out_counts[i] = static_cast<int64_t>(counts[i]);
    }
    return max_groups;
}

// ============================================================================
// JOIN 实现 - V14 预分配
// ============================================================================

size_t inner_join_i32(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result) {
    join::JoinResult jr;
    jr.left_indices = result->left_indices;
    jr.right_indices = result->right_indices;
    jr.count = 0;
    jr.capacity = result->capacity;

    size_t count = join::hash_join_i32_v14(build_keys, build_count,
                                            probe_keys, probe_count,
                                            join::JoinType::INNER, &jr);

    result->left_indices = jr.left_indices;
    result->right_indices = jr.right_indices;
    result->count = jr.count;
    result->capacity = jr.capacity;

    return count;
}

// ============================================================================
// SEMI/ANTI JOIN 实现 - GPU
// ============================================================================

size_t semi_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result) {
    join::JoinResult jr;
    jr.left_indices = result->left_indices;
    jr.right_indices = result->right_indices;
    jr.count = 0;
    jr.capacity = result->capacity;

    size_t count;
    if (probe_count >= g_config.semi_join_gpu_threshold && join::is_semi_join_gpu_available()) {
        count = join::semi_join_gpu(build_keys, build_count,
                                     probe_keys, probe_count, &jr);
    } else {
        count = join::hash_join_i32_v10(build_keys, build_count,
                                         probe_keys, probe_count,
                                         join::JoinType::SEMI, &jr);
    }

    result->left_indices = jr.left_indices;
    result->right_indices = jr.right_indices;
    result->count = jr.count;
    result->capacity = jr.capacity;

    return count;
}

size_t anti_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result) {
    join::JoinResult jr;
    jr.left_indices = result->left_indices;
    jr.right_indices = result->right_indices;
    jr.count = 0;
    jr.capacity = result->capacity;

    size_t count;
    if (probe_count >= g_config.semi_join_gpu_threshold && join::is_semi_join_gpu_available()) {
        count = join::anti_join_gpu(build_keys, build_count,
                                     probe_keys, probe_count, &jr);
    } else {
        count = join::hash_join_i32_v10(build_keys, build_count,
                                         probe_keys, probe_count,
                                         join::JoinType::ANTI, &jr);
    }

    result->left_indices = jr.left_indices;
    result->right_indices = jr.right_indices;
    result->count = jr.count;
    result->capacity = jr.capacity;

    return count;
}

// ============================================================================
// TopK 实现 - V4 采样
// ============================================================================

size_t topk_i32(const int32_t* data, size_t count, size_t k,
                uint32_t* out_indices, int32_t* out_values) {
    sort::topk_max_i32_v4(data, count, k, out_values, out_indices);
    return k;
}

size_t topk_i64(const int64_t* data, size_t count, size_t k,
                uint32_t* out_indices, int64_t* out_values) {
    // 回退到基础版本
    std::vector<std::pair<int64_t, uint32_t>> pairs(count);
    for (size_t i = 0; i < count; ++i) {
        pairs[i] = {data[i], static_cast<uint32_t>(i)};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < k; ++i) {
        out_values[i] = pairs[i].first;
        out_indices[i] = pairs[i].second;
    }
    return k;
}

// ============================================================================
// 设备信息
// ============================================================================

DeviceInfo get_device_info() {
    DeviceInfo info;
    info.gpu_available = filter::is_filter_gpu_available();
    info.npu_available = false;  // 暂不支持 NPU
    info.cpu_cores = 10;  // M4 Max
    info.gpu_memory = 0;
    info.gpu_name = "Apple M4 Max GPU";
    return info;
}

bool is_gpu_available() {
    return filter::is_filter_gpu_available();
}

// ============================================================================
// 版本信息
// ============================================================================

const char* get_version_info() {
    return "V19.0 - Performance Baseline V2 (Filter 2.07x, GROUP BY 3.08x, TopK 4.95x)";
}

} // namespace v19
} // namespace thunderduck
