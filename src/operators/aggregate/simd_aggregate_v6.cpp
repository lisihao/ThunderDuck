/**
 * ThunderDuck - SIMD Aggregation Implementation v6.0 (V9.3)
 *
 * 智能策略选择分组聚合:
 * - 根据数据规模和分组数自动选择最优实现
 * - V4_SINGLE: CPU 单线程 (小数据)
 * - V4_PARALLEL: CPU 多线程 (中等数据，默认最优)
 * - V5_GPU: GPU 两阶段 (超大数据 + 低分组数)
 *
 * 策略阈值基于 M4 基准测试结果:
 * | 场景 | CPU v4-MT | GPU v5 | 结论 |
 * |------|-----------|--------|------|
 * | 10M, 10 groups | 2.54x | 2.53x | CPU 略优 |
 * | 10M, 100 groups | 2.29x | 1.35x | CPU 明显优 |
 * | 10M, 1000 groups | 2.64x | 1.38x | CPU 明显优 |
 *
 * GPU v5 在高竞争场景 (少分组) 表现接近 CPU，
 * 理论上在超大数据 (>50M) 时 GPU 带宽优势更明显。
 */

#include "thunderduck/aggregate.h"
#include <cstring>
#include <cstdio>

namespace thunderduck {
namespace aggregate {

// ============================================================================
// 策略阈值常量 (基于 M4 基准测试)
// ============================================================================

namespace {

// 小于此值使用单线程 (避免线程启动开销)
constexpr size_t SINGLE_THREAD_MAX = 100000;  // 100K

// 大于此值且分组数少时考虑 GPU
constexpr size_t GPU_MIN_COUNT = 50000000;    // 50M

// GPU 适用的最大分组数 (高竞争场景)
constexpr size_t GPU_MAX_GROUPS = 32;

// 策略选择原因
thread_local const char* g_strategy_reason = nullptr;

}  // anonymous namespace

// ============================================================================
// 策略选择器
// ============================================================================

GroupAggregateVersion select_group_aggregate_strategy(
    size_t count, size_t num_groups) {

    // 1. 小数据: 单线程
    if (count < SINGLE_THREAD_MAX) {
        g_strategy_reason = "Small data (<100K), use single-threaded V4";
        return GroupAggregateVersion::V4_SINGLE;
    }

    // 2. 超大数据 + 少分组 + GPU 可用: GPU 两阶段
    if (count >= GPU_MIN_COUNT &&
        num_groups <= GPU_MAX_GROUPS &&
        is_group_aggregate_v2_available()) {
        g_strategy_reason = "Large data (>=50M) with few groups (<=32), use GPU V5";
        return GroupAggregateVersion::V5_GPU;
    }

    // 3. 默认: CPU 多线程 (最佳通用性能)
    g_strategy_reason = "Medium/large data, use multi-threaded V4 (best general performance)";
    return GroupAggregateVersion::V4_PARALLEL;
}

const char* get_group_aggregate_strategy_reason() {
    return g_strategy_reason ? g_strategy_reason : "Unknown";
}

// ============================================================================
// v6.0 智能策略分组求和
// ============================================================================

void group_sum_i32_v6_config(const int32_t* values, const uint32_t* groups,
                              size_t count, size_t num_groups, int64_t* out_sums,
                              const GroupAggregateConfig& config) {
    if (count == 0 || !values || !groups || !out_sums) return;

    GroupAggregateVersion version = config.version;

    // 自动选择策略
    if (version == GroupAggregateVersion::AUTO) {
        version = select_group_aggregate_strategy(count, num_groups);
    }

    // 调试日志
    if (config.debug_log) {
        const char* version_name = "Unknown";
        switch (version) {
            case GroupAggregateVersion::V4_SINGLE:   version_name = "V4_SINGLE"; break;
            case GroupAggregateVersion::V4_PARALLEL: version_name = "V4_PARALLEL"; break;
            case GroupAggregateVersion::V5_GPU:      version_name = "V5_GPU"; break;
            default: break;
        }
        // 使用 stderr 避免影响性能测量
        fprintf(stderr, "[V9.3] Strategy: %s, Reason: %s\n",
                version_name, get_group_aggregate_strategy_reason());
    }

    // 执行选定版本
    switch (version) {
        case GroupAggregateVersion::V4_SINGLE:
            group_sum_i32_v4(values, groups, count, num_groups, out_sums);
            break;

        case GroupAggregateVersion::V4_PARALLEL:
            group_sum_i32_v4_parallel(values, groups, count, num_groups, out_sums);
            break;

        case GroupAggregateVersion::V5_GPU:
            group_sum_i32_v5(values, groups, count, num_groups, out_sums);
            break;

        default:
            // 默认使用多线程版本
            group_sum_i32_v4_parallel(values, groups, count, num_groups, out_sums);
            break;
    }
}

void group_sum_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int64_t* out_sums) {
    GroupAggregateConfig config;
    config.version = GroupAggregateVersion::AUTO;
    config.debug_log = false;
    group_sum_i32_v6_config(values, groups, count, num_groups, out_sums, config);
}

// ============================================================================
// v6.0 智能策略分组计数
// ============================================================================

void group_count_v6(const uint32_t* groups, size_t count,
                     size_t num_groups, size_t* out_counts) {
    if (count == 0 || !groups || !out_counts) return;

    GroupAggregateVersion version = select_group_aggregate_strategy(count, num_groups);

    switch (version) {
        case GroupAggregateVersion::V4_SINGLE:
            group_count_v4(groups, count, num_groups, out_counts);
            break;

        case GroupAggregateVersion::V4_PARALLEL:
            group_count_v4_parallel(groups, count, num_groups, out_counts);
            break;

        case GroupAggregateVersion::V5_GPU:
            group_count_v5(groups, count, num_groups, out_counts);
            break;

        default:
            group_count_v4_parallel(groups, count, num_groups, out_counts);
            break;
    }
}

// ============================================================================
// v6.0 智能策略分组 MIN
// ============================================================================

void group_min_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_mins) {
    if (count == 0 || !values || !groups || !out_mins) return;

    GroupAggregateVersion version = select_group_aggregate_strategy(count, num_groups);

    switch (version) {
        case GroupAggregateVersion::V4_SINGLE:
        case GroupAggregateVersion::V4_PARALLEL:
            // MIN/MAX 目前没有多线程版本，使用 v4
            group_min_i32_v4(values, groups, count, num_groups, out_mins);
            break;

        case GroupAggregateVersion::V5_GPU:
            group_min_i32_v5(values, groups, count, num_groups, out_mins);
            break;

        default:
            group_min_i32_v4(values, groups, count, num_groups, out_mins);
            break;
    }
}

// ============================================================================
// v6.0 智能策略分组 MAX
// ============================================================================

void group_max_i32_v6(const int32_t* values, const uint32_t* groups,
                       size_t count, size_t num_groups, int32_t* out_maxs) {
    if (count == 0 || !values || !groups || !out_maxs) return;

    GroupAggregateVersion version = select_group_aggregate_strategy(count, num_groups);

    switch (version) {
        case GroupAggregateVersion::V4_SINGLE:
        case GroupAggregateVersion::V4_PARALLEL:
            // MIN/MAX 目前没有多线程版本，使用 v4
            group_max_i32_v4(values, groups, count, num_groups, out_maxs);
            break;

        case GroupAggregateVersion::V5_GPU:
            group_max_i32_v5(values, groups, count, num_groups, out_maxs);
            break;

        default:
            group_max_i32_v4(values, groups, count, num_groups, out_maxs);
            break;
    }
}

}  // namespace aggregate
}  // namespace thunderduck
