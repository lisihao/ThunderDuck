/**
 * ThunderDuck V18 - 性能基线 V2
 *
 * 整合所有最优算子 + 智能策略选择
 *
 * 最优算子组合:
 * - Filter:    V4 GPU (1.10x) / V15 CPU (0.82x)
 * - GROUP BY:  V15 8T+unroll (2.69x)
 * - INNER JOIN: V14 pre-alloc (1.63x)
 * - SEMI JOIN: GPU Metal (2.47x) / V10 CPU (0.89x)
 * - TopK:      V4 sample (4.71x)
 *
 * 智能策略:
 * - 根据数据量自动选择 CPU/GPU
 * - 根据硬件能力动态调整线程数
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v18 {

// ============================================================================
// 配置结构
// ============================================================================

struct V18Config {
    // 自动选择阈值
    size_t filter_gpu_threshold = 1000000;     // 1M 以上用 GPU
    size_t semi_join_gpu_threshold = 500000;   // 500K 以上用 GPU
    size_t parallel_threshold = 100000;        // 100K 以上用多线程

    // 线程配置
    int num_threads = 8;                       // 默认 8 线程

    // 设备选择
    bool prefer_gpu = true;                    // 优先使用 GPU
    bool auto_select = true;                   // 自动选择最优策略
};

// 全局配置
V18Config& get_config();

// ============================================================================
// Filter 算子 - 智能选择 V4 GPU / V15 CPU
// ============================================================================

/**
 * V18 Filter - 智能策略选择
 *
 * 策略:
 * - 数据量 >= 1M && GPU 可用: 使用 V4 GPU (1.10x)
 * - 其他情况: 使用 V15 CPU SIMD (0.82x)
 *
 * @return 匹配的索引数量
 */
size_t filter_gt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_lt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_eq_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

// ============================================================================
// GROUP BY SUM 算子 - V15 8T + 循环展开 (2.69x)
// ============================================================================

/**
 * V18 GROUP BY SUM - 8 线程 + 循环展开
 *
 * 策略:
 * - 数据量 >= 100K: 8 线程并行
 * - 其他: 单线程 SIMD
 *
 * @return 分组数量
 */
size_t group_sum_i32(const int32_t* group_ids, const int32_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups);

size_t group_sum_i64(const int32_t* group_ids, const int64_t* values,
                     size_t count, int64_t* out_sums, size_t max_groups);

size_t group_count(const int32_t* group_ids, size_t count,
                   int64_t* out_counts, size_t max_groups);

// ============================================================================
// INNER JOIN 算子 - V14 预分配 (1.63x)
// ============================================================================

struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

/**
 * V18 INNER JOIN - V14 预分配
 *
 * 优化: 预分配输出数组，消除 realloc 开销
 */
size_t inner_join_i32(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result);

// ============================================================================
// SEMI JOIN 算子 - 智能选择 GPU / CPU
// ============================================================================

/**
 * V18 SEMI JOIN - 智能策略选择
 *
 * 策略:
 * - probe_count >= 500K && GPU 可用: 使用 GPU Metal (2.47x)
 * - 其他: 使用 V10 CPU SIMD (0.89x)
 */
size_t semi_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result);

size_t anti_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result);

// ============================================================================
// TopK 算子 - V4 采样 (4.71x)
// ============================================================================

/**
 * V18 TopK - V4 采样算法
 *
 * 优化: 采样确定阈值，SIMD 快速筛选
 */
size_t topk_i32(const int32_t* data, size_t count, size_t k,
                uint32_t* out_indices, int32_t* out_values);

size_t topk_i64(const int64_t* data, size_t count, size_t k,
                uint32_t* out_indices, int64_t* out_values);

// ============================================================================
// 设备信息
// ============================================================================

struct DeviceInfo {
    bool gpu_available;
    bool npu_available;
    int cpu_cores;
    size_t gpu_memory;
    const char* gpu_name;
};

DeviceInfo get_device_info();

// 检查 GPU 是否可用
bool is_gpu_available();

} // namespace v18
} // namespace thunderduck
