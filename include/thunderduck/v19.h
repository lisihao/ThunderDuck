/**
 * ThunderDuck V19 - 性能基线 V2
 *
 * 整合所有最优算子 + 智能策略选择
 *
 * 最优算子组合 (vs DuckDB):
 * - Filter:     V19 两阶段 8T (2.07x) ★ NEW
 * - GROUP BY:   V15 8T+unroll (3.08x)
 * - INNER JOIN: V14 pre-alloc (0.96x)
 * - SEMI JOIN:  GPU Metal (2.95x)
 * - TopK:       V4 sample (4.95x)
 *
 * 智能策略:
 * - 根据数据量自动选择 CPU/GPU
 * - 根据硬件能力动态调整线程数
 * - 两阶段算法消除内存分配开销
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace v19 {

// ============================================================================
// 配置结构
// ============================================================================

struct V19Config {
    // 自动选择阈值
    size_t filter_parallel_threshold = 500000;  // 500K 以上用 V19 8T
    size_t semi_join_gpu_threshold = 500000;    // 500K 以上用 GPU
    size_t parallel_threshold = 100000;         // 100K 以上用多线程

    // 线程配置
    int num_threads = 8;                        // 默认 8 线程 (M4 Max)

    // 设备选择
    bool prefer_gpu = true;                     // 优先使用 GPU
    bool auto_select = true;                    // 自动选择最优策略
};

// 全局配置
V19Config& get_config();

// ============================================================================
// Filter 算子 - V19 两阶段 8T 并行 (2.07x vs DuckDB)
// ============================================================================

/**
 * V19 Filter - 两阶段并行直写
 *
 * 优化:
 * - 阶段 1: 并行统计匹配数
 * - 阶段 2: 直接写入最终输出 (无临时缓冲区)
 * - 8 线程利用 M4 Max 全核
 *
 * @return 匹配的索引数量
 */
size_t filter_gt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_lt_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_eq_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_ge_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_le_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

size_t filter_ne_i32(const int32_t* data, size_t count,
                     int32_t threshold, uint32_t* out_indices);

// ============================================================================
// GROUP BY SUM 算子 - V15 8T + 循环展开 (3.08x vs DuckDB)
// ============================================================================

/**
 * V19 GROUP BY SUM - 8 线程 + 8 路展开
 *
 * 优化:
 * - 8 线程并行
 * - 8 路循环展开
 * - 缓存行对齐避免伪共享
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
// INNER JOIN 算子 - V14 预分配 (0.96x vs DuckDB)
// ============================================================================

struct JoinResult {
    uint32_t* left_indices;
    uint32_t* right_indices;
    size_t count;
    size_t capacity;
};

/**
 * V19 INNER JOIN - V14 两阶段预分配
 *
 * 优化:
 * - 32 分区 (5-bit radix)
 * - 两阶段: 计数 + 精确分配 + 填充
 * - 消除动态扩容开销
 */
size_t inner_join_i32(const int32_t* build_keys, size_t build_count,
                      const int32_t* probe_keys, size_t probe_count,
                      JoinResult* result);

// ============================================================================
// SEMI JOIN 算子 - GPU Metal (2.95x vs DuckDB)
// ============================================================================

/**
 * V19 SEMI JOIN - GPU 并行探测
 *
 * 优化:
 * - GPU 每线程探测一个 probe 键
 * - 原子计数器收集结果
 * - probe >= 500K 自动使用 GPU
 */
size_t semi_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result);

size_t anti_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinResult* result);

// ============================================================================
// TopK 算子 - V4 采样 (4.95x vs DuckDB)
// ============================================================================

/**
 * V19 TopK - V4 采样预过滤
 *
 * 优化:
 * - 采样估计阈值
 * - SIMD 批量跳过 ~90% 元素
 * - 小 K 堆方法
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

// ============================================================================
// 版本信息
// ============================================================================

const char* get_version_info();

} // namespace v19
} // namespace thunderduck
