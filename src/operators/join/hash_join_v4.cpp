/**
 * ThunderDuck - Hash Join v4.0 Implementation
 *
 * 多策略自适应 Hash Join:
 * - PERFECT_HASH: 完美哈希 (连续整数键 O(1) 直接索引)
 * - RADIX256: 256 分区 (8-bit) L1 缓存优化
 * - BLOOMFILTER: CPU Bloom 预过滤 (低选择率时)
 * - NPU: BNNS 加速 (实验性)
 * - GPU: Metal 并行探测 (超大规模)
 *
 * 策略回退链: PERFECT_HASH → GPU → BLOOMFILTER → RADIX_ADAPTIVE → V3
 */

#include "thunderduck/join.h"
#include "thunderduck/bloom_filter.h"
#include "thunderduck/perfect_hash.h"
#include "thunderduck/memory.h"

#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <memory>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// 前向声明策略实现
namespace thunderduck {
namespace join {
namespace v4 {

// RADIX256 策略 (在 hash_join_v4_radix256.cpp)
size_t hash_join_radix256(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

// BloomFilter 策略 (在 hash_join_v4_bloom.cpp)
size_t hash_join_bloom(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

// NPU 策略 (在 bloom_bnns.cpp)
size_t hash_join_npu(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

// GPU 策略 (在 hash_join_metal.mm - 旧版)
size_t hash_join_gpu(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

// GPU v5 策略 (在 hash_join_metal_v5.mm - 新版)
size_t hash_join_gpu_v5(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

// 检查 v5 GPU 可用性
bool is_gpu_v5_ready();

// 检查 NPU 可用性
bool is_npu_ready();

// 检查 GPU 可用性
bool is_gpu_ready();

} // namespace v4
} // namespace join
} // namespace thunderduck

namespace thunderduck {
namespace join {

// ============================================================================
// 策略调度器
// ============================================================================

namespace {

// 策略阈值常量
constexpr size_t SMALL_TABLE_THRESHOLD = 10000;     // 小表阈值
constexpr size_t BLOOM_MIN_BUILD = 500000;          // Bloom 最小 build 数量 (提高阈值)
constexpr size_t BLOOM_MAX_SELECTIVITY = 30;        // Bloom 最大选择率 (30%)
constexpr size_t NPU_MIN_BUILD = 2000000;           // NPU 最小 build 数量 (提高阈值)

// GPU 阈值 (V10 调整 - 降低以启用 GPU 加速)
constexpr size_t GPU_MIN_TOTAL = 5000000;           // GPU 最小总数据量 (5M)
constexpr size_t GPU_MIN_PROBE = 1000000;           // GPU 最小 probe 数量 (1M)
constexpr size_t GPU_MIN_BUILD = 500000;            // GPU 最小 build 数量 (500K)

// 自适应分区阈值
constexpr size_t RADIX_THRESHOLD_0 = 100000;        // < 100K: 不分区
constexpr size_t RADIX_THRESHOLD_16 = 500000;       // < 500K: 16 分区
constexpr size_t RADIX_THRESHOLD_64 = 2000000;      // < 2M: 64 分区
                                                     // >= 2M: 256 分区

/**
 * 选择自适应分区位数
 */
inline int select_adaptive_radix_bits(size_t build_count, size_t probe_count) {
    size_t total = build_count + probe_count;

    if (total < RADIX_THRESHOLD_0) return 0;     // 不分区
    if (total < RADIX_THRESHOLD_16) return 4;    // 16 分区 (同 v3)
    if (total < RADIX_THRESHOLD_64) return 6;    // 64 分区
    return 8;                                     // 256 分区
}

/**
 * 策略调度器
 * 根据数据特征和硬件可用性选择最优策略
 */
class StrategyDispatcher {
public:
    /**
     * 自动选择策略
     */
    static JoinStrategy select_strategy(size_t build_count, size_t probe_count,
                                         const JoinConfigV4& config) {
        // 用户明确指定策略
        if (config.strategy != JoinStrategy::AUTO) {
            // 检查指定策略是否可用
            if (is_strategy_available(config.strategy)) {
                return config.strategy;
            }
            // 不可用时根据 fallback 配置决定
            if (!config.fallback_to_cpu) {
                return config.strategy;  // 强制使用（可能失败）
            }
            // 回退到自动选择
        }

        // AUTO 策略选择逻辑
        // 注意: 完美哈希检查已在 hash_join_i32_v4_config 入口处完成

        // 1. 小表直接使用 v3 (已有完美哈希 + SIMD)
        if (build_count < SMALL_TABLE_THRESHOLD) {
            return JoinStrategy::V3_FALLBACK;
        }

        // 2. GPU: 仅用于超大规模 (500M+ 数据量)
        //    注意: 当前 GPU hash join 实现不如优化的 CPU 版本
        //    保留代码供未来研究，但设置极高阈值以避免性能下降
        size_t total = build_count + probe_count;
        if (total >= GPU_MIN_TOTAL && v4::is_gpu_ready()) {
            return JoinStrategy::GPU;
        }

        // 3. BloomFilter: 用于大表 (500K+ build)
        //    选择率检查在 hash_join_bloom 内部进行
        //    高选择率时会自动跳过 Bloom，直接使用哈希表
        if (build_count >= BLOOM_MIN_BUILD) {
            return JoinStrategy::BLOOMFILTER;
        }

        // 4. 默认: 自适应分区 (RADIX)
        //    - < 100K: 不分区
        //    - < 500K: 16 分区 (同 v3)
        //    - >= 500K: 256 分区
        return JoinStrategy::RADIX256;
    }

    /**
     * 执行策略
     */
    static size_t execute(JoinStrategy strategy,
                          const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count,
                          JoinType join_type, JoinResult* result,
                          const JoinConfigV4& config) {
        switch (strategy) {
            case JoinStrategy::GPU:
                // 优先使用 v5 GPU (分区优化版)
                if (v4::is_gpu_v5_ready()) {
                    return v4::hash_join_gpu_v5(build_keys, build_count,
                                                 probe_keys, probe_count,
                                                 join_type, result, config);
                }
                // 回退到旧版 GPU
                return v4::hash_join_gpu(build_keys, build_count,
                                          probe_keys, probe_count,
                                          join_type, result, config);

            case JoinStrategy::NPU:
                return v4::hash_join_npu(build_keys, build_count,
                                          probe_keys, probe_count,
                                          join_type, result, config);

            case JoinStrategy::BLOOMFILTER:
                return v4::hash_join_bloom(build_keys, build_count,
                                            probe_keys, probe_count,
                                            join_type, result, config);

            case JoinStrategy::RADIX256:
                return v4::hash_join_radix256(build_keys, build_count,
                                               probe_keys, probe_count,
                                               join_type, result, config);

            case JoinStrategy::V3_FALLBACK:
            case JoinStrategy::AUTO:
            default:
                // 回退到 v3
                return hash_join_i32_v3(build_keys, build_count,
                                         probe_keys, probe_count,
                                         join_type, result);
        }
    }
};

// 策略名称
const char* strategy_names[] = {
    "AUTO",
    "RADIX256",
    "BLOOMFILTER",
    "NPU",
    "GPU",
    "V3_FALLBACK"
};

} // anonymous namespace

// ============================================================================
// 公开接口
// ============================================================================

size_t hash_join_i32_v4(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result) {
    JoinConfigV4 default_config;
    return hash_join_i32_v4_config(build_keys, build_count,
                                    probe_keys, probe_count,
                                    join_type, result,
                                    default_config);
}

size_t hash_join_i32_v4_config(const int32_t* build_keys, size_t build_count,
                                const int32_t* probe_keys, size_t probe_count,
                                JoinType join_type,
                                JoinResult* result,
                                const JoinConfigV4& config) {
    if (build_count == 0 || probe_count == 0) {
        result->count = 0;
        return 0;
    }

    // 确保结果缓冲区
    size_t estimated_matches = std::max(build_count, probe_count) * 4;
    if (result->capacity < estimated_matches) {
        grow_join_result(result, estimated_matches);
    }

    // ========== P0 优化: 首先尝试完美哈希 ==========
    // 完美哈希对连续整数键有巨大优势 (O(1) 直接索引)
    // 这是 v3 快于 v4 的主要原因之一
    PerfectHashTable perfect_ht;
    if (perfect_ht.try_build(build_keys, build_count)) {
        size_t matches = perfect_ht.probe(probe_keys, probe_count,
                                           result->left_indices,
                                           result->right_indices);
        result->count = matches;
        return matches;
    }

    // 选择并执行策略
    JoinStrategy selected = StrategyDispatcher::select_strategy(
        build_count, probe_count, config);

    return StrategyDispatcher::execute(selected,
                                        build_keys, build_count,
                                        probe_keys, probe_count,
                                        join_type, result, config);
}

const char* get_selected_strategy_name(size_t build_count, size_t probe_count,
                                        const JoinConfigV4& config) {
    JoinStrategy selected = StrategyDispatcher::select_strategy(
        build_count, probe_count, config);
    return strategy_names[static_cast<int>(selected)];
}

bool is_strategy_available(JoinStrategy strategy) {
    switch (strategy) {
        case JoinStrategy::AUTO:
        case JoinStrategy::V3_FALLBACK:
        case JoinStrategy::RADIX256:
        case JoinStrategy::BLOOMFILTER:
            return true;  // 总是可用

        case JoinStrategy::NPU:
            return v4::is_npu_ready();

        case JoinStrategy::GPU:
            return v4::is_gpu_ready();

        default:
            return false;
    }
}

} // namespace join
} // namespace thunderduck
