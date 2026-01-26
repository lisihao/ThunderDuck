/**
 * ThunderDuck - Join Operator
 * 
 * SIMD 加速的连接算子，支持 Hash Join
 */

#ifndef THUNDERDUCK_JOIN_H
#define THUNDERDUCK_JOIN_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace join {

// ============================================================================
// 连接类型
// ============================================================================

enum class JoinType {
    INNER,      // 内连接
    LEFT,       // 左连接
    RIGHT,      // 右连接
    FULL,       // 全连接
    SEMI,       // 半连接（只返回左表匹配的行）
    ANTI        // 反连接（只返回左表不匹配的行）
};

// ============================================================================
// 连接结果
// ============================================================================

struct JoinResult {
    uint32_t* left_indices;     // 左表匹配索引
    uint32_t* right_indices;    // 右表匹配索引
    size_t count;               // 匹配数量
    size_t capacity;            // 已分配容量
};

// ============================================================================
// 哈希函数
// ============================================================================

/**
 * SIMD 批量哈希计算 - int32 键
 */
void hash_i32(const int32_t* keys, uint32_t* hashes, size_t count);

/**
 * SIMD 批量哈希计算 - int64 键
 */
void hash_i64(const int64_t* keys, uint64_t* hashes, size_t count);

/**
 * 单个键哈希
 */
inline uint32_t hash_one_i32(int32_t key);
inline uint64_t hash_one_i64(int64_t key);

// ============================================================================
// 哈希表
// ============================================================================

/**
 * 缓存友好的哈希表（用于 Hash Join 的 build 侧）
 */
class HashTable {
public:
    explicit HashTable(size_t expected_size = 1024);
    ~HashTable();
    
    // 禁止拷贝
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;
    
    // 允许移动
    HashTable(HashTable&& other) noexcept;
    HashTable& operator=(HashTable&& other) noexcept;
    
    /**
     * 构建哈希表
     * @param keys 键数组
     * @param count 元素数量
     */
    void build_i32(const int32_t* keys, size_t count);
    void build_i64(const int64_t* keys, size_t count);
    
    /**
     * 探测哈希表
     * @param probe_keys 探测键数组
     * @param probe_count 探测数量
     * @param out_build_indices 匹配的 build 侧索引
     * @param out_probe_indices 匹配的 probe 侧索引
     * @return 匹配数量
     */
    size_t probe_i32(const int32_t* probe_keys, size_t probe_count,
                     uint32_t* out_build_indices, uint32_t* out_probe_indices) const;
    
    size_t probe_i64(const int64_t* probe_keys, size_t probe_count,
                     uint32_t* out_build_indices, uint32_t* out_probe_indices) const;
    
    /**
     * 获取统计信息
     */
    size_t size() const;
    size_t bucket_count() const;
    float load_factor() const;

    /**
     * 重置
     */
    void clear();

    /**
     * v5: 只计数不写入 (两阶段算法 Phase 1)
     */
    size_t count_matches_i32(const int32_t* probe_keys, size_t probe_count) const;
    size_t count_matches_i64(const int64_t* probe_keys, size_t probe_count) const;

private:
    struct Impl;
    Impl* impl_;
};

// ============================================================================
// Hash Join 实现
// ============================================================================

/**
 * Hash Join - int32 键
 * 
 * @param build_keys 构建侧键数组
 * @param build_count 构建侧数量
 * @param probe_keys 探测侧键数组
 * @param probe_count 探测侧数量
 * @param join_type 连接类型
 * @param result 输出结果（需预分配或使用动态分配）
 * @return 匹配数量
 */
size_t hash_join_i32(const int32_t* build_keys, size_t build_count,
                     const int32_t* probe_keys, size_t probe_count,
                     JoinType join_type,
                     JoinResult* result);

/**
 * Hash Join - int64 键
 */
size_t hash_join_i64(const int64_t* build_keys, size_t build_count,
                     const int64_t* probe_keys, size_t probe_count,
                     JoinType join_type,
                     JoinResult* result);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 创建连接结果
 */
JoinResult* create_join_result(size_t initial_capacity);

/**
 * 释放连接结果
 */
void free_join_result(JoinResult* result);

/**
 * 扩展结果容量 (动态 2x 增长)
 */
void grow_join_result(JoinResult* result, size_t min_capacity);

/**
 * v5: 精确分配容量 (不拷贝旧数据)
 */
void ensure_join_result_capacity(JoinResult* result, size_t exact_capacity);

// ============================================================================
// SIMD 优化的键比较
// ============================================================================

/**
 * SIMD 批量比较 - 用于哈希表探测
 * 在候选列表中查找匹配的键
 * 
 * @param candidates 候选键数组
 * @param candidate_count 候选数量
 * @param probe_key 探测键
 * @param out_matches 匹配的候选索引
 * @return 匹配数量
 */
size_t simd_find_matches_i32(const int32_t* candidates, size_t candidate_count,
                             int32_t probe_key, uint32_t* out_matches);

size_t simd_find_matches_i64(const int64_t* candidates, size_t candidate_count,
                             int64_t probe_key, uint32_t* out_matches);

// ============================================================================
// v5.0 两阶段算法 - 消除 grow_join_result 瓶颈
// ============================================================================

/**
 * v5 两阶段 Hash Join
 *
 * 优化原理:
 * - Phase 1: 计数遍历，统计总匹配数
 * - Phase 2: 一次性精确分配，填充结果
 *
 * 优势:
 * - 消除动态扩容的 O(n) memcpy
 * - 高匹配场景性能提升 40%+
 */
size_t hash_join_i32_v5(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

size_t hash_join_i64_v5(const int64_t* build_keys, size_t build_count,
                         const int64_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

// ============================================================================
// v2.0 优化版本 - Robin Hood Hash Join
// ============================================================================

/**
 * 优化版 Hash Join - Robin Hood 哈希表 + 批量预取
 */
size_t hash_join_i32_v2(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

/**
 * 分区 Hash Join - 用于超大表
 */
size_t partitioned_hash_join_i32(const int32_t* build_keys, size_t build_count,
                                  const int32_t* probe_keys, size_t probe_count,
                                  JoinType join_type,
                                  JoinResult* result);

// ============================================================================
// v3.0 优化版本 - 针对 Apple M4 深度优化
// ============================================================================

/**
 * Join 配置选项
 */
struct JoinConfig {
    size_t num_threads = 4;           // 并行线程数 (默认使用 P-cores)
    size_t morsel_size = 2048;        // morsel 大小
    int radix_bits = 4;               // 分区位数 (16 分区)
    bool enable_perfect_hash = true;  // 启用完美哈希检测
    bool enable_prefetch = true;      // 启用软件预取
};

/**
 * v3.0 优化版 Hash Join
 *
 * 优化特性:
 * - SOA 哈希表布局 (128-byte 缓存行优化)
 * - SIMD 批量探测 (ARM Neon)
 * - Radix Partitioning (L1/L2 缓存友好)
 * - 完美哈希优化 (小整数键 O(1) 探测)
 * - Morsel-driven 多核并行
 *
 * @param build_keys 构建侧键数组
 * @param build_count 构建侧数量
 * @param probe_keys 探测侧键数组
 * @param probe_count 探测侧数量
 * @param join_type 连接类型
 * @param result 输出结果
 * @return 匹配数量
 */
size_t hash_join_i32_v3(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

/**
 * v3.0 带配置版本
 */
size_t hash_join_i32_v3_config(const int32_t* build_keys, size_t build_count,
                                const int32_t* probe_keys, size_t probe_count,
                                JoinType join_type,
                                JoinResult* result,
                                const JoinConfig& config);

// ============================================================================
// v4.0 优化版本 - 多策略自适应 Hash Join
// ============================================================================

/**
 * v4.0 策略选择
 */
enum class JoinStrategy {
    AUTO,           // 自动选择最优策略
    RADIX256,       // 256 分区 (8-bit, L1 友好)
    BLOOMFILTER,    // CPU Bloom 预过滤
    NPU,            // NPU (BNNS) 加速 Bloom
    GPU,            // Metal GPU 并行
    V3_FALLBACK     // 回退到 v3
};

/**
 * v4.0 配置选项
 */
struct JoinConfigV4 {
    // 策略选择
    JoinStrategy strategy = JoinStrategy::AUTO;

    // 线程控制
    size_t num_threads = 4;           // 并行线程数

    // RADIX256 参数
    int radix_bits = 8;               // 256 分区 (8 bits)

    // Bloom Filter 参数
    double bloom_fpr = 0.01;          // 目标假阳性率 (1%)
    size_t bloom_num_hashes = 7;      // 哈希函数数量

    // NPU/GPU 控制
    bool fallback_to_cpu = true;      // NPU/GPU 不可用时回退到 CPU
    size_t gpu_min_probe_count = 1000000;  // GPU 策略最小 probe 数量
    size_t npu_min_build_count = 500000;   // NPU 策略最小 build 数量

    // 预取优化
    bool enable_prefetch = true;
};

/**
 * v4.0 策略基类 (内部使用)
 */
class JoinStrategyBase {
public:
    virtual ~JoinStrategyBase() = default;

    virtual size_t execute(
        const int32_t* build_keys, size_t build_count,
        const int32_t* probe_keys, size_t probe_count,
        JoinType join_type, JoinResult* result) = 0;

    virtual const char* name() const = 0;
};

/**
 * v4.0 优化版 Hash Join
 *
 * 优化特性:
 * - 多策略自适应选择
 * - RADIX256: 256 分区 (8-bit) L1 缓存优化
 * - BloomFilter: CPU Bloom 预过滤减少探测
 * - NPU: BNNS 加速 Bloom 计算
 * - GPU: Metal 并行探测
 *
 * 策略回退链: GPU → NPU → BLOOMFILTER → RADIX256 → V3
 *
 * @param build_keys 构建侧键数组
 * @param build_count 构建侧数量
 * @param probe_keys 探测侧键数组
 * @param probe_count 探测侧数量
 * @param join_type 连接类型
 * @param result 输出结果
 * @return 匹配数量
 */
size_t hash_join_i32_v4(const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         JoinType join_type,
                         JoinResult* result);

/**
 * v4.0 带配置版本
 */
size_t hash_join_i32_v4_config(const int32_t* build_keys, size_t build_count,
                                const int32_t* probe_keys, size_t probe_count,
                                JoinType join_type,
                                JoinResult* result,
                                const JoinConfigV4& config);

/**
 * 获取当前策略名称 (用于调试/日志)
 */
const char* get_selected_strategy_name(size_t build_count, size_t probe_count,
                                        const JoinConfigV4& config);

/**
 * 检查策略是否可用
 */
bool is_strategy_available(JoinStrategy strategy);

// ============================================================================
// v5.0 GPU 优化版本 (内部使用)
// ============================================================================

namespace v4 {

/**
 * v5 GPU Join (基于 SIGMOD'25 GFTR 模式)
 *
 * 优化特性:
 * - Radix 分区实现顺序内存访问
 * - Threadgroup memory 缓存
 * - SIMD prefix sum 批量结果收集
 */
size_t hash_join_gpu_v5(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

/**
 * 检查 v5 GPU 是否可用
 */
bool is_gpu_v5_ready();

} // namespace v4

// ============================================================================
// UMA 优化版本 - 真正的零拷贝
// ============================================================================

namespace uma {

// 前向声明
struct JoinResultUMA;

/**
 * UMA 优化的 GPU Hash Join
 *
 * 特点:
 * - 尝试零拷贝包装输入数据 (需要页对齐)
 * - 使用缓冲区池减少分配开销
 * - 流水线化 kernel 执行
 * - Shared Events 减少同步
 */
size_t hash_join_gpu_uma(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result,
    const JoinConfigV4& config);

/**
 * 完全零拷贝版本 (使用 JoinResultUMA)
 *
 * 结果直接写入 UMA 缓冲区，无需任何拷贝
 */
size_t hash_join_gpu_uma_zerocopy(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type,
    JoinResultUMA* result);

/**
 * 检查 UMA GPU 是否可用
 */
bool is_uma_gpu_ready();

} // namespace uma

} // namespace join
} // namespace thunderduck

#endif // THUNDERDUCK_JOIN_H
