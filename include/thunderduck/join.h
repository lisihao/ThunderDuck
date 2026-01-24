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
 * 扩展结果容量
 */
void grow_join_result(JoinResult* result, size_t min_capacity);

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

} // namespace join
} // namespace thunderduck

#endif // THUNDERDUCK_JOIN_H
