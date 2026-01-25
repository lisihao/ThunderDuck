/**
 * ThunderDuck - Bloom Filter
 *
 * SIMD 优化的 Bloom Filter，用于 Hash Join v4 预过滤
 * - 128 字节对齐位数组
 * - CRC32 双重哈希
 * - ARM Neon 批量检查
 */

#ifndef THUNDERDUCK_BLOOM_FILTER_H
#define THUNDERDUCK_BLOOM_FILTER_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace thunderduck {
namespace bloom {

// ============================================================================
// 配置常量
// ============================================================================

constexpr size_t BLOOM_CACHE_LINE = 128;      // M4 缓存行大小
constexpr size_t DEFAULT_NUM_HASHES = 7;      // 默认哈希函数数量
constexpr double DEFAULT_FPR = 0.01;          // 默认假阳性率 1%

// ============================================================================
// Bloom Filter 实现
// ============================================================================

/**
 * SIMD 优化的 Bloom Filter
 *
 * 使用双重哈希技术: h_i(x) = h1(x) + i * h2(x)
 * 其中 h1 和 h2 使用 CRC32 变种
 */
class BloomFilter {
public:
    /**
     * 构造函数
     * @param expected_elements 预期元素数量
     * @param fpr 目标假阳性率 (默认 1%)
     * @param num_hashes 哈希函数数量 (默认 7)
     */
    explicit BloomFilter(size_t expected_elements,
                         double fpr = DEFAULT_FPR,
                         size_t num_hashes = DEFAULT_NUM_HASHES);

    ~BloomFilter();

    // 禁止拷贝
    BloomFilter(const BloomFilter&) = delete;
    BloomFilter& operator=(const BloomFilter&) = delete;

    // 允许移动
    BloomFilter(BloomFilter&& other) noexcept;
    BloomFilter& operator=(BloomFilter&& other) noexcept;

    /**
     * 插入单个键
     */
    void insert(int32_t key);

    /**
     * 批量插入
     */
    void insert_batch(const int32_t* keys, size_t count);

    /**
     * 查询单个键 (可能存在)
     * @return true 表示可能存在，false 表示一定不存在
     */
    bool maybe_contains(int32_t key) const;

    /**
     * 批量查询 - SIMD 优化
     * @param keys 查询键数组
     * @param count 查询数量
     * @param out_results 输出结果 (1 = 可能存在, 0 = 不存在)
     */
    void query_batch(const int32_t* keys, size_t count, uint8_t* out_results) const;

    /**
     * 批量查询并返回通过的索引
     * @param keys 查询键数组
     * @param count 查询数量
     * @param out_indices 输出通过过滤的索引
     * @return 通过过滤的数量
     */
    size_t filter_batch(const int32_t* keys, size_t count, uint32_t* out_indices) const;

    /**
     * 重置 Bloom Filter
     */
    void clear();

    /**
     * 获取统计信息
     */
    size_t bit_count() const { return num_bits_; }
    size_t num_hashes() const { return num_hashes_; }
    size_t memory_usage() const { return (num_bits_ + 7) / 8; }
    double estimated_fpr() const;

private:
    // 128 字节对齐的位数组
    uint64_t* bits_;
    size_t num_bits_;
    size_t num_words_;  // 64-bit words
    size_t num_hashes_;

    // 哈希函数
    void compute_hashes(int32_t key, uint32_t* h1, uint32_t* h2) const;
    void set_bit(size_t bit_index);
    bool test_bit(size_t bit_index) const;
};

// ============================================================================
// 工厂函数
// ============================================================================

/**
 * 根据预期元素数量和 FPR 创建 Bloom Filter
 */
BloomFilter* create_bloom_filter(size_t expected_elements,
                                  double fpr = DEFAULT_FPR);

/**
 * 从 build 侧键构建 Bloom Filter
 */
BloomFilter* build_bloom_filter(const int32_t* keys, size_t count,
                                 double fpr = DEFAULT_FPR);

/**
 * 释放 Bloom Filter
 */
void free_bloom_filter(BloomFilter* filter);

} // namespace bloom
} // namespace thunderduck

#endif // THUNDERDUCK_BLOOM_FILTER_H
