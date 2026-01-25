/**
 * ThunderDuck - Bloom Filter Implementation
 *
 * SIMD 优化的 Bloom Filter
 * - CRC32 双重哈希
 * - ARM Neon 批量操作
 * - 128 字节对齐
 */

#include "thunderduck/bloom_filter.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#endif

namespace thunderduck {
namespace bloom {

namespace {

// 对齐分配
void* aligned_alloc_128(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, BLOOM_CACHE_LINE, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void aligned_free(void* ptr) {
    free(ptr);
}

// 计算最优位数组大小
// m = -n * ln(p) / (ln(2)^2)
size_t optimal_bit_count(size_t n, double fpr) {
    if (n == 0) return 64;
    double ln2_sq = 0.480453013918201;  // ln(2)^2
    double m = -static_cast<double>(n) * std::log(fpr) / ln2_sq;
    // 向上取整到 64 的倍数
    size_t bits = static_cast<size_t>(std::ceil(m));
    return ((bits + 63) / 64) * 64;
}

// 计算最优哈希函数数量
// k = (m/n) * ln(2)
size_t optimal_num_hashes(size_t m, size_t n) {
    if (n == 0) return 1;
    double k = (static_cast<double>(m) / n) * 0.693147180559945;  // ln(2)
    return std::max(1UL, std::min(static_cast<size_t>(std::round(k)), 20UL));
}

} // anonymous namespace

// ============================================================================
// BloomFilter 实现
// ============================================================================

BloomFilter::BloomFilter(size_t expected_elements, double fpr, size_t num_hashes)
    : bits_(nullptr), num_bits_(0), num_words_(0), num_hashes_(num_hashes) {

    // 计算最优参数
    num_bits_ = optimal_bit_count(expected_elements, fpr);
    num_words_ = num_bits_ / 64;

    if (num_hashes == 0) {
        num_hashes_ = optimal_num_hashes(num_bits_, expected_elements);
    }

    // 分配对齐内存
    size_t alloc_size = num_words_ * sizeof(uint64_t);
    bits_ = static_cast<uint64_t*>(aligned_alloc_128(alloc_size));

    if (bits_) {
        std::memset(bits_, 0, alloc_size);
    }
}

BloomFilter::~BloomFilter() {
    if (bits_) {
        aligned_free(bits_);
        bits_ = nullptr;
    }
}

BloomFilter::BloomFilter(BloomFilter&& other) noexcept
    : bits_(other.bits_),
      num_bits_(other.num_bits_),
      num_words_(other.num_words_),
      num_hashes_(other.num_hashes_) {
    other.bits_ = nullptr;
    other.num_bits_ = 0;
    other.num_words_ = 0;
}

BloomFilter& BloomFilter::operator=(BloomFilter&& other) noexcept {
    if (this != &other) {
        if (bits_) {
            aligned_free(bits_);
        }
        bits_ = other.bits_;
        num_bits_ = other.num_bits_;
        num_words_ = other.num_words_;
        num_hashes_ = other.num_hashes_;

        other.bits_ = nullptr;
        other.num_bits_ = 0;
        other.num_words_ = 0;
    }
    return *this;
}

void BloomFilter::compute_hashes(int32_t key, uint32_t* h1, uint32_t* h2) const {
#ifdef __aarch64__
    // CRC32 双重哈希
    *h1 = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
    *h2 = __crc32cw(0x12345678, static_cast<uint32_t>(key));  // 不同种子
#else
    // 备用哈希 (MurmurHash 变种)
    uint32_t k = static_cast<uint32_t>(key);

    // h1
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    *h1 = k;

    // h2 (不同混合)
    k = static_cast<uint32_t>(key);
    k ^= k >> 15;
    k *= 0xcc9e2d51;
    k ^= k >> 13;
    k *= 0x1b873593;
    k ^= k >> 16;
    *h2 = k;
#endif
}

void BloomFilter::set_bit(size_t bit_index) {
    size_t word_idx = bit_index / 64;
    size_t bit_pos = bit_index % 64;
    bits_[word_idx] |= (1ULL << bit_pos);
}

bool BloomFilter::test_bit(size_t bit_index) const {
    size_t word_idx = bit_index / 64;
    size_t bit_pos = bit_index % 64;
    return (bits_[word_idx] & (1ULL << bit_pos)) != 0;
}

void BloomFilter::insert(int32_t key) {
    uint32_t h1, h2;
    compute_hashes(key, &h1, &h2);

    // 双重哈希: h_i(x) = h1(x) + i * h2(x)
    for (size_t i = 0; i < num_hashes_; ++i) {
        size_t bit_index = (h1 + i * h2) % num_bits_;
        set_bit(bit_index);
    }
}

void BloomFilter::insert_batch(const int32_t* keys, size_t count) {
#ifdef __aarch64__
    // 批量插入优化
    size_t i = 0;

    // 每次处理 4 个键
    for (; i + 4 <= count; i += 4) {
        // 计算哈希
        uint32_t h1_0 = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i]));
        uint32_t h1_1 = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 1]));
        uint32_t h1_2 = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 2]));
        uint32_t h1_3 = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[i + 3]));

        uint32_t h2_0 = __crc32cw(0x12345678, static_cast<uint32_t>(keys[i]));
        uint32_t h2_1 = __crc32cw(0x12345678, static_cast<uint32_t>(keys[i + 1]));
        uint32_t h2_2 = __crc32cw(0x12345678, static_cast<uint32_t>(keys[i + 2]));
        uint32_t h2_3 = __crc32cw(0x12345678, static_cast<uint32_t>(keys[i + 3]));

        // 设置所有哈希位
        for (size_t j = 0; j < num_hashes_; ++j) {
            set_bit((h1_0 + j * h2_0) % num_bits_);
            set_bit((h1_1 + j * h2_1) % num_bits_);
            set_bit((h1_2 + j * h2_2) % num_bits_);
            set_bit((h1_3 + j * h2_3) % num_bits_);
        }
    }

    // 处理剩余
    for (; i < count; ++i) {
        insert(keys[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        insert(keys[i]);
    }
#endif
}

bool BloomFilter::maybe_contains(int32_t key) const {
    uint32_t h1, h2;
    compute_hashes(key, &h1, &h2);

    for (size_t i = 0; i < num_hashes_; ++i) {
        size_t bit_index = (h1 + i * h2) % num_bits_;
        if (!test_bit(bit_index)) {
            return false;  // 一定不存在
        }
    }
    return true;  // 可能存在
}

void BloomFilter::query_batch(const int32_t* keys, size_t count, uint8_t* out_results) const {
#ifdef __aarch64__
    // SIMD 批量查询
    size_t i = 0;

    // 每次处理 8 个键
    for (; i + 8 <= count; i += 8) {
        // 预取下一批数据
        if (i + 16 < count) {
            __builtin_prefetch(&keys[i + 16], 0, 3);
        }

        // 逐个检查 (由于位操作复杂，这里用标量)
        for (size_t j = 0; j < 8; ++j) {
            out_results[i + j] = maybe_contains(keys[i + j]) ? 1 : 0;
        }
    }

    // 处理剩余
    for (; i < count; ++i) {
        out_results[i] = maybe_contains(keys[i]) ? 1 : 0;
    }
#else
    for (size_t i = 0; i < count; ++i) {
        out_results[i] = maybe_contains(keys[i]) ? 1 : 0;
    }
#endif
}

size_t BloomFilter::filter_batch(const int32_t* keys, size_t count, uint32_t* out_indices) const {
    size_t pass_count = 0;

#ifdef __aarch64__
    // 批量过滤优化
    size_t i = 0;

    // 每次处理 8 个键
    for (; i + 8 <= count; i += 8) {
        // 预取
        if (i + 16 < count) {
            __builtin_prefetch(&keys[i + 16], 0, 3);
        }

        // 检查并收集通过的索引
        for (size_t j = 0; j < 8; ++j) {
            if (maybe_contains(keys[i + j])) {
                out_indices[pass_count++] = static_cast<uint32_t>(i + j);
            }
        }
    }

    // 处理剩余
    for (; i < count; ++i) {
        if (maybe_contains(keys[i])) {
            out_indices[pass_count++] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        if (maybe_contains(keys[i])) {
            out_indices[pass_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return pass_count;
}

void BloomFilter::clear() {
    if (bits_) {
        std::memset(bits_, 0, num_words_ * sizeof(uint64_t));
    }
}

double BloomFilter::estimated_fpr() const {
    // 计算实际 FPR: (1 - e^(-kn/m))^k
    // 这里简化为理论值
    size_t m = num_bits_;
    size_t k = num_hashes_;
    // 假设 n ≈ m / k (理想情况)
    double n_approx = static_cast<double>(m) / k;
    double exponent = -static_cast<double>(k) * n_approx / m;
    double fpr = std::pow(1.0 - std::exp(exponent), static_cast<double>(k));
    return fpr;
}

// ============================================================================
// 工厂函数
// ============================================================================

BloomFilter* create_bloom_filter(size_t expected_elements, double fpr) {
    return new BloomFilter(expected_elements, fpr);
}

BloomFilter* build_bloom_filter(const int32_t* keys, size_t count, double fpr) {
    auto* filter = new BloomFilter(count, fpr);
    filter->insert_batch(keys, count);
    return filter;
}

void free_bloom_filter(BloomFilter* filter) {
    delete filter;
}

} // namespace bloom
} // namespace thunderduck
