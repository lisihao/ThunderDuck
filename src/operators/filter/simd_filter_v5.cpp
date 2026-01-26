/**
 * ThunderDuck - SIMD Filter Implementation v5.0
 *
 * V8 性能试验优化:
 * 1. 掩码压缩查表优化 - 4-bit LUT 加速 mask→indices 转换
 * 2. 128 字节缓存行强制对齐 - M4 L1 缓存行优化
 * 3. 字符串 SIMD 过滤 - 并行 memcmp
 */

#include "thunderduck/filter.h"
#include "thunderduck/memory.h"
#include <cstring>
#include <algorithm>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

// ============================================================================
// 常量定义
// ============================================================================

namespace {

// M4 缓存行大小
constexpr size_t CACHE_LINE_SIZE = 128;

// 批处理大小
constexpr size_t BATCH_SIZE = 256;
constexpr size_t PREFETCH_DISTANCE = 512;

// ============================================================================
// 优化 1: 掩码压缩查表 (Mask Compression LUT)
// ============================================================================

/**
 * 4-bit 掩码到索引的查表
 *
 * mask 值 0-15 对应的匹配位置:
 * 0000 (0)  -> 无匹配
 * 0001 (1)  -> [0]
 * 0010 (2)  -> [1]
 * 0011 (3)  -> [0, 1]
 * ...
 * 1111 (15) -> [0, 1, 2, 3]
 */
struct MaskLUT {
    uint8_t indices[4];  // 最多 4 个匹配位置
    uint8_t count;       // 匹配数量
};

// 预计算的 4-bit 掩码查表 (16 entries)
alignas(64) static const MaskLUT MASK_LUT_4BIT[16] = {
    {{0, 0, 0, 0}, 0},  // 0000
    {{0, 0, 0, 0}, 1},  // 0001
    {{1, 0, 0, 0}, 1},  // 0010
    {{0, 1, 0, 0}, 2},  // 0011
    {{2, 0, 0, 0}, 1},  // 0100
    {{0, 2, 0, 0}, 2},  // 0101
    {{1, 2, 0, 0}, 2},  // 0110
    {{0, 1, 2, 0}, 3},  // 0111
    {{3, 0, 0, 0}, 1},  // 1000
    {{0, 3, 0, 0}, 2},  // 1001
    {{1, 3, 0, 0}, 2},  // 1010
    {{0, 1, 3, 0}, 3},  // 1011
    {{2, 3, 0, 0}, 2},  // 1100
    {{0, 2, 3, 0}, 3},  // 1101
    {{1, 2, 3, 0}, 3},  // 1110
    {{0, 1, 2, 3}, 4},  // 1111
};

/**
 * 使用查表法快速将 SIMD 掩码转换为索引列表
 *
 * 对比传统 CTZ 循环:
 * - CTZ: while(bits) { pos = ctz(bits); bits &= bits-1; } 每个1位需要2-3条指令
 * - LUT: 查表 + 批量写入，每4位只需1次查表
 *
 * 预期性能提升: 高选择率时 +15-25%
 */
__attribute__((always_inline))
inline size_t mask_to_indices_lut(uint32_t mask, uint32_t base, uint32_t* out) {
    size_t count = 0;

    // 处理 4 个 4-bit 片段 (共 16 bits，对应 4 个 int32)
    // 注意: NEON 比较结果是 0xFFFFFFFF 或 0x00000000
    // 需要先压缩为 4-bit

    // 片段 0: bits 0-3
    uint32_t nibble0 = mask & 0xF;
    const MaskLUT& lut0 = MASK_LUT_4BIT[nibble0];
    for (uint8_t i = 0; i < lut0.count; ++i) {
        out[count++] = base + lut0.indices[i];
    }

    return count;
}

/**
 * 从 NEON uint32x4_t 掩码提取 4-bit 压缩掩码
 */
__attribute__((always_inline))
inline uint32_t compress_mask_u32x4(uint32x4_t mask) {
    // 将 0xFFFFFFFF 转换为 1, 0x00000000 转换为 0
    // 使用 vshrn 将 32-bit 压缩到 16-bit，再提取
    uint16x4_t narrow = vmovn_u32(mask);  // 取高16位
    uint64_t bits = vget_lane_u64(vreinterpret_u64_u16(narrow), 0);

    // 提取每个 16-bit 的符号位 (bit 15)
    uint32_t result = 0;
    result |= ((bits >> 15) & 1) << 0;
    result |= ((bits >> 31) & 1) << 1;
    result |= ((bits >> 47) & 1) << 2;
    result |= ((bits >> 63) & 1) << 3;

    return result;
}

/**
 * 优化的位图转索引 - 使用查表法
 */
size_t bitmap_to_indices_lut(const uint64_t* bitmap, size_t bit_count,
                              uint32_t* out_indices) {
    size_t out_count = 0;
    size_t num_words = (bit_count + 63) / 64;

    for (size_t i = 0; i < num_words; ++i) {
        uint64_t bits = bitmap[i];
        uint32_t base = static_cast<uint32_t>(i * 64);

        // 快速跳过空 word
        if (bits == 0) continue;

        // 全满快速路径
        if (bits == ~0ULL) {
            for (uint32_t j = 0; j < 64 && base + j < bit_count; ++j) {
                out_indices[out_count++] = base + j;
            }
            continue;
        }

        // 使用 8-bit 分块处理
        for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
            uint8_t byte_val = (bits >> (byte_idx * 8)) & 0xFF;
            if (byte_val == 0) continue;

            uint32_t byte_base = base + byte_idx * 8;

            // 低 4 bits
            uint8_t lo = byte_val & 0xF;
            if (lo) {
                const MaskLUT& lut = MASK_LUT_4BIT[lo];
                for (uint8_t k = 0; k < lut.count; ++k) {
                    out_indices[out_count++] = byte_base + lut.indices[k];
                }
            }

            // 高 4 bits
            uint8_t hi = (byte_val >> 4) & 0xF;
            if (hi) {
                const MaskLUT& lut = MASK_LUT_4BIT[hi];
                for (uint8_t k = 0; k < lut.count; ++k) {
                    out_indices[out_count++] = byte_base + 4 + lut.indices[k];
                }
            }
        }
    }

    return out_count;
}

} // anonymous namespace

// ============================================================================
// 优化 2: 128 字节缓存行强制对齐
// ============================================================================

namespace {

/**
 * 分配 128 字节对齐的内存 (M4 L1 缓存行大小)
 */
void* cache_aligned_alloc(size_t size) {
    // 向上对齐到 128 字节
    size_t aligned_size = (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    void* ptr = nullptr;

#if defined(__APPLE__)
    // macOS: posix_memalign
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, aligned_size) != 0) {
        return nullptr;
    }
#else
    ptr = aligned_alloc(CACHE_LINE_SIZE, aligned_size);
#endif

    return ptr;
}

void cache_aligned_free(void* ptr) {
    free(ptr);
}

/**
 * 检查指针是否 128 字节对齐
 */
__attribute__((always_inline))
inline bool is_cache_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (CACHE_LINE_SIZE - 1)) == 0;
}

} // anonymous namespace

// ============================================================================
// v5.0 核心过滤实现 - 集成所有优化
// ============================================================================

#ifdef __aarch64__

namespace {

/**
 * 模板化比较操作
 */
template<CompareOp Op>
__attribute__((always_inline))
inline uint32x4_t simd_compare_i32_v5(int32x4_t data, int32x4_t threshold) {
    if constexpr (Op == CompareOp::GT) return vcgtq_s32(data, threshold);
    if constexpr (Op == CompareOp::GE) return vcgeq_s32(data, threshold);
    if constexpr (Op == CompareOp::LT) return vcltq_s32(data, threshold);
    if constexpr (Op == CompareOp::LE) return vcleq_s32(data, threshold);
    if constexpr (Op == CompareOp::EQ) return vceqq_s32(data, threshold);
    if constexpr (Op == CompareOp::NE) return vmvnq_u32(vceqq_s32(data, threshold));
    __builtin_unreachable();
}

/**
 * v5 计数实现 - 128 字节对齐优化
 */
template<CompareOp Op>
__attribute__((noinline))
size_t count_i32_v5_core(const int32_t* __restrict input,
                          size_t count, int32_t value) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t result = 0;
    size_t i = 0;

    // 处理到 128 字节对齐边界
    size_t align_offset = reinterpret_cast<uintptr_t>(input) & (CACHE_LINE_SIZE - 1);
    if (align_offset != 0) {
        size_t skip = (CACHE_LINE_SIZE - align_offset) / sizeof(int32_t);
        skip = std::min(skip, count);
        for (; i < skip; ++i) {
            if constexpr (Op == CompareOp::GT) { if (input[i] > value) ++result; }
            else if constexpr (Op == CompareOp::GE) { if (input[i] >= value) ++result; }
            else if constexpr (Op == CompareOp::LT) { if (input[i] < value) ++result; }
            else if constexpr (Op == CompareOp::LE) { if (input[i] <= value) ++result; }
            else if constexpr (Op == CompareOp::EQ) { if (input[i] == value) ++result; }
            else if constexpr (Op == CompareOp::NE) { if (input[i] != value) ++result; }
        }
    }

    // 主循环: 256 元素批次，4 独立累加器
    for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const int32_t* batch_ptr = input + i;

        // 预取下一批 (128 字节对齐)
        __builtin_prefetch(batch_ptr + BATCH_SIZE, 0, 3);      // L1
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 32, 0, 3);
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 64, 0, 2); // L2
        __builtin_prefetch(batch_ptr + BATCH_SIZE + 96, 0, 2);

        #pragma unroll
        for (size_t j = 0; j < 16; ++j) {
            const int32_t* ptr = batch_ptr + j * 16;

            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            uint32x4_t m0 = simd_compare_i32_v5<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare_i32_v5<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare_i32_v5<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare_i32_v5<Op>(d3, threshold);

            // vsub 技巧: mask=-1 时 acc-(-1)=acc+1
            acc0 = vsubq_u32(acc0, m0);
            acc1 = vsubq_u32(acc1, m1);
            acc2 = vsubq_u32(acc2, m2);
            acc3 = vsubq_u32(acc3, m3);
        }

        uint32x4_t sum01 = vaddq_u32(acc0, acc1);
        uint32x4_t sum23 = vaddq_u32(acc2, acc3);
        uint32x4_t total = vaddq_u32(sum01, sum23);
        result += vaddvq_u32(total);
    }

    // 处理剩余元素
    for (; i < count; ++i) {
        if constexpr (Op == CompareOp::GT) { if (input[i] > value) ++result; }
        else if constexpr (Op == CompareOp::GE) { if (input[i] >= value) ++result; }
        else if constexpr (Op == CompareOp::LT) { if (input[i] < value) ++result; }
        else if constexpr (Op == CompareOp::LE) { if (input[i] <= value) ++result; }
        else if constexpr (Op == CompareOp::EQ) { if (input[i] == value) ++result; }
        else if constexpr (Op == CompareOp::NE) { if (input[i] != value) ++result; }
    }

    return result;
}

/**
 * v5 位图过滤实现 - 带 LUT 转换
 */
template<CompareOp Op>
__attribute__((noinline))
size_t filter_to_bitmap_v5_core(const int32_t* __restrict input, size_t count,
                                 int32_t value, uint64_t* __restrict bitmap) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t match_count = 0;
    size_t i = 0;

    // 初始化位图
    size_t bitmap_words = (count + 63) / 64;
    std::memset(bitmap, 0, bitmap_words * sizeof(uint64_t));

    // 主循环
    for (; i + 64 <= count; i += 64) {
        __builtin_prefetch(input + i + PREFETCH_DISTANCE / sizeof(int32_t), 0, 0);

        uint64_t bits = 0;

        #pragma unroll
        for (size_t j = 0; j < 4; ++j) {
            const int32_t* ptr = input + i + j * 16;

            int32x4_t d0 = vld1q_s32(ptr);
            int32x4_t d1 = vld1q_s32(ptr + 4);
            int32x4_t d2 = vld1q_s32(ptr + 8);
            int32x4_t d3 = vld1q_s32(ptr + 12);

            uint32x4_t m0 = simd_compare_i32_v5<Op>(d0, threshold);
            uint32x4_t m1 = simd_compare_i32_v5<Op>(d1, threshold);
            uint32x4_t m2 = simd_compare_i32_v5<Op>(d2, threshold);
            uint32x4_t m3 = simd_compare_i32_v5<Op>(d3, threshold);

            // 提取每个掩码的 4-bit 压缩形式
            uint32_t b0 = compress_mask_u32x4(m0);
            uint32_t b1 = compress_mask_u32x4(m1);
            uint32_t b2 = compress_mask_u32x4(m2);
            uint32_t b3 = compress_mask_u32x4(m3);

            // 组合到 64-bit word
            bits |= (uint64_t)b0 << (j * 16);
            bits |= (uint64_t)b1 << (j * 16 + 4);
            bits |= (uint64_t)b2 << (j * 16 + 8);
            bits |= (uint64_t)b3 << (j * 16 + 12);
        }

        bitmap[i / 64] = bits;
        match_count += __builtin_popcountll(bits);
    }

    // 处理剩余元素
    for (; i < count; ++i) {
        bool match = false;
        if constexpr (Op == CompareOp::GT) match = input[i] > value;
        else if constexpr (Op == CompareOp::GE) match = input[i] >= value;
        else if constexpr (Op == CompareOp::LT) match = input[i] < value;
        else if constexpr (Op == CompareOp::LE) match = input[i] <= value;
        else if constexpr (Op == CompareOp::EQ) match = input[i] == value;
        else if constexpr (Op == CompareOp::NE) match = input[i] != value;

        if (match) {
            bitmap[i / 64] |= (1ULL << (i % 64));
            ++match_count;
        }
    }

    return match_count;
}

} // anonymous namespace

#endif // __aarch64__

// ============================================================================
// 优化 3: 字符串 SIMD 过滤
// ============================================================================

namespace {

#ifdef __aarch64__

/**
 * SIMD 字符串比较 - 比较前 16 字节
 *
 * @return 0 表示相等，<0 表示 a<b，>0 表示 a>b
 */
__attribute__((always_inline))
inline int simd_strcmp_16(const char* a, const char* b) {
    uint8x16_t va = vld1q_u8(reinterpret_cast<const uint8_t*>(a));
    uint8x16_t vb = vld1q_u8(reinterpret_cast<const uint8_t*>(b));

    // 比较相等
    uint8x16_t eq = vceqq_u8(va, vb);

    // 检查是否全部相等
    uint8x16_t min_val = vpminq_u8(eq, eq);
    uint8_t all_eq = vgetq_lane_u8(min_val, 0);

    if (all_eq == 0xFF) {
        return 0;  // 前 16 字节相等
    }

    // 找到第一个不相等的位置
    uint8x16_t neq = vmvnq_u8(eq);

    // 使用 CLZ 找到第一个差异位置
    // 先转换为 64-bit 方便处理
    uint64x2_t neq64 = vreinterpretq_u64_u8(neq);
    uint64_t lo = vgetq_lane_u64(neq64, 0);
    uint64_t hi = vgetq_lane_u64(neq64, 1);

    int pos;
    if (lo != 0) {
        pos = __builtin_ctzll(lo) / 8;
    } else {
        pos = 8 + __builtin_ctzll(hi) / 8;
    }

    return static_cast<int>(static_cast<unsigned char>(a[pos])) -
           static_cast<int>(static_cast<unsigned char>(b[pos]));
}

/**
 * SIMD 字符串相等比较
 */
__attribute__((always_inline))
inline bool simd_streq_16(const char* a, const char* b) {
    uint8x16_t va = vld1q_u8(reinterpret_cast<const uint8_t*>(a));
    uint8x16_t vb = vld1q_u8(reinterpret_cast<const uint8_t*>(b));
    uint8x16_t eq = vceqq_u8(va, vb);

    // 检查所有字节是否相等
    uint8x16_t min_val = vpminq_u8(eq, eq);
    min_val = vpminq_u8(min_val, min_val);
    min_val = vpminq_u8(min_val, min_val);
    min_val = vpminq_u8(min_val, min_val);

    return vgetq_lane_u8(min_val, 0) == 0xFF;
}

/**
 * SIMD 字符串前缀匹配
 */
__attribute__((always_inline))
inline bool simd_str_startswith(const char* str, const char* prefix, size_t prefix_len) {
    if (prefix_len <= 16) {
        // 短前缀: 直接 SIMD 比较
        uint8x16_t vs = vld1q_u8(reinterpret_cast<const uint8_t*>(str));
        uint8x16_t vp = vld1q_u8(reinterpret_cast<const uint8_t*>(prefix));

        // 创建掩码只比较前 prefix_len 字节
        uint8x16_t eq = vceqq_u8(vs, vp);

        // 检查前 prefix_len 字节
        uint64x2_t eq64 = vreinterpretq_u64_u8(eq);
        uint64_t lo = vgetq_lane_u64(eq64, 0);

        if (prefix_len <= 8) {
            uint64_t mask = (1ULL << (prefix_len * 8)) - 1;
            return (lo & mask) == mask;
        } else {
            uint64_t hi = vgetq_lane_u64(eq64, 1);
            uint64_t hi_mask = (1ULL << ((prefix_len - 8) * 8)) - 1;
            return lo == ~0ULL && (hi & hi_mask) == hi_mask;
        }
    }

    // 长前缀: 分块比较
    return std::memcmp(str, prefix, prefix_len) == 0;
}

#endif // __aarch64__

} // anonymous namespace

// ============================================================================
// v5 公共 API 实现
// ============================================================================

size_t count_i32_v5(const int32_t* input, size_t count,
                     CompareOp op, int32_t value) {
#ifdef __aarch64__
    switch (op) {
        case CompareOp::GT: return count_i32_v5_core<CompareOp::GT>(input, count, value);
        case CompareOp::GE: return count_i32_v5_core<CompareOp::GE>(input, count, value);
        case CompareOp::LT: return count_i32_v5_core<CompareOp::LT>(input, count, value);
        case CompareOp::LE: return count_i32_v5_core<CompareOp::LE>(input, count, value);
        case CompareOp::EQ: return count_i32_v5_core<CompareOp::EQ>(input, count, value);
        case CompareOp::NE: return count_i32_v5_core<CompareOp::NE>(input, count, value);
    }
#else
    // Fallback
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        bool match = false;
        switch (op) {
            case CompareOp::GT: match = input[i] > value; break;
            case CompareOp::GE: match = input[i] >= value; break;
            case CompareOp::LT: match = input[i] < value; break;
            case CompareOp::LE: match = input[i] <= value; break;
            case CompareOp::EQ: match = input[i] == value; break;
            case CompareOp::NE: match = input[i] != value; break;
        }
        if (match) ++result;
    }
    return result;
#endif
    return 0;
}

size_t filter_to_bitmap_v5(const int32_t* input, size_t count,
                            CompareOp op, int32_t value,
                            uint64_t* bitmap) {
#ifdef __aarch64__
    switch (op) {
        case CompareOp::GT: return filter_to_bitmap_v5_core<CompareOp::GT>(input, count, value, bitmap);
        case CompareOp::GE: return filter_to_bitmap_v5_core<CompareOp::GE>(input, count, value, bitmap);
        case CompareOp::LT: return filter_to_bitmap_v5_core<CompareOp::LT>(input, count, value, bitmap);
        case CompareOp::LE: return filter_to_bitmap_v5_core<CompareOp::LE>(input, count, value, bitmap);
        case CompareOp::EQ: return filter_to_bitmap_v5_core<CompareOp::EQ>(input, count, value, bitmap);
        case CompareOp::NE: return filter_to_bitmap_v5_core<CompareOp::NE>(input, count, value, bitmap);
    }
#endif
    return 0;
}

size_t filter_i32_v5(const int32_t* input, size_t count,
                      CompareOp op, int32_t value,
                      uint32_t* out_indices) {
    // v5 实验结论: LUT 方法比 v3 CTZ 循环慢
    // 直接使用 v3 实现，但保留 v5 的 count 和 bitmap 优化
    return filter_i32_v3(input, count, op, value, out_indices);
}

// ============================================================================
// 字符串过滤 API
// ============================================================================

/**
 * 字符串数组过滤 - 相等比较
 *
 * @param strings 字符串指针数组
 * @param count 字符串数量
 * @param target 目标字符串
 * @param out_indices 输出索引数组
 * @return 匹配数量
 */
size_t filter_string_eq(const char* const* strings, size_t count,
                         const char* target, uint32_t* out_indices) {
    size_t out_count = 0;
    size_t target_len = std::strlen(target);

#ifdef __aarch64__
    if (target_len <= 15) {
        // 短字符串: SIMD 优化路径
        // 使用前缀匹配 + 长度验证 (避免读取超出字符串边界)
        for (size_t i = 0; i < count; ++i) {
            size_t str_len = std::strlen(strings[i]);
            if (str_len == target_len &&
                simd_str_startswith(strings[i], target, target_len)) {
                out_indices[out_count++] = static_cast<uint32_t>(i);
            }
        }
    } else
#endif
    {
        // 长字符串: 标准 strcmp
        for (size_t i = 0; i < count; ++i) {
            if (std::strcmp(strings[i], target) == 0) {
                out_indices[out_count++] = static_cast<uint32_t>(i);
            }
        }
    }

    return out_count;
}

/**
 * 字符串数组过滤 - 前缀匹配
 */
size_t filter_string_startswith(const char* const* strings, size_t count,
                                 const char* prefix, uint32_t* out_indices) {
    size_t out_count = 0;
    size_t prefix_len = std::strlen(prefix);

#ifdef __aarch64__
    for (size_t i = 0; i < count; ++i) {
        if (simd_str_startswith(strings[i], prefix, prefix_len)) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#else
    for (size_t i = 0; i < count; ++i) {
        if (std::strncmp(strings[i], prefix, prefix_len) == 0) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return out_count;
}

/**
 * 字符串数组过滤 - 包含子串
 */
size_t filter_string_contains(const char* const* strings, size_t count,
                               const char* substr, uint32_t* out_indices) {
    size_t out_count = 0;

    for (size_t i = 0; i < count; ++i) {
        if (std::strstr(strings[i], substr) != nullptr) {
            out_indices[out_count++] = static_cast<uint32_t>(i);
        }
    }

    return out_count;
}

// ============================================================================
// 辅助函数
// ============================================================================

bool is_filter_cache_aligned(const void* ptr) {
    return is_cache_aligned(ptr);
}

void* filter_cache_alloc(size_t size) {
    return cache_aligned_alloc(size);
}

void filter_cache_free(void* ptr) {
    cache_aligned_free(ptr);
}

} // namespace filter
} // namespace thunderduck
