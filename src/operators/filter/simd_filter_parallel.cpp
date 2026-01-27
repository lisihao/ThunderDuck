/**
 * ThunderDuck - Parallel SIMD Filter Implementation
 *
 * 多线程并行 Filter 优化:
 * - 4 线程并行 (M4 性能核)
 * - 每线程独立输出缓冲区
 * - 最终合并结果
 *
 * 目标: 10M 数据 2.6x → 4-5x
 */

#include "thunderduck/filter.h"
#include "thunderduck/memory.h"
#include <vector>
#include <thread>
#include <algorithm>
#include <cstring>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace thunderduck {
namespace filter {

// ============================================================================
// 配置常量
// ============================================================================

namespace {

constexpr size_t MIN_ELEMENTS_PER_THREAD = 500000;  // 500K 最小每线程元素数
constexpr size_t MAX_THREADS = 4;                   // M4 性能核数量

// 4-bit 掩码查找表
alignas(64) const uint8_t MASK_LUT[16][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 0, 0, 0},
    {2, 0, 1, 0, 0, 0, 0, 0},
    {1, 2, 0, 0, 0, 0, 0, 0},
    {2, 0, 2, 0, 0, 0, 0, 0},
    {2, 1, 2, 0, 0, 0, 0, 0},
    {3, 0, 1, 2, 0, 0, 0, 0},
    {1, 3, 0, 0, 0, 0, 0, 0},
    {2, 0, 3, 0, 0, 0, 0, 0},
    {2, 1, 3, 0, 0, 0, 0, 0},
    {3, 0, 1, 3, 0, 0, 0, 0},
    {2, 2, 3, 0, 0, 0, 0, 0},
    {3, 0, 2, 3, 0, 0, 0, 0},
    {3, 1, 2, 3, 0, 0, 0, 0},
    {4, 0, 1, 2, 3, 0, 0, 0},
};

#ifdef __aarch64__
inline uint32_t extract_mask_4(uint32x4_t mask) {
    uint32x4_t shifted = vshrq_n_u32(mask, 31);
    return vgetq_lane_u32(shifted, 0) |
           (vgetq_lane_u32(shifted, 1) << 1) |
           (vgetq_lane_u32(shifted, 2) << 2) |
           (vgetq_lane_u32(shifted, 3) << 3);
}

// 单线程 SIMD 过滤 (处理数据块)
size_t filter_gt_chunk_simd(const int32_t* input, size_t start, size_t end,
                             int32_t threshold, uint32_t* output) {
    int32x4_t thresh_vec = vdupq_n_s32(threshold);
    size_t out_count = 0;
    size_t i = start;

    // 16 元素展开
    for (; i + 16 <= end; i += 16) {
        __builtin_prefetch(&input[i + 64], 0, 0);

        int32x4_t d0 = vld1q_s32(&input[i]);
        int32x4_t d1 = vld1q_s32(&input[i + 4]);
        int32x4_t d2 = vld1q_s32(&input[i + 8]);
        int32x4_t d3 = vld1q_s32(&input[i + 12]);

        uint32x4_t m0 = vcgtq_s32(d0, thresh_vec);
        uint32x4_t m1 = vcgtq_s32(d1, thresh_vec);
        uint32x4_t m2 = vcgtq_s32(d2, thresh_vec);
        uint32x4_t m3 = vcgtq_s32(d3, thresh_vec);

        uint32_t mask0 = extract_mask_4(m0);
        uint32_t mask1 = extract_mask_4(m1);
        uint32_t mask2 = extract_mask_4(m2);
        uint32_t mask3 = extract_mask_4(m3);

        const uint8_t* lut0 = MASK_LUT[mask0];
        const uint8_t* lut1 = MASK_LUT[mask1];
        const uint8_t* lut2 = MASK_LUT[mask2];
        const uint8_t* lut3 = MASK_LUT[mask3];

        for (int j = 0; j < lut0[0]; ++j) {
            output[out_count++] = static_cast<uint32_t>(i + lut0[j + 1]);
        }
        for (int j = 0; j < lut1[0]; ++j) {
            output[out_count++] = static_cast<uint32_t>(i + 4 + lut1[j + 1]);
        }
        for (int j = 0; j < lut2[0]; ++j) {
            output[out_count++] = static_cast<uint32_t>(i + 8 + lut2[j + 1]);
        }
        for (int j = 0; j < lut3[0]; ++j) {
            output[out_count++] = static_cast<uint32_t>(i + 12 + lut3[j + 1]);
        }
    }

    // 4 元素块
    for (; i + 4 <= end; i += 4) {
        int32x4_t d = vld1q_s32(&input[i]);
        uint32x4_t m = vcgtq_s32(d, thresh_vec);
        uint32_t mask = extract_mask_4(m);
        const uint8_t* lut = MASK_LUT[mask];
        for (int j = 0; j < lut[0]; ++j) {
            output[out_count++] = static_cast<uint32_t>(i + lut[j + 1]);
        }
    }

    // 剩余
    for (; i < end; ++i) {
        if (input[i] > threshold) {
            output[out_count++] = static_cast<uint32_t>(i);
        }
    }

    return out_count;
}
#endif

} // anonymous namespace

// ============================================================================
// 多线程并行 Filter
// ============================================================================

size_t filter_i32_parallel(const int32_t* input, size_t count,
                            CompareOp op, int32_t threshold, uint32_t* output) {
    if (count == 0 || !input || !output) return 0;

    // 目前只优化 GT 操作，其他委托给单线程
    if (op != CompareOp::GT) {
        return filter_i32(input, count, op, threshold, output);
    }

#ifdef __aarch64__
    // 计算线程数
    size_t num_threads = std::min(MAX_THREADS,
                                  std::max(size_t(1), count / MIN_ELEMENTS_PER_THREAD));

    // 小数据量：单线程
    if (num_threads <= 1 || count < MIN_ELEMENTS_PER_THREAD) {
        return filter_i32(input, count, op, threshold, output);
    }

    // 每线程的局部输出缓冲区
    std::vector<std::vector<uint32_t>> local_outputs(num_threads);

    // 计算每线程的数据范围
    size_t chunk_size = (count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // 启动工作线程
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);

        if (start >= end) continue;

        threads.emplace_back([&, t, start, end]() {
            // 分配临时缓冲区
            size_t max_output = end - start;
            local_outputs[t].resize(max_output);

            // 执行过滤
            size_t out_count = filter_gt_chunk_simd(
                input, start, end, threshold,
                local_outputs[t].data());

            local_outputs[t].resize(out_count);
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 计算偏移量
    std::vector<size_t> offsets(num_threads + 1, 0);
    for (size_t t = 0; t < num_threads; ++t) {
        offsets[t + 1] = offsets[t] + local_outputs[t].size();
    }

    size_t total_count = offsets[num_threads];

    // 复制到输出
    for (size_t t = 0; t < num_threads; ++t) {
        if (!local_outputs[t].empty()) {
            std::memcpy(output + offsets[t],
                       local_outputs[t].data(),
                       local_outputs[t].size() * sizeof(uint32_t));
        }
    }

    return total_count;
#else
    return filter_i32(input, count, op, threshold, output);
#endif
}

} // namespace filter
} // namespace thunderduck
