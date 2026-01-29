/**
 * ThunderDuck TPC-H 算子封装 - 实现
 */

#include "tpch_operators.h"
#include "thunderduck/memory.h"
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <thread>
#include <array>

namespace thunderduck {
namespace tpch {
namespace ops {

// ============================================================================
// Filter 算子实现
// ============================================================================

size_t filter_range_i32(const int32_t* data, size_t n,
                        int32_t low, int32_t high,
                        uint32_t* out_indices) {
    size_t count = 0;

#ifdef __aarch64__
    // ARM Neon 向量化实现
    int32x4_t v_lo = vdupq_n_s32(low);
    int32x4_t v_hi = vdupq_n_s32(high);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t values = vld1q_s32(data + i);
        uint32x4_t ge_lo = vcgeq_s32(values, v_lo);
        uint32x4_t lt_hi = vcltq_s32(values, v_hi);
        uint32x4_t mask = vandq_u32(ge_lo, lt_hi);

        // 提取掩码并写入索引
        alignas(16) uint32_t mask_arr[4];
        vst1q_u32(mask_arr, mask);

        for (int j = 0; j < 4; ++j) {
            if (mask_arr[j]) {
                out_indices[count++] = static_cast<uint32_t>(i + j);
            }
        }
    }

    // 标量处理剩余
    for (; i < n; ++i) {
        if (data[i] >= low && data[i] < high) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
#else
    // 标量实现
    for (size_t i = 0; i < n; ++i) {
        if (data[i] >= low && data[i] < high) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }
#endif

    return count;
}

size_t filter_range_inclusive_i64(const int64_t* data, size_t n,
                                  int64_t low, int64_t high,
                                  uint32_t* out_indices) {
    size_t count = 0;

    // 标量实现 (int64 向量化较复杂)
    for (size_t i = 0; i < n; ++i) {
        if (data[i] >= low && data[i] <= high) {
            out_indices[count++] = static_cast<uint32_t>(i);
        }
    }

    return count;
}

// ============================================================================
// MultiFilter 实现
// ============================================================================

MultiFilter::MultiFilter(size_t n) : n_(n), initialized_(false) {
    // 每个 uint64_t 存储 64 行的位
    size_t bitmap_size = (n + 63) / 64;
    bitmap_.resize(bitmap_size, ~0ULL);  // 初始化为全 1
}

void MultiFilter::add_range_i32(const int32_t* data, int32_t low, int32_t high) {
    size_t bitmap_idx = 0;
    uint64_t current_word = 0;
    int bit_pos = 0;

    for (size_t i = 0; i < n_; ++i) {
        bool match = (data[i] >= low && data[i] < high);
        if (match) {
            current_word |= (1ULL << bit_pos);
        }

        if (++bit_pos == 64) {
            if (!initialized_) {
                bitmap_[bitmap_idx] = current_word;
            } else {
                bitmap_[bitmap_idx] &= current_word;
            }
            bitmap_idx++;
            current_word = 0;
            bit_pos = 0;
        }
    }

    // 处理最后一个不完整的 word
    if (bit_pos > 0) {
        if (!initialized_) {
            bitmap_[bitmap_idx] = current_word;
        } else {
            bitmap_[bitmap_idx] &= current_word;
        }
    }

    initialized_ = true;
}

void MultiFilter::add_range_inclusive_i64(const int64_t* data, int64_t low, int64_t high) {
    size_t bitmap_idx = 0;
    uint64_t current_word = 0;
    int bit_pos = 0;

    for (size_t i = 0; i < n_; ++i) {
        bool match = (data[i] >= low && data[i] <= high);
        if (match) {
            current_word |= (1ULL << bit_pos);
        }

        if (++bit_pos == 64) {
            if (!initialized_) {
                bitmap_[bitmap_idx] = current_word;
            } else {
                bitmap_[bitmap_idx] &= current_word;
            }
            bitmap_idx++;
            current_word = 0;
            bit_pos = 0;
        }
    }

    if (bit_pos > 0) {
        if (!initialized_) {
            bitmap_[bitmap_idx] = current_word;
        } else {
            bitmap_[bitmap_idx] &= current_word;
        }
    }

    initialized_ = true;
}

void MultiFilter::add_lt_i64(const int64_t* data, int64_t value) {
    size_t bitmap_idx = 0;
    uint64_t current_word = 0;
    int bit_pos = 0;

    for (size_t i = 0; i < n_; ++i) {
        bool match = (data[i] < value);
        if (match) {
            current_word |= (1ULL << bit_pos);
        }

        if (++bit_pos == 64) {
            if (!initialized_) {
                bitmap_[bitmap_idx] = current_word;
            } else {
                bitmap_[bitmap_idx] &= current_word;
            }
            bitmap_idx++;
            current_word = 0;
            bit_pos = 0;
        }
    }

    if (bit_pos > 0) {
        if (!initialized_) {
            bitmap_[bitmap_idx] = current_word;
        } else {
            bitmap_[bitmap_idx] &= current_word;
        }
    }

    initialized_ = true;
}

void MultiFilter::add_eq_i8(const int8_t* data, int8_t value) {
    size_t bitmap_idx = 0;
    uint64_t current_word = 0;
    int bit_pos = 0;

    for (size_t i = 0; i < n_; ++i) {
        bool match = (data[i] == value);
        if (match) {
            current_word |= (1ULL << bit_pos);
        }

        if (++bit_pos == 64) {
            if (!initialized_) {
                bitmap_[bitmap_idx] = current_word;
            } else {
                bitmap_[bitmap_idx] &= current_word;
            }
            bitmap_idx++;
            current_word = 0;
            bit_pos = 0;
        }
    }

    if (bit_pos > 0) {
        if (!initialized_) {
            bitmap_[bitmap_idx] = current_word;
        } else {
            bitmap_[bitmap_idx] &= current_word;
        }
    }

    initialized_ = true;
}

void MultiFilter::add_ne_i8(const int8_t* data, int8_t value) {
    size_t bitmap_idx = 0;
    uint64_t current_word = 0;
    int bit_pos = 0;

    for (size_t i = 0; i < n_; ++i) {
        bool match = (data[i] != value);
        if (match) {
            current_word |= (1ULL << bit_pos);
        }

        if (++bit_pos == 64) {
            if (!initialized_) {
                bitmap_[bitmap_idx] = current_word;
            } else {
                bitmap_[bitmap_idx] &= current_word;
            }
            bitmap_idx++;
            current_word = 0;
            bit_pos = 0;
        }
    }

    if (bit_pos > 0) {
        if (!initialized_) {
            bitmap_[bitmap_idx] = current_word;
        } else {
            bitmap_[bitmap_idx] &= current_word;
        }
    }

    initialized_ = true;
}

size_t MultiFilter::get_indices(uint32_t* out_indices) const {
    size_t count = 0;
    for (size_t i = 0; i < bitmap_.size(); ++i) {
        uint64_t word = bitmap_[i];
        while (word) {
            int bit = __builtin_ctzll(word);  // 找到最低位的 1
            uint32_t idx = static_cast<uint32_t>(i * 64 + bit);
            if (idx < n_) {
                out_indices[count++] = idx;
            }
            word &= word - 1;  // 清除最低位的 1
        }
    }
    return count;
}

size_t MultiFilter::count() const {
    size_t cnt = 0;
    for (size_t i = 0; i < bitmap_.size(); ++i) {
        cnt += __builtin_popcountll(bitmap_[i]);
    }
    // 修正最后一个 word 可能多计数的情况
    size_t remainder = n_ % 64;
    if (remainder > 0) {
        uint64_t mask = (1ULL << remainder) - 1;
        cnt -= __builtin_popcountll(bitmap_.back() & ~mask);
    }
    return cnt;
}

// ============================================================================
// Aggregate 算子实现
// ============================================================================

int64_t sum_i64_sel(const int64_t* data, const uint32_t* sel, size_t sel_count) {
    int64_t sum = 0;

    // 8 路展开
    size_t i = 0;
    int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    for (; i + 8 <= sel_count; i += 8) {
        s0 += data[sel[i]];
        s1 += data[sel[i + 1]];
        s2 += data[sel[i + 2]];
        s3 += data[sel[i + 3]];
        s4 += data[sel[i + 4]];
        s5 += data[sel[i + 5]];
        s6 += data[sel[i + 6]];
        s7 += data[sel[i + 7]];
    }

    sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);

    for (; i < sel_count; ++i) {
        sum += data[sel[i]];
    }

    return sum;
}

int64_t sum_product_fixed(const int64_t* a, const int64_t* b, size_t n) {
    int64_t sum = 0;

    // 使用 __int128 避免溢出
    __int128 total = 0;

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        total += (__int128)a[i] * b[i];
        total += (__int128)a[i + 1] * b[i + 1];
        total += (__int128)a[i + 2] * b[i + 2];
        total += (__int128)a[i + 3] * b[i + 3];
    }

    for (; i < n; ++i) {
        total += (__int128)a[i] * b[i];
    }

    // 除以 10000 得到正确的定点数结果
    return static_cast<int64_t>(total / 10000);
}

int64_t sum_product_fixed_sel(const int64_t* a, const int64_t* b,
                              const uint32_t* sel, size_t sel_count) {
    __int128 total = 0;

    for (size_t i = 0; i < sel_count; ++i) {
        uint32_t idx = sel[i];
        total += (__int128)a[idx] * b[idx];
    }

    return static_cast<int64_t>(total / 10000);
}

int64_t sum_price_discount(const int64_t* price, const int64_t* discount, size_t n) {
    // SUM(price * (1 - discount)) = SUM(price * (10000 - discount) / 10000)
    __int128 total = 0;

    for (size_t i = 0; i < n; ++i) {
        total += (__int128)price[i] * (10000 - discount[i]);
    }

    return static_cast<int64_t>(total / 10000);
}

int64_t sum_price_discount_sel(const int64_t* price, const int64_t* discount,
                               const uint32_t* sel, size_t sel_count) {
    __int128 total = 0;

    for (size_t i = 0; i < sel_count; ++i) {
        uint32_t idx = sel[i];
        total += (__int128)price[idx] * (10000 - discount[idx]);
    }

    return static_cast<int64_t>(total / 10000);
}

void group_sum_i32_key(const int64_t* values, const int32_t* keys,
                       size_t n, std::unordered_map<int32_t, int64_t>& sums) {
    for (size_t i = 0; i < n; ++i) {
        sums[keys[i]] += values[i];
    }
}

void group_sum_string_key(const int64_t* values, const std::string* keys,
                          size_t n, std::unordered_map<std::string, int64_t>& sums) {
    for (size_t i = 0; i < n; ++i) {
        sums[keys[i]] += values[i];
    }
}

void q1_group_aggregate(
    const int64_t* quantity,
    const int64_t* extendedprice,
    const int64_t* discount,
    const int64_t* tax,
    const int8_t* returnflag,
    const int8_t* linestatus,
    const int32_t* shipdate,
    int32_t date_threshold,
    size_t n,
    std::unordered_map<int16_t, Q1AggResult>& results) {

    // V20.1 优化: 8 线程并行 + 8 路展开
    // 使用固定数组替代哈希表 (只有 3*2=6 个分组)
    // returnflag: A=0, N=1, R=2
    // linestatus: F=0, O=1
    // key = returnflag * 2 + linestatus (0-5)

    constexpr size_t NUM_THREADS = 8;
    constexpr size_t NUM_GROUPS = 6;

    // 每个线程的本地聚合结果 (缓存行对齐避免伪共享)
    alignas(128) std::array<std::array<Q1AggResult, NUM_GROUPS>, NUM_THREADS> thread_groups = {};
    std::array<std::thread, NUM_THREADS> threads;

    size_t chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, n);

        threads[t] = std::thread([&, t, start, end]() {
            auto& local_groups = thread_groups[t];

            // 8 路展开以提高 ILP
            size_t i = start;
            for (; i + 8 <= end; i += 8) {
                // 预取下一批数据
                __builtin_prefetch(shipdate + i + 64, 0, 3);
                __builtin_prefetch(quantity + i + 64, 0, 3);
                __builtin_prefetch(extendedprice + i + 64, 0, 3);
                __builtin_prefetch(discount + i + 64, 0, 3);

                #define PROCESS_ROW(idx) \
                    if (shipdate[i+idx] <= date_threshold) { \
                        int key = returnflag[i+idx] * 2 + linestatus[i+idx]; \
                        auto& r = local_groups[key]; \
                        r.sum_qty += quantity[i+idx]; \
                        r.sum_base_price += extendedprice[i+idx]; \
                        int64_t disc = 10000 - discount[i+idx]; \
                        int64_t disc_price = (extendedprice[i+idx] * disc) / 10000; \
                        r.sum_disc_price += disc_price; \
                        r.sum_charge += (disc_price * (10000 + tax[i+idx])) / 10000; \
                        r.sum_discount += discount[i+idx]; \
                        r.count++; \
                    }

                PROCESS_ROW(0);
                PROCESS_ROW(1);
                PROCESS_ROW(2);
                PROCESS_ROW(3);
                PROCESS_ROW(4);
                PROCESS_ROW(5);
                PROCESS_ROW(6);
                PROCESS_ROW(7);

                #undef PROCESS_ROW
            }

            // 处理剩余元素
            for (; i < end; ++i) {
                if (shipdate[i] <= date_threshold) {
                    int key = returnflag[i] * 2 + linestatus[i];
                    auto& r = local_groups[key];
                    r.sum_qty += quantity[i];
                    r.sum_base_price += extendedprice[i];
                    int64_t disc = 10000 - discount[i];
                    int64_t disc_price = (extendedprice[i] * disc) / 10000;
                    r.sum_disc_price += disc_price;
                    r.sum_charge += (disc_price * (10000 + tax[i])) / 10000;
                    r.sum_discount += discount[i];
                    r.count++;
                }
            }
        });
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }

    // 合并所有线程的结果
    Q1AggResult merged_groups[NUM_GROUPS] = {};
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        for (size_t g = 0; g < NUM_GROUPS; ++g) {
            merged_groups[g].sum_qty += thread_groups[t][g].sum_qty;
            merged_groups[g].sum_base_price += thread_groups[t][g].sum_base_price;
            merged_groups[g].sum_disc_price += thread_groups[t][g].sum_disc_price;
            merged_groups[g].sum_charge += thread_groups[t][g].sum_charge;
            merged_groups[g].sum_discount += thread_groups[t][g].sum_discount;
            merged_groups[g].count += thread_groups[t][g].count;
        }
    }

    // 转换回 unordered_map 格式
    results.clear();
    for (int rf = 0; rf < 3; ++rf) {
        for (int ls = 0; ls < 2; ++ls) {
            int key = rf * 2 + ls;
            if (merged_groups[key].count > 0) {
                int16_t map_key = (static_cast<int16_t>(rf) << 8) | static_cast<int16_t>(ls);
                results[map_key] = merged_groups[key];
            }
        }
    }
}

// ============================================================================
// Join 算子实现
// ============================================================================

void inner_join_i32(const int32_t* build_keys, size_t build_count,
                    const int32_t* probe_keys, size_t probe_count,
                    JoinPairs& result) {
    // 使用 ThunderDuck V19.2 Join (激进预取 + SIMD 并行槽位比较)
    //
    // 重要: hash_join_i32_v19_2 内部的 grow_join_result 使用 aligned_alloc/free
    // 因此我们必须预分配足够大的缓冲区，避免触发 grow_join_result
    // (否则会尝试 aligned_free std::vector 管理的内存导致崩溃)
    //
    // 对于 TPC-H SF=1，最大的 join (lineitem) 可能产生 6M+ 结果
    // 预分配 probe_count * 4 或 10M，取较大者
    size_t initial_capacity = std::max(probe_count * 4, size_t(10000000));

    join::JoinResult jr;
    result.left_indices.resize(initial_capacity);
    result.right_indices.resize(initial_capacity);

    jr.left_indices = result.left_indices.data();
    jr.right_indices = result.right_indices.data();
    jr.count = 0;
    jr.capacity = initial_capacity;

    result.count = join::hash_join_i32_v19_2(
        build_keys, build_count,
        probe_keys, probe_count,
        join::JoinType::INNER,
        &jr
    );

    result.left_indices.resize(result.count);
    result.right_indices.resize(result.count);
}

void semi_join_i32(const int32_t* build_keys, size_t build_count,
                   const int32_t* probe_keys, size_t probe_count,
                   std::vector<uint32_t>& probe_matches) {
    // 使用 GPU SEMI Join (V20.1 最优版本)
    // 数据量 >= 500K 时自动使用 GPU，否则 CPU 回退

    if (join::is_semi_join_gpu_available() && build_count + probe_count >= 500000) {
        // GPU 版本需要使用 aligned_alloc 分配的内存
        // 因为 grow_join_result 使用 aligned_free
        constexpr size_t CACHE_LINE = 128;

        // 预分配足够大的缓冲区，避免 grow_join_result 被调用
        size_t capacity = probe_count;

        uint32_t* left_buf = static_cast<uint32_t*>(
            ::thunderduck::aligned_alloc(capacity * sizeof(uint32_t), CACHE_LINE));
        uint32_t* right_buf = static_cast<uint32_t*>(
            ::thunderduck::aligned_alloc(capacity * sizeof(uint32_t), CACHE_LINE));

        join::JoinResult jr;
        jr.left_indices = left_buf;
        jr.right_indices = right_buf;
        jr.count = 0;
        jr.capacity = capacity;

        size_t count = join::semi_join_gpu(
            build_keys, build_count,
            probe_keys, probe_count,
            &jr
        );

        // 拷贝结果到 std::vector
        // SEMI join 的结果在 right_indices (probe 匹配索引)
        probe_matches.resize(count);
        std::memcpy(probe_matches.data(), jr.right_indices, count * sizeof(uint32_t));

        // 释放 aligned 内存
        ::thunderduck::aligned_free(left_buf);
        ::thunderduck::aligned_free(right_buf);
    } else {
        // CPU 回退实现 (8-way 展开)
        std::unordered_set<int32_t> build_set(build_keys, build_keys + build_count);
        probe_matches.clear();
        probe_matches.reserve(probe_count / 2);

        size_t i = 0;
        for (; i + 8 <= probe_count; i += 8) {
            if (build_set.count(probe_keys[i])) probe_matches.push_back(static_cast<uint32_t>(i));
            if (build_set.count(probe_keys[i+1])) probe_matches.push_back(static_cast<uint32_t>(i+1));
            if (build_set.count(probe_keys[i+2])) probe_matches.push_back(static_cast<uint32_t>(i+2));
            if (build_set.count(probe_keys[i+3])) probe_matches.push_back(static_cast<uint32_t>(i+3));
            if (build_set.count(probe_keys[i+4])) probe_matches.push_back(static_cast<uint32_t>(i+4));
            if (build_set.count(probe_keys[i+5])) probe_matches.push_back(static_cast<uint32_t>(i+5));
            if (build_set.count(probe_keys[i+6])) probe_matches.push_back(static_cast<uint32_t>(i+6));
            if (build_set.count(probe_keys[i+7])) probe_matches.push_back(static_cast<uint32_t>(i+7));
        }
        for (; i < probe_count; ++i) {
            if (build_set.count(probe_keys[i])) {
                probe_matches.push_back(static_cast<uint32_t>(i));
            }
        }
    }
}

void anti_join_i32(const int32_t* build_keys, size_t build_count,
                   const int32_t* probe_keys, size_t probe_count,
                   std::vector<uint32_t>& probe_non_matches) {
    std::unordered_set<int32_t> build_set(build_keys, build_keys + build_count);
    probe_non_matches.clear();
    probe_non_matches.reserve(probe_count / 2);

    for (size_t i = 0; i < probe_count; ++i) {
        if (!build_set.count(probe_keys[i])) {
            probe_non_matches.push_back(static_cast<uint32_t>(i));
        }
    }
}

// ============================================================================
// 组合算子实现
// ============================================================================

int64_t filter_sum_product(
    const int64_t* a,
    const int64_t* b,
    const int32_t* date,
    int32_t date_lo,
    int32_t date_hi,
    const int64_t* filter_col,
    int64_t filter_lo,
    int64_t filter_hi,
    const int64_t* discount_col,
    int64_t discount_lo,
    int64_t discount_hi,
    size_t n) {

    __int128 total = 0;

#ifdef __aarch64__
    // ARM Neon 向量化实现
    int32x4_t v_date_lo = vdupq_n_s32(date_lo);
    int32x4_t v_date_hi = vdupq_n_s32(date_hi);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // 检查日期条件
        int32x4_t v_date = vld1q_s32(date + i);
        uint32x4_t date_ok = vandq_u32(
            vcgeq_s32(v_date, v_date_lo),
            vcltq_s32(v_date, v_date_hi)
        );

        // 检查其他条件并累加 (标量处理以避免复杂的向量逻辑)
        alignas(16) uint32_t mask_arr[4];
        vst1q_u32(mask_arr, date_ok);

        for (int j = 0; j < 4; ++j) {
            size_t idx = i + j;
            if (mask_arr[j] &&
                discount_col[idx] >= discount_lo && discount_col[idx] <= discount_hi &&
                filter_col[idx] < filter_hi) {
                total += (__int128)a[idx] * b[idx];
            }
        }
    }

    // 标量处理剩余
    for (; i < n; ++i) {
        if (date[i] >= date_lo && date[i] < date_hi &&
            discount_col[i] >= discount_lo && discount_col[i] <= discount_hi &&
            filter_col[i] < filter_hi) {
            total += (__int128)a[i] * b[i];
        }
    }
#else
    // 标量实现
    for (size_t i = 0; i < n; ++i) {
        if (date[i] >= date_lo && date[i] < date_hi &&
            discount_col[i] >= discount_lo && discount_col[i] <= discount_hi &&
            filter_col[i] < filter_hi) {
            total += (__int128)a[i] * b[i];
        }
    }
#endif

    return static_cast<int64_t>(total / 10000);
}

void join_group_sum(
    const int32_t* left_keys, size_t left_count,
    const int32_t* right_keys, size_t right_count,
    const int64_t* values,
    const int32_t* group_keys,
    JoinGroupSumResult& result) {

    // 先执行 Join
    JoinPairs pairs;
    inner_join_i32(left_keys, left_count, right_keys, right_count, pairs);

    // 然后分组聚合
    result.sums.clear();
    result.counts.clear();

    for (size_t i = 0; i < pairs.count; ++i) {
        uint32_t right_idx = pairs.right_indices[i];
        int64_t group_key = group_keys[right_idx];
        result.sums[group_key] += values[right_idx];
        result.counts[group_key]++;
    }
}

// ============================================================================
// MultiJoin 实现
// ============================================================================

void MultiJoin::add_join(const int32_t* left_keys, size_t left_count,
                         const int32_t* right_keys, size_t right_count) {
    joins_.push_back({left_keys, left_count, right_keys, right_count});
}

void MultiJoin::execute() {
    if (joins_.empty()) return;

    table_indices_.clear();
    table_indices_.resize(joins_.size() + 1);

    // 第一个 Join
    JoinPairs first_pairs;
    inner_join_i32(joins_[0].left_keys, joins_[0].left_count,
                   joins_[0].right_keys, joins_[0].right_count,
                   first_pairs);

    table_indices_[0] = std::move(first_pairs.left_indices);
    table_indices_[1] = std::move(first_pairs.right_indices);

    // 链式 Join 剩余表
    for (size_t j = 1; j < joins_.size(); ++j) {
        // 使用上一个 Join 的结果作为新的 probe 键
        // (简化实现 - 实际应用中需要更复杂的逻辑)
        JoinPairs pairs;
        inner_join_i32(joins_[j].left_keys, joins_[j].left_count,
                       joins_[j].right_keys, joins_[j].right_count,
                       pairs);

        table_indices_[j + 1] = std::move(pairs.right_indices);
    }
}

} // namespace ops
} // namespace tpch
} // namespace thunderduck
