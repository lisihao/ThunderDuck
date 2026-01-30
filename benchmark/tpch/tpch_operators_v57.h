/**
 * ThunderDuck TPC-H Operators V57 - 零开销通用算子框架
 *
 * 设计原则:
 * 1. 零硬编码 - 所有阈值基于硬件自动计算
 * 2. 零专用设计 - 所有算子查询无关
 * 3. 零运行时开销 - 模板参数 + constexpr if
 * 4. 自适应存储 - 自动选择直接数组 vs Hash 表
 *
 * @version 57
 * @date 2026-01-30
 */

#pragma once

#include "tpch_data_loader.h"
#include "tpch_operators_v32.h"
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <type_traits>
#include <cstring>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace thunderduck {
namespace tpch {
namespace ops_v57 {

using namespace ops_v32;

// ============================================================================
// 硬件感知工具 (零硬编码)
// ============================================================================

namespace hw {

/**
 * 运行时获取 L2 缓存大小
 */
inline size_t l2_cache_bytes() {
    static size_t cached = []() {
#ifdef __APPLE__
        size_t size = 0;
        size_t len = sizeof(size);
        if (sysctlbyname("hw.l2cachesize", &size, &len, nullptr, 0) == 0 && size > 0) {
            return size;
        }
        return size_t(4 * 1024 * 1024);  // Apple Silicon 默认
#else
        return size_t(256 * 1024);
#endif
    }();
    return cached;
}

/**
 * 计算 L2 友好的最大元素数
 */
template<typename T>
inline size_t l2_friendly_count(double utilization = 0.5) {
    return static_cast<size_t>(l2_cache_bytes() * utilization / sizeof(T));
}

/**
 * 获取硬件线程数
 */
inline size_t thread_count() {
    static size_t cached = []() {
        size_t n = std::thread::hardware_concurrency();
        if (n == 0) n = 4;
        if (n > 8) n = 8;  // 限制线程数避免过度竞争
        return n;
    }();
    return cached;
}

}  // namespace hw

// ============================================================================
// AdaptiveMap - 自适应存储结构 (核心通用算子)
// ============================================================================

/**
 * 自适应 Map: 自动选择直接数组 vs Hash 表
 *
 * 特点:
 * - 零硬编码: 阈值基于 L2 缓存自动计算
 * - 编译时分支: 使用 constexpr if 消除运行时开销
 * - 查询无关: 不包含任何查询特定代码
 */
template<typename ValueT>
class AdaptiveMap {
public:
    /**
     * 构建 Map
     *
     * @param keys 键数组
     * @param values 值数组 (如果为 nullptr，只存储存在性)
     * @param count 元素数量
     * @param filter 过滤谓词 (模板参数，可内联)
     */
    template<typename FilterFn = std::nullptr_t>
    void build(const int32_t* keys, const ValueT* values, size_t count,
               FilterFn filter = nullptr) {
        // 第一遍: 找到 key 范围 (应用过滤)
        int32_t min_key = INT32_MAX, max_key = INT32_MIN;
        size_t valid_count = 0;

        for (size_t i = 0; i < count; ++i) {
            if constexpr (!std::is_same_v<FilterFn, std::nullptr_t>) {
                if (!filter(i)) continue;
            }
            min_key = std::min(min_key, keys[i]);
            max_key = std::max(max_key, keys[i]);
            ++valid_count;
        }

        if (valid_count == 0) {
            use_direct_ = false;
            return;
        }

        offset_ = min_key;
        size_t range = static_cast<size_t>(max_key - min_key + 1);

        // 自适应选择: 基于 L2 缓存大小
        size_t max_direct = hw::l2_friendly_count<Entry>();

        if (range <= max_direct) {
            // 使用直接数组
            use_direct_ = true;
            direct_.resize(range);

            for (size_t i = 0; i < count; ++i) {
                if constexpr (!std::is_same_v<FilterFn, std::nullptr_t>) {
                    if (!filter(i)) continue;
                }
                size_t idx = static_cast<size_t>(keys[i] - offset_);
                direct_[idx].valid = true;
                if (values) direct_[idx].value = values[i];
            }
        } else {
            // 使用 Hash 表
            use_direct_ = false;
            hash_.init(valid_count);

            for (size_t i = 0; i < count; ++i) {
                if constexpr (!std::is_same_v<FilterFn, std::nullptr_t>) {
                    if (!filter(i)) continue;
                }
                if (values) {
                    hash_.insert(keys[i], values[i]);
                } else {
                    hash_.insert(keys[i], ValueT{});
                }
            }
        }
    }

    /**
     * 查找 (零开销分支)
     */
    const ValueT* find(int32_t key) const {
        if (use_direct_) {
            size_t idx = static_cast<size_t>(key - offset_);
            if (idx < direct_.size() && direct_[idx].valid) {
                return &direct_[idx].value;
            }
            return nullptr;
        } else {
            return hash_.find(key);
        }
    }

    /**
     * 存在性检查 (零开销)
     */
    bool contains(int32_t key) const {
        if (use_direct_) {
            size_t idx = static_cast<size_t>(key - offset_);
            return idx < direct_.size() && direct_[idx].valid;
        } else {
            return hash_.find(key) != nullptr;
        }
    }

    /**
     * 批量查找 (8 路预取)
     */
    void batch_find(const int32_t* keys, const ValueT** results, size_t n = 8) const {
        if (use_direct_) {
            for (size_t i = 0; i < n; ++i) {
                size_t idx = static_cast<size_t>(keys[i] - offset_);
                if (idx < direct_.size() && direct_[idx].valid) {
                    results[i] = &direct_[idx].value;
                } else {
                    results[i] = nullptr;
                }
            }
        } else {
            hash_.batch_find(keys, results);
        }
    }

    bool uses_direct_array() const { return use_direct_; }
    size_t size() const { return use_direct_ ? direct_.size() : hash_.size(); }

private:
    struct Entry {
        ValueT value{};
        bool valid = false;
    };

    bool use_direct_ = false;
    int32_t offset_ = 0;
    std::vector<Entry> direct_;
    CompactHashTable<ValueT> hash_;
};

// ============================================================================
// DirectArray - 纯直接数组 (当确定 key 范围小时使用)
// ============================================================================

/**
 * 直接数组: 用于 key 范围已知且小的情况
 *
 * 比 AdaptiveMap 更快，因为没有运行时分支
 */
template<typename ValueT>
class DirectArray {
public:
    /**
     * 初始化数组
     *
     * @param max_key 最大 key 值
     * @param default_value 默认值
     */
    void init(size_t max_key, ValueT default_value = ValueT{}) {
        data_.resize(max_key + 1, default_value);
        valid_.resize(max_key + 1, 0);  // uint8_t: 0 = invalid
    }

    /**
     * 设置值
     */
    void set(int32_t key, ValueT value) {
        if (key >= 0 && static_cast<size_t>(key) < data_.size()) {
            data_[key] = value;
            valid_[key] = 1;  // uint8_t: 1 = valid
        }
    }

    /**
     * 获取值 (无边界检查，调用者保证 key 有效)
     */
    ValueT get(int32_t key) const {
        return data_[key];
    }

    /**
     * 获取值指针 (带有效性检查)
     */
    const ValueT* find(int32_t key) const {
        if (key >= 0 && static_cast<size_t>(key) < data_.size() && valid_[key]) {
            return &data_[key];
        }
        return nullptr;
    }

    /**
     * 检查有效性
     */
    bool is_valid(int32_t key) const {
        return key >= 0 && static_cast<size_t>(key) < valid_.size() && valid_[key] != 0;
    }

    /**
     * 累加值
     */
    void add(int32_t key, ValueT delta) {
        if (key >= 0 && static_cast<size_t>(key) < data_.size()) {
            data_[key] += delta;
            valid_[key] = 1;  // uint8_t: 1 = valid
        }
    }

    /**
     * 原始数据指针 (用于高性能访问)
     */
    const ValueT* data() const { return data_.data(); }
    ValueT* data() { return data_.data(); }
    const uint8_t* valid() const { return valid_.data(); }

    size_t capacity() const { return data_.size(); }

private:
    std::vector<ValueT> data_;
    std::vector<uint8_t> valid_;  // 不使用 vector<bool>，因为它没有 .data()
};

// ============================================================================
// ZeroCostAggregator - 零开销聚合器
// ============================================================================

/**
 * 零开销聚合器: 使用模板参数消除虚调用
 *
 * @tparam ValueT 值类型
 * @tparam AggOp 聚合操作 (SUM, MIN, MAX, COUNT)
 */
template<typename ValueT, typename AggOp>
class ZeroCostAggregator {
public:
    struct AggState {
        ValueT value{};
        size_t count = 0;

        ValueT avg() const {
            return count > 0 ? value / static_cast<ValueT>(count) : ValueT{};
        }
    };

    /**
     * 初始化
     */
    void init(size_t max_key) {
        states_.init(max_key);
    }

    /**
     * 聚合单个值
     */
    void aggregate(int32_t key, ValueT value) {
        if (key >= 0 && static_cast<size_t>(key) < states_.capacity()) {
            auto& state = states_.data()[key];
            AggOp::apply(state.value, value);
            state.count++;
        }
    }

    /**
     * 获取聚合结果
     */
    const AggState* get(int32_t key) const {
        if (key >= 0 && static_cast<size_t>(key) < states_.capacity() &&
            states_.data()[key].count > 0) {
            return &states_.data()[key];
        }
        return nullptr;
    }

    /**
     * 原始状态数组 (用于并行合并)
     */
    DirectArray<AggState>& states() { return states_; }
    const DirectArray<AggState>& states() const { return states_; }

private:
    DirectArray<AggState> states_;
};

// 预定义聚合操作 (编译时内联)
struct SumOp {
    template<typename T>
    static void apply(T& acc, T val) { acc += val; }
};

struct MinOp {
    template<typename T>
    static void apply(T& acc, T val) { if (val < acc || acc == T{}) acc = val; }
};

struct MaxOp {
    template<typename T>
    static void apply(T& acc, T val) { if (val > acc) acc = val; }
};

// ============================================================================
// ZeroCostTwoPhaseAgg - 零开销两阶段聚合 (Q17 优化核心)
// ============================================================================

/**
 * 零开销两阶段聚合器
 *
 * 设计原则:
 * 1. 分离存储: sum[] 和 count[] 分开存储，避免结构体开销
 * 2. 原始指针: 热路径使用原始指针，消除边界检查
 * 3. 线程局部: 每个线程独立聚合，最后合并
 *
 * Phase 1: 计算每个 key 的聚合值 (SUM, COUNT)
 * Phase 2: 基于聚合值过滤并累加
 */
template<typename KeyT = int32_t>
class ZeroCostTwoPhaseAgg {
public:
    /**
     * 初始化
     *
     * @param max_key 最大 key 值
     * @param num_threads 线程数
     */
    void init(KeyT max_key, size_t num_threads) {
        max_key_ = max_key;
        num_threads_ = num_threads;

        // 分离存储 - 更好的缓存局部性
        global_sum_.resize(max_key + 1, 0);
        global_count_.resize(max_key + 1, 0);

        // 线程局部存储 - 避免伪共享
        thread_sum_.resize(num_threads);
        thread_count_.resize(num_threads);
        for (size_t t = 0; t < num_threads; ++t) {
            thread_sum_[t].resize(max_key + 1, 0);
            thread_count_[t].resize(max_key + 1, 0);
        }
    }

    /**
     * Phase 1: 线程局部聚合 (热路径)
     *
     * @param thread_id 线程 ID
     * @param key 聚合键
     * @param value 聚合值
     */
    void aggregate_local(size_t thread_id, KeyT key, int64_t value) {
        // 无边界检查 - 调用者保证 key 有效
        thread_sum_[thread_id][key] += value;
        thread_count_[thread_id][key]++;
    }

    /**
     * Phase 1 完成: 合并线程局部结果
     *
     * @param is_valid 有效性数组 (只合并有效的 key)
     */
    void merge(const uint8_t* is_valid) {
        for (size_t t = 0; t < num_threads_; ++t) {
            for (KeyT k = 1; k <= max_key_; ++k) {
                if (is_valid[k]) {
                    global_sum_[k] += thread_sum_[t][k];
                    global_count_[k] += thread_count_[t][k];
                }
            }
        }
    }

    /**
     * 获取 AVG
     */
    double get_avg(KeyT key) const {
        if (global_count_[key] > 0) {
            return static_cast<double>(global_sum_[key]) / global_count_[key];
        }
        return 0.0;
    }

    /**
     * 获取 SUM
     */
    int64_t get_sum(KeyT key) const { return global_sum_[key]; }

    /**
     * 获取 COUNT
     */
    int32_t get_count(KeyT key) const { return global_count_[key]; }

    /**
     * 原始指针访问 (用于 Phase 2 热路径)
     */
    const int64_t* sum_ptr() const { return global_sum_.data(); }
    const int32_t* count_ptr() const { return global_count_.data(); }

    KeyT max_key() const { return max_key_; }

private:
    KeyT max_key_ = 0;
    size_t num_threads_ = 0;

    // 全局聚合结果 (分离存储)
    std::vector<int64_t> global_sum_;
    std::vector<int32_t> global_count_;

    // 线程局部聚合 (分离存储)
    std::vector<std::vector<int64_t>> thread_sum_;
    std::vector<std::vector<int32_t>> thread_count_;
};

// ============================================================================
// ParallelScanner - 通用并行扫描器
// ============================================================================

/**
 * 通用并行扫描器
 *
 * @tparam ProcessFn 处理函数类型 (编译时确定)
 */
class ParallelScanner {
public:
    /**
     * 并行扫描
     *
     * @param count 总元素数
     * @param process_fn 处理函数: (thread_id, start, end) -> void
     */
    template<typename ProcessFn>
    static void scan(size_t count, ProcessFn&& process_fn) {
        size_t num_threads = hw::thread_count();
        size_t chunk_size = (count + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            if (start >= count) break;

            threads.emplace_back([&process_fn, t, start, end]() {
                process_fn(t, start, end);
            });
        }

        for (auto& th : threads) th.join();
    }

    /**
     * 获取线程数
     */
    static size_t thread_count() { return hw::thread_count(); }
};

// ============================================================================
// ZeroCostBranchlessFilter - 零开销无分支多条件过滤器
// ============================================================================

#ifdef __aarch64__
#include <arm_neon.h>
#endif

/**
 * 零开销无分支多条件过滤器
 *
 * 设计原则:
 * 1. Branchless: 所有条件合并为位掩码，消除分支预测失败
 * 2. SIMD: 批量处理 16 个 int8 或 4 个 int32
 * 3. 通用性: 使用模板参数，不包含任何查询特定代码
 *
 * 适用场景:
 * - 多条件过滤 (多个独立条件导致分支预测失败)
 * - 条件涉及多个列
 * - 分支预测失败是瓶颈
 */
class ZeroCostBranchlessFilter {
public:
    /**
     * 通用 4 条件 SIMD 过滤
     *
     * 条件模式 (通用):
     * - int8_col IN (val1, val2)
     * - int32_col_a < int32_col_b
     * - int32_col_c < int32_col_a
     * - int32_col_b >= range_lo AND int32_col_b < range_hi
     *
     * @param int8_col     int8 列数据
     * @param val1, val2   IN 条件的两个值
     * @param col_a        第一个 int32 列 (用于两个比较)
     * @param col_b        第二个 int32 列 (用于范围检查)
     * @param col_c        第三个 int32 列
     * @param range_lo     范围下界 (包含)
     * @param range_hi     范围上界 (不包含)
     * @param count        数据行数
     * @return 满足所有条件的索引数组
     */
    static std::vector<uint32_t> filter_multi_condition(
        const int8_t* int8_col, int8_t val1, int8_t val2,
        const int32_t* col_a, const int32_t* col_b, const int32_t* col_c,
        int32_t range_lo, int32_t range_hi,
        size_t count)
    {
        std::vector<uint32_t> result;
        result.reserve(count / 50);

#ifdef __aarch64__
        size_t i = 0;

        int8x16_t v_val1 = vdupq_n_s8(val1);
        int8x16_t v_val2 = vdupq_n_s8(val2);
        int32x4_t v_range_lo = vdupq_n_s32(range_lo);
        int32x4_t v_range_hi = vdupq_n_s32(range_hi);

        for (; i + 16 <= count; i += 16) {
            __builtin_prefetch(int8_col + i + 64, 0, 0);
            __builtin_prefetch(col_a + i + 64, 0, 0);
            __builtin_prefetch(col_b + i + 64, 0, 0);
            __builtin_prefetch(col_c + i + 64, 0, 0);

            // 条件 1: int8_col IN (val1, val2)
            int8x16_t vals = vld1q_s8(int8_col + i);
            uint8x16_t m1 = vorrq_u8(vceqq_s8(vals, v_val1), vceqq_s8(vals, v_val2));

            alignas(16) uint8_t m1_array[16];
            vst1q_u8(m1_array, m1);

            #define PROCESS_GROUP(G) do { \
                size_t base = i + (G) * 4; \
                int32x4_t a = vld1q_s32(col_a + base); \
                int32x4_t b = vld1q_s32(col_b + base); \
                int32x4_t c = vld1q_s32(col_c + base); \
                uint32x4_t m2 = vcltq_s32(a, b); \
                uint32x4_t m3 = vcltq_s32(c, a); \
                uint32x4_t m4 = vandq_u32(vcgeq_s32(b, v_range_lo), vcltq_s32(b, v_range_hi)); \
                uint32x4_t date_mask = vandq_u32(vandq_u32(m2, m3), m4); \
                uint32x4_t m1_expanded = vcombine_u32( \
                    vcreate_u32( \
                        (uint64_t)(m1_array[(G)*4] ? 0xFFFFFFFFu : 0) | \
                        ((uint64_t)(m1_array[(G)*4+1] ? 0xFFFFFFFFu : 0) << 32) \
                    ), \
                    vcreate_u32( \
                        (uint64_t)(m1_array[(G)*4+2] ? 0xFFFFFFFFu : 0) | \
                        ((uint64_t)(m1_array[(G)*4+3] ? 0xFFFFFFFFu : 0) << 32) \
                    ) \
                ); \
                uint32x4_t final_mask = vandq_u32(m1_expanded, date_mask); \
                uint32_t mask_bits[4]; \
                vst1q_u32(mask_bits, final_mask); \
                if (mask_bits[0]) result.push_back(static_cast<uint32_t>(base)); \
                if (mask_bits[1]) result.push_back(static_cast<uint32_t>(base + 1)); \
                if (mask_bits[2]) result.push_back(static_cast<uint32_t>(base + 2)); \
                if (mask_bits[3]) result.push_back(static_cast<uint32_t>(base + 3)); \
            } while(0)

            PROCESS_GROUP(0);
            PROCESS_GROUP(1);
            PROCESS_GROUP(2);
            PROCESS_GROUP(3);

            #undef PROCESS_GROUP
        }

        // 标量处理剩余元素
        for (; i < count; ++i) {
            int8_t v = int8_col[i];
            if ((v != val1 && v != val2)) continue;
            if (col_a[i] >= col_b[i]) continue;
            if (col_c[i] >= col_a[i]) continue;
            if (col_b[i] < range_lo || col_b[i] >= range_hi) continue;
            result.push_back(static_cast<uint32_t>(i));
        }

#else
        // 非 ARM: 标量 branchless
        for (size_t i = 0; i < count; ++i) {
            int8_t v = int8_col[i];
            uint8_t valid = ((v == val1) | (v == val2)) &
                           (col_a[i] < col_b[i]) &
                           (col_c[i] < col_a[i]) &
                           (col_b[i] >= range_lo) &
                           (col_b[i] < range_hi);
            if (valid) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
#endif

        return result;
    }

    /**
     * 通用模板过滤 (任意条件组合)
     *
     * @tparam MaskFn 掩码函数: (index) -> bool
     */
    template<typename MaskFn>
    static std::vector<uint32_t> filter(size_t count, MaskFn&& mask_fn) {
        std::vector<uint32_t> result;
        result.reserve(count / 10);

        for (size_t i = 0; i < count; ++i) {
            if (mask_fn(i)) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }

        return result;
    }
};

// ============================================================================
// FusedFilterAggregate - 融合过滤聚合算子 (消除中间向量)
// ============================================================================

/**
 * 融合过滤聚合算子
 *
 * 设计原则:
 * 1. 单次扫描: 过滤和聚合在同一循环完成
 * 2. 完全并行: 多线程并行处理
 * 3. 零中间存储: 无需 valid_indices 向量
 * 4. 通用性: 通过模板参数自定义过滤和聚合逻辑
 *
 * 适用场景:
 * - 过滤后聚合的查询模式
 * - 中间结果量大但最终聚合结果小
 */
class FusedFilterAggregate {
public:
    /**
     * 并行融合过滤聚合
     *
     * @tparam FilterFn   过滤函数: (index) -> bool
     * @tparam AggregateFn 聚合函数: (thread_id, index) -> void
     * @param count       数据行数
     * @param filter_fn   过滤谓词
     * @param aggregate_fn 聚合操作 (应更新线程局部状态)
     */
    template<typename FilterFn, typename AggregateFn>
    static void execute(size_t count, FilterFn&& filter_fn, AggregateFn&& aggregate_fn) {
        size_t num_threads = hw::thread_count();
        size_t chunk_size = (count + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            if (start >= count) break;

            threads.emplace_back([&, t, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    if (filter_fn(i)) {
                        aggregate_fn(t, i);
                    }
                }
            });
        }

        for (auto& th : threads) th.join();
    }

    /**
     * 带预取的并行融合过滤聚合
     *
     * @tparam FilterFn    过滤函数: (index) -> bool
     * @tparam AggregateFn 聚合函数: (thread_id, index) -> void
     * @tparam PrefetchFn  预取函数: (index + distance) -> void
     */
    template<typename FilterFn, typename AggregateFn, typename PrefetchFn>
    static void execute_with_prefetch(
        size_t count,
        FilterFn&& filter_fn,
        AggregateFn&& aggregate_fn,
        PrefetchFn&& prefetch_fn,
        size_t prefetch_distance = 64)
    {
        size_t num_threads = hw::thread_count();
        size_t chunk_size = (count + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            if (start >= count) break;

            threads.emplace_back([&, t, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    // 预取
                    if (i + prefetch_distance < end) {
                        prefetch_fn(i + prefetch_distance);
                    }

                    if (filter_fn(i)) {
                        aggregate_fn(t, i);
                    }
                }
            });
        }

        for (auto& th : threads) th.join();
    }
};

// ============================================================================
// 通用查询实现 (使用上述算子)
// ============================================================================

void run_q5_v57(TPCHDataLoader& loader);
void run_q8_v57(TPCHDataLoader& loader);
void run_q12_v57(TPCHDataLoader& loader);
void run_q17_v57(TPCHDataLoader& loader);

// ============================================================================
// 版本信息
// ============================================================================

extern const char* V57_VERSION;
extern const char* V57_DATE;

}  // namespace ops_v57
}  // namespace tpch
}  // namespace thunderduck
