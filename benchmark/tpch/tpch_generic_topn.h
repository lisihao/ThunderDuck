/**
 * ThunderDuck - Generic Top-N Aware Partial Aggregation
 *
 * 通用模式:
 *   并行扫描 → 线程局部聚合 → 局部 Top-K → 全局合并
 *
 * 适用查询: 任何带 ORDER BY + LIMIT 的聚合查询
 *   - Q3: revenue DESC, orderdate ASC, LIMIT 10
 *   - Q10: revenue DESC, LIMIT 20
 *   - Q18: quantity DESC, LIMIT 100
 *
 * @version 1.0
 * @date 2026-01-29
 */

#pragma once

#include "tpch_operators_v25.h"  // ThreadPool
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <future>
#include <algorithm>
#include <optional>

namespace thunderduck {
namespace tpch {
namespace generic {

// ============================================================================
// TopNAwareAggregator - 通用 Top-N 感知聚合器
// ============================================================================

/**
 * 通用 Top-N 感知聚合器
 *
 * @tparam KeyT      聚合键类型 (如 int32_t orderkey)
 * @tparam ValueT    聚合值类型 (如 int64_t revenue)
 * @tparam ResultT   结果类型 (包含完整信息用于排序)
 *
 * 使用方式:
 * 1. 定义 ResultT 结构体，包含 key、value 和排序所需的附加字段
 * 2. 提供比较器 (用于 Top-N 排序)
 * 3. 提供结果构建函数 (key + value → ResultT)
 * 4. 调用 aggregate() 进行并行聚合
 */
template<typename KeyT, typename ValueT, typename ResultT>
class TopNAwareAggregator {
public:
    // 比较器: 返回 true 表示 a 应该排在 b 前面 (用于最终排序)
    using Comparator = std::function<bool(const ResultT&, const ResultT&)>;

    // 结果构建器: 从 key 和 value 构建完整结果
    using ResultBuilder = std::function<ResultT(KeyT key, ValueT value)>;

    // 值合并器: 合并两个同 key 的 value (默认: 累加)
    using ValueMerger = std::function<ValueT(ValueT a, ValueT b)>;

    struct Config {
        size_t final_top_n = 10;      // 最终需要的 Top-N
        size_t local_top_k = 32;      // 每线程保留的 Top-K (应 >= final_top_n * 2)
        size_t num_threads = 8;       // 线程数

        Comparator comparator;        // 排序比较器
        ResultBuilder result_builder; // 结果构建器
        ValueMerger value_merger = [](ValueT a, ValueT b) { return a + b; };
    };

    explicit TopNAwareAggregator(const Config& config) : config_(config) {}

    /**
     * 执行聚合
     *
     * @param data_count    数据总数
     * @param key_extractor 从索引提取 key 的函数 (返回 nullopt 表示跳过)
     * @param value_extractor 从索引提取 value 的函数
     * @return Top-N 结果
     */
    template<typename KeyExtractor, typename ValueExtractor>
    std::vector<ResultT> aggregate(
        size_t data_count,
        KeyExtractor key_extractor,
        ValueExtractor value_extractor
    ) {
        auto& pool = ops_v25::ThreadPool::instance();
        pool.prewarm(config_.num_threads, data_count / config_.num_threads);
        size_t num_threads = pool.size();
        size_t chunk_size = (data_count + num_threads - 1) / num_threads;

        // 每线程的局部 Top-K 结果
        std::vector<std::vector<ResultT>> local_tops(num_threads);
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, data_count);
            if (start >= data_count) break;

            futures.push_back(pool.submit([&, t, start, end]() {
                // 线程局部聚合 map
                std::unordered_map<KeyT, ValueT> local_agg;

                for (size_t i = start; i < end; ++i) {
                    auto key_opt = key_extractor(i);
                    if (!key_opt) continue;

                    KeyT key = *key_opt;
                    ValueT value = value_extractor(i);
                    local_agg[key] = config_.value_merger(local_agg[key], value);
                }

                // 提取局部 Top-K
                extract_local_top_k(local_agg, local_tops[t]);
            }));
        }

        for (auto& f : futures) f.get();

        // 全局合并
        return merge_global(local_tops);
    }

    /**
     * 执行带预过滤的聚合 (更高效)
     *
     * @param data_count      数据总数
     * @param filter          过滤函数 (返回 false 跳过)
     * @param key_extractor   提取 key
     * @param value_extractor 提取 value
     */
    template<typename Filter, typename KeyExtractor, typename ValueExtractor>
    std::vector<ResultT> aggregate_filtered(
        size_t data_count,
        Filter filter,
        KeyExtractor key_extractor,
        ValueExtractor value_extractor
    ) {
        return aggregate(data_count,
            [&](size_t i) -> std::optional<KeyT> {
                if (!filter(i)) return std::nullopt;
                return key_extractor(i);
            },
            value_extractor
        );
    }

private:
    Config config_;

    // Min-heap 比较器 (反转 comparator，使最小元素在顶部)
    struct HeapComparator {
        Comparator comp;
        bool operator()(const ResultT& a, const ResultT& b) const {
            return comp(a, b);  // a 更好则 a 应该在 b 下面
        }
    };

    void extract_local_top_k(
        const std::unordered_map<KeyT, ValueT>& agg,
        std::vector<ResultT>& local_top
    ) {
        HeapComparator heap_comp{config_.comparator};
        std::priority_queue<ResultT, std::vector<ResultT>, HeapComparator>
            heap(heap_comp);

        for (const auto& [key, value] : agg) {
            ResultT result = config_.result_builder(key, value);

            if (heap.size() < config_.local_top_k) {
                heap.push(result);
            } else if (config_.comparator(result, heap.top())) {
                heap.pop();
                heap.push(result);
            }
        }

        local_top.reserve(heap.size());
        while (!heap.empty()) {
            local_top.push_back(heap.top());
            heap.pop();
        }
    }

    std::vector<ResultT> merge_global(
        const std::vector<std::vector<ResultT>>& local_tops
    ) {
        // 合并同 key 的 value
        std::unordered_map<KeyT, ResultT> merged;

        for (const auto& local : local_tops) {
            for (const auto& r : local) {
                auto it = merged.find(get_key(r));
                if (it == merged.end()) {
                    merged[get_key(r)] = r;
                } else {
                    // 合并 value
                    ValueT merged_value = config_.value_merger(
                        get_value(it->second), get_value(r));
                    it->second = config_.result_builder(get_key(r), merged_value);
                }
            }
        }

        // 最终 Top-N
        std::vector<ResultT> results;
        results.reserve(merged.size());
        for (auto& [_, r] : merged) {
            results.push_back(r);
        }

        size_t top_n = std::min(config_.final_top_n, results.size());
        std::partial_sort(results.begin(), results.begin() + top_n, results.end(),
                          config_.comparator);

        if (results.size() > config_.final_top_n) {
            results.resize(config_.final_top_n);
        }

        return results;
    }

    // 从 ResultT 提取 key - 需要在 Config 中提供
    std::function<KeyT(const ResultT&)> get_key_;
    std::function<ValueT(const ResultT&)> get_value_;

public:
    // 设置 key/value 提取器 (必须在 aggregate 之前调用)
    void set_extractors(
        std::function<KeyT(const ResultT&)> get_key,
        std::function<ValueT(const ResultT&)> get_value
    ) {
        get_key_ = get_key;
        get_value_ = get_value;
    }

private:
    KeyT get_key(const ResultT& r) const { return get_key_(r); }
    ValueT get_value(const ResultT& r) const { return get_value_(r); }
};

// ============================================================================
// 便捷工厂函数
// ============================================================================

/**
 * 创建简单的 Top-N 聚合器 (key + value 结构)
 */
template<typename KeyT, typename ValueT>
struct SimpleResult {
    KeyT key;
    ValueT value;
};

template<typename KeyT, typename ValueT>
auto make_simple_topn_aggregator(
    size_t top_n,
    bool descending = true
) {
    using ResultT = SimpleResult<KeyT, ValueT>;

    typename TopNAwareAggregator<KeyT, ValueT, ResultT>::Config config;
    config.final_top_n = top_n;
    config.local_top_k = std::max(top_n * 3, size_t(32));

    if (descending) {
        config.comparator = [](const ResultT& a, const ResultT& b) {
            return a.value > b.value;
        };
    } else {
        config.comparator = [](const ResultT& a, const ResultT& b) {
            return a.value < b.value;
        };
    }

    config.result_builder = [](KeyT k, ValueT v) -> ResultT {
        return {k, v};
    };

    return TopNAwareAggregator<KeyT, ValueT, ResultT>(config);
}

} // namespace generic
} // namespace tpch
} // namespace thunderduck
