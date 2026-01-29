/**
 * ThunderDuck TPC-H 算子封装 V25
 *
 * 优化内容:
 * - 线程池预热与复用
 * - Key Hash 缓存
 * - Join Key Dictionary Encoding
 * - 弱 Hash + 冲突控制
 *
 * @version 25.0
 * @date 2026-01-28
 */

#ifndef TPCH_OPERATORS_V25_H
#define TPCH_OPERATORS_V25_H

#include "tpch_data_loader.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <array>
#include <unordered_map>
#include <memory>

namespace thunderduck {
namespace tpch {
namespace ops_v25 {

using ::thunderduck::tpch::TPCHDataLoader;

// ============================================================================
// 1. 线程池 - 预热与复用
// ============================================================================

/**
 * 高性能线程池
 *
 * 特点:
 * - 支持预热 (prewarm): 提前创建线程
 * - 支持动态扩缩容
 * - 支持批量任务提交
 * - 低开销任务分发
 */
class ThreadPool {
public:
    // 单例模式
    static ThreadPool& instance();

    // 禁止拷贝
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * 预热线程池
     * @param num_threads 预热线程数
     * @param estimated_tasks 预估任务数 (用于队列预分配)
     */
    void prewarm(size_t num_threads, size_t estimated_tasks = 0);

    /**
     * 根据数据量预估并预热
     * @param data_rows 数据行数
     * @param operation_type 操作类型 (filter/join/aggregate)
     */
    void prewarm_for_query(size_t data_rows, const char* operation_type);

    /**
     * 提交单个任务
     */
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())>;

    /**
     * 并行执行分块任务
     * @param total 总元素数
     * @param chunk_size 每块大小 (0 表示自动)
     * @param func 处理函数 (start, end)
     */
    void parallel_for(size_t total, size_t chunk_size,
                      std::function<void(size_t, size_t)> func);

    /**
     * 并行 map-reduce
     */
    template<typename T, typename MapFunc, typename ReduceFunc>
    T parallel_reduce(size_t total, T init, MapFunc map_func, ReduceFunc reduce_func);

    /**
     * 获取当前线程数
     */
    size_t size() const { return workers_.size(); }

    /**
     * 获取活跃线程数
     */
    size_t active_count() const { return active_count_.load(); }

    /**
     * 等待所有任务完成
     */
    void wait_all();

    /**
     * 关闭线程池
     */
    void shutdown();

private:
    ThreadPool();
    ~ThreadPool();

    void worker_loop();

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable cv_finished_;

    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_count_{0};
    std::atomic<size_t> pending_tasks_{0};
};

// ============================================================================
// 2. Hash 优化
// ============================================================================

/**
 * 弱 Hash 函数 (快速，允许一定冲突)
 *
 * 使用乘法 hash，比 std::hash 快 2-3x
 */
inline uint32_t weak_hash_i32(int32_t key) {
    // 黄金比例乘法 hash
    uint32_t h = static_cast<uint32_t>(key) * 2654435769u;
    return h;
}

inline uint32_t weak_hash_i32_mod(int32_t key, uint32_t mask) {
    return weak_hash_i32(key) & mask;
}

/**
 * Key Hash 缓存
 *
 * 预计算 probe 侧的 hash 值，避免重复计算
 */
class KeyHashCache {
public:
    KeyHashCache() = default;

    /**
     * 构建 hash 缓存
     * @param keys 键数组
     * @param count 键数量
     * @param table_size hash 表大小 (2 的幂次)
     */
    void build(const int32_t* keys, size_t count, uint32_t table_size);

    /**
     * 获取缓存的 hash 值
     */
    uint32_t get_hash(size_t idx) const { return hashes_[idx]; }

    /**
     * 获取 hash 数组
     */
    const uint32_t* data() const { return hashes_.data(); }

    size_t size() const { return hashes_.size(); }

private:
    std::vector<uint32_t> hashes_;
};

/**
 * Join Key Dictionary Encoding
 *
 * 将稀疏的 key (如 orderkey) 映射到连续的整数
 * 好处: 可以用数组直接索引，避免 hash 查找
 */
class KeyDictionary {
public:
    KeyDictionary() = default;

    /**
     * 从键数组构建字典
     * @param keys 键数组
     * @param count 键数量
     * @return 字典大小
     */
    size_t build(const int32_t* keys, size_t count);

    /**
     * 编码: key -> dict_id (如果不存在返回 -1)
     */
    int32_t encode(int32_t key) const;

    /**
     * 批量编码
     */
    void encode_batch(const int32_t* keys, size_t count, int32_t* out) const;

    /**
     * 解码: dict_id -> key
     */
    int32_t decode(int32_t dict_id) const { return id_to_key_[dict_id]; }

    /**
     * 字典大小
     */
    size_t dict_size() const { return id_to_key_.size(); }

    /**
     * 是否适合使用字典编码 (基数小于阈值)
     */
    bool is_suitable() const { return suitable_; }

private:
    std::unordered_map<int32_t, int32_t> key_to_id_;
    std::vector<int32_t> id_to_key_;
    bool suitable_ = false;
};

/**
 * 弱 Hash 表 (开放寻址 + 线性探测)
 *
 * 特点:
 * - 使用弱 hash 函数
 * - 固定大小，2 的幂次
 * - 控制负载因子 < 0.7
 * - 支持多值 (链表)
 */
template<typename V>
class WeakHashTable {
public:
    struct Entry {
        int32_t key;
        V value;
        int32_t next;  // 链表下一个 (-1 表示结束)
    };

    WeakHashTable() = default;

    /**
     * 初始化 hash 表
     * @param estimated_size 预估元素数量
     */
    void init(size_t estimated_size);

    /**
     * 插入键值对
     */
    void insert(int32_t key, const V& value);

    /**
     * 查找键 (返回第一个匹配的条目索引，-1 表示未找到)
     */
    int32_t find(int32_t key) const;

    /**
     * 查找键 (使用预计算的 hash)
     */
    int32_t find_with_hash(int32_t key, uint32_t hash) const;

    /**
     * 获取下一个相同 key 的条目
     */
    int32_t get_next(int32_t entry_idx) const;

    /**
     * 获取条目值
     */
    const V& get_value(int32_t entry_idx) const { return entries_[entry_idx].value; }

    /**
     * 获取条目键
     */
    int32_t get_key(int32_t entry_idx) const { return entries_[entry_idx].key; }

    /**
     * 表大小
     */
    size_t table_size() const { return table_size_; }

    /**
     * 元素数量
     */
    size_t entry_count() const { return entries_.size(); }

private:
    std::vector<int32_t> buckets_;   // bucket -> first entry index (-1 if empty)
    std::vector<Entry> entries_;     // 实际条目
    uint32_t table_size_ = 0;
    uint32_t mask_ = 0;
};

// ============================================================================
// 3. V25 优化版 Join 算子
// ============================================================================

/**
 * Hash Join with 预计算 Hash + 弱 Hash 表
 */
struct JoinPairsV25 {
    std::vector<uint32_t> left_indices;
    std::vector<uint32_t> right_indices;
    size_t count = 0;
};

/**
 * 构建侧 Hash 表 (使用弱 hash)
 */
void build_hash_table_v25(
    const int32_t* keys, size_t count,
    WeakHashTable<uint32_t>& table
);

/**
 * Probe with Hash 缓存
 */
void probe_with_hash_cache_v25(
    const WeakHashTable<uint32_t>& build_table,
    const int32_t* probe_keys, size_t probe_count,
    const KeyHashCache& hash_cache,
    JoinPairsV25& result
);

/**
 * Inner Join V25 (完整流程)
 */
void inner_join_v25(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinPairsV25& result
);

/**
 * Semi Join V25 (使用线程池并行)
 */
size_t semi_join_v25(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    uint32_t* out_probe_indices
);

// ============================================================================
// 4. V25 优化版聚合算子
// ============================================================================

/**
 * 并行聚合 (使用线程池)
 */
template<typename T>
T parallel_sum_v25(const T* data, size_t count);

/**
 * 条件并行聚合
 */
int64_t parallel_conditional_sum_v25(
    const int64_t* values, size_t count,
    const uint32_t* selection, size_t sel_count
);

// ============================================================================
// 5. V25 优化版查询
// ============================================================================

void run_q3_v25(TPCHDataLoader& loader);
void run_q5_v25(TPCHDataLoader& loader);
void run_q6_v25(TPCHDataLoader& loader);
void run_q7_v25(TPCHDataLoader& loader);
void run_q9_v25(TPCHDataLoader& loader);
void run_q10_v25(TPCHDataLoader& loader);
void run_q12_v25(TPCHDataLoader& loader);
void run_q14_v25(TPCHDataLoader& loader);
void run_q18_v25(TPCHDataLoader& loader);

/**
 * 初始化 V25 运行时 (预热线程池)
 */
void init_v25_runtime(size_t lineitem_count);

/**
 * 关闭 V25 运行时
 */
void shutdown_v25_runtime();

// ============================================================================
// Template 实现
// ============================================================================

template<typename F>
auto ThreadPool::submit(F&& f) -> std::future<decltype(f())> {
    using return_type = decltype(f());

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    std::future<return_type> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        tasks_.push_back([task]() { (*task)(); });
        pending_tasks_++;
    }
    cv_.notify_one();

    return result;
}

template<typename T, typename MapFunc, typename ReduceFunc>
T ThreadPool::parallel_reduce(size_t total, T init, MapFunc map_func, ReduceFunc reduce_func) {
    if (total == 0) return init;

    size_t num_threads = workers_.size();

    // 如果没有 worker 线程，直接在当前线程执行
    if (num_threads == 0) {
        return reduce_func(init, map_func(0, total));
    }

    size_t chunk_size = (total + num_threads - 1) / num_threads;

    std::vector<std::future<T>> futures;
    futures.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, total);
        if (start >= total) break;

        futures.push_back(submit([=, &map_func]() {
            return map_func(start, end);
        }));
    }

    T result = init;
    for (auto& f : futures) {
        result = reduce_func(result, f.get());
    }

    return result;
}

template<typename V>
void WeakHashTable<V>::init(size_t estimated_size) {
    // 计算表大小 (2 的幂次，负载因子 < 0.7)
    size_t min_size = static_cast<size_t>(estimated_size / 0.7) + 1;
    table_size_ = 1;
    while (table_size_ < min_size) table_size_ <<= 1;
    mask_ = table_size_ - 1;

    buckets_.assign(table_size_, -1);
    entries_.clear();
    entries_.reserve(estimated_size);
}

template<typename V>
void WeakHashTable<V>::insert(int32_t key, const V& value) {
    uint32_t bucket = weak_hash_i32_mod(key, mask_);

    int32_t entry_idx = static_cast<int32_t>(entries_.size());
    entries_.push_back({key, value, buckets_[bucket]});
    buckets_[bucket] = entry_idx;
}

template<typename V>
int32_t WeakHashTable<V>::find(int32_t key) const {
    uint32_t bucket = weak_hash_i32_mod(key, mask_);
    int32_t idx = buckets_[bucket];

    while (idx >= 0) {
        if (entries_[idx].key == key) return idx;
        idx = entries_[idx].next;
    }
    return -1;
}

template<typename V>
int32_t WeakHashTable<V>::find_with_hash(int32_t key, uint32_t hash) const {
    uint32_t bucket = hash & mask_;
    int32_t idx = buckets_[bucket];

    while (idx >= 0) {
        if (entries_[idx].key == key) return idx;
        idx = entries_[idx].next;
    }
    return -1;
}

template<typename V>
int32_t WeakHashTable<V>::get_next(int32_t entry_idx) const {
    int32_t next = entries_[entry_idx].next;
    int32_t key = entries_[entry_idx].key;

    while (next >= 0) {
        if (entries_[next].key == key) return next;
        next = entries_[next].next;
    }
    return -1;
}

template<typename T>
T parallel_sum_v25(const T* data, size_t count) {
    auto& pool = ThreadPool::instance();

    return pool.parallel_reduce<T>(
        count,
        T(0),
        [data](size_t start, size_t end) {
            T sum = 0;
            for (size_t i = start; i < end; ++i) {
                sum += data[i];
            }
            return sum;
        },
        [](T a, T b) { return a + b; }
    );
}

} // namespace ops_v25
} // namespace tpch
} // namespace thunderduck

#endif // TPCH_OPERATORS_V25_H
