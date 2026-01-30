/**
 * ThunderDuck System Catalog - Lightweight Edition
 *
 * 轻量级系统表，适合手机和PC环境
 *
 * 特性:
 * - 环形缓冲区 (固定内存，自动覆盖)
 * - 采样式触发 (低开销检查)
 * - 增量清理 (小批量，避免卡顿)
 * - 内存感知 (自适应阈值)
 *
 * @version 2.0 Lightweight
 * @date 2026-01-29
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <array>
#include <iomanip>
#include <iostream>

// DuckDB 头文件 (用于持久化)
#include "duckdb.hpp"

namespace thunderduck {
namespace catalog {

// ============================================================================
// 时间戳类型
// ============================================================================

using Timestamp = std::chrono::system_clock::time_point;

inline Timestamp now() {
    return std::chrono::system_clock::now();
}

inline int64_t timestamp_to_epoch_ms(Timestamp ts) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        ts.time_since_epoch()).count();
}

// ============================================================================
// 轻量级环形缓冲区
// ============================================================================

template<typename T, size_t Capacity>
class RingBuffer {
public:
    void push(const T& item) {
        data_[write_pos_] = item;
        write_pos_ = (write_pos_ + 1) % Capacity;
        if (size_ < Capacity) {
            size_++;
        }
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == Capacity; }
    static constexpr size_t capacity() { return Capacity; }

    // 获取最近 N 条记录
    std::vector<T> get_recent(size_t n) const {
        std::vector<T> result;
        n = std::min(n, size_);
        result.reserve(n);

        size_t start = (write_pos_ + Capacity - n) % Capacity;
        for (size_t i = 0; i < n; ++i) {
            result.push_back(data_[(start + i) % Capacity]);
        }
        return result;
    }

    // 遍历所有有效元素
    template<typename Func>
    void for_each(Func&& func) const {
        size_t start = (write_pos_ + Capacity - size_) % Capacity;
        for (size_t i = 0; i < size_; ++i) {
            func(data_[(start + i) % Capacity]);
        }
    }

    // 按条件过滤
    template<typename Pred>
    std::vector<T> filter(Pred&& pred) const {
        std::vector<T> result;
        for_each([&](const T& item) {
            if (pred(item)) {
                result.push_back(item);
            }
        });
        return result;
    }

    void clear() {
        size_ = 0;
        write_pos_ = 0;
    }

private:
    std::array<T, Capacity> data_;
    size_t write_pos_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// 紧凑性能指标 (32 字节)
// ============================================================================

struct CompactMetric {
    uint32_t query_hash;            // 查询 ID 哈希 (4 bytes)
    uint16_t version_id;            // 版本 ID (2 bytes)
    uint16_t flags;                 // 标志位 (2 bytes)

    float execution_time_ms;        // 执行时间 (4 bytes)
    uint32_t rows_processed;        // 处理行数 (4 bytes)

    float startup_cost_ms;          // 启动成本 (4 bytes)
    float per_row_cost_us;          // 每行成本 (4 bytes)

    uint32_t timestamp_offset;      // 相对时间戳 (4 bytes, 秒级)
    uint32_t memory_kb;             // 内存使用 KB (4 bytes)

    // 从完整指标构造
    static CompactMetric from_full(
        const std::string& query_id,
        const std::string& version,
        double exec_time,
        size_t rows,
        Timestamp base_time
    );
};

// ============================================================================
// 版本 ID 映射 (节省内存)
// ============================================================================

class VersionRegistry {
public:
    static VersionRegistry& instance() {
        static VersionRegistry reg;
        return reg;
    }

    uint16_t get_or_create_id(const std::string& version) {
        auto it = version_to_id_.find(version);
        if (it != version_to_id_.end()) {
            return it->second;
        }

        uint16_t id = static_cast<uint16_t>(id_to_version_.size());
        version_to_id_[version] = id;
        id_to_version_.push_back(version);
        return id;
    }

    const std::string& get_version(uint16_t id) const {
        static const std::string empty;
        return id < id_to_version_.size() ? id_to_version_[id] : empty;
    }

private:
    VersionRegistry() = default;
    std::unordered_map<std::string, uint16_t> version_to_id_;
    std::vector<std::string> id_to_version_;
};

// ============================================================================
// 轻量级清理策略
// ============================================================================

struct LightweightCleanupPolicy {
    // 环形缓冲区容量 (每个查询)
    static constexpr size_t RING_BUFFER_SIZE = 128;

    // 全局缓冲区容量
    static constexpr size_t GLOBAL_BUFFER_SIZE = 2048;

    // 采样间隔 (每 N 次插入检查一次)
    size_t sample_interval = 64;

    // 内存阈值 (超过则触发清理, KB)
    size_t memory_threshold_kb = 1024;  // 1MB

    // 增量清理批次大小
    size_t cleanup_batch_size = 32;

    // 保留最近 N 秒的数据 (用于 aging)
    uint32_t retention_seconds = 3600;  // 1 小时
};

// ============================================================================
// 时间尺度卷积 Sketch (Time-Scale Sketch)
//
// 设计原理: 类似 RRDtool/Prometheus 的多层时间聚合
// - L0: 原始细粒度数据 (最近 1 小时, 每秒级)
// - L1: 分钟级聚合 (最近 24 小时, 每分钟一个桶)
// - L2: 小时级聚合 (最近 7 天, 每小时一个桶)
// - L3: 日级聚合 (最近 1 年, 每天一个桶)
//
// 优点:
// 1. 固定内存使用 (约 50KB per query-version pair)
// 2. 自动老化: 细粒度数据被淘汰，粗粒度统计保留
// 3. 支持多时间范围的性能分析
// ============================================================================

// 聚合桶 (24 bytes)
struct SketchBucket {
    uint32_t bucket_id;       // 时间桶 ID (epoch / bucket_size)
    uint32_t count;           // 样本数量
    float sum_time_ms;        // 执行时间总和
    float sum_sq;             // 平方和 (用于计算方差)
    float min_time_ms;        // 最小值
    float max_time_ms;        // 最大值

    // 合并另一个桶
    void merge(const SketchBucket& other) {
        if (other.count == 0) return;
        if (count == 0) {
            *this = other;
            return;
        }
        count += other.count;
        sum_time_ms += other.sum_time_ms;
        sum_sq += other.sum_sq;
        min_time_ms = std::min(min_time_ms, other.min_time_ms);
        max_time_ms = std::max(max_time_ms, other.max_time_ms);
    }

    // 添加单个样本
    void add_sample(float time_ms) {
        count++;
        sum_time_ms += time_ms;
        sum_sq += time_ms * time_ms;
        if (count == 1) {
            min_time_ms = max_time_ms = time_ms;
        } else {
            min_time_ms = std::min(min_time_ms, time_ms);
            max_time_ms = std::max(max_time_ms, time_ms);
        }
    }

    // 计算统计值
    float avg() const { return count > 0 ? sum_time_ms / count : 0; }
    float variance() const {
        if (count < 2) return 0;
        float mean = avg();
        return (sum_sq / count) - (mean * mean);
    }
    float stddev() const { return std::sqrt(std::max(0.0f, variance())); }

    void reset() {
        bucket_id = 0;
        count = 0;
        sum_time_ms = 0;
        sum_sq = 0;
        min_time_ms = std::numeric_limits<float>::max();
        max_time_ms = 0;
    }
};

// 时间尺度 Sketch (单个 query-version 对)
class TimeScaleSketch {
public:
    // 时间桶大小
    static constexpr uint32_t L0_BUCKET_SEC = 1;       // 1 秒
    static constexpr uint32_t L1_BUCKET_SEC = 60;      // 1 分钟
    static constexpr uint32_t L2_BUCKET_SEC = 3600;    // 1 小时
    static constexpr uint32_t L3_BUCKET_SEC = 86400;   // 1 天

    // 环形缓冲区容量
    static constexpr size_t L0_CAPACITY = 3600;   // 1 小时的秒级数据
    static constexpr size_t L1_CAPACITY = 1440;   // 24 小时的分钟级数据
    static constexpr size_t L2_CAPACITY = 168;    // 7 天的小时级数据
    static constexpr size_t L3_CAPACITY = 365;    // 1 年的日级数据

    TimeScaleSketch() {
        reset();
    }

    // 添加一个性能样本
    void add_sample(float time_ms, uint64_t timestamp_sec) {
        // L0: 原始数据 (秒级桶)
        uint32_t l0_bucket = static_cast<uint32_t>(timestamp_sec / L0_BUCKET_SEC);
        add_to_level(l0_buckets_, l0_head_, L0_CAPACITY, l0_bucket, time_ms);

        // L1: 分钟级聚合
        uint32_t l1_bucket = static_cast<uint32_t>(timestamp_sec / L1_BUCKET_SEC);
        add_to_level(l1_buckets_, l1_head_, L1_CAPACITY, l1_bucket, time_ms);

        // L2: 小时级聚合
        uint32_t l2_bucket = static_cast<uint32_t>(timestamp_sec / L2_BUCKET_SEC);
        add_to_level(l2_buckets_, l2_head_, L2_CAPACITY, l2_bucket, time_ms);

        // L3: 日级聚合
        uint32_t l3_bucket = static_cast<uint32_t>(timestamp_sec / L3_BUCKET_SEC);
        add_to_level(l3_buckets_, l3_head_, L3_CAPACITY, l3_bucket, time_ms);

        total_samples_++;
    }

    // 获取最近 N 秒的统计
    SketchBucket get_recent_stats(uint64_t current_sec, uint32_t window_sec) const {
        SketchBucket result;
        result.reset();

        uint32_t start_bucket = static_cast<uint32_t>((current_sec - window_sec) / L0_BUCKET_SEC);

        // 根据时间窗口选择合适的层级
        if (window_sec <= 3600) {
            // 最近 1 小时: 使用 L0
            merge_from_level(result, l0_buckets_, l0_head_, L0_CAPACITY, start_bucket);
        } else if (window_sec <= 86400) {
            // 最近 24 小时: 使用 L1
            start_bucket = static_cast<uint32_t>((current_sec - window_sec) / L1_BUCKET_SEC);
            merge_from_level(result, l1_buckets_, l1_head_, L1_CAPACITY, start_bucket);
        } else if (window_sec <= 604800) {
            // 最近 7 天: 使用 L2
            start_bucket = static_cast<uint32_t>((current_sec - window_sec) / L2_BUCKET_SEC);
            merge_from_level(result, l2_buckets_, l2_head_, L2_CAPACITY, start_bucket);
        } else {
            // 更长: 使用 L3
            start_bucket = static_cast<uint32_t>((current_sec - window_sec) / L3_BUCKET_SEC);
            merge_from_level(result, l3_buckets_, l3_head_, L3_CAPACITY, start_bucket);
        }

        return result;
    }

    // 获取各层级统计摘要 (用于持久化)
    struct LevelStats {
        size_t bucket_count;
        uint32_t oldest_bucket;
        uint32_t newest_bucket;
        uint64_t total_samples;
    };

    LevelStats get_level_stats(int level) const {
        LevelStats stats{};
        switch (level) {
            case 0:
                stats = compute_level_stats(l0_buckets_, l0_head_, L0_CAPACITY);
                break;
            case 1:
                stats = compute_level_stats(l1_buckets_, l1_head_, L1_CAPACITY);
                break;
            case 2:
                stats = compute_level_stats(l2_buckets_, l2_head_, L2_CAPACITY);
                break;
            case 3:
                stats = compute_level_stats(l3_buckets_, l3_head_, L3_CAPACITY);
                break;
        }
        return stats;
    }

    uint64_t total_samples() const { return total_samples_; }

    void reset() {
        for (auto& b : l0_buckets_) b.reset();
        for (auto& b : l1_buckets_) b.reset();
        for (auto& b : l2_buckets_) b.reset();
        for (auto& b : l3_buckets_) b.reset();
        l0_head_ = l1_head_ = l2_head_ = l3_head_ = 0;
        total_samples_ = 0;
    }

    // 获取桶数据 (用于持久化)
    const std::array<SketchBucket, L1_CAPACITY>& get_l1_buckets() const { return l1_buckets_; }
    const std::array<SketchBucket, L2_CAPACITY>& get_l2_buckets() const { return l2_buckets_; }
    const std::array<SketchBucket, L3_CAPACITY>& get_l3_buckets() const { return l3_buckets_; }

    // 从持久化数据恢复
    void restore_bucket(int level, size_t idx, const SketchBucket& bucket) {
        switch (level) {
            case 1:
                if (idx < L1_CAPACITY) l1_buckets_[idx] = bucket;
                break;
            case 2:
                if (idx < L2_CAPACITY) l2_buckets_[idx] = bucket;
                break;
            case 3:
                if (idx < L3_CAPACITY) l3_buckets_[idx] = bucket;
                break;
        }
    }

private:
    // 添加样本到指定层级
    template<size_t N>
    void add_to_level(std::array<SketchBucket, N>& buckets, size_t& head,
                      size_t capacity, uint32_t bucket_id, float time_ms) {
        // 查找或创建桶
        size_t idx = head;
        bool found = false;

        // 检查最近的桶是否匹配
        if (buckets[idx].bucket_id == bucket_id && buckets[idx].count > 0) {
            buckets[idx].add_sample(time_ms);
            found = true;
        } else if (buckets[idx].bucket_id < bucket_id || buckets[idx].count == 0) {
            // 需要新桶
            head = (head + 1) % capacity;
            buckets[head].reset();
            buckets[head].bucket_id = bucket_id;
            buckets[head].add_sample(time_ms);
            found = true;
        }

        if (!found) {
            // 旧数据，跳过 (不应该发生)
        }
    }

    // 从层级合并统计
    template<size_t N>
    void merge_from_level(SketchBucket& result,
                          const std::array<SketchBucket, N>& buckets,
                          size_t head, size_t capacity,
                          uint32_t start_bucket) const {
        for (size_t i = 0; i < capacity; ++i) {
            const auto& b = buckets[i];
            if (b.count > 0 && b.bucket_id >= start_bucket) {
                result.merge(b);
            }
        }
    }

    template<size_t N>
    LevelStats compute_level_stats(const std::array<SketchBucket, N>& buckets,
                                   size_t head, size_t capacity) const {
        LevelStats stats{};
        stats.oldest_bucket = UINT32_MAX;
        stats.newest_bucket = 0;

        for (size_t i = 0; i < capacity; ++i) {
            const auto& b = buckets[i];
            if (b.count > 0) {
                stats.bucket_count++;
                stats.total_samples += b.count;
                stats.oldest_bucket = std::min(stats.oldest_bucket, b.bucket_id);
                stats.newest_bucket = std::max(stats.newest_bucket, b.bucket_id);
            }
        }
        return stats;
    }

    // L0: 秒级原始数据
    std::array<SketchBucket, L0_CAPACITY> l0_buckets_;
    size_t l0_head_ = 0;

    // L1: 分钟级聚合
    std::array<SketchBucket, L1_CAPACITY> l1_buckets_;
    size_t l1_head_ = 0;

    // L2: 小时级聚合
    std::array<SketchBucket, L2_CAPACITY> l2_buckets_;
    size_t l2_head_ = 0;

    // L3: 日级聚合
    std::array<SketchBucket, L3_CAPACITY> l3_buckets_;
    size_t l3_head_ = 0;

    uint64_t total_samples_ = 0;
};

// 查询-版本 Sketch 管理器
class SketchManager {
public:
    // 获取或创建指定 query-version 的 Sketch
    TimeScaleSketch& get_sketch(uint32_t query_hash, uint16_t version_id) {
        uint64_t key = make_key(query_hash, version_id);
        return sketches_[key];
    }

    // 添加性能样本
    void add_sample(uint32_t query_hash, uint16_t version_id,
                    float time_ms, uint64_t timestamp_sec) {
        get_sketch(query_hash, version_id).add_sample(time_ms, timestamp_sec);
    }

    // 获取最近统计
    SketchBucket get_recent_stats(uint32_t query_hash, uint16_t version_id,
                                   uint64_t current_sec, uint32_t window_sec) const {
        uint64_t key = make_key(query_hash, version_id);
        auto it = sketches_.find(key);
        if (it == sketches_.end()) {
            SketchBucket empty;
            empty.reset();
            return empty;
        }
        return it->second.get_recent_stats(current_sec, window_sec);
    }

    // 获取所有 sketch 的 key
    std::vector<std::pair<uint32_t, uint16_t>> get_all_keys() const {
        std::vector<std::pair<uint32_t, uint16_t>> keys;
        for (const auto& kv : sketches_) {
            keys.emplace_back(
                static_cast<uint32_t>(kv.first >> 16),
                static_cast<uint16_t>(kv.first & 0xFFFF)
            );
        }
        return keys;
    }

    size_t size() const { return sketches_.size(); }

    // 清除所有数据
    void clear() { sketches_.clear(); }

    // 获取内存使用估算 (KB)
    size_t memory_estimate_kb() const {
        // 每个 TimeScaleSketch 约:
        // L0: 3600 * 24 = 86KB
        // L1: 1440 * 24 = 35KB
        // L2: 168 * 24 = 4KB
        // L3: 365 * 24 = 9KB
        // Total: ~134KB per sketch
        return sketches_.size() * 134;
    }

private:
    static uint64_t make_key(uint32_t query_hash, uint16_t version_id) {
        return (static_cast<uint64_t>(query_hash) << 16) | version_id;
    }

    std::unordered_map<uint64_t, TimeScaleSketch> sketches_;
};

// ============================================================================
// 性能统计摘要 (常驻内存，定期更新)
// ============================================================================

struct VersionSummary {
    uint16_t version_id;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float stddev_ms;
    uint32_t sample_count;
    uint32_t last_update_offset;    // 相对时间戳

    void update(float new_time, uint32_t current_offset) {
        // Welford's online algorithm for incremental stddev
        sample_count++;
        float delta = new_time - avg_time_ms;
        avg_time_ms += delta / sample_count;
        float delta2 = new_time - avg_time_ms;
        m2_ += delta * delta2;
        stddev_ms = sample_count > 1 ? std::sqrt(m2_ / (sample_count - 1)) : 0.0f;

        min_time_ms = std::min(min_time_ms, new_time);
        max_time_ms = std::max(max_time_ms, new_time);
        last_update_offset = current_offset;
    }

    void reset() {
        avg_time_ms = 0;
        min_time_ms = std::numeric_limits<float>::max();
        max_time_ms = 0;
        stddev_ms = 0;
        sample_count = 0;
        m2_ = 0;
    }

private:
    float m2_ = 0;  // For Welford's algorithm
};

// ============================================================================
// 查询统计 (每个查询一个)
// ============================================================================

struct QueryStats {
    uint32_t query_hash;
    std::unordered_map<uint16_t, VersionSummary> versions;

    // 获取最优版本
    uint16_t get_best_version() const {
        uint16_t best = 0;
        float best_time = std::numeric_limits<float>::max();

        for (const auto& kv : versions) {
            if (kv.second.sample_count >= 3 && kv.second.avg_time_ms < best_time) {
                best_time = kv.second.avg_time_ms;
                best = kv.first;
            }
        }
        return best;
    }

    // 获取版本置信度
    float get_confidence(uint16_t version_id) const {
        auto it = versions.find(version_id);
        if (it == versions.end() || it->second.sample_count < 3) {
            return 0.0f;
        }

        const auto& v = it->second;
        float sample_conf = std::min(1.0f, v.sample_count / 30.0f);
        float cv = v.avg_time_ms > 0 ? v.stddev_ms / v.avg_time_ms : 1.0f;
        float cv_conf = std::max(0.0f, 1.0f - cv);

        return 0.6f * sample_conf + 0.4f * cv_conf;
    }
};

// ============================================================================
// 算子元数据 (紧凑版)
// ============================================================================

struct CompactOperatorMeta {
    uint16_t operator_id;
    uint16_t version_id;
    float startup_cost_ms;
    float per_row_cost_us;
    uint32_t min_rows;
    uint32_t max_rows;

    float estimate_cost(uint32_t rows) const {
        return startup_cost_ms + (per_row_cost_us * rows / 1000.0f);
    }
};

// ============================================================================
// 版本选择记录 (紧凑版, 16 bytes)
// ============================================================================

struct CompactSelection {
    uint32_t query_hash;
    uint16_t version_id;
    uint16_t flags;
    uint32_t actual_rows;
    uint32_t timestamp_offset;
};

// ============================================================================
// 轻量级系统目录
// ============================================================================

class SystemCatalog {
public:
    static SystemCatalog& instance() {
        static SystemCatalog catalog;
        return catalog;
    }

    // ========================================================================
    // 性能数据记录 (轻量级)
    // ========================================================================

    void record_metric(
        const std::string& query_id,
        const std::string& version,
        double execution_time_ms,
        size_t rows_processed,
        size_t memory_bytes = 0
    ) {
        std::lock_guard<std::mutex> lock(mutex_);

        uint32_t qhash = hash_query_id(query_id);
        uint16_t vid = VersionRegistry::instance().get_or_create_id(version);

        // 更新统计摘要
        auto& stats = query_stats_[qhash];
        stats.query_hash = qhash;

        auto& summary = stats.versions[vid];
        summary.version_id = vid;

        uint32_t ts_offset = get_timestamp_offset();
        summary.update(static_cast<float>(execution_time_ms), ts_offset);

        // 写入全局环形缓冲区
        CompactMetric metric;
        metric.query_hash = qhash;
        metric.version_id = vid;
        metric.flags = 0;
        metric.execution_time_ms = static_cast<float>(execution_time_ms);
        metric.rows_processed = static_cast<uint32_t>(std::min(rows_processed, size_t(UINT32_MAX)));
        metric.startup_cost_ms = static_cast<float>(execution_time_ms * 0.1);
        metric.per_row_cost_us = rows_processed > 0 ?
            static_cast<float>((execution_time_ms * 1000.0) / rows_processed) : 0;
        metric.timestamp_offset = ts_offset;
        metric.memory_kb = static_cast<uint32_t>(memory_bytes / 1024);

        global_metrics_.push(metric);
        insert_count_++;

        // 写入时间尺度 Sketch (自动卷积聚合)
        uint64_t timestamp_sec = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now().time_since_epoch()).count()
        );
        sketch_manager_.add_sample(qhash, vid,
            static_cast<float>(execution_time_ms), timestamp_sec);

        // 采样式检查
        if (insert_count_ % policy_.sample_interval == 0) {
            maybe_cleanup_locked();
        }
    }


    // ========================================================================
    // 性能查询
    // ========================================================================

    struct VersionStats {
        std::string version;
        float median_time_ms;
        float avg_time_ms;
        float stddev_ms;
        float min_time_ms;
        float max_time_ms;
        size_t sample_count;
    };

    VersionStats get_version_stats(const std::string& query_id, const std::string& version) const {
        std::lock_guard<std::mutex> lock(mutex_);

        VersionStats result{};
        result.version = version;

        uint32_t qhash = hash_query_id(query_id);
        uint16_t vid = VersionRegistry::instance().get_or_create_id(version);

        auto it = query_stats_.find(qhash);
        if (it == query_stats_.end()) return result;

        auto vit = it->second.versions.find(vid);
        if (vit == it->second.versions.end()) return result;

        const auto& s = vit->second;
        result.avg_time_ms = s.avg_time_ms;
        result.median_time_ms = s.avg_time_ms;  // 近似
        result.stddev_ms = s.stddev_ms;
        result.min_time_ms = s.min_time_ms;
        result.max_time_ms = s.max_time_ms;
        result.sample_count = s.sample_count;

        return result;
    }

    float get_version_confidence(const std::string& query_id, const std::string& version) const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint32_t qhash = hash_query_id(query_id);
        uint16_t vid = VersionRegistry::instance().get_or_create_id(version);

        auto it = query_stats_.find(qhash);
        if (it == query_stats_.end()) return 0.0f;

        return it->second.get_confidence(vid);
    }

    // ========================================================================
    // 最优版本选择
    // ========================================================================

    std::pair<std::string, double> select_best_version(
        const std::string& query_id,
        size_t estimated_rows,
        const std::vector<std::string>& candidates
    ) const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint32_t qhash = hash_query_id(query_id);
        auto it = query_stats_.find(qhash);

        std::string best_version;
        double best_time = std::numeric_limits<double>::max();

        for (const auto& version : candidates) {
            uint16_t vid = VersionRegistry::instance().get_or_create_id(version);

            if (it != query_stats_.end()) {
                auto vit = it->second.versions.find(vid);
                if (vit != it->second.versions.end() && vit->second.sample_count >= 3) {
                    double est_time = vit->second.avg_time_ms;
                    if (est_time < best_time) {
                        best_time = est_time;
                        best_version = version;
                    }
                    continue;
                }
            }

            // 回退到算子元数据
            auto oit = operator_meta_.find(vid);
            if (oit != operator_meta_.end()) {
                double est_time = oit->second.estimate_cost(static_cast<uint32_t>(estimated_rows));
                if (est_time < best_time) {
                    best_time = est_time;
                    best_version = version;
                }
            }
        }

        return {best_version, best_time};
    }

    // ========================================================================
    // 版本选择记录
    // ========================================================================

    void record_selection(const std::string& query_id, const std::string& version,
                          size_t actual_rows) {
        std::lock_guard<std::mutex> lock(mutex_);

        CompactSelection sel;
        sel.query_hash = hash_query_id(query_id);
        sel.version_id = VersionRegistry::instance().get_or_create_id(version);
        sel.flags = 0;
        sel.actual_rows = static_cast<uint32_t>(std::min(actual_rows, size_t(UINT32_MAX)));
        sel.timestamp_offset = get_timestamp_offset();

        selections_.push(sel);
    }

    // ========================================================================
    // 算子元数据注册
    // ========================================================================

    void register_operator(uint16_t version_id, float startup_ms, float per_row_us,
                           uint32_t min_rows = 0, uint32_t max_rows = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        CompactOperatorMeta meta;
        meta.operator_id = version_id;
        meta.version_id = version_id;
        meta.startup_cost_ms = startup_ms;
        meta.per_row_cost_us = per_row_us;
        meta.min_rows = min_rows;
        meta.max_rows = max_rows;

        operator_meta_[version_id] = meta;
    }

    void register_operator(const std::string& version, float startup_ms, float per_row_us,
                           uint32_t min_rows = 0, uint32_t max_rows = 0) {
        uint16_t vid = VersionRegistry::instance().get_or_create_id(version);
        register_operator(vid, startup_ms, per_row_us, min_rows, max_rows);
    }

    // ========================================================================
    // 算子成本估算 (优化器调用)
    // ========================================================================

    /**
     * 根据算子名称估算执行成本
     *
     * @param operator_name 算子名称 (如 "V54-NativeDoubleSIMDFilter")
     * @param rows 预计处理行数
     * @return 估算执行时间 (ms), -1 表示算子未注册
     */
    double estimate_operator_cost(const std::string& operator_name, size_t rows) const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint16_t vid = VersionRegistry::instance().get_or_create_id(operator_name);
        auto it = operator_meta_.find(vid);
        if (it == operator_meta_.end()) {
            return -1.0;  // 算子未注册
        }

        return it->second.estimate_cost(static_cast<uint32_t>(rows));
    }

    /**
     * 获取算子元数据
     *
     * @param operator_name 算子名称
     * @return 算子元数据 (如果存在)
     */
    const CompactOperatorMeta* get_operator_meta(const std::string& operator_name) const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint16_t vid = VersionRegistry::instance().get_or_create_id(operator_name);
        auto it = operator_meta_.find(vid);
        return it != operator_meta_.end() ? &it->second : nullptr;
    }

    /**
     * 检查算子是否适用于给定行数
     *
     * @param operator_name 算子名称
     * @param rows 预计处理行数
     * @return true 如果算子适用
     */
    bool is_operator_applicable(const std::string& operator_name, size_t rows) const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint16_t vid = VersionRegistry::instance().get_or_create_id(operator_name);
        auto it = operator_meta_.find(vid);
        if (it == operator_meta_.end()) {
            return false;
        }

        const auto& meta = it->second;
        if (rows < meta.min_rows) return false;
        if (meta.max_rows > 0 && rows > meta.max_rows) return false;
        return true;
    }

    /**
     * 从系统表选择最优算子 (基于成本模型)
     *
     * @param candidates 候选算子名称列表
     * @param rows 预计处理行数
     * @return 最优算子名称和估算成本
     */
    std::pair<std::string, double> select_optimal_operator(
        const std::vector<std::string>& candidates,
        size_t rows
    ) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string best_op;
        double best_cost = std::numeric_limits<double>::max();

        for (const auto& op_name : candidates) {
            uint16_t vid = VersionRegistry::instance().get_or_create_id(op_name);
            auto it = operator_meta_.find(vid);
            if (it == operator_meta_.end()) continue;

            const auto& meta = it->second;
            // 检查适用性
            if (rows < meta.min_rows) continue;
            if (meta.max_rows > 0 && rows > meta.max_rows) continue;

            double cost = meta.estimate_cost(static_cast<uint32_t>(rows));
            if (cost < best_cost) {
                best_cost = cost;
                best_op = op_name;
            }
        }

        return {best_op, best_cost};
    }

    // ========================================================================
    // 清理
    // ========================================================================

    void set_policy(const LightweightCleanupPolicy& policy) {
        std::lock_guard<std::mutex> lock(mutex_);
        policy_ = policy;
    }

    size_t cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        return cleanup_locked();
    }

    // ========================================================================
    // 统计
    // ========================================================================

    struct CatalogStats {
        size_t global_buffer_size;
        size_t global_buffer_capacity;
        size_t queries_tracked;
        size_t total_version_summaries;
        size_t selections_size;
        size_t memory_estimate_kb;
    };

    CatalogStats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        CatalogStats stats{};
        stats.global_buffer_size = global_metrics_.size();
        stats.global_buffer_capacity = global_metrics_.capacity();
        stats.queries_tracked = query_stats_.size();

        for (const auto& kv : query_stats_) {
            stats.total_version_summaries += kv.second.versions.size();
        }

        stats.selections_size = selections_.size();

        // 估算内存使用
        stats.memory_estimate_kb =
            (global_metrics_.capacity() * sizeof(CompactMetric) +
             query_stats_.size() * (sizeof(QueryStats) + 4 * sizeof(VersionSummary)) +
             selections_.capacity() * sizeof(CompactSelection) +
             operator_meta_.size() * sizeof(CompactOperatorMeta)) / 1024;

        return stats;
    }

    // ========================================================================
    // DuckDB 持久化
    // ========================================================================

    /**
     * 初始化 DuckDB 持久化
     * @param db_path 数据库文件路径 (如 ".solar/thunderduck.db")
     * @return 是否成功
     */
    bool init_persistence(const std::string& db_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            // 创建持久化数据库
            persistence_db_ = std::make_unique<duckdb::DuckDB>(db_path);
            persistence_con_ = std::make_unique<duckdb::Connection>(*persistence_db_);

            // 创建系统表
            create_tables();

            // 从数据库加载已有数据
            load_from_db();

            persistence_path_ = db_path;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "DuckDB persistence init failed: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * 保存当前状态到 DuckDB
     */
    bool save_to_db() const {
        if (!persistence_con_) return false;

        std::lock_guard<std::mutex> lock(mutex_);

        try {
            const auto& reg = VersionRegistry::instance();

            // 清空并重新插入算子元数据
            persistence_con_->Query("DELETE FROM td_operators");

            for (const auto& kv : operator_meta_) {
                std::string name = reg.get_version(kv.first);
                std::string sql = "INSERT INTO td_operators VALUES (" +
                    std::to_string(kv.first) + ", '" + name + "', " +
                    std::to_string(kv.second.startup_cost_ms) + ", " +
                    std::to_string(kv.second.per_row_cost_us) + ", " +
                    std::to_string(kv.second.min_rows) + ", " +
                    std::to_string(kv.second.max_rows) + ")";
                persistence_con_->Query(sql);
            }

            // 保存查询统计
            persistence_con_->Query("DELETE FROM td_query_stats");

            for (const auto& kv : query_stats_) {
                for (const auto& vkv : kv.second.versions) {
                    const auto& s = vkv.second;
                    std::string sql = "INSERT INTO td_query_stats VALUES (" +
                        std::to_string(kv.first) + ", " +
                        std::to_string(vkv.first) + ", '" + reg.get_version(vkv.first) + "', " +
                        std::to_string(s.avg_time_ms) + ", " +
                        std::to_string(s.min_time_ms) + ", " +
                        std::to_string(s.max_time_ms) + ", " +
                        std::to_string(s.stddev_ms) + ", " +
                        std::to_string(s.sample_count) + ")";
                    persistence_con_->Query(sql);
                }
            }

            // 保存时间尺度 Sketch (L1-L3 聚合数据)
            persistence_con_->Query("DELETE FROM td_sketch_buckets");

            for (const auto& key : sketch_manager_.get_all_keys()) {
                uint32_t qhash = key.first;
                uint16_t vid = key.second;

                // 获取 Sketch 并保存各层聚合桶
                // 注意: L0 不持久化，重启后自然过期
                auto save_level = [&](int level, const auto& buckets) {
                    for (size_t i = 0; i < buckets.size(); ++i) {
                        const auto& b = buckets[i];
                        if (b.count > 0) {
                            std::string sql = "INSERT INTO td_sketch_buckets VALUES (" +
                                std::to_string(qhash) + ", " +
                                std::to_string(vid) + ", " +
                                std::to_string(level) + ", " +
                                std::to_string(b.bucket_id) + ", " +
                                std::to_string(b.count) + ", " +
                                std::to_string(b.sum_time_ms) + ", " +
                                std::to_string(b.sum_sq) + ", " +
                                std::to_string(b.min_time_ms) + ", " +
                                std::to_string(b.max_time_ms) + ")";
                            persistence_con_->Query(sql);
                        }
                    }
                };

                const auto& sketch = const_cast<SketchManager&>(sketch_manager_).get_sketch(qhash, vid);
                save_level(1, sketch.get_l1_buckets());
                save_level(2, sketch.get_l2_buckets());
                save_level(3, sketch.get_l3_buckets());
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Save to DuckDB failed: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * 从 DuckDB 加载数据
     */
    bool load_from_db() {
        if (!persistence_con_) return false;

        try {
            // 加载算子元数据
            auto result = persistence_con_->Query(
                "SELECT id, name, startup_ms, per_row_us, min_rows, max_rows FROM td_operators");

            if (result->HasError()) return false;

            while (auto chunk = result->Fetch()) {
                for (size_t i = 0; i < chunk->size(); ++i) {
                    uint16_t id = chunk->GetValue(0, i).GetValue<int32_t>();
                    std::string name = chunk->GetValue(1, i).GetValue<std::string>();
                    float startup_ms = chunk->GetValue(2, i).GetValue<float>();
                    float per_row_us = chunk->GetValue(3, i).GetValue<float>();
                    uint32_t min_rows = chunk->GetValue(4, i).GetValue<int32_t>();
                    uint32_t max_rows = chunk->GetValue(5, i).GetValue<int32_t>();

                    // 注册到版本注册表
                    VersionRegistry::instance().get_or_create_id(name);

                    CompactOperatorMeta meta;
                    meta.operator_id = id;
                    meta.version_id = id;
                    meta.startup_cost_ms = startup_ms;
                    meta.per_row_cost_us = per_row_us;
                    meta.min_rows = min_rows;
                    meta.max_rows = max_rows;
                    operator_meta_[id] = meta;
                }
            }

            // 加载查询统计
            auto stats_result = persistence_con_->Query(
                "SELECT query_hash, version_id, version_name, avg_time_ms, min_time_ms, "
                "max_time_ms, stddev_ms, sample_count FROM td_query_stats");

            if (!stats_result->HasError()) {
                while (auto chunk = stats_result->Fetch()) {
                    for (size_t i = 0; i < chunk->size(); ++i) {
                        uint32_t qhash = chunk->GetValue(0, i).GetValue<int64_t>();
                        uint16_t vid = chunk->GetValue(1, i).GetValue<int32_t>();

                        auto& stats = query_stats_[qhash];
                        stats.query_hash = qhash;

                        auto& summary = stats.versions[vid];
                        summary.version_id = vid;
                        summary.avg_time_ms = chunk->GetValue(3, i).GetValue<float>();
                        summary.min_time_ms = chunk->GetValue(4, i).GetValue<float>();
                        summary.max_time_ms = chunk->GetValue(5, i).GetValue<float>();
                        summary.stddev_ms = chunk->GetValue(6, i).GetValue<float>();
                        summary.sample_count = chunk->GetValue(7, i).GetValue<int32_t>();
                    }
                }
            }

            // 加载时间尺度 Sketch (L1-L3 聚合数据)
            auto sketch_result = persistence_con_->Query(
                "SELECT query_hash, version_id, level, bucket_id, count, "
                "sum_time_ms, sum_sq, min_time_ms, max_time_ms FROM td_sketch_buckets");

            if (!sketch_result->HasError()) {
                while (auto chunk = sketch_result->Fetch()) {
                    for (size_t i = 0; i < chunk->size(); ++i) {
                        uint32_t qhash = chunk->GetValue(0, i).GetValue<int64_t>();
                        uint16_t vid = chunk->GetValue(1, i).GetValue<int32_t>();
                        int level = chunk->GetValue(2, i).GetValue<int32_t>();
                        uint32_t bucket_id = chunk->GetValue(3, i).GetValue<int64_t>();

                        SketchBucket bucket;
                        bucket.bucket_id = bucket_id;
                        bucket.count = chunk->GetValue(4, i).GetValue<int32_t>();
                        bucket.sum_time_ms = chunk->GetValue(5, i).GetValue<float>();
                        bucket.sum_sq = chunk->GetValue(6, i).GetValue<float>();
                        bucket.min_time_ms = chunk->GetValue(7, i).GetValue<float>();
                        bucket.max_time_ms = chunk->GetValue(8, i).GetValue<float>();

                        // 恢复到对应的 Sketch
                        auto& sketch = sketch_manager_.get_sketch(qhash, vid);
                        // 简单方式: 直接存储 (假设 bucket_id 可以用作索引)
                        // 实际应该根据 bucket_id 计算正确位置
                        sketch.restore_bucket(level, bucket_id % 1440, bucket);
                    }
                }
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Load from DuckDB failed: " << e.what() << "\n";
            return false;
        }
    }

    /**
     * 获取持久化路径
     */
    const std::string& get_persistence_path() const {
        return persistence_path_;
    }

    /**
     * 检查是否已初始化持久化
     */
    bool is_persistence_initialized() const {
        return persistence_con_ != nullptr;
    }

    // 兼容旧接口
    void set_persistence_path(const std::string& path) {
        init_persistence(path);
    }

    bool save_to_file(const std::string&) const {
        return save_to_db();
    }

private:
    void create_tables() {
        // 创建算子元数据表
        persistence_con_->Query(R"(
            CREATE TABLE IF NOT EXISTS td_operators (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                startup_ms FLOAT,
                per_row_us FLOAT,
                min_rows INTEGER,
                max_rows INTEGER
            )
        )");

        // 创建查询统计表
        persistence_con_->Query(R"(
            CREATE TABLE IF NOT EXISTS td_query_stats (
                query_hash BIGINT,
                version_id INTEGER,
                version_name VARCHAR,
                avg_time_ms FLOAT,
                min_time_ms FLOAT,
                max_time_ms FLOAT,
                stddev_ms FLOAT,
                sample_count INTEGER,
                PRIMARY KEY (query_hash, version_id)
            )
        )");

        // 创建性能指标表 (环形缓冲区持久化)
        persistence_con_->Query(R"(
            CREATE TABLE IF NOT EXISTS td_metrics (
                id INTEGER PRIMARY KEY,
                query_hash BIGINT,
                version_id INTEGER,
                execution_time_ms FLOAT,
                rows_processed BIGINT,
                timestamp_epoch_ms BIGINT
            )
        )");

        // 创建时间尺度 Sketch 表 (L1-L3 聚合数据)
        // L0 不持久化 (仅保留最近 1 小时的原始数据)
        persistence_con_->Query(R"(
            CREATE TABLE IF NOT EXISTS td_sketch_buckets (
                query_hash BIGINT,
                version_id INTEGER,
                level INTEGER,
                bucket_id BIGINT,
                count INTEGER,
                sum_time_ms FLOAT,
                sum_sq FLOAT,
                min_time_ms FLOAT,
                max_time_ms FLOAT,
                PRIMARY KEY (query_hash, version_id, level, bucket_id)
            )
        )");
    }

    // DuckDB 持久化
    mutable std::unique_ptr<duckdb::DuckDB> persistence_db_;
    mutable std::unique_ptr<duckdb::Connection> persistence_con_;

public:
    // ========================================================================
    // 显示所有数据
    // ========================================================================

    /**
     * 打印系统表所有数据
     */
    void print_all() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           ThunderDuck System Catalog - All Data                  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

        // 1. 算子元数据
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 1. Registered Operators (" << operator_meta_.size() << " total)                           │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ ID  │ Name                          │ Startup │ PerRow │ Rows   │\n";
        std::cout << "├─────┼───────────────────────────────┼─────────┼────────┼────────┤\n";

        const auto& reg = VersionRegistry::instance();
        for (const auto& kv : operator_meta_) {
            const auto& m = kv.second;
            std::string name = reg.get_version(kv.first);
            if (name.length() > 29) name = name.substr(0, 26) + "...";

            std::cout << "│ " << std::setw(3) << kv.first << " │ "
                      << std::left << std::setw(29) << name << " │ "
                      << std::right << std::setw(5) << std::fixed << std::setprecision(2) << m.startup_cost_ms << "ms │ "
                      << std::setw(5) << std::setprecision(4) << m.per_row_cost_us << "us │ "
                      << std::setw(6) << m.min_rows << " │\n";
        }
        std::cout << "└─────┴───────────────────────────────┴─────────┴────────┴────────┘\n\n";

        // 2. 查询性能统计
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 2. Query Performance Stats (" << query_stats_.size() << " queries)                       │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";

        if (query_stats_.empty()) {
            std::cout << "│ (No performance data recorded yet)                              │\n";
        } else {
            for (const auto& kv : query_stats_) {
                std::cout << "│ Query Hash: 0x" << std::hex << kv.first << std::dec << "\n";
                for (const auto& vkv : kv.second.versions) {
                    const auto& s = vkv.second;
                    std::cout << "│   " << std::left << std::setw(25) << reg.get_version(vkv.first)
                              << " │ avg: " << std::right << std::setw(6) << std::setprecision(2) << s.avg_time_ms << "ms"
                              << " │ σ: " << std::setw(5) << s.stddev_ms << "ms"
                              << " │ n=" << std::setw(4) << s.sample_count << " │\n";
                }
            }
        }
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";

        // 3. 全局指标缓冲区
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 3. Global Metrics Buffer                                        │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Size: " << std::setw(6) << global_metrics_.size()
                  << " / " << std::setw(6) << global_metrics_.capacity() << " (ring buffer)                  │\n";

        auto recent = global_metrics_.get_recent(10);
        if (!recent.empty()) {
            std::cout << "│ Recent 10 metrics:                                              │\n";
            for (const auto& m : recent) {
                std::cout << "│   hash=0x" << std::hex << std::setw(8) << m.query_hash << std::dec
                          << " ver=" << std::setw(3) << m.version_id
                          << " time=" << std::setw(6) << std::setprecision(2) << m.execution_time_ms << "ms"
                          << " rows=" << std::setw(8) << m.rows_processed << " │\n";
            }
        }
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";

        // 4. 时间尺度 Sketch (多层聚合)
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 4. Time-Scale Sketch (Multi-Resolution Aggregation)            │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Query-Version pairs: " << std::setw(4) << sketch_manager_.size()
                  << " | Memory: ~" << std::setw(5) << sketch_manager_.memory_estimate_kb() << " KB          │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Layers:                                                         │\n";
        std::cout << "│   L0: Raw samples      (1 hour,  second-level, " << std::setw(5) << TimeScaleSketch::L0_CAPACITY << " buckets)  │\n";
        std::cout << "│   L1: Minute aggregate (24 hours, minute-level, " << std::setw(4) << TimeScaleSketch::L1_CAPACITY << " buckets)  │\n";
        std::cout << "│   L2: Hour aggregate   (7 days,  hour-level,   " << std::setw(4) << TimeScaleSketch::L2_CAPACITY << " buckets)  │\n";
        std::cout << "│   L3: Day aggregate    (1 year,  day-level,    " << std::setw(4) << TimeScaleSketch::L3_CAPACITY << " buckets)  │\n";
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";

        // 5. 版本选择历史
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 5. Version Selection History                                    │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Size: " << std::setw(6) << selections_.size()
                  << " / " << std::setw(6) << selections_.capacity() << "                                       │\n";
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";

        // 6. 内存使用
        auto stats = get_stats_locked();
        size_t total_memory = stats.memory_estimate_kb + sketch_manager_.memory_estimate_kb();
        std::cout << "┌─────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ 6. Memory Usage                                                 │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────┤\n";
        std::cout << "│ Base catalog:  " << std::setw(6) << stats.memory_estimate_kb << " KB                                       │\n";
        std::cout << "│ Sketch memory: " << std::setw(6) << sketch_manager_.memory_estimate_kb() << " KB                                       │\n";
        std::cout << "│ Total:         " << std::setw(6) << total_memory << " KB                                       │\n";
        std::cout << "│ Persistence: " << (persistence_path_.empty() ? "(not configured)" : persistence_path_) << "\n";
        std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";
    }

    /**
     * 获取所有算子列表 (用于外部显示)
     */
    std::vector<std::pair<std::string, CompactOperatorMeta>> get_all_operators() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::pair<std::string, CompactOperatorMeta>> result;

        const auto& reg = VersionRegistry::instance();
        for (const auto& kv : operator_meta_) {
            result.emplace_back(reg.get_version(kv.first), kv.second);
        }

        // 按名称排序
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        return result;
    }

private:
    CatalogStats get_stats_locked() const {
        CatalogStats stats{};
        stats.global_buffer_size = global_metrics_.size();
        stats.global_buffer_capacity = global_metrics_.capacity();
        stats.queries_tracked = query_stats_.size();

        for (const auto& kv : query_stats_) {
            stats.total_version_summaries += kv.second.versions.size();
        }

        stats.selections_size = selections_.size();
        stats.memory_estimate_kb =
            (global_metrics_.capacity() * sizeof(CompactMetric) +
             query_stats_.size() * (sizeof(QueryStats) + 4 * sizeof(VersionSummary)) +
             selections_.capacity() * sizeof(CompactSelection) +
             operator_meta_.size() * sizeof(CompactOperatorMeta)) / 1024;

        return stats;
    }

    mutable std::string persistence_path_;  // 持久化路径

    SystemCatalog() : base_time_(now()) {}

    // 查询 ID 哈希
    static uint32_t hash_query_id(const std::string& query_id) {
        uint32_t hash = 5381;
        for (char c : query_id) {
            hash = ((hash << 5) + hash) + static_cast<uint32_t>(c);
        }
        return hash;
    }

    // 获取相对时间戳 (秒)
    uint32_t get_timestamp_offset() const {
        auto diff = std::chrono::duration_cast<std::chrono::seconds>(now() - base_time_);
        return static_cast<uint32_t>(diff.count());
    }

    // 采样式清理检查
    void maybe_cleanup_locked() {
        // 估算内存使用
        size_t est_memory_kb =
            (global_metrics_.size() * sizeof(CompactMetric) +
             query_stats_.size() * 200) / 1024;  // 粗略估算

        if (est_memory_kb > policy_.memory_threshold_kb) {
            cleanup_locked();
        }
    }

    // 增量清理
    size_t cleanup_locked() {
        size_t cleaned = 0;
        uint32_t current_offset = get_timestamp_offset();
        uint32_t cutoff = current_offset > policy_.retention_seconds ?
                          current_offset - policy_.retention_seconds : 0;

        // 清理过期的版本统计 (增量，每次最多处理 batch_size 个查询)
        size_t processed = 0;
        for (auto it = query_stats_.begin();
             it != query_stats_.end() && processed < policy_.cleanup_batch_size;
             ++processed) {

            auto& versions = it->second.versions;
            for (auto vit = versions.begin(); vit != versions.end(); ) {
                if (vit->second.last_update_offset < cutoff &&
                    vit->second.sample_count < 5) {
                    // 样本少且过期的统计直接删除
                    vit = versions.erase(vit);
                    cleaned++;
                } else if (vit->second.last_update_offset < cutoff) {
                    // 样本多的统计衰减
                    vit->second.sample_count = std::max(3u, vit->second.sample_count / 2);
                    ++vit;
                } else {
                    ++vit;
                }
            }

            if (versions.empty()) {
                it = query_stats_.erase(it);
            } else {
                ++it;
            }
        }

        return cleaned;
    }

    mutable std::mutex mutex_;

    // 基准时间 (用于计算相对时间戳)
    Timestamp base_time_;

    // 全局环形缓冲区
    RingBuffer<CompactMetric, LightweightCleanupPolicy::GLOBAL_BUFFER_SIZE> global_metrics_;

    // 查询统计 (常驻内存)
    std::unordered_map<uint32_t, QueryStats> query_stats_;

    // 版本选择历史
    RingBuffer<CompactSelection, 512> selections_;

    // 算子元数据
    std::unordered_map<uint16_t, CompactOperatorMeta> operator_meta_;

    // 清理策略
    LightweightCleanupPolicy policy_;

    // 插入计数 (用于采样)
    size_t insert_count_ = 0;

    // 时间尺度 Sketch 管理器
    SketchManager sketch_manager_;
};

// ============================================================================
// 便捷函数
// ============================================================================

inline SystemCatalog& catalog() {
    return SystemCatalog::instance();
}

// ============================================================================
// 性能采集 RAII 助手
// ============================================================================

class PerformanceCollector {
public:
    PerformanceCollector(
        const std::string& query_id,
        const std::string& version,
        size_t estimated_rows = 0
    ) : query_id_(query_id), version_(version), rows_(estimated_rows) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void set_rows(size_t rows) { rows_ = rows; }

    ~PerformanceCollector() {
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start_).count();
        catalog().record_metric(query_id_, version_, time_ms, rows_);
    }

private:
    std::string query_id_;
    std::string version_;
    size_t rows_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// 兼容性别名 (保持 API 兼容)
// ============================================================================

// 完整版 PerformanceMetric (用于外部接口)
struct PerformanceMetric {
    std::string query_id;
    std::string version;
    std::string operator_name;
    double execution_time_ms;
    size_t rows_processed;
    size_t rows_output;
    size_t memory_used_bytes;
    double startup_cost_ms;
    double per_row_cost_us;
    Timestamp recorded_at;

    double compute_per_row_cost() const {
        if (rows_processed == 0) return 0.0;
        return (execution_time_ms * 1000.0) / rows_processed;
    }
};

struct OperatorMetadata {
    std::string operator_id;
    std::string operator_type;
    std::string version;
    std::string description;
    double avg_startup_cost_ms;
    double avg_per_row_cost_us;
    double stddev_per_row_cost_us;
    size_t avg_memory_per_row;
    size_t min_rows;
    size_t max_rows;
    double min_selectivity;
    double max_selectivity;
    size_t sample_count;
    Timestamp last_updated;

    double estimate_cost(size_t rows) const {
        return avg_startup_cost_ms + (avg_per_row_cost_us * rows / 1000.0);
    }
};

struct VersionSelection {
    std::string query_id;
    std::string selected_version;
    std::string selection_reason;
    size_t actual_rows;
    double actual_speedup;
    double predicted_speedup;
    Timestamp selected_at;
};

} // namespace catalog
} // namespace thunderduck
