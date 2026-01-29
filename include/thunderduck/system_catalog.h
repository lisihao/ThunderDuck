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

private:
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
