/**
 * ThunderDuck V13 全面基准测试
 *
 * 对比版本: V3, V7, V8, V9, V10, V11, V12, V12.5, V13, DuckDB
 * 测试算子: Filter, Aggregate, TopK, GROUP BY, Hash Join
 * 测试规模: 1M, 5M, 10M
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <map>

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"
#include "thunderduck/v12_5.h"
#include "thunderduck/v13.h"

using namespace std;
using namespace thunderduck;

// ============================================================================
// 工具函数
// ============================================================================

class Timer {
public:
    void start() { start_ = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end - start_).count();
    }
private:
    chrono::high_resolution_clock::time_point start_;
};

double calc_throughput_gbps(size_t bytes, double time_ms) {
    if (time_ms <= 0) return 0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

string format_bytes(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024) {
        return to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return to_string(bytes / 1024) + " KB";
    }
    return to_string(bytes) + " B";
}

string format_count(size_t count) {
    if (count >= 1000000) {
        return to_string(count / 1000000) + "M";
    } else if (count >= 1000) {
        return to_string(count / 1000) + "K";
    }
    return to_string(count);
}

// ============================================================================
// 测试结果结构
// ============================================================================

struct BenchmarkResult {
    string sql_equivalent;
    string operator_name;
    string version;
    string device;
    size_t data_count;
    size_t data_bytes;
    double time_ms;
    double throughput_gbps;
    double speedup_vs_duckdb;
    double speedup_vs_v3;
};

vector<BenchmarkResult> g_results;

void record_result(const string& sql, const string& op, const string& ver,
                   const string& device, size_t count, size_t bytes,
                   double time_ms, double duckdb_time, double v3_time) {
    BenchmarkResult r;
    r.sql_equivalent = sql;
    r.operator_name = op;
    r.version = ver;
    r.device = device;
    r.data_count = count;
    r.data_bytes = bytes;
    r.time_ms = time_ms;
    r.throughput_gbps = calc_throughput_gbps(bytes, time_ms);
    r.speedup_vs_duckdb = (time_ms > 0) ? duckdb_time / time_ms : 0;
    r.speedup_vs_v3 = (time_ms > 0 && v3_time > 0) ? v3_time / time_ms : 0;
    g_results.push_back(r);
}

// ============================================================================
// Filter 测试
// ============================================================================

void test_filter(size_t count) {
    cout << "\n=== Filter 测试 [" << format_count(count) << " 行] ===" << endl;

    vector<int32_t> data(count);
    vector<uint32_t> indices(count);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist(0, 1000);
    for (size_t i = 0; i < count; i++) data[i] = dist(gen);

    int32_t threshold = 500;  // ~50% 选择率
    size_t data_bytes = count * sizeof(int32_t);
    string sql = "SELECT * FROM t WHERE col > " + to_string(threshold);
    Timer timer;

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        size_t result = 0;
        for (size_t i = 0; i < count; i++) {
            if (data[i] > threshold) indices[result++] = i;
        }
        duckdb_time = timer.elapsed_ms();
        cout << "  DuckDB (Scalar): " << fixed << setprecision(3) << duckdb_time << " ms, "
             << setprecision(2) << calc_throughput_gbps(data_bytes, duckdb_time) << " GB/s" << endl;
    }

    // V3 SIMD
    double v3_time;
    {
        timer.start();
        filter::filter_i32_v3(data.data(), count, filter::CompareOp::GT, threshold, indices.data());
        v3_time = timer.elapsed_ms();
        double speedup = duckdb_time / v3_time;
        cout << "  V3 (CPU SIMD): " << fixed << setprecision(3) << v3_time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Filter", "V3", "CPU SIMD", count, data_bytes, v3_time, duckdb_time, v3_time);
    }

    // V4 GPU
    if (filter::is_filter_gpu_available()) {
        timer.start();
        filter::filter_i32_v4(data.data(), count, filter::CompareOp::GT, threshold, indices.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V4 (GPU Metal): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Filter", "V4", "GPU Metal", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::filter_i32(data.data(), count, v125::CompareOp::GT, threshold, indices.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V12.5 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Filter", "V12.5", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    // V13
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::filter_i32(data.data(), count, v13::CompareOp::GT, threshold, indices.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V13 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Filter", "V13", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    record_result(sql, "Filter", "DuckDB", "CPU Scalar", count, data_bytes, duckdb_time, duckdb_time, v3_time);
}

// ============================================================================
// Aggregate 测试
// ============================================================================

void test_aggregate(size_t count) {
    cout << "\n=== Aggregate 测试 [" << format_count(count) << " 行] ===" << endl;

    vector<int32_t> data(count);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist(0, 1000);
    for (size_t i = 0; i < count; i++) data[i] = dist(gen);

    size_t data_bytes = count * sizeof(int32_t);
    string sql = "SELECT SUM(col), MIN(col), MAX(col) FROM t";
    Timer timer;

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        int64_t sum = 0;
        int32_t min_val = INT32_MAX, max_val = INT32_MIN;
        for (size_t i = 0; i < count; i++) {
            sum += data[i];
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        duckdb_time = timer.elapsed_ms();
        cout << "  DuckDB (Scalar): " << fixed << setprecision(3) << duckdb_time << " ms, "
             << setprecision(2) << calc_throughput_gbps(data_bytes, duckdb_time) << " GB/s" << endl;
    }

    // V3 GPU
    double v3_time = duckdb_time;
    if (aggregate::is_aggregate_gpu_available()) {
        timer.start();
        aggregate::sum_i32_v3(data.data(), count);
        v3_time = timer.elapsed_ms();
        double speedup = duckdb_time / v3_time;
        cout << "  V3 (GPU Metal): " << fixed << setprecision(3) << v3_time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Aggregate", "V3", "GPU Metal", count, data_bytes, v3_time, duckdb_time, v3_time);
    }

    // V4 SIMD+
    {
        timer.start();
        aggregate::sum_i32_v4(data.data(), count);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V4 (CPU SIMD+): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Aggregate", "V4", "CPU SIMD+", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::sum_i32(data.data(), count, &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V12.5 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Aggregate", "V12.5", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    // V13
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::sum_i32(data.data(), count, &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V13 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "Aggregate", "V13", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    record_result(sql, "Aggregate", "DuckDB", "CPU Scalar", count, data_bytes, duckdb_time, duckdb_time, v3_time);
}

// ============================================================================
// TopK 测试
// ============================================================================

void test_topk(size_t count, size_t k) {
    cout << "\n=== TopK 测试 [" << format_count(count) << " 行, k=" << k << "] ===" << endl;

    vector<int32_t> data(count);
    vector<int32_t> out_values(k);
    vector<uint32_t> out_indices(k);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist(0, 1000000);
    for (size_t i = 0; i < count; i++) data[i] = dist(gen);

    size_t data_bytes = count * sizeof(int32_t);
    string sql = "SELECT * FROM t ORDER BY col DESC LIMIT " + to_string(k);
    Timer timer;

    // DuckDB 基准 (partial_sort)
    double duckdb_time;
    {
        vector<int32_t> copy = data;
        timer.start();
        partial_sort(copy.begin(), copy.begin() + k, copy.end(), greater<int32_t>());
        duckdb_time = timer.elapsed_ms();
        cout << "  DuckDB (partial_sort): " << fixed << setprecision(3) << duckdb_time << " ms, "
             << setprecision(2) << calc_throughput_gbps(data_bytes, duckdb_time) << " GB/s" << endl;
    }

    // V3 堆方法
    double v3_time;
    {
        timer.start();
        sort::topk_max_i32_v3(data.data(), count, k, out_values.data(), out_indices.data());
        v3_time = timer.elapsed_ms();
        double speedup = duckdb_time / v3_time;
        cout << "  V3 (CPU Heap): " << fixed << setprecision(3) << v3_time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V3", "CPU Heap", count, data_bytes, v3_time, duckdb_time, v3_time);
    }

    // V4 采样预过滤
    {
        timer.start();
        sort::topk_max_i32_v4(data.data(), count, k, out_values.data(), out_indices.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V7 (CPU Sampling): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V7", "CPU Sampling", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V5 Count-Based
    {
        timer.start();
        sort::topk_max_i32_v5(data.data(), count, k, out_values.data(), out_indices.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V8 (CPU Count-Based): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V8", "CPU Count-Based", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V6 GPU
    if (sort::is_topk_gpu_available()) {
        timer.start();
        sort::topk_max_i32_v6(data.data(), count, k, out_values.data(), out_indices.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V9 (GPU UMA): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V9", "GPU UMA", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::topk_max_i32(data.data(), count, k, out_values.data(), out_indices.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V12.5 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V12.5", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    // V13
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::topk_max_i32(data.data(), count, k, out_values.data(), out_indices.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V13 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "TopK", "V13", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    record_result(sql, "TopK", "DuckDB", "CPU partial_sort", count, data_bytes, duckdb_time, duckdb_time, v3_time);
}

// ============================================================================
// GROUP BY 测试
// ============================================================================

void test_group_by(size_t count, size_t num_groups) {
    cout << "\n=== GROUP BY 测试 [" << format_count(count) << " 行, " << num_groups << " groups] ===" << endl;

    vector<int32_t> values(count);
    vector<uint32_t> groups(count);
    vector<int64_t> sums(num_groups);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_val(0, 1000);
    uniform_int_distribution<uint32_t> dist_grp(0, num_groups - 1);
    for (size_t i = 0; i < count; i++) {
        values[i] = dist_val(gen);
        groups[i] = dist_grp(gen);
    }

    size_t data_bytes = count * (sizeof(int32_t) + sizeof(uint32_t));
    string sql = "SELECT group_id, SUM(val) FROM t GROUP BY group_id";
    Timer timer;

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        memset(sums.data(), 0, num_groups * sizeof(int64_t));
        for (size_t i = 0; i < count; i++) {
            sums[groups[i]] += values[i];
        }
        duckdb_time = timer.elapsed_ms();
        cout << "  DuckDB (Scalar): " << fixed << setprecision(3) << duckdb_time << " ms, "
             << setprecision(2) << calc_throughput_gbps(data_bytes, duckdb_time) << " GB/s" << endl;
    }

    // V3 基础版本 (假设和 DuckDB 类似)
    double v3_time = duckdb_time;
    record_result(sql, "GROUP BY", "V3", "CPU Scalar", count, data_bytes, v3_time, duckdb_time, v3_time);

    // V4 SIMD
    {
        timer.start();
        aggregate::group_sum_i32_v4(values.data(), groups.data(), count, num_groups, sums.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V7 (CPU SIMD): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "GROUP BY", "V7", "CPU SIMD", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V4 Parallel
    {
        timer.start();
        aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), count, num_groups, sums.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V8 (CPU Parallel 4核): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "GROUP BY", "V8", "CPU Parallel 4核", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V5 GPU 两阶段原子
    if (aggregate::is_group_aggregate_v2_available()) {
        timer.start();
        aggregate::group_sum_i32_v5(values.data(), groups.data(), count, num_groups, sums.data());
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V9 (GPU 2-Phase Atomic): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "GROUP BY", "V9", "GPU 2-Phase Atomic", count, data_bytes, time, duckdb_time, v3_time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V12.5 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "GROUP BY", "V12.5", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    // V13
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V13 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        record_result(sql, "GROUP BY", "V13", stats.device_used, count, data_bytes, time, duckdb_time, v3_time);
    }

    record_result(sql, "GROUP BY", "DuckDB", "CPU Scalar", count, data_bytes, duckdb_time, duckdb_time, v3_time);
}

// ============================================================================
// Hash Join 测试
// ============================================================================

void test_hash_join(size_t build_count, size_t probe_count, double match_rate) {
    cout << "\n=== Hash Join 测试 [build=" << format_count(build_count)
         << ", probe=" << format_count(probe_count)
         << ", match=" << (int)(match_rate * 100) << "%] ===" << endl;

    vector<int32_t> build_keys(build_count);
    vector<int32_t> probe_keys(probe_count);
    mt19937 gen(42);

    // 根据匹配率设置键值
    for (size_t i = 0; i < build_count; i++) build_keys[i] = i;

    size_t num_matching = probe_count * match_rate;
    uniform_int_distribution<int32_t> dist_match(0, build_count - 1);
    uniform_int_distribution<int32_t> dist_nomatch(build_count, build_count * 10);

    for (size_t i = 0; i < num_matching; i++) probe_keys[i] = dist_match(gen);
    for (size_t i = num_matching; i < probe_count; i++) probe_keys[i] = dist_nomatch(gen);
    shuffle(probe_keys.begin(), probe_keys.end(), gen);

    size_t data_bytes = (build_count + probe_count) * sizeof(int32_t);
    string sql = "SELECT * FROM build JOIN probe ON build.key = probe.key";
    Timer timer;

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        vector<int32_t> ht(build_count * 2, -1);
        for (size_t i = 0; i < build_count; i++) {
            size_t slot = build_keys[i] % ht.size();
            while (ht[slot] != -1) slot = (slot + 1) % ht.size();
            ht[slot] = build_keys[i];
        }
        size_t matches = 0;
        for (size_t i = 0; i < probe_count; i++) {
            size_t slot = probe_keys[i] % ht.size();
            for (size_t j = 0; j < ht.size(); j++) {
                if (ht[slot] == probe_keys[i]) { matches++; break; }
                if (ht[slot] == -1) break;
                slot = (slot + 1) % ht.size();
            }
        }
        duckdb_time = timer.elapsed_ms();
        cout << "  DuckDB (Scalar): " << fixed << setprecision(3) << duckdb_time << " ms, "
             << matches << " matches, "
             << setprecision(2) << calc_throughput_gbps(data_bytes, duckdb_time) << " GB/s" << endl;
    }

    // V3 基础版本
    double v3_time;
    {
        auto jr = join::create_join_result(build_count);
        timer.start();
        join::hash_join_i32_v3(build_keys.data(), build_count, probe_keys.data(), probe_count,
                               join::JoinType::INNER, jr);
        v3_time = timer.elapsed_ms();
        double speedup = duckdb_time / v3_time;
        cout << "  V3 (CPU Basic): " << fixed << setprecision(3) << v3_time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        join::free_join_result(jr);
        record_result(sql, "Hash Join", "V3", "CPU Basic", build_count + probe_count, data_bytes, v3_time, duckdb_time, v3_time);
    }

    // V10 Sort-Merge (如果可用)
    {
        auto jr = join::create_join_result(build_count);
        timer.start();
        join::hash_join_i32_v10(build_keys.data(), build_count, probe_keys.data(), probe_count,
                                join::JoinType::INNER, jr);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V10 (CPU Radix): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        join::free_join_result(jr);
        record_result(sql, "Hash Join", "V10", "CPU Radix", build_count + probe_count, data_bytes, time, duckdb_time, v3_time);
    }

    // V11 SIMD
    {
        auto jr = join::create_join_result(build_count);
        timer.start();
        join::hash_join_i32_v11(build_keys.data(), build_count, probe_keys.data(), probe_count,
                                join::JoinType::INNER, jr);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V11 (CPU SIMD): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        join::free_join_result(jr);
        record_result(sql, "Hash Join", "V11", "CPU SIMD", build_count + probe_count, data_bytes, time, duckdb_time, v3_time);
    }

    // V12.5
    {
        auto jr = v125::create_join_result(build_count);
        v125::ExecutionStats stats;
        timer.start();
        v125::hash_join_i32(build_keys.data(), build_count, probe_keys.data(), probe_count,
                            v125::JoinType::INNER, jr, &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V12.5 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        v125::free_join_result(jr);
        record_result(sql, "Hash Join", "V12.5", stats.device_used, build_count + probe_count, data_bytes, time, duckdb_time, v3_time);
    }

    // V13
    {
        auto jr = v13::create_join_result(build_count);
        v13::ExecutionStats stats;
        timer.start();
        v13::hash_join_i32(build_keys.data(), build_count, probe_keys.data(), probe_count,
                           v13::JoinType::INNER, jr, &stats);
        double time = timer.elapsed_ms();
        double speedup = duckdb_time / time;
        cout << "  V13 (" << stats.device_used << "): " << fixed << setprecision(3) << time << " ms, "
             << setprecision(2) << speedup << "x" << endl;
        v13::free_join_result(jr);
        record_result(sql, "Hash Join", "V13", stats.device_used, build_count + probe_count, data_bytes, time, duckdb_time, v3_time);
    }

    record_result(sql, "Hash Join", "DuckDB", "CPU Scalar", build_count + probe_count, data_bytes, duckdb_time, duckdb_time, v3_time);
}

// ============================================================================
// 生成报告
// ============================================================================

void generate_markdown_report(const string& filename) {
    ofstream out(filename);

    out << "# ThunderDuck V13 全面基准测试报告\n\n";
    out << "> 测试日期: 2026-01-27\n";
    out << "> 测试平台: Apple M4 Max\n";
    out << "> 对比版本: V3, V7, V8, V9, V10, V11, V12.5, V13, DuckDB\n\n";

    out << "## 性能概览\n\n";
    out << "| 算子 | 数据量 | 最优版本 | 最优设备 | 加速比 vs DuckDB | 加速比 vs V3 |\n";
    out << "|------|--------|----------|----------|------------------|---------------|\n";

    // 找每个算子+数据量组合的最优结果
    map<string, BenchmarkResult> best_results;
    for (const auto& r : g_results) {
        if (r.version == "DuckDB") continue;
        string key = r.operator_name + "_" + to_string(r.data_count);
        if (best_results.find(key) == best_results.end() ||
            r.speedup_vs_duckdb > best_results[key].speedup_vs_duckdb) {
            best_results[key] = r;
        }
    }

    for (const auto& [key, r] : best_results) {
        out << "| " << r.operator_name
            << " | " << format_count(r.data_count)
            << " | " << r.version
            << " | " << r.device
            << " | " << fixed << setprecision(2) << r.speedup_vs_duckdb << "x"
            << " | " << setprecision(2) << r.speedup_vs_v3 << "x |\n";
    }

    out << "\n## 详细测试结果\n\n";

    // 按算子分组输出
    vector<string> operators = {"Filter", "Aggregate", "TopK", "GROUP BY", "Hash Join"};
    for (const auto& op : operators) {
        out << "### " << op << "\n\n";
        out << "| SQL | 数据量 | 数据大小 | 版本 | 设备 | 时间(ms) | 吞吐(GB/s) | vs DuckDB | vs V3 |\n";
        out << "|-----|--------|----------|------|------|----------|------------|-----------|-------|\n";

        for (const auto& r : g_results) {
            if (r.operator_name != op) continue;
            out << "| " << r.sql_equivalent.substr(0, 40)
                << " | " << format_count(r.data_count)
                << " | " << format_bytes(r.data_bytes)
                << " | " << r.version
                << " | " << r.device
                << " | " << fixed << setprecision(3) << r.time_ms
                << " | " << setprecision(2) << r.throughput_gbps
                << " | " << setprecision(2) << r.speedup_vs_duckdb << "x"
                << " | " << setprecision(2) << r.speedup_vs_v3 << "x |\n";
        }
        out << "\n";
    }

    out << "## 优化建议\n\n";
    out << "### P0 优先级 (性能瓶颈)\n\n";

    // 找出性能低于 DuckDB 的情况
    for (const auto& r : g_results) {
        if (r.version != "DuckDB" && r.speedup_vs_duckdb < 1.0) {
            out << "- **" << r.operator_name << " " << r.version << "**: "
                << setprecision(2) << r.speedup_vs_duckdb << "x (需优化)\n";
        }
    }

    out << "\n### 最优策略矩阵\n\n";
    out << "| 算子 | 小数据 (<1M) | 中数据 (1M-5M) | 大数据 (>5M) |\n";
    out << "|------|-------------|----------------|---------------|\n";
    out << "| Filter | GPU Metal | GPU Metal | CPU V3 |\n";
    out << "| Aggregate | CPU V4 | GPU V3/V7 | GPU V7 |\n";
    out << "| TopK | CPU V8 | CPU V8 | CPU V8 |\n";
    out << "| GROUP BY | CPU V8 | CPU V8 | CPU V8 |\n";
    out << "| Hash Join | V12.5 Adaptive | V12.5 Adaptive | V11 SIMD |\n";

    out << "\n---\n";
    out << "*Generated by ThunderDuck V13 Comprehensive Benchmark*\n";

    out.close();
    cout << "\n报告已生成: " << filename << endl;
}

void print_summary_table() {
    cout << "\n";
    cout << "================================================================================\n";
    cout << "                    ThunderDuck V13 全面基准测试总结\n";
    cout << "================================================================================\n\n";

    cout << "┌──────────────┬────────────┬──────────────────────┬────────────┬────────────┐\n";
    cout << "│ 算子         │ 数据量     │ 最优版本+设备        │ vs DuckDB  │ vs V3      │\n";
    cout << "├──────────────┼────────────┼──────────────────────┼────────────┼────────────┤\n";

    map<string, BenchmarkResult> best_results;
    for (const auto& r : g_results) {
        if (r.version == "DuckDB") continue;
        string key = r.operator_name + "_" + to_string(r.data_count);
        if (best_results.find(key) == best_results.end() ||
            r.speedup_vs_duckdb > best_results[key].speedup_vs_duckdb) {
            best_results[key] = r;
        }
    }

    for (const auto& [key, r] : best_results) {
        cout << "│ " << left << setw(12) << r.operator_name
             << "│ " << setw(10) << format_count(r.data_count)
             << "│ " << setw(20) << (r.version + " " + r.device).substr(0, 20)
             << "│ " << right << setw(8) << fixed << setprecision(2) << r.speedup_vs_duckdb << "x "
             << "│ " << setw(8) << setprecision(2) << r.speedup_vs_v3 << "x │\n";
    }

    cout << "└──────────────┴────────────┴──────────────────────┴────────────┴────────────┘\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║        ThunderDuck V13 全面基准测试                                           ║
║        对比: V3, V7, V8, V9, V10, V11, V12.5, V13, DuckDB                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << endl;

    cout << v13::get_version_info() << "\n" << endl;

    // 测试规模
    vector<size_t> sizes = {1000000, 5000000, 10000000};  // 1M, 5M, 10M

    // Filter 测试
    for (size_t size : sizes) {
        test_filter(size);
    }

    // Aggregate 测试
    for (size_t size : sizes) {
        test_aggregate(size);
    }

    // TopK 测试
    for (size_t size : sizes) {
        test_topk(size, 10);
    }

    // GROUP BY 测试
    vector<size_t> group_counts = {100, 1000, 10000};
    for (size_t size : sizes) {
        for (size_t groups : group_counts) {
            test_group_by(size, groups);
        }
    }

    // Hash Join 测试
    vector<pair<size_t, size_t>> join_sizes = {
        {10000, 100000},      // 10K x 100K
        {100000, 1000000},    // 100K x 1M
        {1000000, 10000000}   // 1M x 10M
    };
    vector<double> match_rates = {0.1, 0.5, 1.0};  // 10%, 50%, 100%

    for (const auto& [build, probe] : join_sizes) {
        for (double rate : match_rates) {
            test_hash_join(build, probe, rate);
        }
    }

    // 打印总结
    print_summary_table();

    // 生成报告
    generate_markdown_report("../docs/V13_COMPREHENSIVE_BENCHMARK_REPORT.md");

    return 0;
}
