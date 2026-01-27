/**
 * ThunderDuck V12/V12.1 全面性能基准测试
 *
 * 详细测试各版本性能：V3, V7, V8, V9, V10, V11, V12, V12.1, DuckDB
 * 输出信息：SQL、数据量、算子、设备、吞吐、时间、加速比
 */

#include "thunderduck/v12_unified.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"

#include <duckdb.hpp>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>

using namespace thunderduck;

// ============================================================================
// 工具类
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

double benchmark_fn(std::function<void()> fn, int warmup, int runs) {
    Timer timer;
    for (int i = 0; i < warmup; i++) fn();

    double total = 0;
    for (int i = 0; i < runs; i++) {
        timer.start();
        fn();
        timer.stop();
        total += timer.elapsed_ms();
    }
    return total / runs;
}

std::vector<int32_t> generate_random_i32(size_t count) {
    std::vector<int32_t> data(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000000);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

std::vector<uint32_t> generate_random_groups(size_t count, size_t num_groups) {
    std::vector<uint32_t> data(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, num_groups - 1);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

double calculate_throughput(size_t count, size_t element_size, double elapsed_ms) {
    if (elapsed_ms <= 0) return 0;
    double bytes = count * element_size;
    double seconds = elapsed_ms / 1000.0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
}

void print_separator() {
    std::cout << "════════════════════════════════════════════════════════════════════════════════════════════════════\n";
}

void print_section(const char* title) {
    std::cout << "\n";
    print_separator();
    std::cout << "  " << title << "\n";
    print_separator();
}

// ============================================================================
// 扩展表格打印
// ============================================================================

void print_extended_header() {
    std::cout << "┌──────────┬──────────────────┬──────────┬───────────┬──────────┬──────────┬──────────┐\n";
    std::cout << "│  版本    │  设备            │ 时间(ms) │ 吞吐GB/s  │ vs DuckDB│  vs V3   │  状态    │\n";
    std::cout << "├──────────┼──────────────────┼──────────┼───────────┼──────────┼──────────┼──────────┤\n";
}

void print_row(const char* version, const char* device, double time_ms,
               double throughput, double vs_duckdb, double vs_v3, bool is_best) {
    char vs_duckdb_str[16];
    if (vs_duckdb >= 1.0) {
        snprintf(vs_duckdb_str, sizeof(vs_duckdb_str), "+%.2fx", vs_duckdb);
    } else {
        snprintf(vs_duckdb_str, sizeof(vs_duckdb_str), "%.2fx", vs_duckdb);
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "│ " << std::left << std::setw(8) << version
              << " │ " << std::setw(16) << device
              << " │ " << std::right << std::setw(8) << time_ms
              << " │ " << std::setw(9) << throughput
              << " │ " << std::setw(8) << vs_duckdb_str
              << " │ " << std::setw(8) << vs_v3 << "x"
              << " │ " << (is_best ? " ★最优  " : "        ")
              << " │\n";
}

void print_footer() {
    std::cout << "└──────────┴──────────────────┴──────────┴───────────┴──────────┴──────────┴──────────┘\n";
}

// ============================================================================
// FILTER 测试
// ============================================================================

void benchmark_filter_full(size_t count) {
    const char* size_label = count < 5000000 ? "1M" : "10M";

    print_section(count < 5000000 ? "FILTER 算子测试 (1M 数据)" : "FILTER 算子测试 (10M 数据)");

    std::cout << "\n【测试配置】\n";
    std::cout << "  SQL语句: SELECT COUNT(*) FROM t WHERE value > 500000\n";
    std::cout << "  数据类型: INT32\n";
    std::cout << "  数据量: " << count << " 行\n";
    std::cout << "  数据大小: " << std::fixed << std::setprecision(2) << (count * 4.0 / 1024 / 1024) << " MB\n";
    std::cout << "  选择率: ~50%\n\n";

    auto data = generate_random_i32(count);
    std::vector<uint32_t> indices(count);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (auto v : data) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    std::cout << "【测试结果】\n\n";
    print_extended_header();

    // DuckDB baseline
    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM t WHERE value > 500000");
    }, 2, 5);

    // V3 CPU SIMD
    double v3_time = benchmark_fn([&]() {
        filter::count_i32_v3(data.data(), count, filter::CompareOp::GT, 500000);
    }, 2, 5);

    // V7 GPU
    double v7_time = v3_time;
    if (filter::is_filter_gpu_available()) {
        v7_time = benchmark_fn([&]() {
            filter::count_i32_v4(data.data(), count, filter::CompareOp::GT, 500000);
        }, 2, 5);
    }

    // V9 CPU SIMD enhanced
    double v9_time = benchmark_fn([&]() {
        filter::count_i32_v6(data.data(), count, filter::CompareOp::GT, 500000);
    }, 2, 5);

    // V12 统一版本 (使用 config 强制特定版本以获取准确时间)
    v12::ExecutionStats stats12;
    double v12_time = benchmark_fn([&]() {
        v12::count_i32(data.data(), count, v12::CompareOp::GT, 500000, &stats12);
    }, 2, 5);

    // 找最优
    std::vector<double> times = {duckdb_time, v3_time, v7_time, v9_time, v12_time};
    double best_time = *std::min_element(times.begin(), times.end());

    print_row("DuckDB", "CPU 标量", duckdb_time,
              calculate_throughput(count, 4, duckdb_time),
              1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_row("V3", "CPU SIMD", v3_time,
              calculate_throughput(count, 4, v3_time),
              duckdb_time / v3_time, 1.0, v3_time == best_time);

    if (filter::is_filter_gpu_available()) {
        print_row("V7", "GPU Metal", v7_time,
                  calculate_throughput(count, 4, v7_time),
                  duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);
    }

    print_row("V9", "CPU SIMD+", v9_time,
              calculate_throughput(count, 4, v9_time),
              duckdb_time / v9_time, v3_time / v9_time, v9_time == best_time);

    print_row("V12", stats12.device_used ? stats12.device_used : "Auto", v12_time,
              calculate_throughput(count, 4, v12_time),
              duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_footer();

    // 分析
    std::cout << "\n【性能分析】\n";
    std::cout << "  - 内存带宽理论上限: ~400 GB/s (M4 Max)\n";
    std::cout << "  - 最优实测带宽: " << std::fixed << std::setprecision(2)
              << calculate_throughput(count, 4, best_time) << " GB/s\n";
    std::cout << "  - 带宽利用率: " << (calculate_throughput(count, 4, best_time) / 400.0 * 100) << "%\n";
}

// ============================================================================
// AGGREGATE 测试
// ============================================================================

void benchmark_aggregate_full(size_t count) {
    print_section(count < 5000000 ? "AGGREGATE 算子测试 (1M 数据)" : "AGGREGATE 算子测试 (10M 数据)");

    std::cout << "\n【测试配置】\n";
    std::cout << "  SQL语句: SELECT SUM(value) FROM t\n";
    std::cout << "  数据类型: INT32 → INT64\n";
    std::cout << "  数据量: " << count << " 行\n";
    std::cout << "  数据大小: " << std::fixed << std::setprecision(2) << (count * 4.0 / 1024 / 1024) << " MB\n\n";

    auto data = generate_random_i32(count);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (auto v : data) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    std::cout << "【测试结果】\n\n";
    print_extended_header();

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT SUM(value) FROM t");
    }, 2, 5);

    // V2 CPU SIMD baseline
    double v2_time = benchmark_fn([&]() {
        aggregate::sum_i32_v2(data.data(), count);
    }, 2, 5);

    // V7 GPU
    double v7_time = v2_time;
    if (aggregate::is_aggregate_gpu_available()) {
        v7_time = benchmark_fn([&]() {
            aggregate::sum_i32_v3(data.data(), count);
        }, 2, 5);
    }

    // V9 CPU SIMD enhanced
    double v9_time = benchmark_fn([&]() {
        aggregate::sum_i32_v4(data.data(), count);
    }, 2, 5);

    // V12
    v12::ExecutionStats stats12;
    double v12_time = benchmark_fn([&]() {
        v12::sum_i32(data.data(), count, &stats12);
    }, 2, 5);

    std::vector<double> times = {duckdb_time, v2_time, v7_time, v9_time, v12_time};
    double best_time = *std::min_element(times.begin(), times.end());

    print_row("DuckDB", "CPU 标量", duckdb_time,
              calculate_throughput(count, 4, duckdb_time),
              1.0, duckdb_time / v2_time, duckdb_time == best_time);

    print_row("V2", "CPU SIMD", v2_time,
              calculate_throughput(count, 4, v2_time),
              duckdb_time / v2_time, 1.0, v2_time == best_time);

    if (aggregate::is_aggregate_gpu_available()) {
        print_row("V7", "GPU Metal", v7_time,
                  calculate_throughput(count, 4, v7_time),
                  duckdb_time / v7_time, v2_time / v7_time, v7_time == best_time);
    }

    print_row("V9", "CPU SIMD+", v9_time,
              calculate_throughput(count, 4, v9_time),
              duckdb_time / v9_time, v2_time / v9_time, v9_time == best_time);

    print_row("V12", stats12.device_used ? stats12.device_used : "Auto", v12_time,
              calculate_throughput(count, 4, v12_time),
              duckdb_time / v12_time, v2_time / v12_time, v12_time == best_time);

    print_footer();
}

// ============================================================================
// GROUP BY 测试 (包含 V12.1 GPU Warp-level)
// ============================================================================

void benchmark_group_by_full(size_t count, size_t num_groups) {
    print_section(count < 5000000 ? "GROUP BY 算子测试 (1M 数据)" : "GROUP BY 算子测试 (10M 数据)");

    std::cout << "\n【测试配置】\n";
    std::cout << "  SQL语句: SELECT group_id, SUM(value) FROM t GROUP BY group_id\n";
    std::cout << "  数据类型: (UINT32, INT32) → INT64\n";
    std::cout << "  数据量: " << count << " 行\n";
    std::cout << "  数据大小: " << std::fixed << std::setprecision(2) << (count * 8.0 / 1024 / 1024) << " MB\n";
    std::cout << "  分组数量: " << num_groups << "\n\n";

    auto values = generate_random_i32(count);
    auto groups = generate_random_groups(count, num_groups);
    std::vector<int64_t> sums(num_groups);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (group_id INTEGER, value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (size_t i = 0; i < count; i++) {
            appender.BeginRow();
            appender.Append<int32_t>(groups[i]);
            appender.Append<int32_t>(values[i]);
            appender.EndRow();
        }
    }

    std::cout << "【测试结果】\n\n";
    print_extended_header();

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT group_id, SUM(value) FROM t GROUP BY group_id");
    }, 2, 5);

    // V7 CPU 单线程
    double v7_time = benchmark_fn([&]() {
        aggregate::group_sum_i32_v4(values.data(), groups.data(), count, num_groups, sums.data());
    }, 2, 5);

    // V8 CPU 多线程
    double v8_time = benchmark_fn([&]() {
        aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), count, num_groups, sums.data());
    }, 2, 5);

    // V9 GPU 两阶段原子
    double v9_gpu_time = v7_time;
    if (aggregate::is_group_aggregate_v2_available()) {
        v9_gpu_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v5(values.data(), groups.data(), count, num_groups, sums.data());
        }, 2, 5);
    }

    // V12.1 GPU Warp-level reduction
    double v12_1_gpu_time = v7_time;
    if (aggregate::is_group_aggregate_v3_available() && num_groups <= 1024) {
        v12_1_gpu_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v12_1(values.data(), groups.data(), count, num_groups, sums.data());
        }, 2, 5);
    }

    // V12 统一版本
    v12::ExecutionStats stats12;
    double v12_time = benchmark_fn([&]() {
        v12::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats12);
    }, 2, 5);

    std::vector<double> times = {duckdb_time, v7_time, v8_time, v9_gpu_time, v12_1_gpu_time, v12_time};
    double best_time = *std::min_element(times.begin(), times.end());

    print_row("DuckDB", "CPU 标量", duckdb_time,
              calculate_throughput(count, 8, duckdb_time),
              1.0, duckdb_time / v7_time, duckdb_time == best_time);

    print_row("V7", "CPU 单线程", v7_time,
              calculate_throughput(count, 8, v7_time),
              duckdb_time / v7_time, 1.0, v7_time == best_time);

    print_row("V8", "CPU 4核并行", v8_time,
              calculate_throughput(count, 8, v8_time),
              duckdb_time / v8_time, v7_time / v8_time, v8_time == best_time);

    if (aggregate::is_group_aggregate_v2_available()) {
        print_row("V9", "GPU 两阶段原子", v9_gpu_time,
                  calculate_throughput(count, 8, v9_gpu_time),
                  duckdb_time / v9_gpu_time, v7_time / v9_gpu_time, v9_gpu_time == best_time);
    }

    if (aggregate::is_group_aggregate_v3_available() && num_groups <= 1024) {
        print_row("V12.1", "GPU Warp-level", v12_1_gpu_time,
                  calculate_throughput(count, 8, v12_1_gpu_time),
                  duckdb_time / v12_1_gpu_time, v7_time / v12_1_gpu_time, v12_1_gpu_time == best_time);
    }

    print_row("V12", stats12.device_used ? stats12.device_used : "Auto", v12_time,
              calculate_throughput(count, 8, v12_time),
              duckdb_time / v12_time, v7_time / v12_time, v12_time == best_time);

    print_footer();

    // V12.1 GPU 改进分析
    if (aggregate::is_group_aggregate_v3_available() && aggregate::is_group_aggregate_v2_available()) {
        std::cout << "\n【V12.1 GPU 优化分析】\n";
        std::cout << "  - V9 GPU (两阶段原子): " << std::fixed << std::setprecision(2)
                  << (duckdb_time / v9_gpu_time) << "x vs DuckDB\n";
        std::cout << "  - V12.1 GPU (Warp-level): " << (duckdb_time / v12_1_gpu_time) << "x vs DuckDB\n";
        std::cout << "  - GPU 改进幅度: " << ((v9_gpu_time / v12_1_gpu_time - 1) * 100) << "%\n";
    }
}

// ============================================================================
// TopK 测试
// ============================================================================

void benchmark_topk_full(size_t count, size_t k) {
    print_section(count < 5000000 ? "TopK 算子测试 (1M 数据)" : "TopK 算子测试 (10M 数据)");

    std::cout << "\n【测试配置】\n";
    std::cout << "  SQL语句: SELECT * FROM t ORDER BY value DESC LIMIT " << k << "\n";
    std::cout << "  数据类型: INT32\n";
    std::cout << "  数据量: " << count << " 行\n";
    std::cout << "  数据大小: " << std::fixed << std::setprecision(2) << (count * 4.0 / 1024 / 1024) << " MB\n";
    std::cout << "  K值: " << k << "\n\n";

    auto data = generate_random_i32(count);
    std::vector<int32_t> values(k);
    std::vector<uint32_t> indices(k);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (auto v : data) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    std::cout << "【测试结果】\n\n";
    print_extended_header();

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT value FROM t ORDER BY value DESC LIMIT 10");
    }, 2, 5);

    // V3 Heap
    double v3_time = benchmark_fn([&]() {
        sort::topk_max_i32_v3(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V7 Sampling
    double v7_time = benchmark_fn([&]() {
        sort::topk_max_i32_v4(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V8 Count-based
    double v8_time = benchmark_fn([&]() {
        sort::topk_max_i32_v5(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V12
    v12::ExecutionStats stats12;
    double v12_time = benchmark_fn([&]() {
        v12::topk_max_i32(data.data(), count, k, values.data(), indices.data(), &stats12);
    }, 2, 5);

    std::vector<double> times = {duckdb_time, v3_time, v7_time, v8_time, v12_time};
    double best_time = *std::min_element(times.begin(), times.end());

    print_row("DuckDB", "CPU Sort", duckdb_time,
              calculate_throughput(count, 4, duckdb_time),
              1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_row("V3", "CPU Heap", v3_time,
              calculate_throughput(count, 4, v3_time),
              duckdb_time / v3_time, 1.0, v3_time == best_time);

    print_row("V7", "CPU Sampling", v7_time,
              calculate_throughput(count, 4, v7_time),
              duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);

    print_row("V8", "CPU Count-Based", v8_time,
              calculate_throughput(count, 4, v8_time),
              duckdb_time / v8_time, v3_time / v8_time, v8_time == best_time);

    print_row("V12", stats12.device_used ? stats12.device_used : "Auto", v12_time,
              calculate_throughput(count, 4, v12_time),
              duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_footer();
}

// ============================================================================
// Hash Join 测试
// ============================================================================

void benchmark_hash_join_full(size_t build_count, size_t probe_count) {
    print_section("HASH JOIN 算子测试");

    std::cout << "\n【测试配置】\n";
    std::cout << "  SQL语句: SELECT * FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key\n";
    std::cout << "  Build表: " << build_count << " 行\n";
    std::cout << "  Probe表: " << probe_count << " 行\n";
    std::cout << "  数据大小: " << std::fixed << std::setprecision(2)
              << ((build_count + probe_count) * 4.0 / 1024 / 1024) << " MB\n";
    std::cout << "  预期匹配率: ~10%\n\n";

    std::mt19937 gen(42);
    std::vector<int32_t> build_keys(build_count);
    std::vector<int32_t> probe_keys(probe_count);

    // Build keys: 0 to build_count-1
    for (size_t i = 0; i < build_count; i++) {
        build_keys[i] = i;
    }
    // Probe keys: random in range [0, build_count * 10)
    std::uniform_int_distribution<int32_t> dist(0, build_count * 10 - 1);
    for (size_t i = 0; i < probe_count; i++) {
        probe_keys[i] = dist(gen);
    }

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE build_t (key INTEGER)");
    conn.Query("CREATE TABLE probe_t (key INTEGER)");

    {
        duckdb::Appender appender(conn, "build_t");
        for (auto k : build_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(k);
            appender.EndRow();
        }
    }
    {
        duckdb::Appender appender(conn, "probe_t");
        for (auto k : probe_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(k);
            appender.EndRow();
        }
    }

    std::cout << "【测试结果】\n\n";
    print_extended_header();

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key");
    }, 2, 5);

    // V3 Radix
    join::JoinResult* result3 = join::create_join_result(probe_count);
    double v3_time = benchmark_fn([&]() {
        result3->count = 0;
        join::hash_join_i32_v3(build_keys.data(), build_count,
                               probe_keys.data(), probe_count,
                               join::JoinType::INNER, result3);
    }, 2, 5);

    // V6 Prefetch
    join::JoinResult* result6 = join::create_join_result(probe_count);
    double v6_time = benchmark_fn([&]() {
        result6->count = 0;
        join::hash_join_i32_v6(build_keys.data(), build_count,
                               probe_keys.data(), probe_count,
                               join::JoinType::INNER, result6);
    }, 2, 5);

    // V7 CPU/GPU Adaptive
    join::JoinResult* result7 = join::create_join_result(probe_count);
    double v7_time = benchmark_fn([&]() {
        result7->count = 0;
        join::hash_join_i32_v4(build_keys.data(), build_count,
                               probe_keys.data(), probe_count,
                               join::JoinType::INNER, result7);
    }, 2, 5);

    // V10 Full Semantic
    join::JoinResult* result10 = join::create_join_result(probe_count);
    double v10_time = benchmark_fn([&]() {
        result10->count = 0;
        join::hash_join_i32_v10(build_keys.data(), build_count,
                                probe_keys.data(), probe_count,
                                join::JoinType::INNER, result10);
    }, 2, 5);

    // V11 SIMD
    join::JoinResult* result11 = join::create_join_result(probe_count);
    double v11_time = benchmark_fn([&]() {
        result11->count = 0;
        join::hash_join_i32_v11(build_keys.data(), build_count,
                                probe_keys.data(), probe_count,
                                join::JoinType::INNER, result11);
    }, 2, 5);

    // V12
    v12::JoinResult* result12 = v12::create_join_result(probe_count);
    v12::ExecutionStats stats12;
    double v12_time = benchmark_fn([&]() {
        result12->count = 0;
        v12::hash_join_i32(build_keys.data(), build_count,
                           probe_keys.data(), probe_count,
                           v12::JoinType::INNER, result12, &stats12);
    }, 2, 5);

    std::vector<double> times = {duckdb_time, v3_time, v6_time, v7_time, v10_time, v11_time, v12_time};
    double best_time = *std::min_element(times.begin(), times.end());
    size_t total_count = build_count + probe_count;

    print_row("DuckDB", "CPU 标量", duckdb_time,
              calculate_throughput(total_count, 4, duckdb_time),
              1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_row("V3", "CPU Radix", v3_time,
              calculate_throughput(total_count, 4, v3_time),
              duckdb_time / v3_time, 1.0, v3_time == best_time);

    print_row("V6", "CPU Prefetch", v6_time,
              calculate_throughput(total_count, 4, v6_time),
              duckdb_time / v6_time, v3_time / v6_time, v6_time == best_time);

    print_row("V7", "CPU/GPU Adaptive", v7_time,
              calculate_throughput(total_count, 4, v7_time),
              duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);

    print_row("V10", "CPU Full Semantic", v10_time,
              calculate_throughput(total_count, 4, v10_time),
              duckdb_time / v10_time, v3_time / v10_time, v10_time == best_time);

    print_row("V11", "CPU SIMD Probe", v11_time,
              calculate_throughput(total_count, 4, v11_time),
              duckdb_time / v11_time, v3_time / v11_time, v11_time == best_time);

    print_row("V12", stats12.device_used ? stats12.device_used : "Auto", v12_time,
              calculate_throughput(total_count, 4, v12_time),
              duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_footer();

    // 清理
    join::free_join_result(result3);
    join::free_join_result(result6);
    join::free_join_result(result7);
    join::free_join_result(result10);
    join::free_join_result(result11);
    v12::free_join_result(result12);
}

// ============================================================================
// 汇总报告
// ============================================================================

void print_summary() {
    print_section("性能汇总报告");

    std::cout << "\n【V12/V12.1 版本信息】\n";
    std::cout << v12::get_version_info() << "\n\n";

    std::cout << "【最优版本矩阵】\n";
    std::cout << v12::get_optimal_versions() << "\n\n";

    std::cout << "【优化建议】\n";
    std::cout << "  P0 (已优化): GROUP BY GPU - 使用 Warp-level reduction\n";
    std::cout << "  P1 (待优化): Filter 10M - GPU vs CPU 策略选择\n";
    std::cout << "  P2 (待优化): Hash Join 高匹配场景 - 两阶段计数分配\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ThunderDuck V12/V12.1 全面性能基准测试                                        ║\n";
    std::cout << "║                    测试平台: Apple M4 Max | 测试日期: 2026-01-27                                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n";

    // 1M 数据测试
    benchmark_filter_full(1000000);
    benchmark_aggregate_full(1000000);
    benchmark_group_by_full(1000000, 1000);
    benchmark_topk_full(1000000, 10);

    // 10M 数据测试
    benchmark_filter_full(10000000);
    benchmark_aggregate_full(10000000);
    benchmark_group_by_full(10000000, 1000);
    benchmark_topk_full(10000000, 10);

    // Hash Join 测试
    benchmark_hash_join_full(100000, 1000000);

    // 汇总
    print_summary();

    std::cout << "\n测试完成!\n";
    return 0;
}
