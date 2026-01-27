/**
 * ThunderDuck V12 综合性能基准测试
 *
 * 详细测试各版本性能：V3, V7, V8, V9, V10, V11, V12, DuckDB
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

struct BenchmarkResult {
    const char* sql;
    const char* operator_name;
    const char* version;
    const char* device;
    size_t data_count;
    double time_ms;
    double throughput_gbps;
    double vs_duckdb;
    double vs_v3;
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

void print_header(const char* title) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::left << std::setw(80) << title << "  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════╝\n";
}

void print_result_table_header() {
    std::cout << "┌───────────┬────────────┬──────────┬─────────────┬────────────┬──────────┬──────────┐\n";
    std::cout << "│   版本    │   设备     │  时间(ms) │  吞吐(GB/s) │ vs DuckDB  │   vs V3  │   状态   │\n";
    std::cout << "├───────────┼────────────┼──────────┼─────────────┼────────────┼──────────┼──────────┤\n";
}

void print_result_row(const char* version, const char* device, double time_ms,
                      double throughput, double vs_duckdb, double vs_v3, bool is_best) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "│ " << std::left << std::setw(9) << version
              << " │ " << std::setw(10) << device
              << " │ " << std::right << std::setw(8) << time_ms
              << " │ " << std::setw(11) << throughput
              << " │ " << std::setw(10) << (vs_duckdb >= 1.0 ? "+" : "") << vs_duckdb << "x"
              << " │ " << std::setw(8) << vs_v3 << "x"
              << " │ " << (is_best ? " ★最优  " : "        ")
              << " │\n";
}

void print_table_footer() {
    std::cout << "└───────────┴────────────┴──────────┴─────────────┴────────────┴──────────┴──────────┘\n";
}

// ============================================================================
// Filter 测试
// ============================================================================

void benchmark_filter(size_t count) {
    std::cout << "\n";
    print_header(count < 5000000 ? "FILTER 测试 (1M 数据)" : "FILTER 测试 (10M 数据)");
    std::cout << "SQL: SELECT COUNT(*) FROM t WHERE value > 500000\n";
    std::cout << "数据量: " << count << " 行 × 4 字节 = " << (count * 4.0 / 1024 / 1024) << " MB\n";
    std::cout << "选择率: ~50%\n\n";

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

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM t WHERE value > 500000");
    }, 2, 5);

    // V3 CPU SIMD
    double v3_time = benchmark_fn([&]() {
        filter::count_i32_v3(data.data(), count, filter::CompareOp::GT, 500000);
    }, 2, 5);

    // V7 GPU (if available)
    double v7_time = v3_time;
    if (filter::is_filter_gpu_available()) {
        v7_time = benchmark_fn([&]() {
            filter::count_i32_v4(data.data(), count, filter::CompareOp::GT, 500000);
        }, 2, 5);
    }

    // V9 CPU SIMD (V6 实现)
    double v9_time = benchmark_fn([&]() {
        filter::count_i32_v6(data.data(), count, filter::CompareOp::GT, 500000);
    }, 2, 5);

    // V12 统一版本
    v12::ExecutionStats stats;
    double v12_time = benchmark_fn([&]() {
        v12::count_i32(data.data(), count, v12::CompareOp::GT, 500000, &stats);
    }, 2, 5);

    print_result_table_header();

    double best_time = std::min({duckdb_time, v3_time, v7_time, v9_time, v12_time});

    print_result_row("DuckDB", "CPU 标量", duckdb_time,
                     calculate_throughput(count, 4, duckdb_time),
                     1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_result_row("V3", "CPU SIMD", v3_time,
                     calculate_throughput(count, 4, v3_time),
                     duckdb_time / v3_time, 1.0, v3_time == best_time);

    if (filter::is_filter_gpu_available()) {
        print_result_row("V7", "GPU Metal", v7_time,
                         calculate_throughput(count, 4, v7_time),
                         duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);
    }

    print_result_row("V9", "CPU SIMD", v9_time,
                     calculate_throughput(count, 4, v9_time),
                     duckdb_time / v9_time, v3_time / v9_time, v9_time == best_time);

    print_result_row("V12", stats.device_used, v12_time,
                     calculate_throughput(count, 4, v12_time),
                     duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_table_footer();
}

// ============================================================================
// Aggregate 测试
// ============================================================================

void benchmark_aggregate(size_t count) {
    std::cout << "\n";
    print_header(count < 5000000 ? "AGGREGATE 测试 (1M 数据)" : "AGGREGATE 测试 (10M 数据)");
    std::cout << "SQL: SELECT SUM(value) FROM t\n";
    std::cout << "数据量: " << count << " 行 × 4 字节 = " << (count * 4.0 / 1024 / 1024) << " MB\n\n";

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

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT SUM(value) FROM t");
    }, 2, 5);

    // V2
    double v2_time = benchmark_fn([&]() {
        aggregate::sum_i32_v2(data.data(), count);
    }, 2, 5);

    // V3 (V3 GPU)
    double v3_time = v2_time;
    if (aggregate::is_aggregate_gpu_available()) {
        v3_time = benchmark_fn([&]() {
            aggregate::sum_i32_v3(data.data(), count);
        }, 2, 5);
    }

    // V4 (V9 CPU SIMD)
    double v9_time = benchmark_fn([&]() {
        aggregate::sum_i32_v4(data.data(), count);
    }, 2, 5);

    // V12 统一版本
    v12::ExecutionStats stats;
    double v12_time = benchmark_fn([&]() {
        v12::sum_i32(data.data(), count, &stats);
    }, 2, 5);

    print_result_table_header();

    double best_time = std::min({duckdb_time, v2_time, v3_time, v9_time, v12_time});

    print_result_row("DuckDB", "CPU 标量", duckdb_time,
                     calculate_throughput(count, 4, duckdb_time),
                     1.0, duckdb_time / v2_time, duckdb_time == best_time);

    print_result_row("V2", "CPU SIMD", v2_time,
                     calculate_throughput(count, 4, v2_time),
                     duckdb_time / v2_time, 1.0, v2_time == best_time);

    if (aggregate::is_aggregate_gpu_available()) {
        print_result_row("V7", "GPU Metal", v3_time,
                         calculate_throughput(count, 4, v3_time),
                         duckdb_time / v3_time, v2_time / v3_time, v3_time == best_time);
    }

    print_result_row("V9", "CPU SIMD", v9_time,
                     calculate_throughput(count, 4, v9_time),
                     duckdb_time / v9_time, v2_time / v9_time, v9_time == best_time);

    print_result_row("V12", stats.device_used, v12_time,
                     calculate_throughput(count, 4, v12_time),
                     duckdb_time / v12_time, v2_time / v12_time, v12_time == best_time);

    print_table_footer();
}

// ============================================================================
// GROUP BY 测试
// ============================================================================

void benchmark_group_by(size_t count, size_t num_groups) {
    std::cout << "\n";
    print_header(count < 5000000 ? "GROUP BY 测试 (1M 数据)" : "GROUP BY 测试 (10M 数据)");
    std::cout << "SQL: SELECT group_id, SUM(value) FROM t GROUP BY group_id\n";
    std::cout << "数据量: " << count << " 行 × 8 字节 = " << (count * 8.0 / 1024 / 1024) << " MB\n";
    std::cout << "分组数: " << num_groups << "\n\n";

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

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT group_id, SUM(value) FROM t GROUP BY group_id");
    }, 2, 5);

    // V4 单线程
    double v4_time = benchmark_fn([&]() {
        aggregate::group_sum_i32_v4(values.data(), groups.data(), count, num_groups, sums.data());
    }, 2, 5);

    // V8 多线程
    double v8_time = benchmark_fn([&]() {
        aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), count, num_groups, sums.data());
    }, 2, 5);

    // V9 GPU (if available)
    double v9_gpu_time = v4_time;
    if (aggregate::is_group_aggregate_v2_available()) {
        v9_gpu_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v5(values.data(), groups.data(), count, num_groups, sums.data());
        }, 2, 5);
    }

    // V12 统一版本
    v12::ExecutionStats stats;
    double v12_time = benchmark_fn([&]() {
        v12::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats);
    }, 2, 5);

    print_result_table_header();

    double best_time = std::min({duckdb_time, v4_time, v8_time, v9_gpu_time, v12_time});

    print_result_row("DuckDB", "CPU 标量", duckdb_time,
                     calculate_throughput(count, 8, duckdb_time),
                     1.0, duckdb_time / v4_time, duckdb_time == best_time);

    print_result_row("V7", "CPU 单线程", v4_time,
                     calculate_throughput(count, 8, v4_time),
                     duckdb_time / v4_time, 1.0, v4_time == best_time);

    print_result_row("V8", "CPU 4核", v8_time,
                     calculate_throughput(count, 8, v8_time),
                     duckdb_time / v8_time, v4_time / v8_time, v8_time == best_time);

    if (aggregate::is_group_aggregate_v2_available()) {
        print_result_row("V9", "GPU Metal", v9_gpu_time,
                         calculate_throughput(count, 8, v9_gpu_time),
                         duckdb_time / v9_gpu_time, v4_time / v9_gpu_time, v9_gpu_time == best_time);
    }

    print_result_row("V12", stats.device_used, v12_time,
                     calculate_throughput(count, 8, v12_time),
                     duckdb_time / v12_time, v4_time / v12_time, v12_time == best_time);

    print_table_footer();
}

// ============================================================================
// TopK 测试
// ============================================================================

void benchmark_topk(size_t count, size_t k) {
    std::cout << "\n";
    print_header(count < 5000000 ? "TopK 测试 (1M 数据)" : "TopK 测试 (10M 数据)");
    std::cout << "SQL: SELECT * FROM t ORDER BY value DESC LIMIT " << k << "\n";
    std::cout << "数据量: " << count << " 行 × 4 字节 = " << (count * 4.0 / 1024 / 1024) << " MB\n";
    std::cout << "K = " << k << "\n\n";

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

    std::string sql = "SELECT * FROM t ORDER BY value DESC LIMIT " + std::to_string(k);
    double duckdb_time = benchmark_fn([&]() {
        conn.Query(sql);
    }, 2, 5);

    // V3 Heap
    double v3_time = benchmark_fn([&]() {
        sort::topk_max_i32_v3(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V7 采样
    double v7_time = benchmark_fn([&]() {
        sort::topk_max_i32_v4(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V8 计数
    double v8_time = benchmark_fn([&]() {
        sort::topk_max_i32_v5(data.data(), count, k, values.data(), indices.data());
    }, 2, 5);

    // V12 统一版本
    v12::ExecutionStats stats;
    double v12_time = benchmark_fn([&]() {
        v12::topk_max_i32(data.data(), count, k, values.data(), indices.data(), &stats);
    }, 2, 5);

    print_result_table_header();

    double best_time = std::min({duckdb_time, v3_time, v7_time, v8_time, v12_time});

    print_result_row("DuckDB", "CPU Sort", duckdb_time,
                     calculate_throughput(count, 4, duckdb_time),
                     1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_result_row("V3", "CPU Heap", v3_time,
                     calculate_throughput(count, 4, v3_time),
                     duckdb_time / v3_time, 1.0, v3_time == best_time);

    print_result_row("V7", "CPU Sample", v7_time,
                     calculate_throughput(count, 4, v7_time),
                     duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);

    print_result_row("V8", "CPU Count", v8_time,
                     calculate_throughput(count, 4, v8_time),
                     duckdb_time / v8_time, v3_time / v8_time, v8_time == best_time);

    print_result_row("V12", stats.device_used, v12_time,
                     calculate_throughput(count, 4, v12_time),
                     duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_table_footer();
}

// ============================================================================
// Hash Join 测试
// ============================================================================

void benchmark_hash_join() {
    std::cout << "\n";
    print_header("HASH JOIN 测试 (100K build × 1M probe)");
    std::cout << "SQL: SELECT * FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key\n";
    std::cout << "数据量: 100K × 4B + 1M × 4B = 4.2 MB\n";
    std::cout << "匹配率: ~10%\n\n";

    size_t build_size = 100000;
    size_t probe_size = 1000000;

    auto build_keys = generate_random_i32(build_size);
    auto probe_keys = generate_random_i32(probe_size);

    join::JoinResult* result = join::create_join_result(probe_size * 2);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE build_t (key INTEGER)");
    conn.Query("CREATE TABLE probe_t (key INTEGER)");
    {
        duckdb::Appender appender(conn, "build_t");
        for (auto v : build_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }
    {
        duckdb::Appender appender(conn, "probe_t");
        for (auto v : probe_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key");
    }, 2, 5);

    // V3 Radix
    double v3_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v3(build_keys.data(), build_size, probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, 2, 5);

    // V7 自适应
    double v7_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v4(build_keys.data(), build_size, probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, 2, 5);

    // V6 预取
    double v6_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v6(build_keys.data(), build_size, probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, 2, 5);

    // V10 完整语义
    double v10_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v10(build_keys.data(), build_size, probe_keys.data(), probe_size,
                                join::JoinType::INNER, result);
    }, 2, 5);

    // V11 SIMD
    double v11_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v11(build_keys.data(), build_size, probe_keys.data(), probe_size,
                                join::JoinType::INNER, result);
    }, 2, 5);

    // V12 统一版本
    v12::ExecutionStats stats;
    auto* v12_result = v12::create_join_result(probe_size * 2);
    double v12_time = benchmark_fn([&]() {
        v12_result->count = 0;
        v12::hash_join_i32(build_keys.data(), build_size, probe_keys.data(), probe_size,
                           v12::JoinType::INNER, v12_result, &stats);
    }, 2, 5);

    print_result_table_header();

    double best_time = std::min({duckdb_time, v3_time, v7_time, v6_time, v10_time, v11_time, v12_time});

    print_result_row("DuckDB", "CPU", duckdb_time,
                     calculate_throughput(build_size + probe_size, 4, duckdb_time),
                     1.0, duckdb_time / v3_time, duckdb_time == best_time);

    print_result_row("V3", "CPU Radix", v3_time,
                     calculate_throughput(build_size + probe_size, 4, v3_time),
                     duckdb_time / v3_time, 1.0, v3_time == best_time);

    print_result_row("V6", "CPU Prefetch", v6_time,
                     calculate_throughput(build_size + probe_size, 4, v6_time),
                     duckdb_time / v6_time, v3_time / v6_time, v6_time == best_time);

    print_result_row("V7", "CPU/GPU", v7_time,
                     calculate_throughput(build_size + probe_size, 4, v7_time),
                     duckdb_time / v7_time, v3_time / v7_time, v7_time == best_time);

    print_result_row("V10", "CPU Full", v10_time,
                     calculate_throughput(build_size + probe_size, 4, v10_time),
                     duckdb_time / v10_time, v3_time / v10_time, v10_time == best_time);

    print_result_row("V11", "CPU SIMD", v11_time,
                     calculate_throughput(build_size + probe_size, 4, v11_time),
                     duckdb_time / v11_time, v3_time / v11_time, v11_time == best_time);

    print_result_row("V12", stats.device_used, v12_time,
                     calculate_throughput(build_size + probe_size, 4, v12_time),
                     duckdb_time / v12_time, v3_time / v12_time, v12_time == best_time);

    print_table_footer();

    join::free_join_result(result);
    v12::free_join_result(v12_result);
}

// ============================================================================
// V12 最优版本矩阵验证
// ============================================================================

void print_v12_summary() {
    std::cout << "\n";
    print_header("V12 最优版本矩阵");
    std::cout << v12::get_optimal_versions() << "\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           ThunderDuck V12 综合性能基准测试                                         ║\n";
    std::cout << "║           测试平台: Apple M4 Max | 测试日期: 2026-01-27                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\n" << v12::get_version_info() << "\n";

    // 1M 数据测试
    benchmark_filter(1000000);
    benchmark_aggregate(1000000);
    benchmark_group_by(1000000, 1000);
    benchmark_topk(1000000, 10);

    // 10M 数据测试
    benchmark_filter(10000000);
    benchmark_aggregate(10000000);
    benchmark_group_by(10000000, 1000);
    benchmark_topk(10000000, 10);

    // Hash Join 测试
    benchmark_hash_join();

    // V12 总结
    print_v12_summary();

    std::cout << "\n测试完成!\n";
    return 0;
}
