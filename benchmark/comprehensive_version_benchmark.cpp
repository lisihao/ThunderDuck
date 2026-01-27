/**
 * ThunderDuck 全面版本基准测试
 *
 * 对比 V3, V7(v4), V8(v5), V9(v6), V10, DuckDB 原版
 *
 * 输出:
 * - SQL 等价语句
 * - 数据量
 * - 操作算子
 * - 执行设备 (CPU/GPU/NPU)
 * - 数据吞吐带宽
 * - 执行时长
 * - 加速比
 */

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
#include <numeric>

using namespace thunderduck;

// ============================================================================
// 测试配置
// ============================================================================

struct BenchmarkConfig {
    size_t small_size = 100000;      // 100K
    size_t medium_size = 1000000;    // 1M
    size_t large_size = 10000000;    // 10M
    size_t xlarge_size = 50000000;   // 50M
    int warmup_runs = 2;
    int test_runs = 5;
    bool verbose = false;
};

// ============================================================================
// 结果记录
// ============================================================================

struct BenchmarkResult {
    const char* sql;                 // SQL 等价语句
    const char* operator_name;       // 算子名称
    const char* version;             // 版本 (V3/V7/V8/V9/V10/DuckDB)
    const char* device;              // 执行设备
    size_t data_bytes;               // 数据量 (字节)
    size_t row_count;                // 行数
    double time_ms;                  // 执行时间 (ms)
    double throughput_gbps;          // 吞吐量 (GB/s)
    double speedup_vs_duckdb;        // 相对 DuckDB 加速比
    double speedup_vs_v3;            // 相对 V3 加速比
};

std::vector<BenchmarkResult> g_results;

// ============================================================================
// 计时工具
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
    // Warmup
    for (int i = 0; i < warmup; i++) fn();

    // Measure
    double total = 0;
    for (int i = 0; i < runs; i++) {
        timer.start();
        fn();
        timer.stop();
        total += timer.elapsed_ms();
    }
    return total / runs;
}

// ============================================================================
// 数据生成
// ============================================================================

std::vector<int32_t> generate_random_i32(size_t count, int32_t min_val = 0, int32_t max_val = 1000000) {
    std::vector<int32_t> data(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

std::vector<uint32_t> generate_groups(size_t count, size_t num_groups) {
    std::vector<uint32_t> groups(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, num_groups - 1);
    for (size_t i = 0; i < count; i++) {
        groups[i] = dist(gen);
    }
    return groups;
}

// ============================================================================
// DuckDB 基准测试
// ============================================================================

class DuckDBBenchmark {
public:
    DuckDBBenchmark() {
        db_ = std::make_unique<duckdb::DuckDB>(nullptr);
        conn_ = std::make_unique<duckdb::Connection>(*db_);
    }

    void load_data(const std::vector<int32_t>& data, const char* table_name) {
        // 创建表
        std::string create_sql = "CREATE OR REPLACE TABLE " + std::string(table_name) + " (value INTEGER)";
        conn_->Query(create_sql);

        // 批量插入
        {
            duckdb::Appender appender(*conn_, table_name);
            for (int32_t v : data) {
                appender.BeginRow();
                appender.Append<int32_t>(v);
                appender.EndRow();
            }
        }
    }

    void load_data_with_groups(const std::vector<int32_t>& values,
                                const std::vector<uint32_t>& groups,
                                const char* table_name) {
        std::string create_sql = "CREATE OR REPLACE TABLE " + std::string(table_name) +
                                  " (group_id INTEGER, value INTEGER)";
        conn_->Query(create_sql);

        {
            duckdb::Appender appender(*conn_, table_name);
            for (size_t i = 0; i < values.size(); i++) {
                appender.BeginRow();
                appender.Append<int32_t>(static_cast<int32_t>(groups[i]));
                appender.Append<int32_t>(values[i]);
                appender.EndRow();
            }
        }
    }

    void load_join_data(const std::vector<int32_t>& build, const std::vector<int32_t>& probe) {
        // Build table
        conn_->Query("CREATE OR REPLACE TABLE build_t (key INTEGER)");
        {
            duckdb::Appender appender(*conn_, "build_t");
            for (int32_t v : build) {
                appender.BeginRow();
                appender.Append<int32_t>(v);
                appender.EndRow();
            }
        }

        // Probe table
        conn_->Query("CREATE OR REPLACE TABLE probe_t (key INTEGER)");
        {
            duckdb::Appender appender(*conn_, "probe_t");
            for (int32_t v : probe) {
                appender.BeginRow();
                appender.Append<int32_t>(v);
                appender.EndRow();
            }
        }
    }

    double run_query(const std::string& sql, int warmup, int runs) {
        return benchmark_fn([&]() {
            auto result = conn_->Query(sql);
            if (result->HasError()) {
                std::cerr << "DuckDB Error: " << result->GetError() << std::endl;
            }
        }, warmup, runs);
    }

private:
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
};

// ============================================================================
// 报告生成
// ============================================================================

void print_header() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              ThunderDuck 全面版本基准测试报告                                                     ║\n";
    std::cout << "║                                   Apple M4 Max | macOS 14.x                                                       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n";
}

void print_section_header(const char* section) {
    std::cout << "\n╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮\n";
    std::cout << "│ " << std::left << std::setw(116) << section << "│\n";
    std::cout << "╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯\n";
}

void print_result_header() {
    std::cout << std::left
              << std::setw(12) << "版本"
              << std::setw(12) << "设备"
              << std::setw(12) << "时间(ms)"
              << std::setw(14) << "吞吐(GB/s)"
              << std::setw(12) << "vs DuckDB"
              << std::setw(12) << "vs V3"
              << "\n";
    std::cout << std::string(74, '-') << "\n";
}

void print_result_row(const BenchmarkResult& r) {
    std::cout << std::left
              << std::setw(12) << r.version
              << std::setw(12) << r.device
              << std::fixed << std::setprecision(3)
              << std::setw(12) << r.time_ms
              << std::setw(14) << r.throughput_gbps
              << std::setw(12);

    if (r.speedup_vs_duckdb >= 1.0) {
        std::cout << std::to_string(r.speedup_vs_duckdb).substr(0, 5) + "x";
    } else {
        std::cout << std::to_string(r.speedup_vs_duckdb).substr(0, 5) + "x";
    }

    std::cout << std::setw(12);
    if (r.speedup_vs_v3 >= 1.0) {
        std::cout << std::to_string(r.speedup_vs_v3).substr(0, 5) + "x";
    } else {
        std::cout << std::to_string(r.speedup_vs_v3).substr(0, 5) + "x";
    }
    std::cout << "\n";
}

// ============================================================================
// Filter 基准测试
// ============================================================================

void benchmark_filter(const BenchmarkConfig& config) {
    print_section_header("FILTER 算子测试: SELECT COUNT(*) FROM t WHERE value > 500000");

    for (size_t size : {config.medium_size, config.large_size}) {
        std::cout << "\n【数据量: " << size/1000000 << "M 行, "
                  << (size * sizeof(int32_t)) / (1024*1024) << " MB】\n\n";

        auto data = generate_random_i32(size);
        size_t data_bytes = size * sizeof(int32_t);

        // DuckDB
        DuckDBBenchmark duckdb;
        duckdb.load_data(data, "filter_t");
        double duckdb_time = duckdb.run_query("SELECT COUNT(*) FROM filter_t WHERE value > 500000",
                                               config.warmup_runs, config.test_runs);

        // V3
        double v3_time = benchmark_fn([&]() {
            filter::count_i32_v3(data.data(), size, filter::CompareOp::GT, 500000);
        }, config.warmup_runs, config.test_runs);

        // V4 (GPU)
        double v4_time = benchmark_fn([&]() {
            filter::count_i32_v4(data.data(), size, filter::CompareOp::GT, 500000);
        }, config.warmup_runs, config.test_runs);

        // V5
        double v5_time = benchmark_fn([&]() {
            filter::count_i32_v5(data.data(), size, filter::CompareOp::GT, 500000);
        }, config.warmup_runs, config.test_runs);

        // V6
        double v6_time = benchmark_fn([&]() {
            filter::count_i32_v6(data.data(), size, filter::CompareOp::GT, 500000);
        }, config.warmup_runs, config.test_runs);

        print_result_header();

        // DuckDB result
        BenchmarkResult r_duckdb = {
            "SELECT COUNT(*) WHERE value > 500000",
            "Filter.Count", "DuckDB", "CPU",
            data_bytes, size, duckdb_time,
            (data_bytes / (1024.0*1024*1024)) / (duckdb_time / 1000.0),
            1.0, duckdb_time / v3_time
        };
        print_result_row(r_duckdb);
        g_results.push_back(r_duckdb);

        // V3 result
        BenchmarkResult r_v3 = {
            "SELECT COUNT(*) WHERE value > 500000",
            "Filter.Count", "V3", "CPU SIMD",
            data_bytes, size, v3_time,
            (data_bytes / (1024.0*1024*1024)) / (v3_time / 1000.0),
            duckdb_time / v3_time, 1.0
        };
        print_result_row(r_v3);
        g_results.push_back(r_v3);

        // V7 (v4) result
        BenchmarkResult r_v4 = {
            "SELECT COUNT(*) WHERE value > 500000",
            "Filter.Count", "V7(GPU)", filter::is_filter_gpu_available() ? "GPU" : "CPU",
            data_bytes, size, v4_time,
            (data_bytes / (1024.0*1024*1024)) / (v4_time / 1000.0),
            duckdb_time / v4_time, v3_time / v4_time
        };
        print_result_row(r_v4);
        g_results.push_back(r_v4);

        // V8 (v5) result
        BenchmarkResult r_v5 = {
            "SELECT COUNT(*) WHERE value > 500000",
            "Filter.Count", "V8", "CPU SIMD",
            data_bytes, size, v5_time,
            (data_bytes / (1024.0*1024*1024)) / (v5_time / 1000.0),
            duckdb_time / v5_time, v3_time / v5_time
        };
        print_result_row(r_v5);
        g_results.push_back(r_v5);

        // V9 (v6) result
        BenchmarkResult r_v6 = {
            "SELECT COUNT(*) WHERE value > 500000",
            "Filter.Count", "V9", "CPU SIMD",
            data_bytes, size, v6_time,
            (data_bytes / (1024.0*1024*1024)) / (v6_time / 1000.0),
            duckdb_time / v6_time, v3_time / v6_time
        };
        print_result_row(r_v6);
        g_results.push_back(r_v6);
    }
}

// ============================================================================
// Aggregate 基准测试
// ============================================================================

void benchmark_aggregate(const BenchmarkConfig& config) {
    print_section_header("AGGREGATE 算子测试: SELECT SUM(value) FROM t");

    for (size_t size : {config.medium_size, config.large_size}) {
        std::cout << "\n【数据量: " << size/1000000 << "M 行, "
                  << (size * sizeof(int32_t)) / (1024*1024) << " MB】\n\n";

        auto data = generate_random_i32(size);
        size_t data_bytes = size * sizeof(int32_t);

        // DuckDB
        DuckDBBenchmark duckdb;
        duckdb.load_data(data, "agg_t");
        double duckdb_time = duckdb.run_query("SELECT SUM(value) FROM agg_t",
                                               config.warmup_runs, config.test_runs);

        // V2
        double v2_time = benchmark_fn([&]() {
            aggregate::sum_i32_v2(data.data(), size);
        }, config.warmup_runs, config.test_runs);

        // V3
        double v3_time = benchmark_fn([&]() {
            aggregate::sum_i32_v3(data.data(), size);
        }, config.warmup_runs, config.test_runs);

        // V4
        double v4_time = benchmark_fn([&]() {
            aggregate::sum_i32_v4(data.data(), size);
        }, config.warmup_runs, config.test_runs);

        print_result_header();

        // DuckDB
        BenchmarkResult r_duckdb = {
            "SELECT SUM(value)", "Aggregate.SUM", "DuckDB", "CPU",
            data_bytes, size, duckdb_time,
            (data_bytes / (1024.0*1024*1024)) / (duckdb_time / 1000.0),
            1.0, duckdb_time / v2_time
        };
        print_result_row(r_duckdb);
        g_results.push_back(r_duckdb);

        // V3 (labeled as baseline)
        BenchmarkResult r_v3 = {
            "SELECT SUM(value)", "Aggregate.SUM", "V3", "CPU SIMD",
            data_bytes, size, v2_time,
            (data_bytes / (1024.0*1024*1024)) / (v2_time / 1000.0),
            duckdb_time / v2_time, 1.0
        };
        print_result_row(r_v3);
        g_results.push_back(r_v3);

        // V7 (v3 GPU)
        BenchmarkResult r_v7 = {
            "SELECT SUM(value)", "Aggregate.SUM", "V7(GPU)",
            aggregate::is_aggregate_gpu_available() ? "GPU" : "CPU",
            data_bytes, size, v3_time,
            (data_bytes / (1024.0*1024*1024)) / (v3_time / 1000.0),
            duckdb_time / v3_time, v2_time / v3_time
        };
        print_result_row(r_v7);
        g_results.push_back(r_v7);

        // V9 (v4)
        BenchmarkResult r_v9 = {
            "SELECT SUM(value)", "Aggregate.SUM", "V9", "CPU SIMD",
            data_bytes, size, v4_time,
            (data_bytes / (1024.0*1024*1024)) / (v4_time / 1000.0),
            duckdb_time / v4_time, v2_time / v4_time
        };
        print_result_row(r_v9);
        g_results.push_back(r_v9);
    }
}

// ============================================================================
// Group Aggregate 基准测试
// ============================================================================

void benchmark_group_aggregate(const BenchmarkConfig& config) {
    print_section_header("GROUP BY 测试: SELECT group_id, SUM(value) FROM t GROUP BY group_id");

    size_t num_groups = 1000;

    for (size_t size : {config.medium_size, config.large_size}) {
        std::cout << "\n【数据量: " << size/1000000 << "M 行, " << num_groups << " 分组】\n\n";

        auto values = generate_random_i32(size);
        auto groups = generate_groups(size, num_groups);
        std::vector<int64_t> out_sums(num_groups);
        size_t data_bytes = size * (sizeof(int32_t) + sizeof(uint32_t));

        // DuckDB
        DuckDBBenchmark duckdb;
        duckdb.load_data_with_groups(values, groups, "group_t");
        double duckdb_time = duckdb.run_query(
            "SELECT group_id, SUM(value) FROM group_t GROUP BY group_id",
            config.warmup_runs, config.test_runs);

        // V4 Single
        double v4_single_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v4(values.data(), groups.data(), size, num_groups, out_sums.data());
        }, config.warmup_runs, config.test_runs);

        // V4 Parallel
        double v4_parallel_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), size, num_groups, out_sums.data());
        }, config.warmup_runs, config.test_runs);

        // V5 GPU
        double v5_gpu_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v5(values.data(), groups.data(), size, num_groups, out_sums.data());
        }, config.warmup_runs, config.test_runs);

        // V6 Auto
        double v6_auto_time = benchmark_fn([&]() {
            aggregate::group_sum_i32_v6(values.data(), groups.data(), size, num_groups, out_sums.data());
        }, config.warmup_runs, config.test_runs);

        print_result_header();

        // DuckDB
        BenchmarkResult r_duckdb = {
            "GROUP BY SUM", "GroupAggregate", "DuckDB", "CPU",
            data_bytes, size, duckdb_time,
            (data_bytes / (1024.0*1024*1024)) / (duckdb_time / 1000.0),
            1.0, duckdb_time / v4_single_time
        };
        print_result_row(r_duckdb);
        g_results.push_back(r_duckdb);

        // V7 (V4 Single)
        BenchmarkResult r_v4s = {
            "GROUP BY SUM", "GroupAggregate", "V7(单线程)", "CPU",
            data_bytes, size, v4_single_time,
            (data_bytes / (1024.0*1024*1024)) / (v4_single_time / 1000.0),
            duckdb_time / v4_single_time, 1.0
        };
        print_result_row(r_v4s);
        g_results.push_back(r_v4s);

        // V8 (V4 Parallel)
        BenchmarkResult r_v4p = {
            "GROUP BY SUM", "GroupAggregate", "V8(并行)", "CPU 4核",
            data_bytes, size, v4_parallel_time,
            (data_bytes / (1024.0*1024*1024)) / (v4_parallel_time / 1000.0),
            duckdb_time / v4_parallel_time, v4_single_time / v4_parallel_time
        };
        print_result_row(r_v4p);
        g_results.push_back(r_v4p);

        // V9 (V5 GPU)
        BenchmarkResult r_v5g = {
            "GROUP BY SUM", "GroupAggregate", "V9(GPU)",
            aggregate::is_group_aggregate_v2_available() ? "GPU" : "CPU",
            data_bytes, size, v5_gpu_time,
            (data_bytes / (1024.0*1024*1024)) / (v5_gpu_time / 1000.0),
            duckdb_time / v5_gpu_time, v4_single_time / v5_gpu_time
        };
        print_result_row(r_v5g);
        g_results.push_back(r_v5g);

        // V9.3 (V6 Auto)
        BenchmarkResult r_v6a = {
            "GROUP BY SUM", "GroupAggregate", "V9.3(智能)", "AUTO",
            data_bytes, size, v6_auto_time,
            (data_bytes / (1024.0*1024*1024)) / (v6_auto_time / 1000.0),
            duckdb_time / v6_auto_time, v4_single_time / v6_auto_time
        };
        print_result_row(r_v6a);
        g_results.push_back(r_v6a);
    }
}

// ============================================================================
// TopK 基准测试
// ============================================================================

void benchmark_topk(const BenchmarkConfig& config) {
    print_section_header("TopK 测试: SELECT * FROM t ORDER BY value DESC LIMIT 10");

    size_t k = 10;

    for (size_t size : {config.medium_size, config.large_size}) {
        std::cout << "\n【数据量: " << size/1000000 << "M 行, K=" << k << "】\n\n";

        auto data = generate_random_i32(size);
        std::vector<int32_t> out_values(k);
        std::vector<uint32_t> out_indices(k);
        size_t data_bytes = size * sizeof(int32_t);

        // DuckDB
        DuckDBBenchmark duckdb;
        duckdb.load_data(data, "topk_t");
        double duckdb_time = duckdb.run_query(
            "SELECT value FROM topk_t ORDER BY value DESC LIMIT 10",
            config.warmup_runs, config.test_runs);

        // V3
        double v3_time = benchmark_fn([&]() {
            sort::topk_max_i32_v3(data.data(), size, k, out_values.data(), out_indices.data());
        }, config.warmup_runs, config.test_runs);

        // V4
        double v4_time = benchmark_fn([&]() {
            sort::topk_max_i32_v4(data.data(), size, k, out_values.data(), out_indices.data());
        }, config.warmup_runs, config.test_runs);

        // V5
        double v5_time = benchmark_fn([&]() {
            sort::topk_max_i32_v5(data.data(), size, k, out_values.data(), out_indices.data());
        }, config.warmup_runs, config.test_runs);

        // V6 (GPU)
        double v6_time = benchmark_fn([&]() {
            sort::topk_max_i32_v6(data.data(), size, k, out_values.data(), out_indices.data());
        }, config.warmup_runs, config.test_runs);

        print_result_header();

        // DuckDB
        BenchmarkResult r_duckdb = {
            "ORDER BY DESC LIMIT 10", "TopK", "DuckDB", "CPU",
            data_bytes, size, duckdb_time,
            (data_bytes / (1024.0*1024*1024)) / (duckdb_time / 1000.0),
            1.0, duckdb_time / v3_time
        };
        print_result_row(r_duckdb);
        g_results.push_back(r_duckdb);

        // V3
        BenchmarkResult r_v3 = {
            "ORDER BY DESC LIMIT 10", "TopK", "V3", "CPU Heap",
            data_bytes, size, v3_time,
            (data_bytes / (1024.0*1024*1024)) / (v3_time / 1000.0),
            duckdb_time / v3_time, 1.0
        };
        print_result_row(r_v3);
        g_results.push_back(r_v3);

        // V7 (v4)
        BenchmarkResult r_v4 = {
            "ORDER BY DESC LIMIT 10", "TopK", "V7(采样)", "CPU SIMD",
            data_bytes, size, v4_time,
            (data_bytes / (1024.0*1024*1024)) / (v4_time / 1000.0),
            duckdb_time / v4_time, v3_time / v4_time
        };
        print_result_row(r_v4);
        g_results.push_back(r_v4);

        // V8 (v5)
        BenchmarkResult r_v5 = {
            "ORDER BY DESC LIMIT 10", "TopK", "V8(计数)", "CPU",
            data_bytes, size, v5_time,
            (data_bytes / (1024.0*1024*1024)) / (v5_time / 1000.0),
            duckdb_time / v5_time, v3_time / v5_time
        };
        print_result_row(r_v5);
        g_results.push_back(r_v5);

        // V9 (v6 GPU)
        BenchmarkResult r_v6 = {
            "ORDER BY DESC LIMIT 10", "TopK", "V9(GPU)",
            sort::is_topk_gpu_available() ? "GPU" : "CPU",
            data_bytes, size, v6_time,
            (data_bytes / (1024.0*1024*1024)) / (v6_time / 1000.0),
            duckdb_time / v6_time, v3_time / v6_time
        };
        print_result_row(r_v6);
        g_results.push_back(r_v6);
    }
}

// ============================================================================
// Hash Join 基准测试
// ============================================================================

void benchmark_hash_join(const BenchmarkConfig& config) {
    print_section_header("HASH JOIN 测试: SELECT * FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key");

    // 测试场景: 100K build x 1M probe, ~10% 匹配率
    size_t build_size = 100000;
    size_t probe_size = 1000000;

    std::cout << "\n【Build: 100K 行, Probe: 1M 行, ~10% 匹配率】\n\n";

    auto build_keys = generate_random_i32(build_size, 0, 1000000);
    auto probe_keys = generate_random_i32(probe_size, 0, 1000000);
    size_t data_bytes = (build_size + probe_size) * sizeof(int32_t);

    // 创建结果缓冲区
    join::JoinResult* result = join::create_join_result(probe_size * 2);

    // DuckDB
    DuckDBBenchmark duckdb;
    duckdb.load_join_data(build_keys, probe_keys);
    double duckdb_time = duckdb.run_query(
        "SELECT COUNT(*) FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key",
        config.warmup_runs, config.test_runs);

    // V3
    double v3_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v3(build_keys.data(), build_size,
                               probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V4
    double v4_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v4(build_keys.data(), build_size,
                               probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V10
    double v10_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v10(build_keys.data(), build_size,
                                probe_keys.data(), probe_size,
                                join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V6 (新版超激进预取)
    double v6_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v6(build_keys.data(), build_size,
                               probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V10.1 (零拷贝优化)
    double v10_1_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v10_1(build_keys.data(), build_size,
                                   probe_keys.data(), probe_size,
                                   join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V10.2 (单次哈希优化)
    double v10_2_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v10_2(build_keys.data(), build_size,
                                   probe_keys.data(), probe_size,
                                   join::JoinType::INNER, result);
    }, config.warmup_runs, config.test_runs);

    // V11 (SIMD 哈希探测 - 简化版)
    double v11_simple_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v11_config(build_keys.data(), build_size,
                                        probe_keys.data(), probe_size,
                                        join::JoinType::INNER, result, false);
    }, config.warmup_runs, config.test_runs);

    // V11 (SIMD 哈希探测 - 并行槽位比较版)
    double v11_parallel_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v11_config(build_keys.data(), build_size,
                                        probe_keys.data(), probe_size,
                                        join::JoinType::INNER, result, true);
    }, config.warmup_runs, config.test_runs);

    print_result_header();

    // DuckDB
    BenchmarkResult r_duckdb = {
        "INNER JOIN", "HashJoin", "DuckDB", "CPU",
        data_bytes, build_size + probe_size, duckdb_time,
        (data_bytes / (1024.0*1024*1024)) / (duckdb_time / 1000.0),
        1.0, duckdb_time / v3_time
    };
    print_result_row(r_duckdb);
    g_results.push_back(r_duckdb);

    // V3
    BenchmarkResult r_v3 = {
        "INNER JOIN", "HashJoin", "V3", "CPU SIMD",
        data_bytes, build_size + probe_size, v3_time,
        (data_bytes / (1024.0*1024*1024)) / (v3_time / 1000.0),
        duckdb_time / v3_time, 1.0
    };
    print_result_row(r_v3);
    g_results.push_back(r_v3);

    // V7 (v4)
    BenchmarkResult r_v4 = {
        "INNER JOIN", "HashJoin", "V7(自适应)", "CPU/GPU",
        data_bytes, build_size + probe_size, v4_time,
        (data_bytes / (1024.0*1024*1024)) / (v4_time / 1000.0),
        duckdb_time / v4_time, v3_time / v4_time
    };
    print_result_row(r_v4);
    g_results.push_back(r_v4);

    // V10
    BenchmarkResult r_v10 = {
        "INNER JOIN", "HashJoin", "V10", "CPU SIMD",
        data_bytes, build_size + probe_size, v10_time,
        (data_bytes / (1024.0*1024*1024)) / (v10_time / 1000.0),
        duckdb_time / v10_time, v3_time / v10_time
    };
    print_result_row(r_v10);
    g_results.push_back(r_v10);

    // V6 (超激进预取)
    BenchmarkResult r_v6 = {
        "INNER JOIN", "HashJoin", "V6(预取优化)", "CPU SIMD",
        data_bytes, build_size + probe_size, v6_time,
        (data_bytes / (1024.0*1024*1024)) / (v6_time / 1000.0),
        duckdb_time / v6_time, v3_time / v6_time
    };
    print_result_row(r_v6);
    g_results.push_back(r_v6);

    // V10.1 (零拷贝优化)
    BenchmarkResult r_v10_1 = {
        "INNER JOIN", "HashJoin", "V10.1(零拷贝)", "CPU SIMD",
        data_bytes, build_size + probe_size, v10_1_time,
        (data_bytes / (1024.0*1024*1024)) / (v10_1_time / 1000.0),
        duckdb_time / v10_1_time, v3_time / v10_1_time
    };
    print_result_row(r_v10_1);
    g_results.push_back(r_v10_1);

    // V10.2 (单次哈希优化)
    BenchmarkResult r_v10_2 = {
        "INNER JOIN", "HashJoin", "V10.2(单哈希)", "CPU SIMD",
        data_bytes, build_size + probe_size, v10_2_time,
        (data_bytes / (1024.0*1024*1024)) / (v10_2_time / 1000.0),
        duckdb_time / v10_2_time, v3_time / v10_2_time
    };
    print_result_row(r_v10_2);
    g_results.push_back(r_v10_2);

    // V11 (SIMD 哈希探测 - 简化版)
    BenchmarkResult r_v11_simple = {
        "INNER JOIN", "HashJoin", "V11(SIMD简化)", "CPU SIMD",
        data_bytes, build_size + probe_size, v11_simple_time,
        (data_bytes / (1024.0*1024*1024)) / (v11_simple_time / 1000.0),
        duckdb_time / v11_simple_time, v3_time / v11_simple_time
    };
    print_result_row(r_v11_simple);
    g_results.push_back(r_v11_simple);

    // V11 (SIMD 哈希探测 - 并行槽位比较版)
    BenchmarkResult r_v11_parallel = {
        "INNER JOIN", "HashJoin", "V11(SIMD并行)", "CPU SIMD",
        data_bytes, build_size + probe_size, v11_parallel_time,
        (data_bytes / (1024.0*1024*1024)) / (v11_parallel_time / 1000.0),
        duckdb_time / v11_parallel_time, v3_time / v11_parallel_time
    };
    print_result_row(r_v11_parallel);
    g_results.push_back(r_v11_parallel);

    join::free_join_result(result);

    // SEMI JOIN 测试
    print_section_header("SEMI JOIN 测试: SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE ...)");

    std::cout << "\n【Build: 100K 行, Probe: 1M 行】\n\n";

    result = join::create_join_result(probe_size);

    // DuckDB SEMI
    double duckdb_semi_time = duckdb.run_query(
        "SELECT COUNT(*) FROM probe_t WHERE key IN (SELECT key FROM build_t)",
        config.warmup_runs, config.test_runs);

    // V4 SEMI
    double v4_semi_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v4(build_keys.data(), build_size,
                               probe_keys.data(), probe_size,
                               join::JoinType::SEMI, result);
    }, config.warmup_runs, config.test_runs);

    // V10 SEMI
    double v10_semi_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v10(build_keys.data(), build_size,
                                probe_keys.data(), probe_size,
                                join::JoinType::SEMI, result);
    }, config.warmup_runs, config.test_runs);

    print_result_header();

    BenchmarkResult r_duckdb_semi = {
        "SEMI JOIN", "HashJoin.SEMI", "DuckDB", "CPU",
        data_bytes, build_size + probe_size, duckdb_semi_time,
        (data_bytes / (1024.0*1024*1024)) / (duckdb_semi_time / 1000.0),
        1.0, 1.0
    };
    print_result_row(r_duckdb_semi);

    BenchmarkResult r_v4_semi = {
        "SEMI JOIN", "HashJoin.SEMI", "V7", "CPU",
        data_bytes, build_size + probe_size, v4_semi_time,
        (data_bytes / (1024.0*1024*1024)) / (v4_semi_time / 1000.0),
        duckdb_semi_time / v4_semi_time, 1.0
    };
    print_result_row(r_v4_semi);

    BenchmarkResult r_v10_semi = {
        "SEMI JOIN", "HashJoin.SEMI", "V10(优化)", "CPU",
        data_bytes, build_size + probe_size, v10_semi_time,
        (data_bytes / (1024.0*1024*1024)) / (v10_semi_time / 1000.0),
        duckdb_semi_time / v10_semi_time, v4_semi_time / v10_semi_time
    };
    print_result_row(r_v10_semi);

    join::free_join_result(result);
}

// ============================================================================
// V10 特性测试
// ============================================================================

void benchmark_v10_features(const BenchmarkConfig& config) {
    print_section_header("V10 新特性测试");

    // Range Join 测试
    std::cout << "\n【Range Join: 100K left x 1K ranges】\n\n";

    size_t left_count = 100000;
    size_t right_count = 1000;

    auto left_keys = generate_random_i32(left_count, 0, 100000);
    std::vector<int32_t> right_lo(right_count), right_hi(right_count);

    std::mt19937 gen(42);
    for (size_t i = 0; i < right_count; i++) {
        int32_t lo = gen() % 99000;
        right_lo[i] = lo;
        right_hi[i] = lo + (gen() % 1000) + 100;
    }

    join::JoinResult* result = join::create_join_result(left_count * 2);

    // Range Join 标量
    join::JoinConfigV10 cfg_scalar;
    cfg_scalar.range_join_simd = false;
    double range_scalar_time = benchmark_fn([&]() {
        result->count = 0;
        join::range_join_i32_config(left_keys.data(), left_count,
                                     right_lo.data(), right_hi.data(), right_count,
                                     result, cfg_scalar);
    }, config.warmup_runs, config.test_runs);

    // Range Join SIMD
    join::JoinConfigV10 cfg_simd;
    cfg_simd.range_join_simd = true;
    double range_simd_time = benchmark_fn([&]() {
        result->count = 0;
        join::range_join_i32_config(left_keys.data(), left_count,
                                     right_lo.data(), right_hi.data(), right_count,
                                     result, cfg_simd);
    }, config.warmup_runs, config.test_runs);

    size_t data_bytes = (left_count + right_count * 2) * sizeof(int32_t);

    std::cout << std::left
              << std::setw(20) << "方法"
              << std::setw(15) << "时间(ms)"
              << std::setw(15) << "加速比"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    std::cout << std::setw(20) << "Range(标量)"
              << std::fixed << std::setprecision(3)
              << std::setw(15) << range_scalar_time
              << std::setw(15) << "1.00x"
              << "\n";

    std::cout << std::setw(20) << "Range(SIMD)"
              << std::setw(15) << range_simd_time
              << std::setw(15) << std::to_string(range_scalar_time / range_simd_time).substr(0, 5) + "x"
              << "\n";

    join::free_join_result(result);
}

// ============================================================================
// 总结报告
// ============================================================================

void print_summary() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                           性能总结                                                               ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n";

    // 统计各版本的平均加速比
    std::map<std::string, std::vector<double>> speedups;
    for (const auto& r : g_results) {
        if (std::string(r.version) != "DuckDB") {
            speedups[r.version].push_back(r.speedup_vs_duckdb);
        }
    }

    std::cout << "\n【各版本 vs DuckDB 平均加速比】\n\n";
    std::cout << std::left << std::setw(20) << "版本" << std::setw(15) << "平均加速比" << "\n";
    std::cout << std::string(35, '-') << "\n";

    for (const auto& [version, speeds] : speedups) {
        double avg = std::accumulate(speeds.begin(), speeds.end(), 0.0) / speeds.size();
        std::cout << std::setw(20) << version
                  << std::fixed << std::setprecision(2) << avg << "x\n";
    }

    std::cout << "\n【优化建议】\n\n";

    // 找出性能最差的场景
    double worst_speedup = 999;
    const BenchmarkResult* worst = nullptr;
    for (const auto& r : g_results) {
        if (std::string(r.version) != "DuckDB" && r.speedup_vs_duckdb < worst_speedup) {
            worst_speedup = r.speedup_vs_duckdb;
            worst = &r;
        }
    }

    if (worst && worst_speedup < 1.5) {
        std::cout << "  ! " << worst->operator_name << " (" << worst->version << "): "
                  << std::fixed << std::setprecision(2) << worst_speedup << "x"
                  << " - 需要进一步优化\n";
    }

    // 找出最佳性能
    double best_speedup = 0;
    const BenchmarkResult* best = nullptr;
    for (const auto& r : g_results) {
        if (std::string(r.version) != "DuckDB" && r.speedup_vs_duckdb > best_speedup) {
            best_speedup = r.speedup_vs_duckdb;
            best = &r;
        }
    }

    if (best) {
        std::cout << "  * " << best->operator_name << " (" << best->version << "): "
                  << std::fixed << std::setprecision(2) << best_speedup << "x"
                  << " - 最佳性能\n";
    }

    std::cout << "\n╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--small") == 0) {
            config.medium_size = 100000;
            config.large_size = 1000000;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        }
    }

    print_header();

    std::cout << "\n配置:\n";
    std::cout << "  - 预热轮数: " << config.warmup_runs << "\n";
    std::cout << "  - 测试轮数: " << config.test_runs << "\n";
    std::cout << "  - 中等数据: " << config.medium_size/1000 << "K 行\n";
    std::cout << "  - 大数据: " << config.large_size/1000000 << "M 行\n";

    // 运行测试
    benchmark_filter(config);
    benchmark_aggregate(config);
    benchmark_group_aggregate(config);
    benchmark_topk(config);
    benchmark_hash_join(config);
    benchmark_v10_features(config);

    // 打印总结
    print_summary();

    return 0;
}
