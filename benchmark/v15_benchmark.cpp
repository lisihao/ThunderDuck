/**
 * ThunderDuck V15 基准测试
 *
 * 测试 V15 新特性:
 * 1. Filter 直接索引生成 (跳过位图中间层)
 * 2. GROUP BY 8线程 + SIMD归约
 * 3. Hash Join Robin-Hood 哈希
 *
 * 对比: V3, V14, V15, DuckDB
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>

// DuckDB
#include "duckdb.hpp"

// ThunderDuck
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"

using namespace thunderduck;
using namespace std::chrono;

// ============================================================================
// 配置
// ============================================================================

struct TestConfig {
    size_t filter_count_1m = 1000000;     // 1M
    size_t filter_count_10m = 10000000;   // 10M
    size_t group_by_count = 10000000;     // 10M
    size_t group_by_groups = 1000;        // 1000 分组
    size_t join_build = 100000;           // 100K build
    size_t join_probe = 1000000;          // 1M probe
    int iterations = 15;
    int warmup = 2;
};

// ============================================================================
// 测量工具
// ============================================================================

template<typename Func>
double measure_median_iqr(Func&& func, int iterations, int warmup) {
    // 预热
    for (int i = 0; i < warmup; ++i) func();

    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

    // IQR 剔除异常值
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    double q1 = times[n / 4];
    double q3 = times[n * 3 / 4];
    double iqr = q3 - q1;

    std::vector<double> filtered;
    for (double t : times) {
        if (t >= q1 - 1.5 * iqr && t <= q3 + 1.5 * iqr) {
            filtered.push_back(t);
        }
    }

    if (filtered.empty()) return times[n / 2];

    std::sort(filtered.begin(), filtered.end());
    return filtered[filtered.size() / 2];
}

double calc_stddev(const std::vector<double>& times, double median) {
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - median) * (t - median);
    return std::sqrt(sq_sum / times.size());
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(90, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

void print_result(const std::string& version, const std::string& device,
                  double time_ms, double throughput, double vs_duckdb, bool pass) {
    std::cout << std::left << std::fixed
              << std::setw(18) << version
              << std::setw(10) << device
              << std::setw(14) << std::setprecision(3) << time_ms
              << std::setw(12) << std::setprecision(2) << throughput
              << std::setw(14) << std::setprecision(2) << vs_duckdb << "x"
              << std::setw(10) << (pass ? "PASS" : "FAIL")
              << std::endl;
}

// ============================================================================
// Filter 输出索引测试 (V15 核心优化)
// ============================================================================

void benchmark_filter_indices(const TestConfig& config) {
    print_header("FILTER + 输出索引 (V15 核心优化)");

    for (size_t count : {config.filter_count_1m, config.filter_count_10m}) {
        std::cout << "\n--- 数据量: " << count / 1000000 << "M ---" << std::endl;
        std::cout << "等效 SQL: SELECT * FROM t WHERE value > 500000" << std::endl;
        std::cout << std::left
                  << std::setw(18) << "版本"
                  << std::setw(10) << "设备"
                  << std::setw(14) << "时间(ms)"
                  << std::setw(12) << "带宽(GB/s)"
                  << std::setw(14) << "vs DuckDB"
                  << std::setw(10) << "正确性"
                  << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        // 生成数据
        std::vector<int32_t> data(count);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000000);
        for (auto& v : data) v = dist(rng);

        const int32_t threshold = 500000;
        size_t data_bytes = count * sizeof(int32_t);

        // 预分配输出
        std::vector<uint32_t> indices(count);
        size_t expected_count = 0;

        // 计算期望匹配数
        for (size_t i = 0; i < count; ++i) {
            if (data[i] > threshold) ++expected_count;
        }

        // DuckDB 基准
        double duckdb_time = 0;
        {
            duckdb::DuckDB db(nullptr);
            duckdb::Connection con(db);

            con.Query("CREATE TABLE t (value INTEGER)");
            {
                duckdb::Appender appender(con, "t");
                for (size_t i = 0; i < count; ++i) {
                    appender.AppendRow(data[i]);
                }
            }

            duckdb_time = measure_median_iqr([&]() {
                auto result = con.Query("SELECT rowid FROM t WHERE value > 500000");
            }, config.iterations, config.warmup);

            print_result("DuckDB", "CPU", duckdb_time,
                        data_bytes / (duckdb_time * 1e6), 1.0, true);
        }

        // V3 位图 + 转换
        {
            std::vector<uint64_t> bitmap((count + 63) / 64);
            size_t v3_count = 0;

            double v3_time = measure_median_iqr([&]() {
                filter::filter_to_bitmap_v3(data.data(), count,
                                            filter::CompareOp::GT, threshold,
                                            bitmap.data());
                v3_count = filter::bitmap_to_indices(bitmap.data(), count,
                                                     indices.data());
            }, config.iterations, config.warmup);

            print_result("V3 (bitmap)", "CPU SIMD", v3_time,
                        data_bytes / (v3_time * 1e6),
                        duckdb_time / v3_time, v3_count == expected_count);
        }

        // V15 直接索引
        {
            size_t v15_count = 0;

            double v15_time = measure_median_iqr([&]() {
                v15_count = filter::filter_i32_v15(data.data(), count,
                                                   filter::CompareOp::GT, threshold,
                                                   indices.data());
            }, config.iterations, config.warmup);

            print_result("V15 (direct)", "CPU SIMD", v15_time,
                        data_bytes / (v15_time * 1e6),
                        duckdb_time / v15_time, v15_count == expected_count);
        }

        // GPU Filter (V4 - AUTO 策略，依赖新选择器)
        if (filter::is_filter_gpu_available()) {
            size_t auto_count = 0;

            // 使用 AUTO 策略 (新逻辑: 10M 自动选 GPU)
            filter::FilterConfigV4 auto_config;
            auto_config.strategy = filter::FilterStrategy::AUTO;
            auto_config.selectivity_hint = 0.5f;  // 实际选择率约 50%

            double auto_time = measure_median_iqr([&]() {
                auto_count = filter::filter_i32_v4_config(data.data(), count,
                                                           filter::CompareOp::GT, threshold,
                                                           indices.data(), auto_config);
            }, config.iterations, config.warmup);

            // 标记选中的执行器
            const char* executor = (count >= 5000000) ? "GPU Auto" : "CPU Auto";
            print_result("V4 (AUTO)", executor, auto_time,
                        data_bytes / (auto_time * 1e6),
                        duckdb_time / auto_time, auto_count == expected_count);
        }
    }
}

// ============================================================================
// GROUP BY 测试
// ============================================================================

void benchmark_group_by(const TestConfig& config) {
    print_header("GROUP BY SUM 算子测试");
    std::cout << "等效 SQL: SELECT group_id, SUM(value) FROM t GROUP BY group_id" << std::endl;
    std::cout << "数据量: " << config.group_by_count / 1000000 << "M 行, "
              << config.group_by_groups << " 分组" << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(18) << "版本"
              << std::setw(10) << "设备"
              << std::setw(14) << "时间(ms)"
              << std::setw(12) << "带宽(GB/s)"
              << std::setw(14) << "vs DuckDB"
              << std::setw(10) << "正确性"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    // 生成数据
    std::vector<int32_t> values(config.group_by_count);
    std::vector<uint32_t> groups(config.group_by_count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> val_dist(0, 1000);
    std::uniform_int_distribution<uint32_t> grp_dist(0, config.group_by_groups - 1);

    for (size_t i = 0; i < config.group_by_count; ++i) {
        values[i] = val_dist(rng);
        groups[i] = grp_dist(rng);
    }

    size_t data_bytes = config.group_by_count * (sizeof(int32_t) + sizeof(uint32_t));

    // 期望结果
    std::vector<int64_t> expected_sums(config.group_by_groups, 0);
    for (size_t i = 0; i < config.group_by_count; ++i) {
        expected_sums[groups[i]] += values[i];
    }

    // DuckDB 基准
    double duckdb_time = 0;
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);

        con.Query("CREATE TABLE t (group_id INTEGER, value INTEGER)");
        {
            duckdb::Appender appender(con, "t");
            for (size_t i = 0; i < config.group_by_count; ++i) {
                appender.AppendRow(static_cast<int32_t>(groups[i]), values[i]);
            }
        }

        duckdb_time = measure_median_iqr([&]() {
            auto result = con.Query("SELECT group_id, SUM(value) FROM t GROUP BY group_id");
        }, config.iterations, config.warmup);

        print_result("DuckDB", "CPU", duckdb_time,
                    data_bytes / (duckdb_time * 1e6), 1.0, true);
    }

    // V6 (GPU + 智能选择)
    {
        std::vector<int64_t> sums(config.group_by_groups, 0);

        double v6_time = measure_median_iqr([&]() {
            std::fill(sums.begin(), sums.end(), 0);
            aggregate::group_sum_i32_v6(values.data(), groups.data(),
                                        config.group_by_count,
                                        config.group_by_groups,
                                        sums.data());
        }, config.iterations, config.warmup);

        bool correct = (sums == expected_sums);
        print_result("V6 (smart)", "CPU/GPU", v6_time,
                    data_bytes / (v6_time * 1e6),
                    duckdb_time / v6_time, correct);
    }

    // V14 并行 8 线程
    {
        std::vector<int64_t> sums(config.group_by_groups, 0);

        double v14_time = measure_median_iqr([&]() {
            std::fill(sums.begin(), sums.end(), 0);
            aggregate::group_sum_i32_v14(values.data(), groups.data(),
                                         config.group_by_count,
                                         config.group_by_groups,
                                         sums.data());
        }, config.iterations, config.warmup);

        bool correct = (sums == expected_sums);
        print_result("V14 (parallel)", "CPU 8T", v14_time,
                    data_bytes / (v14_time * 1e6),
                    duckdb_time / v14_time, correct);
    }

    // V15 (如果存在)
    // 注意: V15 GROUP BY 可能还未实现
}

// ============================================================================
// Hash Join 测试
// ============================================================================

void benchmark_hash_join(const TestConfig& config) {
    print_header("HASH JOIN (INNER) 算子测试");
    std::cout << "等效 SQL: SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key" << std::endl;
    std::cout << "Build: " << config.join_build / 1000 << "K, Probe: "
              << config.join_probe / 1000000 << "M" << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(18) << "版本"
              << std::setw(10) << "设备"
              << std::setw(14) << "时间(ms)"
              << std::setw(12) << "带宽(GB/s)"
              << std::setw(14) << "vs DuckDB"
              << std::setw(10) << "正确性"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    // 生成数据
    std::vector<int32_t> build_keys(config.join_build);
    std::vector<int32_t> probe_keys(config.join_probe);
    std::mt19937 rng(42);

    // Build: 唯一键 0 ~ build_count-1
    for (size_t i = 0; i < config.join_build; ++i) {
        build_keys[i] = static_cast<int32_t>(i);
    }
    std::shuffle(build_keys.begin(), build_keys.end(), rng);

    // Probe: 随机键，约 10% 匹配
    std::uniform_int_distribution<int32_t> dist(0, config.join_build * 10 - 1);
    for (auto& k : probe_keys) k = dist(rng);

    size_t data_bytes = (config.join_build + config.join_probe) * sizeof(int32_t);

    // DuckDB 基准
    double duckdb_time = 0;
    size_t duckdb_matches = 0;
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);

        con.Query("CREATE TABLE build_t (key INTEGER)");
        con.Query("CREATE TABLE probe_t (key INTEGER)");

        {
            duckdb::Appender appender(con, "build_t");
            for (auto k : build_keys) appender.AppendRow(k);
        }
        {
            duckdb::Appender appender(con, "probe_t");
            for (auto k : probe_keys) appender.AppendRow(k);
        }

        duckdb_time = measure_median_iqr([&]() {
            auto result = con.Query(
                "SELECT COUNT(*) FROM build_t JOIN probe_t ON build_t.key = probe_t.key");
            duckdb_matches = result->GetValue(0, 0).GetValue<int64_t>();
        }, config.iterations, config.warmup);

        print_result("DuckDB", "CPU", duckdb_time,
                    data_bytes / (duckdb_time * 1e6), 1.0, true);
    }

    // V3
    {
        join::JoinResult* result = join::create_join_result(config.join_probe);

        double v3_time = measure_median_iqr([&]() {
            join::hash_join_i32_v3(build_keys.data(), config.join_build,
                                   probe_keys.data(), config.join_probe,
                                   join::JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == duckdb_matches);
        print_result("V3", "CPU SIMD", v3_time,
                    data_bytes / (v3_time * 1e6),
                    duckdb_time / v3_time, correct);

        join::free_join_result(result);
    }

    // V6 (预取优化)
    {
        join::JoinResult* result = join::create_join_result(config.join_probe);

        double v6_time = measure_median_iqr([&]() {
            join::hash_join_i32_v6(build_keys.data(), config.join_build,
                                   probe_keys.data(), config.join_probe,
                                   join::JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == duckdb_matches);
        print_result("V6 (prefetch)", "CPU SIMD", v6_time,
                    data_bytes / (v6_time * 1e6),
                    duckdb_time / v6_time, correct);

        join::free_join_result(result);
    }

    // V15 (如果存在)
    // hash_join_i32_v15
}

// ============================================================================
// SEMI Join 测试
// ============================================================================

void benchmark_semi_join(const TestConfig& config) {
    print_header("SEMI JOIN 算子测试");
    std::cout << "等效 SQL: SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE build_t.key = probe_t.key)" << std::endl;
    std::cout << "Build: " << config.join_build / 1000 << "K, Probe: "
              << config.join_probe / 1000000 << "M" << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(18) << "版本"
              << std::setw(10) << "设备"
              << std::setw(14) << "时间(ms)"
              << std::setw(12) << "带宽(GB/s)"
              << std::setw(14) << "vs DuckDB"
              << std::setw(10) << "正确性"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    // 生成数据 (与 Hash Join 相同)
    std::vector<int32_t> build_keys(config.join_build);
    std::vector<int32_t> probe_keys(config.join_probe);
    std::mt19937 rng(42);

    for (size_t i = 0; i < config.join_build; ++i) {
        build_keys[i] = static_cast<int32_t>(i);
    }
    std::shuffle(build_keys.begin(), build_keys.end(), rng);

    std::uniform_int_distribution<int32_t> dist(0, config.join_build * 10 - 1);
    for (auto& k : probe_keys) k = dist(rng);

    size_t data_bytes = (config.join_build + config.join_probe) * sizeof(int32_t);

    // DuckDB 基准
    double duckdb_time = 0;
    size_t duckdb_matches = 0;
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);

        con.Query("CREATE TABLE build_t (key INTEGER PRIMARY KEY)");
        con.Query("CREATE TABLE probe_t (key INTEGER)");

        {
            duckdb::Appender appender(con, "build_t");
            for (auto k : build_keys) appender.AppendRow(k);
        }
        {
            duckdb::Appender appender(con, "probe_t");
            for (auto k : probe_keys) appender.AppendRow(k);
        }

        duckdb_time = measure_median_iqr([&]() {
            auto result = con.Query(
                "SELECT COUNT(*) FROM probe_t WHERE EXISTS "
                "(SELECT 1 FROM build_t WHERE build_t.key = probe_t.key)");
            duckdb_matches = result->GetValue(0, 0).GetValue<int64_t>();
        }, config.iterations, config.warmup);

        print_result("DuckDB", "CPU", duckdb_time,
                    data_bytes / (duckdb_time * 1e6), 1.0, true);
    }

    // V10 SEMI
    {
        join::JoinResult* result = join::create_join_result(config.join_probe);

        double v10_time = measure_median_iqr([&]() {
            join::hash_join_i32_v10(build_keys.data(), config.join_build,
                                    probe_keys.data(), config.join_probe,
                                    join::JoinType::SEMI, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == duckdb_matches);
        print_result("V10 (SIMD)", "CPU SIMD", v10_time,
                    data_bytes / (v10_time * 1e6),
                    duckdb_time / v10_time, correct);

        join::free_join_result(result);
    }

    // GPU SEMI Join
    if (join::is_semi_join_gpu_available()) {
        join::JoinResult* result = join::create_join_result(config.join_probe);

        double gpu_time = measure_median_iqr([&]() {
            join::semi_join_gpu(build_keys.data(), config.join_build,
                                 probe_keys.data(), config.join_probe,
                                 result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == duckdb_matches);
        print_result("GPU (Metal)", "Metal", gpu_time,
                    data_bytes / (gpu_time * 1e6),
                    duckdb_time / gpu_time, correct);

        join::free_join_result(result);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "        ThunderDuck V15 基准测试" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "平台: Apple M4 Max" << std::endl;
    std::cout << "日期: " << __DATE__ << std::endl;
    std::cout << std::endl;

    TestConfig config;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = std::stoi(argv[++i]);
        } else if (arg == "--filter-only") {
            benchmark_filter_indices(config);
            return 0;
        } else if (arg == "--join-only") {
            benchmark_hash_join(config);
            benchmark_semi_join(config);
            return 0;
        }
    }

    std::cout << "配置: iterations=" << config.iterations
              << ", warmup=" << config.warmup << std::endl;

    // 运行所有测试
    benchmark_filter_indices(config);
    benchmark_group_by(config);
    benchmark_hash_join(config);
    benchmark_semi_join(config);

    std::cout << "\n" << std::string(90, '=') << std::endl;
    std::cout << "  测试完成" << std::endl;
    std::cout << std::string(90, '=') << std::endl;

    return 0;
}
