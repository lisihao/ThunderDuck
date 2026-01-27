/**
 * ThunderDuck V14 全面基准测试
 *
 * 测试所有算子版本与 DuckDB 的性能对比
 *
 * 算子覆盖:
 * - Filter: V3, V5, V6, V15, V4-GPU, Parallel
 * - GROUP BY SUM: V4, V6, V14, V15
 * - INNER JOIN: V3, V6, V10, V11, V13, V14
 * - SEMI JOIN: V10, GPU
 * - TopK: V3, V4, V5, V6, V13
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/sort.h"

#include "duckdb.hpp"

using namespace std::chrono;

// ============================================================================
// 测试配置
// ============================================================================

struct BenchConfig {
    size_t iterations = 15;        // 测试迭代次数
    size_t warmup = 2;             // 预热次数

    // Filter 测试数据量
    size_t filter_1m = 1000000;    // 1M
    size_t filter_10m = 10000000;  // 10M

    // GROUP BY 测试数据量
    size_t group_count = 10000000; // 10M 行
    size_t num_groups = 1000;      // 1000 分组

    // JOIN 测试数据量
    size_t join_build = 100000;    // 100K build
    size_t join_probe = 1000000;   // 1M probe

    // TopK 测试数据量
    size_t topk_count = 10000000;  // 10M
    size_t topk_k = 10;            // Top 10
};

// ============================================================================
// 测试结果结构
// ============================================================================

struct TestResult {
    std::string operator_name;     // 算子名称
    std::string version;           // 版本名称
    std::string device;            // 设备 (CPU SIMD/GPU/NPU)
    std::string sql;               // 等效 SQL
    size_t data_size;              // 数据量
    double time_ms;                // 执行时间 (ms)
    double bandwidth_gbs;          // 带宽 (GB/s)
    double vs_duckdb;              // vs DuckDB 加速比
    double vs_v3;                  // vs V3 加速比
    bool correct;                  // 正确性
};

std::vector<TestResult> all_results;

// ============================================================================
// 统计工具函数
// ============================================================================

template<typename Func>
double measure_median_iqr(Func&& func, size_t iterations, size_t warmup) {
    std::vector<double> times;
    times.reserve(iterations);

    // 预热
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }

    // 正式测试
    for (size_t i = 0; i < iterations; ++i) {
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
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    times.erase(std::remove_if(times.begin(), times.end(),
        [&](double t) { return t < lower || t > upper; }), times.end());

    // 返回中位数
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(90, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

void print_header() {
    std::cout << std::left
              << std::setw(20) << "版本"
              << std::setw(12) << "设备"
              << std::setw(14) << "时间(ms)"
              << std::setw(14) << "带宽(GB/s)"
              << std::setw(12) << "vs DuckDB"
              << std::setw(12) << "vs V3"
              << std::setw(8) << "正确性"
              << std::endl;
    std::cout << std::string(90, '-') << std::endl;
}

void print_result(const TestResult& r) {
    std::cout << std::left
              << std::setw(20) << r.version
              << std::setw(12) << r.device
              << std::fixed << std::setprecision(3)
              << std::setw(14) << r.time_ms
              << std::setw(14) << r.bandwidth_gbs
              << std::setw(12) << (std::to_string(int(r.vs_duckdb * 100) / 100.0).substr(0, 4) + "x")
              << std::setw(12) << (r.vs_v3 > 0 ? std::to_string(int(r.vs_v3 * 100) / 100.0).substr(0, 4) + "x" : "-")
              << std::setw(8) << (r.correct ? "PASS" : "FAIL")
              << std::endl;
}

// ============================================================================
// Filter 算子测试
// ============================================================================

void test_filter(const BenchConfig& config) {
    using namespace thunderduck::filter;

    print_section("FILTER 算子测试");
    std::cout << "等效 SQL: SELECT * FROM t WHERE value > threshold" << std::endl;

    for (size_t data_size : {config.filter_1m, config.filter_10m}) {
        std::cout << "\n--- 数据量: " << data_size / 1000000 << "M ---" << std::endl;
        print_header();

        // 准备数据
        std::vector<int32_t> data(data_size);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000000);
        for (auto& v : data) v = dist(rng);

        int32_t threshold = 500000;  // 约 50% 选择率
        std::vector<uint32_t> indices(data_size);
        size_t data_bytes = data_size * sizeof(int32_t);

        double duckdb_time = 0, v3_time = 0;
        size_t expected_count = 0;

        // DuckDB 基准
        {
            duckdb::DuckDB db(nullptr);
            duckdb::Connection con(db);
            con.Query("CREATE TABLE t (value INTEGER)");
            {
                duckdb::Appender appender(con, "t");
                for (auto v : data) appender.AppendRow(v);
            }

            // 先获取预期计数
            {
                auto result = con.Query("SELECT COUNT(*) FROM t WHERE value > 500000");
                expected_count = result->GetValue(0, 0).GetValue<int64_t>();
            }

            // 测试返回索引 (与 ThunderDuck 对等比较)
            duckdb_time = measure_median_iqr([&]() {
                con.Query("SELECT rowid FROM t WHERE value > 500000");
            }, config.iterations, config.warmup);

            TestResult r{"Filter", "DuckDB", "CPU",
                "SELECT * FROM t WHERE value > 500000",
                data_size, duckdb_time, data_bytes / (duckdb_time * 1e6),
                1.0, 0, true};
            print_result(r);
            all_results.push_back(r);
        }

        // V3 (bitmap)
        {
            v3_time = measure_median_iqr([&]() {
                filter_i32_v3(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_v3(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            TestResult r{"Filter", "V3 (bitmap)", "CPU SIMD",
                "SELECT * FROM t WHERE value > 500000",
                data_size, v3_time, data_bytes / (v3_time * 1e6),
                duckdb_time / v3_time, 1.0, correct};
            print_result(r);
            all_results.push_back(r);
        }

        // V5 (LUT)
        {
            double time = measure_median_iqr([&]() {
                filter_i32_v5(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_v5(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            TestResult r{"Filter", "V5 (LUT)", "CPU SIMD",
                "SELECT * FROM t WHERE value > 500000",
                data_size, time, data_bytes / (time * 1e6),
                duckdb_time / time, v3_time / time, correct};
            print_result(r);
            all_results.push_back(r);
        }

        // V6 (prefetch)
        {
            double time = measure_median_iqr([&]() {
                filter_i32_v6(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_v6(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            TestResult r{"Filter", "V6 (prefetch)", "CPU SIMD",
                "SELECT * FROM t WHERE value > 500000",
                data_size, time, data_bytes / (time * 1e6),
                duckdb_time / time, v3_time / time, correct};
            print_result(r);
            all_results.push_back(r);
        }

        // V15 (direct)
        {
            double time = measure_median_iqr([&]() {
                filter_i32_v15(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_v15(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            TestResult r{"Filter", "V15 (direct)", "CPU SIMD",
                "SELECT * FROM t WHERE value > 500000",
                data_size, time, data_bytes / (time * 1e6),
                duckdb_time / time, v3_time / time, correct};
            print_result(r);
            all_results.push_back(r);
        }

        // V4 GPU (AUTO)
        if (is_filter_gpu_available()) {
            double time = measure_median_iqr([&]() {
                filter_i32_v4(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_v4(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            std::string device = (data_size >= 5000000) ? "GPU Auto" : "CPU Auto";
            TestResult r{"Filter", "V4 (AUTO)", device,
                "SELECT * FROM t WHERE value > 500000",
                data_size, time, data_bytes / (time * 1e6),
                duckdb_time / time, v3_time / time, correct};
            print_result(r);
            all_results.push_back(r);
        }

        // Parallel
        if (data_size >= 1000000) {
            double time = measure_median_iqr([&]() {
                filter_i32_parallel(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            }, config.iterations, config.warmup);

            size_t count = filter_i32_parallel(data.data(), data_size, CompareOp::GT, threshold, indices.data());
            bool correct = (count == expected_count);

            TestResult r{"Filter", "Parallel (4T)", "CPU 4T",
                "SELECT * FROM t WHERE value > 500000",
                data_size, time, data_bytes / (time * 1e6),
                duckdb_time / time, v3_time / time, correct};
            print_result(r);
            all_results.push_back(r);
        }
    }
}

// ============================================================================
// GROUP BY SUM 算子测试
// ============================================================================

void test_group_by(const BenchConfig& config) {
    using namespace thunderduck::aggregate;

    print_section("GROUP BY SUM 算子测试");
    std::cout << "等效 SQL: SELECT group_id, SUM(value) FROM t GROUP BY group_id" << std::endl;
    std::cout << "数据量: " << config.group_count / 1000000 << "M 行, "
              << config.num_groups << " 分组" << std::endl;
    print_header();

    // 准备数据
    std::vector<int32_t> values(config.group_count);
    std::vector<uint32_t> groups(config.group_count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> val_dist(1, 1000);
    std::uniform_int_distribution<uint32_t> grp_dist(0, config.num_groups - 1);

    for (size_t i = 0; i < config.group_count; ++i) {
        values[i] = val_dist(rng);
        groups[i] = grp_dist(rng);
    }

    std::vector<int64_t> out_sums(config.num_groups);
    size_t data_bytes = config.group_count * (sizeof(int32_t) + sizeof(uint32_t));

    double duckdb_time = 0, v4_time = 0;
    std::vector<int64_t> expected_sums(config.num_groups);

    // DuckDB 基准
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);
        con.Query("CREATE TABLE t (group_id INTEGER, value INTEGER)");
        {
            duckdb::Appender appender(con, "t");
            for (size_t i = 0; i < config.group_count; ++i) {
                appender.AppendRow((int32_t)groups[i], values[i]);
            }
        }

        duckdb_time = measure_median_iqr([&]() {
            con.Query("SELECT group_id, SUM(value) FROM t GROUP BY group_id");
        }, config.iterations, config.warmup);

        // 获取预期结果
        auto result = con.Query("SELECT group_id, SUM(value) FROM t GROUP BY group_id ORDER BY group_id");
        for (size_t i = 0; i < result->RowCount(); ++i) {
            int32_t gid = result->GetValue(0, i).GetValue<int32_t>();
            int64_t sum = result->GetValue(1, i).GetValue<int64_t>();
            expected_sums[gid] = sum;
        }

        TestResult r{"GROUP BY SUM", "DuckDB", "CPU",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, duckdb_time, data_bytes / (duckdb_time * 1e6),
            1.0, 0, true};
        print_result(r);
        all_results.push_back(r);
    }

    auto verify_sums = [&](const int64_t* sums) -> bool {
        for (size_t i = 0; i < config.num_groups; ++i) {
            if (sums[i] != expected_sums[i]) return false;
        }
        return true;
    };

    // V4 (single)
    {
        std::fill(out_sums.begin(), out_sums.end(), 0);
        v4_time = measure_median_iqr([&]() {
            std::fill(out_sums.begin(), out_sums.end(), 0);
            group_sum_i32_v4(values.data(), groups.data(), config.group_count,
                            config.num_groups, out_sums.data());
        }, config.iterations, config.warmup);

        std::fill(out_sums.begin(), out_sums.end(), 0);
        group_sum_i32_v4(values.data(), groups.data(), config.group_count,
                        config.num_groups, out_sums.data());
        bool correct = verify_sums(out_sums.data());

        TestResult r{"GROUP BY SUM", "V4 (single)", "CPU SIMD",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, v4_time, data_bytes / (v4_time * 1e6),
            duckdb_time / v4_time, 1.0, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V4 Parallel
    {
        std::fill(out_sums.begin(), out_sums.end(), 0);
        double time = measure_median_iqr([&]() {
            std::fill(out_sums.begin(), out_sums.end(), 0);
            group_sum_i32_v4_parallel(values.data(), groups.data(), config.group_count,
                                     config.num_groups, out_sums.data());
        }, config.iterations, config.warmup);

        std::fill(out_sums.begin(), out_sums.end(), 0);
        group_sum_i32_v4_parallel(values.data(), groups.data(), config.group_count,
                                 config.num_groups, out_sums.data());
        bool correct = verify_sums(out_sums.data());

        TestResult r{"GROUP BY SUM", "V4 (parallel)", "CPU 4T",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v4_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V6 (smart)
    {
        std::fill(out_sums.begin(), out_sums.end(), 0);
        double time = measure_median_iqr([&]() {
            std::fill(out_sums.begin(), out_sums.end(), 0);
            group_sum_i32_v6(values.data(), groups.data(), config.group_count,
                            config.num_groups, out_sums.data());
        }, config.iterations, config.warmup);

        std::fill(out_sums.begin(), out_sums.end(), 0);
        group_sum_i32_v6(values.data(), groups.data(), config.group_count,
                        config.num_groups, out_sums.data());
        bool correct = verify_sums(out_sums.data());

        TestResult r{"GROUP BY SUM", "V6 (smart)", "CPU/GPU Auto",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v4_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V14 (parallel)
    {
        std::fill(out_sums.begin(), out_sums.end(), 0);
        double time = measure_median_iqr([&]() {
            std::fill(out_sums.begin(), out_sums.end(), 0);
            group_sum_i32_v14_parallel(values.data(), groups.data(), config.group_count,
                                       config.num_groups, out_sums.data());
        }, config.iterations, config.warmup);

        std::fill(out_sums.begin(), out_sums.end(), 0);
        group_sum_i32_v14_parallel(values.data(), groups.data(), config.group_count,
                                   config.num_groups, out_sums.data());
        bool correct = verify_sums(out_sums.data());

        TestResult r{"GROUP BY SUM", "V14 (parallel)", "CPU 8T",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v4_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V15
    {
        std::fill(out_sums.begin(), out_sums.end(), 0);
        double time = measure_median_iqr([&]() {
            std::fill(out_sums.begin(), out_sums.end(), 0);
            group_sum_i32_v15(values.data(), groups.data(), config.group_count,
                             config.num_groups, out_sums.data());
        }, config.iterations, config.warmup);

        std::fill(out_sums.begin(), out_sums.end(), 0);
        group_sum_i32_v15(values.data(), groups.data(), config.group_count,
                         config.num_groups, out_sums.data());
        bool correct = verify_sums(out_sums.data());

        TestResult r{"GROUP BY SUM", "V15 (8T+展开)", "CPU 8T",
            "SELECT group_id, SUM(value) FROM t GROUP BY group_id",
            config.group_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v4_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }
}

// ============================================================================
// HASH JOIN (INNER) 算子测试
// ============================================================================

void test_hash_join(const BenchConfig& config) {
    using namespace thunderduck::join;

    print_section("HASH JOIN (INNER) 算子测试");
    std::cout << "等效 SQL: SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key" << std::endl;
    std::cout << "Build: " << config.join_build / 1000 << "K, Probe: "
              << config.join_probe / 1000000 << "M" << std::endl;
    print_header();

    // 准备数据
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

    double duckdb_time = 0, v3_time = 0;
    size_t expected_count = 0;

    // DuckDB 基准
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
            expected_count = result->GetValue(0, 0).GetValue<int64_t>();
        }, config.iterations, config.warmup);

        TestResult r{"INNER JOIN", "DuckDB", "CPU",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, duckdb_time,
            data_bytes / (duckdb_time * 1e6), 1.0, 0, true};
        print_result(r);
        all_results.push_back(r);
    }

    // V3
    {
        JoinResult* result = create_join_result(config.join_probe);

        v3_time = measure_median_iqr([&]() {
            hash_join_i32_v3(build_keys.data(), config.join_build,
                            probe_keys.data(), config.join_probe,
                            JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V3", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, v3_time,
            data_bytes / (v3_time * 1e6), duckdb_time / v3_time, 1.0, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // V6 (prefetch)
    {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            hash_join_i32_v6(build_keys.data(), config.join_build,
                            probe_keys.data(), config.join_probe,
                            JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V6 (prefetch)", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // V10
    {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            hash_join_i32_v10(build_keys.data(), config.join_build,
                             probe_keys.data(), config.join_probe,
                             JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V10", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // V11
    {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            hash_join_i32_v11(build_keys.data(), config.join_build,
                             probe_keys.data(), config.join_probe,
                             JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V11 (SIMD probe)", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // V13
    {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            hash_join_i32_v13(build_keys.data(), config.join_build,
                             probe_keys.data(), config.join_probe,
                             JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V13 (两阶段)", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // V14
    {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            hash_join_i32_v14(build_keys.data(), config.join_build,
                             probe_keys.data(), config.join_probe,
                             JoinType::INNER, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"INNER JOIN", "V14 (预分配)", "CPU SIMD",
            "SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }
}

// ============================================================================
// SEMI JOIN 算子测试
// ============================================================================

void test_semi_join(const BenchConfig& config) {
    using namespace thunderduck::join;

    print_section("SEMI JOIN 算子测试");
    std::cout << "等效 SQL: SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE build_t.key = probe_t.key)" << std::endl;
    std::cout << "Build: " << config.join_build / 1000 << "K, Probe: "
              << config.join_probe / 1000000 << "M" << std::endl;
    print_header();

    // 准备数据
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

    double duckdb_time = 0, v10_time = 0;
    size_t expected_count = 0;

    // DuckDB 基准
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
            expected_count = result->GetValue(0, 0).GetValue<int64_t>();
        }, config.iterations, config.warmup);

        TestResult r{"SEMI JOIN", "DuckDB", "CPU",
            "SELECT * FROM probe_t WHERE EXISTS (...)",
            config.join_build + config.join_probe, duckdb_time,
            data_bytes / (duckdb_time * 1e6), 1.0, 0, true};
        print_result(r);
        all_results.push_back(r);
    }

    // V10
    {
        JoinResult* result = create_join_result(config.join_probe);

        v10_time = measure_median_iqr([&]() {
            hash_join_i32_v10(build_keys.data(), config.join_build,
                             probe_keys.data(), config.join_probe,
                             JoinType::SEMI, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"SEMI JOIN", "V10", "CPU SIMD",
            "SELECT * FROM probe_t WHERE EXISTS (...)",
            config.join_build + config.join_probe, v10_time,
            data_bytes / (v10_time * 1e6), duckdb_time / v10_time, 1.0, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }

    // GPU SEMI Join
    if (is_semi_join_gpu_available()) {
        JoinResult* result = create_join_result(config.join_probe);

        double time = measure_median_iqr([&]() {
            semi_join_gpu(build_keys.data(), config.join_build,
                         probe_keys.data(), config.join_probe, result);
        }, config.iterations, config.warmup);

        bool correct = (result->count == expected_count);

        TestResult r{"SEMI JOIN", "GPU (Metal)", "Metal",
            "SELECT * FROM probe_t WHERE EXISTS (...)",
            config.join_build + config.join_probe, time,
            data_bytes / (time * 1e6), duckdb_time / time, v10_time / time, correct};
        print_result(r);
        all_results.push_back(r);

        free_join_result(result);
    }
}

// ============================================================================
// TopK 算子测试
// ============================================================================

void test_topk(const BenchConfig& config) {
    using namespace thunderduck::sort;

    print_section("TopK 算子测试");
    std::cout << "等效 SQL: SELECT * FROM t ORDER BY value DESC LIMIT " << config.topk_k << std::endl;
    std::cout << "数据量: " << config.topk_count / 1000000 << "M, K=" << config.topk_k << std::endl;
    print_header();

    // 准备数据
    std::vector<int32_t> data(config.topk_count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000000);
    for (auto& v : data) v = dist(rng);

    std::vector<int32_t> out_values(config.topk_k);
    std::vector<uint32_t> out_indices(config.topk_k);
    size_t data_bytes = config.topk_count * sizeof(int32_t);

    double duckdb_time = 0, v3_time = 0;
    std::vector<int32_t> expected_values(config.topk_k);

    // DuckDB 基准
    {
        duckdb::DuckDB db(nullptr);
        duckdb::Connection con(db);
        con.Query("CREATE TABLE t (value INTEGER)");
        {
            duckdb::Appender appender(con, "t");
            for (auto v : data) appender.AppendRow(v);
        }

        duckdb_time = measure_median_iqr([&]() {
            con.Query("SELECT value FROM t ORDER BY value DESC LIMIT 10");
        }, config.iterations, config.warmup);

        auto result = con.Query("SELECT value FROM t ORDER BY value DESC LIMIT 10");
        for (size_t i = 0; i < config.topk_k; ++i) {
            expected_values[i] = result->GetValue(0, i).GetValue<int32_t>();
        }

        TestResult r{"TopK", "DuckDB", "CPU",
            "SELECT * FROM t ORDER BY value DESC LIMIT 10",
            config.topk_count, duckdb_time, data_bytes / (duckdb_time * 1e6),
            1.0, 0, true};
        print_result(r);
        all_results.push_back(r);
    }

    auto verify_topk = [&](const int32_t* values) -> bool {
        std::vector<int32_t> sorted_values(values, values + config.topk_k);
        std::sort(sorted_values.begin(), sorted_values.end(), std::greater<int32_t>());
        return sorted_values == expected_values;
    };

    // V3
    {
        v3_time = measure_median_iqr([&]() {
            topk_max_i32_v3(data.data(), config.topk_count, config.topk_k,
                           out_values.data(), out_indices.data());
        }, config.iterations, config.warmup);

        topk_max_i32_v3(data.data(), config.topk_count, config.topk_k,
                       out_values.data(), out_indices.data());
        bool correct = verify_topk(out_values.data());

        TestResult r{"TopK", "V3 (adaptive)", "CPU",
            "SELECT * FROM t ORDER BY value DESC LIMIT 10",
            config.topk_count, v3_time, data_bytes / (v3_time * 1e6),
            duckdb_time / v3_time, 1.0, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V4 (sample)
    {
        double time = measure_median_iqr([&]() {
            topk_max_i32_v4(data.data(), config.topk_count, config.topk_k,
                           out_values.data(), out_indices.data());
        }, config.iterations, config.warmup);

        topk_max_i32_v4(data.data(), config.topk_count, config.topk_k,
                       out_values.data(), out_indices.data());
        bool correct = verify_topk(out_values.data());

        TestResult r{"TopK", "V4 (sample)", "CPU SIMD",
            "SELECT * FROM t ORDER BY value DESC LIMIT 10",
            config.topk_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V5 (count-based)
    {
        double time = measure_median_iqr([&]() {
            topk_max_i32_v5(data.data(), config.topk_count, config.topk_k,
                           out_values.data(), out_indices.data());
        }, config.iterations, config.warmup);

        topk_max_i32_v5(data.data(), config.topk_count, config.topk_k,
                       out_values.data(), out_indices.data());
        bool correct = verify_topk(out_values.data());

        TestResult r{"TopK", "V5 (count)", "CPU",
            "SELECT * FROM t ORDER BY value DESC LIMIT 10",
            config.topk_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }

    // V6 (UMA)
    if (is_topk_gpu_available()) {
        double time = measure_median_iqr([&]() {
            topk_max_i32_v6(data.data(), config.topk_count, config.topk_k,
                           out_values.data(), out_indices.data());
        }, config.iterations, config.warmup);

        topk_max_i32_v6(data.data(), config.topk_count, config.topk_k,
                       out_values.data(), out_indices.data());
        bool correct = verify_topk(out_values.data());

        TestResult r{"TopK", "V6 (UMA)", "CPU/GPU Auto",
            "SELECT * FROM t ORDER BY value DESC LIMIT 10",
            config.topk_count, time, data_bytes / (time * 1e6),
            duckdb_time / time, v3_time / time, correct};
        print_result(r);
        all_results.push_back(r);
    }
}

// ============================================================================
// 生成报告
// ============================================================================

void generate_report(const BenchConfig& config) {
    std::ofstream report("/Users/sihaoli/ThunderDuck/docs/V14_BENCHMARK_REPORT.md");

    report << "# ThunderDuck V14 全面性能基准报告\n\n";
    report << "> **测试日期**: " << __DATE__ << " " << __TIME__ << "\n";
    report << "> **平台**: Apple M4 Max\n";
    report << "> **测试配置**: iterations=" << config.iterations << ", warmup=" << config.warmup << "\n\n";

    report << "## 一、执行摘要\n\n";
    report << "本报告测试了 ThunderDuck 各算子版本与 DuckDB 的性能对比。\n\n";

    // 按算子分组统计最佳性能
    std::map<std::string, TestResult> best_results;
    for (const auto& r : all_results) {
        if (r.version == "DuckDB") continue;
        std::string key = r.operator_name + "_" + std::to_string(r.data_size);
        if (best_results.find(key) == best_results.end() ||
            r.vs_duckdb > best_results[key].vs_duckdb) {
            best_results[key] = r;
        }
    }

    report << "### 最佳性能摘要\n\n";
    report << "| 算子 | 数据量 | 最佳版本 | 设备 | vs DuckDB |\n";
    report << "|------|--------|----------|------|-----------|\n";
    for (const auto& [key, r] : best_results) {
        std::string size_str = (r.data_size >= 1000000) ?
            std::to_string(r.data_size / 1000000) + "M" :
            std::to_string(r.data_size / 1000) + "K";
        report << "| " << r.operator_name << " | " << size_str << " | "
               << r.version << " | " << r.device << " | "
               << std::fixed << std::setprecision(2) << r.vs_duckdb << "x |\n";
    }

    report << "\n## 二、详细测试结果\n\n";

    // Filter 结果
    report << "### 2.1 Filter 算子\n\n";
    report << "**等效 SQL**: `SELECT * FROM t WHERE value > 500000`\n\n";
    report << "| 版本 | 设备 | 数据量 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |\n";
    report << "|------|------|--------|----------|------------|-----------|-------|--------|\n";
    for (const auto& r : all_results) {
        if (r.operator_name != "Filter") continue;
        std::string size_str = (r.data_size >= 1000000) ?
            std::to_string(r.data_size / 1000000) + "M" :
            std::to_string(r.data_size / 1000) + "K";
        report << "| " << r.version << " | " << r.device << " | " << size_str << " | "
               << std::fixed << std::setprecision(3) << r.time_ms << " | "
               << std::setprecision(2) << r.bandwidth_gbs << " | "
               << r.vs_duckdb << "x | "
               << (r.vs_v3 > 0 ? std::to_string(r.vs_v3).substr(0, 4) + "x" : "-") << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    // GROUP BY 结果
    report << "\n### 2.2 GROUP BY SUM 算子\n\n";
    report << "**等效 SQL**: `SELECT group_id, SUM(value) FROM t GROUP BY group_id`\n\n";
    report << "| 版本 | 设备 | 数据量 | 分组数 | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V4 | 正确性 |\n";
    report << "|------|------|--------|--------|----------|------------|-----------|-------|--------|\n";
    for (const auto& r : all_results) {
        if (r.operator_name != "GROUP BY SUM") continue;
        report << "| " << r.version << " | " << r.device << " | "
               << r.data_size / 1000000 << "M | " << config.num_groups << " | "
               << std::fixed << std::setprecision(3) << r.time_ms << " | "
               << std::setprecision(2) << r.bandwidth_gbs << " | "
               << r.vs_duckdb << "x | "
               << (r.vs_v3 > 0 ? std::to_string(r.vs_v3).substr(0, 4) + "x" : "-") << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    // JOIN 结果
    report << "\n### 2.3 INNER JOIN 算子\n\n";
    report << "**等效 SQL**: `SELECT * FROM build_t JOIN probe_t ON build_t.key = probe_t.key`\n\n";
    report << "| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |\n";
    report << "|------|------|-------|-------|----------|------------|-----------|-------|--------|\n";
    for (const auto& r : all_results) {
        if (r.operator_name != "INNER JOIN") continue;
        report << "| " << r.version << " | " << r.device << " | "
               << config.join_build / 1000 << "K | " << config.join_probe / 1000000 << "M | "
               << std::fixed << std::setprecision(3) << r.time_ms << " | "
               << std::setprecision(2) << r.bandwidth_gbs << " | "
               << r.vs_duckdb << "x | "
               << (r.vs_v3 > 0 ? std::to_string(r.vs_v3).substr(0, 4) + "x" : "-") << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    // SEMI JOIN 结果
    report << "\n### 2.4 SEMI JOIN 算子\n\n";
    report << "**等效 SQL**: `SELECT * FROM probe_t WHERE EXISTS (SELECT 1 FROM build_t WHERE ...)`\n\n";
    report << "| 版本 | 设备 | Build | Probe | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V10 | 正确性 |\n";
    report << "|------|------|-------|-------|----------|------------|-----------|--------|--------|\n";
    for (const auto& r : all_results) {
        if (r.operator_name != "SEMI JOIN") continue;
        report << "| " << r.version << " | " << r.device << " | "
               << config.join_build / 1000 << "K | " << config.join_probe / 1000000 << "M | "
               << std::fixed << std::setprecision(3) << r.time_ms << " | "
               << std::setprecision(2) << r.bandwidth_gbs << " | "
               << r.vs_duckdb << "x | "
               << (r.vs_v3 > 0 ? std::to_string(r.vs_v3).substr(0, 4) + "x" : "-") << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    // TopK 结果
    report << "\n### 2.5 TopK 算子\n\n";
    report << "**等效 SQL**: `SELECT * FROM t ORDER BY value DESC LIMIT 10`\n\n";
    report << "| 版本 | 设备 | 数据量 | K | 时间(ms) | 带宽(GB/s) | vs DuckDB | vs V3 | 正确性 |\n";
    report << "|------|------|--------|---|----------|------------|-----------|-------|--------|\n";
    for (const auto& r : all_results) {
        if (r.operator_name != "TopK") continue;
        report << "| " << r.version << " | " << r.device << " | "
               << r.data_size / 1000000 << "M | " << config.topk_k << " | "
               << std::fixed << std::setprecision(3) << r.time_ms << " | "
               << std::setprecision(2) << r.bandwidth_gbs << " | "
               << r.vs_duckdb << "x | "
               << (r.vs_v3 > 0 ? std::to_string(r.vs_v3).substr(0, 4) + "x" : "-") << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    report << "\n## 三、优化建议\n\n";
    report << "### 潜在优化点\n\n";

    // 找出 vs DuckDB < 1.0 的算子
    report << "#### 需要优化 (vs DuckDB < 1.0x):\n\n";
    for (const auto& r : all_results) {
        if (r.version == "DuckDB") continue;
        if (r.vs_duckdb < 1.0 && r.correct) {
            report << "- **" << r.operator_name << " " << r.version << "**: "
                   << std::fixed << std::setprecision(2) << r.vs_duckdb << "x\n";
        }
    }

    report << "\n#### 表现优异 (vs DuckDB >= 2.0x):\n\n";
    for (const auto& r : all_results) {
        if (r.version == "DuckDB") continue;
        if (r.vs_duckdb >= 2.0 && r.correct) {
            report << "- **" << r.operator_name << " " << r.version << "**: "
                   << std::fixed << std::setprecision(2) << r.vs_duckdb << "x\n";
        }
    }

    report << "\n## 四、版本历史对比\n\n";
    report << "| 版本 | 主要优化 | 关键算子性能 |\n";
    report << "|------|----------|-------------|\n";
    report << "| V3 | 基础 SIMD | 基准版本 |\n";
    report << "| V6 | 预取优化 | Join 1.3x |\n";
    report << "| V10 | 完整语义 | SEMI/ANTI Join |\n";
    report << "| V11 | SIMD 探测 | - |\n";
    report << "| V13 | 两阶段算法 | Join 1.2x |\n";
    report << "| V14 | GPU 加速 | SEMI Join 2.2x, Filter 1.5x |\n";

    report.close();
    std::cout << "\n报告已生成: docs/V14_BENCHMARK_REPORT.md\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================================\n";
    std::cout << "        ThunderDuck V14 全面基准测试\n";
    std::cout << "============================================================\n";
    std::cout << "平台: Apple M4 Max\n";
    std::cout << "日期: " << __DATE__ << " " << __TIME__ << "\n\n";

    BenchConfig config;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            config.iterations = 5;
            config.warmup = 1;
            config.filter_10m = 1000000;
            config.group_count = 1000000;
            config.topk_count = 1000000;
        }
    }

    std::cout << "配置: iterations=" << config.iterations
              << ", warmup=" << config.warmup << "\n";

    // 运行所有测试
    test_filter(config);
    test_group_by(config);
    test_hash_join(config);
    test_semi_join(config);
    test_topk(config);

    // 生成报告
    generate_report(config);

    print_section("测试完成");

    return 0;
}
