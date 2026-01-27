/**
 * ThunderDuck V14 基准测试
 *
 * 测试 V14 优化效果:
 * - Hash Join: 两阶段预分配 (目标 8x+)
 * - GROUP BY: 寄存器缓冲 + 多路分流 (目标 4x+)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "thunderduck/join.h"
#include "thunderduck/aggregate.h"

using namespace thunderduck;
using namespace std::chrono;

// ============================================================================
// 基准测试工具
// ============================================================================

struct BenchmarkResult {
    double median_ms;
    double stddev_ms;
    double throughput_gbps;
};

template<typename Func>
BenchmarkResult run_benchmark(Func&& func, size_t data_bytes, int iterations = 10) {
    std::vector<double> times;
    times.reserve(iterations);

    // 预热
    func();

    // 多次测量
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        double ms = duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // IQR 剔除异常值
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    double q1 = times[n / 4];
    double q3 = times[n * 3 / 4];
    double iqr = q3 - q1;
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    std::vector<double> filtered;
    for (double t : times) {
        if (t >= lower && t <= upper) {
            filtered.push_back(t);
        }
    }

    // 中位数
    std::sort(filtered.begin(), filtered.end());
    double median = filtered[filtered.size() / 2];

    // 标准差
    double sum_sq = 0;
    for (double t : filtered) {
        sum_sq += (t - median) * (t - median);
    }
    double stddev = std::sqrt(sum_sq / filtered.size());

    // 吞吐量
    double throughput = (data_bytes / 1e9) / (median / 1000.0);

    return {median, stddev, throughput};
}

void print_result(const char* name, const BenchmarkResult& result) {
    std::cout << std::left << std::setw(30) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << result.median_ms << " ms (σ=" << std::setprecision(2) << result.stddev_ms << ")"
              << "  " << std::setprecision(2) << result.throughput_gbps << " GB/s"
              << std::endl;
}

void print_comparison(const char* name, const BenchmarkResult& baseline,
                      const BenchmarkResult& optimized) {
    double speedup = baseline.median_ms / optimized.median_ms;
    std::cout << std::left << std::setw(30) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << optimized.median_ms << " ms"
              << " vs " << std::setprecision(3) << baseline.median_ms << " ms"
              << "  " << std::setprecision(2) << speedup << "x speedup"
              << std::endl;
}

// ============================================================================
// Hash Join 基准测试
// ============================================================================

void benchmark_hash_join() {
    std::cout << "\n=== Hash Join 基准测试 ===" << std::endl;
    std::cout << "测试场景: 100K build × 1M probe (100% 匹配率)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    const size_t BUILD_COUNT = 100000;
    const size_t PROBE_COUNT = 1000000;

    // 生成测试数据: 100% 匹配率
    std::vector<int32_t> build_keys(BUILD_COUNT);
    std::vector<int32_t> probe_keys(PROBE_COUNT);

    std::mt19937 rng(42);
    for (size_t i = 0; i < BUILD_COUNT; ++i) {
        build_keys[i] = static_cast<int32_t>(i);
    }
    std::shuffle(build_keys.begin(), build_keys.end(), rng);

    for (size_t i = 0; i < PROBE_COUNT; ++i) {
        probe_keys[i] = static_cast<int32_t>(i % BUILD_COUNT);
    }
    std::shuffle(probe_keys.begin(), probe_keys.end(), rng);

    size_t data_bytes = (BUILD_COUNT + PROBE_COUNT) * sizeof(int32_t);

    // V3 基准
    join::JoinResult* result_v3 = join::create_join_result(PROBE_COUNT);
    auto bench_v3 = run_benchmark([&]() {
        join::hash_join_i32_v3(build_keys.data(), BUILD_COUNT,
                               probe_keys.data(), PROBE_COUNT,
                               join::JoinType::INNER, result_v3);
    }, data_bytes);
    std::cout << "V3 matches: " << result_v3->count << std::endl;

    // V10 当前最优
    join::JoinResult* result_v10 = join::create_join_result(PROBE_COUNT);
    auto bench_v10 = run_benchmark([&]() {
        join::hash_join_i32_v10(build_keys.data(), BUILD_COUNT,
                                probe_keys.data(), PROBE_COUNT,
                                join::JoinType::INNER, result_v10);
    }, data_bytes);

    // V14 优化版本
    join::JoinResult* result_v14 = join::create_join_result(PROBE_COUNT);
    auto bench_v14 = run_benchmark([&]() {
        join::hash_join_i32_v14(build_keys.data(), BUILD_COUNT,
                                probe_keys.data(), PROBE_COUNT,
                                join::JoinType::INNER, result_v14);
    }, data_bytes);
    std::cout << "V14 matches: " << result_v14->count << std::endl;

    std::cout << std::endl;
    print_result("V3 (Radix 16-partition)", bench_v3);
    print_result("V10 (Current Best)", bench_v10);
    print_result("V14 (Two-Phase 32-part)", bench_v14);

    std::cout << std::endl;
    std::cout << "加速比对比:" << std::endl;
    print_comparison("V14 vs V3", bench_v3, bench_v14);
    print_comparison("V14 vs V10", bench_v10, bench_v14);

    // 验证正确性
    bool correct = (result_v3->count == result_v14->count);
    std::cout << "\n正确性验证: " << (correct ? "PASS" : "FAIL") << std::endl;

    join::free_join_result(result_v3);
    join::free_join_result(result_v10);
    join::free_join_result(result_v14);
}

void benchmark_hash_join_various_sizes() {
    std::cout << "\n=== Hash Join 不同数据量测试 ===" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::vector<std::pair<size_t, size_t>> sizes = {
        {10000, 100000},    // 10K × 100K
        {50000, 500000},    // 50K × 500K
        {100000, 1000000},  // 100K × 1M
        {200000, 2000000},  // 200K × 2M
    };

    for (auto [build_count, probe_count] : sizes) {
        std::vector<int32_t> build_keys(build_count);
        std::vector<int32_t> probe_keys(probe_count);

        std::mt19937 rng(42);
        for (size_t i = 0; i < build_count; ++i) {
            build_keys[i] = static_cast<int32_t>(i);
        }
        for (size_t i = 0; i < probe_count; ++i) {
            probe_keys[i] = static_cast<int32_t>(i % build_count);
        }

        size_t data_bytes = (build_count + probe_count) * sizeof(int32_t);

        join::JoinResult* result_v10 = join::create_join_result(probe_count);
        join::JoinResult* result_v14 = join::create_join_result(probe_count);

        auto bench_v10 = run_benchmark([&]() {
            join::hash_join_i32_v10(build_keys.data(), build_count,
                                    probe_keys.data(), probe_count,
                                    join::JoinType::INNER, result_v10);
        }, data_bytes, 5);

        auto bench_v14 = run_benchmark([&]() {
            join::hash_join_i32_v14(build_keys.data(), build_count,
                                    probe_keys.data(), probe_count,
                                    join::JoinType::INNER, result_v14);
        }, data_bytes, 5);

        double speedup = bench_v10.median_ms / bench_v14.median_ms;

        std::cout << build_count / 1000 << "K × " << probe_count / 1000 << "K: "
                  << "V10=" << std::fixed << std::setprecision(2) << bench_v10.median_ms << "ms, "
                  << "V14=" << bench_v14.median_ms << "ms, "
                  << "Speedup=" << speedup << "x" << std::endl;

        join::free_join_result(result_v10);
        join::free_join_result(result_v14);
    }
}

// ============================================================================
// GROUP BY 基准测试
// ============================================================================

void benchmark_group_by() {
    std::cout << "\n=== GROUP BY 基准测试 ===" << std::endl;
    std::cout << "测试场景: 10M 数据, 1000 分组" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    const size_t COUNT = 10000000;
    const size_t NUM_GROUPS = 1000;

    std::vector<int32_t> values(COUNT);
    std::vector<uint32_t> groups(COUNT);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> val_dist(1, 1000);
    std::uniform_int_distribution<uint32_t> grp_dist(0, NUM_GROUPS - 1);

    for (size_t i = 0; i < COUNT; ++i) {
        values[i] = val_dist(rng);
        groups[i] = grp_dist(rng);
    }

    size_t data_bytes = COUNT * (sizeof(int32_t) + sizeof(uint32_t));

    // V4 单线程
    std::vector<int64_t> sums_v4(NUM_GROUPS);
    auto bench_v4_single = run_benchmark([&]() {
        aggregate::group_sum_i32_v4(values.data(), groups.data(), COUNT,
                                    NUM_GROUPS, sums_v4.data());
    }, data_bytes);

    // V4 并行 (当前最优)
    std::vector<int64_t> sums_v4_par(NUM_GROUPS);
    auto bench_v4_parallel = run_benchmark([&]() {
        aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), COUNT,
                                              NUM_GROUPS, sums_v4_par.data());
    }, data_bytes);

    // V14 优化版本
    std::vector<int64_t> sums_v14(NUM_GROUPS);
    auto bench_v14 = run_benchmark([&]() {
        aggregate::group_sum_i32_v14(values.data(), groups.data(), COUNT,
                                     NUM_GROUPS, sums_v14.data());
    }, data_bytes);

    std::cout << std::endl;
    print_result("V4 Single-thread", bench_v4_single);
    print_result("V4 Parallel (Current Best)", bench_v4_parallel);
    print_result("V14 RegBuf+Parallel", bench_v14);

    std::cout << std::endl;
    std::cout << "加速比对比:" << std::endl;
    print_comparison("V14 vs V4 Single", bench_v4_single, bench_v14);
    print_comparison("V14 vs V4 Parallel", bench_v4_parallel, bench_v14);

    // 验证正确性
    bool correct = true;
    for (size_t g = 0; g < NUM_GROUPS && correct; ++g) {
        if (sums_v4[g] != sums_v14[g]) {
            correct = false;
            std::cout << "Mismatch at group " << g << ": V4=" << sums_v4[g]
                      << ", V14=" << sums_v14[g] << std::endl;
        }
    }
    std::cout << "\n正确性验证: " << (correct ? "PASS" : "FAIL") << std::endl;
}

void benchmark_group_by_low_cardinality() {
    std::cout << "\n=== GROUP BY 低基数测试 (寄存器缓冲优化) ===" << std::endl;
    std::cout << "测试场景: 10M 数据, 8/16/32/64 分组" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    const size_t COUNT = 10000000;
    std::vector<size_t> group_counts = {8, 16, 32, 64};

    std::vector<int32_t> values(COUNT);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> val_dist(1, 1000);

    for (size_t i = 0; i < COUNT; ++i) {
        values[i] = val_dist(rng);
    }

    for (size_t num_groups : group_counts) {
        std::vector<uint32_t> groups(COUNT);
        std::uniform_int_distribution<uint32_t> grp_dist(0, num_groups - 1);
        for (size_t i = 0; i < COUNT; ++i) {
            groups[i] = grp_dist(rng);
        }

        size_t data_bytes = COUNT * (sizeof(int32_t) + sizeof(uint32_t));

        std::vector<int64_t> sums_v4(num_groups);
        std::vector<int64_t> sums_v14(num_groups);

        auto bench_v4 = run_benchmark([&]() {
            aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), COUNT,
                                                  num_groups, sums_v4.data());
        }, data_bytes, 5);

        auto bench_v14 = run_benchmark([&]() {
            aggregate::group_sum_i32_v14(values.data(), groups.data(), COUNT,
                                         num_groups, sums_v14.data());
        }, data_bytes, 5);

        double speedup = bench_v4.median_ms / bench_v14.median_ms;

        std::cout << num_groups << " groups: "
                  << "V4 Parallel=" << std::fixed << std::setprecision(2) << bench_v4.median_ms << "ms, "
                  << "V14=" << bench_v14.median_ms << "ms, "
                  << "Speedup=" << speedup << "x" << std::endl;
    }
}

void benchmark_group_by_various_sizes() {
    std::cout << "\n=== GROUP BY 不同数据量测试 ===" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::vector<size_t> sizes = {1000000, 5000000, 10000000, 20000000};
    const size_t NUM_GROUPS = 1000;

    for (size_t count : sizes) {
        std::vector<int32_t> values(count);
        std::vector<uint32_t> groups(count);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> val_dist(1, 1000);
        std::uniform_int_distribution<uint32_t> grp_dist(0, NUM_GROUPS - 1);

        for (size_t i = 0; i < count; ++i) {
            values[i] = val_dist(rng);
            groups[i] = grp_dist(rng);
        }

        size_t data_bytes = count * (sizeof(int32_t) + sizeof(uint32_t));

        std::vector<int64_t> sums_v4(NUM_GROUPS);
        std::vector<int64_t> sums_v14(NUM_GROUPS);

        auto bench_v4 = run_benchmark([&]() {
            aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), count,
                                                  NUM_GROUPS, sums_v4.data());
        }, data_bytes, 5);

        auto bench_v14 = run_benchmark([&]() {
            aggregate::group_sum_i32_v14(values.data(), groups.data(), count,
                                         NUM_GROUPS, sums_v14.data());
        }, data_bytes, 5);

        double speedup = bench_v4.median_ms / bench_v14.median_ms;

        std::cout << count / 1000000 << "M rows: "
                  << "V4 Parallel=" << std::fixed << std::setprecision(2) << bench_v4.median_ms << "ms, "
                  << "V14=" << bench_v14.median_ms << "ms, "
                  << "Speedup=" << speedup << "x" << std::endl;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "  ThunderDuck V14 性能基准测试" << std::endl;
    std::cout << "=============================================" << std::endl;

    std::cout << "\nV14 优化目标:" << std::endl;
    std::cout << "  - Hash Join: 4.28x → 8x+" << std::endl;
    std::cout << "  - GROUP BY: 2.66x → 4x+" << std::endl;

    // Hash Join 测试
    benchmark_hash_join();
    benchmark_hash_join_various_sizes();

    // GROUP BY 测试
    benchmark_group_by();
    benchmark_group_by_low_cardinality();
    benchmark_group_by_various_sizes();

    std::cout << "\n=============================================" << std::endl;
    std::cout << "  测试完成" << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}
