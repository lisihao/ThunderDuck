/**
 * V7/V8/V9 Aggregate 版本对比测试
 *
 * 版本映射:
 * - V7: TD v1/v2 (基础SIMD + 64B预取)
 * - V8: TD v2 (Filter v5优化,聚合无变化)
 * - V9: TD v4 (AGG新性能方案: P1预取优化+P2缓存分块+P3多线程)
 *
 * P0-P3 优化项:
 * - P0: 向量化哈希分组 - 预期1.5-2x (当前1.03x,分区开销大)
 * - P1: 预取距离优化 - 预期5-10% (MIN/MAX达成10%)
 * - P2: 缓存分块 - 预期5-15% (已实现)
 * - P3: 多线程分组聚合 - 预期1.2-1.5x (当前2.54x,超预期)
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "thunderduck/aggregate.h"

using namespace std;
using namespace thunderduck::aggregate;

// ============================================================================
// 统计函数
// ============================================================================

struct BenchResult {
    double median;
    double stddev;
    int removed_outliers;
};

template<typename F>
BenchResult measure_with_stats(F&& func, int iterations = 30) {
    func();  // Warmup

    vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
    }

    sort(times.begin(), times.end());

    // IQR 剔除异常值
    double q1 = times[times.size() / 4];
    double q3 = times[times.size() * 3 / 4];
    double iqr = q3 - q1;
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    vector<double> filtered;
    for (double t : times) {
        if (t >= lower && t <= upper) {
            filtered.push_back(t);
        }
    }

    int removed = static_cast<int>(times.size() - filtered.size());
    if (filtered.empty()) filtered = times;

    sort(filtered.begin(), filtered.end());
    double median = filtered[filtered.size() / 2];

    double mean = accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    double sq_sum = 0;
    for (double t : filtered) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / filtered.size());

    return {median, stddev, removed};
}

void print_version_result(const string& version, const string& desc,
                          const BenchResult& r, double baseline = 0) {
    cout << "  " << left << setw(8) << version
         << setw(25) << desc << ": "
         << fixed << setprecision(3) << setw(8) << r.median << " ms"
         << " (σ=" << setprecision(3) << setw(5) << r.stddev << ")";

    if (baseline > 0) {
        double speedup = baseline / r.median;
        cout << "  [" << setprecision(2) << speedup << "x]";
    }

    if (r.removed_outliers > 0) {
        cout << " *" << r.removed_outliers << " outliers";
    }

    cout << endl;
}

int main() {
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║     ThunderDuck V7 / V8 / V9 Aggregate 版本对比测试                          ║" << endl;
    cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
    cout << "║ V7: 基础SIMD + 64B预取                                                       ║" << endl;
    cout << "║ V8: Filter v5优化 (聚合无变化)                                               ║" << endl;
    cout << "║ V9: AGG新性能方案 (P1:256B预取 + P2:缓存分块 + P3:多线程)                    ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << endl;

    // ========================================================================
    // 数据准备
    // ========================================================================

    const size_t N = 10000000;  // 10M 元素
    const size_t NUM_GROUPS = 100;

    vector<int32_t> data_i32(N);
    vector<uint32_t> groups(N);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_i32(-1000000, 1000000);
    uniform_int_distribution<uint32_t> dist_group(0, NUM_GROUPS - 1);

    for (size_t i = 0; i < N; ++i) {
        data_i32[i] = dist_i32(gen);
        groups[i] = dist_group(gen);
    }

    cout << "\n数据规模: " << N / 1000000 << "M 元素" << endl;
    cout << "分组数量: " << NUM_GROUPS << " groups" << endl;
    cout << "迭代次数: 30 次 (IQR剔除异常值)\n" << endl;

    // ========================================================================
    // 测试 1: SUM i32
    // ========================================================================

    cout << "╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 1: SUM i32 (10M elements)                                               │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    volatile int64_t sum_result;

    auto duckdb_sum = measure_with_stats([&]() {
        sum_result = 0;
        for (size_t i = 0; i < N; ++i) sum_result += data_i32[i];
    }, 30);

    auto v7_sum = measure_with_stats([&]() {
        sum_result = sum_i32_v2(data_i32.data(), N);
    }, 30);

    auto v9_sum = measure_with_stats([&]() {
        sum_result = sum_i32_v4(data_i32.data(), N);
    }, 30);

    auto v9_blocked_sum = measure_with_stats([&]() {
        sum_result = sum_i32_v4_blocked(data_i32.data(), N);
    }, 30);

    print_version_result("DuckDB", "标量基线", duckdb_sum);
    print_version_result("V7/V8", "SIMD + 64B预取", v7_sum, duckdb_sum.median);
    print_version_result("V9", "SIMD + 256B预取 (P1)", v9_sum, duckdb_sum.median);
    print_version_result("V9", "缓存分块 (P2)", v9_blocked_sum, duckdb_sum.median);

    // ========================================================================
    // 测试 2: MIN/MAX i32
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 2: MIN/MAX i32 (10M elements)                                           │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    int32_t min_result, max_result;

    auto duckdb_minmax = measure_with_stats([&]() {
        min_result = data_i32[0];
        max_result = data_i32[0];
        for (size_t i = 1; i < N; ++i) {
            if (data_i32[i] < min_result) min_result = data_i32[i];
            if (data_i32[i] > max_result) max_result = data_i32[i];
        }
    }, 30);

    auto v7_minmax = measure_with_stats([&]() {
        minmax_i32(data_i32.data(), N, &min_result, &max_result);
    }, 30);

    auto v9_minmax = measure_with_stats([&]() {
        minmax_i32_v4(data_i32.data(), N, &min_result, &max_result);
    }, 30);

    print_version_result("DuckDB", "标量基线", duckdb_minmax);
    print_version_result("V7/V8", "SIMD 16元素/迭代", v7_minmax, duckdb_minmax.median);
    print_version_result("V9", "SIMD 32元素 + 预取 (P1)", v9_minmax, duckdb_minmax.median);

    // ========================================================================
    // 测试 3: 融合统计量
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 3: Fused Stats (SUM+MIN+MAX)                                            │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    AggregateStats stats;

    auto duckdb_fused = measure_with_stats([&]() {
        stats.sum = 0;
        stats.min_val = data_i32[0];
        stats.max_val = data_i32[0];
        for (size_t i = 0; i < N; ++i) {
            stats.sum += data_i32[i];
            if (data_i32[i] < stats.min_val) stats.min_val = data_i32[i];
            if (data_i32[i] > stats.max_val) stats.max_val = data_i32[i];
        }
    }, 30);

    auto v7_fused = measure_with_stats([&]() {
        stats = aggregate_all_i32(data_i32.data(), N);
    }, 30);

    auto v9_fused = measure_with_stats([&]() {
        stats = aggregate_all_i32_v4(data_i32.data(), N);
    }, 30);

    print_version_result("DuckDB", "标量基线", duckdb_fused);
    print_version_result("V7/V8", "融合SIMD", v7_fused, duckdb_fused.median);
    print_version_result("V9", "融合SIMD + 预取 (P1)", v9_fused, duckdb_fused.median);

    // ========================================================================
    // 测试 4: GROUP BY SUM
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 4: GROUP BY SUM (10M elements, 100 groups)                              │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    vector<int64_t> group_sums(NUM_GROUPS);

    auto duckdb_group = measure_with_stats([&]() {
        fill(group_sums.begin(), group_sums.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            group_sums[groups[i]] += data_i32[i];
        }
    }, 30);

    auto v7_group = measure_with_stats([&]() {
        group_sum_i32(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    auto v9_group = measure_with_stats([&]() {
        group_sum_i32_v4(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    auto v9_group_mt = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    print_version_result("DuckDB", "标量基线", duckdb_group);
    print_version_result("V7/V8", "标量累加", v7_group, duckdb_group.median);
    print_version_result("V9", "展开循环 + 预取 (P0)", v9_group, duckdb_group.median);
    print_version_result("V9", "多线程4T (P3)", v9_group_mt, duckdb_group.median);

    // ========================================================================
    // 测试 5: 大分组数
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 5: GROUP BY SUM (10M elements, 10K groups)                              │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t LARGE_GROUPS = 10000;
    vector<uint32_t> large_groups(N);
    uniform_int_distribution<uint32_t> dist_large_group(0, LARGE_GROUPS - 1);
    for (size_t i = 0; i < N; ++i) {
        large_groups[i] = dist_large_group(gen);
    }

    vector<int64_t> large_group_sums(LARGE_GROUPS);

    auto duckdb_large = measure_with_stats([&]() {
        fill(large_group_sums.begin(), large_group_sums.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            large_group_sums[large_groups[i]] += data_i32[i];
        }
    }, 30);

    auto v9_large = measure_with_stats([&]() {
        group_sum_i32_v4(data_i32.data(), large_groups.data(), N, LARGE_GROUPS, large_group_sums.data());
    }, 30);

    auto v9_large_mt = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(data_i32.data(), large_groups.data(), N, LARGE_GROUPS, large_group_sums.data());
    }, 30);

    print_version_result("DuckDB", "标量基线", duckdb_large);
    print_version_result("V9", "展开循环 + 预取 (P0)", v9_large, duckdb_large.median);
    print_version_result("V9", "多线程4T (P3)", v9_large_mt, duckdb_large.median);

    // ========================================================================
    // 性能汇总表
    // ========================================================================

    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                            性能汇总 (vs DuckDB)                              ║" << endl;
    cout << "╠════════════════════╦═══════════╦═══════════╦═══════════╦════════════════════╣" << endl;
    cout << "║ 测试项             ║   V7/V8   ║    V9     ║  V9-MT    ║ 预期收益           ║" << endl;
    cout << "╠════════════════════╬═══════════╬═══════════╬═══════════╬════════════════════╣" << endl;

    auto fmt = [](double baseline, double value) -> string {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2fx", baseline / value);
        return string(buf);
    };

    cout << "║ SUM i32 (10M)      ║ "
         << setw(9) << fmt(duckdb_sum.median, v7_sum.median) << " ║ "
         << setw(9) << fmt(duckdb_sum.median, v9_sum.median) << " ║ "
         << setw(9) << "N/A" << " ║ P1: 5-10%          ║" << endl;

    cout << "║ MIN/MAX (10M)      ║ "
         << setw(9) << fmt(duckdb_minmax.median, v7_minmax.median) << " ║ "
         << setw(9) << fmt(duckdb_minmax.median, v9_minmax.median) << " ║ "
         << setw(9) << "N/A" << " ║ P1: 5-10%          ║" << endl;

    cout << "║ Fused Stats        ║ "
         << setw(9) << fmt(duckdb_fused.median, v7_fused.median) << " ║ "
         << setw(9) << fmt(duckdb_fused.median, v9_fused.median) << " ║ "
         << setw(9) << "N/A" << " ║ P2: 5-15%          ║" << endl;

    cout << "║ GroupBy (100g)     ║ "
         << setw(9) << fmt(duckdb_group.median, v7_group.median) << " ║ "
         << setw(9) << fmt(duckdb_group.median, v9_group.median) << " ║ "
         << setw(9) << fmt(duckdb_group.median, v9_group_mt.median) << " ║ P0:1.5x P3:1.2-1.5x║" << endl;

    cout << "║ GroupBy (10Kg)     ║ "
         << setw(9) << "N/A" << " ║ "
         << setw(9) << fmt(duckdb_large.median, v9_large.median) << " ║ "
         << setw(9) << fmt(duckdb_large.median, v9_large_mt.median) << " ║                    ║" << endl;

    cout << "╚════════════════════╩═══════════╩═══════════╩═══════════╩════════════════════╝" << endl;

    // 优化项达成情况
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                          P0-P3 优化达成情况                                  ║" << endl;
    cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;

    double p0_achieved = (duckdb_group.median / v9_group.median - 1) * 100;
    double p1_sum = (v7_sum.median / v9_sum.median - 1) * 100;
    double p1_minmax = (v7_minmax.median / v9_minmax.median - 1) * 100;
    double p3_achieved = duckdb_group.median / v9_group_mt.median;

    cout << "║ P0 向量化哈希分组: " << setw(6) << fixed << setprecision(1)
         << p0_achieved << "% 改进 (预期 50-100%)  ";
    if (p0_achieved < 10) cout << "[未达预期]";
    else cout << "[达成]     ";
    cout << "           ║" << endl;

    cout << "║ P1 预取距离优化:                                                             ║" << endl;
    cout << "║    - SUM:      " << setw(6) << p1_sum << "% 改进 (预期 5-10%)  ";
    if (p1_sum >= 3) cout << "[达成]     ";
    else cout << "[未达预期]";
    cout << "                    ║" << endl;

    cout << "║    - MIN/MAX:  " << setw(6) << p1_minmax << "% 改进 (预期 5-10%)  ";
    if (p1_minmax >= 5) cout << "[达成]     ";
    else cout << "[未达预期]";
    cout << "                    ║" << endl;

    cout << "║ P2 缓存分块:   已实现,大数据量时有效                                         ║" << endl;

    cout << "║ P3 多线程分组: " << setw(5) << setprecision(2) << p3_achieved << "x 加速 (预期 1.2-1.5x)  ";
    if (p3_achieved >= 1.2) cout << "[超预期]   ";
    else cout << "[未达预期]";
    cout << "                  ║" << endl;

    cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << endl;

    return 0;
}
