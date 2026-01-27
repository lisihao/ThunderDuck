/**
 * V9.2 GPU 两阶段分组聚合 对比测试
 *
 * 对比版本:
 * - DuckDB: 标量基线
 * - V7/V8 v1: CPU 标量累加
 * - V9 v4: CPU 多线程 (4T)
 * - V9 v3: GPU 原子版 (V8)
 * - V9.2 v5: GPU 两阶段 (本次优化)
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

void print_result(const string& name, const BenchResult& r, double baseline = 0) {
    cout << "  " << left << setw(30) << name << ": "
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
    cout << "║     V9.2 GPU 两阶段分组聚合 对比测试                                         ║" << endl;
    cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
    cout << "║ V7/V8 v1: CPU 标量累加                                                       ║" << endl;
    cout << "║ V9 v4-MT: CPU 多线程 (4T)                                                    ║" << endl;
    cout << "║ V9 v3:    GPU 原子版 (V8 实现)                                               ║" << endl;
    cout << "║ V9.2 v5:  GPU 两阶段 (本地累加 + 全局合并)                                   ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << endl;

    // 检查 GPU 可用性
    bool gpu_v2_available = is_group_aggregate_v2_available();
    bool gpu_v3_available = is_aggregate_gpu_available();

    cout << "\nGPU 状态: V8=" << (gpu_v3_available ? "可用" : "不可用")
         << ", V9.2=" << (gpu_v2_available ? "可用" : "不可用") << endl;

    // ========================================================================
    // 数据准备
    // ========================================================================

    const size_t N = 10000000;  // 10M 元素

    vector<int32_t> values(N);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_val(-1000, 1000);

    for (size_t i = 0; i < N; ++i) {
        values[i] = dist_val(gen);
    }

    cout << "\n数据规模: " << N / 1000000 << "M 元素" << endl;
    cout << "迭代次数: 30 次 (IQR剔除异常值)\n" << endl;

    // ========================================================================
    // 测试 1: 100 分组
    // ========================================================================

    cout << "╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 1: GROUP BY SUM (10M elements, 100 groups)                              │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t GROUPS_100 = 100;
    vector<uint32_t> groups_100(N);
    uniform_int_distribution<uint32_t> dist_g100(0, GROUPS_100 - 1);
    for (size_t i = 0; i < N; ++i) {
        groups_100[i] = dist_g100(gen);
    }

    vector<int64_t> sums(GROUPS_100);

    // DuckDB 模拟
    auto duckdb_100 = measure_with_stats([&]() {
        fill(sums.begin(), sums.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            sums[groups_100[i]] += values[i];
        }
    }, 30);

    // V7/V8 v1
    auto v1_100 = measure_with_stats([&]() {
        group_sum_i32(values.data(), groups_100.data(), N, GROUPS_100, sums.data());
    }, 30);

    // V9 v4 多线程
    auto v4_mt_100 = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values.data(), groups_100.data(), N, GROUPS_100, sums.data());
    }, 30);

    // V9 v3 GPU 原子
    auto v3_gpu_100 = measure_with_stats([&]() {
        group_sum_i32_v3(values.data(), groups_100.data(), N, GROUPS_100, sums.data());
    }, 30);

    // V9.2 v5 GPU 两阶段
    auto v5_gpu_100 = measure_with_stats([&]() {
        group_sum_i32_v5(values.data(), groups_100.data(), N, GROUPS_100, sums.data());
    }, 30);

    print_result("DuckDB (标量基线)", duckdb_100);
    print_result("V7/V8 v1 (CPU 标量)", v1_100, duckdb_100.median);
    print_result("V9 v4-MT (CPU 4线程)", v4_mt_100, duckdb_100.median);
    print_result("V9 v3 (GPU 原子)", v3_gpu_100, duckdb_100.median);
    print_result("V9.2 v5 (GPU 两阶段)", v5_gpu_100, duckdb_100.median);

    // ========================================================================
    // 测试 2: 1000 分组
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 2: GROUP BY SUM (10M elements, 1000 groups)                             │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t GROUPS_1K = 1000;
    vector<uint32_t> groups_1k(N);
    vector<int64_t> sums_1k(GROUPS_1K);
    uniform_int_distribution<uint32_t> dist_g1k(0, GROUPS_1K - 1);
    for (size_t i = 0; i < N; ++i) {
        groups_1k[i] = dist_g1k(gen);
    }

    auto duckdb_1k = measure_with_stats([&]() {
        fill(sums_1k.begin(), sums_1k.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            sums_1k[groups_1k[i]] += values[i];
        }
    }, 30);

    auto v4_mt_1k = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values.data(), groups_1k.data(), N, GROUPS_1K, sums_1k.data());
    }, 30);

    auto v3_gpu_1k = measure_with_stats([&]() {
        group_sum_i32_v3(values.data(), groups_1k.data(), N, GROUPS_1K, sums_1k.data());
    }, 30);

    auto v5_gpu_1k = measure_with_stats([&]() {
        group_sum_i32_v5(values.data(), groups_1k.data(), N, GROUPS_1K, sums_1k.data());
    }, 30);

    print_result("DuckDB (标量基线)", duckdb_1k);
    print_result("V9 v4-MT (CPU 4线程)", v4_mt_1k, duckdb_1k.median);
    print_result("V9 v3 (GPU 原子)", v3_gpu_1k, duckdb_1k.median);
    print_result("V9.2 v5 (GPU 两阶段)", v5_gpu_1k, duckdb_1k.median);

    // ========================================================================
    // 测试 3: 10 分组 (高竞争)
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 3: GROUP BY SUM (10M elements, 10 groups) - 高原子竞争                  │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t GROUPS_10 = 10;
    vector<uint32_t> groups_10(N);
    vector<int64_t> sums_10(GROUPS_10);
    uniform_int_distribution<uint32_t> dist_g10(0, GROUPS_10 - 1);
    for (size_t i = 0; i < N; ++i) {
        groups_10[i] = dist_g10(gen);
    }

    auto duckdb_10 = measure_with_stats([&]() {
        fill(sums_10.begin(), sums_10.end(), 0);
        for (size_t i = 0; i < N; ++i) {
            sums_10[groups_10[i]] += values[i];
        }
    }, 30);

    auto v4_mt_10 = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values.data(), groups_10.data(), N, GROUPS_10, sums_10.data());
    }, 30);

    auto v3_gpu_10 = measure_with_stats([&]() {
        group_sum_i32_v3(values.data(), groups_10.data(), N, GROUPS_10, sums_10.data());
    }, 30);

    auto v5_gpu_10 = measure_with_stats([&]() {
        group_sum_i32_v5(values.data(), groups_10.data(), N, GROUPS_10, sums_10.data());
    }, 30);

    print_result("DuckDB (标量基线)", duckdb_10);
    print_result("V9 v4-MT (CPU 4线程)", v4_mt_10, duckdb_10.median);
    print_result("V9 v3 (GPU 原子)", v3_gpu_10, duckdb_10.median);
    print_result("V9.2 v5 (GPU 两阶段)", v5_gpu_10, duckdb_10.median);

    // ========================================================================
    // 性能汇总
    // ========================================================================

    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                            V9.2 性能汇总                                     ║" << endl;
    cout << "╠════════════════════╦═══════════╦═══════════╦═══════════╦═══════════╗" << endl;
    cout << "║ 分组数             ║  CPU v4   ║  GPU v3   ║  GPU v5   ║ v5 vs v3  ║" << endl;
    cout << "╠════════════════════╬═══════════╬═══════════╬═══════════╬═══════════╣" << endl;

    auto fmt = [](double baseline, double value) -> string {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2fx", baseline / value);
        return string(buf);
    };

    auto fmt_pct = [](double v3, double v5) -> string {
        char buf[32];
        double pct = (v3 / v5 - 1) * 100;
        snprintf(buf, sizeof(buf), "%+.1f%%", pct);
        return string(buf);
    };

    cout << "║ 10 groups (高竞争) ║ "
         << setw(9) << fmt(duckdb_10.median, v4_mt_10.median) << " ║ "
         << setw(9) << fmt(duckdb_10.median, v3_gpu_10.median) << " ║ "
         << setw(9) << fmt(duckdb_10.median, v5_gpu_10.median) << " ║ "
         << setw(9) << fmt_pct(v3_gpu_10.median, v5_gpu_10.median) << " ║" << endl;

    cout << "║ 100 groups         ║ "
         << setw(9) << fmt(duckdb_100.median, v4_mt_100.median) << " ║ "
         << setw(9) << fmt(duckdb_100.median, v3_gpu_100.median) << " ║ "
         << setw(9) << fmt(duckdb_100.median, v5_gpu_100.median) << " ║ "
         << setw(9) << fmt_pct(v3_gpu_100.median, v5_gpu_100.median) << " ║" << endl;

    cout << "║ 1000 groups        ║ "
         << setw(9) << fmt(duckdb_1k.median, v4_mt_1k.median) << " ║ "
         << setw(9) << fmt(duckdb_1k.median, v3_gpu_1k.median) << " ║ "
         << setw(9) << fmt(duckdb_1k.median, v5_gpu_1k.median) << " ║ "
         << setw(9) << fmt_pct(v3_gpu_1k.median, v5_gpu_1k.median) << " ║" << endl;

    cout << "╚════════════════════╩═══════════╩═══════════╩═══════════╩═══════════╝" << endl;

    cout << "\n说明:" << endl;
    cout << "  - v5 vs v3: GPU 两阶段相对于 GPU 原子版的改进" << endl;
    cout << "  - 正值表示 v5 更快" << endl;

    return 0;
}
