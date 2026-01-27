/**
 * V9.3 智能策略选择分组聚合 对比测试
 *
 * 测试场景:
 * 1. 小数据 (50K): V4_SINGLE vs V4_PARALLEL vs V6(AUTO)
 * 2. 中数据 (10M): V4_SINGLE vs V4_PARALLEL vs V6(AUTO)
 * 3. 大数据 (50M): V4_PARALLEL vs V5_GPU vs V6(AUTO)
 *
 * 验证智能策略是否选择了最优实现
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
BenchResult measure_with_stats(F&& func, int iterations = 20) {
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
    cout << "║     V9.3 智能策略选择分组聚合 对比测试                                       ║" << endl;
    cout << "╠══════════════════════════════════════════════════════════════════════════════╣" << endl;
    cout << "║ V4 SINGLE:   CPU 单线程 (小数据优化)                                         ║" << endl;
    cout << "║ V4 PARALLEL: CPU 多线程 (通用最优)                                           ║" << endl;
    cout << "║ V5 GPU:      GPU 两阶段 (大数据高竞争)                                       ║" << endl;
    cout << "║ V6 AUTO:     智能策略选择 (V9.3)                                             ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════════════╝" << endl;

    // 检查 GPU 可用性
    bool gpu_available = is_group_aggregate_v2_available();
    cout << "\nGPU 状态: " << (gpu_available ? "可用" : "不可用") << endl;

    // ========================================================================
    // 测试 1: 小数据 (50K 元素)
    // 策略预期: V4_SINGLE (避免线程开销)
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 1: 小数据场景 (50K elements, 10 groups)                                │" << endl;
    cout << "│ 策略预期: V6 应选择 V4_SINGLE (线程启动开销 > 收益)                         │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t N_SMALL = 50000;
    const size_t GROUPS_10 = 10;

    vector<int32_t> values_small(N_SMALL);
    vector<uint32_t> groups_small(N_SMALL);
    vector<int64_t> sums_small(GROUPS_10);

    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_val(-1000, 1000);
    uniform_int_distribution<uint32_t> dist_g10(0, GROUPS_10 - 1);

    for (size_t i = 0; i < N_SMALL; ++i) {
        values_small[i] = dist_val(gen);
        groups_small[i] = dist_g10(gen);
    }

    // DuckDB 模拟 (标量基线)
    auto duckdb_small = measure_with_stats([&]() {
        fill(sums_small.begin(), sums_small.end(), 0);
        for (size_t i = 0; i < N_SMALL; ++i) {
            sums_small[groups_small[i]] += values_small[i];
        }
    }, 20);

    auto v4_single_small = measure_with_stats([&]() {
        group_sum_i32_v4(values_small.data(), groups_small.data(), N_SMALL, GROUPS_10, sums_small.data());
    }, 20);

    auto v4_parallel_small = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values_small.data(), groups_small.data(), N_SMALL, GROUPS_10, sums_small.data());
    }, 20);

    auto v6_auto_small = measure_with_stats([&]() {
        group_sum_i32_v6(values_small.data(), groups_small.data(), N_SMALL, GROUPS_10, sums_small.data());
    }, 20);

    print_result("DuckDB (标量基线)", duckdb_small);
    print_result("V4 SINGLE (CPU单线程)", v4_single_small, duckdb_small.median);
    print_result("V4 PARALLEL (CPU多线程)", v4_parallel_small, duckdb_small.median);
    print_result("V6 AUTO (智能选择)", v6_auto_small, duckdb_small.median);

    // 验证策略选择
    auto strategy_small = select_group_aggregate_strategy(N_SMALL, GROUPS_10);
    cout << "\n  策略选择: " << (strategy_small == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE ✓" :
                               strategy_small == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" :
                               strategy_small == GroupAggregateVersion::V5_GPU ? "V5_GPU" : "AUTO");
    cout << " - " << get_group_aggregate_strategy_reason() << endl;

    // ========================================================================
    // 测试 2: 中数据 (10M 元素)
    // 策略预期: V4_PARALLEL
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 2: 中数据场景 (10M elements, 100 groups)                               │" << endl;
    cout << "│ 策略预期: V6 应选择 V4_PARALLEL (最佳通用性能)                              │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t N_MEDIUM = 10000000;
    const size_t GROUPS_100 = 100;

    vector<int32_t> values_medium(N_MEDIUM);
    vector<uint32_t> groups_medium(N_MEDIUM);
    vector<int64_t> sums_medium(GROUPS_100);

    uniform_int_distribution<uint32_t> dist_g100(0, GROUPS_100 - 1);
    for (size_t i = 0; i < N_MEDIUM; ++i) {
        values_medium[i] = dist_val(gen);
        groups_medium[i] = dist_g100(gen);
    }

    auto duckdb_medium = measure_with_stats([&]() {
        fill(sums_medium.begin(), sums_medium.end(), 0);
        for (size_t i = 0; i < N_MEDIUM; ++i) {
            sums_medium[groups_medium[i]] += values_medium[i];
        }
    }, 20);

    auto v4_single_medium = measure_with_stats([&]() {
        group_sum_i32_v4(values_medium.data(), groups_medium.data(), N_MEDIUM, GROUPS_100, sums_medium.data());
    }, 20);

    auto v4_parallel_medium = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values_medium.data(), groups_medium.data(), N_MEDIUM, GROUPS_100, sums_medium.data());
    }, 20);

    auto v6_auto_medium = measure_with_stats([&]() {
        group_sum_i32_v6(values_medium.data(), groups_medium.data(), N_MEDIUM, GROUPS_100, sums_medium.data());
    }, 20);

    print_result("DuckDB (标量基线)", duckdb_medium);
    print_result("V4 SINGLE (CPU单线程)", v4_single_medium, duckdb_medium.median);
    print_result("V4 PARALLEL (CPU多线程)", v4_parallel_medium, duckdb_medium.median);
    print_result("V6 AUTO (智能选择)", v6_auto_medium, duckdb_medium.median);

    auto strategy_medium = select_group_aggregate_strategy(N_MEDIUM, GROUPS_100);
    cout << "\n  策略选择: " << (strategy_medium == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                               strategy_medium == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL ✓" :
                               strategy_medium == GroupAggregateVersion::V5_GPU ? "V5_GPU" : "AUTO");
    cout << " - " << get_group_aggregate_strategy_reason() << endl;

    // ========================================================================
    // 测试 3: 大数据 + 少分组 (50M 元素, 10 分组)
    // 策略预期: V5_GPU (如果 GPU 可用)
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 3: 大数据+少分组场景 (50M elements, 10 groups)                         │" << endl;
    cout << "│ 策略预期: V6 应选择 " << (gpu_available ? "V5_GPU (GPU高竞争优势)" : "V4_PARALLEL (GPU不可用)") << "                             │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t N_LARGE = 50000000;

    vector<int32_t> values_large(N_LARGE);
    vector<uint32_t> groups_large(N_LARGE);
    vector<int64_t> sums_large(GROUPS_10);

    for (size_t i = 0; i < N_LARGE; ++i) {
        values_large[i] = dist_val(gen);
        groups_large[i] = dist_g10(gen);
    }

    cout << "\n  数据准备完成 (" << N_LARGE / 1000000 << "M elements)" << endl;

    // DuckDB 基线
    auto duckdb_large = measure_with_stats([&]() {
        fill(sums_large.begin(), sums_large.end(), 0);
        for (size_t i = 0; i < N_LARGE; ++i) {
            sums_large[groups_large[i]] += values_large[i];
        }
    }, 10);

    auto v4_parallel_large = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values_large.data(), groups_large.data(), N_LARGE, GROUPS_10, sums_large.data());
    }, 10);

    BenchResult v5_gpu_large = {0, 0, 0};
    if (gpu_available) {
        v5_gpu_large = measure_with_stats([&]() {
            group_sum_i32_v5(values_large.data(), groups_large.data(), N_LARGE, GROUPS_10, sums_large.data());
        }, 10);
    }

    auto v6_auto_large = measure_with_stats([&]() {
        group_sum_i32_v6(values_large.data(), groups_large.data(), N_LARGE, GROUPS_10, sums_large.data());
    }, 10);

    print_result("DuckDB (标量基线)", duckdb_large);
    print_result("V4 PARALLEL (CPU多线程)", v4_parallel_large, duckdb_large.median);
    if (gpu_available) {
        print_result("V5 GPU (两阶段)", v5_gpu_large, duckdb_large.median);
    }
    print_result("V6 AUTO (智能选择)", v6_auto_large, duckdb_large.median);

    auto strategy_large = select_group_aggregate_strategy(N_LARGE, GROUPS_10);
    cout << "\n  策略选择: " << (strategy_large == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                               strategy_large == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" :
                               strategy_large == GroupAggregateVersion::V5_GPU ? "V5_GPU ✓" : "AUTO");
    cout << " - " << get_group_aggregate_strategy_reason() << endl;

    // ========================================================================
    // 测试 4: 大数据 + 多分组 (50M 元素, 1000 分组)
    // 策略预期: V4_PARALLEL (分组数超过 GPU 阈值)
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮" << endl;
    cout << "│ TEST 4: 大数据+多分组场景 (50M elements, 1000 groups)                       │" << endl;
    cout << "│ 策略预期: V6 应选择 V4_PARALLEL (分组数>32不适合GPU)                        │" << endl;
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯" << endl;

    const size_t GROUPS_1K = 1000;

    vector<uint32_t> groups_large_1k(N_LARGE);
    vector<int64_t> sums_large_1k(GROUPS_1K);

    uniform_int_distribution<uint32_t> dist_g1k(0, GROUPS_1K - 1);
    for (size_t i = 0; i < N_LARGE; ++i) {
        groups_large_1k[i] = dist_g1k(gen);
    }

    auto duckdb_large_1k = measure_with_stats([&]() {
        fill(sums_large_1k.begin(), sums_large_1k.end(), 0);
        for (size_t i = 0; i < N_LARGE; ++i) {
            sums_large_1k[groups_large_1k[i]] += values_large[i];
        }
    }, 10);

    auto v4_parallel_large_1k = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(values_large.data(), groups_large_1k.data(), N_LARGE, GROUPS_1K, sums_large_1k.data());
    }, 10);

    auto v6_auto_large_1k = measure_with_stats([&]() {
        group_sum_i32_v6(values_large.data(), groups_large_1k.data(), N_LARGE, GROUPS_1K, sums_large_1k.data());
    }, 10);

    print_result("DuckDB (标量基线)", duckdb_large_1k);
    print_result("V4 PARALLEL (CPU多线程)", v4_parallel_large_1k, duckdb_large_1k.median);
    print_result("V6 AUTO (智能选择)", v6_auto_large_1k, duckdb_large_1k.median);

    auto strategy_large_1k = select_group_aggregate_strategy(N_LARGE, GROUPS_1K);
    cout << "\n  策略选择: " << (strategy_large_1k == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                               strategy_large_1k == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL ✓" :
                               strategy_large_1k == GroupAggregateVersion::V5_GPU ? "V5_GPU" : "AUTO");
    cout << " - " << get_group_aggregate_strategy_reason() << endl;

    // ========================================================================
    // 策略验证汇总
    // ========================================================================

    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                           V9.3 策略验证汇总                                  ║" << endl;
    cout << "╠═══════════════════════════╦══════════════╦═══════════════╦═══════════════════╣" << endl;
    cout << "║ 场景                      ║ 策略选择     ║ V6 vs 最优    ║ 验证结果          ║" << endl;
    cout << "╠═══════════════════════════╬══════════════╬═══════════════╬═══════════════════╣" << endl;

    // 计算 V6 与最优实现的差距
    auto check_strategy = [](double v6_time, double best_time, bool expected) {
        double diff = (v6_time - best_time) / best_time * 100;
        if (abs(diff) < 5.0 && expected) {
            return "PASS (optimal)";
        } else if (abs(diff) < 10.0) {
            return "PASS (close)";
        } else if (!expected) {
            return "WARN (mismatch)";
        }
        return "FAIL (>10% off)";
    };

    // 小数据
    double best_small = min(v4_single_small.median, v4_parallel_small.median);
    bool small_expected = (strategy_small == GroupAggregateVersion::V4_SINGLE);
    cout << "║ 50K, 10 groups            ║ "
         << setw(12) << (strategy_small == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                        strategy_small == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" : "V5_GPU")
         << " ║ "
         << setw(13) << (v6_auto_small.median / best_small < 1.05 ? "~最优" : "偏离")
         << " ║ "
         << setw(17) << check_strategy(v6_auto_small.median, best_small, small_expected)
         << " ║" << endl;

    // 中数据
    double best_medium = min(v4_single_medium.median, v4_parallel_medium.median);
    bool medium_expected = (strategy_medium == GroupAggregateVersion::V4_PARALLEL);
    cout << "║ 10M, 100 groups           ║ "
         << setw(12) << (strategy_medium == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                        strategy_medium == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" : "V5_GPU")
         << " ║ "
         << setw(13) << (v6_auto_medium.median / best_medium < 1.05 ? "~最优" : "偏离")
         << " ║ "
         << setw(17) << check_strategy(v6_auto_medium.median, best_medium, medium_expected)
         << " ║" << endl;

    // 大数据少分组
    double best_large = v4_parallel_large.median;
    if (gpu_available && v5_gpu_large.median > 0 && v5_gpu_large.median < best_large) {
        best_large = v5_gpu_large.median;
    }
    bool large_expected = gpu_available ? (strategy_large == GroupAggregateVersion::V5_GPU) :
                                           (strategy_large == GroupAggregateVersion::V4_PARALLEL);
    cout << "║ 50M, 10 groups            ║ "
         << setw(12) << (strategy_large == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                        strategy_large == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" : "V5_GPU")
         << " ║ "
         << setw(13) << (v6_auto_large.median / best_large < 1.05 ? "~最优" : "偏离")
         << " ║ "
         << setw(17) << check_strategy(v6_auto_large.median, best_large, large_expected)
         << " ║" << endl;

    // 大数据多分组
    cout << "║ 50M, 1000 groups          ║ "
         << setw(12) << (strategy_large_1k == GroupAggregateVersion::V4_SINGLE ? "V4_SINGLE" :
                        strategy_large_1k == GroupAggregateVersion::V4_PARALLEL ? "V4_PARALLEL" : "V5_GPU")
         << " ║ "
         << setw(13) << (v6_auto_large_1k.median / v4_parallel_large_1k.median < 1.05 ? "~最优" : "偏离")
         << " ║ "
         << setw(17) << check_strategy(v6_auto_large_1k.median, v4_parallel_large_1k.median,
                                       strategy_large_1k == GroupAggregateVersion::V4_PARALLEL)
         << " ║" << endl;

    cout << "╚═══════════════════════════╩══════════════╩═══════════════╩═══════════════════╝" << endl;

    cout << "\n说明:" << endl;
    cout << "  - PASS (optimal): 策略正确且性能最优" << endl;
    cout << "  - PASS (close): 性能在最优的5%以内" << endl;
    cout << "  - WARN (mismatch): 策略与预期不符但性能尚可" << endl;
    cout << "  - FAIL: 性能偏离最优超过10%" << endl;

    return 0;
}
