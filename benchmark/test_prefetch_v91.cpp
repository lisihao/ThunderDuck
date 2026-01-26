/**
 * V9.1 预取优化对比测试
 *
 * 测试 Filter v6 和 TopK 多级预取优化
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "thunderduck/filter.h"
#include "thunderduck/sort.h"

using namespace std;
using namespace thunderduck::filter;
using namespace thunderduck::sort;

// ============================================================================
// 符合 CLAUDE.md 基准测试规则的测量函数
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
    cout << "  " << left << setw(25) << name << ": "
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
    cout << "\n╔══════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║           V9.1 预取优化对比测试                                      ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════╝" << endl;

    // ========================================================================
    // 数据准备
    // ========================================================================

    const size_t N = 10000000;  // 10M 元素
    const size_t K = 100;       // TopK

    vector<int32_t> data_i32(N);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_i32(0, 10000000);

    for (size_t i = 0; i < N; ++i) {
        data_i32[i] = dist_i32(gen);
    }

    cout << "\n数据规模: " << N << " 元素 (10M)" << endl;
    cout << "迭代次数: 30 次 (IQR剔除异常值)\n" << endl;

    // ========================================================================
    // 测试 1: Filter COUNT
    // ========================================================================

    cout << "=== Filter COUNT (10M, >5000000, ~50% 选择率) ===" << endl;

    auto filter_v3 = measure_with_stats([&]() {
        count_i32_v3(data_i32.data(), N, CompareOp::GT, 5000000);
    }, 30);

    auto filter_v5 = measure_with_stats([&]() {
        count_i32_v5(data_i32.data(), N, CompareOp::GT, 5000000);
    }, 30);

    auto filter_v6 = measure_with_stats([&]() {
        count_i32_v6(data_i32.data(), N, CompareOp::GT, 5000000);
    }, 30);

    print_result("Filter v3 (64B prefetch)", filter_v3);
    print_result("Filter v5 (V8)", filter_v5, filter_v3.median);
    print_result("Filter v6 (V9.1 multi-lvl)", filter_v6, filter_v3.median);

    // ========================================================================
    // 测试 2: Filter 索引输出
    // ========================================================================

    cout << "\n=== Filter INDEX (10M, >9000000, ~10% 选择率) ===" << endl;

    vector<uint32_t> indices(N);

    auto filter_idx_v3 = measure_with_stats([&]() {
        filter_i32_v3(data_i32.data(), N, CompareOp::GT, 9000000, indices.data());
    }, 30);

    auto filter_idx_v5 = measure_with_stats([&]() {
        filter_i32_v5(data_i32.data(), N, CompareOp::GT, 9000000, indices.data());
    }, 30);

    auto filter_idx_v6 = measure_with_stats([&]() {
        filter_i32_v6(data_i32.data(), N, CompareOp::GT, 9000000, indices.data());
    }, 30);

    print_result("Filter v3 (64B prefetch)", filter_idx_v3);
    print_result("Filter v5 (V8)", filter_idx_v5, filter_idx_v3.median);
    print_result("Filter v6 (V9.1 multi-lvl)", filter_idx_v6, filter_idx_v3.median);

    // ========================================================================
    // 测试 3: TopK
    // ========================================================================

    cout << "\n=== TopK (10M, K=100) ===" << endl;

    vector<int32_t> topk_values(K);
    vector<uint32_t> topk_indices(K);

    auto topk_v4 = measure_with_stats([&]() {
        topk_max_i32_v4(data_i32.data(), N, K, topk_values.data(), topk_indices.data());
    }, 30);

    print_result("TopK v4 (V9.1 multi-lvl)", topk_v4);

    // ========================================================================
    // 测试 4: TopK 不同 K 值
    // ========================================================================

    cout << "\n=== TopK 不同 K 值 ===" << endl;

    for (size_t test_k : {10, 100, 500, 1000}) {
        vector<int32_t> v(test_k);
        vector<uint32_t> idx(test_k);

        auto result = measure_with_stats([&]() {
            topk_max_i32_v4(data_i32.data(), N, test_k, v.data(), idx.data());
        }, 30);

        cout << "  K=" << setw(5) << test_k << ": "
             << fixed << setprecision(3) << result.median << " ms"
             << " (σ=" << result.stddev << ")" << endl;
    }

    // ========================================================================
    // 汇总
    // ========================================================================

    cout << "\n" << string(70, '=') << endl;
    cout << "V9.1 预取优化汇总" << endl;
    cout << string(70, '-') << endl;

    cout << "Filter COUNT:  v3=" << fixed << setprecision(3) << filter_v3.median << "ms"
         << "  v6=" << filter_v6.median << "ms"
         << "  改进=" << setprecision(1) << ((filter_v3.median / filter_v6.median - 1) * 100) << "%" << endl;

    cout << "Filter INDEX:  v3=" << setprecision(3) << filter_idx_v3.median << "ms"
         << "  v6=" << filter_idx_v6.median << "ms"
         << "  改进=" << setprecision(1) << ((filter_idx_v3.median / filter_idx_v6.median - 1) * 100) << "%" << endl;

    cout << string(70, '=') << endl;

    return 0;
}
