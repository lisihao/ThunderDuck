/**
 * TopK 快速测试 - 验证 100K K=10 优化
 */

#include "thunderduck/sort.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

using namespace std;
using namespace thunderduck::sort;

template<typename Func>
double measure_time(Func&& func, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 3; ++i) func();

    double total = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, micro>(end - start).count();
    }
    return total / iterations;
}

// Baseline WITHOUT indices (like simple DuckDB partial_sort)
double baseline_topk_no_idx(const vector<int32_t>& data, size_t k, vector<int32_t>& result) {
    return measure_time([&]() {
        vector<int32_t> temp = data;
        partial_sort(temp.begin(), temp.begin() + k, temp.end(), greater<int32_t>());
        copy(temp.begin(), temp.begin() + k, result.begin());
    });
}

// Baseline WITH indices (fair comparison)
double baseline_topk_with_idx(const vector<int32_t>& data, size_t k,
                              vector<int32_t>& result, vector<uint32_t>& indices) {
    return measure_time([&]() {
        vector<pair<int32_t, uint32_t>> indexed(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            indexed[i] = {data[i], static_cast<uint32_t>(i)};
        }
        partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < k; ++i) {
            result[i] = indexed[i].first;
            indices[i] = indexed[i].second;
        }
    });
}

int main() {
    cout << "═══════════════════════════════════════════════════════════════\n";
    cout << "          TopK 优化验证测试 (v5 partial_sort)\n";
    cout << "═══════════════════════════════════════════════════════════════\n\n";

    random_device rd;
    mt19937 gen(rd());

    struct TestCase {
        size_t count;
        size_t k;
        const char* desc;
    };

    vector<TestCase> tests = {
        {100000,  10,   "100K K=10  (目标场景)"},
        {100000,  100,  "100K K=100"},
        {500000,  10,   "500K K=10"},
        {1000000, 10,   "1M K=10"},
        {1000000, 100,  "1M K=100"},
        {5000000, 10,   "5M K=10"},
        {10000000, 10,  "10M K=10"},
    };

    printf("%-25s %10s %10s %10s %10s %8s\n",
           "场景", "NoIdx(μs)", "WithIdx(μs)", "v5(μs)", "v4(μs)", "v5/Idx");
    printf("────────────────────────────────────────────────────────────────────\n");

    for (const auto& test : tests) {
        // Generate random data
        vector<int32_t> data(test.count);
        uniform_int_distribution<int32_t> dist(-1000000, 1000000);
        for (size_t i = 0; i < test.count; ++i) {
            data[i] = dist(gen);
        }

        vector<int32_t> result(test.k);
        vector<uint32_t> indices(test.k);
        vector<int32_t> result_v4(test.k);
        vector<uint32_t> indices_v4(test.k);

        // Baseline without indices
        double baseline_no_idx = baseline_topk_no_idx(data, test.k, result);

        // Baseline with indices (fair comparison)
        double baseline_with_idx = baseline_topk_with_idx(data, test.k, result, indices);

        // v5 (optimized)
        double v5_time = measure_time([&]() {
            topk_max_i32_v5(data.data(), data.size(), test.k,
                           result.data(), indices.data());
        });

        // v4 for comparison
        double v4_time = measure_time([&]() {
            topk_max_i32_v4(data.data(), data.size(), test.k,
                           result_v4.data(), indices_v4.data());
        });

        double speedup = baseline_with_idx / v5_time;

        printf("%-25s %10.1f %10.1f %10.1f %10.1f %7.2fx\n",
               test.desc, baseline_no_idx, baseline_with_idx, v5_time, v4_time, speedup);
    }

    cout << "\n═══════════════════════════════════════════════════════════════\n";
    cout << "注: Baseline 使用 std::partial_sort (类似 DuckDB)\n";
    cout << "    v5 使用优化的自适应策略\n";
    cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
