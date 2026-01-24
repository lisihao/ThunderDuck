/**
 * TopK v4.0 Benchmark
 *
 * 专门测试 T4 场景 (10M 行, K=10) 的性能优化效果
 */

#include "thunderduck/sort.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>

using namespace thunderduck::sort;
using namespace std::chrono;

// 生成随机数据
std::vector<int32_t> generate_data(size_t n, int32_t min_val = 0, int32_t max_val = 1000000) {
    std::vector<int32_t> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

// 验证结果正确性
bool verify_topk_max(const int32_t* data, size_t n, size_t k,
                     const int32_t* result_values, const uint32_t* result_indices) {
    // 使用标准库计算参考结果
    std::vector<std::pair<int32_t, uint32_t>> pairs(n);
    for (size_t i = 0; i < n; ++i) {
        pairs[i] = {data[i], static_cast<uint32_t>(i)};
    }

    std::nth_element(pairs.begin(), pairs.begin() + k, pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::sort(pairs.begin(), pairs.begin() + k,
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // 比较值
    for (size_t i = 0; i < k; ++i) {
        if (result_values[i] != pairs[i].first) {
            std::cerr << "Value mismatch at " << i << ": expected "
                      << pairs[i].first << ", got " << result_values[i] << std::endl;
            return false;
        }
    }

    return true;
}

// 基准测试函数
template<typename Func>
double benchmark(Func func, int warmup = 3, int iterations = 10) {
    // 预热
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // 测量
    double total_ms = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        total_ms += duration<double, std::milli>(end - start).count();
    }

    return total_ms / iterations;
}

void run_benchmark(size_t n, size_t k, const char* test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "数据量: " << n / 1000000.0 << "M 行, K=" << k << std::endl;

    auto data = generate_data(n);

    std::vector<int32_t> values_v3(k);
    std::vector<uint32_t> indices_v3(k);
    std::vector<int32_t> values_v4(k);
    std::vector<uint32_t> indices_v4(k);

    // 测试 v3
    double time_v3 = benchmark([&]() {
        topk_max_i32_v3(data.data(), data.size(), k,
                        values_v3.data(), indices_v3.data());
    });

    // 测试 v4
    double time_v4 = benchmark([&]() {
        topk_max_i32_v4(data.data(), data.size(), k,
                        values_v4.data(), indices_v4.data());
    });

    // 验证正确性
    bool v3_correct = verify_topk_max(data.data(), n, k, values_v3.data(), indices_v3.data());
    bool v4_correct = verify_topk_max(data.data(), n, k, values_v4.data(), indices_v4.data());

    // 输出结果
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "v3.0: " << time_v3 << " ms" << (v3_correct ? " ✓" : " ✗") << std::endl;
    std::cout << "v4.0: " << time_v4 << " ms" << (v4_correct ? " ✓" : " ✗") << std::endl;

    double speedup = time_v3 / time_v4;
    std::cout << "v4 加速比: " << speedup << "x"
              << (speedup > 1.0 ? " (提升)" : " (下降)") << std::endl;

    // 对比之前的 DuckDB 基准 (T4: 2.02ms)
    if (n == 10000000 && k == 10) {
        double duckdb_time = 2.02;  // 之前测试的 DuckDB 时间
        double vs_duckdb = duckdb_time / time_v4;
        std::cout << "vs DuckDB (2.02ms): " << vs_duckdb << "x" << std::endl;
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          ThunderDuck TopK v4.0 性能基准测试                 ║" << std::endl;
    std::cout << "║          采样预过滤 + SIMD 批量跳过优化                     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    // T1: 1M, K=10 (v3 已经赢了)
    run_benchmark(1000000, 10, "T1: 1M rows, K=10");

    // T2: 1M, K=100
    run_benchmark(1000000, 100, "T2: 1M rows, K=100");

    // T3: 1M, K=1000
    run_benchmark(1000000, 1000, "T3: 1M rows, K=1000");

    // T4: 10M, K=10 (核心优化目标!)
    run_benchmark(10000000, 10, "T4: 10M rows, K=10 (核心优化目标)");

    // T5: 10M, K=100
    run_benchmark(10000000, 100, "T5: 10M rows, K=100");

    // T6: 10M, K=1000
    run_benchmark(10000000, 1000, "T6: 10M rows, K=1000");

    std::cout << "\n=== 总结 ===" << std::endl;
    std::cout << "v4.0 核心优化: 采样预过滤 + SIMD 批量跳过" << std::endl;
    std::cout << "目标场景: N >= 1M 且 K <= 64 的大数据量小K场景" << std::endl;

    return 0;
}
