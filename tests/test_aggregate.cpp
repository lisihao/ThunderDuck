/**
 * ThunderDuck - Aggregation Operator Tests
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/memory.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <cmath>

using namespace thunderduck;
using namespace thunderduck::aggregate;

void test_sum_i32() {
    std::cout << "Testing sum_i32... ";
    
    alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    size_t count = 10;
    
    int64_t result = sum_i32(data, count);
    assert(result == 55);  // 1+2+...+10 = 55
    
    std::cout << "PASSED (sum=" << result << ")\n";
}

void test_sum_f32() {
    std::cout << "Testing sum_f32... ";
    
    alignas(128) float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
    size_t count = 4;
    
    double result = sum_f32(data, count);
    assert(std::abs(result - 12.0) < 0.001);
    
    std::cout << "PASSED (sum=" << result << ")\n";
}

void test_min_max_i32() {
    std::cout << "Testing min/max_i32... ";
    
    alignas(128) int32_t data[] = {5, 2, 8, 1, 9, 3, 7, 4, 6, 10};
    size_t count = 10;
    
    int32_t min_val = min_i32(data, count);
    int32_t max_val = max_i32(data, count);
    
    assert(min_val == 1);
    assert(max_val == 10);
    
    std::cout << "PASSED (min=" << min_val << ", max=" << max_val << ")\n";
}

void test_avg_i32() {
    std::cout << "Testing avg_i32... ";
    
    alignas(128) int32_t data[] = {10, 20, 30, 40, 50};
    size_t count = 5;
    
    double result = avg_i32(data, count);
    assert(std::abs(result - 30.0) < 0.001);
    
    std::cout << "PASSED (avg=" << result << ")\n";
}

void test_group_sum() {
    std::cout << "Testing group_sum_i32... ";
    
    // 值数组
    alignas(128) int32_t values[] = {10, 20, 30, 40, 50, 60};
    // 分组 ID
    alignas(128) uint32_t groups[] = {0, 1, 0, 1, 0, 1};
    size_t count = 6;
    size_t num_groups = 2;
    
    std::vector<int64_t> sums(num_groups);
    group_sum_i32(values, groups, count, num_groups, sums.data());
    
    // 组 0: 10 + 30 + 50 = 90
    // 组 1: 20 + 40 + 60 = 120
    assert(sums[0] == 90);
    assert(sums[1] == 120);
    
    std::cout << "PASSED (group0=" << sums[0] << ", group1=" << sums[1] << ")\n";
}

void test_group_count() {
    std::cout << "Testing group_count... ";
    
    alignas(128) uint32_t groups[] = {0, 1, 2, 0, 1, 0, 2, 2};
    size_t count = 8;
    size_t num_groups = 3;
    
    std::vector<size_t> counts(num_groups);
    group_count(groups, count, num_groups, counts.data());
    
    // 组 0: 3 次
    // 组 1: 2 次
    // 组 2: 3 次
    assert(counts[0] == 3);
    assert(counts[1] == 2);
    assert(counts[2] == 3);
    
    std::cout << "PASSED\n";
}

void benchmark_sum_i32() {
    std::cout << "Benchmarking sum_i32... ";
    
    const size_t N = 10000000;
    std::vector<int32_t> data(N);
    
    // 生成随机数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    
    // 预热
    sum_i32(data.data(), N);
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    volatile int64_t result = 0;
    for (int i = 0; i < iterations; ++i) {
        result = sum_i32(data.data(), N);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ms_per_iter = duration.count() / 1000.0 / iterations;
    double elements_per_sec = N / (ms_per_iter / 1000.0);
    double bandwidth_gb = (N * sizeof(int32_t)) / (ms_per_iter / 1000.0) / 1e9;
    
    std::cout << "DONE\n";
    std::cout << "  " << ms_per_iter << " ms per iteration\n";
    std::cout << "  " << (elements_per_sec / 1e6) << " M elements/sec\n";
    std::cout << "  " << bandwidth_gb << " GB/s memory bandwidth\n";
}

int main() {
    std::cout << "=== ThunderDuck Aggregation Tests ===\n\n";
    
    thunderduck::initialize();
    
    // 功能测试
    test_sum_i32();
    test_sum_f32();
    test_min_max_i32();
    test_avg_i32();
    test_group_sum();
    test_group_count();
    
    std::cout << "\n";
    
    // 性能测试
    benchmark_sum_i32();
    
    std::cout << "\n=== All tests passed! ===\n";
    
    thunderduck::shutdown();
    return 0;
}
