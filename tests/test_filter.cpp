/**
 * ThunderDuck - Filter Operator Tests
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/filter.h"
#include "thunderduck/memory.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>

using namespace thunderduck;
using namespace thunderduck::filter;

void test_filter_gt_i32() {
    std::cout << "Testing filter_i32 (GT)... ";
    
    // 测试数据
    alignas(128) int32_t data[] = {1, 5, 3, 8, 2, 9, 4, 7, 6, 10};
    size_t count = 10;
    
    std::vector<uint32_t> indices(count);
    
    // 过滤 > 5
    size_t result_count = filter_i32(data, count, CompareOp::GT, 5, indices.data());
    
    // data: 1,5,3,8,2,9,4,7,6,10
    // > 5: 8(idx=3), 9(idx=5), 7(idx=7), 6(idx=8), 10(idx=9) = 5 个
    assert(result_count == 5);
    
    std::cout << "PASSED (found " << result_count << " elements)\n";
}

void test_filter_eq_i32() {
    std::cout << "Testing filter_i32 (EQ)... ";
    
    alignas(128) int32_t data[] = {1, 2, 3, 2, 5, 2, 7, 8};
    size_t count = 8;
    
    std::vector<uint32_t> indices(count);
    
    // 过滤 == 2
    size_t result_count = filter_i32(data, count, CompareOp::EQ, 2, indices.data());
    
    assert(result_count == 3);  // 索引 1, 3, 5
    assert(indices[0] == 1);
    assert(indices[1] == 3);
    assert(indices[2] == 5);
    
    std::cout << "PASSED\n";
}

void test_filter_range() {
    std::cout << "Testing filter_i32_range... ";
    
    alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    size_t count = 10;
    
    std::vector<uint32_t> indices(count);
    
    // 过滤 3 <= x < 7
    size_t result_count = filter_i32_range(data, count, 3, 7, indices.data());
    
    assert(result_count == 4);  // 3, 4, 5, 6
    
    std::cout << "PASSED\n";
}

void test_count_i32() {
    std::cout << "Testing count_i32... ";
    
    alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t count = 12;
    
    // 计数 > 6
    size_t result = count_i32(data, count, CompareOp::GT, 6);
    
    assert(result == 6);  // 7,8,9,10,11,12
    
    std::cout << "PASSED\n";
}

void benchmark_filter_i32() {
    std::cout << "Benchmarking filter_i32... ";
    
    const size_t N = 1000000;
    std::vector<int32_t> data(N);
    std::vector<uint32_t> indices(N);
    
    // 生成随机数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    
    // 预热
    filter_i32(data.data(), N, CompareOp::GT, 500, indices.data());
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    for (int i = 0; i < iterations; ++i) {
        filter_i32(data.data(), N, CompareOp::GT, 500, indices.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ms_per_iter = duration.count() / 1000.0 / iterations;
    double elements_per_sec = N / (ms_per_iter / 1000.0);
    
    std::cout << "DONE\n";
    std::cout << "  " << ms_per_iter << " ms per iteration\n";
    std::cout << "  " << (elements_per_sec / 1e6) << " M elements/sec\n";
}

int main() {
    std::cout << "=== ThunderDuck Filter Tests ===\n\n";
    
    // 初始化
    thunderduck::initialize();
    
    // 功能测试
    test_filter_gt_i32();
    test_filter_eq_i32();
    test_filter_range();
    test_count_i32();
    
    std::cout << "\n";
    
    // 性能测试
    benchmark_filter_i32();
    
    std::cout << "\n=== All tests passed! ===\n";
    
    thunderduck::shutdown();
    return 0;
}
