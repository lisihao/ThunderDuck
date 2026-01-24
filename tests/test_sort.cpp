/**
 * ThunderDuck - Sort Operator Tests
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/sort.h"
#include "thunderduck/memory.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>

using namespace thunderduck;
using namespace thunderduck::sort;

void test_sort_4_i32() {
    std::cout << "Testing sort_4_i32... ";
    
    alignas(16) int32_t data[] = {4, 2, 3, 1};
    sort_4_i32(data, SortOrder::ASC);
    
    assert(data[0] == 1);
    assert(data[1] == 2);
    assert(data[2] == 3);
    assert(data[3] == 4);
    
    std::cout << "PASSED\n";
}

void test_sort_8_i32() {
    std::cout << "Testing sort_8_i32... ";
    
    alignas(16) int32_t data[] = {8, 3, 5, 1, 7, 2, 6, 4};
    sort_8_i32(data, SortOrder::ASC);
    
    for (int i = 0; i < 7; ++i) {
        assert(data[i] <= data[i + 1]);
    }
    
    std::cout << "PASSED\n";
}

void test_sort_i32() {
    std::cout << "Testing sort_i32... ";
    
    std::vector<int32_t> data = {9, 3, 7, 1, 8, 2, 6, 4, 5, 10};
    sort_i32(data.data(), data.size(), SortOrder::ASC);
    
    for (size_t i = 0; i < data.size() - 1; ++i) {
        assert(data[i] <= data[i + 1]);
    }
    
    // 测试降序
    sort_i32(data.data(), data.size(), SortOrder::DESC);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        assert(data[i] >= data[i + 1]);
    }
    
    std::cout << "PASSED\n";
}

void test_argsort_i32() {
    std::cout << "Testing argsort_i32... ";
    
    int32_t data[] = {30, 10, 40, 20};
    std::vector<uint32_t> indices(4);
    
    argsort_i32(data, 4, indices.data(), SortOrder::ASC);
    
    // 排序后顺序应该是: 10(idx=1), 20(idx=3), 30(idx=0), 40(idx=2)
    assert(indices[0] == 1);
    assert(indices[1] == 3);
    assert(indices[2] == 0);
    assert(indices[3] == 2);
    
    std::cout << "PASSED\n";
}

void test_topk_min() {
    std::cout << "Testing topk_min_i32... ";
    
    int32_t data[] = {50, 20, 80, 10, 30, 90, 40, 60, 70};
    size_t count = 9;
    size_t k = 3;
    
    std::vector<int32_t> values(k);
    std::vector<uint32_t> indices(k);
    
    topk_min_i32(data, count, k, values.data(), indices.data());
    
    // 最小的 3 个: 10, 20, 30
    assert(values[0] == 10);
    assert(values[1] == 20);
    assert(values[2] == 30);
    
    std::cout << "PASSED\n";
}

void test_topk_max() {
    std::cout << "Testing topk_max_i32... ";
    
    int32_t data[] = {50, 20, 80, 10, 30, 90, 40, 60, 70};
    size_t count = 9;
    size_t k = 3;
    
    std::vector<int32_t> values(k);
    std::vector<uint32_t> indices(k);
    
    topk_max_i32(data, count, k, values.data(), indices.data());
    
    // 最大的 3 个: 90, 80, 70
    assert(values[0] == 90);
    assert(values[1] == 80);
    assert(values[2] == 70);
    
    std::cout << "PASSED\n";
}

void benchmark_sort_i32() {
    std::cout << "Benchmarking sort_i32... ";
    
    const size_t N = 1000000;
    std::vector<int32_t> data(N);
    std::vector<int32_t> data_copy(N);
    
    // 生成随机数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000000);
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    
    // 预热
    data_copy = data;
    sort_i32(data_copy.data(), N, SortOrder::ASC);
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 10;
    
    for (int i = 0; i < iterations; ++i) {
        data_copy = data;
        sort_i32(data_copy.data(), N, SortOrder::ASC);
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
    std::cout << "=== ThunderDuck Sort Tests ===\n\n";
    
    thunderduck::initialize();
    
    // 功能测试
    test_sort_4_i32();
    test_sort_8_i32();
    test_sort_i32();
    test_argsort_i32();
    test_topk_min();
    test_topk_max();
    
    std::cout << "\n";
    
    // 性能测试
    benchmark_sort_i32();
    
    std::cout << "\n=== All tests passed! ===\n";
    
    thunderduck::shutdown();
    return 0;
}
