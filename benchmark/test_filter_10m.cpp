/**
 * 10M Filter 性能测试
 * 验证 GPU 高带宽过滤
 */

#include "thunderduck/filter.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

using namespace thunderduck::filter;

constexpr size_t DATA_SIZE = 10000000;  // 10M
constexpr int ITERATIONS = 5;
constexpr int WARMUP = 2;

double measure_bandwidth(size_t data_size, double time_ms) {
    // 读取 data_size 个 int32_t + 写入约一半的 uint32_t 索引
    double bytes = data_size * sizeof(int32_t) + (data_size / 2) * sizeof(uint32_t);
    return bytes / (time_ms / 1000.0) / (1e9);  // GB/s
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║         10M Filter Performance Test                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // 生成测试数据
    std::cout << "Generating " << DATA_SIZE / 1000000 << "M test data...\n";
    std::vector<int32_t> data(DATA_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 100);
    
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        data[i] = dist(rng);
    }
    
    // 页对齐的输出缓冲区
    std::vector<uint32_t> out_indices(DATA_SIZE);
    
    std::cout << "Data ready. Testing filter operations...\n\n";

    // 测试不同版本
    struct TestCase {
        const char* name;
        std::function<size_t()> func;
    };

    int32_t threshold = 50;  // > 50: 约 50% 选择率
    
    TestCase tests[] = {
        {"v3 (SIMD)", [&]() {
            return filter_i32_v3(data.data(), DATA_SIZE, CompareOp::GT, threshold, out_indices.data());
        }},
        {"v4 (AUTO)", [&]() {
            return filter_i32_v4(data.data(), DATA_SIZE, CompareOp::GT, threshold, out_indices.data());
        }},
        {"v5 (Multi-thread)", [&]() {
            return filter_i32_v5(data.data(), DATA_SIZE, CompareOp::GT, threshold, out_indices.data());
        }},
        {"GPU Bandwidth", [&]() {
            return filter_i32_gpu_bandwidth(data.data(), DATA_SIZE, CompareOp::GT, threshold, out_indices.data());
        }},
    };

    std::cout << "┌────────────────────┬────────────┬────────────┬────────────┐\n";
    std::cout << "│ Version            │ Time (ms)  │ Matches    │ GB/s       │\n";
    std::cout << "├────────────────────┼────────────┼────────────┼────────────┤\n";

    double v3_time = 0;
    
    for (auto& test : tests) {
        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            test.func();
        }
        
        // Measure
        double total_time = 0;
        size_t matches = 0;
        
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            matches = test.func();
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        
        double avg_time = total_time / ITERATIONS;
        double bandwidth = measure_bandwidth(DATA_SIZE, avg_time);
        
        if (strcmp(test.name, "v3 (SIMD)") == 0) {
            v3_time = avg_time;
        }
        
        std::cout << "│ " << std::setw(18) << std::left << test.name
                  << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << matches
                  << " │ " << std::setw(10) << std::setprecision(1) << bandwidth
                  << " │\n";
    }

    std::cout << "└────────────────────┴────────────┴────────────┴────────────┘\n";

    std::cout << "\n理论内存带宽: ~400 GB/s (M4 Max)\n";
    std::cout << "目标: 200-300 GB/s (50-75% 利用率)\n";

    return 0;
}
