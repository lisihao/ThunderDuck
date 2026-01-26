/**
 * 10M Filter Debug 测试
 */

#include "thunderduck/filter.h"
#include "thunderduck/uma_memory.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>

using namespace thunderduck::filter;

constexpr size_t DATA_SIZE = 10000000;  // 10M
constexpr int ITERATIONS = 5;
constexpr int WARMUP = 2;

double measure_bandwidth(size_t data_size, double time_ms) {
    double bytes = data_size * sizeof(int32_t) + (data_size / 2) * sizeof(uint32_t);
    return bytes / (time_ms / 1000.0) / (1e9);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║    10M Filter Debug Test (Page-Aligned)              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // 页对齐分配
    void* data_raw = nullptr;
    void* out_raw = nullptr;
    posix_memalign(&data_raw, 16384, DATA_SIZE * sizeof(int32_t));
    posix_memalign(&out_raw, 16384, DATA_SIZE * sizeof(uint32_t));
    
    int32_t* data = static_cast<int32_t*>(data_raw);
    uint32_t* out_indices = static_cast<uint32_t*>(out_raw);
    
    std::cout << "Data pointer: " << data << " (page aligned: " 
              << thunderduck::uma::is_page_aligned(data) << ")\n";
    std::cout << "Output pointer: " << out_indices << " (page aligned: " 
              << thunderduck::uma::is_page_aligned(out_indices) << ")\n\n";

    // 生成测试数据
    std::cout << "Generating " << DATA_SIZE / 1000000 << "M test data...\n";
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, 100);
    
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        data[i] = dist(rng);
    }

    std::cout << "GPU Filter available: " << is_filter_gpu_available() << "\n\n";

    int32_t threshold = 50;

    // 测试 v3 (baseline)
    double v3_time = 0;
    size_t v3_matches = 0;
    for (int i = 0; i < WARMUP; ++i) {
        filter_i32_v3(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
    }
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        v3_matches = filter_i32_v3(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
        auto end = std::chrono::high_resolution_clock::now();
        v3_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    v3_time /= ITERATIONS;
    
    std::cout << "v3 SIMD:         " << std::fixed << std::setprecision(2) 
              << v3_time << " ms, " << v3_matches << " matches, "
              << measure_bandwidth(DATA_SIZE, v3_time) << " GB/s\n";

    // 测试 v5 (multi-thread)
    double v5_time = 0;
    size_t v5_matches = 0;
    for (int i = 0; i < WARMUP; ++i) {
        filter_i32_v5(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
    }
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        v5_matches = filter_i32_v5(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
        auto end = std::chrono::high_resolution_clock::now();
        v5_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    v5_time /= ITERATIONS;
    
    std::cout << "v5 Multi-thread: " << std::fixed << std::setprecision(2) 
              << v5_time << " ms, " << v5_matches << " matches, "
              << measure_bandwidth(DATA_SIZE, v5_time) << " GB/s\n";

    // 测试 GPU bandwidth
    double gpu_time = 0;
    size_t gpu_matches = 0;
    for (int i = 0; i < WARMUP; ++i) {
        filter_i32_gpu_bandwidth(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
    }
    for (int i = 0; i < ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        gpu_matches = filter_i32_gpu_bandwidth(data, DATA_SIZE, CompareOp::GT, threshold, out_indices);
        auto end = std::chrono::high_resolution_clock::now();
        gpu_time += std::chrono::duration<double, std::milli>(end - start).count();
    }
    gpu_time /= ITERATIONS;
    
    std::cout << "GPU Bandwidth:   " << std::fixed << std::setprecision(2) 
              << gpu_time << " ms, " << gpu_matches << " matches, "
              << measure_bandwidth(DATA_SIZE, gpu_time) << " GB/s\n";

    std::cout << "\nSpeedup v5 vs v3: " << v3_time / v5_time << "x\n";
    std::cout << "Speedup GPU vs v3: " << v3_time / gpu_time << "x\n";
    
    std::cout << "\n理论带宽: ~400 GB/s (M4 Max)\n";
    std::cout << "实际带宽: v5 达到 " << measure_bandwidth(DATA_SIZE, v5_time) / 400 * 100 << "% 利用率\n";

    free(data_raw);
    free(out_raw);
    return 0;
}
