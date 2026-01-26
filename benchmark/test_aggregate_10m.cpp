/**
 * 10M Aggregate 性能测试
 * 验证多线程 + vDSP 优化
 */

#include "thunderduck/aggregate.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <thread>

using namespace thunderduck::aggregate;

constexpr size_t DATA_SIZE = 10000000;  // 10M
constexpr int ITERATIONS = 5;
constexpr int WARMUP = 2;

double measure_bandwidth(size_t data_size, double time_ms, size_t elem_size) {
    double bytes = data_size * elem_size;
    return bytes / (time_ms / 1000.0) / (1e9);  // GB/s
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║         10M Aggregate Performance Test               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // 页对齐分配
    void* i32_raw = nullptr;
    void* f32_raw = nullptr;
    posix_memalign(&i32_raw, 16384, DATA_SIZE * sizeof(int32_t));
    posix_memalign(&f32_raw, 16384, DATA_SIZE * sizeof(float));

    int32_t* data_i32 = static_cast<int32_t*>(i32_raw);
    float* data_f32 = static_cast<float*>(f32_raw);

    // 生成测试数据
    std::cout << "Generating " << DATA_SIZE / 1000000 << "M test data...\n";
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist_i32(0, 1000);
    std::uniform_real_distribution<float> dist_f32(0.0f, 1000.0f);

    for (size_t i = 0; i < DATA_SIZE; ++i) {
        data_i32[i] = dist_i32(rng);
        data_f32[i] = dist_f32(rng);
    }

    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "GPU Aggregate available: " << is_aggregate_gpu_available() << "\n\n";

    // ========================================================================
    // INT32 SUM 测试
    // ========================================================================
    std::cout << "=== INT32 SUM Tests ===\n";
    std::cout << "┌────────────────────┬────────────┬────────────┬────────────┐\n";
    std::cout << "│ Version            │ Time (ms)  │ Result     │ GB/s       │\n";
    std::cout << "├────────────────────┼────────────┼────────────┼────────────┤\n";

    // v2 SIMD (baseline)
    {
        for (int i = 0; i < WARMUP; ++i) sum_i32_v2(data_i32, DATA_SIZE);
        double total_time = 0;
        int64_t result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = sum_i32_v2(data_i32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v2 (SIMD)          │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << result / 1000000 << "M"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // v3 Parallel
    {
        for (int i = 0; i < WARMUP; ++i) sum_i32_v3(data_i32, DATA_SIZE);
        double total_time = 0;
        int64_t result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = sum_i32_v3(data_i32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v3 (Parallel SIMD) │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << result / 1000000 << "M"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // vDSP (single-thread)
    {
        for (int i = 0; i < WARMUP; ++i) vdsp_sum_i32(data_i32, DATA_SIZE);
        double total_time = 0;
        int64_t result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = vdsp_sum_i32(data_i32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ vDSP (Single-th)   │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << result / 1000000 << "M"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    std::cout << "└────────────────────┴────────────┴────────────┴────────────┘\n\n";

    // ========================================================================
    // FLOAT32 SUM 测试
    // ========================================================================
    std::cout << "=== FLOAT32 SUM Tests ===\n";
    std::cout << "┌────────────────────┬────────────┬────────────┬────────────┐\n";
    std::cout << "│ Version            │ Time (ms)  │ Result     │ GB/s       │\n";
    std::cout << "├────────────────────┼────────────┼────────────┼────────────┤\n";

    // v2 SIMD (baseline)
    {
        for (int i = 0; i < WARMUP; ++i) sum_f32_v2(data_f32, DATA_SIZE);
        double total_time = 0;
        double result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = sum_f32_v2(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v2 (SIMD)          │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // v3 Parallel
    {
        for (int i = 0; i < WARMUP; ++i) sum_f32_v3(data_f32, DATA_SIZE);
        double total_time = 0;
        double result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = sum_f32_v3(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v3 (Parallel SIMD) │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // vDSP (single-thread)
    {
        for (int i = 0; i < WARMUP; ++i) vdsp_sum_f32(data_f32, DATA_SIZE);
        double total_time = 0;
        float result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = vdsp_sum_f32(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ vDSP (Single-th)   │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // vDSP Parallel (multi-thread + AMX)
    {
        for (int i = 0; i < WARMUP; ++i) vdsp_sum_f32_parallel(data_f32, DATA_SIZE);
        double total_time = 0;
        double result = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = vdsp_sum_f32_parallel(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ vDSP Parallel      │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    std::cout << "└────────────────────┴────────────┴────────────┴────────────┘\n\n";

    // ========================================================================
    // ALL STATS (SUM/MIN/MAX) 测试
    // ========================================================================
    std::cout << "=== ALL STATS (SUM/MIN/MAX) Tests ===\n";
    std::cout << "┌────────────────────┬────────────┬────────────┬────────────┐\n";
    std::cout << "│ Version            │ Time (ms)  │ Sum        │ GB/s       │\n";
    std::cout << "├────────────────────┼────────────┼────────────┼────────────┤\n";

    // v2 (baseline)
    {
        for (int i = 0; i < WARMUP; ++i) aggregate_all_i32(data_i32, DATA_SIZE);
        double total_time = 0;
        AggregateStats stats;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            stats = aggregate_all_i32(data_i32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v2 (SIMD)          │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << stats.sum / 1000000 << "M"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // v3 Parallel
    {
        for (int i = 0; i < WARMUP; ++i) aggregate_all_i32_v3(data_i32, DATA_SIZE);
        double total_time = 0;
        AggregateStats stats;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            stats = aggregate_all_i32_v3(data_i32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ v3 (Parallel SIMD) │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << stats.sum / 1000000 << "M"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // vDSP Fused Float (single-thread)
    {
        for (int i = 0; i < WARMUP; ++i) vdsp_fused_aggregate_f32(data_f32, DATA_SIZE);
        double total_time = 0;
        VDSPAggregateResult result;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = vdsp_fused_aggregate_f32(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ vDSP Fused (1-th)  │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result.sum / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    // vDSP Fused Parallel
    {
        for (int i = 0; i < WARMUP; ++i) vdsp_fused_aggregate_f32_parallel(data_f32, DATA_SIZE);
        double total_time = 0;
        VDSPAggregateResult result;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            result = vdsp_fused_aggregate_f32_parallel(data_f32, DATA_SIZE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_time = total_time / ITERATIONS;
        std::cout << "│ vDSP Fused Parallel│ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                  << " │ " << std::setw(10) << std::setprecision(0) << result.sum / 1e9 << "G"
                  << " │ " << std::setw(10) << std::setprecision(1) << measure_bandwidth(DATA_SIZE, avg_time, 4)
                  << " │\n";
    }

    std::cout << "└────────────────────┴────────────┴────────────┴────────────┘\n\n";

    std::cout << "理论内存带宽: ~400 GB/s (M4 Max)\n";
    std::cout << "目标: 100-200 GB/s (25-50% 利用率)\n";

    free(i32_raw);
    free(f32_raw);
    return 0;
}
