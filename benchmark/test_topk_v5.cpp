/**
 * TopK v5.0 Comprehensive Benchmark
 *
 * Tests v3 vs v4 vs v5 at different scales and cardinalities
 */

#include "thunderduck/sort.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>

using namespace thunderduck::sort;

constexpr int WARMUP = 2;
constexpr int ITERATIONS = 5;

struct TestConfig {
    const char* name;
    size_t count;
    size_t k;
    size_t unique_values;  // 0 = fully random (high cardinality)
};

void benchmark_topk(const TestConfig& config) {
    std::cout << "\n=== " << config.name << " ===" << std::endl;
    std::cout << "N=" << config.count << ", K=" << config.k;
    if (config.unique_values > 0) {
        std::cout << ", Cardinality=" << config.unique_values;
    } else {
        std::cout << ", Cardinality=HIGH";
    }
    std::cout << std::endl;

    // Allocate page-aligned memory
    void* data_raw = nullptr;
    posix_memalign(&data_raw, 16384, config.count * sizeof(int32_t));
    int32_t* data = static_cast<int32_t*>(data_raw);

    // Generate data
    std::mt19937 rng(42);
    if (config.unique_values > 0) {
        // Low cardinality: limited unique values
        std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(config.unique_values - 1));
        for (size_t i = 0; i < config.count; ++i) {
            data[i] = dist(rng);
        }
    } else {
        // High cardinality: fully random
        std::uniform_int_distribution<int32_t> dist(-1000000000, 1000000000);
        for (size_t i = 0; i < config.count; ++i) {
            data[i] = dist(rng);
        }
    }

    // Allocate result buffers
    std::vector<int32_t> out_values(config.k);
    std::vector<uint32_t> out_indices(config.k);

    std::cout << "┌────────────┬────────────┬────────────┐" << std::endl;
    std::cout << "│ Version    │ Time (ms)  │ vs v3      │" << std::endl;
    std::cout << "├────────────┼────────────┼────────────┤" << std::endl;

    double v3_time = 0;

    // v3 baseline
    {
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v3(data, config.count, config.k, out_values.data(), out_indices.data());
        }

        double total = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            topk_max_i32_v3(data, config.count, config.k, out_values.data(), out_indices.data());
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        v3_time = total / ITERATIONS;

        std::cout << "│ v3         │ " << std::setw(10) << std::fixed << std::setprecision(2) << v3_time
                  << " │ " << std::setw(10) << "1.00x" << " │" << std::endl;
    }

    // v4 sampling prefilter
    {
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v4(data, config.count, config.k, out_values.data(), out_indices.data());
        }

        double total = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            topk_max_i32_v4(data, config.count, config.k, out_values.data(), out_indices.data());
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg = total / ITERATIONS;

        std::cout << "│ v4         │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg
                  << " │ " << std::setw(9) << std::setprecision(2) << v3_time / avg << "x │" << std::endl;
    }

    // v5 adaptive (count-based for low cardinality)
    {
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v5(data, config.count, config.k, out_values.data(), out_indices.data());
        }

        double total = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            topk_max_i32_v5(data, config.count, config.k, out_values.data(), out_indices.data());
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg = total / ITERATIONS;

        std::cout << "│ v5         │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg
                  << " │ " << std::setw(9) << std::setprecision(2) << v3_time / avg << "x │" << std::endl;
    }

    // v6 GPU (if available)
    if (is_topk_gpu_available()) {
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v6(data, config.count, config.k, out_values.data(), out_indices.data());
        }

        double total = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            topk_max_i32_v6(data, config.count, config.k, out_values.data(), out_indices.data());
            auto end = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg = total / ITERATIONS;

        std::cout << "│ v6 GPU     │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg
                  << " │ " << std::setw(9) << std::setprecision(2) << v3_time / avg << "x │" << std::endl;
    }

    std::cout << "└────────────┴────────────┴────────────┘" << std::endl;

    free(data_raw);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         TopK v5.0 Comprehensive Benchmark                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    // Test cases
    std::vector<TestConfig> tests = {
        // 1M scale - avoid regression
        {"1M High Cardinality", 1000000, 10, 0},
        {"1M Low Cardinality (100)", 1000000, 10, 100},

        // 5M scale - count-based threshold
        {"5M High Cardinality", 5000000, 10, 0},
        {"5M Low Cardinality (100)", 5000000, 10, 100},
        {"5M Low Cardinality (1000)", 5000000, 10, 1000},

        // 10M scale - main optimization target
        {"10M High Cardinality", 10000000, 10, 0},
        {"10M Low Cardinality (100)", 10000000, 10, 100},
        {"10M Low Cardinality (1000)", 10000000, 10, 1000},
        {"10M Medium Cardinality (10000)", 10000000, 10, 10000},
    };

    for (const auto& test : tests) {
        benchmark_topk(test);
    }

    std::cout << "\n目标:" << std::endl;
    std::cout << "- 1M: v5 不回退 (>= 1.0x vs v3)" << std::endl;
    std::cout << "- 10M Low Cardinality: v5 显著提升 (>= 2.0x vs v3)" << std::endl;

    return 0;
}
