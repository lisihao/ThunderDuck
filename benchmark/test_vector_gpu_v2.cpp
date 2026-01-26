/**
 * Vector Similarity GPU v2.0 Benchmark
 *
 * Tests GPU float4 vectorization and zero-copy optimizations
 * Focus: Memory bandwidth utilization at large batches (100K+)
 */

#include "thunderduck/vector_ops.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdlib>

using namespace thunderduck;
using namespace std::chrono;

constexpr int WARMUP = 3;
constexpr int ITERATIONS = 5;

struct TestCase {
    const char* name;
    size_t dim;
    size_t num_candidates;
};

double calculate_bandwidth_gbps(size_t bytes, double time_us) {
    // bytes / time_us = bytes per microsecond
    // Convert to GB/s: (bytes / time_us) * 1e6 / 1e9 = bytes / time_us * 1e-3
    return (bytes / time_us) * 1e-3;
}

void benchmark_batch_dot_product(const TestCase& test) {
    std::cout << "\n=== " << test.name << " ===" << std::endl;
    std::cout << "dim=" << test.dim << ", candidates=" << test.num_candidates << std::endl;

    // Allocate page-aligned memory for zero-copy GPU access
    void* query_raw = nullptr;
    void* candidates_raw = nullptr;
    void* scores_raw = nullptr;

    posix_memalign(&query_raw, 16384, test.dim * sizeof(float));
    posix_memalign(&candidates_raw, 16384, test.num_candidates * test.dim * sizeof(float));
    posix_memalign(&scores_raw, 16384, test.num_candidates * sizeof(float));

    float* query = static_cast<float*>(query_raw);
    float* candidates = static_cast<float*>(candidates_raw);
    float* scores = static_cast<float*>(scores_raw);

    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < test.dim; ++i) {
        query[i] = dist(rng);
    }
    for (size_t i = 0; i < test.num_candidates * test.dim; ++i) {
        candidates[i] = dist(rng);
    }

    // Calculate data sizes
    size_t candidates_bytes = test.num_candidates * test.dim * sizeof(float);
    size_t total_bytes = test.dim * sizeof(float) + candidates_bytes + test.num_candidates * sizeof(float);

    std::cout << "Data size: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    std::cout << "┌─────────────┬────────────┬────────────┬────────────┐" << std::endl;
    std::cout << "│ Method      │ Time (us)  │ BW (GB/s)  │ vs AMX     │" << std::endl;
    std::cout << "├─────────────┼────────────┼────────────┼────────────┤" << std::endl;

    double amx_time = 0;

    // AMX/BLAS (baseline) - use batch_dot_product_f32 which uses AMX by default
    {
        for (int i = 0; i < WARMUP; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }
        auto end = high_resolution_clock::now();
        amx_time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;
        double bw = calculate_bandwidth_gbps(candidates_bytes, amx_time);

        std::cout << "│ AMX/BLAS    │ " << std::setw(10) << std::fixed << std::setprecision(1) << amx_time
                  << " │ " << std::setw(10) << std::setprecision(1) << bw
                  << " │ " << std::setw(10) << "1.00x" << " │" << std::endl;
    }

    // GPU Metal
    if (vector::gpu::is_gpu_vector_ready()) {
        for (int i = 0; i < WARMUP; ++i) {
            vector::gpu::batch_dot_product_gpu(query, candidates, test.dim, test.num_candidates, scores);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            vector::gpu::batch_dot_product_gpu(query, candidates, test.dim, test.num_candidates, scores);
        }
        auto end = high_resolution_clock::now();
        double time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;
        double bw = calculate_bandwidth_gbps(candidates_bytes, time);

        std::cout << "│ GPU Metal   │ " << std::setw(10) << std::fixed << std::setprecision(1) << time
                  << " │ " << std::setw(10) << std::setprecision(1) << bw
                  << " │ " << std::setw(9) << std::setprecision(2) << amx_time / time << "x │" << std::endl;
    }

    std::cout << "└─────────────┴────────────┴────────────┴────────────┘" << std::endl;

    free(query_raw);
    free(candidates_raw);
    free(scores_raw);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         Vector Similarity GPU v2.0 Benchmark                     ║" << std::endl;
    std::cout << "║         Focus: Memory Bandwidth at Large Batches                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\nGPU Available: " << (vector::gpu::is_gpu_vector_ready() ? "Yes" : "No") << std::endl;
    std::cout << "AMX Available: " << (vector::is_amx_available() ? "Yes" : "No") << std::endl;

    std::vector<TestCase> tests = {
        // Small batches (AMX optimal)
        {"Small (10K, dim=256)", 256, 10000},

        // Medium batches (transition zone)
        {"Medium (50K, dim=256)", 256, 50000},

        // Large batches (GPU should excel)
        {"Large (100K, dim=256)", 256, 100000},
        {"Large (100K, dim=512)", 512, 100000},
        {"Large (100K, dim=128)", 128, 100000},

        // Very large batches (GPU optimal)
        {"XLarge (500K, dim=256)", 256, 500000},
        {"XLarge (1M, dim=128)", 128, 1000000},
    };

    for (const auto& test : tests) {
        benchmark_batch_dot_product(test);
    }

    std::cout << "\n目标:" << std::endl;
    std::cout << "- Large batches (100K+): GPU 带宽 > 150 GB/s" << std::endl;
    std::cout << "- GPU vs AMX at 100K: > 1.5x speedup" << std::endl;
    std::cout << "- 理论 M4 UMA 带宽: ~400 GB/s" << std::endl;

    return 0;
}
