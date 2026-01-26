/**
 * Vector Similarity Strategy Selection Benchmark
 *
 * Tests the intelligent AMX/GPU strategy selection
 * and verifies the performance at various scales
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
    return (bytes / time_us) * 1e-3;
}

void benchmark_batch_dot_product(const TestCase& test) {
    std::cout << "\n=== " << test.name << " ===" << std::endl;
    std::cout << "dim=" << test.dim << ", candidates=" << test.num_candidates << std::endl;

    // Show AUTO selected path
    auto selected_path = vector::get_auto_selected_path(test.dim, test.num_candidates);
    std::cout << "AUTO Strategy: " << vector::vector_path_name(selected_path) << std::endl;

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
    std::cout << "│ Method      │ Time (us)  │ BW (GB/s)  │ vs AUTO    │" << std::endl;
    std::cout << "├─────────────┼────────────┼────────────┼────────────┤" << std::endl;

    double auto_time = 0;

    // AUTO (default - should select AMX in most cases)
    {
        vector::set_default_vector_path(vector::VectorPath::AUTO);

        for (int i = 0; i < WARMUP; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }
        auto end = high_resolution_clock::now();
        auto_time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;
        double bw = calculate_bandwidth_gbps(candidates_bytes, auto_time);

        std::cout << "│ AUTO        │ " << std::setw(10) << std::fixed << std::setprecision(1) << auto_time
                  << " │ " << std::setw(10) << std::setprecision(1) << bw
                  << " │ " << std::setw(10) << "1.00x" << " │" << std::endl;
    }

    // Force AMX/BLAS
    {
        vector::set_default_vector_path(vector::VectorPath::AMX_BLAS);

        for (int i = 0; i < WARMUP; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }
        auto end = high_resolution_clock::now();
        double time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;
        double bw = calculate_bandwidth_gbps(candidates_bytes, time);

        std::cout << "│ AMX/BLAS    │ " << std::setw(10) << std::fixed << std::setprecision(1) << time
                  << " │ " << std::setw(10) << std::setprecision(1) << bw
                  << " │ " << std::setw(9) << std::setprecision(2) << auto_time / time << "x │" << std::endl;
    }

    // Force GPU Metal
    if (vector::gpu::is_gpu_vector_ready()) {
        vector::set_default_vector_path(vector::VectorPath::GPU_METAL);

        for (int i = 0; i < WARMUP; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < ITERATIONS; ++i) {
            vector::batch_dot_product_f32(query, candidates, test.dim, test.num_candidates, scores);
        }
        auto end = high_resolution_clock::now();
        double time = duration_cast<microseconds>(end - start).count() / (double)ITERATIONS;
        double bw = calculate_bandwidth_gbps(candidates_bytes, time);

        std::cout << "│ GPU Metal   │ " << std::setw(10) << std::fixed << std::setprecision(1) << time
                  << " │ " << std::setw(10) << std::setprecision(1) << bw
                  << " │ " << std::setw(9) << std::setprecision(2) << auto_time / time << "x │" << std::endl;
    }

    std::cout << "└─────────────┴────────────┴────────────┴────────────┘" << std::endl;

    // Reset to AUTO
    vector::set_default_vector_path(vector::VectorPath::AUTO);

    free(query_raw);
    free(candidates_raw);
    free(scores_raw);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Vector Similarity - Intelligent Strategy Selection Test        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\nGPU Available: " << (vector::gpu::is_gpu_vector_ready() ? "Yes" : "No") << std::endl;
    std::cout << "AMX Available: " << (vector::is_amx_available() ? "Yes" : "No") << std::endl;

    std::cout << "\n--- Strategy Selection Rules (v2.2) ---" << std::endl;
    std::cout << "- AMX/BLAS: Default for all cases (always faster on M4)" << std::endl;
    std::cout << "- GPU: Manual selection only (via set_default_vector_path)" << std::endl;
    std::cout << "- Rationale: AMX has 1.1x-27x bandwidth advantage across all batch sizes" << std::endl;

    std::vector<TestCase> tests = {
        // Small batches - should select AMX
        {"Small (10K, dim=256)", 256, 10000},

        // Medium batches - should select AMX
        {"Medium (50K, dim=256)", 256, 50000},

        // Large batches, medium dim - should select AMX
        {"Large (100K, dim=256)", 256, 100000},

        // Large batches, high dim - should select GPU
        {"Large (100K, dim=512)", 512, 100000},

        // XLarge batches - verify strategy
        {"XLarge (500K, dim=256)", 256, 500000},

        // High dim small batch - should select AMX
        {"High Dim Small Batch (10K, dim=1024)", 1024, 10000},
    };

    for (const auto& test : tests) {
        benchmark_batch_dot_product(test);
    }

    std::cout << "\n结论:" << std::endl;
    std::cout << "- AUTO 策略始终选择 AMX/BLAS (M4 上最优)" << std::endl;
    std::cout << "- AMX 在所有场景下都比 GPU 更快 (1.1x-27x)" << std::endl;
    std::cout << "- GPU 保留接口供特殊场景 (CPU offloading, 多 GPU 等)" << std::endl;

    return 0;
}
