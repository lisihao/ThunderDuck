/**
 * Hash Join 高匹配场景性能测试
 * 测试 1M 匹配场景的性能
 */

#include "thunderduck/join.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>

using namespace thunderduck::join;
using namespace thunderduck::join::v6;

constexpr int ITERATIONS = 3;
constexpr int WARMUP = 1;

struct TestCase {
    const char* name;
    size_t build_size;
    size_t probe_size;
    size_t expected_matches;
};

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Hash Join High-Match Performance Test                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // 测试场景: 高匹配 (每个 probe key 匹配多个 build key)
    // 每个 build key 有 10 个重复 (10% 唯一键)
    TestCase tests[] = {
        {"50K build, 50K probe, ~500K match", 50000, 50000, 0},
        {"100K build, 100K probe, ~1M match", 100000, 100000, 0},
        {"200K build, 200K probe, ~2M match", 200000, 200000, 0},
        {"500K build, 500K probe, ~5M match", 500000, 500000, 0},
    };

    for (auto& test : tests) {
        std::cout << "=== " << test.name << " ===\n";

        // 页对齐分配
        void* build_raw = nullptr;
        void* probe_raw = nullptr;
        posix_memalign(&build_raw, 16384, test.build_size * sizeof(int32_t));
        posix_memalign(&probe_raw, 16384, test.probe_size * sizeof(int32_t));

        int32_t* build_keys = static_cast<int32_t*>(build_raw);
        int32_t* probe_keys = static_cast<int32_t*>(probe_raw);

        // 生成高匹配数据: probe keys 从 build keys 中采样
        std::mt19937 rng(42);
        size_t unique_keys = test.build_size / 10;  // 10% 唯一键 = 每个键约 10 个匹配

        for (size_t i = 0; i < test.build_size; ++i) {
            build_keys[i] = static_cast<int32_t>(i % unique_keys);
        }

        std::uniform_int_distribution<size_t> dist(0, unique_keys - 1);
        for (size_t i = 0; i < test.probe_size; ++i) {
            probe_keys[i] = static_cast<int32_t>(dist(rng));
        }

        // 创建结果缓冲区 (预分配更大空间避免重分配)
        // 预估: 每个 probe key 约 10 匹配
        size_t estimated_matches = test.probe_size * 10;
        JoinResult* result = create_join_result(estimated_matches);

        std::cout << "┌────────────────────┬────────────┬────────────┬────────────┐\n";
        std::cout << "│ Version            │ Time (ms)  │ Matches    │ Speedup    │\n";
        std::cout << "├────────────────────┼────────────┼────────────┼────────────┤\n";

        double v3_time = 0;

        // v3 (baseline)
        {
            std::cout << "Starting v3 warmup..." << std::flush;
            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                hash_join_i32_v3(build_keys, test.build_size, probe_keys, test.probe_size,
                                  JoinType::INNER, result);
            }
            std::cout << " done. Starting measurements..." << std::flush;

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = hash_join_i32_v3(build_keys, test.build_size, probe_keys, test.probe_size,
                                            JoinType::INNER, result);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;
            v3_time = avg_time;

            std::cout << "│ v3 (SIMD)          │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << "1.00x"
                      << " │\n";

            test.expected_matches = matches;
        }

        // v4 AUTO
        {
            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                hash_join_i32_v4(build_keys, test.build_size, probe_keys, test.probe_size,
                                  JoinType::INNER, result);
            }

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = hash_join_i32_v4(build_keys, test.build_size, probe_keys, test.probe_size,
                                            JoinType::INNER, result);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;

            std::cout << "│ v4 (AUTO)          │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << std::setprecision(2) << v3_time / avg_time << "x"
                      << " │\n";
        }

        // v5 Two-Phase (direct call)
        {
            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                v5::hash_join_i32_v5_twophase(build_keys, test.build_size, probe_keys, test.probe_size,
                                               JoinType::INNER, result);
            }

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = v5::hash_join_i32_v5_twophase(build_keys, test.build_size, probe_keys, test.probe_size,
                                                         JoinType::INNER, result);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;

            std::cout << "│ v5 Two-Phase       │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << std::setprecision(2) << v3_time / avg_time << "x"
                      << " │\n";
        }

        // v4 with TWOPHASE_CPU strategy
        {
            JoinConfigV4 config;
            config.strategy = JoinStrategy::TWOPHASE_CPU;

            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                hash_join_i32_v4_config(build_keys, test.build_size, probe_keys, test.probe_size,
                                         JoinType::INNER, result, config);
            }

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = hash_join_i32_v4_config(build_keys, test.build_size, probe_keys, test.probe_size,
                                                   JoinType::INNER, result, config);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;

            std::cout << "│ v4 TWOPHASE_CPU    │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << std::setprecision(2) << v3_time / avg_time << "x"
                      << " │\n";
        }

        // v4 with GPU_PARALLEL strategy (if available)
        {
            JoinConfigV4 config;
            config.strategy = JoinStrategy::GPU_PARALLEL;

            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                hash_join_i32_v4_config(build_keys, test.build_size, probe_keys, test.probe_size,
                                         JoinType::INNER, result, config);
            }

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = hash_join_i32_v4_config(build_keys, test.build_size, probe_keys, test.probe_size,
                                                   JoinType::INNER, result, config);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;

            std::cout << "│ v4 GPU_PARALLEL    │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << std::setprecision(2) << v3_time / avg_time << "x"
                      << " │\n";
        }

        // v6 Chain (链式哈希表优化)
        {
            for (int i = 0; i < WARMUP; ++i) {
                result->count = 0;
                hash_join_i32_v6_chain(build_keys, test.build_size, probe_keys, test.probe_size,
                                        JoinType::INNER, result);
            }

            double total_time = 0;
            size_t matches = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                result->count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                matches = hash_join_i32_v6_chain(build_keys, test.build_size, probe_keys, test.probe_size,
                                                  JoinType::INNER, result);
                auto end = std::chrono::high_resolution_clock::now();
                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
            double avg_time = total_time / ITERATIONS;

            std::cout << "│ v6 Chain           │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg_time
                      << " │ " << std::setw(10) << matches
                      << " │ " << std::setw(10) << std::setprecision(2) << v3_time / avg_time << "x"
                      << " │\n";
        }

        std::cout << "└────────────────────┴────────────┴────────────┴────────────┘\n\n";

        free_join_result(result);
        free(build_raw);
        free(probe_raw);
    }

    std::cout << "目标: 高匹配场景 v4 vs v3 提升 1.5x+\n";

    return 0;
}
