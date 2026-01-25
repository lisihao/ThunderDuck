/**
 * ThunderDuck - 全面 UMA 性能基准测试
 *
 * 测试内容:
 * - Filter: v3 (CPU SIMD) vs v4 (UMA GPU)
 * - Aggregate: v2 (CPU SIMD) vs v3 (UMA GPU)
 * - Join: v3 (CPU) vs v4 (CPU优化) vs UMA GPU
 * - TopK: v4/v5 (CPU) vs v6 (UMA GPU)
 *
 * 输出:
 * - SQL 语义
 * - 数据量
 * - 执行时间
 * - 吞吐量
 * - 加速比
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <map>

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/sort.h"
#include "thunderduck/uma_memory.h"

using namespace thunderduck;
using namespace std::chrono;

// 为避免 uma 命名空间歧义，使用别名
namespace uma_mem = thunderduck::uma;
namespace uma_join = thunderduck::join::uma;

// ============================================================================
// 工具函数
// ============================================================================

void* page_aligned_alloc(size_t size) {
    void* ptr = nullptr;
    posix_memalign(&ptr, 16384, size);
    return ptr;
}

void page_aligned_free(void* ptr) {
    free(ptr);
}

struct BenchmarkResult {
    std::string name;
    std::string sql_semantic;
    std::string operator_type;
    std::string accelerator;    // CPU/GPU/NPU
    size_t data_size;           // 数据量 (行数)
    size_t data_bytes;          // 数据字节数
    double time_ms;             // 执行时间
    double throughput_mrows;    // 吞吐量 (M rows/s)
    double bandwidth_gbps;      // 带宽 (GB/s)
    double vs_baseline;         // vs baseline 加速比
    size_t result_count;        // 结果数量
};

std::vector<BenchmarkResult> all_results;

void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "│ " << std::setw(20) << std::left << r.name
              << " │ " << std::setw(8) << r.accelerator
              << " │ " << std::setw(10) << r.time_ms
              << " │ " << std::setw(10) << r.throughput_mrows
              << " │ " << std::setw(10) << r.bandwidth_gbps
              << " │ " << std::setw(8) << r.vs_baseline << "x"
              << " │" << std::endl;
}

void print_section_header(const char* title) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ " << std::setw(88) << std::left << title << " ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
}

void print_table_header() {
    std::cout << "┌──────────────────────┬──────────┬────────────┬────────────┬────────────┬──────────┐" << std::endl;
    std::cout << "│ 版本                 │ 加速器   │ 时间(ms)   │ 吞吐(M/s)  │ 带宽(GB/s) │ 加速比   │" << std::endl;
    std::cout << "├──────────────────────┼──────────┼────────────┼────────────┼────────────┼──────────┤" << std::endl;
}

void print_table_footer() {
    std::cout << "└──────────────────────┴──────────┴────────────┴────────────┴────────────┴──────────┘" << std::endl;
}

// ============================================================================
// Filter 基准测试
// ============================================================================

void benchmark_filter() {
    print_section_header("Filter 算子测试 - SELECT * FROM t WHERE col > value");

    const size_t RUNS = 5;

    // 测试不同数据规模
    struct TestCase {
        size_t count;
        const char* name;
        float selectivity;  // 预估选择率
    };

    TestCase cases[] = {
        {100000,    "F1: 100K rows",   0.5f},
        {1000000,   "F2: 1M rows",     0.5f},
        {10000000,  "F3: 10M rows",    0.5f},
        {50000000,  "F4: 50M rows",    0.5f},
    };

    for (const auto& tc : cases) {
        std::cout << "\n【" << tc.name << "】" << std::endl;
        std::cout << "SQL: SELECT * FROM t WHERE value > 500000000" << std::endl;
        std::cout << "数据量: " << tc.count << " rows, "
                  << (tc.count * sizeof(int32_t) / 1024 / 1024) << " MB" << std::endl;

        // 分配页对齐内存
        int32_t* data = (int32_t*)page_aligned_alloc(tc.count * sizeof(int32_t));
        uint32_t* indices = (uint32_t*)page_aligned_alloc(tc.count * sizeof(uint32_t));

        // 生成随机数据
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000000000);
        for (size_t i = 0; i < tc.count; i++) {
            data[i] = dist(rng);
        }

        int32_t threshold = 500000000;  // ~50% 选择率

        // 预热
        filter::filter_i32_v3(data, tc.count, filter::CompareOp::GT, threshold, indices);

        print_table_header();

        // v3 (CPU SIMD) - baseline
        double best_v3 = 1e9;
        size_t result_count = 0;
        for (size_t i = 0; i < RUNS; i++) {
            auto start = high_resolution_clock::now();
            result_count = filter::filter_i32_v3(data, tc.count,
                filter::CompareOp::GT, threshold, indices);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_v3 = std::min(best_v3, ms);
        }

        BenchmarkResult r_v3;
        r_v3.name = "Filter v3";
        r_v3.sql_semantic = "WHERE col > val";
        r_v3.operator_type = "Filter";
        r_v3.accelerator = "CPU SIMD";
        r_v3.data_size = tc.count;
        r_v3.data_bytes = tc.count * sizeof(int32_t);
        r_v3.time_ms = best_v3;
        r_v3.throughput_mrows = tc.count / best_v3 / 1000.0;
        r_v3.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_v3 / 1e6;
        r_v3.vs_baseline = 1.0;
        r_v3.result_count = result_count;
        print_result(r_v3);
        all_results.push_back(r_v3);

        // v4 (UMA GPU)
        if (filter::is_filter_gpu_available()) {
            double best_v4 = 1e9;
            for (size_t i = 0; i < RUNS; i++) {
                auto start = high_resolution_clock::now();
                result_count = filter::filter_i32_v4(data, tc.count,
                    filter::CompareOp::GT, threshold, indices);
                auto end = high_resolution_clock::now();
                double ms = duration<double, std::milli>(end - start).count();
                best_v4 = std::min(best_v4, ms);
            }

            BenchmarkResult r_v4;
            r_v4.name = "Filter v4 UMA";
            r_v4.sql_semantic = "WHERE col > val";
            r_v4.operator_type = "Filter";
            r_v4.accelerator = "GPU";
            r_v4.data_size = tc.count;
            r_v4.data_bytes = tc.count * sizeof(int32_t);
            r_v4.time_ms = best_v4;
            r_v4.throughput_mrows = tc.count / best_v4 / 1000.0;
            r_v4.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_v4 / 1e6;
            r_v4.vs_baseline = best_v3 / best_v4;
            r_v4.result_count = result_count;
            print_result(r_v4);
            all_results.push_back(r_v4);
        }

        print_table_footer();
        std::cout << "结果行数: " << result_count << " ("
                  << (100.0 * result_count / tc.count) << "%)" << std::endl;

        page_aligned_free(data);
        page_aligned_free(indices);
    }
}

// ============================================================================
// Aggregate 基准测试
// ============================================================================

void benchmark_aggregate() {
    print_section_header("Aggregate 算子测试 - SELECT SUM(col), MIN(col), MAX(col) FROM t");

    const size_t RUNS = 5;

    struct TestCase {
        size_t count;
        const char* name;
    };

    TestCase cases[] = {
        {100000,    "A1: 100K rows"},
        {1000000,   "A2: 1M rows"},
        {10000000,  "A3: 10M rows"},
        {50000000,  "A4: 50M rows"},
    };

    for (const auto& tc : cases) {
        std::cout << "\n【" << tc.name << "】" << std::endl;
        std::cout << "SQL: SELECT SUM(value), MIN(value), MAX(value) FROM t" << std::endl;
        std::cout << "数据量: " << tc.count << " rows, "
                  << (tc.count * sizeof(int32_t) / 1024 / 1024) << " MB" << std::endl;

        int32_t* data = (int32_t*)page_aligned_alloc(tc.count * sizeof(int32_t));

        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
        for (size_t i = 0; i < tc.count; i++) {
            data[i] = dist(rng);
        }

        // 预热
        aggregate::sum_i32_v2(data, tc.count);

        print_table_header();

        // v2 (CPU SIMD) - SUM
        double best_sum_v2 = 1e9;
        int64_t sum_result = 0;
        for (size_t i = 0; i < RUNS; i++) {
            auto start = high_resolution_clock::now();
            sum_result = aggregate::sum_i32_v2(data, tc.count);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_sum_v2 = std::min(best_sum_v2, ms);
        }

        BenchmarkResult r_sum_v2;
        r_sum_v2.name = "SUM v2";
        r_sum_v2.sql_semantic = "SUM(col)";
        r_sum_v2.operator_type = "Aggregate";
        r_sum_v2.accelerator = "CPU SIMD";
        r_sum_v2.data_size = tc.count;
        r_sum_v2.data_bytes = tc.count * sizeof(int32_t);
        r_sum_v2.time_ms = best_sum_v2;
        r_sum_v2.throughput_mrows = tc.count / best_sum_v2 / 1000.0;
        r_sum_v2.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_sum_v2 / 1e6;
        r_sum_v2.vs_baseline = 1.0;
        print_result(r_sum_v2);
        all_results.push_back(r_sum_v2);

        // v3 (UMA GPU) - SUM
        if (aggregate::is_aggregate_gpu_available()) {
            double best_sum_v3 = 1e9;
            for (size_t i = 0; i < RUNS; i++) {
                auto start = high_resolution_clock::now();
                sum_result = aggregate::sum_i32_v3(data, tc.count);
                auto end = high_resolution_clock::now();
                double ms = duration<double, std::milli>(end - start).count();
                best_sum_v3 = std::min(best_sum_v3, ms);
            }

            BenchmarkResult r_sum_v3;
            r_sum_v3.name = "SUM v3 UMA";
            r_sum_v3.sql_semantic = "SUM(col)";
            r_sum_v3.operator_type = "Aggregate";
            r_sum_v3.accelerator = "GPU";
            r_sum_v3.data_size = tc.count;
            r_sum_v3.data_bytes = tc.count * sizeof(int32_t);
            r_sum_v3.time_ms = best_sum_v3;
            r_sum_v3.throughput_mrows = tc.count / best_sum_v3 / 1000.0;
            r_sum_v3.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_sum_v3 / 1e6;
            r_sum_v3.vs_baseline = best_sum_v2 / best_sum_v3;
            print_result(r_sum_v3);
            all_results.push_back(r_sum_v3);
        }

        // MINMAX v2 (CPU)
        double best_minmax_v2 = 1e9;
        int32_t min_val, max_val;
        for (size_t i = 0; i < RUNS; i++) {
            auto start = high_resolution_clock::now();
            aggregate::minmax_i32(data, tc.count, &min_val, &max_val);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_minmax_v2 = std::min(best_minmax_v2, ms);
        }

        BenchmarkResult r_mm_v2;
        r_mm_v2.name = "MINMAX v2";
        r_mm_v2.sql_semantic = "MIN/MAX(col)";
        r_mm_v2.operator_type = "Aggregate";
        r_mm_v2.accelerator = "CPU SIMD";
        r_mm_v2.data_size = tc.count;
        r_mm_v2.data_bytes = tc.count * sizeof(int32_t);
        r_mm_v2.time_ms = best_minmax_v2;
        r_mm_v2.throughput_mrows = tc.count / best_minmax_v2 / 1000.0;
        r_mm_v2.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_minmax_v2 / 1e6;
        r_mm_v2.vs_baseline = 1.0;
        print_result(r_mm_v2);
        all_results.push_back(r_mm_v2);

        // MINMAX v3 (GPU)
        if (aggregate::is_aggregate_gpu_available()) {
            double best_minmax_v3 = 1e9;
            for (size_t i = 0; i < RUNS; i++) {
                auto start = high_resolution_clock::now();
                aggregate::minmax_i32_v3(data, tc.count, &min_val, &max_val);
                auto end = high_resolution_clock::now();
                double ms = duration<double, std::milli>(end - start).count();
                best_minmax_v3 = std::min(best_minmax_v3, ms);
            }

            BenchmarkResult r_mm_v3;
            r_mm_v3.name = "MINMAX v3 UMA";
            r_mm_v3.sql_semantic = "MIN/MAX(col)";
            r_mm_v3.operator_type = "Aggregate";
            r_mm_v3.accelerator = "GPU";
            r_mm_v3.data_size = tc.count;
            r_mm_v3.data_bytes = tc.count * sizeof(int32_t);
            r_mm_v3.time_ms = best_minmax_v3;
            r_mm_v3.throughput_mrows = tc.count / best_minmax_v3 / 1000.0;
            r_mm_v3.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_minmax_v3 / 1e6;
            r_mm_v3.vs_baseline = best_minmax_v2 / best_minmax_v3;
            print_result(r_mm_v3);
            all_results.push_back(r_mm_v3);
        }

        print_table_footer();
        std::cout << "SUM 结果: " << sum_result << ", MIN: " << min_val
                  << ", MAX: " << max_val << std::endl;

        page_aligned_free(data);
    }
}

// ============================================================================
// Join 基准测试
// ============================================================================

void benchmark_join() {
    print_section_header("Hash Join 算子测试 - SELECT * FROM t1 JOIN t2 ON t1.key = t2.key");

    const size_t RUNS = 3;

    struct TestCase {
        size_t build_count;
        size_t probe_count;
        const char* name;
        bool random_keys;
    };

    TestCase cases[] = {
        {10000,    100000,    "J1: 10K × 100K",   false},
        {100000,   1000000,   "J2: 100K × 1M",    true},
        {1000000,  10000000,  "J3: 1M × 10M",     true},
        {5000000,  50000000,  "J4: 5M × 50M",     true},
    };

    for (const auto& tc : cases) {
        std::cout << "\n【" << tc.name << "】" << std::endl;
        std::cout << "SQL: SELECT * FROM t1 JOIN t2 ON t1.key = t2.key" << std::endl;
        std::cout << "Build 表: " << tc.build_count << " rows, Probe 表: " << tc.probe_count << " rows" << std::endl;
        std::cout << "数据量: " << ((tc.build_count + tc.probe_count) * sizeof(int32_t) / 1024 / 1024) << " MB" << std::endl;
        std::cout << "键类型: " << (tc.random_keys ? "随机" : "连续") << std::endl;

        int32_t* build_keys = (int32_t*)page_aligned_alloc(tc.build_count * sizeof(int32_t));
        int32_t* probe_keys = (int32_t*)page_aligned_alloc(tc.probe_count * sizeof(int32_t));

        if (tc.random_keys) {
            std::mt19937 rng(42);
            std::uniform_int_distribution<int32_t> dist(1, 100000000);
            for (size_t i = 0; i < tc.build_count; i++) {
                build_keys[i] = dist(rng);
            }
            for (size_t i = 0; i < tc.probe_count; i++) {
                if (i < tc.build_count) {
                    probe_keys[i] = build_keys[i % tc.build_count];
                } else {
                    probe_keys[i] = dist(rng);
                }
            }
        } else {
            for (size_t i = 0; i < tc.build_count; i++) {
                build_keys[i] = static_cast<int32_t>(i);
            }
            for (size_t i = 0; i < tc.probe_count; i++) {
                probe_keys[i] = static_cast<int32_t>(i % tc.build_count);
            }
        }

        join::JoinResult* result = join::create_join_result(tc.probe_count * 2);

        // 预热
        join::hash_join_i32_v3(build_keys, tc.build_count, probe_keys, tc.probe_count,
                               join::JoinType::INNER, result);

        print_table_header();

        size_t total_keys = tc.build_count + tc.probe_count;
        size_t total_bytes = total_keys * sizeof(int32_t);

        // v3 (CPU baseline)
        double best_v3 = 1e9;
        size_t matches = 0;
        for (size_t i = 0; i < RUNS; i++) {
            result->count = 0;
            auto start = high_resolution_clock::now();
            matches = join::hash_join_i32_v3(build_keys, tc.build_count,
                probe_keys, tc.probe_count, join::JoinType::INNER, result);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_v3 = std::min(best_v3, ms);
        }

        BenchmarkResult r_v3;
        r_v3.name = "Join v3";
        r_v3.sql_semantic = "INNER JOIN";
        r_v3.operator_type = "Hash Join";
        r_v3.accelerator = "CPU SIMD";
        r_v3.data_size = total_keys;
        r_v3.data_bytes = total_bytes;
        r_v3.time_ms = best_v3;
        r_v3.throughput_mrows = total_keys / best_v3 / 1000.0;
        r_v3.bandwidth_gbps = total_bytes / best_v3 / 1e6;
        r_v3.vs_baseline = 1.0;
        r_v3.result_count = matches;
        print_result(r_v3);
        all_results.push_back(r_v3);

        // v4 (CPU 优化)
        double best_v4 = 1e9;
        for (size_t i = 0; i < RUNS; i++) {
            result->count = 0;
            auto start = high_resolution_clock::now();
            matches = join::hash_join_i32_v4(build_keys, tc.build_count,
                probe_keys, tc.probe_count, join::JoinType::INNER, result);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_v4 = std::min(best_v4, ms);
        }

        BenchmarkResult r_v4;
        r_v4.name = "Join v4";
        r_v4.sql_semantic = "INNER JOIN";
        r_v4.operator_type = "Hash Join";
        r_v4.accelerator = "CPU Opt";
        r_v4.data_size = total_keys;
        r_v4.data_bytes = total_bytes;
        r_v4.time_ms = best_v4;
        r_v4.throughput_mrows = total_keys / best_v4 / 1000.0;
        r_v4.bandwidth_gbps = total_bytes / best_v4 / 1e6;
        r_v4.vs_baseline = best_v3 / best_v4;
        r_v4.result_count = matches;
        print_result(r_v4);
        all_results.push_back(r_v4);

        // UMA GPU
        if (uma_join::is_uma_gpu_ready()) {
            double best_uma = 1e9;
            join::JoinConfigV4 config;
            for (size_t i = 0; i < RUNS; i++) {
                result->count = 0;
                auto start = high_resolution_clock::now();
                matches = uma_join::hash_join_gpu_uma(build_keys, tc.build_count,
                    probe_keys, tc.probe_count, join::JoinType::INNER, result, config);
                auto end = high_resolution_clock::now();
                double ms = duration<double, std::milli>(end - start).count();
                best_uma = std::min(best_uma, ms);
            }

            BenchmarkResult r_uma;
            r_uma.name = "Join UMA GPU";
            r_uma.sql_semantic = "INNER JOIN";
            r_uma.operator_type = "Hash Join";
            r_uma.accelerator = "GPU";
            r_uma.data_size = total_keys;
            r_uma.data_bytes = total_bytes;
            r_uma.time_ms = best_uma;
            r_uma.throughput_mrows = total_keys / best_uma / 1000.0;
            r_uma.bandwidth_gbps = total_bytes / best_uma / 1e6;
            r_uma.vs_baseline = best_v3 / best_uma;
            r_uma.result_count = matches;
            print_result(r_uma);
            all_results.push_back(r_uma);
        }

        print_table_footer();
        std::cout << "匹配行数: " << matches << std::endl;

        join::free_join_result(result);
        page_aligned_free(build_keys);
        page_aligned_free(probe_keys);
    }
}

// ============================================================================
// TopK 基准测试
// ============================================================================

void benchmark_topk() {
    print_section_header("TopK 算子测试 - SELECT * FROM t ORDER BY col DESC LIMIT K");

    const size_t RUNS = 5;

    struct TestCase {
        size_t count;
        size_t k;
        const char* name;
    };

    TestCase cases[] = {
        {1000000,   10,     "T1: 1M rows, K=10"},
        {1000000,   100,    "T2: 1M rows, K=100"},
        {10000000,  10,     "T3: 10M rows, K=10"},
        {10000000,  100,    "T4: 10M rows, K=100"},
        {50000000,  10,     "T5: 50M rows, K=10"},
    };

    for (const auto& tc : cases) {
        std::cout << "\n【" << tc.name << "】" << std::endl;
        std::cout << "SQL: SELECT * FROM t ORDER BY value DESC LIMIT " << tc.k << std::endl;
        std::cout << "数据量: " << tc.count << " rows, "
                  << (tc.count * sizeof(int32_t) / 1024 / 1024) << " MB" << std::endl;

        int32_t* data = (int32_t*)page_aligned_alloc(tc.count * sizeof(int32_t));
        int32_t* out_values = (int32_t*)page_aligned_alloc(tc.k * sizeof(int32_t));
        uint32_t* out_indices = (uint32_t*)page_aligned_alloc(tc.k * sizeof(uint32_t));

        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000000000);
        for (size_t i = 0; i < tc.count; i++) {
            data[i] = dist(rng);
        }

        // 预热
        sort::topk_max_i32_v5(data, tc.count, tc.k, out_values, out_indices);

        print_table_header();

        // v4 (CPU 采样预过滤)
        double best_v4 = 1e9;
        for (size_t i = 0; i < RUNS; i++) {
            auto start = high_resolution_clock::now();
            sort::topk_max_i32_v4(data, tc.count, tc.k, out_values, out_indices);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_v4 = std::min(best_v4, ms);
        }

        BenchmarkResult r_v4;
        r_v4.name = "TopK v4";
        r_v4.sql_semantic = "ORDER BY LIMIT";
        r_v4.operator_type = "TopK";
        r_v4.accelerator = "CPU Sample";
        r_v4.data_size = tc.count;
        r_v4.data_bytes = tc.count * sizeof(int32_t);
        r_v4.time_ms = best_v4;
        r_v4.throughput_mrows = tc.count / best_v4 / 1000.0;
        r_v4.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_v4 / 1e6;
        r_v4.vs_baseline = 1.0;
        r_v4.result_count = tc.k;
        print_result(r_v4);
        all_results.push_back(r_v4);

        // v5 (CPU Count-Based)
        double best_v5 = 1e9;
        for (size_t i = 0; i < RUNS; i++) {
            auto start = high_resolution_clock::now();
            sort::topk_max_i32_v5(data, tc.count, tc.k, out_values, out_indices);
            auto end = high_resolution_clock::now();
            double ms = duration<double, std::milli>(end - start).count();
            best_v5 = std::min(best_v5, ms);
        }

        BenchmarkResult r_v5;
        r_v5.name = "TopK v5";
        r_v5.sql_semantic = "ORDER BY LIMIT";
        r_v5.operator_type = "TopK";
        r_v5.accelerator = "CPU Count";
        r_v5.data_size = tc.count;
        r_v5.data_bytes = tc.count * sizeof(int32_t);
        r_v5.time_ms = best_v5;
        r_v5.throughput_mrows = tc.count / best_v5 / 1000.0;
        r_v5.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_v5 / 1e6;
        r_v5.vs_baseline = best_v4 / best_v5;
        r_v5.result_count = tc.k;
        print_result(r_v5);
        all_results.push_back(r_v5);

        // v6 (UMA GPU)
        if (sort::is_topk_gpu_available()) {
            double best_v6 = 1e9;
            for (size_t i = 0; i < RUNS; i++) {
                auto start = high_resolution_clock::now();
                sort::topk_max_i32_v6(data, tc.count, tc.k, out_values, out_indices);
                auto end = high_resolution_clock::now();
                double ms = duration<double, std::milli>(end - start).count();
                best_v6 = std::min(best_v6, ms);
            }

            BenchmarkResult r_v6;
            r_v6.name = "TopK v6 UMA";
            r_v6.sql_semantic = "ORDER BY LIMIT";
            r_v6.operator_type = "TopK";
            r_v6.accelerator = "GPU";
            r_v6.data_size = tc.count;
            r_v6.data_bytes = tc.count * sizeof(int32_t);
            r_v6.time_ms = best_v6;
            r_v6.throughput_mrows = tc.count / best_v6 / 1000.0;
            r_v6.bandwidth_gbps = (tc.count * sizeof(int32_t)) / best_v6 / 1e6;
            r_v6.vs_baseline = best_v4 / best_v6;
            r_v6.result_count = tc.k;
            print_result(r_v6);
            all_results.push_back(r_v6);
        }

        print_table_footer();
        std::cout << "Top-" << tc.k << " 结果: [" << out_values[0];
        for (size_t i = 1; i < std::min(tc.k, (size_t)5); i++) {
            std::cout << ", " << out_values[i];
        }
        if (tc.k > 5) std::cout << ", ...";
        std::cout << "]" << std::endl;

        page_aligned_free(data);
        page_aligned_free(out_values);
        page_aligned_free(out_indices);
    }
}

// ============================================================================
// 综合报告
// ============================================================================

void print_summary() {
    print_section_header("综合性能报告");

    std::cout << "\n=== 各算子最佳 GPU 加速比 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // 按算子类型分组
    std::map<std::string, std::pair<double, std::string>> best_by_operator;

    for (const auto& r : all_results) {
        if (r.accelerator == "GPU" && r.vs_baseline > 1.0) {
            std::string key = r.operator_type;
            if (best_by_operator.find(key) == best_by_operator.end() ||
                r.vs_baseline > best_by_operator[key].first) {
                best_by_operator[key] = {r.vs_baseline, r.name};
            }
        }
    }

    std::cout << "\n┌───────────────┬────────────────────────┬────────────┐" << std::endl;
    std::cout << "│ 算子类型      │ 最佳版本               │ GPU 加速比 │" << std::endl;
    std::cout << "├───────────────┼────────────────────────┼────────────┤" << std::endl;

    for (const auto& kv : best_by_operator) {
        std::cout << "│ " << std::setw(13) << std::left << kv.first
                  << " │ " << std::setw(22) << kv.second.second
                  << " │ " << std::setw(9) << kv.second.first << "x │" << std::endl;
    }
    std::cout << "└───────────────┴────────────────────────┴────────────┘" << std::endl;

    // 找出需要优化的点
    std::cout << "\n=== 优化建议 ===" << std::endl;

    for (const auto& r : all_results) {
        if (r.accelerator == "GPU" && r.vs_baseline < 1.0) {
            std::cout << "⚠️  " << r.name << " (" << r.operator_type << ", "
                      << r.data_size << " rows): GPU 慢于 CPU ("
                      << r.vs_baseline << "x)" << std::endl;
            std::cout << "   建议: 提高 GPU 阈值或优化 kernel" << std::endl;
        }
    }

    // 带宽分析
    std::cout << "\n=== 带宽利用率分析 (M4 理论峰值 ~400 GB/s) ===" << std::endl;

    double max_bw = 0;
    std::string max_bw_name;
    for (const auto& r : all_results) {
        if (r.bandwidth_gbps > max_bw) {
            max_bw = r.bandwidth_gbps;
            max_bw_name = r.name + " (" + std::to_string((int)(r.data_size/1000000)) + "M)";
        }
    }

    std::cout << "最高带宽: " << max_bw << " GB/s (" << max_bw_name << ")" << std::endl;
    std::cout << "带宽利用率: " << (max_bw / 400.0 * 100) << "%" << std::endl;

    if (max_bw < 100) {
        std::cout << "⚠️  带宽利用率较低，建议:" << std::endl;
        std::cout << "   - 增加批处理大小" << std::endl;
        std::cout << "   - 减少 kernel 启动开销" << std::endl;
        std::cout << "   - 优化内存访问模式" << std::endl;
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    ThunderDuck UMA 全面性能基准测试                                       ║" << std::endl;
    std::cout << "║                    Apple M4 - 统一内存架构优化                                            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;

    // 系统状态
    std::cout << "\n=== 系统状态 ===" << std::endl;
    std::cout << "UMA 内存管理器: " << (uma_mem::UMAMemoryManager::instance().is_available() ? "✓ 可用" : "✗ 不可用") << std::endl;
    std::cout << "Filter GPU: " << (filter::is_filter_gpu_available() ? "✓ 可用" : "✗ 不可用") << std::endl;
    std::cout << "Aggregate GPU: " << (aggregate::is_aggregate_gpu_available() ? "✓ 可用" : "✗ 不可用") << std::endl;
    std::cout << "Join GPU: " << (uma_join::is_uma_gpu_ready() ? "✓ 可用" : "✗ 不可用") << std::endl;
    std::cout << "TopK GPU: " << (sort::is_topk_gpu_available() ? "✓ 可用" : "✗ 不可用") << std::endl;

    // 运行基准测试
    benchmark_filter();
    benchmark_aggregate();
    benchmark_join();
    benchmark_topk();

    // 打印综合报告
    print_summary();

    return 0;
}
