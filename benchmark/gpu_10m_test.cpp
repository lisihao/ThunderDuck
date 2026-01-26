/**
 * GPU 10M+ 数据规模测试
 * 验证 GPU 在大数据量下是否有优势
 */

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"
#include "thunderduck/uma_memory.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
using namespace thunderduck;

template<typename Func>
double measure_time(Func&& func, int iterations = 5) {
    // Warmup
    for (int i = 0; i < 2; ++i) func();

    double total = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, milli>(end - start).count();
    }
    return total / iterations;
}

void test_filter(size_t count) {
    cout << "\n=== Filter " << count/1000000 << "M rows ===\n";

    // 生成测试数据
    vector<int32_t> data(count);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int32_t> dist(0, 1000000000);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }

    vector<uint32_t> indices(count);
    size_t result_count = 0;

    // CPU v3
    double cpu_time = measure_time([&]() {
        result_count = filter::filter_i32_v3(data.data(), count, filter::CompareOp::GT,
                                              500000000, indices.data());
    });

    // GPU v4 (UMA)
    double gpu_time = cpu_time;
    if (filter::is_filter_gpu_available()) {
        gpu_time = measure_time([&]() {
            result_count = filter::filter_i32_v4(data.data(), count, filter::CompareOp::GT,
                                                  500000000, indices.data());
        });
    }

    double bandwidth_cpu = (count * sizeof(int32_t)) / (cpu_time * 1e6);
    double bandwidth_gpu = (count * sizeof(int32_t)) / (gpu_time * 1e6);

    printf("CPU v3:  %8.2f ms, %.2f GB/s\n", cpu_time, bandwidth_cpu);
    printf("GPU v4:  %8.2f ms, %.2f GB/s\n", gpu_time, bandwidth_gpu);
    printf("GPU/CPU: %.2fx\n", cpu_time / gpu_time);
    printf("选择率:  %.1f%%\n", result_count * 100.0 / count);
}

void test_aggregate(size_t count) {
    cout << "\n=== Aggregate " << count/1000000 << "M rows ===\n";

    vector<int32_t> data(count);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }

    int64_t sum = 0;
    int32_t min_val = 0, max_val = 0;

    // CPU baseline (v2 sum, minmax_i32)
    double cpu_time = measure_time([&]() {
        sum = aggregate::sum_i32_v2(data.data(), count);
        aggregate::minmax_i32(data.data(), count, &min_val, &max_val);
    });

    // GPU v3 (UMA)
    double gpu_time = cpu_time;
    if (aggregate::is_aggregate_gpu_available()) {
        gpu_time = measure_time([&]() {
            sum = aggregate::sum_i32_v3(data.data(), count);
            aggregate::minmax_i32_v3(data.data(), count, &min_val, &max_val);
        });
    }

    double bandwidth_cpu = (count * sizeof(int32_t) * 2) / (cpu_time * 1e6);  // *2 for both SUM and MINMAX
    double bandwidth_gpu = (count * sizeof(int32_t) * 2) / (gpu_time * 1e6);

    printf("CPU v2:  %8.2f ms, %.2f GB/s\n", cpu_time, bandwidth_cpu);
    printf("GPU v3:  %8.2f ms, %.2f GB/s\n", gpu_time, bandwidth_gpu);
    printf("GPU/CPU: %.2fx\n", cpu_time / gpu_time);
}

void test_hash_join(size_t build_count, size_t probe_count) {
    cout << "\n=== Hash Join " << build_count/1000 << "K × " << probe_count/1000000 << "M ===\n";

    // 生成 build 表
    vector<int32_t> build_keys(build_count);
    for (size_t i = 0; i < build_count; ++i) {
        build_keys[i] = static_cast<int32_t>(i);
    }

    // 生成 probe 表 (随机键)
    vector<int32_t> probe_keys(probe_count);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int32_t> dist(0, build_count * 2);  // 50% 匹配率
    for (size_t i = 0; i < probe_count; ++i) {
        probe_keys[i] = dist(gen);
    }

    join::JoinResult result_cpu;
    join::JoinResult result_gpu;
    size_t match_count = 0;

    // CPU v4
    double cpu_time = measure_time([&]() {
        match_count = join::hash_join_i32_v4(
            build_keys.data(), build_count,
            probe_keys.data(), probe_count,
            join::JoinType::INNER, &result_cpu);
    });

    // GPU UMA
    double gpu_time = cpu_time;
    if (join::uma::is_uma_gpu_ready()) {
        join::JoinConfigV4 config;
        config.strategy = join::JoinStrategy::GPU;
        gpu_time = measure_time([&]() {
            match_count = join::uma::hash_join_gpu_uma(
                build_keys.data(), build_count,
                probe_keys.data(), probe_count,
                join::JoinType::INNER, &result_gpu, config);
        });
    }

    printf("CPU v4:  %8.2f ms\n", cpu_time);
    printf("GPU:     %8.2f ms\n", gpu_time);
    printf("GPU/CPU: %.2fx\n", cpu_time / gpu_time);
    printf("匹配数:  %zu (%.1f%%)\n", match_count, match_count * 100.0 / probe_count);
}

int main() {
    cout << "═══════════════════════════════════════════════════════════════\n";
    cout << "          GPU 10M+ 数据规模测试\n";
    cout << "═══════════════════════════════════════════════════════════════\n";

    // 检查 GPU 可用性
    cout << "Checking Filter GPU... " << flush;
    bool filter_gpu = filter::is_filter_gpu_available();
    cout << (filter_gpu ? "✓" : "✗") << "\n";

    cout << "Checking Aggregate GPU... " << flush;
    bool agg_gpu = aggregate::is_aggregate_gpu_available();
    cout << (agg_gpu ? "✓" : "✗") << "\n";

    // 暂时跳过 Join GPU 检查 (可能有 bug)
    // cout << "Checking Join GPU UMA... " << flush;
    // bool join_gpu = join::uma::is_uma_gpu_ready();
    // cout << (join_gpu ? "✓" : "✗") << "\n";

    // Filter 测试
    test_filter(10000000);   // 10M
    test_filter(50000000);   // 50M
    // test_filter(100000000);  // 100M - 可能内存不足

    // Aggregate 测试
    test_aggregate(10000000);   // 10M
    test_aggregate(50000000);   // 50M
    // test_aggregate(100000000);  // 100M

    // 暂时跳过 Hash Join 测试 (GPU 初始化有 bug)
    // test_hash_join(100000, 10000000);   // 100K × 10M
    // test_hash_join(1000000, 10000000);  // 1M × 10M

    cout << "\n═══════════════════════════════════════════════════════════════\n";
    return 0;
}
