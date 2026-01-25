/**
 * UMA GPU Hash Join 性能测试
 *
 * 对比:
 * - v3 (CPU baseline)
 * - v4 GPU (旧实现，有数据拷贝)
 * - UMA GPU (新实现，零拷贝)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

#include "thunderduck/join.h"
#include "thunderduck/uma_memory.h"

using namespace thunderduck::join;
using namespace thunderduck::uma;

// 页对齐分配
void* page_aligned_alloc(size_t size) {
    void* ptr = nullptr;
    posix_memalign(&ptr, 16384, size);  // 16KB 对齐
    return ptr;
}

void page_aligned_free(void* ptr) {
    free(ptr);
}

struct TestResult {
    double time_ms;
    size_t matches;
    double throughput;  // M keys/s
};

TestResult run_v3(const int32_t* build, size_t build_count,
                  const int32_t* probe, size_t probe_count,
                  JoinResult* result) {
    result->count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    size_t matches = hash_join_i32_v3(build, build_count,
                                       probe, probe_count,
                                       JoinType::INNER, result);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = (build_count + probe_count) / ms / 1000.0;

    return {ms, matches, throughput};
}

TestResult run_v4_gpu(const int32_t* build, size_t build_count,
                      const int32_t* probe, size_t probe_count,
                      JoinResult* result) {
    result->count = 0;
    JoinConfigV4 config;
    config.strategy = JoinStrategy::GPU;

    auto start = std::chrono::high_resolution_clock::now();
    size_t matches = hash_join_i32_v4_config(build, build_count,
                                              probe, probe_count,
                                              JoinType::INNER, result, config);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = (build_count + probe_count) / ms / 1000.0;

    return {ms, matches, throughput};
}

TestResult run_uma_gpu(const int32_t* build, size_t build_count,
                       const int32_t* probe, size_t probe_count,
                       JoinResult* result) {
    result->count = 0;
    JoinConfigV4 config;

    auto start = std::chrono::high_resolution_clock::now();
    size_t matches = uma::hash_join_gpu_uma(build, build_count,
                                             probe, probe_count,
                                             JoinType::INNER, result, config);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = (build_count + probe_count) / ms / 1000.0;

    return {ms, matches, throughput};
}

void run_test(const char* name, size_t build_count, size_t probe_count, bool use_random_keys) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ " << std::setw(64) << std::left << name << " ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "Build: " << build_count << ", Probe: " << probe_count << std::endl;
    std::cout << "Keys: " << (use_random_keys ? "随机 (避免完美哈希)" : "连续 (触发完美哈希)") << std::endl;

    // 使用页对齐内存 (支持零拷贝)
    int32_t* build_keys = (int32_t*)page_aligned_alloc(build_count * sizeof(int32_t));
    int32_t* probe_keys = (int32_t*)page_aligned_alloc(probe_count * sizeof(int32_t));

    // 生成数据
    if (use_random_keys) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(1, 100000000);

        for (size_t i = 0; i < build_count; i++) {
            build_keys[i] = dist(rng);
        }
        // Probe 部分来自 build，部分随机
        for (size_t i = 0; i < probe_count; i++) {
            if (i < build_count) {
                probe_keys[i] = build_keys[i % build_count];
            } else {
                probe_keys[i] = dist(rng);
            }
        }
    } else {
        // 连续整数
        for (size_t i = 0; i < build_count; i++) {
            build_keys[i] = static_cast<int32_t>(i);
        }
        for (size_t i = 0; i < probe_count; i++) {
            probe_keys[i] = static_cast<int32_t>(i % build_count);
        }
    }

    // 创建结果缓冲区
    JoinResult* result = create_join_result(probe_count * 2);

    // 预热
    run_v3(build_keys, build_count, probe_keys, probe_count, result);

    // 测试
    const int RUNS = 3;

    std::cout << "\n运行 " << RUNS << " 次取最佳结果...\n" << std::endl;

    // v3
    TestResult best_v3 = {1e9, 0, 0};
    for (int i = 0; i < RUNS; i++) {
        auto r = run_v3(build_keys, build_count, probe_keys, probe_count, result);
        if (r.time_ms < best_v3.time_ms) best_v3 = r;
    }

    // v4 GPU
    TestResult best_v4_gpu = {1e9, 0, 0};
    if (is_strategy_available(JoinStrategy::GPU)) {
        for (int i = 0; i < RUNS; i++) {
            auto r = run_v4_gpu(build_keys, build_count, probe_keys, probe_count, result);
            if (r.time_ms < best_v4_gpu.time_ms) best_v4_gpu = r;
        }
    }

    // UMA GPU
    TestResult best_uma = {1e9, 0, 0};
    if (uma::is_uma_gpu_ready()) {
        for (int i = 0; i < RUNS; i++) {
            auto r = run_uma_gpu(build_keys, build_count, probe_keys, probe_count, result);
            if (r.time_ms < best_uma.time_ms) best_uma = r;
        }
    }

    // 输出结果
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "┌────────────────┬────────────┬────────────┬────────────┬────────────┐" << std::endl;
    std::cout << "│ 实现           │ 时间 (ms)  │ 吞吐量 M/s │ vs v3      │ 匹配数     │" << std::endl;
    std::cout << "├────────────────┼────────────┼────────────┼────────────┼────────────┤" << std::endl;

    std::cout << "│ v3 (CPU)       │ " << std::setw(10) << best_v3.time_ms
              << " │ " << std::setw(10) << best_v3.throughput
              << " │ " << std::setw(10) << "1.00x"
              << " │ " << std::setw(10) << best_v3.matches << " │" << std::endl;

    if (is_strategy_available(JoinStrategy::GPU)) {
        double speedup = best_v3.time_ms / best_v4_gpu.time_ms;
        std::cout << "│ v4 GPU (旧)    │ " << std::setw(10) << best_v4_gpu.time_ms
                  << " │ " << std::setw(10) << best_v4_gpu.throughput
                  << " │ " << std::setw(9) << speedup << "x"
                  << " │ " << std::setw(10) << best_v4_gpu.matches << " │" << std::endl;
    }

    if (uma::is_uma_gpu_ready()) {
        double speedup = best_v3.time_ms / best_uma.time_ms;
        std::cout << "│ UMA GPU (新)   │ " << std::setw(10) << best_uma.time_ms
                  << " │ " << std::setw(10) << best_uma.throughput
                  << " │ " << std::setw(9) << speedup << "x"
                  << " │ " << std::setw(10) << best_uma.matches << " │" << std::endl;
    }

    std::cout << "└────────────────┴────────────┴────────────┴────────────┴────────────┘" << std::endl;

    // 检查 UMA vs v4 改进
    if (is_strategy_available(JoinStrategy::GPU) && uma::is_uma_gpu_ready()) {
        double uma_improvement = best_v4_gpu.time_ms / best_uma.time_ms;
        std::cout << "\nUMA vs 旧 GPU 改进: " << std::setprecision(2) << uma_improvement << "x" << std::endl;
    }

    // 清理
    free_join_result(result);
    page_aligned_free(build_keys);
    page_aligned_free(probe_keys);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       UMA GPU Hash Join 性能测试                                 ║" << std::endl;
    std::cout << "║       对比: v3 (CPU) vs v4 GPU vs UMA GPU                        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n系统状态:" << std::endl;
    std::cout << "  UMA 内存管理器: " << (UMAMemoryManager::instance().is_available() ? "可用" : "不可用") << std::endl;
    std::cout << "  UMA GPU Join: " << (uma::is_uma_gpu_ready() ? "可用" : "不可用") << std::endl;
    std::cout << "  v4 GPU Join: " << (is_strategy_available(JoinStrategy::GPU) ? "可用" : "不可用") << std::endl;

    // 测试用例
    // 1. 小数据量，连续键 (触发完美哈希)
    run_test("J1: 10K × 100K (连续键)", 10000, 100000, false);

    // 2. 中等数据量，随机键
    run_test("J2: 100K × 1M (随机键)", 100000, 1000000, true);

    // 3. 大数据量，随机键
    run_test("J3: 1M × 10M (随机键)", 1000000, 10000000, true);

    // 4. 超大数据量
    run_test("J4: 5M × 50M (随机键)", 5000000, 50000000, true);

    return 0;
}
