/**
 * GPU v5 随机键测试
 * 使用随机键避免完美哈希优化，真正测试 GPU 性能
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "thunderduck/join.h"

using namespace thunderduck::join;

void run_test(size_t build_count, size_t probe_count, const char* label) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    std::cout << "Build: " << build_count << ", Probe: " << probe_count << std::endl;

    // 生成随机键 (范围足够大避免完美哈希)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(1, 100000000);  // 1 亿范围

    std::vector<int32_t> build_keys(build_count);
    std::vector<int32_t> probe_keys(probe_count);

    for (size_t i = 0; i < build_count; i++) {
        build_keys[i] = dist(rng);
    }

    // Probe keys 部分来自 build (确保有匹配)
    size_t num_matches = std::min(build_count, probe_count);
    for (size_t i = 0; i < num_matches; i++) {
        probe_keys[i] = build_keys[i % build_count];
    }
    // 剩余部分随机
    for (size_t i = num_matches; i < probe_count; i++) {
        probe_keys[i] = dist(rng);
    }

    // 打乱顺序
    std::shuffle(probe_keys.begin(), probe_keys.end(), rng);

    JoinResult* result = create_join_result(probe_count * 2);
    JoinConfigV4 config;

    // 测试 v3
    std::cout << "\nv3:" << std::endl;
    result->count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    size_t matches = hash_join_i32_v3(
        build_keys.data(), build_count,
        probe_keys.data(), probe_count,
        JoinType::INNER, result);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  匹配数: " << matches << std::endl;
    std::cout << "  耗时: " << ms << " ms" << std::endl;
    std::cout << "  吞吐量: " << (build_count + probe_count) / ms / 1000.0 << " M/s" << std::endl;
    double v3_time = ms;

    // 测试 v4 AUTO
    std::cout << "\nv4 AUTO:" << std::endl;
    const char* strategy = get_selected_strategy_name(build_count, probe_count, config);
    std::cout << "  选择策略: " << strategy << std::endl;
    result->count = 0;

    start = std::chrono::high_resolution_clock::now();
    matches = hash_join_i32_v4_config(
        build_keys.data(), build_count,
        probe_keys.data(), probe_count,
        JoinType::INNER, result, config);
    end = std::chrono::high_resolution_clock::now();

    ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  匹配数: " << matches << std::endl;
    std::cout << "  耗时: " << ms << " ms" << std::endl;
    std::cout << "  吞吐量: " << (build_count + probe_count) / ms / 1000.0 << " M/s" << std::endl;
    std::cout << "  vs v3: " << v3_time / ms << "x" << std::endl;

    // 直接测试 GPU v5
    if (v4::is_gpu_v5_ready()) {
        std::cout << "\nGPU v5 直接调用:" << std::endl;
        result->count = 0;

        start = std::chrono::high_resolution_clock::now();
        matches = v4::hash_join_gpu_v5(
            build_keys.data(), build_count,
            probe_keys.data(), probe_count,
            JoinType::INNER, result, config);
        end = std::chrono::high_resolution_clock::now();

        ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  匹配数: " << matches << std::endl;
        std::cout << "  耗时: " << ms << " ms" << std::endl;
        std::cout << "  吞吐量: " << (build_count + probe_count) / ms / 1000.0 << " M/s" << std::endl;
        std::cout << "  vs v3: " << v3_time / ms << "x" << std::endl;
    }

    free_join_result(result);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         GPU v5 随机键性能测试                                    ║" << std::endl;
    std::cout << "║         避免完美哈希，测试真实 GPU 性能                          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    // J2 规模
    run_test(100000, 1000000, "J2: 100K × 1M (随机键)");

    // J3 规模
    run_test(1000000, 10000000, "J3: 1M × 10M (随机键)");

    // 更大规模
    run_test(5000000, 50000000, "J4: 5M × 50M (随机键)");

    return 0;
}
