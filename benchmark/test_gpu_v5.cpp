/**
 * GPU v5 诊断测试
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "thunderduck/join.h"

using namespace thunderduck::join;

int main() {
    std::cout << "=== GPU v5 诊断测试 ===" << std::endl;

    // 检查 GPU 可用性
    bool gpu_ready = is_strategy_available(JoinStrategy::GPU);
    std::cout << "GPU 策略可用: " << (gpu_ready ? "是" : "否") << std::endl;

    // 检查 v5 GPU 可用性
    bool gpu_v5_ready = v4::is_gpu_v5_ready();
    std::cout << "GPU v5 可用: " << (gpu_v5_ready ? "是" : "否") << std::endl;

    if (!gpu_v5_ready) {
        std::cout << "GPU v5 不可用，无法继续测试" << std::endl;
        return 1;
    }

    // 测试数据
    const size_t build_count = 100000;
    const size_t probe_count = 1000000;

    std::vector<int32_t> build_keys(build_count);
    std::vector<int32_t> probe_keys(probe_count);

    for (size_t i = 0; i < build_count; i++) {
        build_keys[i] = static_cast<int32_t>(i);
    }
    for (size_t i = 0; i < probe_count; i++) {
        probe_keys[i] = static_cast<int32_t>(i % build_count);
    }

    JoinResult* result = create_join_result(probe_count);
    JoinConfigV4 config;

    // 直接调用 v5 GPU
    std::cout << "\n直接调用 v5 GPU..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    size_t matches = v4::hash_join_gpu_v5(
        build_keys.data(), build_count,
        probe_keys.data(), probe_count,
        JoinType::INNER, result, config);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  匹配数: " << matches << " (预期: " << probe_count << ")" << std::endl;
    std::cout << "  耗时: " << ms << " ms" << std::endl;
    std::cout << "  吞吐量: " << (build_count + probe_count) / ms / 1000.0 << " M/s" << std::endl;

    // 对比 v3
    std::cout << "\n对比 v3..." << std::endl;
    result->count = 0;

    start = std::chrono::high_resolution_clock::now();
    matches = hash_join_i32_v3(
        build_keys.data(), build_count,
        probe_keys.data(), probe_count,
        JoinType::INNER, result);
    end = std::chrono::high_resolution_clock::now();

    ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  匹配数: " << matches << std::endl;
    std::cout << "  耗时: " << ms << " ms" << std::endl;
    std::cout << "  吞吐量: " << (build_count + probe_count) / ms / 1000.0 << " M/s" << std::endl;

    free_join_result(result);
    return 0;
}
