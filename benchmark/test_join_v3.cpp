/**
 * Hash Join v3.0 测试程序
 *
 * 测试正确性和性能对比（v2 vs v3）
 */

#include <thunderduck/join.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <unordered_set>

using namespace thunderduck::join;

// 高精度计时
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

// 生成测试数据
std::vector<int32_t> generate_keys(size_t count, int32_t min_val, int32_t max_val, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);

    std::vector<int32_t> keys(count);
    for (size_t i = 0; i < count; ++i) {
        keys[i] = dist(gen);
    }
    return keys;
}

// 生成唯一键 (用于 build 侧)
std::vector<int32_t> generate_unique_keys(size_t count, int seed = 42) {
    std::vector<int32_t> keys(count);
    for (size_t i = 0; i < count; ++i) {
        keys[i] = static_cast<int32_t>(i + 1);  // 1 到 count
    }

    // 打乱
    std::mt19937 gen(seed);
    std::shuffle(keys.begin(), keys.end(), gen);

    return keys;
}

// 使用 STL 计算参考答案
size_t compute_reference_join(const std::vector<int32_t>& build_keys,
                               const std::vector<int32_t>& probe_keys) {
    std::unordered_multiset<int32_t> build_set(build_keys.begin(), build_keys.end());

    size_t matches = 0;
    for (int32_t key : probe_keys) {
        matches += build_set.count(key);
    }
    return matches;
}

// 正确性测试
void test_correctness() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "正确性测试" << std::endl;
    std::cout << "========================================\n" << std::endl;

    struct TestCase {
        size_t build_count;
        size_t probe_count;
        int32_t key_range;
        const char* desc;
    };

    std::vector<TestCase> test_cases = {
        {100, 1000, 1000, "小表 Join"},
        {1000, 10000, 10000, "中表 Join"},
        {10000, 100000, 100000, "大表 Join"},
        {100, 1000, 50, "高冲突 Join"},       // 完美哈希场景
        {10000, 100000, 20000, "中等冲突"},
    };

    for (const auto& tc : test_cases) {
        // 生成数据
        auto build_keys = generate_keys(tc.build_count, 1, tc.key_range, 42);
        auto probe_keys = generate_keys(tc.probe_count, 1, tc.key_range, 123);

        // 计算参考答案
        size_t expected = compute_reference_join(build_keys, probe_keys);

        // 测试 v2
        JoinResult* result_v2 = create_join_result(tc.probe_count);
        size_t v2_matches = hash_join_i32_v2(build_keys.data(), build_keys.size(),
                                              probe_keys.data(), probe_keys.size(),
                                              JoinType::INNER, result_v2);

        // 测试 v3
        JoinResult* result_v3 = create_join_result(tc.probe_count);
        size_t v3_matches = hash_join_i32_v3(build_keys.data(), build_keys.size(),
                                              probe_keys.data(), probe_keys.size(),
                                              JoinType::INNER, result_v3);

        bool v2_ok = (v2_matches == expected);
        bool v3_ok = (v3_matches == expected);

        std::cout << std::left << std::setw(20) << tc.desc
                  << " | Expected: " << std::setw(8) << expected
                  << " | v2: " << std::setw(8) << v2_matches << (v2_ok ? " ✓" : " ✗")
                  << " | v3: " << std::setw(8) << v3_matches << (v3_ok ? " ✓" : " ✗")
                  << std::endl;

        free_join_result(result_v2);
        free_join_result(result_v3);
    }
}

// 性能测试
void test_performance() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "性能测试 - Join v2 vs v3" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int WARMUP = 3;
    const int ITERATIONS = 10;

    struct TestCase {
        size_t build_count;
        size_t probe_count;
        int32_t key_range;
        const char* desc;
    };

    std::vector<TestCase> test_cases = {
        {1000, 10000, 2000, "1K x 10K (小表)"},
        {10000, 100000, 20000, "10K x 100K (中表)"},
        {100000, 1000000, 200000, "100K x 1M (大表)"},
        {100000, 1000000, 100000, "100K x 1M (高选择率)"},
        {100000, 1000000, 100000, "完美哈希场景"},
    };

    std::cout << std::left << std::setw(25) << "测试"
              << std::right << std::setw(12) << "v2 (ms)"
              << std::setw(12) << "v3 (ms)"
              << std::setw(12) << "加速比"
              << std::setw(12) << "匹配数" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (const auto& tc : test_cases) {
        // 生成数据
        auto build_keys = generate_unique_keys(tc.build_count, 42);
        auto probe_keys = generate_keys(tc.probe_count, 1, tc.build_count, 123);

        JoinResult* result_v2 = create_join_result(tc.probe_count * 2);
        JoinResult* result_v3 = create_join_result(tc.probe_count * 2);

        // Warmup v2
        for (int i = 0; i < WARMUP; ++i) {
            result_v2->count = 0;
            hash_join_i32_v2(build_keys.data(), build_keys.size(),
                             probe_keys.data(), probe_keys.size(),
                             JoinType::INNER, result_v2);
        }

        // Benchmark v2
        Timer timer_v2;
        size_t v2_matches = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            result_v2->count = 0;
            v2_matches = hash_join_i32_v2(build_keys.data(), build_keys.size(),
                                          probe_keys.data(), probe_keys.size(),
                                          JoinType::INNER, result_v2);
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        // Warmup v3
        for (int i = 0; i < WARMUP; ++i) {
            result_v3->count = 0;
            hash_join_i32_v3(build_keys.data(), build_keys.size(),
                             probe_keys.data(), probe_keys.size(),
                             JoinType::INNER, result_v3);
        }

        // Benchmark v3
        Timer timer_v3;
        size_t v3_matches = 0;
        for (int i = 0; i < ITERATIONS; ++i) {
            result_v3->count = 0;
            v3_matches = hash_join_i32_v3(build_keys.data(), build_keys.size(),
                                          probe_keys.data(), probe_keys.size(),
                                          JoinType::INNER, result_v3);
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        double speedup = time_v2 / time_v3;

        std::cout << std::left << std::setw(25) << tc.desc
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << speedup << "x"
                  << std::setw(12) << v3_matches << std::endl;

        free_join_result(result_v2);
        free_join_result(result_v3);
    }
}

// 分区效果测试
void test_partitioning() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "分区效果分析 (100K x 1M)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const size_t BUILD_COUNT = 100000;
    const size_t PROBE_COUNT = 1000000;
    const int ITERATIONS = 5;

    auto build_keys = generate_unique_keys(BUILD_COUNT, 42);
    auto probe_keys = generate_keys(PROBE_COUNT, 1, BUILD_COUNT, 123);

    JoinResult* result = create_join_result(PROBE_COUNT * 2);

    // 测试不同线程数
    std::cout << "线程数测试:\n";
    std::cout << std::left << std::setw(15) << "线程数"
              << std::right << std::setw(12) << "时间 (ms)"
              << std::setw(12) << "相对 1 线程" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    double time_1_thread = 0;

    for (int threads = 1; threads <= 4; ++threads) {
        JoinConfig config;
        config.num_threads = threads;

        // Warmup
        for (int i = 0; i < 2; ++i) {
            result->count = 0;
            hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                     probe_keys.data(), probe_keys.size(),
                                     JoinType::INNER, result, config);
        }

        // Benchmark
        Timer timer;
        for (int i = 0; i < ITERATIONS; ++i) {
            result->count = 0;
            hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                     probe_keys.data(), probe_keys.size(),
                                     JoinType::INNER, result, config);
        }
        double time = timer.elapsed_ms() / ITERATIONS;

        if (threads == 1) time_1_thread = time;

        std::cout << std::left << std::setw(15) << threads
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time
                  << std::setw(11) << (time_1_thread / time) << "x" << std::endl;
    }

    free_join_result(result);
}

// 完美哈希效果测试
void test_perfect_hash() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "完美哈希优化效果" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 小整数键，密集分布 - 完美哈希最佳场景
    const size_t BUILD_COUNT = 10000;
    const size_t PROBE_COUNT = 100000;
    const int ITERATIONS = 10;

    // 生成连续键 1-10000
    std::vector<int32_t> build_keys(BUILD_COUNT);
    for (size_t i = 0; i < BUILD_COUNT; ++i) {
        build_keys[i] = static_cast<int32_t>(i + 1);
    }

    auto probe_keys = generate_keys(PROBE_COUNT, 1, BUILD_COUNT, 123);

    JoinResult* result = create_join_result(PROBE_COUNT * 2);

    // 启用完美哈希
    JoinConfig config_perfect;
    config_perfect.enable_perfect_hash = true;

    // 禁用完美哈希
    JoinConfig config_no_perfect;
    config_no_perfect.enable_perfect_hash = false;

    // Warmup
    for (int i = 0; i < 3; ++i) {
        result->count = 0;
        hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                 probe_keys.data(), probe_keys.size(),
                                 JoinType::INNER, result, config_perfect);
    }

    // 测试完美哈希
    Timer timer_perfect;
    for (int i = 0; i < ITERATIONS; ++i) {
        result->count = 0;
        hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                 probe_keys.data(), probe_keys.size(),
                                 JoinType::INNER, result, config_perfect);
    }
    double time_perfect = timer_perfect.elapsed_ms() / ITERATIONS;

    // 测试普通哈希
    for (int i = 0; i < 3; ++i) {
        result->count = 0;
        hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                 probe_keys.data(), probe_keys.size(),
                                 JoinType::INNER, result, config_no_perfect);
    }

    Timer timer_normal;
    for (int i = 0; i < ITERATIONS; ++i) {
        result->count = 0;
        hash_join_i32_v3_config(build_keys.data(), build_keys.size(),
                                 probe_keys.data(), probe_keys.size(),
                                 JoinType::INNER, result, config_no_perfect);
    }
    double time_normal = timer_normal.elapsed_ms() / ITERATIONS;

    std::cout << "场景: 10K x 100K, 键范围 1-10000 (密集)" << std::endl;
    std::cout << "完美哈希: " << std::fixed << std::setprecision(3) << time_perfect << " ms" << std::endl;
    std::cout << "普通哈希: " << std::fixed << std::setprecision(3) << time_normal << " ms" << std::endl;
    std::cout << "加速比: " << std::fixed << std::setprecision(2) << (time_normal / time_perfect) << "x" << std::endl;

    free_join_result(result);
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "     ThunderDuck Hash Join v3.0 测试" << std::endl;
    std::cout << "================================================" << std::endl;

    test_correctness();
    test_performance();
    test_partitioning();
    test_perfect_hash();

    std::cout << "\n测试完成！" << std::endl;

    return 0;
}
