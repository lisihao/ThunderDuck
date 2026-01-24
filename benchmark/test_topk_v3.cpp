/**
 * TopK v3.0 测试程序
 *
 * 测试正确性和性能对比（v2 vs v3）
 */

#include <thunderduck/sort.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>

using namespace thunderduck::sort;

// 高精度计时
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// 生成测试数据
std::vector<int32_t> generate_random_data(size_t count, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);

    std::vector<int32_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// 使用 STL 验证结果
bool verify_topk_max(const int32_t* data, size_t count, size_t k,
                     const int32_t* result, const uint32_t* indices) {
    // 获取参考答案
    std::vector<std::pair<int32_t, size_t>> pairs(count);
    for (size_t i = 0; i < count; ++i) {
        pairs[i] = {data[i], i};
    }

    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // 验证值
    std::vector<int32_t> expected_values(k);
    for (size_t i = 0; i < k; ++i) {
        expected_values[i] = pairs[i].first;
    }
    std::sort(expected_values.begin(), expected_values.end(), std::greater<int32_t>());

    std::vector<int32_t> actual_values(result, result + k);
    std::sort(actual_values.begin(), actual_values.end(), std::greater<int32_t>());

    if (expected_values != actual_values) {
        std::cerr << "  Value mismatch!" << std::endl;
        return false;
    }

    // 验证索引（如果提供）
    if (indices != nullptr) {
        for (size_t i = 0; i < k; ++i) {
            if (data[indices[i]] != result[i]) {
                std::cerr << "  Index mismatch at " << i << ": indices[" << i << "]="
                          << indices[i] << " -> " << data[indices[i]] << " != " << result[i] << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool verify_topk_min(const int32_t* data, size_t count, size_t k,
                     const int32_t* result, const uint32_t* indices) {
    // 获取参考答案
    std::vector<std::pair<int32_t, size_t>> pairs(count);
    for (size_t i = 0; i < count; ++i) {
        pairs[i] = {data[i], i};
    }

    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    // 验证值
    std::vector<int32_t> expected_values(k);
    for (size_t i = 0; i < k; ++i) {
        expected_values[i] = pairs[i].first;
    }
    std::sort(expected_values.begin(), expected_values.end());

    std::vector<int32_t> actual_values(result, result + k);
    std::sort(actual_values.begin(), actual_values.end());

    if (expected_values != actual_values) {
        std::cerr << "  Value mismatch!" << std::endl;
        return false;
    }

    // 验证索引
    if (indices != nullptr) {
        for (size_t i = 0; i < k; ++i) {
            if (data[indices[i]] != result[i]) {
                std::cerr << "  Index mismatch at " << i << std::endl;
                return false;
            }
        }
    }

    return true;
}

// 正确性测试
void test_correctness() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "正确性测试" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const size_t N = 100000;
    auto data = generate_random_data(N);

    std::vector<size_t> k_values = {10, 50, 100, 500, 1000, 2000, 5000, 10000};

    for (size_t k : k_values) {
        std::vector<int32_t> values(k);
        std::vector<uint32_t> indices(k);

        // 测试 TopK Max v3
        topk_max_i32_v3(data.data(), N, k, values.data(), indices.data());
        bool max_ok = verify_topk_max(data.data(), N, k, values.data(), indices.data());

        // 测试 TopK Min v3
        topk_min_i32_v3(data.data(), N, k, values.data(), indices.data());
        bool min_ok = verify_topk_min(data.data(), N, k, values.data(), indices.data());

        std::cout << "K=" << std::setw(5) << k
                  << ": Max " << (max_ok ? "✓" : "✗")
                  << "  Min " << (min_ok ? "✓" : "✗") << std::endl;
    }
}

// 性能测试
void test_performance() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "性能测试 - TopK v2 vs v3" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const size_t N = 1000000;
    const int WARMUP = 3;
    const int ITERATIONS = 10;

    auto data = generate_random_data(N);

    std::vector<std::pair<size_t, const char*>> test_cases = {
        {10, "Small K (堆策略)"},
        {64, "K=64 边界"},
        {100, "Medium K"},
        {500, "K=500"},
        {1000, "K=1000 (SIMD堆)"},
        {2000, "K=2000 (分块)"},
        {5000, "K=5000 (分块)"},
        {10000, "Large K (nth_element)"},
        {50000, "Very Large K"}
    };

    std::cout << std::left << std::setw(25) << "测试"
              << std::right << std::setw(12) << "v2 (ms)"
              << std::setw(12) << "v3 (ms)"
              << std::setw(12) << "加速比" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& [k, desc] : test_cases) {
        std::vector<int32_t> values(k);
        std::vector<uint32_t> indices(k);

        // Warmup v2
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v2(data.data(), N, k, values.data(), indices.data());
        }

        // Benchmark v2
        Timer timer_v2;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v2(data.data(), N, k, values.data(), indices.data());
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        // Warmup v3
        for (int i = 0; i < WARMUP; ++i) {
            topk_max_i32_v3(data.data(), N, k, values.data(), indices.data());
        }

        // Benchmark v3
        Timer timer_v3;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v3(data.data(), N, k, values.data(), indices.data());
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        double speedup = time_v2 / time_v3;

        std::cout << std::left << std::setw(25) << desc
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << speedup << "x" << std::endl;
    }
}

// 不同数据分布测试
void test_distributions() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "数据分布测试 (K=1000)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const size_t N = 1000000;
    const size_t K = 1000;
    const int ITERATIONS = 10;

    std::vector<int32_t> values(K);
    std::vector<uint32_t> indices(K);

    std::cout << std::left << std::setw(25) << "分布类型"
              << std::right << std::setw(12) << "v2 (ms)"
              << std::setw(12) << "v3 (ms)"
              << std::setw(12) << "加速比" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // 1. 均匀分布
    {
        auto data = generate_random_data(N);

        Timer timer_v2;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v2(data.data(), N, K, values.data(), indices.data());
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        Timer timer_v3;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v3(data.data(), N, K, values.data(), indices.data());
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        std::cout << std::left << std::setw(25) << "均匀分布"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << (time_v2/time_v3) << "x" << std::endl;
    }

    // 2. 已排序（最差情况）
    {
        std::vector<int32_t> data(N);
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<int32_t>(i);
        }

        Timer timer_v2;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v2(data.data(), N, K, values.data(), indices.data());
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        Timer timer_v3;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v3(data.data(), N, K, values.data(), indices.data());
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        std::cout << std::left << std::setw(25) << "升序排列（最差）"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << (time_v2/time_v3) << "x" << std::endl;
    }

    // 3. 逆序排列（最佳情况）
    {
        std::vector<int32_t> data(N);
        for (size_t i = 0; i < N; ++i) {
            data[i] = static_cast<int32_t>(N - i);
        }

        Timer timer_v2;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v2(data.data(), N, K, values.data(), indices.data());
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        Timer timer_v3;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v3(data.data(), N, K, values.data(), indices.data());
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        std::cout << std::left << std::setw(25) << "降序排列（最佳）"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << (time_v2/time_v3) << "x" << std::endl;
    }

    // 4. 大量重复值
    {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int32_t> dist(0, 100);  // 只有 101 种值

        std::vector<int32_t> data(N);
        for (size_t i = 0; i < N; ++i) {
            data[i] = dist(gen);
        }

        Timer timer_v2;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v2(data.data(), N, K, values.data(), indices.data());
        }
        double time_v2 = timer_v2.elapsed_ms() / ITERATIONS;

        Timer timer_v3;
        for (int i = 0; i < ITERATIONS; ++i) {
            topk_max_i32_v3(data.data(), N, K, values.data(), indices.data());
        }
        double time_v3 = timer_v3.elapsed_ms() / ITERATIONS;

        std::cout << std::left << std::setw(25) << "大量重复值"
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << time_v2
                  << std::setw(12) << time_v3
                  << std::setw(11) << (time_v2/time_v3) << "x" << std::endl;
    }
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "     ThunderDuck TopK v3.0 测试" << std::endl;
    std::cout << "================================================" << std::endl;

    test_correctness();
    test_performance();
    test_distributions();

    std::cout << "\n测试完成！" << std::endl;

    return 0;
}
