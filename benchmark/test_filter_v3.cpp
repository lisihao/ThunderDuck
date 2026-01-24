/**
 * ThunderDuck Filter v3.0 Performance Test
 *
 * 快速对比 v2 和 v3 的性能差异
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>

#include "thunderduck/filter.h"
#include "thunderduck/memory.h"

using namespace std::chrono;
using namespace thunderduck::filter;

// ============================================================================
// 计时工具
// ============================================================================

class Timer {
public:
    void start() { start_ = high_resolution_clock::now(); }
    void stop() { end_ = high_resolution_clock::now(); }
    double ms() const {
        return duration_cast<nanoseconds>(end_ - start_).count() / 1000000.0;
    }
private:
    high_resolution_clock::time_point start_, end_;
};

// ============================================================================
// 测试结构
// ============================================================================

struct TestResult {
    std::string name;
    double v2_min_ms, v2_avg_ms, v2_max_ms;
    double v3_min_ms, v3_avg_ms, v3_max_ms;
    double speedup;
    size_t result_count;
};

// ============================================================================
// 基准测试函数
// ============================================================================

TestResult run_count_benchmark(const std::string& name,
                                const int32_t* data, size_t count,
                                CompareOp op, int32_t value,
                                int warmup = 5, int iterations = 20) {
    TestResult result;
    result.name = name;

    std::vector<double> v2_times, v3_times;
    Timer timer;
    size_t v2_count = 0, v3_count = 0;

    // 预热 v2
    for (int i = 0; i < warmup; ++i) {
        count_i32_v2(data, count, op, value);
    }

    // 运行 v2
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        v2_count = count_i32_v2(data, count, op, value);
        timer.stop();
        v2_times.push_back(timer.ms());
    }

    // 预热 v3
    for (int i = 0; i < warmup; ++i) {
        count_i32_v3(data, count, op, value);
    }

    // 运行 v3
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        v3_count = count_i32_v3(data, count, op, value);
        timer.stop();
        v3_times.push_back(timer.ms());
    }

    // 验证结果一致性
    if (v2_count != v3_count) {
        std::cerr << "ERROR: Result mismatch! v2=" << v2_count << " v3=" << v3_count << "\n";
    }
    result.result_count = v3_count;

    // 计算统计
    std::sort(v2_times.begin(), v2_times.end());
    std::sort(v3_times.begin(), v3_times.end());

    result.v2_min_ms = v2_times.front();
    result.v2_max_ms = v2_times.back();
    result.v2_avg_ms = std::accumulate(v2_times.begin(), v2_times.end(), 0.0) / iterations;

    result.v3_min_ms = v3_times.front();
    result.v3_max_ms = v3_times.back();
    result.v3_avg_ms = std::accumulate(v3_times.begin(), v3_times.end(), 0.0) / iterations;

    result.speedup = result.v2_avg_ms / result.v3_avg_ms;

    return result;
}

TestResult run_range_benchmark(const std::string& name,
                                const int32_t* data, size_t count,
                                int32_t low, int32_t high,
                                int warmup = 5, int iterations = 20) {
    TestResult result;
    result.name = name;

    std::vector<double> v2_times, v3_times;
    Timer timer;
    size_t v2_count = 0, v3_count = 0;

    // 预热和运行 v2
    for (int i = 0; i < warmup; ++i) {
        count_i32_range_v2(data, count, low, high);
    }
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        v2_count = count_i32_range_v2(data, count, low, high);
        timer.stop();
        v2_times.push_back(timer.ms());
    }

    // 预热和运行 v3
    for (int i = 0; i < warmup; ++i) {
        count_i32_range_v3(data, count, low, high);
    }
    for (int i = 0; i < iterations; ++i) {
        timer.start();
        v3_count = count_i32_range_v3(data, count, low, high);
        timer.stop();
        v3_times.push_back(timer.ms());
    }

    // 验证结果一致性
    if (v2_count != v3_count) {
        std::cerr << "ERROR: Result mismatch! v2=" << v2_count << " v3=" << v3_count << "\n";
    }
    result.result_count = v3_count;

    // 计算统计
    std::sort(v2_times.begin(), v2_times.end());
    std::sort(v3_times.begin(), v3_times.end());

    result.v2_min_ms = v2_times.front();
    result.v2_max_ms = v2_times.back();
    result.v2_avg_ms = std::accumulate(v2_times.begin(), v2_times.end(), 0.0) / iterations;

    result.v3_min_ms = v3_times.front();
    result.v3_max_ms = v3_times.back();
    result.v3_avg_ms = std::accumulate(v3_times.begin(), v3_times.end(), 0.0) / iterations;

    result.speedup = result.v2_avg_ms / result.v3_avg_ms;

    return result;
}

void print_result(const TestResult& r) {
    std::string color = r.speedup >= 1.0 ? "\033[32m" : "\033[33m";
    std::string reset = "\033[0m";

    std::cout << "│ " << std::left << std::setw(24) << r.name << " │ "
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(8) << r.v2_avg_ms << " │ "
              << std::setw(8) << r.v3_avg_ms << " │ "
              << color << std::setw(7) << std::setprecision(2) << r.speedup << "x" << reset << " │ "
              << std::setw(10) << r.result_count << " │\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ThunderDuck Filter v3.0 Performance Test                        ║\n";
    std::cout << "║              v2 (baseline) vs v3 (optimized)                                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n\n";

    // 数据规模
    size_t data_size = 5000000;  // 5M 元素
    if (argc > 1) {
        data_size = std::stoul(argv[1]);
    }

    std::cout << "Configuration:\n";
    std::cout << "  Data Size:    " << data_size << " elements\n";
    std::cout << "  Data Volume:  " << (data_size * 4 / 1024 / 1024) << " MB\n";
    std::cout << "  Warmup:       5 iterations\n";
    std::cout << "  Benchmark:    20 iterations\n\n";

    // 生成测试数据
    std::cout << "Generating test data... " << std::flush;
    std::vector<int32_t> data(data_size);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(1, 50);
    for (auto& v : data) {
        v = dist(rng);
    }
    std::cout << "done\n\n";

    // 运行测试
    std::vector<TestResult> results;

    std::cout << "Running benchmarks...\n\n";

    std::cout << "┌──────────────────────────┬──────────┬──────────┬─────────┬────────────┐\n";
    std::cout << "│ Test                     │ v2 (ms)  │ v3 (ms)  │ Speedup │ Match Count│\n";
    std::cout << "├──────────────────────────┼──────────┼──────────┼─────────┼────────────┤\n";

    // Test 1: GT (Greater Than)
    auto r1 = run_count_benchmark("GT (quantity > 25)", data.data(), data.size(),
                                   CompareOp::GT, 25);
    print_result(r1);
    results.push_back(r1);

    // Test 2: EQ (Equal)
    auto r2 = run_count_benchmark("EQ (quantity == 30)", data.data(), data.size(),
                                   CompareOp::EQ, 30);
    print_result(r2);
    results.push_back(r2);

    // Test 3: LT (Less Than)
    auto r3 = run_count_benchmark("LT (quantity < 20)", data.data(), data.size(),
                                   CompareOp::LT, 20);
    print_result(r3);
    results.push_back(r3);

    // Test 4: GE (Greater or Equal)
    auto r4 = run_count_benchmark("GE (quantity >= 40)", data.data(), data.size(),
                                   CompareOp::GE, 40);
    print_result(r4);
    results.push_back(r4);

    // Test 5: NE (Not Equal)
    auto r5 = run_count_benchmark("NE (quantity != 25)", data.data(), data.size(),
                                   CompareOp::NE, 25);
    print_result(r5);
    results.push_back(r5);

    // Test 6: Range
    auto r6 = run_range_benchmark("Range (10 <= x < 40)", data.data(), data.size(),
                                   10, 40);
    print_result(r6);
    results.push_back(r6);

    std::cout << "└──────────────────────────┴──────────┴──────────┴─────────┴────────────┘\n\n";

    // 汇总统计
    double total_speedup = 0;
    int improved = 0;
    for (const auto& r : results) {
        total_speedup += r.speedup;
        if (r.speedup >= 1.0) improved++;
    }
    double avg_speedup = total_speedup / results.size();

    std::cout << "═══════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "  Total Tests:     " << results.size() << "\n";
    std::cout << "  Improved:        " << improved << "/" << results.size() << "\n";
    std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x\n\n";

    if (avg_speedup >= 1.0) {
        std::cout << "\033[32m✓ v3 is faster than v2 on average!\033[0m\n\n";
    } else {
        std::cout << "\033[33m✗ v3 is slower than v2 on average. Need more optimization.\033[0m\n\n";
    }

    // 优化效果分析
    std::cout << "Optimization Analysis:\n";
    std::cout << "  1. Template specialization: Eliminated switch-case in inner loop\n";
    std::cout << "  2. Independent accumulators: 4 parallel accumulation streams\n";
    std::cout << "  3. vsub optimization: mask subtraction instead of shift+add\n";
    std::cout << "  4. 256-element batching: Reduced horizontal reduction frequency\n\n";

    return 0;
}
