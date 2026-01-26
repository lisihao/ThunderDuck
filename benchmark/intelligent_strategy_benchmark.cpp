/**
 * ThunderDuck - 智能策略选择系统综合性能测试
 *
 * 测试内容:
 * 1. Join: 不同数据规模 (小/中/大/超大)
 * 2. Join: 不同选择率 (高/低)
 * 3. Filter: 不同数据规模和选择率
 * 4. TopK: 不同基数和K值
 * 5. 对比: 智能选择 vs 固定路径
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>

#include "thunderduck/query_optimizer.h"
#include "thunderduck/statistics.h"
#include "thunderduck/runtime_stats.h"
#include "thunderduck/execution_path.h"
#include "thunderduck/join.h"
#include "thunderduck/filter.h"
#include "thunderduck/sort.h"

using namespace thunderduck;
using namespace std::chrono;

// ============================================================================
// 辅助函数
// ============================================================================

class Timer {
public:
    void start() { start_ = high_resolution_clock::now(); }
    double stop_us() {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_).count();
    }
    double stop_ms() { return stop_us() / 1000.0; }
private:
    high_resolution_clock::time_point start_;
};

void print_header(const std::string& title) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(76) << title << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
}

void print_section(const std::string& title) {
    std::cout << "\n┌──────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(76) << title << " │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";
}

void print_result(const std::string& test, const std::string& path,
                  double time_ms, size_t result_count, double throughput_mps = 0) {
    std::cout << "│ " << std::left << std::setw(30) << test
              << "│ " << std::setw(20) << path
              << "│ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << time_ms << " ms"
              << " │ " << std::setw(10) << result_count << " │\n";
}

void print_comparison(const std::string& test, double smart_ms, double fixed_ms,
                      const std::string& smart_path, const std::string& fixed_path) {
    double speedup = fixed_ms / smart_ms;
    std::string speedup_str = (speedup >= 1.0) ?
        "\033[32m" + std::to_string(speedup).substr(0,4) + "x ↑\033[0m" :
        "\033[31m" + std::to_string(1.0/speedup).substr(0,4) + "x ↓\033[0m";

    std::cout << "│ " << std::left << std::setw(25) << test
              << "│ " << std::setw(15) << smart_path
              << "│ " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << smart_ms
              << " │ " << std::setw(8) << fixed_ms
              << " │ " << std::setw(10) << speedup_str << " │\n";
}

// 生成随机数据
std::vector<int32_t> generate_random_data(size_t count, int32_t min_val, int32_t max_val,
                                          unsigned seed = 42) {
    std::vector<int32_t> data(count);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

// 生成低基数数据
std::vector<int32_t> generate_low_cardinality_data(size_t count, int32_t distinct_values,
                                                    unsigned seed = 42) {
    std::vector<int32_t> data(count);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(0, distinct_values - 1);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

// 生成顺序数据 (用于 join key)
std::vector<int32_t> generate_sequential_data(size_t count, int32_t start = 0) {
    std::vector<int32_t> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = start + static_cast<int32_t>(i);
    }
    return data;
}

// ============================================================================
// Join 测试
// ============================================================================

struct JoinTestResult {
    std::string scenario;
    std::string smart_path;
    std::string fixed_path;
    double smart_time_ms;
    double fixed_time_ms;
    size_t matches;
};

JoinTestResult test_join_scenario(const std::string& scenario,
                                  size_t build_count, size_t probe_count,
                                  float selectivity, int iterations = 5) {
    JoinTestResult result;
    result.scenario = scenario;

    // 生成数据
    auto build_keys = generate_sequential_data(build_count);

    // 根据选择率生成 probe keys
    std::vector<int32_t> probe_keys(probe_count);
    std::mt19937 gen(42);
    size_t match_count = static_cast<size_t>(probe_count * selectivity);
    std::uniform_int_distribution<int32_t> match_dist(0, build_count - 1);
    std::uniform_int_distribution<int32_t> nomatch_dist(build_count, build_count * 2);

    for (size_t i = 0; i < match_count; i++) {
        probe_keys[i] = match_dist(gen);
    }
    for (size_t i = match_count; i < probe_count; i++) {
        probe_keys[i] = nomatch_dist(gen);
    }
    std::shuffle(probe_keys.begin(), probe_keys.end(), gen);

    auto& optimizer = optimizer::QueryOptimizer::instance();

    // 设置特征
    stats::JoinCharacteristics chars;
    chars.build_count = build_count;
    chars.probe_count = probe_count;
    chars.build_cardinality = 1.0f;  // 顺序数据，完全唯一
    chars.probe_cardinality = selectivity;
    chars.estimated_selectivity = selectivity;

    // 智能选择路径
    auto smart_path = optimizer.selectJoinPath(chars);
    result.smart_path = execution::getJoinPathName(smart_path);

    // 固定路径 (使用 v3 作为基准)
    result.fixed_path = "CPU_V3_RADIX256";

    // 预热
    join::JoinResult warmup_result;
    join::hash_join_i32_v3(build_keys.data(), build_count,
                           probe_keys.data(), probe_count,
                           join::JoinType::INNER, &warmup_result);

    // 测试智能选择
    Timer timer;
    double smart_total = 0;
    for (int i = 0; i < iterations; i++) {
        join::JoinResult jr;
        timer.start();
        optimizer.executeJoin(build_keys.data(), build_count,
                              probe_keys.data(), probe_count,
                              join::JoinType::INNER, &jr, &chars);
        smart_total += timer.stop_ms();
        result.matches = jr.count;
    }
    result.smart_time_ms = smart_total / iterations;

    // 测试固定路径 v3
    double fixed_total = 0;
    for (int i = 0; i < iterations; i++) {
        join::JoinResult jr;
        timer.start();
        join::hash_join_i32_v3(build_keys.data(), build_count,
                               probe_keys.data(), probe_count,
                               join::JoinType::INNER, &jr);
        fixed_total += timer.stop_ms();
    }
    result.fixed_time_ms = fixed_total / iterations;

    return result;
}

void run_join_tests() {
    print_header("JOIN 性能测试 - 智能策略选择");

    std::vector<JoinTestResult> results;

    // 场景 1: 小数据
    std::cout << "  Testing J1: Small data (10K x 100K)..." << std::flush;
    results.push_back(test_join_scenario("J1: 10K x 100K", 10000, 100000, 0.3f));
    std::cout << " done\n";

    // 场景 2: 中等数据
    std::cout << "  Testing J2: Medium data (100K x 1M)..." << std::flush;
    results.push_back(test_join_scenario("J2: 100K x 1M", 100000, 1000000, 0.3f));
    std::cout << " done\n";

    // 场景 3: 大数据 (J3 经典场景)
    std::cout << "  Testing J3: Large data (1M x 10M)..." << std::flush;
    results.push_back(test_join_scenario("J3: 1M x 10M", 1000000, 10000000, 0.3f, 3));
    std::cout << " done\n";

    // 场景 4: 低选择率 (应该选 Bloom)
    std::cout << "  Testing J4: Low selectivity (100K x 1M, 5%)..." << std::flush;
    results.push_back(test_join_scenario("J4: Low sel (5%)", 100000, 1000000, 0.05f));
    std::cout << " done\n";

    // 场景 5: 高选择率
    std::cout << "  Testing J5: High selectivity (100K x 1M, 80%)..." << std::flush;
    results.push_back(test_join_scenario("J5: High sel (80%)", 100000, 1000000, 0.8f));
    std::cout << " done\n";

    // 输出结果
    print_section("Join 测试结果对比");
    std::cout << "│ " << std::left << std::setw(25) << "Scenario"
              << "│ " << std::setw(15) << "Smart Path"
              << "│ " << std::setw(8) << "Smart"
              << " │ " << std::setw(8) << "Fixed"
              << " │ " << std::setw(10) << "Speedup" << " │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";

    for (const auto& r : results) {
        print_comparison(r.scenario, r.smart_time_ms, r.fixed_time_ms,
                        r.smart_path, r.fixed_path);
    }
    std::cout << "└──────────────────────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// Filter 测试
// ============================================================================

struct FilterTestResult {
    std::string scenario;
    std::string smart_path;
    double smart_time_ms;
    double v3_time_ms;
    size_t result_count;
};

FilterTestResult test_filter_scenario(const std::string& scenario,
                                      size_t count, float selectivity,
                                      int iterations = 5) {
    FilterTestResult result;
    result.scenario = scenario;

    // 生成数据: 范围 [0, 1000]
    auto data = generate_random_data(count, 0, 1000);

    // 计算阈值以达到目标选择率
    int32_t threshold = static_cast<int32_t>(1000 * (1.0f - selectivity));

    auto& optimizer = optimizer::QueryOptimizer::instance();

    // 设置特征
    stats::FilterCharacteristics chars;
    chars.row_count = count;
    chars.estimated_selectivity = selectivity;

    // 智能选择路径
    auto smart_path = optimizer.selectFilterPath(chars);
    result.smart_path = execution::getFilterPathName(smart_path);

    std::vector<uint32_t> indices(count);

    // 预热
    filter::filter_i32(data.data(), count, filter::CompareOp::GT, threshold, indices.data());

    // 测试智能选择
    Timer timer;
    double smart_total = 0;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        result.result_count = optimizer.executeFilter(data.data(), count,
                                                       filter::CompareOp::GT, threshold,
                                                       indices.data(), &chars);
        smart_total += timer.stop_ms();
    }
    result.smart_time_ms = smart_total / iterations;

    // 测试基础路径 (filter_i32)
    double v3_total = 0;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        filter::filter_i32(data.data(), count, filter::CompareOp::GT, threshold, indices.data());
        v3_total += timer.stop_ms();
    }
    result.v3_time_ms = v3_total / iterations;

    return result;
}

void run_filter_tests() {
    print_header("FILTER 性能测试 - 智能策略选择");

    std::vector<FilterTestResult> results;

    // 场景 1: 小数据
    std::cout << "  Testing F1: Small data (1M rows)..." << std::flush;
    results.push_back(test_filter_scenario("F1: 1M rows", 1000000, 0.5f));
    std::cout << " done\n";

    // 场景 2: 中等数据
    std::cout << "  Testing F2: Medium data (10M rows)..." << std::flush;
    results.push_back(test_filter_scenario("F2: 10M rows", 10000000, 0.5f));
    std::cout << " done\n";

    // 场景 3: 大数据 (应该选多线程或GPU)
    std::cout << "  Testing F3: Large data (50M rows)..." << std::flush;
    results.push_back(test_filter_scenario("F3: 50M rows", 50000000, 0.5f, 3));
    std::cout << " done\n";

    // 场景 4: 低选择率
    std::cout << "  Testing F4: Low selectivity (10M, 5%)..." << std::flush;
    results.push_back(test_filter_scenario("F4: Low sel (5%)", 10000000, 0.05f));
    std::cout << " done\n";

    // 场景 5: 高选择率
    std::cout << "  Testing F5: High selectivity (10M, 90%)..." << std::flush;
    results.push_back(test_filter_scenario("F5: High sel (90%)", 10000000, 0.9f));
    std::cout << " done\n";

    // 输出结果
    print_section("Filter 测试结果对比");
    std::cout << "│ " << std::left << std::setw(25) << "Scenario"
              << "│ " << std::setw(18) << "Smart Path"
              << "│ " << std::setw(10) << "Smart(ms)"
              << "│ " << std::setw(10) << "V3(ms)"
              << "│ " << std::setw(10) << "Speedup" << " │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";

    for (const auto& r : results) {
        double speedup = r.v3_time_ms / r.smart_time_ms;
        std::string speedup_str = (speedup >= 1.0) ?
            "\033[32m" + std::to_string(speedup).substr(0,4) + "x ↑\033[0m" :
            "\033[31m" + std::to_string(1.0/speedup).substr(0,4) + "x ↓\033[0m";

        std::cout << "│ " << std::left << std::setw(25) << r.scenario
                  << "│ " << std::setw(18) << r.smart_path
                  << "│ " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.smart_time_ms
                  << "│ " << std::setw(10) << r.v3_time_ms
                  << "│ " << std::setw(10) << speedup_str << " │\n";
    }
    std::cout << "└──────────────────────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// TopK 测试
// ============================================================================

struct TopKTestResult {
    std::string scenario;
    std::string smart_path;
    double smart_time_ms;
    double v3_time_ms;
    size_t k;
};

TopKTestResult test_topk_scenario(const std::string& scenario,
                                  size_t count, size_t k, float cardinality_ratio,
                                  int iterations = 5) {
    TopKTestResult result;
    result.scenario = scenario;
    result.k = k;

    // 根据基数生成数据
    std::vector<int32_t> data;
    if (cardinality_ratio < 0.01f) {
        // 低基数
        int32_t distinct = static_cast<int32_t>(count * cardinality_ratio);
        if (distinct < 100) distinct = 100;
        data = generate_low_cardinality_data(count, distinct);
    } else {
        // 高基数
        data = generate_random_data(count, 0, static_cast<int32_t>(count));
    }

    auto& optimizer = optimizer::QueryOptimizer::instance();

    // 设置特征
    stats::TopKCharacteristics chars;
    chars.row_count = count;
    chars.k = k;
    chars.cardinality_ratio = cardinality_ratio;

    // 智能选择路径
    auto smart_path = optimizer.selectTopKPath(chars);
    result.smart_path = execution::getTopKPathName(smart_path);

    std::vector<int32_t> values(k);
    std::vector<uint32_t> indices(k);

    // 预热
    sort::topk_max_i32_v3(data.data(), count, k, values.data(), indices.data());

    // 测试智能选择
    Timer timer;
    double smart_total = 0;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        optimizer.executeTopKMax(data.data(), count, k, values.data(), indices.data(), &chars);
        smart_total += timer.stop_ms();
    }
    result.smart_time_ms = smart_total / iterations;

    // 测试 v3 固定路径
    double v3_total = 0;
    for (int i = 0; i < iterations; i++) {
        timer.start();
        sort::topk_max_i32_v3(data.data(), count, k, values.data(), indices.data());
        v3_total += timer.stop_ms();
    }
    result.v3_time_ms = v3_total / iterations;

    return result;
}

void run_topk_tests() {
    print_header("TOPK 性能测试 - 智能策略选择");

    std::vector<TopKTestResult> results;

    // 场景 1: 小K，高基数
    std::cout << "  Testing T1: Small K (10M, k=10, high card)..." << std::flush;
    results.push_back(test_topk_scenario("T1: k=10, high card", 10000000, 10, 0.8f));
    std::cout << " done\n";

    // 场景 2: 中K，高基数
    std::cout << "  Testing T2: Medium K (10M, k=100, high card)..." << std::flush;
    results.push_back(test_topk_scenario("T2: k=100, high card", 10000000, 100, 0.8f));
    std::cout << " done\n";

    // 场景 3: 大K，高基数
    std::cout << "  Testing T3: Large K (10M, k=1000, high card)..." << std::flush;
    results.push_back(test_topk_scenario("T3: k=1000, high card", 10000000, 1000, 0.8f));
    std::cout << " done\n";

    // 场景 4: 低基数 (应该选 counting sort)
    std::cout << "  Testing T4: Low cardinality (10M, k=100, 0.1%)..." << std::flush;
    results.push_back(test_topk_scenario("T4: k=100, low card", 10000000, 100, 0.001f));
    std::cout << " done\n";

    // 场景 5: 超大数据
    std::cout << "  Testing T5: Large data (50M, k=100)..." << std::flush;
    results.push_back(test_topk_scenario("T5: 50M, k=100", 50000000, 100, 0.5f, 3));
    std::cout << " done\n";

    // 输出结果
    print_section("TopK 测试结果对比");
    std::cout << "│ " << std::left << std::setw(25) << "Scenario"
              << "│ " << std::setw(15) << "Smart Path"
              << "│ " << std::setw(10) << "Smart(ms)"
              << "│ " << std::setw(10) << "V3(ms)"
              << "│ " << std::setw(10) << "Speedup" << " │\n";
    std::cout << "├──────────────────────────────────────────────────────────────────────────────┤\n";

    for (const auto& r : results) {
        double speedup = r.v3_time_ms / r.smart_time_ms;
        std::string speedup_str = (speedup >= 1.0) ?
            "\033[32m" + std::to_string(speedup).substr(0,4) + "x ↑\033[0m" :
            "\033[31m" + std::to_string(1.0/speedup).substr(0,4) + "x ↓\033[0m";

        std::cout << "│ " << std::left << std::setw(25) << r.scenario
                  << "│ " << std::setw(15) << r.smart_path
                  << "│ " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.smart_time_ms
                  << "│ " << std::setw(10) << r.v3_time_ms
                  << "│ " << std::setw(10) << speedup_str << " │\n";
    }
    std::cout << "└──────────────────────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// DuckDB 对比测试
// ============================================================================

void run_duckdb_comparison() {
    print_header("DuckDB 对比测试 - 智能策略 vs DuckDB");

    std::cout << "\n使用 benchmark_app 进行完整对比测试...\n";
    std::cout << "(运行 ./build/benchmark_app --test all --size 1000000)\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           ThunderDuck 智能策略选择系统 - 综合性能测试                        ║\n";
    std::cout << "║                    Intelligent Strategy Selection Benchmark                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Version:  1.0.0                                                             ║\n";
    std::cout << "║  Platform: Apple Silicon M4                                                  ║\n";
    std::cout << "║  Features: SIMD (Neon), GPU (Metal), NPU (BNNS), Adaptive Strategy          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    // 显示系统配置
    auto& optimizer = optimizer::QueryOptimizer::instance();
    auto& stats_mgr = runtime::RuntimeStatsManager::instance();

    std::cout << "\n系统配置:\n";
    std::cout << "  GPU Available: " << (optimizer.isGpuAvailable() ? "Yes" : "No") << "\n";
    std::cout << "  NPU Available: " << (optimizer.isNpuAvailable() ? "Yes" : "No") << "\n";
    std::cout << "  Adaptive Mode: " << (stats_mgr.isAdaptiveEnabled() ? "Enabled" : "Disabled") << "\n";

    auto thresholds = stats_mgr.getThresholds();
    std::cout << "\n当前阈值设置:\n";
    std::cout << "  Join GPU Min Probe:    " << thresholds.join_gpu_min_probe << "\n";
    std::cout << "  Join Bloom Selectivity: " << thresholds.join_bloom_selectivity << "\n";
    std::cout << "  Filter MT Min:         " << thresholds.filter_mt_min << "\n";
    std::cout << "  Filter GPU Min:        " << thresholds.filter_gpu_min << "\n";
    std::cout << "  TopK GPU Min:          " << thresholds.topk_gpu_min << "\n";
    std::cout << "  TopK Low Cardinality:  " << thresholds.topk_low_cardinality << "\n";

    // 运行测试
    run_join_tests();
    run_filter_tests();
    run_topk_tests();

    // 总结
    print_header("测试总结");
    std::cout << R"(
智能策略选择系统根据数据特征自动选择最优执行路径:

┌─────────────────────────────────────────────────────────────────────────────┐
│                            路径选择策略                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ JOIN:                                                                       │
│   • 小数据 (<10K build)      → CPU_V3_RADIX16  (低开销)                     │
│   • 中等数据                  → CPU_V4_RADIX256 (缓存友好)                   │
│   • 低选择率 (<10%)          → CPU_V4_BLOOM    (预过滤)                     │
│   • 大数据 (>1M probe)       → GPU_UMA_DIRECT  (并行加速)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ FILTER:                                                                     │
│   • 小数据 (<5M)             → CPU_V3_SIMD     (单线程SIMD)                 │
│   • 中等数据 (5M-10M)        → CPU_V5_MT       (多线程)                     │
│   • 大数据 (>10M)            → GPU_SCAN        (GPU加速)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ TOPK:                                                                       │
│   • 高基数，小K              → CPU_V4_SAMPLE   (采样预过滤)                 │
│   • 低基数 (<1%)             → CPU_V5_COUNT    (计数排序)                   │
│   • 大数据 (>50M)            → GPU_FILTER      (GPU加速)                    │
└─────────────────────────────────────────────────────────────────────────────┘
)";

    std::cout << "\n测试完成!\n";
    return 0;
}
