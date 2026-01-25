/**
 * Hash Join v4.0 全面性能测试
 *
 * 测试维度:
 * - 所有 v4 策略 (RADIX256, BLOOMFILTER, NPU, GPU)
 * - 多种数据规模 (10K×100K, 100K×1M, 1M×10M)
 * - 与 v3 和 DuckDB 对比
 * - 策略自动选择验证
 */

#include <thunderduck/join.h>
#include <thunderduck/bloom_filter.h>
#include <duckdb.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <map>

using namespace thunderduck::join;

// ============================================================================
// 计时器
// ============================================================================

class PrecisionTimer {
    std::chrono::high_resolution_clock::time_point start_;
    std::vector<double> samples_;

public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        samples_.push_back(ms);
        return ms;
    }

    void reset() { samples_.clear(); }

    double min() const {
        return samples_.empty() ? 0 : *std::min_element(samples_.begin(), samples_.end());
    }

    double max() const {
        return samples_.empty() ? 0 : *std::max_element(samples_.begin(), samples_.end());
    }

    double avg() const {
        if (samples_.empty()) return 0;
        return std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
    }

    double median() const {
        if (samples_.empty()) return 0;
        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        return n % 2 == 0 ? (sorted[n/2-1] + sorted[n/2]) / 2 : sorted[n/2];
    }
};

// ============================================================================
// 测试结果
// ============================================================================

struct JoinTestResult {
    std::string test_id;
    std::string scale;
    std::string strategy;
    size_t build_count;
    size_t probe_count;
    size_t match_count;

    double time_ms;
    double time_min_ms;
    double time_max_ms;

    double throughput_mps;  // Million probes per second
    double speedup_vs_v3;
    double speedup_vs_duckdb;
};

std::vector<JoinTestResult> all_results;

// ============================================================================
// 数据生成
// ============================================================================

std::vector<int32_t> generate_keys(size_t count, int32_t min_val, int32_t max_val, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);

    std::vector<int32_t> keys(count);
    for (size_t i = 0; i < count; ++i) {
        keys[i] = dist(gen);
    }
    return keys;
}

std::vector<int32_t> generate_unique_keys(size_t count, int seed = 42) {
    std::vector<int32_t> keys(count);
    for (size_t i = 0; i < count; ++i) {
        keys[i] = static_cast<int32_t>(i + 1);
    }

    std::mt19937 gen(seed);
    std::shuffle(keys.begin(), keys.end(), gen);

    return keys;
}

// 参考答案
size_t compute_reference_join(const std::vector<int32_t>& build_keys,
                               const std::vector<int32_t>& probe_keys) {
    std::unordered_multiset<int32_t> build_set(build_keys.begin(), build_keys.end());

    size_t matches = 0;
    for (int32_t key : probe_keys) {
        matches += build_set.count(key);
    }
    return matches;
}

// ============================================================================
// 格式化
// ============================================================================

std::string format_number(size_t n) {
    if (n >= 1000000000) return std::to_string(n / 1000000000) + "B";
    if (n >= 1000000) return std::to_string(n / 1000000) + "M";
    if (n >= 1000) return std::to_string(n / 1000) + "K";
    return std::to_string(n);
}

std::string format_time(double ms) {
    std::ostringstream oss;
    if (ms < 1) {
        oss << std::fixed << std::setprecision(3) << ms << " ms";
    } else if (ms < 1000) {
        oss << std::fixed << std::setprecision(2) << ms << " ms";
    } else {
        oss << std::fixed << std::setprecision(2) << ms / 1000 << " s";
    }
    return oss.str();
}

const char* strategy_name(JoinStrategy s) {
    switch (s) {
        case JoinStrategy::AUTO: return "AUTO";
        case JoinStrategy::RADIX256: return "RADIX256";
        case JoinStrategy::BLOOMFILTER: return "BLOOMFILTER";
        case JoinStrategy::NPU: return "NPU";
        case JoinStrategy::GPU: return "GPU";
        case JoinStrategy::V3_FALLBACK: return "V3_FALLBACK";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 主测试类
// ============================================================================

class JoinV4Benchmark {
    const int WARMUP = 3;
    const int ITERATIONS = 10;

    duckdb::DuckDB db_;
    duckdb::Connection conn_;

    // 测试数据
    std::vector<int32_t> build_small_;   // 10K
    std::vector<int32_t> build_medium_;  // 100K
    std::vector<int32_t> build_large_;   // 1M
    std::vector<int32_t> probe_small_;   // 100K
    std::vector<int32_t> probe_medium_;  // 1M
    std::vector<int32_t> probe_large_;   // 10M

public:
    JoinV4Benchmark() : db_(nullptr), conn_(db_) {}

    void run_all() {
        print_header();
        setup_data();
        setup_duckdb();

        std::cout << "\n";
        print_section("正确性验证");
        verify_correctness();

        std::cout << "\n";
        print_section("策略可用性检测");
        check_strategy_availability();

        std::cout << "\n";
        print_section("性能测试: J1 (10K × 100K)");
        run_scale_tests("J1", build_small_, probe_small_, "build_small", "probe_small");

        std::cout << "\n";
        print_section("性能测试: J2 (100K × 1M)");
        run_scale_tests("J2", build_medium_, probe_medium_, "build_medium", "probe_medium");

        std::cout << "\n";
        print_section("性能测试: J3 (1M × 10M)");
        run_scale_tests("J3", build_large_, probe_large_, "build_large", "probe_large");

        std::cout << "\n";
        print_section("AUTO 策略选择验证");
        test_auto_strategy();

        std::cout << "\n";
        print_summary();
        generate_report();
    }

private:
    void print_header() {
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Hash Join v4.0 全面性能测试                                     ║\n";
        std::cout << "║                                                                              ║\n";
        std::cout << "║  策略: RADIX256 | BLOOMFILTER | NPU | GPU | AUTO                             ║\n";
        std::cout << "║  平台: Apple Silicon M4 | ARM Neon | Metal GPU | Accelerate NPU             ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    }

    void print_section(const std::string& title) {
        std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ " << std::left << std::setw(75) << title << "│\n";
        std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n\n";
    }

    void setup_data() {
        std::cout << "\n[1/4] 生成测试数据...\n";

        build_small_ = generate_unique_keys(10000, 42);
        build_medium_ = generate_unique_keys(100000, 42);
        build_large_ = generate_unique_keys(1000000, 42);

        probe_small_ = generate_keys(100000, 1, 10000, 123);
        probe_medium_ = generate_keys(1000000, 1, 100000, 123);
        probe_large_ = generate_keys(10000000, 1, 1000000, 123);

        std::cout << "  ✓ J1: 10K build × 100K probe\n";
        std::cout << "  ✓ J2: 100K build × 1M probe\n";
        std::cout << "  ✓ J3: 1M build × 10M probe\n";
    }

    void setup_duckdb() {
        std::cout << "\n[2/4] 初始化 DuckDB...\n";

        conn_.Query("CREATE TABLE build_small (key INTEGER)");
        conn_.Query("CREATE TABLE build_medium (key INTEGER)");
        conn_.Query("CREATE TABLE build_large (key INTEGER)");
        conn_.Query("CREATE TABLE probe_small (key INTEGER)");
        conn_.Query("CREATE TABLE probe_medium (key INTEGER)");
        conn_.Query("CREATE TABLE probe_large (key INTEGER)");

        load_keys("build_small", build_small_);
        load_keys("build_medium", build_medium_);
        load_keys("build_large", build_large_);
        load_keys("probe_small", probe_small_);
        load_keys("probe_medium", probe_medium_);
        load_keys("probe_large", probe_large_);

        std::cout << "  ✓ DuckDB 加载完成\n";
    }

    void load_keys(const std::string& table, const std::vector<int32_t>& keys) {
        duckdb::Appender appender(conn_, table);
        for (int32_t k : keys) {
            appender.BeginRow();
            appender.Append<int32_t>(k);
            appender.EndRow();
        }
        appender.Close();
    }

    void verify_correctness() {
        std::cout << "验证各策略结果一致性...\n\n";

        struct TestCase {
            const std::vector<int32_t>* build;
            const std::vector<int32_t>* probe;
            const char* name;
        };

        std::vector<TestCase> cases = {
            {&build_small_, &probe_small_, "J1 (10K×100K)"},
            {&build_medium_, &probe_medium_, "J2 (100K×1M)"},
        };

        for (const auto& tc : cases) {
            size_t expected = compute_reference_join(*tc.build, *tc.probe);

            // 测试 v3
            JoinResult* result_v3 = create_join_result(tc.probe->size() * 2);
            size_t v3_count = hash_join_i32_v3(
                tc.build->data(), tc.build->size(),
                tc.probe->data(), tc.probe->size(),
                JoinType::INNER, result_v3);
            free_join_result(result_v3);

            // 测试 v4 各策略
            std::vector<std::pair<JoinStrategy, size_t>> v4_results;

            JoinStrategy strategies[] = {
                JoinStrategy::RADIX256,
                JoinStrategy::BLOOMFILTER,
                JoinStrategy::AUTO
            };

            for (JoinStrategy s : strategies) {
                if (is_strategy_available(s)) {
                    JoinResult* result = create_join_result(tc.probe->size() * 2);
                    JoinConfigV4 config;
                    config.strategy = s;

                    size_t count = hash_join_i32_v4_config(
                        tc.build->data(), tc.build->size(),
                        tc.probe->data(), tc.probe->size(),
                        JoinType::INNER, result, config);

                    v4_results.push_back({s, count});
                    free_join_result(result);
                }
            }

            // 输出结果
            std::cout << "  " << std::left << std::setw(15) << tc.name
                      << " | Expected: " << std::setw(10) << expected
                      << " | v3: " << std::setw(10) << v3_count;

            bool all_correct = (v3_count == expected);
            for (const auto& [s, count] : v4_results) {
                std::cout << " | " << strategy_name(s) << ": " << count;
                if (count != expected) all_correct = false;
            }

            std::cout << (all_correct ? " ✓" : " ✗") << "\n";
        }
    }

    void check_strategy_availability() {
        std::cout << "检测各策略可用性...\n\n";

        JoinStrategy strategies[] = {
            JoinStrategy::V3_FALLBACK,
            JoinStrategy::RADIX256,
            JoinStrategy::BLOOMFILTER,
            JoinStrategy::NPU,
            JoinStrategy::GPU
        };

        for (JoinStrategy s : strategies) {
            bool available = is_strategy_available(s);
            std::cout << "  " << std::left << std::setw(15) << strategy_name(s)
                      << ": " << (available ? "\033[32m可用\033[0m" : "\033[31m不可用\033[0m") << "\n";
        }
    }

    void run_scale_tests(const std::string& scale_id,
                          const std::vector<int32_t>& build_keys,
                          const std::vector<int32_t>& probe_keys,
                          const std::string& build_table,
                          const std::string& probe_table) {

        // DuckDB 基准
        double duckdb_time = benchmark_duckdb(build_table, probe_table);

        // v3 基准
        double v3_time = benchmark_v3(build_keys, probe_keys);

        std::cout << "基准测试:\n";
        std::cout << "  DuckDB:     " << std::fixed << std::setprecision(3) << duckdb_time << " ms\n";
        std::cout << "  v3:         " << std::fixed << std::setprecision(3) << v3_time << " ms"
                  << " (vs DuckDB: " << std::setprecision(2) << (duckdb_time / v3_time) << "x)\n\n";

        // 测试各 v4 策略
        std::cout << "v4 策略性能:\n";
        std::cout << std::left << std::setw(15) << "策略"
                  << std::right << std::setw(12) << "时间 (ms)"
                  << std::setw(12) << "vs v3"
                  << std::setw(12) << "vs DuckDB"
                  << std::setw(15) << "吞吐量 (M/s)"
                  << std::setw(12) << "匹配数" << "\n";
        std::cout << std::string(78, '-') << "\n";

        JoinStrategy strategies[] = {
            JoinStrategy::RADIX256,
            JoinStrategy::BLOOMFILTER,
            JoinStrategy::NPU,
            JoinStrategy::GPU,
            JoinStrategy::AUTO
        };

        for (JoinStrategy s : strategies) {
            if (!is_strategy_available(s) && s != JoinStrategy::AUTO) {
                std::cout << std::left << std::setw(15) << strategy_name(s)
                          << " (不可用)\n";
                continue;
            }

            auto result = benchmark_v4_strategy(scale_id, s, build_keys, probe_keys,
                                                 v3_time, duckdb_time);
            all_results.push_back(result);

            std::cout << std::left << std::setw(15) << result.strategy
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(12) << result.time_ms
                      << std::setw(11) << std::setprecision(2) << result.speedup_vs_v3 << "x"
                      << std::setw(11) << result.speedup_vs_duckdb << "x"
                      << std::setw(15) << std::setprecision(1) << result.throughput_mps
                      << std::setw(12) << format_number(result.match_count) << "\n";
        }
    }

    double benchmark_duckdb(const std::string& build_table, const std::string& probe_table) {
        std::string sql = "SELECT COUNT(*) FROM " + build_table + " b INNER JOIN " +
                          probe_table + " p ON b.key = p.key";

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            conn_.Query(sql);
        }

        PrecisionTimer timer;
        for (int i = 0; i < ITERATIONS; ++i) {
            timer.start();
            conn_.Query(sql);
            timer.stop();
        }

        return timer.avg();
    }

    double benchmark_v3(const std::vector<int32_t>& build_keys,
                        const std::vector<int32_t>& probe_keys) {
        JoinResult* result = create_join_result(probe_keys.size() * 2);

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            result->count = 0;
            hash_join_i32_v3(build_keys.data(), build_keys.size(),
                             probe_keys.data(), probe_keys.size(),
                             JoinType::INNER, result);
        }

        PrecisionTimer timer;
        for (int i = 0; i < ITERATIONS; ++i) {
            result->count = 0;
            timer.start();
            hash_join_i32_v3(build_keys.data(), build_keys.size(),
                             probe_keys.data(), probe_keys.size(),
                             JoinType::INNER, result);
            timer.stop();
        }

        free_join_result(result);
        return timer.avg();
    }

    JoinTestResult benchmark_v4_strategy(const std::string& scale_id,
                                          JoinStrategy strategy,
                                          const std::vector<int32_t>& build_keys,
                                          const std::vector<int32_t>& probe_keys,
                                          double v3_time,
                                          double duckdb_time) {
        JoinTestResult result;
        result.test_id = scale_id;
        result.scale = scale_id;
        result.strategy = strategy_name(strategy);
        result.build_count = build_keys.size();
        result.probe_count = probe_keys.size();

        JoinResult* join_result = create_join_result(probe_keys.size() * 2);
        JoinConfigV4 config;
        config.strategy = strategy;

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            join_result->count = 0;
            hash_join_i32_v4_config(build_keys.data(), build_keys.size(),
                                     probe_keys.data(), probe_keys.size(),
                                     JoinType::INNER, join_result, config);
        }

        PrecisionTimer timer;
        for (int i = 0; i < ITERATIONS; ++i) {
            join_result->count = 0;
            timer.start();
            result.match_count = hash_join_i32_v4_config(
                build_keys.data(), build_keys.size(),
                probe_keys.data(), probe_keys.size(),
                JoinType::INNER, join_result, config);
            timer.stop();
        }

        result.time_ms = timer.avg();
        result.time_min_ms = timer.min();
        result.time_max_ms = timer.max();
        result.throughput_mps = (probe_keys.size() / result.time_ms) / 1000.0;  // Million/s
        result.speedup_vs_v3 = v3_time / result.time_ms;
        result.speedup_vs_duckdb = duckdb_time / result.time_ms;

        free_join_result(join_result);
        return result;
    }

    void test_auto_strategy() {
        std::cout << "验证 AUTO 策略自动选择...\n\n";

        struct TestCase {
            const std::vector<int32_t>* build;
            const std::vector<int32_t>* probe;
            const char* name;
        };

        std::vector<TestCase> cases = {
            {&build_small_, &probe_small_, "J1 (10K×100K)"},
            {&build_medium_, &probe_medium_, "J2 (100K×1M)"},
            {&build_large_, &probe_large_, "J3 (1M×10M)"},
        };

        for (const auto& tc : cases) {
            JoinConfigV4 config;
            config.strategy = JoinStrategy::AUTO;

            const char* selected = get_selected_strategy_name(
                tc.build->size(), tc.probe->size(), config);

            std::cout << "  " << std::left << std::setw(15) << tc.name
                      << " → 选择: " << selected << "\n";
        }
    }

    void print_summary() {
        print_section("性能总结");

        // 按 scale 分组统计
        std::map<std::string, std::vector<JoinTestResult>> by_scale;
        for (const auto& r : all_results) {
            by_scale[r.scale].push_back(r);
        }

        std::cout << "各规模最佳策略:\n\n";
        std::cout << std::left << std::setw(10) << "规模"
                  << std::setw(15) << "最佳策略"
                  << std::right << std::setw(12) << "时间"
                  << std::setw(12) << "vs v3"
                  << std::setw(12) << "vs DuckDB" << "\n";
        std::cout << std::string(61, '-') << "\n";

        for (const auto& [scale, results] : by_scale) {
            auto best = std::min_element(results.begin(), results.end(),
                [](const auto& a, const auto& b) { return a.time_ms < b.time_ms; });

            std::cout << std::left << std::setw(10) << scale
                      << std::setw(15) << best->strategy
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << format_time(best->time_ms)
                      << std::setw(11) << best->speedup_vs_v3 << "x"
                      << std::setw(11) << best->speedup_vs_duckdb << "x" << "\n";
        }

        std::cout << "\n";

        // 总体统计
        double avg_vs_v3 = 0, avg_vs_duckdb = 0;
        int count = 0;
        for (const auto& r : all_results) {
            if (r.strategy != "AUTO") {
                avg_vs_v3 += r.speedup_vs_v3;
                avg_vs_duckdb += r.speedup_vs_duckdb;
                count++;
            }
        }

        if (count > 0) {
            avg_vs_v3 /= count;
            avg_vs_duckdb /= count;
        }

        std::cout << "总体统计:\n";
        std::cout << "  • v4 vs v3 平均加速比:     " << std::fixed << std::setprecision(2) << avg_vs_v3 << "x\n";
        std::cout << "  • v4 vs DuckDB 平均加速比: " << std::fixed << std::setprecision(2) << avg_vs_duckdb << "x\n";
    }

    void generate_report() {
        std::cout << "\n正在生成报告...\n";

        // 生成 Markdown 格式的测试数据供后续更新 DETAILED_BENCHMARK_REPORT.md
        std::ofstream report("join_v4_benchmark_results.md");

        report << "# Hash Join v4 性能测试结果\n\n";
        report << "> **测试时间**: " << __DATE__ << " " << __TIME__ << "\n";
        report << "> **平台**: Apple Silicon M4 | ARM Neon | Metal | Accelerate\n\n";

        report << "## 详细测试结果\n\n";
        report << "| 规模 | 策略 | 时间 (ms) | vs v3 | vs DuckDB | 吞吐量 (M/s) | 匹配数 |\n";
        report << "|------|------|-----------|-------|-----------|-------------|--------|\n";

        for (const auto& r : all_results) {
            report << "| " << r.scale
                   << " | " << r.strategy
                   << " | " << std::fixed << std::setprecision(3) << r.time_ms
                   << " | " << std::setprecision(2) << r.speedup_vs_v3 << "x"
                   << " | " << r.speedup_vs_duckdb << "x"
                   << " | " << std::setprecision(1) << r.throughput_mps
                   << " | " << format_number(r.match_count) << " |\n";
        }

        report << "\n## 策略可用性\n\n";
        report << "| 策略 | 状态 |\n";
        report << "|------|------|\n";

        JoinStrategy strategies[] = {
            JoinStrategy::V3_FALLBACK,
            JoinStrategy::RADIX256,
            JoinStrategy::BLOOMFILTER,
            JoinStrategy::NPU,
            JoinStrategy::GPU
        };

        for (JoinStrategy s : strategies) {
            report << "| " << strategy_name(s)
                   << " | " << (is_strategy_available(s) ? "✅ 可用" : "❌ 不可用") << " |\n";
        }

        report.close();

        std::cout << "  ✓ 报告已保存到: join_v4_benchmark_results.md\n";
    }
};

// ============================================================================
// 主函数
// ============================================================================

int main() {
    JoinV4Benchmark benchmark;
    benchmark.run_all();
    return 0;
}
