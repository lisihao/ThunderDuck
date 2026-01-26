/**
 * ThunderDuck V5 Comprehensive Benchmark
 *
 * 全面对比所有版本的性能:
 * - Filter: v2, v3, v4
 * - Aggregate: v2, v3
 * - TopK: v2, v3, v4, v5, v6
 * - Hash Join: v2, v3, v4, GPU-v5, GPU-UMA
 *
 * 测试指标:
 * - SQL 语义
 * - 数据量
 * - 硬件路径 (CPU/GPU)
 * - 执行时间
 * - 数据吞吐带宽
 * - vs DuckDB 加速比
 */

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace thunderduck;

// ============================================================================
// 配置
// ============================================================================

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURE_RUNS = 5;
constexpr double M4_MEMORY_BW_GBS = 400.0;

// ============================================================================
// 计时工具
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// DuckDB 基线模拟
// ============================================================================

namespace baseline {

size_t filter_gt(const int32_t* data, size_t count, int32_t threshold, uint32_t* indices) {
    size_t result = 0;
    for (size_t i = 0; i < count; ++i) {
        if (data[i] > threshold) indices[result++] = static_cast<uint32_t>(i);
    }
    return result;
}

int64_t sum_i32(const int32_t* data, size_t count) {
    int64_t sum = 0;
    for (size_t i = 0; i < count; ++i) sum += data[i];
    return sum;
}

void topk(const int32_t* data, size_t count, size_t k, int32_t* result) {
    std::vector<int32_t> tmp(data, data + count);
    std::partial_sort(tmp.begin(), tmp.begin() + k, tmp.end(), std::greater<int32_t>());
    std::copy(tmp.begin(), tmp.begin() + k, result);
}

size_t hash_join(const int32_t* build, size_t bc, const int32_t* probe, size_t pc,
                  uint32_t* bi, uint32_t* pi) {
    std::vector<std::pair<int32_t, uint32_t>> ht;
    ht.reserve(bc);
    for (size_t i = 0; i < bc; ++i) ht.push_back({build[i], static_cast<uint32_t>(i)});
    std::sort(ht.begin(), ht.end());

    size_t matches = 0;
    for (size_t i = 0; i < pc; ++i) {
        auto range = std::equal_range(ht.begin(), ht.end(),
            std::make_pair(probe[i], 0u),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto it = range.first; it != range.second; ++it) {
            bi[matches] = it->second;
            pi[matches] = static_cast<uint32_t>(i);
            ++matches;
        }
    }
    return matches;
}

} // namespace baseline

// ============================================================================
// 数据生成
// ============================================================================

void gen_data(int32_t* data, size_t count, int32_t min_val, int32_t max_val, std::mt19937& rng) {
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) data[i] = dist(rng);
}

// ============================================================================
// 打印工具
// ============================================================================

void print_header(const char* title) {
    printf("\n");
    printf("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n");
    printf("┃ %-125s ┃\n", title);
    printf("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n");
}

void print_table_header() {
    printf("┌──────────────────────────────────────────────┬────────────┬──────────┬────────────┬────────────┬──────────┬───────────┬───────────┐\n");
    printf("│ SQL 语义                                     │ 数据量     │ 硬件路径 │ 时间(μs)   │ 带宽(GB/s) │ 带宽利用 │ vs 基准   │ vs DuckDB │\n");
    printf("├──────────────────────────────────────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼───────────┼───────────┤\n");
}

void print_row(const char* sql, size_t data_bytes, const char* hw, double time_us,
               double bandwidth, double bw_util, double vs_base, double vs_duckdb) {
    char size_str[16];
    if (data_bytes >= 1024 * 1024) {
        snprintf(size_str, sizeof(size_str), "%.1f MB", data_bytes / (1024.0 * 1024.0));
    } else {
        snprintf(size_str, sizeof(size_str), "%.1f KB", data_bytes / 1024.0);
    }
    printf("│ %-44s │ %10s │ %-8s │ %10.1f │ %10.2f │ %7.1f%% │ %8.2fx │ %8.1fx │\n",
           sql, size_str, hw, time_us, bandwidth, bw_util, vs_base, vs_duckdb);
}

void print_table_footer() {
    printf("└──────────────────────────────────────────────┴────────────┴──────────┴────────────┴────────────┴──────────┴───────────┴───────────┘\n");
}

// ============================================================================
// Filter 测试
// ============================================================================

void run_filter_benchmarks(std::mt19937& rng) {
    print_header("Filter 算子版本对比 (v2/v3/v4)");
    print_table_header();

    struct Test { size_t count; int32_t thresh; const char* sql; };
    std::vector<Test> tests = {
        {1000000,  500, "SELECT * WHERE val > 500 (1M 50%)"},
        {5000000,  500, "SELECT * WHERE val > 500 (5M 50%)"},
        {10000000, 500, "SELECT * WHERE val > 500 (10M 50%)"},
        {10000000, 900, "SELECT * WHERE val > 900 (10M 10%)"},
    };

    for (const auto& t : tests) {
        std::vector<int32_t> data(t.count);
        gen_data(data.data(), t.count, 0, 1000, rng);
        std::vector<uint32_t> indices(t.count);
        size_t data_bytes = t.count * sizeof(int32_t);

        Timer timer;

        // DuckDB baseline
        double dk_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            baseline::filter_gt(data.data(), t.count, t.thresh, indices.data());
            dk_time += timer.elapsed_us();
        }
        dk_time /= MEASURE_RUNS;

        // v2
        for (int i = 0; i < WARMUP_RUNS; ++i)
            filter::filter_i32_v2(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
        double v2_time = 0;
        size_t result_count = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            result_count = filter::filter_i32_v2(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
            v2_time += timer.elapsed_us();
        }
        v2_time /= MEASURE_RUNS;
        double v2_bw = (data_bytes + result_count * 4) / (v2_time * 1000.0);
        print_row(t.sql, data_bytes, "v2 Neon", v2_time, v2_bw, v2_bw / M4_MEMORY_BW_GBS * 100, 1.0, dk_time / v2_time);

        // v3
        for (int i = 0; i < WARMUP_RUNS; ++i)
            filter::filter_i32_v3(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
        double v3_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            result_count = filter::filter_i32_v3(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
            v3_time += timer.elapsed_us();
        }
        v3_time /= MEASURE_RUNS;
        double v3_bw = (data_bytes + result_count * 4) / (v3_time * 1000.0);
        print_row(t.sql, data_bytes, "v3 Neon", v3_time, v3_bw, v3_bw / M4_MEMORY_BW_GBS * 100, v2_time / v3_time, dk_time / v3_time);

        // v4
        for (int i = 0; i < WARMUP_RUNS; ++i)
            filter::filter_i32_v4(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
        double v4_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            result_count = filter::filter_i32_v4(data.data(), t.count, filter::CompareOp::GT, t.thresh, indices.data());
            v4_time += timer.elapsed_us();
        }
        v4_time /= MEASURE_RUNS;
        double v4_bw = (data_bytes + result_count * 4) / (v4_time * 1000.0);
        print_row(t.sql, data_bytes, "v4 Auto", v4_time, v4_bw, v4_bw / M4_MEMORY_BW_GBS * 100, v2_time / v4_time, dk_time / v4_time);

        printf("├──────────────────────────────────────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼───────────┼───────────┤\n");
    }
    print_table_footer();
}

// ============================================================================
// Aggregate 测试
// ============================================================================

void run_aggregate_benchmarks(std::mt19937& rng) {
    print_header("Aggregate 算子版本对比 (v2/v3)");
    print_table_header();

    struct Test { size_t count; const char* sql; };
    std::vector<Test> tests = {
        {1000000,  "SELECT SUM(val) FROM t (1M)"},
        {5000000,  "SELECT SUM(val) FROM t (5M)"},
        {10000000, "SELECT SUM(val) FROM t (10M)"},
    };

    for (const auto& t : tests) {
        std::vector<int32_t> data(t.count);
        gen_data(data.data(), t.count, 0, 1000, rng);
        size_t data_bytes = t.count * sizeof(int32_t);

        Timer timer;

        // DuckDB baseline
        double dk_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            baseline::sum_i32(data.data(), t.count);
            dk_time += timer.elapsed_us();
        }
        dk_time /= MEASURE_RUNS;

        // v2
        for (int i = 0; i < WARMUP_RUNS; ++i)
            aggregate::sum_i32_v2(data.data(), t.count);
        double v2_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            aggregate::sum_i32_v2(data.data(), t.count);
            v2_time += timer.elapsed_us();
        }
        v2_time /= MEASURE_RUNS;
        double v2_bw = data_bytes / (v2_time * 1000.0);
        print_row(t.sql, data_bytes, "v2 Neon", v2_time, v2_bw, v2_bw / M4_MEMORY_BW_GBS * 100, 1.0, dk_time / v2_time);

        // v3
        for (int i = 0; i < WARMUP_RUNS; ++i)
            aggregate::sum_i32_v3(data.data(), t.count);
        double v3_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            aggregate::sum_i32_v3(data.data(), t.count);
            v3_time += timer.elapsed_us();
        }
        v3_time /= MEASURE_RUNS;
        double v3_bw = data_bytes / (v3_time * 1000.0);
        print_row(t.sql, data_bytes, "v3 Neon", v3_time, v3_bw, v3_bw / M4_MEMORY_BW_GBS * 100, v2_time / v3_time, dk_time / v3_time);

        printf("├──────────────────────────────────────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼───────────┼───────────┤\n");
    }
    print_table_footer();
}

// ============================================================================
// TopK 测试
// ============================================================================

void run_topk_benchmarks(std::mt19937& rng) {
    print_header("TopK 算子版本对比 (v3/v4/v5/v6)");
    print_table_header();

    struct Test { size_t count; size_t k; const char* sql; };
    std::vector<Test> tests = {
        {100000,   10,   "SELECT TOP 10 FROM t (100K)"},
        {100000,   100,  "SELECT TOP 100 FROM t (100K)"},
        {1000000,  10,   "SELECT TOP 10 FROM t (1M)"},
        {1000000,  100,  "SELECT TOP 100 FROM t (1M)"},
        {5000000,  10,   "SELECT TOP 10 FROM t (5M)"},
        {10000000, 10,   "SELECT TOP 10 FROM t (10M)"},
        {10000000, 100,  "SELECT TOP 100 FROM t (10M)"},
        {10000000, 1000, "SELECT TOP 1000 FROM t (10M)"},
    };

    for (const auto& t : tests) {
        std::vector<int32_t> data(t.count);
        gen_data(data.data(), t.count, 0, 1000000, rng);
        std::vector<int32_t> result(t.k);
        size_t data_bytes = t.count * sizeof(int32_t);

        Timer timer;

        // DuckDB baseline
        double dk_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            baseline::topk(data.data(), t.count, t.k, result.data());
            dk_time += timer.elapsed_us();
        }
        dk_time /= MEASURE_RUNS;

        // v3
        for (int i = 0; i < WARMUP_RUNS; ++i)
            sort::topk_max_i32_v3(data.data(), t.count, t.k, result.data());
        double v3_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            sort::topk_max_i32_v3(data.data(), t.count, t.k, result.data());
            v3_time += timer.elapsed_us();
        }
        v3_time /= MEASURE_RUNS;
        double v3_bw = data_bytes / (v3_time * 1000.0);
        print_row(t.sql, data_bytes, "v3 Heap", v3_time, v3_bw, v3_bw / M4_MEMORY_BW_GBS * 100, 1.0, dk_time / v3_time);

        // v4
        for (int i = 0; i < WARMUP_RUNS; ++i)
            sort::topk_max_i32_v4(data.data(), t.count, t.k, result.data());
        double v4_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            sort::topk_max_i32_v4(data.data(), t.count, t.k, result.data());
            v4_time += timer.elapsed_us();
        }
        v4_time /= MEASURE_RUNS;
        double v4_bw = data_bytes / (v4_time * 1000.0);
        print_row(t.sql, data_bytes, "v4 Sample", v4_time, v4_bw, v4_bw / M4_MEMORY_BW_GBS * 100, v3_time / v4_time, dk_time / v4_time);

        // v5
        for (int i = 0; i < WARMUP_RUNS; ++i)
            sort::topk_max_i32_v5(data.data(), t.count, t.k, result.data());
        double v5_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            sort::topk_max_i32_v5(data.data(), t.count, t.k, result.data());
            v5_time += timer.elapsed_us();
        }
        v5_time /= MEASURE_RUNS;
        double v5_bw = data_bytes / (v5_time * 1000.0);
        print_row(t.sql, data_bytes, "v5 Auto", v5_time, v5_bw, v5_bw / M4_MEMORY_BW_GBS * 100, v3_time / v5_time, dk_time / v5_time);

        // v6
        for (int i = 0; i < WARMUP_RUNS; ++i)
            sort::topk_max_i32_v6(data.data(), t.count, t.k, result.data());
        double v6_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            sort::topk_max_i32_v6(data.data(), t.count, t.k, result.data());
            v6_time += timer.elapsed_us();
        }
        v6_time /= MEASURE_RUNS;
        double v6_bw = data_bytes / (v6_time * 1000.0);
        print_row(t.sql, data_bytes, "v6 GPU", v6_time, v6_bw, v6_bw / M4_MEMORY_BW_GBS * 100, v3_time / v6_time, dk_time / v6_time);

        printf("├──────────────────────────────────────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼───────────┼───────────┤\n");
    }
    print_table_footer();
}

// ============================================================================
// Hash Join 测试
// ============================================================================

void run_join_benchmarks(std::mt19937& rng) {
    print_header("Hash Join 算子版本对比 (v2/v3/v4/GPU-UMA)");
    print_table_header();

    struct Test { size_t bc; size_t pc; double match; const char* sql; };
    std::vector<Test> tests = {
        {10000,   100000,  1.0, "JOIN 10K×100K (full match)"},
        {10000,   100000,  0.1, "JOIN 10K×100K (10% match)"},
        {100000,  1000000, 1.0, "JOIN 100K×1M (full match)"},
        {100000,  1000000, 0.1, "JOIN 100K×1M (10% match)"},
        {1000000, 1000000, 1.0, "JOIN 1M×1M (full match)"},
        {1000000, 1000000, 0.1, "JOIN 1M×1M (10% match)"},
    };

    for (const auto& t : tests) {
        std::vector<int32_t> build(t.bc), probe(t.pc);
        for (size_t i = 0; i < t.bc; ++i) build[i] = static_cast<int32_t>(i);
        std::shuffle(build.begin(), build.end(), rng);

        size_t match_count = static_cast<size_t>(t.pc * t.match);
        for (size_t i = 0; i < match_count; ++i) probe[i] = build[rng() % t.bc];
        for (size_t i = match_count; i < t.pc; ++i) probe[i] = static_cast<int32_t>(t.bc + i);
        std::shuffle(probe.begin(), probe.end(), rng);

        size_t data_bytes = (t.bc + t.pc) * sizeof(int32_t);
        size_t max_matches = std::min(t.bc * 10, size_t(10000000));

        Timer timer;

        // DuckDB baseline
        std::vector<uint32_t> bi(max_matches), pi(max_matches);
        double dk_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            timer.start();
            baseline::hash_join(build.data(), t.bc, probe.data(), t.pc, bi.data(), pi.data());
            dk_time += timer.elapsed_us();
        }
        dk_time /= MEASURE_RUNS;

        join::JoinResult* result = join::create_join_result(max_matches);

        // v2
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            result->count = 0;
            join::hash_join_i32_v2(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
        }
        double v2_time = 0;
        size_t matches = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            result->count = 0;
            timer.start();
            matches = join::hash_join_i32_v2(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
            v2_time += timer.elapsed_us();
        }
        v2_time /= MEASURE_RUNS;
        double v2_bw = (data_bytes + matches * 8) / (v2_time * 1000.0);
        print_row(t.sql, data_bytes, "v2 Robin", v2_time, v2_bw, v2_bw / M4_MEMORY_BW_GBS * 100, 1.0, dk_time / v2_time);

        // v3
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            result->count = 0;
            join::hash_join_i32_v3(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
        }
        double v3_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            result->count = 0;
            timer.start();
            matches = join::hash_join_i32_v3(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
            v3_time += timer.elapsed_us();
        }
        v3_time /= MEASURE_RUNS;
        double v3_bw = (data_bytes + matches * 8) / (v3_time * 1000.0);
        print_row(t.sql, data_bytes, "v3 SOA", v3_time, v3_bw, v3_bw / M4_MEMORY_BW_GBS * 100, v2_time / v3_time, dk_time / v3_time);

        // v4
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            result->count = 0;
            join::hash_join_i32_v4(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
        }
        double v4_time = 0;
        for (int i = 0; i < MEASURE_RUNS; ++i) {
            result->count = 0;
            timer.start();
            matches = join::hash_join_i32_v4(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result);
            v4_time += timer.elapsed_us();
        }
        v4_time /= MEASURE_RUNS;
        double v4_bw = (data_bytes + matches * 8) / (v4_time * 1000.0);
        print_row(t.sql, data_bytes, "v4 Multi", v4_time, v4_bw, v4_bw / M4_MEMORY_BW_GBS * 100, v2_time / v4_time, dk_time / v4_time);

        // GPU-UMA (大规模测试)
        if (t.pc >= 1000000 && join::uma::is_uma_gpu_ready()) {
            join::JoinConfigV4 cfg;
            cfg.strategy = join::JoinStrategy::GPU;
            for (int i = 0; i < WARMUP_RUNS; ++i) {
                result->count = 0;
                join::uma::hash_join_gpu_uma(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result, cfg);
            }
            double gpu_time = 0;
            for (int i = 0; i < MEASURE_RUNS; ++i) {
                result->count = 0;
                timer.start();
                matches = join::uma::hash_join_gpu_uma(build.data(), t.bc, probe.data(), t.pc, join::JoinType::INNER, result, cfg);
                gpu_time += timer.elapsed_us();
            }
            gpu_time /= MEASURE_RUNS;
            double gpu_bw = (data_bytes + matches * 8) / (gpu_time * 1000.0);
            print_row(t.sql, data_bytes, "GPU UMA", gpu_time, gpu_bw, gpu_bw / M4_MEMORY_BW_GBS * 100, v2_time / gpu_time, dk_time / gpu_time);
        }

        join::free_join_result(result);
        printf("├──────────────────────────────────────────────┼────────────┼──────────┼────────────┼────────────┼──────────┼───────────┼───────────┤\n");
    }
    print_table_footer();
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                    ThunderDuck V5 Comprehensive Benchmark                                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  平台: Apple Silicon M4 (UMA)                                                                                                 ║\n");
    printf("║  理论内存带宽: 400 GB/s                                                                                                       ║\n");
    printf("║  预热: %d 次, 测量: %d 次                                                                                                      ║\n", WARMUP_RUNS, MEASURE_RUNS);
    printf("║  对比版本: v2 → v3 → v4 → v5 → v6 / GPU                                                                                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    std::mt19937 rng(42);

    run_filter_benchmarks(rng);
    run_aggregate_benchmarks(rng);
    run_topk_benchmarks(rng);
    run_join_benchmarks(rng);

    // 汇总
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                              版本演进分析                                                                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                                                                               ║\n");
    printf("║  Filter:     v2 (基线) → v3 (SIMD 批量) → v4 (自适应策略)                                                                     ║\n");
    printf("║  Aggregate:  v2 (基线) → v3 (循环展开 + 预取)                                                                                 ║\n");
    printf("║  TopK:       v3 (堆) → v4 (采样预过滤) → v5 (自适应) → v6 (GPU)                                                               ║\n");
    printf("║  Hash Join:  v2 (Robin Hood) → v3 (SOA 缓存优化) → v4 (多策略) → GPU-UMA                                                      ║\n");
    printf("║                                                                                                                               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    return 0;
}
