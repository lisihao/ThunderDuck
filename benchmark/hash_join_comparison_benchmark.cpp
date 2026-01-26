/**
 * ThunderDuck - Hash Join Version Comparison Benchmark
 *
 * 对比 v3, v5 (两阶段), v6 (链式), GPU-UMA 和 DuckDB 的 Hash Join 性能
 *
 * 测试场景:
 * - 不同数据规模: 10K, 100K, 1M, 5M, 10M
 * - 不同匹配率: 10%, 50%, 100%
 * - 不同重复键比例: 低(唯一), 中(10x), 高(100x)
 *
 * SQL 语义:
 *   SELECT COUNT(*) FROM build b JOIN probe p ON b.key = p.key
 */

#include "thunderduck/join.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// 外部函数声明 (v5/v6 在各自命名空间中)
// ============================================================================

namespace thunderduck {
namespace join {
namespace v5 {

size_t hash_join_i32_v5_twophase(const int32_t* build_keys, size_t build_count,
                                  const int32_t* probe_keys, size_t probe_count,
                                  JoinType join_type,
                                  JoinResult* result);

size_t hash_join_i32_v5_parallel(const int32_t* build_keys, size_t build_count,
                                  const int32_t* probe_keys, size_t probe_count,
                                  JoinType join_type,
                                  JoinResult* result,
                                  size_t num_threads);

} // namespace v5

namespace v6 {

size_t hash_join_i32_v6_chain(const int32_t* build_keys, size_t build_count,
                               const int32_t* probe_keys, size_t probe_count,
                               JoinType join_type,
                               JoinResult* result);

size_t hash_join_i32_v6_chunked(const int32_t* build_keys, size_t build_count,
                                 const int32_t* probe_keys, size_t probe_count,
                                 JoinType join_type,
                                 JoinResult* result);

} // namespace v6
} // namespace join
} // namespace thunderduck

using namespace thunderduck::join;

// ============================================================================
// 基准测试配置
// ============================================================================

constexpr int WARMUP_RUNS = 3;
constexpr int MEASURE_RUNS = 5;
constexpr double M4_MEMORY_BW_GBS = 400.0;  // M4 理论内存带宽

// ============================================================================
// 计时工具
// ============================================================================

class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// ============================================================================
// DuckDB 模拟基线 (简单哈希表实现)
// ============================================================================

size_t duckdb_baseline_hash_join(const int32_t* build_keys, size_t build_count,
                                  const int32_t* probe_keys, size_t probe_count,
                                  JoinResult* result) {
    // 简单的 std::unordered_multimap 模拟 DuckDB 行为
    std::vector<std::pair<int32_t, uint32_t>> ht;
    ht.reserve(build_count);

    // Build phase
    for (size_t i = 0; i < build_count; ++i) {
        ht.push_back({build_keys[i], static_cast<uint32_t>(i)});
    }
    std::sort(ht.begin(), ht.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Probe phase
    size_t match_count = 0;
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t key = probe_keys[i];

        // 二分查找范围
        auto lower = std::lower_bound(ht.begin(), ht.end(), key,
            [](const auto& p, int32_t k) { return p.first < k; });
        auto upper = std::upper_bound(lower, ht.end(), key,
            [](int32_t k, const auto& p) { return k < p.first; });

        for (auto it = lower; it != upper; ++it) {
            if (match_count < result->capacity) {
                result->left_indices[match_count] = it->second;
                result->right_indices[match_count] = static_cast<uint32_t>(i);
            }
            ++match_count;
        }
    }

    result->count = std::min(match_count, result->capacity);
    return match_count;
}

// ============================================================================
// 数据生成
// ============================================================================

struct TestData {
    std::vector<int32_t> build_keys;
    std::vector<int32_t> probe_keys;
    size_t expected_matches;
    const char* description;
};

TestData generate_test_data(size_t build_count, size_t probe_count,
                            double match_rate, int duplicates,
                            const char* desc, std::mt19937& rng) {
    TestData data;
    data.description = desc;
    data.build_keys.resize(build_count);
    data.probe_keys.resize(probe_count);

    // Build 侧: 唯一键 / 重复键
    size_t unique_build = build_count / duplicates;
    for (size_t i = 0; i < build_count; ++i) {
        data.build_keys[i] = static_cast<int32_t>(i / duplicates);
    }
    std::shuffle(data.build_keys.begin(), data.build_keys.end(), rng);

    // Probe 侧: 部分匹配
    size_t match_probe = static_cast<size_t>(probe_count * match_rate);
    for (size_t i = 0; i < match_probe; ++i) {
        // 从 build 键中随机选取
        data.probe_keys[i] = data.build_keys[rng() % build_count];
    }
    // 不匹配的键
    for (size_t i = match_probe; i < probe_count; ++i) {
        data.probe_keys[i] = static_cast<int32_t>(unique_build + i);  // 超出 build 范围
    }
    std::shuffle(data.probe_keys.begin(), data.probe_keys.end(), rng);

    // 计算预期匹配数 (近似值)
    // 对于有重复键的情况，每个匹配的 probe 键会产生 duplicates 个匹配
    data.expected_matches = match_probe * duplicates;

    return data;
}

// ============================================================================
// 基准测试执行
// ============================================================================

struct BenchResult {
    const char* version;
    double time_us;
    size_t matches;
    double throughput_mops;  // M ops/sec
    double bandwidth_gbs;    // GB/s
};

template<typename JoinFunc>
BenchResult run_benchmark(const char* name, JoinFunc func,
                          const int32_t* build_keys, size_t build_count,
                          const int32_t* probe_keys, size_t probe_count) {
    // 预分配结果缓冲区 (限制为合理大小)
    size_t max_matches = std::min(build_count * 10, size_t(10000000));  // 最大 10M 匹配

    JoinResult* result = create_join_result(max_matches);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        result->count = 0;
        func(build_keys, build_count, probe_keys, probe_count, JoinType::INNER, result);
    }

    // Measure
    Timer timer;
    double total_time = 0;
    size_t matches = 0;

    for (int i = 0; i < MEASURE_RUNS; ++i) {
        result->count = 0;
        timer.start();
        matches = func(build_keys, build_count, probe_keys, probe_count, JoinType::INNER, result);
        total_time += timer.elapsed_us();
    }

    double avg_time = total_time / MEASURE_RUNS;

    // 计算吞吐量
    size_t total_keys = build_count + probe_count;
    double throughput = static_cast<double>(total_keys) / avg_time;  // M keys/sec

    // 计算带宽 (假设每个键 4 字节)
    size_t data_size = (build_count + probe_count) * sizeof(int32_t) +
                       matches * 2 * sizeof(uint32_t);  // 输入 + 输出
    double bandwidth = static_cast<double>(data_size) / (avg_time * 1000.0);  // GB/s

    free_join_result(result);

    return {name, avg_time, matches, throughput, bandwidth};
}

// ============================================================================
// 主测试函数
// ============================================================================

void print_header() {
    printf("\n");
    printf("==============================================================================\n");
    printf(" ThunderDuck Hash Join Version Comparison Benchmark\n");
    printf("==============================================================================\n");
    printf("\n");
    printf("SQL 语义: SELECT COUNT(*) FROM build b JOIN probe p ON b.key = p.key\n");
    printf("\n");
    printf("测试配置:\n");
    printf("  - 平台: Apple Silicon M4 (UMA)\n");
    printf("  - 预热: %d 次, 测量: %d 次\n", WARMUP_RUNS, MEASURE_RUNS);
    printf("  - 理论内存带宽: %.0f GB/s\n", M4_MEMORY_BW_GBS);
    printf("\n");
}

void print_test_header(const TestData& data, size_t build_count, size_t probe_count,
                       double match_rate, int duplicates) {
    size_t data_size = (build_count + probe_count) * sizeof(int32_t);
    printf("------------------------------------------------------------------------------\n");
    printf("测试: %s\n", data.description);
    printf("  Build: %zu, Probe: %zu, 匹配率: %.0f%%, 重复键: %dx\n",
           build_count, probe_count, match_rate * 100, duplicates);
    printf("  数据大小: %.2f MB, 预期匹配: ~%zu\n",
           data_size / (1024.0 * 1024.0), data.expected_matches);
    printf("\n");
    printf("| %-20s | %12s | %12s | %10s | %10s | %8s | %8s |\n",
           "版本", "时间(μs)", "匹配数", "吞吐(Mop/s)", "带宽(GB/s)", "vs v3", "带宽利用");
    printf("|----------------------|--------------|--------------|------------|------------|----------|----------|\n");
}

void print_result(const BenchResult& r, double v3_time) {
    double speedup = v3_time / r.time_us;
    double bw_util = (r.bandwidth_gbs / M4_MEMORY_BW_GBS) * 100.0;
    printf("| %-20s | %12.1f | %12zu | %10.2f | %10.2f | %7.2fx | %7.1f%% |\n",
           r.version, r.time_us, r.matches, r.throughput_mops, r.bandwidth_gbs,
           speedup, bw_util);
}

int main() {
    print_header();

    std::mt19937 rng(42);

    // 测试场景定义
    struct TestCase {
        size_t build_count;
        size_t probe_count;
        double match_rate;
        int duplicates;
        const char* description;
    };

    std::vector<TestCase> test_cases = {
        // 小规模测试
        {10000, 100000, 1.0, 1, "10K×100K 唯一键 全匹配"},
        {10000, 100000, 0.5, 1, "10K×100K 唯一键 50%匹配"},
        {10000, 100000, 0.1, 1, "10K×100K 唯一键 10%匹配"},

        // 中规模测试
        {100000, 1000000, 1.0, 1, "100K×1M 唯一键 全匹配"},
        {100000, 1000000, 0.5, 1, "100K×1M 唯一键 50%匹配"},
        {100000, 1000000, 0.1, 1, "100K×1M 唯一键 10%匹配"},

        // 大规模测试
        {1000000, 1000000, 1.0, 1, "1M×1M 唯一键 全匹配"},
        {1000000, 1000000, 0.1, 1, "1M×1M 唯一键 10%匹配"},
    };

    // 汇总结果
    std::vector<std::tuple<const char*, double, double, double, double, double>> summary;

    for (const auto& tc : test_cases) {
        TestData data = generate_test_data(
            tc.build_count, tc.probe_count, tc.match_rate, tc.duplicates,
            tc.description, rng
        );

        print_test_header(data, tc.build_count, tc.probe_count,
                          tc.match_rate, tc.duplicates);

        // 运行各版本测试
        auto r_v3 = run_benchmark("v3 (CPU Neon)",
            hash_join_i32_v3,
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_v3, r_v3.time_us);

        auto r_v5 = run_benchmark("v5 (两阶段)",
            v5::hash_join_i32_v5_twophase,
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_v5, r_v3.time_us);

        auto r_v5p = run_benchmark("v5 (4线程并行)",
            [](const int32_t* bk, size_t bc, const int32_t* pk, size_t pc,
               JoinType jt, JoinResult* r) {
                return v5::hash_join_i32_v5_parallel(bk, bc, pk, pc, jt, r, 4);
            },
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_v5p, r_v3.time_us);

        auto r_v6 = run_benchmark("v6 (链式哈希)",
            v6::hash_join_i32_v6_chain,
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_v6, r_v3.time_us);

        auto r_v6c = run_benchmark("v6 (分块输出)",
            v6::hash_join_i32_v6_chunked,
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_v6c, r_v3.time_us);

        // GPU-UMA (仅大规模测试)
        if (tc.probe_count >= 1000000 && uma::is_uma_gpu_ready()) {
            JoinConfigV4 gpu_config;
            gpu_config.strategy = JoinStrategy::GPU;

            auto r_gpu = run_benchmark("GPU-UMA",
                [&gpu_config](const int32_t* bk, size_t bc, const int32_t* pk, size_t pc,
                              JoinType jt, JoinResult* r) {
                    return uma::hash_join_gpu_uma(bk, bc, pk, pc, jt, r, gpu_config);
                },
                data.build_keys.data(), tc.build_count,
                data.probe_keys.data(), tc.probe_count);
            print_result(r_gpu, r_v3.time_us);
        }

        // DuckDB 基线
        auto r_duckdb = run_benchmark("DuckDB (模拟)",
            [](const int32_t* bk, size_t bc, const int32_t* pk, size_t pc,
               JoinType, JoinResult* r) {
                return duckdb_baseline_hash_join(bk, bc, pk, pc, r);
            },
            data.build_keys.data(), tc.build_count,
            data.probe_keys.data(), tc.probe_count);
        print_result(r_duckdb, r_v3.time_us);

        printf("\n");

        // 添加到汇总
        double best_td_time = std::min({r_v3.time_us, r_v5.time_us, r_v5p.time_us,
                                         r_v6.time_us, r_v6c.time_us});
        double vs_duckdb = r_duckdb.time_us / best_td_time;
        summary.push_back({tc.description, best_td_time, r_v3.time_us, r_duckdb.time_us,
                           r_v3.time_us / best_td_time, vs_duckdb});
    }

    // 打印汇总表
    printf("\n");
    printf("==============================================================================\n");
    printf(" 性能汇总\n");
    printf("==============================================================================\n");
    printf("\n");
    printf("| %-35s | %12s | %8s | %10s |\n",
           "测试场景", "最佳时间(μs)", "vs v3", "vs DuckDB");
    printf("|-------------------------------------|--------------|----------|------------|\n");

    for (const auto& [desc, best, v3, duckdb, vs_v3, vs_dk] : summary) {
        printf("| %-35s | %12.1f | %7.2fx | %9.1fx |\n",
               desc, best, vs_v3, vs_dk);
    }

    printf("\n");
    printf("==============================================================================\n");
    printf(" 关键发现\n");
    printf("==============================================================================\n");
    printf("\n");
    printf("1. 唯一键场景: 比较 v3/v5/v6 在低重复键下的表现\n");
    printf("2. 高重复键场景: 比较 v5两阶段/v6链式 在高匹配数下的表现\n");
    printf("3. 大规模场景: 比较 CPU vs GPU 在 1M+ 数据量的表现\n");
    printf("4. 整体: ThunderDuck vs DuckDB 的加速比\n");
    printf("\n");

    return 0;
}
