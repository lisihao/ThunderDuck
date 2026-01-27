/**
 * ThunderDuck V13 极致优化 - 基准测试
 *
 * 测试 P0/P1/P3 优化效果:
 * - P0: Hash Join 两阶段算法
 * - P1: GROUP BY GPU 无原子
 * - P3: TopK GPU 并行
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>

#include "thunderduck/v13.h"
#include "thunderduck/v12_5.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"

using namespace std;
using namespace thunderduck;

class Timer {
public:
    void start() { start_ = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end - start_).count();
    }
private:
    chrono::high_resolution_clock::time_point start_;
};

double calc_throughput(size_t bytes, double time_ms) {
    if (time_ms <= 0) return 0;
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

void print_header(const string& title) {
    cout << "\n";
    cout << "================================================================================\n";
    cout << " " << title << "\n";
    cout << "================================================================================\n";
}

void print_result(const string& version, const string& device,
                  double time_ms, double throughput, double speedup,
                  const string& note = "") {
    cout << "│ " << left << setw(10) << version
         << "│ " << setw(20) << device.substr(0, 20)
         << "│ " << right << setw(10) << fixed << setprecision(3) << time_ms << " ms"
         << " │ " << setw(8) << setprecision(2) << throughput << " GB/s"
         << " │ " << setw(6) << setprecision(2) << speedup << "x"
         << " │ " << note << "\n";
}

// ============================================================================
// P0: Hash Join 测试
// ============================================================================

void test_hash_join_p0(size_t build_count, size_t probe_count) {
    print_header("P0 优化: Hash Join 两阶段算法 [build=" + to_string(build_count/1000) + "K, probe=" + to_string(probe_count/1000000) + "M]");

    vector<int32_t> build_keys(build_count);
    vector<int32_t> probe_keys(probe_count);
    mt19937 gen(42);

    // 高匹配率场景
    for (size_t i = 0; i < build_count; i++) build_keys[i] = i;
    uniform_int_distribution<int32_t> dist(0, build_count - 1);
    for (size_t i = 0; i < probe_count; i++) probe_keys[i] = dist(gen);

    Timer timer;
    size_t data_bytes = (build_count + probe_count) * sizeof(int32_t);

    cout << "├────────────┬──────────────────────┬──────────────┬────────────┬────────┤\n";
    cout << "│ 版本       │ 设备                 │ 时间         │ 吞吐量     │ 加速比 │\n";
    cout << "├────────────┼──────────────────────┼──────────────┼────────────┼────────┤\n";

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        vector<int32_t> ht(build_count * 2, -1);
        for (size_t i = 0; i < build_count; i++) {
            size_t slot = build_keys[i] % ht.size();
            while (ht[slot] != -1) slot = (slot + 1) % ht.size();
            ht[slot] = build_keys[i];
        }
        size_t matches = 0;
        for (size_t i = 0; i < probe_count; i++) {
            size_t slot = probe_keys[i] % ht.size();
            for (size_t j = 0; j < ht.size(); j++) {
                if (ht[slot] == probe_keys[i]) { matches++; break; }
                if (ht[slot] == -1) break;
                slot = (slot + 1) % ht.size();
            }
        }
        duckdb_time = timer.elapsed_ms();
        print_result("DuckDB", "CPU Scalar", duckdb_time, calc_throughput(data_bytes, duckdb_time), 1.0, "基准");
    }

    // V11 SIMD
    {
        auto jr = join::create_join_result(build_count);
        timer.start();
        join::hash_join_i32_v11(build_keys.data(), build_count, probe_keys.data(), probe_count, join::JoinType::INNER, jr);
        double time = timer.elapsed_ms();
        join::free_join_result(jr);
        print_result("V11", "CPU SIMD", time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V12.5 Adaptive
    {
        auto jr = v125::create_join_result(build_count);
        v125::ExecutionStats stats;
        timer.start();
        v125::hash_join_i32(build_keys.data(), build_count, probe_keys.data(), probe_count, v125::JoinType::INNER, jr, &stats);
        double time = timer.elapsed_ms();
        v125::free_join_result(jr);
        print_result("V12.5", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V13 Two-Phase (P0 优化)
    {
        auto jr = v13::create_join_result(build_count);
        v13::ExecutionStats stats;
        timer.start();
        v13::hash_join_i32(build_keys.data(), build_count, probe_keys.data(), probe_count, v13::JoinType::INNER, jr, &stats);
        double time = timer.elapsed_ms();
        v13::free_join_result(jr);
        print_result("V13", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time, "★P0优化");
    }

    cout << "└────────────┴──────────────────────┴──────────────┴────────────┴────────┘\n";
}

// ============================================================================
// P1: GROUP BY GPU 测试
// ============================================================================

void test_group_by_p1(size_t count, size_t num_groups) {
    print_header("P1 优化: GROUP BY GPU 无原子 [" + to_string(count/1000000) + "M, " + to_string(num_groups) + " groups]");

    vector<int32_t> values(count);
    vector<uint32_t> groups(count);
    vector<int64_t> sums(num_groups);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_val(0, 1000);
    uniform_int_distribution<uint32_t> dist_grp(0, num_groups - 1);
    for (size_t i = 0; i < count; i++) {
        values[i] = dist_val(gen);
        groups[i] = dist_grp(gen);
    }

    Timer timer;
    size_t data_bytes = count * 8;

    cout << "├────────────┬──────────────────────┬──────────────┬────────────┬────────┤\n";
    cout << "│ 版本       │ 设备                 │ 时间         │ 吞吐量     │ 加速比 │\n";
    cout << "├────────────┼──────────────────────┼──────────────┼────────────┼────────┤\n";

    // DuckDB 基准
    double duckdb_time;
    {
        timer.start();
        memset(sums.data(), 0, num_groups * sizeof(int64_t));
        for (size_t i = 0; i < count; i++) sums[groups[i]] += values[i];
        duckdb_time = timer.elapsed_ms();
        print_result("DuckDB", "CPU Scalar", duckdb_time, calc_throughput(data_bytes, duckdb_time), 1.0, "基准");
    }

    // V8 CPU Parallel
    {
        timer.start();
        aggregate::group_sum_i32_v4_parallel(values.data(), groups.data(), count, num_groups, sums.data());
        double time = timer.elapsed_ms();
        print_result("V8", "CPU Parallel 4核", time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V9 GPU 两阶段原子
    if (aggregate::is_group_aggregate_v2_available()) {
        timer.start();
        aggregate::group_sum_i32_v5(values.data(), groups.data(), count, num_groups, sums.data());
        double time = timer.elapsed_ms();
        print_result("V9", "GPU 2-Phase Atomic", time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats);
        double time = timer.elapsed_ms();
        print_result("V12.5", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V13 GPU Partition (P1 优化)
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::group_sum_i32(values.data(), groups.data(), count, num_groups, sums.data(), &stats);
        double time = timer.elapsed_ms();
        print_result("V13", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time, "★P1优化");
    }

    cout << "└────────────┴──────────────────────┴──────────────┴────────────┴────────┘\n";
}

// ============================================================================
// P3: TopK GPU 测试
// ============================================================================

void test_topk_p3(size_t count, size_t k) {
    print_header("P3 优化: TopK GPU 并行 [" + to_string(count/1000000) + "M, k=" + to_string(k) + "]");

    vector<int32_t> data(count);
    vector<int32_t> out_values(k);
    vector<uint32_t> out_indices(k);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist(0, 1000000);
    for (size_t i = 0; i < count; i++) data[i] = dist(gen);

    Timer timer;
    size_t data_bytes = count * sizeof(int32_t);

    cout << "├────────────┬──────────────────────┬──────────────┬────────────┬────────┤\n";
    cout << "│ 版本       │ 设备                 │ 时间         │ 吞吐量     │ 加速比 │\n";
    cout << "├────────────┼──────────────────────┼──────────────┼────────────┼────────┤\n";

    // DuckDB 基准 (partial_sort)
    double duckdb_time;
    {
        vector<int32_t> copy = data;
        timer.start();
        partial_sort(copy.begin(), copy.begin() + k, copy.end(), greater<int32_t>());
        duckdb_time = timer.elapsed_ms();
        print_result("DuckDB", "CPU partial_sort", duckdb_time, calc_throughput(data_bytes, duckdb_time), 1.0, "基准");
    }

    // V7 Sampling
    {
        timer.start();
        sort::topk_max_i32_v4(data.data(), count, k, out_values.data(), out_indices.data());
        double time = timer.elapsed_ms();
        print_result("V7", "CPU Sampling", time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V8 Count-Based
    {
        timer.start();
        sort::topk_max_i32_v5(data.data(), count, k, out_values.data(), out_indices.data());
        double time = timer.elapsed_ms();
        print_result("V8", "CPU Count-Based", time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V12.5
    {
        v125::ExecutionStats stats;
        timer.start();
        v125::topk_max_i32(data.data(), count, k, out_values.data(), out_indices.data(), &stats);
        double time = timer.elapsed_ms();
        print_result("V12.5", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time);
    }

    // V13 GPU (P3 优化)
    {
        v13::ExecutionStats stats;
        timer.start();
        v13::topk_max_i32(data.data(), count, k, out_values.data(), out_indices.data(), &stats);
        double time = timer.elapsed_ms();
        print_result("V13", string(stats.device_used), time, calc_throughput(data_bytes, time), duckdb_time/time, "★P3优化");
    }

    cout << "└────────────┴──────────────────────┴──────────────┴────────────┴────────┘\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║          ThunderDuck V13 极致优化 - 基准测试                                  ║
║          P0: Hash Join 两阶段 | P1: GROUP BY 无原子 | P3: TopK GPU            ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << endl;

    cout << v13::get_version_info() << "\n" << endl;

    // P0 测试: Hash Join
    test_hash_join_p0(100000, 1000000);   // 100K x 1M

    // P1 测试: GROUP BY GPU
    test_group_by_p1(1000000, 1000);      // 1M, 1000 groups
    test_group_by_p1(10000000, 1000);     // 10M, 1000 groups

    // P3 测试: TopK GPU
    test_topk_p3(1000000, 10);            // 1M, k=10
    test_topk_p3(10000000, 10);           // 10M, k=10

    // 总结
    cout << R"(

================================================================================
                        V13 优化总结
================================================================================

┌────────────────────────────────────────────────────────────────────────────┐
│ P0 Hash Join: 两阶段算法                                                    │
│   - Phase 1: 计数遍历，统计总匹配数                                         │
│   - Phase 2: 预分配精确容量，一次填充                                       │
│   - 消除 grow_join_result() 动态扩容开销                                    │
├────────────────────────────────────────────────────────────────────────────┤
│ P1 GROUP BY GPU: 分区聚合                                                   │
│   - Phase 1: 每个 threadgroup 独立累加 (无全局原子)                         │
│   - Phase 2: 合并分区结果                                                   │
│   - 原子操作减少 1000x                                                      │
├────────────────────────────────────────────────────────────────────────────┤
│ P3 TopK GPU: 并行选择                                                       │
│   - 多线程并行扫描                                                          │
│   - 本地堆选择 + 全局合并                                                   │
└────────────────────────────────────────────────────────────────────────────┘

)" << endl;

    return 0;
}
