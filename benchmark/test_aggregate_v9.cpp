/**
 * V9 Aggregate 性能对比测试
 *
 * 对比 V7, V8, V9, DuckDB
 * 使用中位数 + 标准差 + IQR剔除异常值
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "thunderduck/aggregate.h"
#include "duckdb.hpp"

using namespace std;
using namespace thunderduck::aggregate;

// ============================================================================
// 符合 CLAUDE.md 基准测试规则的测量函数
// ============================================================================

struct BenchResult {
    double median;
    double stddev;
    int removed_outliers;
};

template<typename F>
BenchResult measure_with_stats(F&& func, int iterations = 30) {
    // Warmup
    func();

    vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
    }

    // 排序用于计算四分位数
    sort(times.begin(), times.end());

    // 计算 Q1, Q3, IQR
    double q1 = times[times.size() / 4];
    double q3 = times[times.size() * 3 / 4];
    double iqr = q3 - q1;

    // 剔除异常值 (IQR 方法)
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    vector<double> filtered;
    for (double t : times) {
        if (t >= lower && t <= upper) {
            filtered.push_back(t);
        }
    }

    int removed = static_cast<int>(times.size() - filtered.size());

    // 使用过滤后的数据
    if (filtered.empty()) filtered = times;  // 保底

    // 计算中位数
    sort(filtered.begin(), filtered.end());
    double median = filtered[filtered.size() / 2];

    // 计算标准差
    double mean = accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    double sq_sum = 0;
    for (double t : filtered) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / filtered.size());

    return {median, stddev, removed};
}

// ============================================================================
// 测试函数
// ============================================================================

void print_result(const string& name, const BenchResult& r, double baseline = 0) {
    cout << "  " << left << setw(25) << name << ": "
         << fixed << setprecision(3) << setw(8) << r.median << " ms"
         << " (σ=" << setprecision(3) << setw(5) << r.stddev << ")";

    if (baseline > 0) {
        double speedup = baseline / r.median;
        cout << "  [" << setprecision(2) << speedup << "x]";
    }

    if (r.removed_outliers > 0) {
        cout << " *" << r.removed_outliers << " outliers removed";
    }

    cout << endl;
}

int main() {
    cout << "\n╔══════════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║           V9 Aggregate 性能对比测试 (中位数+IQR)                     ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════════╝" << endl;

    // ========================================================================
    // 数据准备
    // ========================================================================

    const size_t N = 4000000;  // 4M 元素

    vector<int32_t> data_i32(N);
    mt19937 gen(42);
    uniform_int_distribution<int32_t> dist_i32(1, 50);

    for (size_t i = 0; i < N; ++i) {
        data_i32[i] = dist_i32(gen);
    }

    cout << "\n数据规模: " << N << " 元素 (4M)" << endl;
    cout << "迭代次数: 30 次 (剔除异常值)\n" << endl;

    // ========================================================================
    // 测试 1: 简单聚合 SUM
    // ========================================================================

    cout << "=== SUM i32 对比 ===" << endl;

    // DuckDB baseline
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    con.Query("CREATE TABLE test (quantity INTEGER)");

    con.Query("BEGIN TRANSACTION");
    for (int batch = 0; batch < 40; ++batch) {
        string values;
        for (int i = 0; i < 100000; ++i) {
            if (i > 0) values += ",";
            values += "(" + to_string(data_i32[batch * 100000 + i]) + ")";
        }
        con.Query("INSERT INTO test VALUES " + values);
    }
    con.Query("COMMIT");

    auto duckdb_sum = measure_with_stats([&]() {
        con.Query("SELECT SUM(quantity) FROM test");
    }, 30);

    // ThunderDuck v1 (基础版)
    auto td_v1 = measure_with_stats([&]() {
        sum_i32(data_i32.data(), N);
    }, 30);

    // ThunderDuck v2 (预取优化)
    auto td_v2 = measure_with_stats([&]() {
        sum_i32_v2(data_i32.data(), N);
    }, 30);

    // ThunderDuck v4 (V9: 更激进预取)
    auto td_v4 = measure_with_stats([&]() {
        sum_i32_v4(data_i32.data(), N);
    }, 30);

    // ThunderDuck v4 blocked
    auto td_v4_blk = measure_with_stats([&]() {
        sum_i32_v4_blocked(data_i32.data(), N);
    }, 30);

    print_result("DuckDB", duckdb_sum);
    print_result("TD v1 (SIMD)", td_v1, duckdb_sum.median);
    print_result("TD v2 (prefetch 64B)", td_v2, duckdb_sum.median);
    print_result("TD v4 (prefetch 256B)", td_v4, duckdb_sum.median);
    print_result("TD v4 blocked (3MB)", td_v4_blk, duckdb_sum.median);

    // ========================================================================
    // 测试 2: MIN/MAX
    // ========================================================================

    cout << "\n=== MIN/MAX i32 对比 ===" << endl;

    auto duckdb_minmax = measure_with_stats([&]() {
        con.Query("SELECT MIN(quantity), MAX(quantity) FROM test");
    }, 30);

    auto td_minmax_v2 = measure_with_stats([&]() {
        int32_t min_val, max_val;
        minmax_i32(data_i32.data(), N, &min_val, &max_val);
    }, 30);

    auto td_minmax_v4 = measure_with_stats([&]() {
        int32_t min_val, max_val;
        minmax_i32_v4(data_i32.data(), N, &min_val, &max_val);
    }, 30);

    print_result("DuckDB", duckdb_minmax);
    print_result("TD v2 (16 elem)", td_minmax_v2, duckdb_minmax.median);
    print_result("TD v4 (32 elem)", td_minmax_v4, duckdb_minmax.median);

    // ========================================================================
    // 测试 3: 融合统计量
    // ========================================================================

    cout << "\n=== 融合统计量 (SUM+MIN+MAX) ===" << endl;

    auto duckdb_all = measure_with_stats([&]() {
        con.Query("SELECT SUM(quantity), MIN(quantity), MAX(quantity) FROM test");
    }, 30);

    auto td_all_v2 = measure_with_stats([&]() {
        aggregate_all_i32(data_i32.data(), N);
    }, 30);

    auto td_all_v4 = measure_with_stats([&]() {
        aggregate_all_i32_v4(data_i32.data(), N);
    }, 30);

    print_result("DuckDB (3 queries)", duckdb_all);
    print_result("TD v2 (fused)", td_all_v2, duckdb_all.median);
    print_result("TD v4 (fused+prefetch)", td_all_v4, duckdb_all.median);

    // ========================================================================
    // 测试 4: 分组聚合
    // ========================================================================

    cout << "\n=== 分组聚合 GROUP BY ===" << endl;

    // 准备分组数据
    const size_t NUM_GROUPS = 1000;
    vector<uint32_t> groups(N);
    uniform_int_distribution<uint32_t> group_dist(0, NUM_GROUPS - 1);
    for (size_t i = 0; i < N; ++i) {
        groups[i] = group_dist(gen);
    }

    vector<int64_t> group_sums(NUM_GROUPS);
    vector<size_t> group_counts(NUM_GROUPS);

    // DuckDB group by
    con.Query("DROP TABLE test");
    con.Query("CREATE TABLE test (group_id INTEGER, quantity INTEGER)");

    con.Query("BEGIN TRANSACTION");
    for (int batch = 0; batch < 40; ++batch) {
        string values;
        for (int i = 0; i < 100000; ++i) {
            if (i > 0) values += ",";
            int idx = batch * 100000 + i;
            values += "(" + to_string(groups[idx]) + "," + to_string(data_i32[idx]) + ")";
        }
        con.Query("INSERT INTO test VALUES " + values);
    }
    con.Query("COMMIT");

    auto duckdb_group = measure_with_stats([&]() {
        con.Query("SELECT group_id, SUM(quantity) FROM test GROUP BY group_id");
    }, 30);

    // ThunderDuck 基础分组
    auto td_group_v1 = measure_with_stats([&]() {
        group_sum_i32(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    // ThunderDuck v4 分区优化
    auto td_group_v4 = measure_with_stats([&]() {
        group_sum_i32_v4(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    // ThunderDuck v4 多线程
    auto td_group_v4_mt = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(data_i32.data(), groups.data(), N, NUM_GROUPS, group_sums.data());
    }, 30);

    print_result("DuckDB", duckdb_group);
    print_result("TD v1 (scalar)", td_group_v1, duckdb_group.median);
    print_result("TD v4 (partition)", td_group_v4, duckdb_group.median);
    print_result("TD v4 parallel (4T)", td_group_v4_mt, duckdb_group.median);

    // ========================================================================
    // 测试 5: 大分组数测试 (10K groups)
    // ========================================================================

    cout << "\n=== 大分组数 (10K groups) ===" << endl;

    const size_t LARGE_GROUPS = 10000;
    vector<uint32_t> large_groups(N);
    uniform_int_distribution<uint32_t> large_group_dist(0, LARGE_GROUPS - 1);
    for (size_t i = 0; i < N; ++i) {
        large_groups[i] = large_group_dist(gen);
    }

    vector<int64_t> large_group_sums(LARGE_GROUPS);

    auto td_large_v1 = measure_with_stats([&]() {
        group_sum_i32(data_i32.data(), large_groups.data(), N, LARGE_GROUPS, large_group_sums.data());
    }, 30);

    auto td_large_v4 = measure_with_stats([&]() {
        group_sum_i32_v4(data_i32.data(), large_groups.data(), N, LARGE_GROUPS, large_group_sums.data());
    }, 30);

    auto td_large_v4_mt = measure_with_stats([&]() {
        group_sum_i32_v4_parallel(data_i32.data(), large_groups.data(), N, LARGE_GROUPS, large_group_sums.data());
    }, 30);

    print_result("TD v1 (scalar)", td_large_v1);
    print_result("TD v4 (partition)", td_large_v4, td_large_v1.median);
    print_result("TD v4 parallel (4T)", td_large_v4_mt, td_large_v1.median);

    // ========================================================================
    // 汇总
    // ========================================================================

    cout << "\n" << string(70, '=') << endl;
    cout << "性能汇总 (vs DuckDB)" << endl;
    cout << string(70, '-') << endl;

    cout << "SUM:       v1=" << fixed << setprecision(2) << (duckdb_sum.median / td_v1.median) << "x"
         << "  v2=" << (duckdb_sum.median / td_v2.median) << "x"
         << "  v4=" << (duckdb_sum.median / td_v4.median) << "x" << endl;

    cout << "MIN/MAX:   v2=" << (duckdb_minmax.median / td_minmax_v2.median) << "x"
         << "  v4=" << (duckdb_minmax.median / td_minmax_v4.median) << "x" << endl;

    cout << "Fused:     v2=" << (duckdb_all.median / td_all_v2.median) << "x"
         << "  v4=" << (duckdb_all.median / td_all_v4.median) << "x" << endl;

    cout << "GroupBy:   v1=" << (duckdb_group.median / td_group_v1.median) << "x"
         << "  v4=" << (duckdb_group.median / td_group_v4.median) << "x"
         << "  v4-MT=" << (duckdb_group.median / td_group_v4_mt.median) << "x" << endl;

    cout << string(70, '=') << endl;

    return 0;
}
