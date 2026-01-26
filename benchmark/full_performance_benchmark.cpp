/**
 * ThunderDuck V6 全面性能基准测试
 */

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"
#include "thunderduck/uma_memory.h"
#include "duckdb.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

using namespace std;
using namespace thunderduck;

constexpr int WARMUP = 3;
constexpr int MEASURE = 5;

template<typename Func>
double measure_us(Func&& func) {
    for (int i = 0; i < WARMUP; ++i) func();
    double total = 0;
    for (int i = 0; i < MEASURE; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        total += chrono::duration<double, micro>(end - start).count();
    }
    return total / MEASURE;
}

double bw(size_t bytes, double us) { return (bytes / 1e9) / (us / 1e6); }
double tp(size_t rows, double us) { return (rows / 1e6) / (us / 1e6); }

class DuckDBBench {
public:
    duckdb::DuckDB db;
    duckdb::Connection conn;
    DuckDBBench() : db(nullptr), conn(db) {}

    double filter_bench(const vector<int32_t>& data, int32_t threshold) {
        conn.Query("DROP TABLE IF EXISTS t");
        conn.Query("CREATE TABLE t (val INTEGER)");
        {
            duckdb::Appender app(conn, "t");
            for (int32_t v : data) app.AppendRow(v);
            app.Close();
        }
        string sql = "SELECT COUNT(*) FROM t WHERE val > " + to_string(threshold);
        return measure_us([&]() { conn.Query(sql); });
    }

    double sum_bench(const vector<int32_t>& data) {
        conn.Query("DROP TABLE IF EXISTS t");
        conn.Query("CREATE TABLE t (val INTEGER)");
        {
            duckdb::Appender app(conn, "t");
            for (int32_t v : data) app.AppendRow(v);
            app.Close();
        }
        return measure_us([&]() { conn.Query("SELECT SUM(val) FROM t"); });
    }

    double minmax_bench(const vector<int32_t>& data) {
        conn.Query("DROP TABLE IF EXISTS t");
        conn.Query("CREATE TABLE t (val INTEGER)");
        {
            duckdb::Appender app(conn, "t");
            for (int32_t v : data) app.AppendRow(v);
            app.Close();
        }
        return measure_us([&]() { conn.Query("SELECT MIN(val), MAX(val) FROM t"); });
    }

    double topk_bench(const vector<int32_t>& data, size_t k) {
        conn.Query("DROP TABLE IF EXISTS t");
        conn.Query("CREATE TABLE t (val INTEGER)");
        {
            duckdb::Appender app(conn, "t");
            for (int32_t v : data) app.AppendRow(v);
            app.Close();
        }
        string sql = "SELECT val FROM t ORDER BY val DESC LIMIT " + to_string(k);
        return measure_us([&]() { conn.Query(sql); });
    }

    double join_bench(const vector<int32_t>& build, const vector<int32_t>& probe) {
        conn.Query("DROP TABLE IF EXISTS b; DROP TABLE IF EXISTS p");
        conn.Query("CREATE TABLE b (key INTEGER)");
        conn.Query("CREATE TABLE p (key INTEGER)");
        {
            duckdb::Appender a1(conn, "b");
            for (int32_t k : build) a1.AppendRow(k);
            a1.Close();
        }
        {
            duckdb::Appender a2(conn, "p");
            for (int32_t k : probe) a2.AppendRow(k);
            a2.Close();
        }
        return measure_us([&]() { conn.Query("SELECT COUNT(*) FROM b JOIN p ON b.key=p.key"); });
    }
};

ofstream g_report;

void test_filter() {
    cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║ Filter 算子: SELECT * FROM t WHERE val > threshold                                        ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";

    g_report << "\n## Filter 算子\n\n";
    g_report << "**SQL**: `SELECT * FROM t WHERE val > threshold`\n\n";
    g_report << "| 数据量 | 选择率 | 版本 | 硬件 | 时间(μs) | 带宽(GB/s) | 吞吐(M/s) | vs v3 | vs DuckDB |\n";
    g_report << "|--------|--------|------|------|----------|------------|-----------|-------|----------|\n";

    struct Test { size_t n; float sel; };
    vector<Test> tests = {{100000,0.5},{1000000,0.5},{1000000,0.1},{1000000,0.9},{10000000,0.5},{10000000,0.1},{50000000,0.5}};

    DuckDBBench duck;

    for (auto& t : tests) {
        cout << "\n【" << t.n/1000000 << "M rows, " << (int)(t.sel*100) << "% selectivity】\n";
        printf("┌──────────────┬──────────┬────────────┬───────────┬───────────┬─────────┬──────────┐\n");
        printf("│ 版本         │ 硬件     │ 时间(μs)   │ 带宽(GB/s)│ 吞吐(M/s) │ vs v3   │ vs DuckDB│\n");
        printf("├──────────────┼──────────┼────────────┼───────────┼───────────┼─────────┼──────────┤\n");

        vector<int32_t> data(t.n);
        mt19937 gen(42);
        uniform_int_distribution<int32_t> dist(0, 1000000000);
        for (size_t i = 0; i < t.n; ++i) data[i] = dist(gen);

        int32_t threshold = (int32_t)((1.0f - t.sel) * 1000000000);
        vector<uint32_t> idx(t.n);
        size_t cnt = 0, bytes = t.n * 4;

        double duck_t = duck.filter_bench(data, threshold);
        double v2_t = measure_us([&](){ cnt = filter::filter_i32_v2(data.data(), t.n, filter::CompareOp::GT, threshold, idx.data()); });
        double v3_t = measure_us([&](){ cnt = filter::filter_i32_v3(data.data(), t.n, filter::CompareOp::GT, threshold, idx.data()); });
        double v4_t = measure_us([&](){ cnt = filter::filter_i32_v4(data.data(), t.n, filter::CompareOp::GT, threshold, idx.data()); });

        auto pr = [&](const char* ver, const char* hw, double tm, double vs3, double vsd) {
            printf("│ %-12s │ %-8s │ %10.1f │ %9.2f │ %9.2f │ %7.2fx│ %8.2fx│\n", ver, hw, tm, bw(bytes,tm), tp(t.n,tm), vs3, vsd);
            g_report << "| " << t.n/1000000 << "M | " << (int)(t.sel*100) << "% | " << ver << " | " << hw
                     << " | " << fixed << setprecision(1) << tm << " | " << setprecision(2) << bw(bytes,tm)
                     << " | " << tp(t.n,tm) << " | " << vs3 << "x | " << vsd << "x |\n";
        };

        pr("v2", "CPU SIMD", v2_t, v3_t/v2_t, duck_t/v2_t);
        pr("v3", "CPU SIMD", v3_t, 1.0, duck_t/v3_t);
        pr("v4 (自适应)", "AUTO", v4_t, v3_t/v4_t, duck_t/v4_t);
        pr("DuckDB", "CPU", duck_t, v3_t/duck_t, 1.0);

        printf("├──────────────┴──────────┴────────────┴───────────┴───────────┴─────────┴──────────┤\n");
        printf("│ 结果行数: %zu (%.1f%%)                                                             │\n", cnt, cnt*100.0/t.n);
        printf("└────────────────────────────────────────────────────────────────────────────────────┘\n");
    }
}

void test_aggregate() {
    cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║ Aggregate 算子: SELECT SUM/MIN/MAX(val) FROM t                                            ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";

    g_report << "\n## Aggregate 算子\n\n";
    g_report << "**SQL**: `SELECT SUM(val), MIN(val), MAX(val) FROM t`\n\n";
    g_report << "| 数据量 | 操作 | 版本 | 硬件 | 时间(μs) | 带宽(GB/s) | vs v3 | vs DuckDB |\n";
    g_report << "|--------|------|------|------|----------|------------|-------|----------|\n";

    vector<size_t> sizes = {100000, 1000000, 10000000, 50000000};
    DuckDBBench duck;

    for (size_t n : sizes) {
        cout << "\n【" << n/1000000 << "M rows】\n";
        printf("┌──────────────┬──────────┬────────────┬───────────┬─────────┬──────────┐\n");
        printf("│ 操作/版本    │ 硬件     │ 时间(μs)   │ 带宽(GB/s)│ vs v3   │ vs DuckDB│\n");
        printf("├──────────────┼──────────┼────────────┼───────────┼─────────┼──────────┤\n");

        vector<int32_t> data(n);
        mt19937 gen(42);
        uniform_int_distribution<int32_t> dist(-1000000, 1000000);
        for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
        size_t bytes = n * 4;

        int64_t sum; int32_t mn, mx;

        double duck_sum = duck.sum_bench(data);
        double v2_sum = measure_us([&](){ sum = aggregate::sum_i32_v2(data.data(), n); });
        double v3_sum = measure_us([&](){ sum = aggregate::sum_i32_v3(data.data(), n); });

        double duck_mm = duck.minmax_bench(data);
        double v2_mm = measure_us([&](){ aggregate::minmax_i32(data.data(), n, &mn, &mx); });
        double v3_mm = measure_us([&](){ aggregate::minmax_i32_v3(data.data(), n, &mn, &mx); });

        auto pr = [&](const char* op, const char* ver, const char* hw, double tm, double vs3, double vsd) {
            printf("│ %-12s │ %-8s │ %10.1f │ %9.2f │ %7.2fx│ %8.2fx│\n", (string(op)+"/"+ver).c_str(), hw, tm, bw(bytes,tm), vs3, vsd);
            g_report << "| " << n/1000000 << "M | " << op << " | " << ver << " | " << hw
                     << " | " << fixed << setprecision(1) << tm << " | " << setprecision(2) << bw(bytes,tm)
                     << " | " << vs3 << "x | " << vsd << "x |\n";
        };

        pr("SUM", "v2", "CPU SIMD", v2_sum, v3_sum/v2_sum, duck_sum/v2_sum);
        pr("SUM", "v3", "AUTO", v3_sum, 1.0, duck_sum/v3_sum);
        pr("SUM", "DuckDB", "CPU", duck_sum, v3_sum/duck_sum, 1.0);
        pr("MINMAX", "v2", "CPU SIMD", v2_mm, v3_mm/v2_mm, duck_mm/v2_mm);
        pr("MINMAX", "v3", "AUTO", v3_mm, 1.0, duck_mm/v3_mm);
        pr("MINMAX", "DuckDB", "CPU", duck_mm, v3_mm/duck_mm, 1.0);

        printf("└──────────────┴──────────┴────────────┴───────────┴─────────┴──────────┘\n");
    }
}

void test_topk() {
    cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║ TopK 算子: SELECT val FROM t ORDER BY val DESC LIMIT K                                    ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";

    g_report << "\n## TopK 算子\n\n";
    g_report << "**SQL**: `SELECT val FROM t ORDER BY val DESC LIMIT K`\n\n";
    g_report << "| 数据量 | K | 版本 | 时间(μs) | 吞吐(M/s) | vs v3 | vs DuckDB |\n";
    g_report << "|--------|---|------|----------|-----------|-------|----------|\n";

    struct Test { size_t n; size_t k; };
    vector<Test> tests = {{100000,10},{100000,100},{1000000,10},{1000000,100},{1000000,1000},{10000000,10},{10000000,100},{10000000,1000}};
    DuckDBBench duck;

    for (auto& t : tests) {
        cout << "\n【" << t.n/1000000 << "M rows, K=" << t.k << "】\n";
        printf("┌──────────────┬────────────┬───────────┬─────────┬──────────┐\n");
        printf("│ 版本         │ 时间(μs)   │ 吞吐(M/s) │ vs v3   │ vs DuckDB│\n");
        printf("├──────────────┼────────────┼───────────┼─────────┼──────────┤\n");

        vector<int32_t> data(t.n);
        mt19937 gen(42);
        uniform_int_distribution<int32_t> dist(-1000000000, 1000000000);
        for (size_t i = 0; i < t.n; ++i) data[i] = dist(gen);

        vector<int32_t> vals(t.k);
        vector<uint32_t> idx(t.k);

        double duck_t = duck.topk_bench(data, t.k);
        double v3_t = measure_us([&](){ sort::topk_max_i32_v3(data.data(), t.n, t.k, vals.data(), idx.data()); });
        double v4_t = measure_us([&](){ sort::topk_max_i32_v4(data.data(), t.n, t.k, vals.data(), idx.data()); });
        double v5_t = measure_us([&](){ sort::topk_max_i32_v5(data.data(), t.n, t.k, vals.data(), idx.data()); });

        auto pr = [&](const char* ver, double tm, double vs3, double vsd) {
            printf("│ %-12s │ %10.1f │ %9.2f │ %7.2fx│ %8.2fx│\n", ver, tm, tp(t.n,tm), vs3, vsd);
            g_report << "| " << t.n/1000000 << "M | " << t.k << " | " << ver
                     << " | " << fixed << setprecision(1) << tm << " | " << setprecision(2) << tp(t.n,tm)
                     << " | " << vs3 << "x | " << vsd << "x |\n";
        };

        pr("v3 (堆)", v3_t, 1.0, duck_t/v3_t);
        pr("v4 (采样)", v4_t, v3_t/v4_t, duck_t/v4_t);
        pr("v5 (自适应)", v5_t, v3_t/v5_t, duck_t/v5_t);
        pr("DuckDB", duck_t, v3_t/duck_t, 1.0);

        printf("└──────────────┴────────────┴───────────┴─────────┴──────────┘\n");
    }
}

void test_join() {
    cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║ Hash Join 算子: SELECT * FROM b JOIN p ON b.key = p.key                                   ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";

    g_report << "\n## Hash Join 算子\n\n";
    g_report << "**SQL**: `SELECT * FROM b JOIN p ON b.key = p.key`\n\n";
    g_report << "| 规模 | 匹配率 | 版本 | 硬件 | 时间(μs) | vs v3 | vs DuckDB |\n";
    g_report << "|------|--------|------|------|----------|-------|----------|\n";

    struct Test { size_t build; size_t probe; float match; };
    vector<Test> tests = {{10000,100000,1.0},{10000,100000,0.5},{10000,100000,0.1},{100000,1000000,1.0},{100000,1000000,0.5},{1000000,1000000,1.0},{1000000,1000000,0.5},{1000000,10000000,0.5}};
    DuckDBBench duck;

    for (auto& t : tests) {
        cout << "\n【" << t.build/1000 << "K×" << t.probe/1000000 << "M, " << (int)(t.match*100) << "% match】\n";
        printf("┌──────────────┬──────────┬────────────┬─────────┬──────────┐\n");
        printf("│ 版本         │ 硬件     │ 时间(μs)   │ vs v3   │ vs DuckDB│\n");
        printf("├──────────────┼──────────┼────────────┼─────────┼──────────┤\n");

        vector<int32_t> build(t.build), probe(t.probe);
        for (size_t i = 0; i < t.build; ++i) build[i] = (int32_t)i;
        mt19937 gen(42);
        size_t range = (size_t)(t.build / t.match);
        uniform_int_distribution<int32_t> dist(0, range-1);
        for (size_t i = 0; i < t.probe; ++i) probe[i] = dist(gen);

        join::JoinResult res;
        size_t cnt = 0;

        double duck_t = duck.join_bench(build, probe);
        double v2_t = measure_us([&](){ cnt = join::hash_join_i32_v2(build.data(), t.build, probe.data(), t.probe, join::JoinType::INNER, &res); });
        double v3_t = measure_us([&](){ cnt = join::hash_join_i32_v3(build.data(), t.build, probe.data(), t.probe, join::JoinType::INNER, &res); });
        double v4_t = measure_us([&](){ cnt = join::hash_join_i32_v4(build.data(), t.build, probe.data(), t.probe, join::JoinType::INNER, &res); });

        auto pr = [&](const char* ver, const char* hw, double tm, double vs3, double vsd) {
            printf("│ %-12s │ %-8s │ %10.1f │ %7.2fx│ %8.2fx│\n", ver, hw, tm, vs3, vsd);
            g_report << "| " << t.build/1000 << "K×" << t.probe/1000000 << "M | " << (int)(t.match*100) << "% | "
                     << ver << " | " << hw << " | " << fixed << setprecision(1) << tm
                     << " | " << setprecision(2) << vs3 << "x | " << vsd << "x |\n";
        };

        pr("v2 (Robin)", "CPU", v2_t, v3_t/v2_t, duck_t/v2_t);
        pr("v3 (SOA)", "CPU SIMD", v3_t, 1.0, duck_t/v3_t);
        pr("v4 (多策略)", "CPU Opt", v4_t, v3_t/v4_t, duck_t/v4_t);
        pr("DuckDB", "CPU", duck_t, v3_t/duck_t, 1.0);

        printf("├──────────────┴──────────┴────────────┴─────────┴──────────┤\n");
        printf("│ 匹配: %zu (%.1f%%)                                         │\n", cnt, cnt*100.0/t.probe);
        printf("└────────────────────────────────────────────────────────────┘\n");
    }
}

int main() {
    cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                    ThunderDuck V6 全面性能基准测试                                        ║\n";
    cout << "║                    Apple M4 - 统一内存架构优化                                            ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";

    cout << "\n=== 系统状态 ===\n";
    cout << "UMA: " << (uma::UMAMemoryManager::instance().is_available() ? "✓" : "✗") << "\n";
    cout << "Filter GPU: " << (filter::is_filter_gpu_available() ? "✓" : "✗") << "\n";
    cout << "Aggregate GPU: " << (aggregate::is_aggregate_gpu_available() ? "✓" : "✗") << "\n";

    g_report.open("docs/FULL_BENCHMARK_REPORT_V6.md");
    g_report << "# ThunderDuck V6 全面性能基准测试报告\n\n";
    g_report << "> **版本**: V6 | **日期**: 2026-01-26 | **平台**: Apple Silicon M4 (UMA)\n\n";
    g_report << "## 测试环境\n\n";
    g_report << "| 配置项 | 详情 |\n|-------|------|\n";
    g_report << "| 平台 | Apple Silicon M4 |\n| 理论带宽 | 400 GB/s |\n| CPU | ARM Neon SIMD |\n";
    g_report << "| 预热 | 3 次 |\n| 测量 | 5 次 |\n\n---\n";

    test_filter();
    test_aggregate();
    test_topk();
    test_join();

    g_report << "\n---\n*报告生成: ThunderDuck Benchmark Suite - 2026-01-26*\n";
    g_report.close();

    cout << "\n✓ 报告已保存到 docs/FULL_BENCHMARK_REPORT_V6.md\n";
    return 0;
}
