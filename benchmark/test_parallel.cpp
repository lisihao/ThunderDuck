/**
 * ThunderDuck - 并行优化测试
 *
 * 测试新的并行 Filter/Aggregate 实现
 */

#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/join.h"

#include <duckdb.hpp>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

using namespace thunderduck;

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

double benchmark_fn(std::function<void()> fn, int warmup, int runs) {
    Timer timer;
    for (int i = 0; i < warmup; i++) fn();

    double total = 0;
    for (int i = 0; i < runs; i++) {
        timer.start();
        fn();
        timer.stop();
        total += timer.elapsed_ms();
    }
    return total / runs;
}

std::vector<int32_t> generate_random_i32(size_t count) {
    std::vector<int32_t> data(count);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000000);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    return data;
}

void test_parallel_filter() {
    std::cout << "\n=== 并行 Filter 测试 (10M 数据) ===\n\n";

    size_t count = 10000000;  // 10M
    auto data = generate_random_i32(count);
    std::vector<uint32_t> indices(count);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (auto v : data) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM t WHERE value > 500000");
    }, 2, 5);

    // V3 (单线程 SIMD)
    double v3_time = benchmark_fn([&]() {
        filter::count_i32_v3(data.data(), count, filter::CompareOp::GT, 500000);
    }, 2, 5);

    // 并行 Filter
    double parallel_time = benchmark_fn([&]() {
        filter::filter_i32_parallel(data.data(), count, filter::CompareOp::GT, 500000, indices.data());
    }, 2, 5);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "DuckDB:           " << duckdb_time << " ms (1.00x)\n";
    std::cout << "V3 (单线程):      " << v3_time << " ms (" << duckdb_time/v3_time << "x)\n";
    std::cout << "V10 (多线程):     " << parallel_time << " ms (" << duckdb_time/parallel_time << "x)\n";
}

void test_parallel_aggregate() {
    std::cout << "\n=== 并行 Aggregate 测试 (10M 数据) ===\n\n";

    size_t count = 10000000;  // 10M
    auto data = generate_random_i32(count);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);
    conn.Query("CREATE TABLE t (value INTEGER)");
    {
        duckdb::Appender appender(conn, "t");
        for (auto v : data) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT SUM(value) FROM t");
    }, 2, 5);

    // V2 (单线程 SIMD)
    double v2_time = benchmark_fn([&]() {
        aggregate::sum_i32_v2(data.data(), count);
    }, 2, 5);

    // 并行 SUM
    double parallel_time = benchmark_fn([&]() {
        aggregate::sum_i32_parallel(data.data(), count);
    }, 2, 5);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "DuckDB:           " << duckdb_time << " ms (1.00x)\n";
    std::cout << "V2 (单线程):      " << v2_time << " ms (" << duckdb_time/v2_time << "x)\n";
    std::cout << "V10 (多线程):     " << parallel_time << " ms (" << duckdb_time/parallel_time << "x)\n";
}

void test_hash_join() {
    std::cout << "\n=== Hash Join 测试 (100K build x 1M probe) ===\n\n";

    size_t build_size = 100000;
    size_t probe_size = 1000000;

    auto build_keys = generate_random_i32(build_size);
    auto probe_keys = generate_random_i32(probe_size);

    join::JoinResult* result = join::create_join_result(probe_size * 2);

    // DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);

    conn.Query("CREATE TABLE build_t (key INTEGER)");
    conn.Query("CREATE TABLE probe_t (key INTEGER)");

    {
        duckdb::Appender appender(conn, "build_t");
        for (auto v : build_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }
    {
        duckdb::Appender appender(conn, "probe_t");
        for (auto v : probe_keys) {
            appender.BeginRow();
            appender.Append<int32_t>(v);
            appender.EndRow();
        }
    }

    double duckdb_time = benchmark_fn([&]() {
        conn.Query("SELECT COUNT(*) FROM build_t INNER JOIN probe_t ON build_t.key = probe_t.key");
    }, 2, 5);

    // V7 (自适应)
    double v7_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v4(build_keys.data(), build_size,
                               probe_keys.data(), probe_size,
                               join::JoinType::INNER, result);
    }, 2, 5);

    // V11 (SIMD简化)
    double v11_time = benchmark_fn([&]() {
        result->count = 0;
        join::hash_join_i32_v11(build_keys.data(), build_size,
                                probe_keys.data(), probe_size,
                                join::JoinType::INNER, result);
    }, 2, 5);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "DuckDB:           " << duckdb_time << " ms (1.00x)\n";
    std::cout << "V7 (自适应):      " << v7_time << " ms (" << duckdb_time/v7_time << "x)\n";
    std::cout << "V11 (SIMD):       " << v11_time << " ms (" << duckdb_time/v11_time << "x)\n";

    join::free_join_result(result);
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           ThunderDuck 并行优化测试                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    test_parallel_filter();
    test_parallel_aggregate();
    test_hash_join();

    std::cout << "\n测试完成!\n";
    return 0;
}
