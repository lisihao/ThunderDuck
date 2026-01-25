/**
 * TopK 基数敏感性测试 - v4 vs v5 对比
 *
 * 目标: 验证 v5 Count-Based TopK 在低基数下的性能提升
 *
 * 测试变量:
 * - 数据量: 10M (固定，这是 T4 场景)
 * - K 值: 10 (固定)
 * - 基数: 从 10 到 10,000,000 (对数增长)
 */

#include "thunderduck/sort.h"
#include <duckdb.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std::chrono;

// 配置
constexpr size_t DATA_SIZE = 10000000;  // 10M 行
constexpr size_t K = 10;
constexpr int WARMUP = 3;
constexpr int ITERATIONS = 10;

// 生成指定基数的随机数据
std::vector<int32_t> generate_data_with_cardinality(size_t n, size_t cardinality, int seed = 42) {
    std::vector<int32_t> data(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(1, static_cast<int32_t>(cardinality));

    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// 计时辅助
template<typename Func>
double benchmark(Func func, int warmup = WARMUP, int iterations = ITERATIONS) {
    for (int i = 0; i < warmup; ++i) func();

    double total_ms = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        total_ms += duration<double, std::milli>(end - start).count();
    }
    return total_ms / iterations;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    TopK 基数敏感性测试 - v4 vs v5 vs DuckDB                                       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  数据量: 10M 行 | K=10 | v5: Count-Based 低基数优化                                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    // 初始化 DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);

    // 测试不同基数 (精简版，聚焦关键点)
    std::vector<size_t> cardinalities = {
        10, 20, 50,                          // 极低基数
        100, 200, 500,                       // 低基数
        1000, 2000, 5000,                    // 中低基数
        10000, 50000,                        // 中等基数
        100000, 1000000,                     // 高基数
        10000000                             // 完全唯一
    };

    std::cout << "┌────────────┬─────────────┬─────────────┬─────────────┬──────────┬──────────┬────────────┐\n";
    std::cout << "│   基数     │ DuckDB (ms) │  v4 (ms)    │  v5 (ms)    │ v5加速比 │ v5/v4    │ 胜者       │\n";
    std::cout << "├────────────┼─────────────┼─────────────┼─────────────┼──────────┼──────────┼────────────┤\n";

    size_t v4_crossover = 0, v5_crossover = 0;
    bool found_v4_crossover = false, found_v5_crossover = false;
    int v5_wins = 0, duckdb_wins = 0;

    for (size_t cardinality : cardinalities) {
        // 生成数据
        auto data = generate_data_with_cardinality(DATA_SIZE, cardinality);

        // 加载到 DuckDB (使用批量 INSERT)
        conn.Query("DROP TABLE IF EXISTS test_data");
        conn.Query("CREATE TABLE test_data (val INTEGER)");

        constexpr size_t BATCH_SIZE = 10000;
        for (size_t batch_start = 0; batch_start < data.size(); batch_start += BATCH_SIZE) {
            std::string insert_sql = "INSERT INTO test_data VALUES ";
            size_t batch_end = std::min(batch_start + BATCH_SIZE, data.size());

            for (size_t i = batch_start; i < batch_end; ++i) {
                if (i > batch_start) insert_sql += ",";
                insert_sql += "(" + std::to_string(data[i]) + ")";
            }
            conn.Query(insert_sql);
        }

        // 测试 DuckDB
        std::string sql = "SELECT val FROM test_data ORDER BY val DESC LIMIT 10";
        double duckdb_time = benchmark([&]() {
            conn.Query(sql);
        });

        // 测试 ThunderDuck v4
        std::vector<int32_t> topk_values(K);
        std::vector<uint32_t> topk_indices(K);
        double v4_time = benchmark([&]() {
            thunderduck::sort::topk_max_i32_v4(data.data(), data.size(), K,
                                               topk_values.data(), topk_indices.data());
        });

        // 测试 ThunderDuck v5
        double v5_time = benchmark([&]() {
            thunderduck::sort::topk_max_i32_v5(data.data(), data.size(), K,
                                               topk_values.data(), topk_indices.data());
        });

        double v5_speedup = duckdb_time / v5_time;
        double v5_vs_v4 = v4_time / v5_time;
        const char* winner = v5_speedup >= 1.0 ? "Thunder v5" : "DuckDB";

        if (v5_speedup >= 1.0) v5_wins++; else duckdb_wins++;

        // 检测临界点
        if (!found_v4_crossover && duckdb_time / v4_time >= 1.0) {
            v4_crossover = cardinality;
            found_v4_crossover = true;
        }
        if (!found_v5_crossover && v5_speedup >= 1.0) {
            v5_crossover = cardinality;
            found_v5_crossover = true;
        }

        // 格式化输出
        std::cout << "│ " << std::setw(10) << cardinality << " │ ";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(11) << duckdb_time << " │ ";
        std::cout << std::setw(11) << v4_time << " │ ";
        std::cout << std::setw(11) << v5_time << " │ ";

        // 加速比显示
        if (v5_speedup >= 1.0) {
            std::cout << "\033[32m" << std::setw(7) << v5_speedup << "x\033[0m │ ";
        } else {
            std::cout << "\033[31m" << std::setw(7) << v5_speedup << "x\033[0m │ ";
        }

        // v5 vs v4 提升
        if (v5_vs_v4 >= 1.0) {
            std::cout << "\033[32m" << std::setw(7) << v5_vs_v4 << "x\033[0m │ ";
        } else {
            std::cout << "\033[33m" << std::setw(7) << v5_vs_v4 << "x\033[0m │ ";
        }

        // 胜者
        if (v5_speedup >= 1.0) {
            std::cout << "\033[32m" << std::setw(10) << winner << "\033[0m │\n";
        } else {
            std::cout << "\033[31m" << std::setw(10) << winner << "\033[0m │\n";
        }
    }

    std::cout << "└────────────┴─────────────┴─────────────┴─────────────┴──────────┴──────────┴────────────┘\n\n";

    // 分析结论
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "                                      分析结论                                            \n";
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "📊 总体战绩: ThunderDuck v5 胜 " << v5_wins << " / DuckDB 胜 " << duckdb_wins << "\n\n";

    if (found_v5_crossover) {
        std::cout << "🎯 v5 临界基数: " << v5_crossover << "\n";
        std::cout << "   当基数 >= " << v5_crossover << " 时，ThunderDuck v5 开始反超 DuckDB\n\n";
    }

    if (found_v4_crossover) {
        std::cout << "📈 v4 临界基数: " << v4_crossover << " (对比参考)\n";
        if (found_v5_crossover && v5_crossover < v4_crossover) {
            std::cout << "   ✅ v5 将临界基数从 " << v4_crossover << " 降低到 " << v5_crossover << "!\n";
        }
    }

    std::cout << "\n══════════════════════════════════════════════════════════════════════════════════════════\n";
    std::cout << "                                      v5 优化效果                                          \n";
    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════\n\n";

    std::cout << "v5 核心优化: Count-Based TopK\n";
    std::cout << "  - 低基数: 统计计数 O(n) + 唯一值排序 O(cardinality)\n";
    std::cout << "  - 高基数: 复用 v4 采样预过滤\n\n";

    std::cout << "══════════════════════════════════════════════════════════════════════════════════════════\n";

    return 0;
}
