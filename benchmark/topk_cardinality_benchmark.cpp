/**
 * TopK 基数敏感性测试
 *
 * 目标: 找出 ThunderDuck 反超 DuckDB 的基数临界点
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
    std::cout << "╔═══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           TopK 基数敏感性测试 - 寻找反超临界点                         ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  数据量: 10M 行 | K=10 | 测试: ThunderDuck v4 vs DuckDB               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════╝\n\n";

    // 初始化 DuckDB
    duckdb::DuckDB db(nullptr);
    duckdb::Connection conn(db);

    // 测试不同基数
    std::vector<size_t> cardinalities = {
        10, 20, 50,                          // 极低基数
        100, 200, 500,                       // 低基数
        1000, 2000, 5000,                    // 中低基数
        10000, 20000, 50000,                 // 中等基数
        100000, 200000, 500000,              // 中高基数
        1000000, 2000000, 5000000,           // 高基数
        10000000                             // 完全唯一
    };

    std::cout << "┌────────────┬───────────────┬───────────────┬──────────┬──────────┐\n";
    std::cout << "│  基数      │ DuckDB (ms)   │ Thunder (ms)  │ 加速比   │ 胜者     │\n";
    std::cout << "├────────────┼───────────────┼───────────────┼──────────┼──────────┤\n";

    size_t crossover_cardinality = 0;
    bool found_crossover = false;

    for (size_t cardinality : cardinalities) {
        // 生成数据
        auto data = generate_data_with_cardinality(DATA_SIZE, cardinality);

        // 加载到 DuckDB (使用批量 INSERT)
        conn.Query("DROP TABLE IF EXISTS test_data");
        conn.Query("CREATE TABLE test_data (val INTEGER)");

        // 批量插入 (每批 10000 行)
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
        double thunder_time = benchmark([&]() {
            thunderduck::sort::topk_max_i32_v4(data.data(), data.size(), K,
                                               topk_values.data(), topk_indices.data());
        });

        double speedup = duckdb_time / thunder_time;
        const char* winner = speedup >= 1.0 ? "Thunder" : "DuckDB";

        // 检测临界点
        if (!found_crossover && speedup >= 1.0) {
            crossover_cardinality = cardinality;
            found_crossover = true;
        }

        // 格式化输出
        std::cout << "│ " << std::setw(10) << cardinality << " │ ";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(13) << duckdb_time << " │ ";
        std::cout << std::setw(13) << thunder_time << " │ ";

        if (speedup >= 1.0) {
            std::cout << "\033[32m" << std::setw(7) << speedup << "x\033[0m │ ";
            std::cout << "\033[32m" << std::setw(8) << winner << "\033[0m │\n";
        } else {
            std::cout << "\033[31m" << std::setw(7) << speedup << "x\033[0m │ ";
            std::cout << "\033[31m" << std::setw(8) << winner << "\033[0m │\n";
        }
    }

    std::cout << "└────────────┴───────────────┴───────────────┴──────────┴──────────┘\n\n";

    // 分析结论
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                           分析结论                                    \n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    if (found_crossover) {
        std::cout << "🎯 临界基数: " << crossover_cardinality << "\n";
        std::cout << "   当基数 >= " << crossover_cardinality << " 时，ThunderDuck 开始反超 DuckDB\n\n";

        double unique_ratio = static_cast<double>(crossover_cardinality) / DATA_SIZE * 100;
        std::cout << "📊 唯一值比例: " << std::fixed << std::setprecision(2) << unique_ratio << "%\n";
        std::cout << "   (基数 " << crossover_cardinality << " / 数据量 " << DATA_SIZE << ")\n\n";

        double avg_duplicates = static_cast<double>(DATA_SIZE) / crossover_cardinality;
        std::cout << "📈 平均重复次数: " << std::fixed << std::setprecision(1) << avg_duplicates << "\n";
        std::cout << "   每个唯一值平均出现 " << avg_duplicates << " 次\n\n";
    } else {
        std::cout << "⚠️  未找到临界点，DuckDB 在所有测试基数下均胜出\n";
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                           优化建议                                    \n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    if (found_crossover) {
        std::cout << "✅ 推荐使用 ThunderDuck 的场景:\n";
        std::cout << "   - 用户ID、订单ID等高基数字段的 TopK 查询\n";
        std::cout << "   - 时间戳、价格等连续数值的 TopK 查询\n";
        std::cout << "   - 基数 >= " << crossover_cardinality << " 的任何数据\n\n";

        std::cout << "⚠️  可能需要回退到 DuckDB 的场景:\n";
        std::cout << "   - 状态码、类别等低基数字段的 TopK 查询\n";
        std::cout << "   - 基数 < " << crossover_cardinality << " 的数据\n";
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";

    return 0;
}
