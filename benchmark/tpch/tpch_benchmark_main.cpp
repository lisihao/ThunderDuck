/**
 * ThunderDuck TPC-H 基准测试 - 主入口
 *
 * 运行完整的 TPC-H 基准测试并生成报告
 *
 * 用法:
 *   ./tpch_benchmark [options]
 *
 * 选项:
 *   --sf <n>       设置 Scale Factor (默认 1)
 *   --category <A|B|C>  只运行指定类别
 *   --query <Qn>   只运行指定查询
 *   --iterations <n>    测量迭代次数 (默认 10)
 *   --warmup <n>        预热次数 (默认 2)
 *   --output <file>     报告输出路径
 *   -q, --quiet         安静模式
 *   -h, --help          显示帮助
 *
 * @version 1.0
 * @date 2026-01-28
 */

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>

#include "tpch_data_loader.h"
#include "tpch_executor.h"
#include "tpch_queries.h"
#include "tpch_report.h"

#include "duckdb.hpp"

using namespace thunderduck::tpch;

// ============================================================================
// 命令行参数解析
// ============================================================================

struct Options {
    int scale_factor = 1;
    size_t iterations = 10;
    size_t warmup = 2;
    std::string category;     // 空 = 全部
    std::string query;        // 空 = 全部
    std::string output;       // 空 = 默认路径
    bool quiet = false;
    bool help = false;
};

void print_help() {
    std::cout << R"(
ThunderDuck TPC-H Benchmark v1.0

用法: tpch_benchmark [options]

选项:
  --sf <n>              Scale Factor (默认: 1)
  --category <A|B|C>    只运行指定类别的查询
  --query <Qn>          只运行指定查询 (如 Q6)
  --iterations <n>      测量迭代次数 (默认: 10)
  --warmup <n>          预热次数 (默认: 2)
  --output <file>       报告输出路径
  -q, --quiet           安静模式
  -h, --help            显示帮助

类别说明:
  A: 完全可优化 (Q1,Q3,Q5,Q6,Q7,Q9,Q10,Q12,Q14,Q18) - 预期 1.5-3x
  B: 部分可优化 (Q2,Q4,Q11,Q15,Q16,Q19) - 预期 1.0-1.5x
  C: DuckDB 回退 (Q8,Q13,Q17,Q20,Q21,Q22) - 预期 ~1.0x

示例:
  tpch_benchmark --sf 1                    # 运行 SF=1 完整测试
  tpch_benchmark --sf 10 --category A      # 运行 SF=10 Category A
  tpch_benchmark --query Q6                # 只运行 Q6
  tpch_benchmark --sf 1 --output report.md # 输出到指定文件

)";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.help = true;
        } else if (arg == "-q" || arg == "--quiet") {
            opts.quiet = true;
        } else if (arg == "--sf" && i + 1 < argc) {
            opts.scale_factor = std::stoi(argv[++i]);
        } else if (arg == "--category" && i + 1 < argc) {
            opts.category = argv[++i];
        } else if (arg == "--query" && i + 1 < argc) {
            opts.query = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            opts.iterations = std::stoul(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = std::stoul(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else {
            std::cerr << "未知选项: " << arg << std::endl;
            opts.help = true;
        }
    }

    return opts;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Options opts = parse_args(argc, argv);

    if (opts.help) {
        print_help();
        return 0;
    }

    // 打印标题
    if (!opts.quiet) {
        std::cout << "============================================================\n";
        std::cout << "    ThunderDuck TPC-H 完整基准测试系统 v1.0                 \n";
        std::cout << "    对比: DuckDB vs ThunderDuck                             \n";
        std::cout << "============================================================\n";
    }

    // 创建内存数据库
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    // 创建数据加载器
    TPCHDataLoader loader(con);

    // 生成 TPC-H 数据
    if (!loader.generate_data(opts.scale_factor)) {
        std::cerr << "TPC-H 数据生成失败!" << std::endl;
        return 1;
    }

    // 提取数据到内存
    loader.extract_all_tables();
    loader.print_stats();

    // 创建执行器
    TPCHExecutor executor(loader, opts.iterations, opts.warmup);

    BenchmarkResult result;

    // 运行测试
    if (!opts.query.empty()) {
        // 运行单个查询
        if (!opts.quiet) {
            std::cout << "\n运行查询 " << opts.query << "...\n";
        }
        auto qr = executor.run_query(opts.query, !opts.quiet);
        result.queries.push_back(qr);
        result.scale_factor = opts.scale_factor;

        // 获取日期
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&time));
        result.date = buf;

        result.total_duckdb_ms = qr.duckdb_ms;
        result.total_thunderduck_ms = qr.thunderduck_ms;
        result.geometric_mean_speedup = qr.speedup;
        if (qr.speedup > 1.05) result.faster_count = 1;
        else if (qr.speedup < 0.95) result.slower_count = 1;
        else result.same_count = 1;

    } else if (!opts.category.empty()) {
        // 运行指定类别
        QueryCategory cat;
        if (opts.category == "A") cat = QueryCategory::A;
        else if (opts.category == "B") cat = QueryCategory::B;
        else if (opts.category == "C") cat = QueryCategory::C;
        else {
            std::cerr << "无效的类别: " << opts.category << std::endl;
            return 1;
        }

        result = executor.run_category(cat, !opts.quiet);

    } else {
        // 运行所有查询
        result = executor.run_all_queries(!opts.quiet);
    }

    // 打印汇总
    print_summary(result);

    // 生成报告
    std::string output_path = opts.output;
    if (output_path.empty()) {
        output_path = "docs/TPCH_BENCHMARK_REPORT_SF" +
                      std::to_string(opts.scale_factor) + ".md";
    }
    generate_report(result, output_path);

    // 成功标准检查
    if (!opts.quiet) {
        std::cout << "\n";
        std::cout << "============================================================\n";

        bool success = true;
        std::cout << "成功标准检查:\n";

        // 1. 22 条查询全部运行通过
        bool all_passed = true;
        for (const auto& q : result.queries) {
            if (!q.correct) all_passed = false;
        }
        std::cout << "  [" << (all_passed ? "PASS" : "FAIL") << "] 所有查询运行通过\n";
        if (!all_passed) success = false;

        // 2. Category A 平均加速比 >= 1.5x
        double cat_a_speedup = 0;
        int cat_a_count = 0;
        for (const auto& q : result.queries) {
            if (q.category == "A") {
                cat_a_speedup += q.speedup;
                cat_a_count++;
            }
        }
        if (cat_a_count > 0) {
            cat_a_speedup /= cat_a_count;
            bool cat_a_ok = (cat_a_speedup >= 1.5);
            std::cout << "  [" << (cat_a_ok ? "PASS" : "FAIL")
                      << "] Category A 平均加速比 >= 1.5x (实际: "
                      << std::fixed << std::setprecision(2) << cat_a_speedup << "x)\n";
            if (!cat_a_ok) success = false;
        }

        // 3. 几何平均加速比 >= 1.3x
        bool geo_ok = (result.geometric_mean_speedup >= 1.3);
        std::cout << "  [" << (geo_ok ? "PASS" : "FAIL")
                  << "] 几何平均加速比 >= 1.3x (实际: "
                  << result.geometric_mean_speedup << "x)\n";
        if (!geo_ok) success = false;

        std::cout << "============================================================\n";
        std::cout << "最终结果: " << (success ? "SUCCESS" : "NEEDS IMPROVEMENT") << "\n";
        std::cout << "============================================================\n";
    }

    return 0;
}
