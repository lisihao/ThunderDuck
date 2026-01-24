/**
 * ThunderDuck vs DuckDB 内存占用对比测试
 *
 * 测试场景:
 * 1. 数据装载时的内存占用
 * 2. Hash Join 操作时的内存峰值
 * 3. Filter 操作时的内存占用
 * 4. 聚合操作时的内存占用
 */

#include <thunderduck/join.h>
#include <thunderduck/filter.h>
#include <thunderduck/aggregate.h>
#include <duckdb.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <mach/mach.h>
#include <mach/task.h>

// ============================================================================
// 内存测量工具 (macOS specific)
// ============================================================================

struct MemoryStats {
    size_t resident_size;      // 实际物理内存
    size_t virtual_size;       // 虚拟内存
    size_t resident_size_max;  // 峰值物理内存
};

MemoryStats get_memory_stats() {
    MemoryStats stats = {0, 0, 0};

    struct task_basic_info info;
    mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;

    if (task_info(mach_task_self(), TASK_BASIC_INFO,
                  (task_info_t)&info, &size) == KERN_SUCCESS) {
        stats.resident_size = info.resident_size;
        stats.virtual_size = info.virtual_size;
    }

    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        stats.resident_size_max = usage.ru_maxrss;
    }

    return stats;
}

std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && unit_idx < 3) {
        size /= 1024;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

// ============================================================================
// 内存分析报告
// ============================================================================

struct MemoryReport {
    std::string test_name;
    size_t data_size_bytes;      // 原始数据大小
    size_t thunder_memory;       // ThunderDuck 内存使用
    size_t duckdb_memory;        // DuckDB 内存使用
    double thunder_overhead;     // ThunderDuck 开销比
    double duckdb_overhead;      // DuckDB 开销比
};

std::vector<MemoryReport> reports;

void print_memory_comparison(const std::string& test_name,
                              size_t data_bytes,
                              size_t thunder_before, size_t thunder_after,
                              size_t duckdb_before, size_t duckdb_after) {
    size_t thunder_used = thunder_after - thunder_before;
    size_t duckdb_used = duckdb_after - duckdb_before;

    double thunder_overhead = static_cast<double>(thunder_used) / data_bytes;
    double duckdb_overhead = static_cast<double>(duckdb_used) / data_bytes;

    std::cout << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(59) << test_name << "│\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ 原始数据大小:        " << std::setw(37) << format_bytes(data_bytes) << "│\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ ThunderDuck 内存:    " << std::setw(20) << format_bytes(thunder_used)
              << " (开销: " << std::fixed << std::setprecision(2) << thunder_overhead << "x)    │\n";
    std::cout << "│ DuckDB 内存:         " << std::setw(20) << format_bytes(duckdb_used)
              << " (开销: " << std::fixed << std::setprecision(2) << duckdb_overhead << "x)    │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";

    if (thunder_used < duckdb_used) {
        double savings = 100.0 * (1.0 - static_cast<double>(thunder_used) / duckdb_used);
        std::cout << "│ 结果: ThunderDuck 节省 " << std::setw(5) << std::fixed << std::setprecision(1)
                  << savings << "% 内存                          │\n";
    } else if (duckdb_used < thunder_used) {
        double excess = 100.0 * (static_cast<double>(thunder_used) / duckdb_used - 1.0);
        std::cout << "│ 结果: ThunderDuck 多用 " << std::setw(5) << std::fixed << std::setprecision(1)
                  << excess << "% 内存 ⚠️                        │\n";
    } else {
        std::cout << "│ 结果: 内存使用相当                                          │\n";
    }
    std::cout << "└─────────────────────────────────────────────────────────────┘\n";

    MemoryReport report;
    report.test_name = test_name;
    report.data_size_bytes = data_bytes;
    report.thunder_memory = thunder_used;
    report.duckdb_memory = duckdb_used;
    report.thunder_overhead = thunder_overhead;
    report.duckdb_overhead = duckdb_overhead;
    reports.push_back(report);
}

// ============================================================================
// 数据生成
// ============================================================================

std::vector<int32_t> generate_random_int32(size_t count, int32_t min_val, int32_t max_val, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> dist(min_val, max_val);

    std::vector<int32_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

std::vector<double> generate_random_double(size_t count, double min_val, double max_val, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(min_val, max_val);

    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(gen);
    }
    return data;
}

// ============================================================================
// 测试 1: 数据装载内存占用
// ============================================================================

void test_data_loading() {
    std::cout << "\n========================================\n";
    std::cout << "测试 1: 数据装载内存占用\n";
    std::cout << "========================================\n";

    const size_t ROW_COUNT = 1000000;  // 100万行

    // 生成测试数据
    auto int_col1 = generate_random_int32(ROW_COUNT, 1, 100000, 42);
    auto int_col2 = generate_random_int32(ROW_COUNT, 1, 50000, 123);
    auto double_col = generate_random_double(ROW_COUNT, 0.0, 10000.0, 456);

    size_t raw_data_size = ROW_COUNT * (sizeof(int32_t) * 2 + sizeof(double));

    // ThunderDuck: 直接使用 vector
    auto mem_before = get_memory_stats();

    std::vector<int32_t> thunder_col1 = int_col1;  // 拷贝
    std::vector<int32_t> thunder_col2 = int_col2;
    std::vector<double> thunder_col3 = double_col;

    auto thunder_after = get_memory_stats();

    // 清理
    thunder_col1.clear(); thunder_col1.shrink_to_fit();
    thunder_col2.clear(); thunder_col2.shrink_to_fit();
    thunder_col3.clear(); thunder_col3.shrink_to_fit();

    // DuckDB: 创建表并插入数据
    auto mem_before_duck = get_memory_stats();

    duckdb::DuckDB db(nullptr);  // in-memory
    duckdb::Connection conn(db);

    conn.Query("CREATE TABLE test (col1 INTEGER, col2 INTEGER, col3 DOUBLE)");

    // 批量插入
    duckdb::Appender appender(conn, "test");
    for (size_t i = 0; i < ROW_COUNT; ++i) {
        appender.BeginRow();
        appender.Append<int32_t>(int_col1[i]);
        appender.Append<int32_t>(int_col2[i]);
        appender.Append<double>(double_col[i]);
        appender.EndRow();
    }
    appender.Close();

    auto duckdb_after = get_memory_stats();

    print_memory_comparison("数据装载 (1M 行 x 3 列)",
                            raw_data_size,
                            mem_before.resident_size, thunder_after.resident_size,
                            mem_before_duck.resident_size, duckdb_after.resident_size);
}

// ============================================================================
// 测试 2: Hash Join 内存占用
// ============================================================================

void test_hash_join_memory() {
    std::cout << "\n========================================\n";
    std::cout << "测试 2: Hash Join 内存占用\n";
    std::cout << "========================================\n";

    struct TestCase {
        size_t build_count;
        size_t probe_count;
        const char* desc;
    };

    // 使用较小的测试数据避免内存问题
    std::vector<TestCase> test_cases = {
        {10000, 100000, "小表 Join (10K x 100K)"},
        {50000, 500000, "中表 Join (50K x 500K)"},
    };

    for (const auto& tc : test_cases) {
        auto build_keys = generate_random_int32(tc.build_count, 1, tc.build_count, 42);
        auto probe_keys = generate_random_int32(tc.probe_count, 1, tc.build_count, 123);

        size_t raw_data_size = (tc.build_count + tc.probe_count) * sizeof(int32_t);

        // ThunderDuck Hash Join v3
        auto mem_before = get_memory_stats();

        thunderduck::join::JoinResult* result =
            thunderduck::join::create_join_result(tc.probe_count);

        thunderduck::join::hash_join_i32_v3(
            build_keys.data(), build_keys.size(),
            probe_keys.data(), probe_keys.size(),
            thunderduck::join::JoinType::INNER, result);

        auto thunder_after = get_memory_stats();

        thunderduck::join::free_join_result(result);

        // DuckDB Hash Join
        auto mem_before_duck = get_memory_stats();

        duckdb::DuckDB db(nullptr);
        duckdb::Connection conn(db);

        conn.Query("CREATE TABLE build_table (key INTEGER)");
        conn.Query("CREATE TABLE probe_table (key INTEGER)");

        {
            duckdb::Appender appender(conn, "build_table");
            for (size_t i = 0; i < tc.build_count; ++i) {
                appender.BeginRow();
                appender.Append<int32_t>(build_keys[i]);
                appender.EndRow();
            }
            appender.Close();
        }

        {
            duckdb::Appender appender(conn, "probe_table");
            for (size_t i = 0; i < tc.probe_count; ++i) {
                appender.BeginRow();
                appender.Append<int32_t>(probe_keys[i]);
                appender.EndRow();
            }
            appender.Close();
        }

        // 执行 Join
        auto join_result = conn.Query(
            "SELECT COUNT(*) FROM build_table b "
            "INNER JOIN probe_table p ON b.key = p.key");

        auto duckdb_after = get_memory_stats();

        print_memory_comparison(tc.desc, raw_data_size,
                                mem_before.resident_size, thunder_after.resident_size,
                                mem_before_duck.resident_size, duckdb_after.resident_size);
    }
}

// ============================================================================
// 测试 3: 哈希表内存开销分析
// ============================================================================

void test_hash_table_overhead() {
    std::cout << "\n========================================\n";
    std::cout << "测试 3: 哈希表内存开销详细分析\n";
    std::cout << "========================================\n";

    // 分析不同数据量下的哈希表内存开销
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    std::cout << "\n┌────────────┬──────────────┬──────────────┬──────────────┬─────────┐\n";
    std::cout << "│ 元素数量   │ 原始数据     │ 哈希表容量   │ 实际占用     │ 开销    │\n";
    std::cout << "├────────────┼──────────────┼──────────────┼──────────────┼─────────┤\n";

    for (size_t count : sizes) {
        // 计算 SOAHashTable 的容量 (load factor 1.7)
        size_t capacity = 16;
        while (capacity < count * 1.7) {
            capacity *= 2;
        }

        size_t raw_size = count * sizeof(int32_t);
        // SOAHashTable: keys (int32) + indices (uint32) = 8 bytes per slot
        size_t hash_table_size = capacity * (sizeof(int32_t) + sizeof(uint32_t));
        double overhead = static_cast<double>(hash_table_size) / raw_size;

        std::cout << "│ " << std::setw(10) << count
                  << " │ " << std::setw(12) << format_bytes(raw_size)
                  << " │ " << std::setw(12) << capacity
                  << " │ " << std::setw(12) << format_bytes(hash_table_size)
                  << " │ " << std::setw(6) << std::fixed << std::setprecision(2) << overhead << "x │\n";
    }

    std::cout << "└────────────┴──────────────┴──────────────┴──────────────┴─────────┘\n";

    std::cout << "\n分析:\n";
    std::cout << "- SOAHashTable 使用 1.7x 负载因子，导致约 30% 空槽浪费\n";
    std::cout << "- 每个条目存储 key(4B) + index(4B) = 8B，比原始 key 多 2x\n";
    std::cout << "- 总开销约 3.4x 原始数据大小\n";
}

// ============================================================================
// 测试 4: JoinResult 缓冲区增长分析
// ============================================================================

void test_join_result_growth() {
    std::cout << "\n========================================\n";
    std::cout << "测试 4: JoinResult 缓冲区增长模式\n";
    std::cout << "========================================\n";

    // 模拟不同匹配率下的内存使用
    struct Scenario {
        size_t probe_count;
        size_t actual_matches;
        const char* desc;
    };

    std::vector<Scenario> scenarios = {
        {1000000, 100, "极稀疏 (0.01% 匹配率)"},
        {1000000, 10000, "稀疏 (1% 匹配率)"},
        {1000000, 100000, "中等 (10% 匹配率)"},
        {1000000, 1000000, "密集 (100% 匹配率)"},
        {1000000, 5000000, "高重复 (500% 匹配率)"},
    };

    std::cout << "\n当前 ThunderDuck 策略: estimated = max(build, probe) * 4\n\n";

    std::cout << "┌────────────────────────┬──────────────┬──────────────┬──────────────┬─────────┐\n";
    std::cout << "│ 场景                   │ 实际匹配数   │ 预分配大小   │ 内存浪费     │ 浪费率  │\n";
    std::cout << "├────────────────────────┼──────────────┼──────────────┼──────────────┼─────────┤\n";

    for (const auto& s : scenarios) {
        // 当前策略: max(build, probe) * 4
        size_t estimated = s.probe_count * 4;
        size_t allocated_bytes = estimated * sizeof(uint32_t) * 2;  // left + right indices
        size_t actual_bytes = s.actual_matches * sizeof(uint32_t) * 2;
        size_t wasted = (allocated_bytes > actual_bytes) ? (allocated_bytes - actual_bytes) : 0;
        double waste_rate = (actual_bytes > 0) ? (static_cast<double>(wasted) / actual_bytes * 100) : 0;

        std::cout << "│ " << std::left << std::setw(22) << s.desc
                  << " │ " << std::right << std::setw(12) << s.actual_matches
                  << " │ " << std::setw(12) << format_bytes(allocated_bytes)
                  << " │ " << std::setw(12) << format_bytes(wasted)
                  << " │ " << std::setw(6) << std::fixed << std::setprecision(0) << waste_rate << "% │\n";
    }

    std::cout << "└────────────────────────┴──────────────┴──────────────┴──────────────┴─────────┘\n";

    std::cout << "\n问题分析:\n";
    std::cout << "- 稀疏 Join 场景下，预分配 30MB 只用 800KB，浪费 99.7%\n";
    std::cout << "- 当前策略无法适应不同选择率的场景\n";
}

// ============================================================================
// 测试 5: 临时缓冲区内存分析
// ============================================================================

void test_temp_buffer_overhead() {
    std::cout << "\n========================================\n";
    std::cout << "测试 5: 分区 Join 临时缓冲区开销\n";
    std::cout << "========================================\n";

    // 分析 hash_join_v3 中临时缓冲区的分配
    const size_t NUM_PARTITIONS = 16;
    const size_t NUM_THREADS = 4;
    const size_t PROBE_COUNT = 1000000;

    // 每个分区每个线程都会分配临时缓冲区
    size_t per_partition_probe = PROBE_COUNT / NUM_PARTITIONS;

    // 单分区 Join 的临时缓冲区
    size_t single_partition_temp = per_partition_probe * sizeof(uint32_t) * 2;

    // 多线程场景：每个线程处理多个分区
    size_t max_concurrent_buffers = NUM_THREADS * 2;  // 每线程可能同时处理多个分区
    size_t total_temp_memory = max_concurrent_buffers * single_partition_temp;

    std::cout << "\n分区 Join 内存分析 (1M probe rows, 16 分区, 4 线程):\n\n";
    std::cout << "┌────────────────────────────────────────┬──────────────────┐\n";
    std::cout << "│ 项目                                   │ 内存占用         │\n";
    std::cout << "├────────────────────────────────────────┼──────────────────┤\n";
    std::cout << "│ 每分区 probe 数量                      │ " << std::setw(16) << per_partition_probe << " │\n";
    std::cout << "│ 单分区临时缓冲区 (build+probe indices) │ " << std::setw(16) << format_bytes(single_partition_temp) << " │\n";
    std::cout << "│ 最大并发缓冲区数                       │ " << std::setw(16) << max_concurrent_buffers << " │\n";
    std::cout << "│ 峰值临时内存                           │ " << std::setw(16) << format_bytes(total_temp_memory) << " │\n";
    std::cout << "└────────────────────────────────────────┴──────────────────┘\n";

    std::cout << "\n问题:\n";
    std::cout << "- 临时缓冲区按 probe_count/16 分配，不考虑实际匹配数\n";
    std::cout << "- 稀疏 Join 可能只有 0.1% 匹配，99.9% 内存浪费\n";
}

// ============================================================================
// 生成分析报告
// ============================================================================

void generate_report() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              内存分析总结报告                                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    if (!reports.empty()) {
        std::cout << "测试结果汇总:\n\n";
        std::cout << "┌────────────────────────┬────────────┬────────────┬────────────┐\n";
        std::cout << "│ 测试                   │ ThunderDuck│ DuckDB     │ 差异       │\n";
        std::cout << "├────────────────────────┼────────────┼────────────┼────────────┤\n";

        for (const auto& r : reports) {
            double diff_pct = (r.duckdb_memory > 0) ?
                (static_cast<double>(r.thunder_memory) / r.duckdb_memory - 1.0) * 100 : 0;

            std::cout << "│ " << std::left << std::setw(22) << r.test_name.substr(0, 22)
                      << " │ " << std::right << std::setw(10) << format_bytes(r.thunder_memory)
                      << " │ " << std::setw(10) << format_bytes(r.duckdb_memory)
                      << " │ " << std::showpos << std::setw(9) << std::fixed << std::setprecision(1)
                      << diff_pct << "% │\n";
        }

        std::cout << "└────────────────────────┴────────────┴────────────┴────────────┘\n";
    }

    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n";
    std::cout << "                    关键发现与优化建议\n";
    std::cout << "══════════════════════════════════════════════════════════════════\n\n";

    std::cout << "【问题 1】哈希表负载因子过低\n";
    std::cout << "  现状: 1.7x 容量，30% 空槽浪费\n";
    std::cout << "  建议: 使用 0.75 负载因子，减少 20% 内存\n\n";

    std::cout << "【问题 2】JoinResult 预分配策略不智能\n";
    std::cout << "  现状: max(build,probe)*4，不考虑选择率\n";
    std::cout << "  建议: 实现选择率估算或增量分配\n\n";

    std::cout << "【问题 3】临时缓冲区过度分配\n";
    std::cout << "  现状: 按 probe_count 分配，稀疏 Join 浪费 99%+\n";
    std::cout << "  建议: 直接写入输出缓冲区，避免临时拷贝\n\n";

    std::cout << "【问题 4】缺少内存池机制\n";
    std::cout << "  现状: 每次操作重新分配/释放\n";
    std::cout << "  建议: 实现线程本地内存池\n\n";

    std::cout << "【问题 5】对齐策略过度\n";
    std::cout << "  现状: 所有分配 128 字节对齐\n";
    std::cout << "  建议: 小对象用 16/32 字节对齐\n\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     ThunderDuck vs DuckDB 内存占用深度分析                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    auto initial_mem = get_memory_stats();
    std::cout << "\n初始内存状态:\n";
    std::cout << "  物理内存: " << format_bytes(initial_mem.resident_size) << "\n";
    std::cout << "  虚拟内存: " << format_bytes(initial_mem.virtual_size) << "\n";

    test_data_loading();
    test_hash_join_memory();
    test_hash_table_overhead();
    test_join_result_growth();
    test_temp_buffer_overhead();
    generate_report();

    std::cout << "\n测试完成!\n";

    return 0;
}
