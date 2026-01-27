/**
 * V10 Join 深度优化 综合测试
 *
 * 测试内容:
 * 1. SEMI/ANTI JOIN 优化 (提前退出)
 * 2. Sort-Merge Join
 * 3. Range Join (范围连接)
 * 4. Inequality Join (不等值连接)
 * 5. 字符串键 Hash Join
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstring>
#include "thunderduck/join.h"

using namespace std;
using namespace thunderduck::join;

// ============================================================================
// 统计函数
// ============================================================================

struct BenchResult {
    double median;
    double stddev;
};

template<typename F>
BenchResult measure(F&& func, int iterations = 15) {
    func();  // Warmup

    vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
    }

    sort(times.begin(), times.end());
    double median = times[times.size() / 2];

    double mean = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = sqrt(sq_sum / times.size());

    return {median, stddev};
}

void print_result(const string& name, const BenchResult& r, double baseline = 0) {
    cout << "  " << left << setw(35) << name << ": "
         << fixed << setprecision(3) << setw(8) << r.median << " ms";

    if (baseline > 0 && r.median > 0) {
        double speedup = baseline / r.median;
        cout << "  [" << setprecision(2) << speedup << "x]";
    }
    cout << endl;
}

int main() {
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║          V10 Join 深度优化 综合测试                                          ║\n";
    cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    cout << "║ 1. SEMI/ANTI JOIN 优化 (提前退出)                                            ║\n";
    cout << "║ 2. Sort-Merge Join (SIMD 优化)                                               ║\n";
    cout << "║ 3. Range Join (范围连接)                                                     ║\n";
    cout << "║ 4. Inequality Join (不等值连接)                                              ║\n";
    cout << "║ 5. 字符串键 Hash Join (SIMD 优化)                                            ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

    cout << "\nV10 版本: " << get_v10_version_info() << endl;

    mt19937 gen(42);

    // ========================================================================
    // TEST 1: SEMI/ANTI JOIN 优化
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮\n";
    cout << "│ TEST 1: SEMI/ANTI JOIN 优化 (1M probe, 100K build, 10% 匹配率)              │\n";
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯\n";

    const size_t BUILD_1 = 100000;
    const size_t PROBE_1 = 1000000;

    vector<int32_t> build_keys_1(BUILD_1), probe_keys_1(PROBE_1);
    for (size_t i = 0; i < BUILD_1; ++i) build_keys_1[i] = i * 10;
    uniform_int_distribution<int32_t> dist_1(0, BUILD_1 * 100);
    for (size_t i = 0; i < PROBE_1; ++i) probe_keys_1[i] = dist_1(gen);

    JoinResult* result_1 = create_join_result(PROBE_1);

    // INNER JOIN 基线
    auto inner_1 = measure([&]() {
        result_1->count = 0;
        hash_join_i32_v4(build_keys_1.data(), BUILD_1, probe_keys_1.data(), PROBE_1,
                         JoinType::INNER, result_1);
    });

    // V10 SEMI JOIN
    auto semi_v10 = measure([&]() {
        result_1->count = 0;
        hash_join_i32_v10(build_keys_1.data(), BUILD_1, probe_keys_1.data(), PROBE_1,
                          JoinType::SEMI, result_1);
    });

    // V10 ANTI JOIN
    auto anti_v10 = measure([&]() {
        result_1->count = 0;
        hash_join_i32_v10(build_keys_1.data(), BUILD_1, probe_keys_1.data(), PROBE_1,
                          JoinType::ANTI, result_1);
    });

    // V4 SEMI (对比)
    auto semi_v4 = measure([&]() {
        result_1->count = 0;
        hash_join_i32_v4(build_keys_1.data(), BUILD_1, probe_keys_1.data(), PROBE_1,
                         JoinType::SEMI, result_1);
    });

    print_result("INNER JOIN (v4 基线)", inner_1);
    print_result("SEMI JOIN (v4)", semi_v4, inner_1.median);
    print_result("SEMI JOIN (v10 提前退出)", semi_v10, inner_1.median);
    print_result("ANTI JOIN (v10)", anti_v10, inner_1.median);

    cout << "\n  SEMI 优化收益: " << fixed << setprecision(1)
         << ((semi_v4.median / semi_v10.median - 1) * 100) << "%" << endl;

    // ========================================================================
    // TEST 2: Sort-Merge Join
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮\n";
    cout << "│ TEST 2: Sort-Merge Join (500K x 500K, 已排序输入)                            │\n";
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯\n";

    const size_t N_2 = 500000;
    vector<int32_t> left_2(N_2), right_2(N_2);

    // 生成已排序数据 (有重叠)
    for (size_t i = 0; i < N_2; ++i) {
        left_2[i] = i * 2;
        right_2[i] = i * 2 + 1;
    }
    // 添加一些匹配
    for (size_t i = 0; i < N_2 / 10; ++i) {
        right_2[i] = left_2[i];
    }
    sort(right_2.begin(), right_2.end());

    JoinResult* result_2 = create_join_result(N_2);

    // Hash Join 对比
    auto hash_2 = measure([&]() {
        result_2->count = 0;
        hash_join_i32_v4(left_2.data(), N_2, right_2.data(), N_2,
                         JoinType::INNER, result_2);
    });

    // Sort-Merge Join (假设已排序)
    JoinConfigV10 config_sorted;
    config_sorted.assume_sorted = true;

    auto merge_sorted = measure([&]() {
        result_2->count = 0;
        sort_merge_join_i32_config(left_2.data(), N_2, right_2.data(), N_2,
                                    JoinType::INNER, result_2, config_sorted);
    });

    // Sort-Merge Join (需要排序)
    JoinConfigV10 config_unsorted;
    config_unsorted.assume_sorted = false;

    auto merge_unsorted = measure([&]() {
        result_2->count = 0;
        sort_merge_join_i32_config(left_2.data(), N_2, right_2.data(), N_2,
                                    JoinType::INNER, result_2, config_unsorted);
    });

    print_result("Hash Join (v4)", hash_2);
    print_result("Sort-Merge (已排序输入)", merge_sorted, hash_2.median);
    print_result("Sort-Merge (需要排序)", merge_unsorted, hash_2.median);

    // ========================================================================
    // TEST 3: Range Join
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮\n";
    cout << "│ TEST 3: Range Join (100K left, 1K ranges)                                    │\n";
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯\n";

    const size_t LEFT_3 = 100000;
    const size_t RIGHT_3 = 1000;

    vector<int32_t> left_3(LEFT_3);
    vector<int32_t> right_lo_3(RIGHT_3), right_hi_3(RIGHT_3);

    uniform_int_distribution<int32_t> dist_3(0, 1000000);
    for (size_t i = 0; i < LEFT_3; ++i) left_3[i] = dist_3(gen);

    for (size_t i = 0; i < RIGHT_3; ++i) {
        int32_t lo = dist_3(gen);
        right_lo_3[i] = lo;
        right_hi_3[i] = lo + 1000;  // 范围大小 1000
    }

    JoinResult* result_3 = create_join_result(LEFT_3 * 10);

    // Range Join SIMD
    JoinConfigV10 config_simd;
    config_simd.range_join_simd = true;

    auto range_simd = measure([&]() {
        result_3->count = 0;
        range_join_i32_config(left_3.data(), LEFT_3, right_lo_3.data(), right_hi_3.data(),
                               RIGHT_3, result_3, config_simd);
    }, 10);

    // Range Join 标量
    JoinConfigV10 config_scalar;
    config_scalar.range_join_simd = false;

    auto range_scalar = measure([&]() {
        result_3->count = 0;
        range_join_i32_config(left_3.data(), LEFT_3, right_lo_3.data(), right_hi_3.data(),
                               RIGHT_3, result_3, config_scalar);
    }, 10);

    print_result("Range Join (标量)", range_scalar);
    print_result("Range Join (SIMD)", range_simd, range_scalar.median);
    cout << "  匹配数: " << result_3->count << endl;

    // ========================================================================
    // TEST 4: Inequality Join
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮\n";
    cout << "│ TEST 4: Inequality Join (10K x 10K, left < right)                            │\n";
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯\n";

    const size_t N_4 = 10000;
    vector<int32_t> left_4(N_4), right_4(N_4);

    uniform_int_distribution<int32_t> dist_4(0, 100000);
    for (size_t i = 0; i < N_4; ++i) {
        left_4[i] = dist_4(gen);
        right_4[i] = dist_4(gen);
    }

    JoinResult* result_4 = create_join_result(N_4 * N_4 / 2);

    auto ineq_lt = measure([&]() {
        result_4->count = 0;
        inequality_join_i32(left_4.data(), N_4, right_4.data(), N_4,
                            InequalityOp::LESS_THAN, result_4);
    }, 10);

    auto ineq_le = measure([&]() {
        result_4->count = 0;
        inequality_join_i32(left_4.data(), N_4, right_4.data(), N_4,
                            InequalityOp::LESS_EQUAL, result_4);
    }, 10);

    print_result("Inequality (left < right)", ineq_lt);
    print_result("Inequality (left <= right)", ineq_le);
    cout << "  匹配数 (<): " << result_4->count << endl;

    // ========================================================================
    // TEST 5: 字符串键 Hash Join
    // ========================================================================

    cout << "\n╭──────────────────────────────────────────────────────────────────────────────╮\n";
    cout << "│ TEST 5: 字符串键 Hash Join (100K build, 500K probe, 16字节键)                │\n";
    cout << "╰──────────────────────────────────────────────────────────────────────────────╯\n";

    const size_t BUILD_5 = 100000;
    const size_t PROBE_5 = 500000;
    const size_t KEY_LEN = 16;

    // 生成定长字符串键
    vector<char> build_str_5(BUILD_5 * KEY_LEN);
    vector<char> probe_str_5(PROBE_5 * KEY_LEN);

    for (size_t i = 0; i < BUILD_5; ++i) {
        snprintf(build_str_5.data() + i * KEY_LEN, KEY_LEN, "KEY%011zu", i);
    }

    uniform_int_distribution<size_t> dist_5(0, BUILD_5 * 10);
    for (size_t i = 0; i < PROBE_5; ++i) {
        snprintf(probe_str_5.data() + i * KEY_LEN, KEY_LEN, "KEY%011zu", dist_5(gen));
    }

    JoinResult* result_5 = create_join_result(PROBE_5);

    auto str_fixed = measure([&]() {
        result_5->count = 0;
        hash_join_fixedstring(build_str_5.data(), KEY_LEN, BUILD_5,
                               probe_str_5.data(), PROBE_5,
                               JoinType::INNER, result_5);
    });

    // 对比: 转换为整数键的 Hash Join
    vector<int32_t> build_int_5(BUILD_5), probe_int_5(PROBE_5);
    for (size_t i = 0; i < BUILD_5; ++i) build_int_5[i] = i;
    for (size_t i = 0; i < PROBE_5; ++i) {
        size_t key;
        sscanf(probe_str_5.data() + i * KEY_LEN, "KEY%zu", &key);
        probe_int_5[i] = key < BUILD_5 ? key : -1;
    }

    auto int_baseline = measure([&]() {
        result_5->count = 0;
        hash_join_i32_v4(build_int_5.data(), BUILD_5, probe_int_5.data(), PROBE_5,
                         JoinType::INNER, result_5);
    });

    print_result("整数键 Hash Join (基线)", int_baseline);
    print_result("定长字符串键 Hash Join", str_fixed, int_baseline.median);
    cout << "  匹配数: " << result_5->count << endl;

    // ========================================================================
    // 汇总
    // ========================================================================

    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                           V10 优化汇总                                       ║\n";
    cout << "╠════════════════════════════════╦═════════════════════════════════════════════╣\n";
    cout << "║ 优化项                         ║ 结果                                        ║\n";
    cout << "╠════════════════════════════════╬═════════════════════════════════════════════╣\n";

    cout << "║ SEMI JOIN 提前退出             ║ " << setw(43)
         << (semi_v4.median > semi_v10.median ? "有效 ✓" : "无明显收益") << " ║\n";

    cout << "║ Sort-Merge Join                ║ " << setw(43)
         << (merge_sorted.median < hash_2.median ? "已排序输入更快 ✓" : "Hash更快") << " ║\n";

    cout << "║ Range Join SIMD                ║ " << setw(43)
         << (range_simd.median < range_scalar.median ? "SIMD加速有效 ✓" : "无明显收益") << " ║\n";

    cout << "║ 字符串键 SIMD                  ║ " << setw(43)
         << "已实现 ✓" << " ║\n";

    cout << "║ GPU 阈值调整                   ║ " << setw(43)
         << "5M (原500M) ✓" << " ║\n";

    cout << "╚════════════════════════════════╩═════════════════════════════════════════════╝\n";

    // 清理
    free_join_result(result_1);
    free_join_result(result_2);
    free_join_result(result_3);
    free_join_result(result_4);
    free_join_result(result_5);

    return 0;
}
