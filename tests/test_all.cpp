/**
 * ThunderDuck - Complete Test Suite
 * 
 * 完整的功能测试和性能基准测试
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/memory.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <set>

using namespace thunderduck;

// ============================================================================
// 测试工具
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop() { end_ = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() / 1000.0;
    }
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) \
    std::cout << "  " << std::setw(40) << std::left << name << std::flush; \
    try {

#define EXPECT(cond) \
    if (!(cond)) { \
        throw std::runtime_error("Assertion failed: " #cond); \
    }

#define END_TEST \
        std::cout << "\033[32m[PASS]\033[0m\n"; \
        tests_passed++; \
    } catch (const std::exception& e) { \
        std::cout << "\033[31m[FAIL]\033[0m " << e.what() << "\n"; \
        tests_failed++; \
    }

// ============================================================================
// Filter 测试
// ============================================================================

void test_filter() {
    std::cout << "\n\033[1m=== Filter Operator Tests ===\033[0m\n";
    
    TEST("filter_i32 GT") {
        alignas(128) int32_t data[] = {1, 5, 3, 8, 2, 9, 4, 7, 6, 10};
        std::vector<uint32_t> indices(10);
        size_t count = filter::filter_i32(data, 10, filter::CompareOp::GT, 5, indices.data());
        EXPECT(count == 5);  // 8, 9, 7, 6, 10
    } END_TEST
    
    TEST("filter_i32 EQ") {
        alignas(128) int32_t data[] = {1, 2, 3, 2, 5, 2, 7, 8};
        std::vector<uint32_t> indices(8);
        size_t count = filter::filter_i32(data, 8, filter::CompareOp::EQ, 2, indices.data());
        EXPECT(count == 3);
        EXPECT(indices[0] == 1 && indices[1] == 3 && indices[2] == 5);
    } END_TEST
    
    TEST("filter_i32 LT") {
        alignas(128) int32_t data[] = {5, 2, 8, 1, 9, 3};
        std::vector<uint32_t> indices(6);
        size_t count = filter::filter_i32(data, 6, filter::CompareOp::LT, 5, indices.data());
        EXPECT(count == 3);  // 2, 1, 3
    } END_TEST
    
    TEST("filter_i32 range (3 <= x < 7)") {
        alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<uint32_t> indices(10);
        size_t count = filter::filter_i32_range(data, 10, 3, 7, indices.data());
        EXPECT(count == 4);  // 3, 4, 5, 6
    } END_TEST
    
    TEST("filter_i32_values") {
        alignas(128) int32_t data[] = {10, 20, 30, 40, 50};
        std::vector<int32_t> values(5);
        size_t count = filter::filter_i32_values(data, 5, filter::CompareOp::GT, 25, values.data());
        EXPECT(count == 3);
        EXPECT(values[0] == 30 && values[1] == 40 && values[2] == 50);
    } END_TEST
    
    TEST("filter_f32 GT") {
        alignas(128) float data[] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        std::vector<uint32_t> indices(5);
        size_t count = filter::filter_f32(data, 5, filter::CompareOp::GT, 3.0f, indices.data());
        EXPECT(count == 3);  // 3.5, 4.5, 5.5
    } END_TEST
    
    TEST("count_i32") {
        alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        size_t count = filter::count_i32(data, 12, filter::CompareOp::GT, 6);
        EXPECT(count == 6);  // 7, 8, 9, 10, 11, 12
    } END_TEST
    
    TEST("filter_i32_and (compound)") {
        alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<uint32_t> indices(10);
        // 3 < x <= 7
        size_t count = filter::filter_i32_and(data, 10, 
            filter::CompareOp::GT, 3, filter::CompareOp::LE, 7, indices.data());
        EXPECT(count == 4);  // 4, 5, 6, 7
    } END_TEST
}

// ============================================================================
// Aggregation 测试
// ============================================================================

void test_aggregate() {
    std::cout << "\n\033[1m=== Aggregation Operator Tests ===\033[0m\n";
    
    TEST("sum_i32") {
        alignas(128) int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int64_t sum = aggregate::sum_i32(data, 10);
        EXPECT(sum == 55);
    } END_TEST
    
    TEST("sum_i64") {
        alignas(128) int64_t data[] = {1000000000LL, 2000000000LL, 3000000000LL};
        int64_t sum = aggregate::sum_i64(data, 3);
        EXPECT(sum == 6000000000LL);
    } END_TEST
    
    TEST("sum_f32") {
        alignas(128) float data[] = {1.1f, 2.2f, 3.3f, 4.4f};
        double sum = aggregate::sum_f32(data, 4);
        EXPECT(std::abs(sum - 11.0) < 0.01);
    } END_TEST
    
    TEST("sum_f64") {
        alignas(128) double data[] = {1.111, 2.222, 3.333, 4.444};
        double sum = aggregate::sum_f64(data, 4);
        EXPECT(std::abs(sum - 11.11) < 0.001);
    } END_TEST
    
    TEST("min_i32") {
        alignas(128) int32_t data[] = {5, 2, 8, 1, 9, 3, 7};
        int32_t min = aggregate::min_i32(data, 7);
        EXPECT(min == 1);
    } END_TEST
    
    TEST("max_i32") {
        alignas(128) int32_t data[] = {5, 2, 8, 1, 9, 3, 7};
        int32_t max = aggregate::max_i32(data, 7);
        EXPECT(max == 9);
    } END_TEST
    
    TEST("min_f32") {
        alignas(128) float data[] = {5.5f, 2.2f, 8.8f, 1.1f, 9.9f};
        float min = aggregate::min_f32(data, 5);
        EXPECT(std::abs(min - 1.1f) < 0.01f);
    } END_TEST
    
    TEST("max_f32") {
        alignas(128) float data[] = {5.5f, 2.2f, 8.8f, 1.1f, 9.9f};
        float max = aggregate::max_f32(data, 5);
        EXPECT(std::abs(max - 9.9f) < 0.01f);
    } END_TEST
    
    TEST("avg_i32") {
        alignas(128) int32_t data[] = {10, 20, 30, 40, 50};
        double avg = aggregate::avg_i32(data, 5);
        EXPECT(std::abs(avg - 30.0) < 0.001);
    } END_TEST
    
    TEST("group_sum_i32") {
        alignas(128) int32_t values[] = {10, 20, 30, 40, 50, 60};
        alignas(128) uint32_t groups[] = {0, 1, 0, 1, 0, 1};
        std::vector<int64_t> sums(2);
        aggregate::group_sum_i32(values, groups, 6, 2, sums.data());
        EXPECT(sums[0] == 90);   // 10 + 30 + 50
        EXPECT(sums[1] == 120);  // 20 + 40 + 60
    } END_TEST
    
    TEST("group_count") {
        alignas(128) uint32_t groups[] = {0, 1, 2, 0, 1, 0, 2, 2};
        std::vector<size_t> counts(3);
        aggregate::group_count(groups, 8, 3, counts.data());
        EXPECT(counts[0] == 3);
        EXPECT(counts[1] == 2);
        EXPECT(counts[2] == 3);
    } END_TEST
    
    TEST("group_min_i32") {
        alignas(128) int32_t values[] = {5, 3, 8, 2, 7, 1};
        alignas(128) uint32_t groups[] = {0, 1, 0, 1, 0, 1};
        std::vector<int32_t> mins(2);
        aggregate::group_min_i32(values, groups, 6, 2, mins.data());
        EXPECT(mins[0] == 5);  // min(5, 8, 7)
        EXPECT(mins[1] == 1);  // min(3, 2, 1)
    } END_TEST
    
    TEST("group_max_i32") {
        alignas(128) int32_t values[] = {5, 3, 8, 2, 7, 1};
        alignas(128) uint32_t groups[] = {0, 1, 0, 1, 0, 1};
        std::vector<int32_t> maxs(2);
        aggregate::group_max_i32(values, groups, 6, 2, maxs.data());
        EXPECT(maxs[0] == 8);  // max(5, 8, 7)
        EXPECT(maxs[1] == 3);  // max(3, 2, 1)
    } END_TEST
}

// ============================================================================
// Sort 测试
// ============================================================================

void test_sort() {
    std::cout << "\n\033[1m=== Sort Operator Tests ===\033[0m\n";
    
    TEST("sort_4_i32 ASC") {
        alignas(16) int32_t data[] = {4, 2, 3, 1};
        sort::sort_4_i32(data, sort::SortOrder::ASC);
        EXPECT(data[0] == 1 && data[1] == 2 && data[2] == 3 && data[3] == 4);
    } END_TEST
    
    TEST("sort_4_i32 DESC") {
        alignas(16) int32_t data[] = {1, 3, 2, 4};
        sort::sort_4_i32(data, sort::SortOrder::DESC);
        EXPECT(data[0] == 4 && data[1] == 3 && data[2] == 2 && data[3] == 1);
    } END_TEST
    
    TEST("sort_8_i32 ASC") {
        alignas(16) int32_t data[] = {8, 3, 5, 1, 7, 2, 6, 4};
        sort::sort_8_i32(data, sort::SortOrder::ASC);
        for (int i = 0; i < 7; ++i) EXPECT(data[i] <= data[i + 1]);
    } END_TEST
    
    TEST("sort_16_i32 ASC") {
        alignas(16) int32_t data[] = {16, 3, 9, 1, 14, 7, 11, 5, 13, 2, 10, 6, 15, 4, 12, 8};
        sort::sort_16_i32(data, sort::SortOrder::ASC);
        for (int i = 0; i < 15; ++i) EXPECT(data[i] <= data[i + 1]);
    } END_TEST
    
    TEST("sort_i32 large array") {
        std::vector<int32_t> data = {9, 3, 7, 1, 8, 2, 6, 4, 5, 10, 15, 12, 11, 14, 13};
        sort::sort_i32(data.data(), data.size(), sort::SortOrder::ASC);
        for (size_t i = 0; i < data.size() - 1; ++i) EXPECT(data[i] <= data[i + 1]);
    } END_TEST
    
    TEST("sort_i32 DESC") {
        std::vector<int32_t> data = {1, 5, 3, 9, 7, 2, 8, 4, 6, 10};
        sort::sort_i32(data.data(), data.size(), sort::SortOrder::DESC);
        for (size_t i = 0; i < data.size() - 1; ++i) EXPECT(data[i] >= data[i + 1]);
    } END_TEST
    
    TEST("argsort_i32") {
        int32_t data[] = {30, 10, 40, 20};
        std::vector<uint32_t> indices(4);
        sort::argsort_i32(data, 4, indices.data(), sort::SortOrder::ASC);
        EXPECT(indices[0] == 1);  // 10
        EXPECT(indices[1] == 3);  // 20
        EXPECT(indices[2] == 0);  // 30
        EXPECT(indices[3] == 2);  // 40
    } END_TEST
    
    TEST("topk_min_i32") {
        int32_t data[] = {50, 20, 80, 10, 30, 90, 40, 60, 70};
        std::vector<int32_t> values(3);
        sort::topk_min_i32(data, 9, 3, values.data(), nullptr);
        EXPECT(values[0] == 10);
        EXPECT(values[1] == 20);
        EXPECT(values[2] == 30);
    } END_TEST
    
    TEST("topk_max_i32") {
        int32_t data[] = {50, 20, 80, 10, 30, 90, 40, 60, 70};
        std::vector<int32_t> values(3);
        sort::topk_max_i32(data, 9, 3, values.data(), nullptr);
        EXPECT(values[0] == 90);
        EXPECT(values[1] == 80);
        EXPECT(values[2] == 70);
    } END_TEST
    
    TEST("merge_sorted_i32") {
        int32_t a[] = {1, 3, 5, 7};
        int32_t b[] = {2, 4, 6, 8};
        std::vector<int32_t> out(8);
        sort::merge_sorted_i32(a, 4, b, 4, out.data(), sort::SortOrder::ASC);
        for (int i = 0; i < 8; ++i) EXPECT(out[i] == i + 1);
    } END_TEST
}

// ============================================================================
// Join 测试
// ============================================================================

void test_join() {
    std::cout << "\n\033[1m=== Join Operator Tests ===\033[0m\n";
    
    TEST("hash_i32 uniqueness") {
        int32_t keys[] = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<uint32_t> hashes(8);
        join::hash_i32(keys, hashes.data(), 8);
        // 检查哈希值分布
        std::set<uint32_t> unique(hashes.begin(), hashes.end());
        EXPECT(unique.size() >= 4);  // 至少一半是唯一的
    } END_TEST
    
    TEST("HashTable build/probe basic") {
        int32_t build[] = {10, 20, 30, 20, 40};
        int32_t probe[] = {20, 30, 50};
        
        join::HashTable ht(5);
        ht.build_i32(build, 5);
        
        std::vector<uint32_t> build_idx(10), probe_idx(10);
        size_t matches = ht.probe_i32(probe, 3, build_idx.data(), probe_idx.data());
        EXPECT(matches == 3);  // 20 匹配 2 次, 30 匹配 1 次
    } END_TEST
    
    TEST("HashTable statistics") {
        int32_t keys[] = {1, 2, 3, 4, 5};
        join::HashTable ht(10);
        ht.build_i32(keys, 5);
        
        EXPECT(ht.size() == 5);
        EXPECT(ht.bucket_count() >= 8);  // 应该是 2 的幂
        EXPECT(ht.load_factor() < 1.0f);
    } END_TEST
    
    TEST("hash_join_i32 INNER") {
        int32_t build[] = {1, 2, 3, 4, 5};
        int32_t probe[] = {2, 4, 6, 8};
        
        join::JoinResult* result = join::create_join_result(100);
        size_t matches = join::hash_join_i32(build, 5, probe, 4, 
                                              join::JoinType::INNER, result);
        EXPECT(matches == 2);  // 2 和 4 匹配
        
        join::free_join_result(result);
    } END_TEST
    
    TEST("simd_find_matches_i32") {
        int32_t candidates[] = {10, 20, 30, 20, 40, 20, 50};
        std::vector<uint32_t> matches(7);
        size_t count = join::simd_find_matches_i32(candidates, 7, 20, matches.data());
        EXPECT(count == 3);
        EXPECT(matches[0] == 1 && matches[1] == 3 && matches[2] == 5);
    } END_TEST
    
    TEST("hash_join_i32 with duplicates") {
        int32_t build[] = {1, 1, 2, 2, 3};
        int32_t probe[] = {1, 2, 3, 4};
        
        join::JoinResult* result = join::create_join_result(100);
        size_t matches = join::hash_join_i32(build, 5, probe, 4,
                                              join::JoinType::INNER, result);
        EXPECT(matches == 5);  // 1 匹配 2 次, 2 匹配 2 次, 3 匹配 1 次
        
        join::free_join_result(result);
    } END_TEST
}

// ============================================================================
// 性能基准测试
// ============================================================================

void benchmark() {
    std::cout << "\n\033[1m=== Performance Benchmarks ===\033[0m\n";
    std::cout << std::fixed << std::setprecision(2);
    
    Timer timer;
    const int WARMUP = 3;
    const int ITERATIONS = 20;
    
    // Filter benchmark
    {
        const size_t N = 10000000;
        std::vector<int32_t> data(N);
        std::vector<uint32_t> indices(N);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000);
        for (size_t i = 0; i < N; ++i) data[i] = dist(rng);
        
        for (int i = 0; i < WARMUP; ++i)
            filter::filter_i32(data.data(), N, filter::CompareOp::GT, 500, indices.data());
        
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i)
            filter::filter_i32(data.data(), N, filter::CompareOp::GT, 500, indices.data());
        timer.stop();
        
        double ms = timer.ms() / ITERATIONS;
        double throughput = N / (ms / 1000.0) / 1e6;
        std::cout << "  Filter (10M i32):      " << std::setw(8) << ms << " ms, "
                  << std::setw(8) << throughput << " M elem/s\n";
    }
    
    // Aggregation benchmark
    {
        const size_t N = 100000000;
        std::vector<int32_t> data(N);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000);
        for (size_t i = 0; i < N; ++i) data[i] = dist(rng);
        
        for (int i = 0; i < WARMUP; ++i)
            aggregate::sum_i32(data.data(), N);
        
        timer.start();
        volatile int64_t result = 0;
        for (int i = 0; i < ITERATIONS; ++i)
            result = aggregate::sum_i32(data.data(), N);
        timer.stop();
        
        double ms = timer.ms() / ITERATIONS;
        double throughput = N / (ms / 1000.0) / 1e6;
        double bandwidth = (N * sizeof(int32_t)) / (ms / 1000.0) / 1e9;
        std::cout << "  Sum (100M i32):        " << std::setw(8) << ms << " ms, "
                  << std::setw(8) << throughput << " M elem/s, " 
                  << std::setw(6) << bandwidth << " GB/s\n";
    }
    
    // Sort benchmark
    {
        const size_t N = 1000000;
        std::vector<int32_t> data(N), data_copy(N);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 1000000);
        for (size_t i = 0; i < N; ++i) data[i] = dist(rng);
        
        for (int i = 0; i < WARMUP; ++i) {
            data_copy = data;
            sort::sort_i32(data_copy.data(), N, sort::SortOrder::ASC);
        }
        
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            data_copy = data;
            sort::sort_i32(data_copy.data(), N, sort::SortOrder::ASC);
        }
        timer.stop();
        
        double ms = timer.ms() / ITERATIONS;
        double throughput = N / (ms / 1000.0) / 1e6;
        std::cout << "  Sort (1M i32):         " << std::setw(8) << ms << " ms, "
                  << std::setw(8) << throughput << " M elem/s\n";
    }
    
    // Join benchmark
    {
        const size_t BUILD_N = 100000;
        const size_t PROBE_N = 1000000;
        
        std::vector<int32_t> build(BUILD_N), probe(PROBE_N);
        
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, BUILD_N * 2);
        for (size_t i = 0; i < BUILD_N; ++i) build[i] = dist(rng);
        for (size_t i = 0; i < PROBE_N; ++i) probe[i] = dist(rng);
        
        join::JoinResult* result = join::create_join_result(PROBE_N);
        
        for (int i = 0; i < WARMUP; ++i) {
            result->count = 0;
            join::hash_join_i32(build.data(), BUILD_N, probe.data(), PROBE_N,
                                join::JoinType::INNER, result);
        }
        
        timer.start();
        for (int i = 0; i < ITERATIONS; ++i) {
            result->count = 0;
            join::hash_join_i32(build.data(), BUILD_N, probe.data(), PROBE_N,
                                join::JoinType::INNER, result);
        }
        timer.stop();
        
        double ms = timer.ms() / ITERATIONS;
        double throughput = PROBE_N / (ms / 1000.0) / 1e6;
        std::cout << "  Hash Join (100K/1M):   " << std::setw(8) << ms << " ms, "
                  << std::setw(8) << throughput << " M probe/s\n";
        
        join::free_join_result(result);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         ThunderDuck - Complete Test Suite                    ║\n";
    std::cout << "║         Apple M4 Optimized DuckDB Operators                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // 初始化
    std::cout << "\n\033[1m=== System Information ===\033[0m\n";
    thunderduck::initialize();
    
    // 运行测试
    test_filter();
    test_aggregate();
    test_sort();
    test_join();
    
    // 运行基准测试
    benchmark();
    
    // 总结
    std::cout << "\n\033[1m=== Test Summary ===\033[0m\n";
    std::cout << "  Tests Passed: \033[32m" << tests_passed << "\033[0m\n";
    std::cout << "  Tests Failed: \033[" << (tests_failed ? "31" : "32") << "m" 
              << tests_failed << "\033[0m\n";
    
    if (tests_failed == 0) {
        std::cout << "\n\033[32m✓ All tests passed!\033[0m\n\n";
    } else {
        std::cout << "\n\033[31m✗ Some tests failed!\033[0m\n\n";
    }
    
    thunderduck::shutdown();
    return tests_failed > 0 ? 1 : 0;
}
