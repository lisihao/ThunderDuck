/**
 * ThunderDuck - Join Operator Tests
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/join.h"
#include "thunderduck/memory.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <random>
#include <set>

using namespace thunderduck;
using namespace thunderduck::join;

void test_hash_i32() {
    std::cout << "Testing hash_i32... ";
    
    int32_t keys[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint32_t> hashes(8);
    
    hash_i32(keys, hashes.data(), 8);
    
    // 确保哈希值不全相同
    std::set<uint32_t> unique_hashes(hashes.begin(), hashes.end());
    assert(unique_hashes.size() > 1);
    
    std::cout << "PASSED\n";
}

void test_hash_table_build_probe() {
    std::cout << "Testing HashTable build/probe... ";
    
    // 构建侧: [10, 20, 30, 20, 40]
    int32_t build_keys[] = {10, 20, 30, 20, 40};
    size_t build_count = 5;
    
    // 探测侧: [20, 30, 50]
    int32_t probe_keys[] = {20, 30, 50};
    size_t probe_count = 3;
    
    HashTable ht(build_count);
    ht.build_i32(build_keys, build_count);
    
    std::vector<uint32_t> build_indices(build_count * probe_count);
    std::vector<uint32_t> probe_indices(build_count * probe_count);
    
    size_t match_count = ht.probe_i32(probe_keys, probe_count, 
                                       build_indices.data(), probe_indices.data());
    
    // 20 匹配 2 次（build 索引 1 和 3）
    // 30 匹配 1 次（build 索引 2）
    // 50 不匹配
    assert(match_count == 3);
    
    std::cout << "PASSED (matches=" << match_count << ")\n";
}

void test_hash_join_inner() {
    std::cout << "Testing hash_join_i32 (INNER)... ";
    
    // 构建侧
    int32_t build_keys[] = {1, 2, 3, 4, 5};
    size_t build_count = 5;
    
    // 探测侧
    int32_t probe_keys[] = {2, 4, 6, 8};
    size_t probe_count = 4;
    
    JoinResult* result = create_join_result(1024);
    
    size_t match_count = hash_join_i32(build_keys, build_count,
                                        probe_keys, probe_count,
                                        JoinType::INNER, result);
    
    // 匹配: 2(build=1, probe=0), 4(build=3, probe=1)
    assert(match_count == 2);
    
    free_join_result(result);
    
    std::cout << "PASSED (matches=" << match_count << ")\n";
}

void test_simd_find_matches() {
    std::cout << "Testing simd_find_matches_i32... ";
    
    int32_t candidates[] = {10, 20, 30, 20, 40, 20, 50};
    std::vector<uint32_t> matches(7);
    
    size_t count = simd_find_matches_i32(candidates, 7, 20, matches.data());
    
    // 20 出现在索引 1, 3, 5
    assert(count == 3);
    assert(matches[0] == 1);
    assert(matches[1] == 3);
    assert(matches[2] == 5);
    
    std::cout << "PASSED\n";
}

void benchmark_hash_join() {
    std::cout << "Benchmarking hash_join_i32... ";
    
    const size_t BUILD_SIZE = 100000;
    const size_t PROBE_SIZE = 1000000;
    
    std::vector<int32_t> build_keys(BUILD_SIZE);
    std::vector<int32_t> probe_keys(PROBE_SIZE);
    
    // 生成数据
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(0, BUILD_SIZE * 2);
    
    for (size_t i = 0; i < BUILD_SIZE; ++i) {
        build_keys[i] = dist(rng);
    }
    for (size_t i = 0; i < PROBE_SIZE; ++i) {
        probe_keys[i] = dist(rng);
    }
    
    JoinResult* result = create_join_result(PROBE_SIZE);
    
    // 预热
    hash_join_i32(build_keys.data(), BUILD_SIZE,
                  probe_keys.data(), PROBE_SIZE,
                  JoinType::INNER, result);
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 10;
    
    for (int i = 0; i < iterations; ++i) {
        result->count = 0;
        hash_join_i32(build_keys.data(), BUILD_SIZE,
                      probe_keys.data(), PROBE_SIZE,
                      JoinType::INNER, result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ms_per_iter = duration.count() / 1000.0 / iterations;
    double probes_per_sec = PROBE_SIZE / (ms_per_iter / 1000.0);
    
    std::cout << "DONE\n";
    std::cout << "  Build size: " << BUILD_SIZE << ", Probe size: " << PROBE_SIZE << "\n";
    std::cout << "  Matches: " << result->count << "\n";
    std::cout << "  " << ms_per_iter << " ms per join\n";
    std::cout << "  " << (probes_per_sec / 1e6) << " M probes/sec\n";
    
    free_join_result(result);
}

int main() {
    std::cout << "=== ThunderDuck Join Tests ===\n\n";
    
    thunderduck::initialize();
    
    // 功能测试
    test_hash_i32();
    test_hash_table_build_probe();
    test_hash_join_inner();
    test_simd_find_matches();
    
    std::cout << "\n";
    
    // 性能测试
    benchmark_hash_join();
    
    std::cout << "\n=== All tests passed! ===\n";
    
    thunderduck::shutdown();
    return 0;
}
