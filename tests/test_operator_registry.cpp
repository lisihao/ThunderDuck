/**
 * ThunderDuck Operator Registry Test
 *
 * 验证算子注册和查找功能
 */

#include "thunderduck/operator_registry.h"
#include <iostream>
#include <cassert>

using namespace thunderduck;

void test_registration() {
    std::cout << "=== Test: Operator Registration ===" << std::endl;

    // 注册所有算子
    operators::register_all_operators();

    auto& reg = OperatorRegistry::instance();
    auto all_ops = reg.list_all();

    std::cout << "Total operators registered: " << all_ops.size() << std::endl;
    assert(all_ops.size() >= 10 && "Should have at least 10 operators");

    std::cout << "PASSED" << std::endl;
}

void test_find_by_name() {
    std::cout << "\n=== Test: Find by Name ===" << std::endl;

    auto& reg = OperatorRegistry::instance();

    // 查找 SIMD Filter
    const auto* simd_filter = reg.find("ThunderDuck::SIMDFilterInt32");
    assert(simd_filter != nullptr && "Should find SIMDFilterInt32");
    assert(simd_filter->type == OperatorType::FILTER);

    std::cout << "Found: " << simd_filter->name << std::endl;
    std::cout << "  Description: " << simd_filter->description << std::endl;
    std::cout << "  Per-row cost: " << simd_filter->cost.per_row_cost << " ns" << std::endl;

    // 查找不存在的算子
    const auto* not_found = reg.find("NonExistent");
    assert(not_found == nullptr && "Should not find non-existent operator");

    std::cout << "PASSED" << std::endl;
}

void test_find_by_type() {
    std::cout << "\n=== Test: Find by Type ===" << std::endl;

    auto& reg = OperatorRegistry::instance();

    // 查找所有 Filter 算子
    auto filters = reg.find_by_type(OperatorType::FILTER);
    std::cout << "Filter operators: " << filters.size() << std::endl;
    assert(filters.size() >= 2 && "Should have at least 2 filter operators");

    for (const auto* meta : filters) {
        std::cout << "  - " << meta->name << std::endl;
    }

    // 查找所有 Join 算子
    auto joins = reg.find_by_type(OperatorType::HASH_JOIN);
    std::cout << "Hash Join operators: " << joins.size() << std::endl;
    assert(joins.size() >= 2 && "Should have at least 2 hash join operators");

    std::cout << "PASSED" << std::endl;
}

void test_select_best() {
    std::cout << "\n=== Test: Select Best Operator ===" << std::endl;

    auto& reg = OperatorRegistry::instance();

    // 场景1: 大数据量 INT32 过滤
    const auto* best_filter = reg.select_best(
        OperatorType::FILTER,
        {DataType::INT32},
        1000000  // 1M rows
    );
    assert(best_filter != nullptr && "Should find best filter");
    std::cout << "Best filter for 1M INT32: " << best_filter->name << std::endl;

    // 场景2: 小数据量 - 应该返回 nullptr (不满足 min_rows)
    const auto* no_filter = reg.select_best(
        OperatorType::FILTER,
        {DataType::INT32},
        100  // 100 rows - too small
    );
    // 可能找到也可能找不到，取决于 min_rows_for_benefit
    std::cout << "Best filter for 100 rows: "
              << (no_filter ? no_filter->name : "None (too small)") << std::endl;

    // 场景3: 大数据量 Join
    const auto* best_join = reg.select_best(
        OperatorType::HASH_JOIN,
        {DataType::INT32},
        500000
    );
    assert(best_join != nullptr && "Should find best join");
    std::cout << "Best hash join for 500K: " << best_join->name << std::endl;

    std::cout << "PASSED" << std::endl;
}

void test_cost_model() {
    std::cout << "\n=== Test: Cost Model ===" << std::endl;

    auto& reg = OperatorRegistry::instance();

    const auto* simd_filter = reg.find("ThunderDuck::SIMDFilterInt32");
    assert(simd_filter != nullptr);

    // 估算不同数据量的成本
    for (size_t rows : {10000, 100000, 1000000, 10000000}) {
        double cost = simd_filter->cost.estimate(rows);
        std::cout << "  " << rows << " rows: " << cost << " ms" << std::endl;
    }

    // 验证成本随数据量增长
    double cost_10k = simd_filter->cost.estimate(10000);
    double cost_1m = simd_filter->cost.estimate(1000000);
    assert(cost_1m > cost_10k && "Cost should increase with rows");

    std::cout << "PASSED" << std::endl;
}

void test_print_registry() {
    std::cout << "\n=== Test: Print Registry ===" << std::endl;

    auto& reg = OperatorRegistry::instance();
    reg.print_registry();

    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " ThunderDuck Operator Registry Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_registration();
    test_find_by_name();
    test_find_by_type();
    test_select_best();
    test_cost_model();
    test_print_registry();

    std::cout << "\n========================================" << std::endl;
    std::cout << " All tests PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
