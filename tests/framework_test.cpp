/**
 * ThunderDuck V23 Framework 测试
 *
 * 验证:
 * 1. 算子注册表工作正常
 * 2. 零拷贝视图正确
 * 3. 选择向量操作正确
 * 4. 算子选择逻辑正确
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>

// Framework
#include "framework/core/framework.hpp"

using namespace thunderduck::framework;
using namespace std;

// ============================================================================
// 测试辅助
// ============================================================================

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            cerr << "FAILED: " << msg << endl; \
            return false; \
        } \
    } while(0)

// ============================================================================
// VectorView 测试
// ============================================================================

bool test_vector_view() {
    cout << "Testing VectorView..." << endl;

    // 创建测试数据
    vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 基本视图
    VectorView<int32_t> view(data.data(), data.size());
    TEST_ASSERT(view.size() == 10, "size() should be 10");
    TEST_ASSERT(view[0] == 1, "first element should be 1");
    TEST_ASSERT(view[9] == 10, "last element should be 10");

    // 带选择向量的视图
    vector<uint32_t> selection = {0, 2, 4, 6, 8};  // 选择偶数索引
    view.SetSelection(selection.data(), selection.size());

    TEST_ASSERT(view.size() == 5, "size() with selection should be 5");
    TEST_ASSERT(view[0] == 1, "view[0] should be data[0] = 1");
    TEST_ASSERT(view[1] == 3, "view[1] should be data[2] = 3");
    TEST_ASSERT(view[4] == 9, "view[4] should be data[8] = 9");

    // 清除选择向量
    view.ClearSelection();
    TEST_ASSERT(view.size() == 10, "size() after clear should be 10");

    cout << "  VectorView: PASSED" << endl;
    return true;
}

// ============================================================================
// SelectionVector 测试
// ============================================================================

bool test_selection_vector() {
    cout << "Testing SelectionVector..." << endl;

    // 基本操作
    SelectionVector sv(100);
    sv.push_back(1);
    sv.push_back(3);
    sv.push_back(5);

    TEST_ASSERT(sv.size() == 3, "size should be 3");
    TEST_ASSERT(sv[0] == 1, "first element should be 1");
    TEST_ASSERT(sv[2] == 5, "third element should be 5");

    // 全选择向量
    SelectionVector all = SelectionVector::All(5);
    TEST_ASSERT(all.size() == 5, "All(5) should have size 5");
    TEST_ASSERT(all[0] == 0, "All(5)[0] should be 0");
    TEST_ASSERT(all[4] == 4, "All(5)[4] should be 4");

    // 排序检测
    TEST_ASSERT(sv.IsSorted(), "sv should be sorted");

    cout << "  SelectionVector: PASSED" << endl;
    return true;
}

// ============================================================================
// ColumnBatch 测试
// ============================================================================

bool test_column_batch() {
    cout << "Testing ColumnBatch..." << endl;

    // 创建测试数据
    vector<int32_t> prices = {100, 200, 300, 400, 500};
    vector<int64_t> quantities = {10, 20, 30, 40, 50};

    // 创建 ColumnBatch
    ColumnBatch batch;
    batch.AddColumn("price", prices.data(), prices.size());
    batch.AddColumn("quantity", quantities.data(), quantities.size());

    TEST_ASSERT(batch.GetRowCount() == 5, "row count should be 5");
    TEST_ASSERT(batch.GetColumnCount() == 2, "column count should be 2");
    TEST_ASSERT(batch.HasColumn("price"), "should have price column");
    TEST_ASSERT(batch.HasColumn("quantity"), "should have quantity column");
    TEST_ASSERT(!batch.HasColumn("invalid"), "should not have invalid column");

    // 获取列
    auto price_view = batch.GetColumn<int32_t>("price");
    TEST_ASSERT(price_view.size() == 5, "price view size should be 5");
    TEST_ASSERT(price_view[0] == 100, "first price should be 100");

    // 应用选择向量
    SelectionVector sel(3);
    sel.push_back(0);
    sel.push_back(2);
    sel.push_back(4);
    batch.ApplySelection(sel);

    TEST_ASSERT(batch.GetRowCount() == 3, "row count after selection should be 3");

    auto filtered_price = batch.GetColumn<int32_t>("price");
    TEST_ASSERT(filtered_price.size() == 3, "filtered price size should be 3");
    TEST_ASSERT(filtered_price[0] == 100, "filtered[0] should be 100");
    TEST_ASSERT(filtered_price[1] == 300, "filtered[1] should be 300");
    TEST_ASSERT(filtered_price[2] == 500, "filtered[2] should be 500");

    // 物化
    auto materialized = batch.Materialize<int32_t>("price");
    TEST_ASSERT(materialized.size() == 3, "materialized size should be 3");
    TEST_ASSERT(materialized[0] == 100, "materialized[0] should be 100");
    TEST_ASSERT(materialized[1] == 300, "materialized[1] should be 300");
    TEST_ASSERT(materialized[2] == 500, "materialized[2] should be 500");

    cout << "  ColumnBatch: PASSED" << endl;
    return true;
}

// ============================================================================
// OperatorRegistry 测试
// ============================================================================

bool test_operator_registry() {
    cout << "Testing OperatorRegistry..." << endl;

    auto& registry = OperatorRegistry::Instance();

    // 检查已注册的算子数量 (自动注册)
    size_t total = registry.GetTotalOperatorCount();
    cout << "  Registered operators: " << total << endl;
    TEST_ASSERT(total > 0, "should have registered operators");

    // 检查各类型算子
    TEST_ASSERT(registry.HasOperator(OperatorType::FILTER), "should have Filter operators");
    TEST_ASSERT(registry.HasOperator(OperatorType::AGGREGATE_SUM), "should have SUM operators");
    TEST_ASSERT(registry.HasOperator(OperatorType::GROUP_BY), "should have GROUP BY operators");
    TEST_ASSERT(registry.HasOperator(OperatorType::HASH_JOIN), "should have JOIN operators");

    // 测试 GetBestOperator
    OperatorContext ctx;
    ctx.SetRowCount(1000000)
       .SetDataType(DataType::INT32)
       .SetPredicateType(PredicateType::SIMPLE_COMPARISON);

    auto* filter_op = registry.GetBestOperator(OperatorType::FILTER, ctx);
    TEST_ASSERT(filter_op != nullptr, "should find a Filter operator");
    cout << "  Best Filter operator: " << filter_op->GetDescription() << endl;

    // 小数据量应该找不到合适的算子
    ctx.SetRowCount(100);
    auto* small_filter = registry.GetBestOperator(OperatorType::FILTER, ctx);
    TEST_ASSERT(small_filter == nullptr, "should not find operator for small data");

    cout << "  OperatorRegistry: PASSED" << endl;
    return true;
}

// ============================================================================
// Framework 初始化测试
// ============================================================================

bool test_framework_init() {
    cout << "Testing Framework initialization..." << endl;

    ThunderDuckFramework::Initialize();
    TEST_ASSERT(ThunderDuckFramework::IsInitialized(), "framework should be initialized");
    TEST_ASSERT(ThunderDuckFramework::IsEnabled(), "framework should be enabled");

    cout << "  Framework version: " << ThunderDuckFramework::GetVersion() << endl;
    cout << "  Min row threshold: " << ThunderDuckFramework::GetMinRowThreshold() << endl;

    // 修改配置
    ThunderDuckFramework::SetMinRowThreshold(50000);
    TEST_ASSERT(ThunderDuckFramework::GetMinRowThreshold() == 50000, "threshold should be 50000");

    // 恢复默认
    ThunderDuckFramework::SetMinRowThreshold(10000);

    cout << "  Framework: PASSED" << endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "    ThunderDuck V23 Framework Unit Tests" << endl;
    cout << "============================================================" << endl;

    int passed = 0;
    int failed = 0;

    if (test_vector_view()) passed++; else failed++;
    if (test_selection_vector()) passed++; else failed++;
    if (test_column_batch()) passed++; else failed++;
    if (test_operator_registry()) passed++; else failed++;
    if (test_framework_init()) passed++; else failed++;

    cout << "============================================================" << endl;
    cout << "    Results: " << passed << " passed, " << failed << " failed" << endl;
    cout << "============================================================" << endl;

    return failed > 0 ? 1 : 0;
}
