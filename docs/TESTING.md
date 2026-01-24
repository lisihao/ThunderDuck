# ThunderDuck 测试文档

> **版本**: 2.0 | **日期**: 2026-01-24
>
> 定义 ThunderDuck 项目的测试策略、用例和验证方法

---

## 一、测试策略

### 1.1 测试层次

```
┌─────────────────────────────────────────────────┐
│            端到端测试 (E2E Tests)                │
│         TPC-H Benchmark, 真实工作负载            │
├─────────────────────────────────────────────────┤
│            性能测试 (Performance Tests)          │
│      微基准测试, 与 DuckDB 对比测试              │
├─────────────────────────────────────────────────┤
│            集成测试 (Integration Tests)          │
│         算子组合测试, DuckDB 接口测试            │
├─────────────────────────────────────────────────┤
│            单元测试 (Unit Tests)                 │
│      各算子功能测试, 边界条件, 正确性验证        │
└─────────────────────────────────────────────────┘
```

### 1.2 测试原则

| 原则 | 描述 |
|------|------|
| **正确性优先** | 性能测试前必须通过正确性验证 |
| **与 DuckDB 对比** | 所有结果与 DuckDB 输出比对 |
| **边界覆盖** | 覆盖空输入、单元素、大数据等边界 |
| **可重复性** | 固定随机种子，确保结果可重现 |

---

## 二、单元测试

### 2.1 Filter 算子测试

#### TC-F001: 基本比较操作

| 测试ID | 操作 | 输入 | 预期结果 |
|--------|------|------|----------|
| TC-F001.1 | GT | [1,2,3,4,5], threshold=3 | count=2 (4,5) |
| TC-F001.2 | GE | [1,2,3,4,5], threshold=3 | count=3 (3,4,5) |
| TC-F001.3 | LT | [1,2,3,4,5], threshold=3 | count=2 (1,2) |
| TC-F001.4 | LE | [1,2,3,4,5], threshold=3 | count=3 (1,2,3) |
| TC-F001.5 | EQ | [1,2,3,4,5], threshold=3 | count=1 (3) |
| TC-F001.6 | NE | [1,2,3,4,5], threshold=3 | count=4 (1,2,4,5) |

#### TC-F002: 边界条件

| 测试ID | 场景 | 输入 | 预期结果 |
|--------|------|------|----------|
| TC-F002.1 | 空输入 | [], threshold=0 | count=0 |
| TC-F002.2 | 单元素匹配 | [5], threshold=3, GT | count=1 |
| TC-F002.3 | 单元素不匹配 | [1], threshold=3, GT | count=0 |
| TC-F002.4 | 全部匹配 | [4,5,6], threshold=3, GT | count=3 |
| TC-F002.5 | 全部不匹配 | [1,2,3], threshold=10, GT | count=0 |
| TC-F002.6 | INT32_MAX | [INT32_MAX], threshold=0, GT | count=1 |
| TC-F002.7 | INT32_MIN | [INT32_MIN], threshold=0, LT | count=1 |

#### TC-F003: 范围过滤

| 测试ID | 场景 | 输入 | 预期结果 |
|--------|------|------|----------|
| TC-F003.1 | 正常范围 | [1..10], low=3, high=7 | count=4 (3,4,5,6) |
| TC-F003.2 | 空范围 | [1..10], low=5, high=5 | count=0 |
| TC-F003.3 | 全范围 | [1..10], low=1, high=11 | count=10 |

#### TC-F004: SIMD 对齐测试

| 测试ID | 场景 | 输入大小 | 说明 |
|--------|------|----------|------|
| TC-F004.1 | 4 的倍数 | 16 | 完美 SIMD 对齐 |
| TC-F004.2 | 4 的倍数 + 1 | 17 | 1 个标量尾部 |
| TC-F004.3 | 4 的倍数 + 2 | 18 | 2 个标量尾部 |
| TC-F004.4 | 4 的倍数 + 3 | 19 | 3 个标量尾部 |
| TC-F004.5 | 16 的倍数 | 256 | 完美 16 元素批次 |
| TC-F004.6 | 非对齐大数组 | 1000003 | 大量尾部处理 |

### 2.2 Aggregation 算子测试

#### TC-A001: SUM 测试

| 测试ID | 场景 | 输入 | 预期结果 |
|--------|------|------|----------|
| TC-A001.1 | 正数求和 | [1,2,3,4,5] | 15 |
| TC-A001.2 | 负数求和 | [-1,-2,-3] | -6 |
| TC-A001.3 | 混合求和 | [-5,0,5] | 0 |
| TC-A001.4 | 空输入 | [] | 0 |
| TC-A001.5 | 溢出测试 | [INT32_MAX, 1] | 检查 int64 累加 |

#### TC-A002: MIN/MAX 测试

| 测试ID | 场景 | 输入 | 预期 MIN | 预期 MAX |
|--------|------|------|----------|----------|
| TC-A002.1 | 正常 | [3,1,4,1,5,9] | 1 | 9 |
| TC-A002.2 | 单元素 | [42] | 42 | 42 |
| TC-A002.3 | 全相同 | [5,5,5,5] | 5 | 5 |
| TC-A002.4 | 极值 | [INT32_MIN, INT32_MAX] | INT32_MIN | INT32_MAX |

#### TC-A003: AVG 测试

| 测试ID | 场景 | 输入 | 预期结果 |
|--------|------|------|----------|
| TC-A003.1 | 整数平均 | [2,4,6] | 4.0 |
| TC-A003.2 | 带小数 | [1,2] | 1.5 |
| TC-A003.3 | 单元素 | [10] | 10.0 |

#### TC-A004: COUNT(*) 测试

| 测试ID | 场景 | 输入大小 | 预期结果 |
|--------|------|----------|----------|
| TC-A004.1 | 空 | 0 | 0 |
| TC-A004.2 | 小 | 10 | 10 |
| TC-A004.3 | 大 | 1000000 | 1000000 |

### 2.3 Sort 算子测试

#### TC-S001: 基本排序

| 测试ID | 场景 | 输入 | 预期结果 (ASC) |
|--------|------|------|----------------|
| TC-S001.1 | 已排序 | [1,2,3,4,5] | [1,2,3,4,5] |
| TC-S001.2 | 逆序 | [5,4,3,2,1] | [1,2,3,4,5] |
| TC-S001.3 | 随机 | [3,1,4,1,5] | [1,1,3,4,5] |
| TC-S001.4 | 单元素 | [42] | [42] |
| TC-S001.5 | 两元素 | [2,1] | [1,2] |

#### TC-S002: 边界条件

| 测试ID | 场景 | 说明 |
|--------|------|------|
| TC-S002.1 | 空输入 | 不崩溃，返回空 |
| TC-S002.2 | 全相同 | 保持原样 |
| TC-S002.3 | 极值 | 包含 INT32_MIN, INT32_MAX |
| TC-S002.4 | 负数 | 正确处理符号 |

#### TC-S003: Radix Sort 阈值

| 测试ID | 大小 | 预期算法 |
|--------|------|----------|
| TC-S003.1 | 16 | std::sort |
| TC-S003.2 | 64 | std::sort |
| TC-S003.3 | 256 | Radix Sort |
| TC-S003.4 | 1000000 | Radix Sort |

### 2.4 TopK 算子测试

#### TC-T001: Top-K 最大值

| 测试ID | K | 输入 | 预期结果 (值) |
|--------|---|------|---------------|
| TC-T001.1 | 3 | [1,5,2,8,3,9] | [9,8,5] |
| TC-T001.2 | 1 | [3,1,4,1,5] | [5] |
| TC-T001.3 | 5 | [1,2,3] | [3,2,1] (K > n) |

#### TC-T002: K 值边界

| 测试ID | 场景 | K | 数据大小 |
|--------|------|---|----------|
| TC-T002.1 | K=1 | 1 | 1000 |
| TC-T002.2 | K=n | 1000 | 1000 |
| TC-T002.3 | K>n | 2000 | 1000 |
| TC-T002.4 | K=0 | 0 | 1000 |

### 2.5 Join 算子测试

#### TC-J001: Hash Join

| 测试ID | 场景 | Build | Probe | 预期匹配 |
|--------|------|-------|-------|----------|
| TC-J001.1 | 全匹配 | [1,2,3] | [1,2,3] | 3 |
| TC-J001.2 | 部分匹配 | [1,2,3] | [2,3,4] | 2 |
| TC-J001.3 | 无匹配 | [1,2,3] | [4,5,6] | 0 |
| TC-J001.4 | 重复键 | [1,1,2] | [1,2] | 3 |

---

## 三、性能测试

### 3.1 微基准测试配置

```cpp
struct BenchmarkConfig {
    size_t warmup_iterations = 3;
    size_t test_iterations = 10;
    bool report_min = true;
    bool report_avg = true;
    bool report_max = true;
    bool report_stddev = true;
};
```

### 3.2 测试数据集

| 数据集 | lineitem | orders | customer | 总大小 |
|--------|----------|--------|----------|--------|
| **Small** | 200K | 50K | 5K | ~8 MB |
| **Medium** | 2M | 500K | 50K | ~80 MB |
| **Large** | 5M | 1M | 100K | ~200 MB |

### 3.3 性能测试用例

#### PT-001: Filter 性能

| 测试ID | SQL | 数据集 | 目标 |
|--------|-----|--------|------|
| PT-001.1 | `WHERE qty > 25` | Large | < DuckDB |
| PT-001.2 | `WHERE qty = 30` | Large | < DuckDB |
| PT-001.3 | `WHERE qty BETWEEN 10 AND 40` | Large | < DuckDB |
| PT-001.4 | `WHERE price > 500` | Large | < DuckDB |

#### PT-002: Aggregation 性能

| 测试ID | SQL | 数据集 | 目标 |
|--------|-----|--------|------|
| PT-002.1 | `SUM(quantity)` | Large | 2x+ vs DuckDB |
| PT-002.2 | `MIN(qty), MAX(qty)` | Large | 3x+ vs DuckDB |
| PT-002.3 | `AVG(price)` | Large | 3x+ vs DuckDB |
| PT-002.4 | `COUNT(*)` | Large | 1000x+ vs DuckDB |

#### PT-003: Sort 性能

| 测试ID | SQL | 数据集 | 目标 |
|--------|-----|--------|------|
| PT-003.1 | `ORDER BY price ASC` | 1M rows | 4x+ vs DuckDB |
| PT-003.2 | `ORDER BY price DESC` | 1M rows | 4x+ vs DuckDB |

#### PT-004: TopK 性能

| 测试ID | SQL | K | 目标 |
|--------|-----|---|------|
| PT-004.1 | `LIMIT 10` | 10 | 2x+ vs DuckDB |
| PT-004.2 | `LIMIT 100` | 100 | 2x+ vs DuckDB |
| PT-004.3 | `LIMIT 1000` | 1000 | < DuckDB (v3.0 优化) |

#### PT-005: Join 性能

| 测试ID | SQL | 数据集 | 目标 |
|--------|-----|--------|------|
| PT-005.1 | `orders JOIN customer` | 1M × 100K | 改进中 |

---

## 四、正确性验证

### 4.1 与 DuckDB 结果对比

```cpp
void verify_correctness(const char* sql) {
    // 1. 执行 DuckDB 查询
    auto duckdb_result = duckdb_execute(sql);

    // 2. 执行 ThunderDuck 操作
    auto thunder_result = thunder_execute(sql);

    // 3. 比对结果
    ASSERT_EQ(duckdb_result.size(), thunder_result.size());
    for (size_t i = 0; i < duckdb_result.size(); ++i) {
        ASSERT_EQ(duckdb_result[i], thunder_result[i]);
    }
}
```

### 4.2 浮点精度验证

```cpp
constexpr double EPSILON = 1e-9;

void verify_float_result(double expected, double actual) {
    double diff = std::abs(expected - actual);
    double relative_error = diff / std::max(std::abs(expected), 1e-10);
    ASSERT_LT(relative_error, EPSILON);
}
```

---

## 五、回归测试

### 5.1 回归测试套件

| 套件 | 测试数 | 执行时间 | 触发条件 |
|------|--------|----------|----------|
| **Quick** | 50 | ~10s | 每次提交 |
| **Standard** | 200 | ~2min | PR 合并前 |
| **Full** | 500+ | ~10min | 发布前 |

### 5.2 性能回归检测

```cpp
struct PerformanceBaseline {
    std::string test_name;
    double baseline_ms;
    double tolerance_percent = 10.0;  // 允许 10% 波动
};

void check_performance_regression(const std::string& test,
                                    double current_ms,
                                    const PerformanceBaseline& baseline) {
    double change = (current_ms - baseline.baseline_ms) / baseline.baseline_ms * 100;
    if (change > baseline.tolerance_percent) {
        FAIL() << "Performance regression: " << test
               << " was " << baseline.baseline_ms << "ms"
               << " now " << current_ms << "ms"
               << " (" << change << "% slower)";
    }
}
```

---

## 六、测试执行

### 6.1 命令行

```bash
# 运行所有单元测试
./run_tests

# 运行特定测试
./run_tests --filter=Filter*

# 运行性能测试
./build/benchmark_app --medium

# 运行详细性能报告
./build/detailed_benchmark_app

# 生成测试覆盖率
make coverage
```

### 6.2 CI 集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-14  # M-series runner
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: make clean && make
      - name: Unit Tests
        run: ./run_tests
      - name: Quick Benchmark
        run: ./build/benchmark_app --small
```

---

## 七、测试报告

### 7.1 报告格式

```markdown
# ThunderDuck Test Report

## Summary
- Total: 200
- Passed: 198
- Failed: 2
- Skipped: 0

## Failed Tests
- TC-F004.6: SIMD alignment with 1000003 elements
  - Expected: 500123
  - Actual: 500122

## Performance Summary
| Test | DuckDB | ThunderDuck | Speedup |
|------|--------|-------------|---------|
| ...  | ...    | ...         | ...     |
```

### 7.2 性能趋势追踪

| 版本 | 总体胜率 | Filter | Sort | Agg | TopK | Join |
|------|---------|--------|------|-----|------|------|
| v1.0 | 50% | 0% | 50% | 50% | 100% | 0% |
| v2.0 | 71% | 50% | 100% | 100% | 66% | 0% |
| v3.0 | 目标 86% | 100% | 100% | 100% | 100% | 0% |

---

## 八、测试环境

### 8.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | Apple M1 | Apple M4 |
| RAM | 8 GB | 16 GB |
| SSD | 10 GB | 50 GB |

### 8.2 软件依赖

| 软件 | 版本 |
|------|------|
| macOS | 14.0+ |
| Xcode | 15.0+ |
| Clang | 17.0+ |
| DuckDB | 1.1.3 |

---

*ThunderDuck 测试文档 v2.0*
