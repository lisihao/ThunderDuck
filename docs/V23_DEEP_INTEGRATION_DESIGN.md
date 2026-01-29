# ThunderDuck V23 深度集成架构设计

> 版本: 23.0 | 日期: 2026-01-28

## 一、问题分析

### 1.1 当前架构的缺陷

```
当前架构 (V22):
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   DuckDB    │───▶│  数据提取    │───▶│ ThunderDuck │
│   执行 SQL  │    │  (开销大)    │    │   算子      │
└─────────────┘    └──────────────┘    └─────────────┘
                         │
                   问题所在:
                   1. 完整物化所有列
                   2. 内存分配/拷贝开销
                   3. 手动改写查询计划
```

**核心问题：**
1. **手动改写 SQL** - 我们手动将 JOIN 改写为 SEMI JOIN + INNER JOIN，这不公平
2. **完整物化** - 提取所有数据到内存，即使只需要部分列
3. **多次遍历** - 数据被多次读取/写入

### 1.2 目标架构

```
目标架构 (V23):
┌─────────────────────────────────────────────────────┐
│                     DuckDB                          │
│  ┌─────────────┐    ┌─────────────────────────────┐│
│  │ SQL Parser  │───▶│      Query Optimizer        ││
│  └─────────────┘    │  (自动选择最优算子)         ││
│                     └──────────────┬──────────────┘│
│                                    │               │
│  ┌─────────────────────────────────▼──────────────┐│
│  │              Execution Engine                  ││
│  │  ┌─────────┐  ┌─────────┐  ┌────────────────┐ ││
│  │  │ DuckDB  │  │ DuckDB  │  │  ThunderDuck   │ ││
│  │  │ Filter  │  │  Join   │  │  替换算子      │ ││
│  │  └─────────┘  └─────────┘  │  (深度集成)    │ ││
│  │                            └────────────────┘ ││
│  └───────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## 二、深度集成策略

### 2.1 DuckDB 扩展接口

DuckDB 支持以下扩展点：

```cpp
// 1. Replacement Scan - 替换表扫描
void RegisterReplacementScan(DatabaseInstance &db, replacement_scan_t scan);

// 2. Custom Optimizer - 自定义优化器规则
void RegisterOptimizer(Optimizer &optimizer, OptimizerType type);

// 3. Custom Physical Operator - 自定义物理算子
class ThunderDuckPhysicalOperator : public PhysicalOperator {
    // 直接操作 DuckDB 的 DataChunk
};
```

### 2.2 晚期物化 (Late Materialization)

**原则：只在最后一刻才物化需要的列**

```cpp
// 早期物化 (当前方式，效率低):
vector<int64_t> prices = extract_column("l_extendedprice");  // 提取所有 6M 行
vector<int64_t> discounts = extract_column("l_discount");    // 再提取 6M 行
int64_t sum = compute_sum(prices, discounts);                // 计算

// 晚期物化 (目标方式，效率高):
// 只传递行索引，最后才读取实际值
vector<uint32_t> matching_indices = filter_step1();  // 只传递索引
matching_indices = filter_step2(matching_indices);   // 继续过滤
// 最后一步才物化需要的列
int64_t sum = sum_with_indices(prices_ptr, discounts_ptr, matching_indices);
```

### 2.3 集成层级

```
Level 1: DataChunk 直接操作 (最深度集成)
├── 直接操作 DuckDB 的 Vector/DataChunk
├── 零拷贝
└── 需要修改 DuckDB 源码

Level 2: Replacement Scan (中度集成)
├── 注册自定义表扫描
├── 可以控制数据流
└── 使用 DuckDB 扩展 API

Level 3: UDF/Table Function (轻度集成) ← 当前推荐
├── 注册自定义函数
├── 对特定操作进行加速
└── 最容易实现
```

## 三、实现方案

### 3.1 Phase 1: 公平基准测试

**原则：不手动改写 SQL，让优化器自己选择**

```cpp
// 修改后的基准测试
QueryResult run_query(const std::string& sql) {
    // DuckDB 原生执行
    auto duckdb_result = con.Query(sql);
    double duckdb_time = measure_time();

    // ThunderDuck 执行 (使用扩展)
    auto td_result = con_with_extension.Query(sql);  // 同样的 SQL!
    double td_time = measure_time();

    return {duckdb_time, td_time, verify(duckdb_result, td_result)};
}
```

### 3.2 Phase 2: 算子替换扩展

```cpp
// thunderduck_extension.cpp
class ThunderDuckExtension : public Extension {
public:
    void Load(DuckDB &db) override {
        // 注册替换算子
        RegisterPhysicalOperatorReplacement(db, "HASH_JOIN",
            [](PhysicalOperator* op) -> unique_ptr<PhysicalOperator> {
                // 检查是否适合使用 ThunderDuck 算子
                if (ShouldUseThunderDuck(op)) {
                    return make_unique<ThunderDuckHashJoin>(op);
                }
                return nullptr;  // 使用原生 DuckDB
            });

        // 注册聚合函数替换
        RegisterAggregateFunctionReplacement(db, "SUM",
            CreateThunderDuckSum());
    }
};
```

### 3.3 Phase 3: 晚期物化实现

```cpp
// 选择向量传递，避免物化
class ThunderDuckFilter : public PhysicalOperator {
    DataChunk GetData() override {
        // 获取输入
        DataChunk input = child->GetData();

        // 只计算选择向量，不物化数据
        SelectionVector sel;
        idx_t count = FilterWithSIMD(
            input.data[filter_col].GetData(),  // 直接指针，零拷贝
            input.size(),
            predicate,
            sel
        );

        // 将选择向量传递给下游
        DataChunk output;
        output.Slice(input, sel, count);  // 零拷贝切片
        return output;
    }
};
```

## 四、实现步骤

### Step 1: 创建 DuckDB 扩展框架
```
src/extension/
├── thunderduck_extension.cpp    # 扩展入口
├── operators/
│   ├── td_filter.cpp            # Filter 替换
│   ├── td_hash_join.cpp         # Hash Join 替换
│   └── td_aggregate.cpp         # Aggregate 替换
└── CMakeLists.txt
```

### Step 2: 实现 DataChunk 直接操作
- 直接操作 DuckDB 的 Vector 数据
- 使用 SelectionVector 传递过滤结果
- 避免数据拷贝

### Step 3: 实现算子替换逻辑
- 检测适合 ThunderDuck 的场景
- 自动选择最优实现
- 保证结果正确性

### Step 4: 修正基准测试
- 使用原生 TPC-H SQL
- 不手动改写查询
- 公平对比

## 五、关键技术点

### 5.1 零拷贝数据访问

```cpp
// DuckDB Vector 直接访问
void ProcessVector(Vector &vec, idx_t count) {
    // 获取原始数据指针，零拷贝
    auto data = FlatVector::GetData<int32_t>(vec);

    // 直接在原地使用 SIMD 处理
    simd_filter_gt_i32(data, count, threshold, selection);
}
```

### 5.2 选择向量传播

```cpp
// 使用 SelectionVector 避免物化
SelectionVector sel(STANDARD_VECTOR_SIZE);
idx_t result_count = 0;

// 过滤操作只更新 SelectionVector
for (idx_t i = 0; i < count; i++) {
    if (data[i] > threshold) {
        sel.set_index(result_count++, i);
    }
}

// 传递 SelectionVector 到下一个算子
output.Slice(input, sel, result_count);
```

### 5.3 算子选择决策

```cpp
bool ShouldUseThunderDuck(PhysicalOperator* op) {
    // 1. 数据量检查
    if (op->estimated_cardinality < 100000) return false;

    // 2. 数据类型检查
    if (!IsNumericType(op->types)) return false;

    // 3. 操作类型检查
    if (op->type == PhysicalOperatorType::FILTER) {
        return HasSimplePredicates(op);
    }

    return true;
}
```

## 六、预期收益

| 优化点 | 当前开销 | 优化后 | 预期提升 |
|--------|----------|--------|----------|
| 数据提取 | 全量物化 | 按需物化 | 2-5x |
| 内存分配 | 多次分配 | 复用缓冲区 | 1.5-2x |
| 数据拷贝 | 多次拷贝 | 零拷贝 | 2-3x |
| 算子选择 | 手动 | 自动优化 | - |

## 七、风险与挑战

1. **DuckDB 版本兼容性** - 内部 API 可能变化
2. **调试难度** - 深度集成后调试复杂
3. **正确性验证** - 需要严格测试

## 八、下一步行动

1. [ ] 研究 DuckDB Extension API
2. [ ] 实现最小可行扩展
3. [ ] 验证 DataChunk 直接操作
4. [ ] 实现 Filter 算子替换
5. [ ] 运行公平基准测试

---

*ThunderDuck V23 - 深度集成，零拷贝，晚期物化*
