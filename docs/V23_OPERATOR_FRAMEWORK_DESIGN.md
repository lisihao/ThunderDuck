# ThunderDuck V23 算子替换框架设计

> 版本: 23.0 | 日期: 2026-01-28

## 一、设计目标

1. **可扩展性** - 新算子可以通过简单的注册机制添加
2. **高性能** - 零拷贝数据访问，晚期物化
3. **灵活性** - 可根据条件选择是否使用 ThunderDuck 算子
4. **可维护性** - 清晰的分层架构

## 二、框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ThunderDuck Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Operator Registry (算子注册表)            │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │ │
│  │  │  Filter  │ │   SUM    │ │ GROUP BY │ │   JOIN   │  │ │
│  │  │ Operator │ │ Operator │ │ Operator │ │ Operator │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌───────────────────────────▼────────────────────────────┐ │
│  │            Zero-Copy Data Layer (零拷贝数据层)          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │ VectorView  │  │ SelectionVec│  │ ColumnBatch │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌───────────────────────────▼────────────────────────────┐ │
│  │              DuckDB Integration Layer                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │ │
│  │  │ Replacement │  │   Custom    │  │  Extension  │    │ │
│  │  │    Scan     │  │    UDF      │  │  Callback   │    │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 三、核心组件

### 3.1 算子基类 (BaseOperator)

```cpp
class BaseOperator {
public:
    virtual ~BaseOperator() = default;

    // 算子元信息
    virtual const char* GetName() const = 0;
    virtual OperatorType GetType() const = 0;

    // 能力检测 - 决定是否使用 ThunderDuck 实现
    virtual bool CanHandle(const OperatorContext& ctx) const = 0;

    // 性能预估 - 用于优化器决策
    virtual double EstimateCost(const OperatorContext& ctx) const = 0;
};
```

### 3.2 算子注册表 (OperatorRegistry)

```cpp
class OperatorRegistry {
public:
    // 单例访问
    static OperatorRegistry& Instance();

    // 注册算子
    template<typename T>
    void Register() {
        auto op = std::make_unique<T>();
        operators_[op->GetType()].push_back(std::move(op));
    }

    // 获取最佳算子
    BaseOperator* GetBestOperator(OperatorType type, const OperatorContext& ctx);

private:
    std::map<OperatorType, std::vector<std::unique_ptr<BaseOperator>>> operators_;
};

// 自动注册宏
#define REGISTER_OPERATOR(OpClass) \
    static bool _reg_##OpClass = []() { \
        OperatorRegistry::Instance().Register<OpClass>(); \
        return true; \
    }()
```

### 3.3 零拷贝数据视图 (VectorView)

```cpp
template<typename T>
class VectorView {
public:
    // 从 DuckDB Vector 创建 (零拷贝)
    static VectorView FromDuckDB(duckdb::Vector& vec, size_t count);

    // 从原始指针创建
    VectorView(const T* data, size_t count);

    // 数据访问
    const T* data() const { return data_; }
    size_t size() const { return count_; }
    const T& operator[](size_t i) const { return data_[i]; }

    // 选择向量支持
    void SetSelection(const uint32_t* sel, size_t sel_count);
    bool HasSelection() const { return selection_ != nullptr; }

private:
    const T* data_;
    size_t count_;
    const uint32_t* selection_ = nullptr;
    size_t sel_count_ = 0;
};
```

### 3.4 晚期物化批次 (ColumnBatch)

```cpp
class ColumnBatch {
public:
    // 添加列视图 (零拷贝)
    template<typename T>
    void AddColumn(const std::string& name, VectorView<T> view);

    // 获取列
    template<typename T>
    VectorView<T> GetColumn(const std::string& name) const;

    // 选择向量操作
    void ApplySelection(const uint32_t* sel, size_t count);
    void PropagateSelection();  // 向下游传播

    // 物化 (仅在需要时调用)
    template<typename T>
    std::vector<T> Materialize(const std::string& name) const;

private:
    std::map<std::string, std::any> columns_;
    std::vector<uint32_t> selection_;
    bool has_selection_ = false;
};
```

## 四、算子实现示例

### 4.1 Filter 算子

```cpp
class ThunderDuckFilter : public BaseOperator {
public:
    const char* GetName() const override { return "ThunderDuckFilter"; }
    OperatorType GetType() const override { return OperatorType::FILTER; }

    bool CanHandle(const OperatorContext& ctx) const override {
        // 条件: 数据量 > 100K 且是简单谓词
        return ctx.row_count > 100000 &&
               ctx.predicate_type == PredicateType::SIMPLE_COMPARISON;
    }

    double EstimateCost(const OperatorContext& ctx) const override {
        // 基于数据量和选择率估算成本
        return ctx.row_count * 0.1;  // 比 DuckDB 快约 2.5x
    }

    // 执行过滤
    size_t Execute(const VectorView<int32_t>& input,
                   CompareOp op, int32_t value,
                   uint32_t* out_indices) {
        return filter::filter_i32_v19(input.data(), input.size(),
                                       op, value, out_indices);
    }
};

REGISTER_OPERATOR(ThunderDuckFilter);
```

### 4.2 Aggregate 算子

```cpp
class ThunderDuckAggregate : public BaseOperator {
public:
    const char* GetName() const override { return "ThunderDuckAggregate"; }
    OperatorType GetType() const override { return OperatorType::AGGREGATE; }

    bool CanHandle(const OperatorContext& ctx) const override {
        return ctx.row_count > 100000 &&
               ctx.agg_type == AggType::SUM &&
               (ctx.data_type == DataType::INT32 ||
                ctx.data_type == DataType::INT64);
    }

    // SUM 执行
    int64_t ExecuteSum(const VectorView<int64_t>& input) {
        return aggregate::sum_i64_v21(input.data(), input.size());
    }

    // GROUP BY SUM 执行
    void ExecuteGroupSum(const VectorView<int32_t>& values,
                         const VectorView<uint32_t>& groups,
                         size_t num_groups,
                         int64_t* out_sums) {
        aggregate::group_sum_i32_v15(values.data(), groups.data(),
                                      values.size(), num_groups, out_sums);
    }
};

REGISTER_OPERATOR(ThunderDuckAggregate);
```

## 五、DuckDB 集成方式

### 5.1 方式 A: 自定义函数 (UDF)

```cpp
void RegisterThunderDuckFunctions(duckdb::Connection& conn) {
    // 注册 td_sum 函数
    conn.CreateScalarFunction<int64_t, int64_t>(
        "td_sum",
        [](int64_t* data, size_t count) {
            return aggregate::sum_i64_v21(data, count);
        }
    );
}
```

### 5.2 方式 B: 替换扫描 (Replacement Scan)

```cpp
unique_ptr<TableRef> ThunderDuckReplacementScan(
    ClientContext& context,
    ReplacementScanInput& input,
    optional_ptr<ReplacementScanData> data) {

    // 检查是否应该使用 ThunderDuck
    if (ShouldUseThunderDuck(input)) {
        return CreateThunderDuckTableRef(input);
    }
    return nullptr;  // 使用默认 DuckDB
}
```

### 5.3 方式 C: 执行钩子 (推荐)

```cpp
class ThunderDuckExecutionHook {
public:
    // 在算子执行前检查是否替换
    static bool OnBeforeExecute(PhysicalOperator* op,
                                 ExecutionContext& ctx) {
        auto* best_op = OperatorRegistry::Instance()
            .GetBestOperator(op->type, CreateContext(op));

        if (best_op && best_op->CanHandle(ctx)) {
            // 使用 ThunderDuck 执行
            return ExecuteWithThunderDuck(best_op, op, ctx);
        }
        return false;  // 使用 DuckDB 默认执行
    }
};
```

## 六、使用流程

### 6.1 初始化

```cpp
// 1. 创建 DuckDB 连接
duckdb::DuckDB db(nullptr);
duckdb::Connection conn(db);

// 2. 初始化 ThunderDuck 框架
ThunderDuckFramework::Initialize(conn);

// 3. 执行 SQL (自动使用优化算子)
auto result = conn.Query("SELECT SUM(l_quantity) FROM lineitem WHERE ...");
```

### 6.2 添加新算子

```cpp
// 1. 继承 BaseOperator
class MyCustomOperator : public BaseOperator {
    // 实现接口...
};

// 2. 注册
REGISTER_OPERATOR(MyCustomOperator);

// 完成! 框架会自动发现并使用新算子
```

## 七、性能保证

### 7.1 零拷贝保证

- VectorView 只存储指针，不复制数据
- 选择向量传播，避免物化中间结果
- 仅在最终输出时物化

### 7.2 自动选择保证

- CanHandle() 检查算子适用性
- EstimateCost() 估算执行成本
- 框架选择成本最低的实现

### 7.3 回退保证

- 如果 ThunderDuck 算子不适用，自动回退到 DuckDB
- 保证功能正确性

## 八、文件结构

```
src/framework/
├── core/
│   ├── base_operator.hpp       # 算子基类
│   ├── operator_registry.hpp   # 算子注册表
│   ├── operator_context.hpp    # 执行上下文
│   └── framework.hpp           # 框架入口
├── data/
│   ├── vector_view.hpp         # 零拷贝视图
│   ├── column_batch.hpp        # 列批次
│   └── selection_vector.hpp    # 选择向量
├── operators/
│   ├── filter_operator.hpp     # Filter 算子
│   ├── aggregate_operator.hpp  # Aggregate 算子
│   ├── join_operator.hpp       # Join 算子
│   └── sort_operator.hpp       # Sort 算子
└── integration/
    ├── duckdb_hooks.hpp        # DuckDB 钩子
    └── udf_registry.hpp        # UDF 注册
```

---

*ThunderDuck V23 Framework - 可扩展、高性能、深度集成*
