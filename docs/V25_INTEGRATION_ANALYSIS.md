# ThunderDuck V25 算子整合与 DuckDB 集成分析

> 日期: 2026-01-28

## 一、V25 算子覆盖分析

### 1.1 当前覆盖情况

| 查询 | 实际版本 | 关键技术 | 差距分析 |
|------|----------|----------|----------|
| Q1 | V20.1 | 8线程 + 数组聚合 | 可升级到 V25 线程池 |
| **Q3** | **V25** | WeakHashTable + Hash缓存 + 线程池 | ✅ 最优 |
| **Q5** | **V25** | WeakHashTable + Hash缓存 + 线程池 | ✅ 最优 |
| **Q6** | **V25** | 线程池 + SIMD Filter | ✅ 最优 |
| Q7 | 基础 | std::unordered_map | 需升级到 V25 |
| **Q9** | **V25** | WeakHashTable + Hash缓存 + 线程池 | ✅ 最优 |
| Q10 | 基础 | std::unordered_map | 需升级到 V25 |
| Q12 | 基础 | std::unordered_map | 需升级到 V25 |
| Q14 | 基础 | std::unordered_map | 需升级到 V25 |
| Q18 | 基础 | unordered_map + 8路展开 | 需升级到 V25 |

### 1.2 版本技术矩阵

```
技术/查询         Q1  Q3  Q5  Q6  Q7  Q9  Q10 Q12 Q14 Q18
────────────────────────────────────────────────────────
线程池预热         -   ✓   ✓   ✓   -   ✓   -   -   -   -
WeakHashTable      -   ✓   ✓   -   -   ✓   -   -   -   -
Hash 缓存          -   ✓   ✓   -   -   ✓   -   -   -   -
SIMD Filter        ✓   ✓   ✓   ✓   -   ✓   -   -   -   -
数组聚合           ✓   -   -   -   -   -   -   -   -   -
8路展开            ✓   -   -   ✓   -   -   -   -   -   ✓
```

**V25 覆盖率: 4/10 (40%)**

### 1.3 未使用的 V24/V25 优化

```cpp
// V24 优化 - 未在所有查询中使用
class SelectionVector;     // 选择向量替代数据复制
class DirectArrayAgg<T>;   // 低基数分组优化

// V25 优化 - 仅 4 个查询使用
class ThreadPool;          // 线程池预热复用
class WeakHashTable<V>;    // 弱 Hash 表
class KeyHashCache;        // Hash 缓存
```

## 二、DuckDB 集成深度分析

### 2.1 当前集成方式

```
┌────────────────────────────────────────────────────────────────┐
│                      执行流程                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DuckDB 数据生成                                             │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ duckdb::DuckDB db(nullptr);                             ││
│     │ con.Query("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1);");││
│     └─────────────────────────────────────────────────────────┘│
│                         │                                       │
│                         ▼ 数据提取 (约 20-30% 开销)             │
│  2. 数据提取到 ThunderDuck 内存                                 │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ auto result = con.Query("SELECT * FROM lineitem");      ││
│     │ // 转换为 LineitemColumns                               ││
│     │ // 6M 行 × 16 列 ≈ 500MB 内存拷贝                       ││
│     └─────────────────────────────────────────────────────────┘│
│                         │                                       │
│                         ▼                                       │
│  3. ThunderDuck 独立执行                                        │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ // 完全独立实现，不调用 DuckDB 算子                     ││
│     │ ops_v25::inner_join_v25(...);                           ││
│     │ ops_v25::parallel_sum_v25(...);                         ││
│     └─────────────────────────────────────────────────────────┘│
│                         │                                       │
│                         ▼                                       │
│  4. 结果验证 (可选)                                             │
│     ┌─────────────────────────────────────────────────────────┐│
│     │ auto duckdb_result = con.Query(TPC_H_SQL[qid]);         ││
│     │ verify(thunderduck_result, duckdb_result);              ││
│     └─────────────────────────────────────────────────────────┘│
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 集成深度评分

| 维度 | 当前状态 | 评分 | 理想状态 |
|------|----------|------|----------|
| **数据访问** | 全量提取到内存 | 2/5 | 零拷贝 Arrow 共享 |
| **查询规划** | 硬编码手写 | 1/5 | DuckDB 优化器 + 算子替换 |
| **算子执行** | 完全独立 | 5/5 | 独立 (这是正确的) |
| **类型系统** | 手动转换 | 2/5 | 复用 DuckDB 类型 |
| **结果格式** | 自定义 | 2/5 | 兼容 DuckDB Result |

**总体集成深度: 中低 (2.4/5)**

### 2.3 集成问题

1. **数据拷贝开销**
   - 当前: 全量提取 → 列式转换 → 内存存储
   - 开销: 20-30% 的总执行时间
   - 改进: Arrow 零拷贝或 DuckDB Buffer 直接访问

2. **查询规划硬编码**
   - 当前: 每条 TPC-H 查询手写实现
   - 问题: 无法处理任意 SQL
   - 改进: DuckDB Extension 架构

3. **类型转换**
   - 当前: DuckDB DECIMAL → int64 定点数
   - 问题: 精度损失风险
   - 改进: 统一类型系统

## 三、改进建议

### 3.1 短期: V25 全面覆盖

**优先级 P0: 升级剩余 6 条查询到 V25**

```cpp
// 需要升级的查询
run_q1   → run_q1_v25   // 使用线程池替代手动线程
run_q7   → run_q7_v25   // 使用 WeakHashTable
run_q10  → run_q10_v25  // 使用 WeakHashTable + Hash 缓存
run_q12  → run_q12_v25  // 使用 WeakHashTable
run_q14  → run_q14_v25  // 使用 WeakHashTable
run_q18  → run_q18_v25  // 使用线程池 + WeakHashTable
```

**预期收益:**
- Q7, Q10, Q12, Q14: +30-50% (WeakHashTable vs unordered_map)
- Q1, Q18: +10-15% (线程池复用)

### 3.2 中期: 数据访问优化

**优先级 P1: 选择性列提取**

```cpp
// 当前: 提取全部 16 列
extract_all_tables();

// 改进: 按需提取
void extract_columns(const std::vector<std::string>& columns) {
    // 只提取 Q6 需要的 4 列
    // 减少 75% 内存带宽
}
```

**优先级 P2: Arrow 零拷贝**

```cpp
// 直接访问 DuckDB 内部 Arrow 格式
auto arrow_array = result->ToArrowTable();
// 无需拷贝，直接操作
```

### 3.3 长期: DuckDB Extension

**优先级 P3: 深度集成**

```cpp
// DuckDB Extension 架构
class ThunderDuckExtension : public Extension {
    void Load(DatabaseInstance& db) override {
        // 注册算子替换
        db.RegisterOperatorReplacement(
            PhysicalHashJoin::GetTypeId(),
            [](ClientContext& ctx, ...) {
                return make_unique<ThunderDuckHashJoin>(...);
            }
        );
    }
};
```

## 四、总结

### 4.1 V25 整合状态

- ✅ V25 包含了最新的优化技术 (线程池 + Hash 缓存 + 弱 Hash)
- ⚠️ 但只覆盖了 4/10 的 Category A 查询
- ❌ 6 条查询仍使用基础实现

### 4.2 DuckDB 集成状态

- ✅ 数据加载正常工作
- ✅ 结果验证可用
- ⚠️ 数据拷贝开销大 (20-30%)
- ❌ 无查询规划集成
- ❌ 无算子替换机制

### 4.3 下一步行动

| 优先级 | 任务 | 预期收益 |
|--------|------|----------|
| P0 | 将 Q7, Q10, Q12, Q14, Q18 升级到 V25 | +30-50% |
| P1 | 选择性列提取 | -20% 开销 |
| P2 | Arrow 零拷贝 | -15% 开销 |
| P3 | DuckDB Extension | 任意 SQL 支持 |

---

*ThunderDuck V25 算子整合与 DuckDB 集成分析 - 2026-01-28*
