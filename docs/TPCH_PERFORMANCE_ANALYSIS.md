# ThunderDuck TPC-H 全面性能分析报告

> 日期: 2026-01-28 | SF=1 | 平台: Apple M4 Max

## 一、执行摘要

### 版本对比总览

| 指标 | V24 | V25 | 变化 |
|------|-----|-----|------|
| 几何平均加速比 | 0.76x | **1.26x** | +66% |
| 更快查询数 | 1/5 | 7/10 | - |
| 更慢查询数 | 4/5 | 3/10 | - |

### V25 性能分布

```
 优秀 (>2x)  : ████████████████████ Q1 (9.15x)
 良好 (1.5-2x): ████████████ Q5 (1.90x), Q7 (1.94x), Q10 (1.67x)
 一般 (1-1.5x): ████████ Q6 (1.31x), Q9 (1.41x), Q14 (1.29x)
 回退 (<1x)  : ████ Q3 (0.45x), Q12 (0.79x), Q18 (0.22x) ← 需重点优化
```

## 二、详细性能数据

### 2.1 数据集统计 (SF=1)

| 表 | 行数 | 大小 | 说明 |
|----|------|------|------|
| LINEITEM | 6,001,215 | ~229 MB | 主要事实表 |
| ORDERS | 1,500,000 | ~57 MB | 订单表 |
| CUSTOMER | 150,000 | ~6 MB | 客户表 |
| PART | 200,000 | ~8 MB | 零件表 |
| PARTSUPP | 800,000 | ~30 MB | 零件供应表 |
| SUPPLIER | 10,000 | ~0.4 MB | 供应商表 |
| NATION | 25 | <1 KB | 国家表 |
| REGION | 5 | <1 KB | 区域表 |

### 2.2 查询级别详细分析

---

#### Q1: 定价汇总报告 ✅ 优秀

| 指标 | 值 |
|------|-----|
| **SQL 概要** | `GROUP BY l_returnflag, l_linestatus` + 6 列聚合 |
| **访问表** | lineitem (6M 行) |
| **关键操作** | Filter → GROUP BY → SUM/AVG/COUNT |
| **DuckDB** | 18.55 ms |
| **V24** | 2.01 ms (**9.21x**) |
| **V25** | 2.03 ms (**9.15x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 8 路展开 |
| Aggregate | V15 8T | 8 线程 | 直接数组聚合 (6 分组) |

**为什么快**: 低基数分组 (仅 6 个组) + 直接数组聚合避免 hash 开销

---

#### Q3: 运输优先级 ❌ 严重回退

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 3 表 JOIN + GROUP BY + Top 10 |
| **访问表** | customer (150K), orders (1.5M), lineitem (6M) |
| **关键操作** | Filter → JOIN → JOIN → GROUP BY → 排序 |
| **DuckDB** | 10.91 ms |
| **V24** | 36.26 ms (0.30x) |
| **V25** | 24.37 ms (**0.45x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 两阶段并行 |
| Join | WeakHashTable | CPU | KeyHashCache 预计算 |
| Aggregate | ThreadPool | 8 线程 | 并行 GROUP BY |

**瓶颈分析**:
1. **多次全表遍历**: lineitem 被扫描 2+ 次
2. **中间 vector 开销**: li_sel (~300 万), li_orderkeys (~300 万)
3. **多线程合并开销**: 8 个 unordered_map 合并

**优化方向**: Filter-Join-Aggregate 融合，单遍扫描

---

#### Q5: 本地供应商收入 ✅ 良好

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 6 表 JOIN + GROUP BY |
| **访问表** | customer, orders, lineitem, supplier, nation, region |
| **关键操作** | 多表 JOIN → GROUP BY |
| **DuckDB** | 17.24 ms |
| **V24** | 22.40 ms (0.77x) |
| **V25** | 9.08 ms (**1.90x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 日期范围过滤 |
| Join | WeakHashTable | CPU | 多表 KeyHashCache |
| Aggregate | ThreadPool | 8 线程 | 低基数分组 (5 nations) |

**为什么快**: 区域过滤后数据量大幅减少 + 低基数分组

---

#### Q6: 预测收入变化 ⚠️ 一般

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 简单 Filter + SUM |
| **访问表** | lineitem (6M 行) |
| **关键操作** | Filter (4 条件) → SUM |
| **DuckDB** | 2.60 ms |
| **V24** | 6.33 ms (0.41x) |
| **V25** | 1.97 ms (**1.31x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | SIMD 并行过滤 |
| Aggregate | parallel_sum_v25 | ThreadPool | 8 线程并行求和 |

**优化空间**: Filter + SUM 融合（当前分两步）

---

#### Q7: 体量运输 ✅ 良好

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 6 表 JOIN + 复合分组 |
| **访问表** | supplier, lineitem, orders, customer, nation×2 |
| **关键操作** | 多表 JOIN → 复合 GROUP BY (nation_pair × year) |
| **DuckDB** | 29.04 ms |
| **V24** | N/A |
| **V25** | 15.00 ms (**1.94x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 日期范围 + 国家过滤 |
| Join | 4×WeakHashTable | CPU | supp→nation, cust→nation, orders, lineitem |
| Aggregate | ThreadPool | 8 线程 | 复合 key 编码 (nation_pair × year) |

---

#### Q9: 产品类型利润 ⚠️ 一般

| 指标 | 值 |
|------|-----|
| **SQL 概要** | LIKE 过滤 + 6 表 JOIN + 利润计算 |
| **访问表** | part, supplier, lineitem, partsupp, orders, nation |
| **关键操作** | LIKE Filter → 多表 JOIN → 聚合 |
| **DuckDB** | 34.08 ms |
| **V24** | 114.54 ms (0.30x) |
| **V25** | 24.21 ms (**1.41x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD + LIKE | ARM Neon | LIKE '%green%' 预过滤 |
| Join | 3×WeakHashTable | CPU | green_parts, supp_to_nation, order_to_year |
| Aggregate | ThreadPool | 8 线程 | 3 路 KeyHashCache |

**V24→V25 提升原因**: WeakHashTable 比 V14 HashJoin 更适合多表 JOIN 场景

---

#### Q10: 退货报告 ✅ 良好

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 4 表 JOIN + GROUP BY + Top 20 |
| **访问表** | customer, orders, lineitem, nation |
| **关键操作** | Filter → JOIN → GROUP BY → 排序 |
| **DuckDB** | 19.29 ms |
| **V24** | N/A |
| **V25** | 11.59 ms (**1.67x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 日期范围 + returnflag='R' |
| Join | 2×WeakHashTable | CPU | order_cust, part_name |
| Aggregate | ThreadPool | 8 线程 | 并行客户收入聚合 |

---

#### Q12: 运输模式订单优先级 ❌ 回退

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 2 表 JOIN + CASE 聚合 |
| **访问表** | orders (1.5M), lineitem (6M) |
| **关键操作** | Filter (5 条件) → JOIN → CASE 聚合 |
| **DuckDB** | 18.69 ms |
| **V24** | N/A |
| **V25** | 23.66 ms (**0.79x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 5 条件组合过滤 |
| Join | WeakHashTable | CPU | order_priority |
| Aggregate | ThreadPool | 8 线程 | CASE 计数 |

**瓶颈分析**:
1. 过滤后数据量小，ThreadPool 调度开销 > 并行收益
2. 低基数分组 (2 个 shipmode)，不需要并行

**优化方向**: 自适应并行策略（小数据量用单线程）

---

#### Q14: 促销效果 ⚠️ 一般

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 2 表 JOIN + 条件聚合 |
| **访问表** | lineitem (6M), part (200K) |
| **关键操作** | Filter → JOIN → 条件 SUM |
| **DuckDB** | 7.88 ms |
| **V24** | N/A |
| **V25** | 6.08 ms (**1.29x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | V19 SIMD | ARM Neon | 日期范围过滤 |
| Join | WeakHashTable | CPU | part_promo 映射 |
| Aggregate | parallel_sum_v25 | ThreadPool | 条件 PROMO% 前缀匹配 |

---

#### Q18: 大批量客户 ❌ 严重回退

| 指标 | 值 |
|------|-----|
| **SQL 概要** | 子查询 GROUP BY HAVING + JOIN + Top 100 |
| **访问表** | customer, orders, lineitem (6M) |
| **关键操作** | GROUP BY → HAVING → JOIN → Top 100 |
| **DuckDB** | 28.50 ms |
| **V24** | N/A |
| **V25** | 128.94 ms (**0.22x**) |

**算子配置 (V25)**:
| 算子 | 版本 | 加速器 | 优化技术 |
|------|------|--------|----------|
| Filter | - | - | - |
| Join | WeakHasher | CPU 标量 | 自定义 hash 函数 |
| Aggregate | unordered_map | CPU 标量 | **高基数 GROUP BY fallback** |

**瓶颈分析**:
```
lineitem: 600 万行
GROUP BY l_orderkey: ~150 万个 unique key
当前实现: for (j=0; j<6M; j++) order_qty_map[orderkey[j]] += qty[j]
```

1. **unordered_map 逐行插入**: 600 万次 hash + 可能的内存分配
2. **WeakHashTable 无法原地更新**: 只能用于存在性检查，不能累加
3. **完全标量执行**: 无 SIMD 向量化

**DuckDB 优势**: 向量化 GROUP BY + 批量 hash + 预排序分组

**优化方向**:
1. MutableWeakHashTable (支持原地更新)
2. 向量化 GROUP BY SUM
3. Partition-based 聚合

## 三、版本演进对比

### 3.1 算子版本矩阵

| 查询 | Filter | Join | Aggregate | V24 状态 | V25 状态 |
|------|--------|------|-----------|----------|----------|
| Q1 | V19 SIMD | - | V15 8T | ✅ 9.21x | ✅ 9.15x |
| Q3 | V19 SIMD | V14→WeakHT | V15→ThreadPool | ❌ 0.30x | ❌ 0.45x |
| Q5 | V19 SIMD | V14→WeakHT | V15→ThreadPool | ❌ 0.77x | ✅ 1.90x |
| Q6 | V19 SIMD | - | V21→parallel | ❌ 0.41x | ✅ 1.31x |
| Q7 | V19 SIMD | 4×WeakHT | ThreadPool | N/A | ✅ 1.94x |
| Q9 | V19 SIMD | V14→3×WeakHT | V15→ThreadPool | ❌ 0.30x | ✅ 1.41x |
| Q10 | V19 SIMD | 2×WeakHT | ThreadPool | N/A | ✅ 1.67x |
| Q12 | V19 SIMD | WeakHT | ThreadPool | N/A | ❌ 0.79x |
| Q14 | V19 SIMD | WeakHT | parallel_sum | N/A | ✅ 1.29x |
| Q18 | - | WeakHasher | unordered_map | N/A | ❌ 0.22x |

### 3.2 V24 → V25 关键改进

| 改进 | 影响查询 | 效果 |
|------|----------|------|
| WeakHashTable 替换 V14 HashJoin | Q5, Q9 | 0.77x→1.90x, 0.30x→1.41x |
| ThreadPool 替换手动线程 | Q5, Q6, Q9 | 减少线程创建开销 |
| KeyHashCache 预计算 | 所有 JOIN | 避免重复 hash 计算 |
| 新增 Q7, Q10, Q12, Q14, Q18 | 5 个查询 | 扩大覆盖面 |

### 3.3 加速器使用统计

| 加速器 | 使用查询 | 效果 |
|--------|----------|------|
| ARM Neon SIMD | 全部 | Filter 2x+ |
| ThreadPool (8T) | Q3-Q14 | 聚合 1.5-2x |
| GPU Metal | 未使用 (V25) | - |
| 直接数组聚合 | Q1 | 9x (低基数) |

## 四、优化建议

### 4.1 P0: Q18 (0.22x → 目标 2x)

**问题**: 高基数 GROUP BY 使用标量 unordered_map

**方案**:
```cpp
// V26 新增: MutableWeakHashTable
template<typename V>
class MutableWeakHashTable {
    void add_or_insert(int32_t key, V delta);  // 支持原地更新
    void batch_add(const int32_t* keys, const V* deltas, size_t n);
};

// 或: 向量化 GROUP BY
class VectorizedGroupBySum {
    void batch_hash_keys(const int32_t* keys, size_t n, uint32_t* hashes);
    void aggregate_partitioned(const int32_t* keys, const int64_t* values, size_t n);
};
```

### 4.2 P1: Q3 (0.45x → 目标 1.5x)

**问题**: 多次全表遍历 + 中间 vector 开销

**方案**:
```cpp
// V26: Filter-Join-Aggregate 融合
void fused_filter_join_aggregate(
    const LineitemColumns& li,
    const WeakHashTable<OrderInfo>& orders,
    int32_t date_filter,
    AggregateResult& result  // 直接输出到结果
);
```

### 4.3 P2: Q12 (0.79x → 目标 1.2x)

**问题**: 小数据量并行调度开销

**方案**:
```cpp
// V26: 自适应并行策略
enum class ParallelStrategy { SERIAL, LIGHT_PARALLEL, FULL_PARALLEL };

ParallelStrategy choose_strategy(size_t data_size, size_t work_per_row) {
    if (data_size * work_per_row < 100000) return SERIAL;
    if (data_size * work_per_row < 1000000) return LIGHT_PARALLEL;
    return FULL_PARALLEL;
}
```

## 五、下一步计划

| 优先级 | 任务 | 影响查询 | 预期收益 |
|--------|------|----------|----------|
| P0 | MutableWeakHashTable | Q18 | 0.22x → 1.5x |
| P0 | 向量化 GROUP BY SUM | Q18 | 额外 1.5x |
| P1 | Filter-Join-Aggregate 融合 | Q3 | 0.45x → 1.5x |
| P2 | 自适应并行策略 | Q12, Q14 | 0.79x → 1.2x |
| P3 | GPU Metal 聚合 | Q1, Q6 | 额外 1.5x |

**V26 目标**:
- 消除所有回退 (所有查询 >= 1.0x)
- 几何平均加速比 >= 2.0x
- Category A 平均 >= 2.5x
