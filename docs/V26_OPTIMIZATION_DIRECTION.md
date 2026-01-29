# ThunderDuck V26 优化方向分析

> 基于 TPC-H SF=1 完整对比测试 | 日期: 2026-01-28

## 一、基准测试结果

### DuckDB vs ThunderDuck V25 对比

| Query | Category | DuckDB(ms) | V25(ms) | Speedup | 状态 |
|-------|----------|------------|---------|---------|------|
| Q1 | A | 26.08 | 2.04 | **12.81x** | ✅ 优秀 |
| Q10 | A | 28.42 | 12.19 | **2.33x** | ✅ 良好 |
| Q5 | A | 13.96 | 9.08 | 1.54x | ⚠️ 一般 |
| Q6 | A | 2.54 | 1.62 | 1.56x | ⚠️ 一般 |
| Q9 | A | 33.30 | 23.29 | 1.43x | ⚠️ 一般 |
| Q14 | A | 7.90 | 6.03 | 1.31x | ⚠️ 一般 |
| Q7 | A | 12.45 | 14.97 | **0.83x** | ❌ 回退 |
| Q12 | A | 18.91 | 23.98 | **0.79x** | ❌ 回退 |
| Q3 | A | 11.07 | 24.24 | **0.46x** | ❌ 严重回退 |
| Q18 | A | 29.26 | 120.98 | **0.24x** | ❌ 严重回退 |

**汇总**: 几何平均 1.26x，但 4 个查询性能低于 DuckDB

## 二、瓶颈分析

### P0: Q18 - 最严重 (0.24x, 比 DuckDB 慢 4x)

**问题诊断**:
```
lineitem 表: 600 万行
当前实现: for (j=0; j<6M; j++) order_qty_map[orderkey[j]] += qty[j]
```

1. **unordered_map 逐行插入**: 600 万次 hash + 内存分配
2. **WeakHashTable 无法原地更新**: fallback 到 unordered_map
3. **无向量化**: 完全标量执行

**DuckDB 优势**:
- 向量化 GROUP BY
- 批量 hash
- 预排序 + 分组

### P1: Q3 - 严重 (0.46x, 比 DuckDB 慢 2x)

**问题诊断**:
```cpp
// Step 3: 全表扫描构建中间 vector
for (i=0; i<6M; i++) if (l_shipdate[i] > date) li_sel.push_back(i);

// Step 4: 再次遍历
for (j=0; j<li_sel.size; j++) { ... }
```

1. **多次全表遍历**: lineitem 被扫描 2+ 次
2. **中间 vector 开销**: li_sel (~300万), li_orderkeys (~300万)
3. **多线程合并开销**: 8 个 unordered_map 合并

### P2: Q7/Q12 - 回退 (0.79x-0.83x)

**Q7 问题**:
- 复合 key 处理复杂
- 4 个 WeakHashTable 构建开销
- 多线程调度开销 > 并行收益

**Q12 问题**:
- lineitem 过滤后数据量小
- ThreadPool 调度开销成为主要开销

## 三、优化方向

### 方向 1: 向量化聚合器 (解决 Q18)

```cpp
// V26 新增: 向量化 GROUP BY SUM
class VectorizedGroupBySum {
    // 使用 SIMD 批量 hash
    void batch_hash_keys(const int32_t* keys, size_t n, uint32_t* hashes);

    // 直接数组聚合 (适合低基数)
    void aggregate_direct_array(const int32_t* keys, const int64_t* values, size_t n);

    // Partition-based 聚合 (适合高基数)
    void aggregate_partitioned(const int32_t* keys, const int64_t* values, size_t n);
};
```

**预期收益**: Q18 从 120ms → 30ms (4x 提升)

### 方向 2: 融合 Filter-Join-Aggregate (解决 Q3)

```cpp
// V26: 单遍扫描融合
void fused_filter_join_aggregate(
    const LineitemColumns& li,
    const WeakHashTable<OrderInfo>& orders,
    int32_t date_filter,
    AggregateResult& result  // 输出直接到结果
) {
    // 单遍扫描: Filter + Hash Lookup + Aggregate
    for (size_t i = 0; i < li.count; i += 8) {
        // SIMD 过滤
        uint32x4_t mask = vcgtq_s32(vld1q_s32(&li.l_shipdate[i]), date_vec);

        // 批量 hash + 查找
        // 直接聚合到结果
    }
}
```

**预期收益**: Q3 从 24ms → 10ms (2.4x 提升)

### 方向 3: 自适应并行策略 (解决 Q7/Q12)

```cpp
// V26: 根据数据量自动选择策略
enum class ParallelStrategy {
    SERIAL,           // < 100K 行: 单线程
    LIGHT_PARALLEL,   // 100K-1M 行: 2-4 线程
    FULL_PARALLEL     // > 1M 行: 8 线程
};

ParallelStrategy choose_strategy(size_t data_size, size_t work_per_row) {
    size_t total_work = data_size * work_per_row;
    if (total_work < 100000) return SERIAL;
    if (total_work < 1000000) return LIGHT_PARALLEL;
    return FULL_PARALLEL;
}
```

**预期收益**: Q7/Q12 从 0.8x → 1.2x

### 方向 4: WeakHashTable 增强

```cpp
// V26: 支持原地更新的 hash 表
template<typename V>
class MutableWeakHashTable {
    // 支持 += 操作
    void add_or_insert(int32_t key, V delta);

    // 批量更新
    void batch_add(const int32_t* keys, const V* deltas, size_t n);
};
```

**预期收益**: Q18 聚合阶段 3x 提升

## 四、优先级排序

| 优先级 | 方向 | 影响查询 | 预期收益 | 实现难度 |
|--------|------|----------|----------|----------|
| P0 | 向量化聚合器 | Q18, Q1 | 4x | 中 |
| P1 | Filter-Join 融合 | Q3, Q5, Q10 | 2x | 高 |
| P2 | MutableWeakHashTable | Q18, Q7 | 2x | 低 |
| P3 | 自适应并行策略 | Q7, Q12, Q14 | 1.5x | 低 |

## 五、V26 目标

1. **消除回退**: 所有 Category A 查询 >= 1.0x (不能比 DuckDB 慢)
2. **平均加速**: Category A 几何平均 >= 2.0x
3. **峰值保持**: Q1, Q10 保持 > 2x

## 六、实现计划

### Phase 1: 基础设施 (Day 1)
- [ ] MutableWeakHashTable 实现
- [ ] 自适应并行策略

### Phase 2: 核心优化 (Day 2-3)
- [ ] 向量化 GROUP BY SUM
- [ ] Filter-Join 融合框架

### Phase 3: 查询重写 (Day 4-5)
- [ ] Q18 重写
- [ ] Q3 重写
- [ ] Q7, Q12 优化

### Phase 4: 验证 (Day 6)
- [ ] 完整 TPC-H 回归测试
- [ ] 性能报告生成
