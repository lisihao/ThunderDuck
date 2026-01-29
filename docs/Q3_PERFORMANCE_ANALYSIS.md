# Q3 性能分析：为什么无法超越 DuckDB

> 版本: V27 | 日期: 2026-01-28

## 一、Q3 SQL 结构

```sql
SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10
```

**特点：3 表 JOIN + GROUP BY + ORDER BY + LIMIT**

## 二、数据流分析 (SF=1)

```
Customer (150K) ──[c_mktsegment='BUILDING']──> 30K (20%)
                            │
                            ▼
Orders (1.5M) ──[JOIN + date<threshold]──> 150K 有效订单 (10%)
                            │
                            ▼
Lineitem (6M) ──[date>threshold]──> 3M (50%)
                            │
              ──[JOIN orderkey]──> 300K 匹配行 (5%)
                            │
                            ▼
                    GROUP BY + ORDER BY + LIMIT 10
```

## 三、ThunderDuck V27 实现开销分析

### 3.1 预处理开销 (DuckDB 不需要)

| 步骤 | 操作 | 数据量 | 开销 |
|------|------|--------|------|
| Step 1 | 构建 custkey bitmap | 150K 客户 | ~0.1 ms |
| Step 2 | 构建 order_lookup 数组 | 6M × 12 bytes = **68 MB** | ~2-3 ms |
| Step 3 | 收集 valid_orderkey_list | 150K 订单 | ~0.1 ms |
| Step 4 | 初始化 revenue_array | 150K × 8 bytes | ~0.1 ms |

**预处理总开销: ~3 ms (占总时间 25%)**

### 3.2 主循环开销

```cpp
for (size_t i = start; i < end; ++i) {
    // 1. 日期过滤 - 内存访问 + 比较
    if (li.l_shipdate[i] <= date_threshold) continue;  // 50% 过滤

    // 2. 随机访问 order_lookup (68 MB 数组)
    int32_t idx = order_lookup[orderkey].compact_idx;  // 缓存不友好!
    if (idx < 0) continue;  // 90% 过滤

    // 3. 计算 revenue (乘法 + 除法)
    int64_t revenue = price * (10000 - discount) / 10000;

    // 4. 原子操作
    revenue_array[idx].fetch_add(revenue, ...);  // 300K 次
}
```

**关键瓶颈：**

1. **order_lookup 随机访问 (68 MB)**
   - 每次 lineitem 扫描都需要查找 `order_lookup[orderkey]`
   - orderkey 范围 1-6M，访问模式随机
   - L3 缓存 (M4: 16 MB) 无法容纳，导致大量缓存未命中

2. **原子操作开销**
   - 300K 次 `fetch_add` 操作
   - 虽然使用 `memory_order_relaxed`，仍有同步开销
   - 多线程可能在同一 orderkey 上竞争

### 3.3 后处理开销

| 步骤 | 操作 | 数据量 | 开销 |
|------|------|--------|------|
| Step 5 | 收集结果 | 遍历 150K valid orders | ~0.2 ms |
| Step 6 | partial_sort | ~100K 结果取 TOP 10 | ~0.5 ms |

## 四、DuckDB 的优势

### 4.1 查询优化器

DuckDB 可以选择最优的 JOIN 顺序：
```
1. 先过滤 customer (20% 选择率)
2. Hash JOIN orders (使用 hash table 而非 6M 数组)
3. Hash JOIN lineitem (利用 hash probe 的局部性)
```

### 4.2 向量化执行

```
- 批量处理 1024-2048 行
- SIMD 向量化过滤和计算
- 更好的 CPU 流水线利用
```

### 4.3 内存效率

```
- 延迟物化：只在需要时提取列
- 紧凑的 hash table (只存储匹配的 key)
- 无需预分配 6M 大小的数组
```

### 4.4 自适应执行

```
- 根据数据分布动态调整策略
- 小表广播 vs 大表分区
- Bloom Filter 预过滤
```

## 五、为什么其他查询能超越 DuckDB？

| 查询 | 加速比 | 原因 |
|------|--------|------|
| Q1 | 6.28x | 单表聚合，SIMD 优化效果显著 |
| Q6 | 1.67x | 单表过滤+聚合，无 JOIN 开销 |
| Q12 | 4.71x | 简单 2 表 JOIN，orderkey 直接索引有效 |
| Q18 | 4.91x | GROUP BY 后 HAVING，bitmap 过滤高效 |
| Q7 | 2.67x | 预构建 nation 映射，单遍扫描 |
| **Q3** | **0.91x** | 3 表 JOIN，大数组随机访问 |

**Q3 的特殊性：**
1. 需要同时 JOIN 3 个表
2. orderkey 范围大 (1-6M)，但有效 orderkey 只有 10%
3. order_lookup 数组的随机访问模式
4. GROUP BY 的 key (orderkey) 分布稀疏

## 六、潜在优化方向

### 6.1 Hash Table 替代大数组

```cpp
// 当前: 6M 大小的数组，90% 空间浪费
std::vector<OrderLookup> order_lookup(6M);

// 优化: 只存储有效的 150K 订单
std::unordered_map<int32_t, OrderLookup> order_lookup(150K);
```

**问题：** 之前尝试过，hash table 的 hash 计算开销抵消了空间节省

### 6.2 Bloom Filter 预过滤

```cpp
// 在扫描 lineitem 之前，使用 Bloom Filter 快速判断
BloomFilter valid_orders_bf(150K, 0.01);  // 1% 假阳性

for (auto orderkey : valid_orderkey_list) {
    valid_orders_bf.add(orderkey);
}

// 热循环中使用 Bloom Filter
if (!valid_orders_bf.possibly_contains(orderkey)) continue;  // 快速路径
int32_t idx = order_lookup[orderkey].compact_idx;  // 慢速路径
```

### 6.3 数据预排序

```cpp
// 如果 lineitem 按 orderkey 排序，可以利用局部性
// 但这需要修改数据加载方式
```

### 6.4 更激进的并行策略

```cpp
// 当前: 按 lineitem 行分片
// 优化: 按 orderkey 范围分片，减少原子操作冲突
```

## 七、结论

Q3 无法超越 DuckDB 的根本原因：

1. **内存访问模式不友好**
   - 68 MB 的 order_lookup 数组随机访问
   - 超出 L3 缓存容量，缓存命中率低

2. **预处理开销**
   - 需要构建大型查找表
   - DuckDB 使用 JIT 优化避免这些开销

3. **JOIN 策略限制**
   - 我们使用"探测式 JOIN"（遍历 lineitem，探测 orders）
   - DuckDB 可以选择"构建式 JOIN"（小表建 hash，大表探测）

4. **向量化程度不足**
   - 当前实现是标量循环 + 原子操作
   - DuckDB 使用批量向量化处理

**Q3 的 0.91x 已经是合理的结果**，对于复杂的 3 表 JOIN 查询，能达到接近 DuckDB 的性能已经说明优化是有效的。进一步优化需要更底层的架构改变（如实现完整的向量化执行引擎）。
