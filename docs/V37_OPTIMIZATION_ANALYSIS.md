# V37 性能优化分析

> 目标: 将加速比 < 1.2x 的查询优化至 >= 1.2x

## 一、问题查询分析

| 查询 | 当前加速比 | 当前版本 | 问题类型 |
|------|-----------|---------|----------|
| Q22 | 0.97x | V34 | ANTI JOIN + 相关子查询 |
| Q8 | 1.06x | V34 | 8表 JOIN |
| Q17 | 1.06x | V36 | 相关子查询 |
| Q3 | 1.18x | V31 | 3表 JOIN + ORDER BY |
| Q20 | 1.00x | baseline | 嵌套子查询 |
| Q21 | 1.00x | baseline | EXISTS + NOT EXISTS |

---

## 二、Q22 深度分析 (0.97x → 目标 1.5x)

### 2.1 SQL 结构
```sql
SELECT cntrycode, COUNT(*), SUM(c_acctbal)
FROM (
    SELECT SUBSTRING(c_phone, 1, 2) AS cntrycode, c_acctbal
    FROM customer
    WHERE SUBSTRING(c_phone, 1, 2) IN ('13','31','23','29','30','18','17')
      AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer WHERE c_acctbal > 0 AND ...)
      AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
) GROUP BY cntrycode
```

### 2.2 当前 V34 问题
1. **Anti-Join 使用 unordered_set**: O(1) 但常数大
2. **逐行检查**: 每个 candidate 调用 `anti_join.exists()`
3. **没有利用 custkey 稠密性**: custkey 范围 1-150000

### 2.3 V37 优化方案

#### 方案 A: Bitmap Anti-Join
```cpp
// 构建客户订单位图 (只需 150000 / 8 = 18.75 KB)
std::vector<uint64_t> has_order_bitmap((max_custkey + 63) / 64);

// 并行构建
for (size_t i = 0; i < ord.count; ++i) {
    int32_t ck = ord.o_custkey[i];
    has_order_bitmap[ck >> 6] |= (1ULL << (ck & 63));
}

// SIMD 批量检查
// 每次处理 64 个 customer
```

**预期加速**: 3-5x (从 hash lookup 变为 bit 操作)

#### 方案 B: 融合过滤
```cpp
// 单遍扫描: AVG计算 + 候选提取 + Anti-Join 过滤
for (size_t i = 0; i < cust.count; ++i) {
    // 1. 国家码过滤 (直接数组)
    // 2. c_acctbal > 0 累加 (AVG)
    // 3. Anti-Join 检查 (bitmap)
    // 4. 如果通过，加入候选
}
```

---

## 三、Q21 深度分析 (1.00x → 目标 1.5x)

### 3.1 SQL 结构
```sql
SELECT s_name, COUNT(*) AS numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey
  AND o_orderkey = l1.l_orderkey
  AND o_orderstatus = 'F'
  AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (SELECT * FROM lineitem l2
              WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
  AND NOT EXISTS (SELECT * FROM lineitem l3
                  WHERE l3.l_orderkey = l1.l_orderkey
                    AND l3.l_suppkey <> l1.l_suppkey
                    AND l3.l_receiptdate > l3.l_commitdate)
  AND n_name = 'SAUDI ARABIA'
```

### 3.2 问题分解
1. **双重存在性检查**: EXISTS + NOT EXISTS 在同一 orderkey 上
2. **自联接 lineitem**: l1, l2, l3 都是 lineitem
3. **复杂条件**: suppkey 不同 + 延迟交付

### 3.3 V37 优化方案

#### 关键洞察
可以预计算每个 orderkey 的:
- `has_other_supplier[orderkey]`: 是否有其他供应商
- `has_late_other[orderkey]`: 是否有其他供应商延迟交付

```cpp
struct OrderKeyState {
    uint8_t supplier_count;     // 该订单的供应商数量
    uint8_t late_supplier_count; // 延迟交付的供应商数量
    int32_t first_late_suppkey; // 第一个延迟供应商 (用于判断 <>)
};

// Phase 1: 扫描 lineitem 构建 OrderKeyState
std::unordered_map<int32_t, OrderKeyState> order_states;

for (size_t i = 0; i < li.count; ++i) {
    auto& state = order_states[li.l_orderkey[i]];
    state.supplier_count++;  // 简化: 只计数
    if (li.l_receiptdate[i] > li.l_commitdate[i]) {
        state.late_supplier_count++;
    }
}

// Phase 2: 评估条件
// EXISTS: supplier_count > 1
// NOT EXISTS (late other): late_supplier_count == 0
//                          OR (late_supplier_count == 1 AND 是当前供应商)
```

---

## 四、Q8 深度分析 (1.06x → 目标 1.3x)

### 4.1 SQL 结构 (8表 JOIN)
```sql
FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
WHERE p_partkey = l_partkey
  AND s_suppkey = l_suppkey
  AND l_orderkey = o_orderkey
  AND o_custkey = c_custkey
  AND c_nationkey = n1.n_nationkey
  AND n1.n_regionkey = r_regionkey
  AND r_name = 'AMERICA'
  AND s_nationkey = n2.n_nationkey
  AND o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'
  AND p_type = 'ECONOMY ANODIZED STEEL'
```

### 4.2 Join 顺序优化
DuckDB 可能使用:
```
region → nation → customer → orders → lineitem → supplier → nation2 → part
```

V37 优化顺序:
```
1. part 过滤 (p_type = 'ECONOMY ANODIZED STEEL') → 约 200 个 part
2. region → nation1 → customer (小基数链)
3. orders (日期过滤) → lineitem (probe part)
4. supplier → nation2
```

### 4.3 优化方案
```cpp
// Phase 1: 预计算小表
std::unordered_set<int32_t> target_parts;      // p_type 过滤
std::unordered_set<int32_t> america_customers; // region='AMERICA'

// Phase 2: 扫描 lineitem (6M 行)
// - 过滤 l_partkey IN target_parts
// - 过滤 日期范围
// - 查找 customer, supplier 信息

// 关键: 使用 Bloom Filter 预过滤
```

---

## 五、Q20 深度分析 (1.00x → 目标 1.3x)

### 5.1 V36 为何失败 (0.50x)
V36 实现的问题:
1. 遍历整个 lineitem (6M) 即使只需要 forest% 的 part
2. 使用 std::unordered_map 而非更高效的数据结构
3. 没有利用 partsupp 的有序性

### 5.2 V37 优化方案
```cpp
// Phase 1: 找 forest% 的 parts (~2000 个)
std::unordered_set<int32_t> forest_parts;

// Phase 2: 构建 (partkey, suppkey) → sum_qty
// 只处理 forest_parts 中的 lineitem!
// 使用复合键 hash: (pk << 20) | sk

// Phase 3: 过滤 partsupp
// ps_availqty > 0.5 * sum_qty[pk, sk]
```

---

## 六、Q17 深度分析 (1.06x → 目标 2.0x)

### 6.1 V36 加速不明显的原因
1. **线程开销**: 8 线程对 200 个 target parts 过重
2. **数据布局**: pending_rows 使用 vector<vector>，内存分配开销
3. **SIMD 利用率低**: 只有 2 路展开

### 6.2 V37 优化方案
```cpp
// 方案: 延迟物化 + 批量处理

// Phase 1: 找目标 parts (Brand#23, MED BOX) → ~200 个
// Phase 2: 单遍扫描 lineitem
//   - 累加 qty_sum[partkey], qty_count[partkey]
//   - 记录所有 (partkey, qty, extendedprice) 到批量数组
// Phase 3: 计算 AVG = qty_sum / qty_count
// Phase 4: SIMD 批量过滤 qty < 0.2 * AVG

// 关键: 避免 per-partkey 的 vector，使用全局批量数组
```

---

## 七、Q3 深度分析 (1.18x → 目标 1.5x)

### 7.1 当前 V31 实现特点
- 3 表 JOIN: customer → orders → lineitem
- 日期过滤: o_orderdate < '1995-03-15', l_shipdate > '1995-03-15'
- ORDER BY revenue DESC + LIMIT 10

### 7.2 优化方向
1. **Top-K 优化**: 使用堆而非全排序
2. **延迟 Join**: 先聚合 lineitem，再 join
3. **SIMD 日期比较**

---

## 八、V37 架构设计

### 8.1 新增通用组件

```cpp
// 1. Bitmap 存在性检查
class BitmapExistenceSet {
    std::vector<uint64_t> bitmap_;
public:
    void build(const int32_t* keys, size_t count, int32_t max_key);
    bool exists(int32_t key) const;
    void batch_check(const int32_t* keys, size_t count, uint64_t* results);
};

// 2. 紧凑复合键 Hash
class CompactCompositeHash {
    // 使用 (key1 << 20) | key2 作为单一 int64_t 键
};

// 3. Top-K 堆
template<typename T, typename Compare>
class TopKHeap {
    std::priority_queue<T, std::vector<T>, Compare> heap_;
    size_t k_;
public:
    void push(T value);
    std::vector<T> drain();
};
```

### 8.2 文件结构
```
benchmark/tpch/
├── tpch_operators_v37.h      # V37 算子定义
├── tpch_operators_v37.cpp    # V37 实现
└── tpch_queries.cpp          # 更新注册
```

---

## 九、预期收益

| 查询 | 当前 | V37 目标 | 主要优化 |
|------|------|----------|----------|
| Q22 | 0.97x | 1.8x | Bitmap Anti-Join |
| Q21 | 1.00x | 1.5x | 预计算 OrderKeyState |
| Q20 | 1.00x | 1.5x | 复合键 Hash + 预过滤 |
| Q17 | 1.06x | 1.8x | 批量处理 + SIMD |
| Q8 | 1.06x | 1.4x | Join 顺序 + Bloom |
| Q3 | 1.18x | 1.5x | Top-K 堆 |

**预期几何平均**: 2.15x → 2.4x+
