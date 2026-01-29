# V35 优化方案设计

> **版本**: V35 | **日期**: 2026-01-29 | **基线**: V34 (2.10x)

## 一、当前性能状态 (V34)

| 查询 | 加速比 | 状态 | V35 目标 |
|------|--------|------|----------|
| Q22 | 0.96x | ❌ 低于 DuckDB | >= 1.15x |
| Q8 | 1.06x | ⚠️ 接近持平 | >= 1.25x |
| Q3 | 1.12x | ⚠️ 微弱优势 | >= 1.40x |
| Q14 | 1.18x | ⚠️ 微弱优势 | >= 1.50x |
| Q17 | 1.00x | 回退 | 保持 |
| Q20 | 1.00x | 回退 | 保持 |
| Q21 | 1.00x | 回退 | >= 1.20x (可选) |

---

## 二、优化优先级

### P0 (高优先级) - 预期收益大

| 查询 | 优化策略 | 预期提升 | 工作量 |
|------|----------|----------|--------|
| **Q3** | Filter-JOIN-Aggregate 融合 | +25-34% → 1.40x | 中 |
| **Q14** | 并行两阶段聚合 + 直接数组 | +27-35% → 1.50x | 中 |
| **Q22** | SIMD SUBSTRING + AVG 融合 | +20-25% → 1.15x | 中 |

### P1 (中优先级)

| 查询 | 优化策略 | 预期提升 | 工作量 |
|------|----------|----------|--------|
| **Q8** | 缓存友好查找表 + SIMD CASE | +18-27% → 1.30x | 中 |

### P2 (低优先级 - 可选)

| 查询 | 优化策略 | 预期提升 | 工作量 |
|------|----------|----------|--------|
| **Q21** | SEMI + ANTI JOIN 优化 | 1.20-1.50x | 高 |

### Skip (不推荐)

| 查询 | 原因 |
|------|------|
| Q17 | 相关子查询，工作量过大 |
| Q20 | 多层嵌套，工作量过大 |

---

## 三、详细优化方案

### 3.1 Q3: 多表 JOIN (1.12x → 1.40x)

**当前瓶颈**:
- 三表 JOIN (customer-orders-lineitem) 数据拷贝开销
- GROUP BY 聚合使用 hash map

**优化策略**:
```cpp
// 1. Early Filter Push-down
// 在 JOIN 前先过滤 customer (BUILDING) 和 orders (date < 1995-03-15)
std::vector<bool> valid_customers(max_custkey + 1, false);
std::vector<uint32_t> valid_orders;

// 2. 融合 JOIN + Aggregate
// 直接在 lineitem 扫描时完成聚合，避免中间结果
struct Q3AggKey {
    int32_t orderkey;
    int32_t orderdate;
    int32_t shippriority;
};
CompactHashTable<Q3AggKey, int64_t> results;

// 3. 并行分区聚合
// 按 orderkey 范围分区，每个线程独立聚合
```

**预期收益**: +25-34%

---

### 3.2 Q14: 促销效果 (1.18x → 1.50x)

**当前瓶颈**:
- 条件聚合 (CASE WHEN) 分支开销
- SUM 累加的数据依赖

**优化策略**:
```cpp
// 1. 直接数组索引替代 hash 查找
// part 表按 p_partkey 直接索引
std::vector<bool> is_promo(max_partkey + 1, false);
for (size_t i = 0; i < part.count; ++i) {
    if (part.p_type[i].starts_with("PROMO")) {
        is_promo[part.p_partkey[i]] = true;
    }
}

// 2. 并行两阶段聚合
struct Q14ThreadLocal {
    int64_t promo_revenue = 0;
    int64_t total_revenue = 0;
};
std::vector<Q14ThreadLocal> thread_results(num_threads);

// 3. SIMD 批量计算 revenue
// l_extendedprice * (1 - l_discount) 向量化
```

**预期收益**: +27-35%

---

### 3.3 Q22: 全球销售机会 (0.96x → 1.15x)

**当前瓶颈**:
- SUBSTRING 字符串操作
- AVG 子查询重复扫描
- NOT EXISTS 检查

**优化策略**:
```cpp
// 1. SIMD 批量提取电话前缀
// 直接用 NEON 加载 2 字节并转换为数字
int8x16_t extract_country_codes_simd(const char** phones, size_t count);

// 2. AVG 子查询预计算融合
// 单遍扫描同时计算 AVG 和分类
struct Q22PreScan {
    int64_t sum_positive = 0;
    int64_t count_positive = 0;
    std::vector<uint32_t> candidate_indices;  // acctbal > 0 且 code 匹配
};

// 3. Bloom Filter 优化 NOT EXISTS
// 用 orders.o_custkey 构建 Bloom Filter
SingleHashBloomFilter order_customers;
order_customers.build(ord.o_custkey.data(), ord.count);

// 先 Bloom 过滤，再精确检查
for (uint32_t idx : candidates) {
    if (order_customers.possibly_contains(cust.c_custkey[idx])) {
        // 精确检查
    } else {
        // 确定不存在订单，直接计入结果
    }
}
```

**预期收益**: +20-25%

---

### 3.4 Q8: 国家市场份额 (1.06x → 1.30x)

**当前瓶颈**:
- 8 表 JOIN 查找开销
- CASE WHEN 条件分支

**优化策略**:
```cpp
// 1. 合并查找表减少访问次数
// 将 supplier → nation → region 合并为单一查找
struct SupplierInfo {
    int32_t nation_key;
    bool is_brazil;
};
std::vector<SupplierInfo> supp_info(max_suppkey + 1);

// 2. 预计算订单有效性
struct OrderValidity {
    int16_t year;      // 1995 or 1996
    bool is_america;   // customer in AMERICA region
};
std::vector<OrderValidity> order_valid(max_orderkey + 1);

// 3. SIMD CASE 评估
// 批量判断 nation == BRAZIL
uint8x16_t is_brazil_simd(const int32_t* nation_keys, size_t count, int32_t brazil_key);
```

**预期收益**: +18-27%

---

### 3.5 Q21: 供应商等待 (1.00x → 1.20x, 可选)

**当前瓶颈**:
- EXISTS + NOT EXISTS 组合
- 多层相关子查询

**优化策略**:
```cpp
// 1. SEMI JOIN 支持 EXISTS
// l_orderkey 存在于 late_lineitems 中
SemiJoin semi;
semi.build(late_lineitems);
auto exists_indices = semi.probe(all_lineitems);

// 2. ANTI JOIN 支持 NOT EXISTS
// 不存在其他供应商
AntiJoin anti;
anti.build(other_suppliers);
auto not_exists_indices = anti.probe(current_supplier_lineitems);

// 3. 组合条件
// EXISTS(l2) AND NOT EXISTS(l3)
```

**预期收益**: 20-50%
**风险**: 实现复杂度高，需要新增 SEMI/ANTI JOIN 算子

---

## 四、实现计划

### Phase 1: P0 查询优化

```
Week 1:
├── Q3 Filter-JOIN-Aggregate 融合
├── Q14 并行两阶段聚合
└── Q22 SIMD SUBSTRING + AVG 融合
```

### Phase 2: P1 查询优化

```
Week 2:
├── Q8 缓存友好查找表
└── Q8 SIMD CASE 评估
```

### Phase 3: P2 查询优化 (可选)

```
Week 3:
├── SEMI JOIN 算子
├── ANTI JOIN 算子
└── Q21 优化实现
```

---

## 五、预期成果

### V35 目标性能

| 查询 | V34 | V35 目标 | 提升 |
|------|-----|----------|------|
| Q3 | 1.12x | 1.40x | +25% |
| Q8 | 1.06x | 1.30x | +23% |
| Q14 | 1.18x | 1.50x | +27% |
| Q22 | 0.96x | 1.15x | +20% |
| Q21 | 1.00x | 1.20x | +20% (可选) |

### 几何平均加速比

- **V34**: 2.10x
- **V35 目标**: >= 2.25x (+7%)

---

## 六、风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| SIMD 优化不显著 | 中 | 中 | 先 profile，确认热点 |
| 并行开销抵消收益 | 低 | 中 | 设置最小数据量阈值 |
| Q21 复杂度过高 | 高 | 低 | 作为 P2 可选项 |

---

## 七、验证方法

1. **正确性**: 对比 DuckDB 查询结果
2. **性能**: 30 次迭代，IQR 中位数
3. **回归**: 确保其他查询不回退

---

*ThunderDuck V35 - 精准优化，目标 2.25x*
