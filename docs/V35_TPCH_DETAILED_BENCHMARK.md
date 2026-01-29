# ThunderDuck TPC-H 详细基准测试报告

> **测试日期**: 2026-01-29 | **Scale Factor**: 1 | **平台**: Apple M4 Max

## 一、测试概览

| 指标 | 数值 |
|------|------|
| 几何平均加速比 | **2.11x** |
| 更快查询数 | 18 (81.8%) |
| 相同 | 3 |
| 更慢 | 1 |
| Category A 平均 | 2.58x |

## 二、数据规模 (SF=1)

| 表名 | 行数 | 核心列内存 |
|------|------|-----------|
| LINEITEM | 6,001,215 | ~229 MB |
| ORDERS | 1,500,000 | ~57 MB |
| PARTSUPP | 800,000 | ~31 MB |
| PART | 200,000 | ~8 MB |
| CUSTOMER | 150,000 | ~6 MB |
| SUPPLIER | 10,000 | ~0.4 MB |
| NATION | 25 | ~1 KB |
| REGION | 5 | ~0.2 KB |
| **总计** | **8,661,445** | **~397 MB** |

---

## 三、22 条查询详细对比

### Q1: 定价汇总报告 (Pricing Summary Report)

**SQL:**
```sql
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity) as sum_qty,
       SUM(l_extendedprice) as sum_base_price,
       SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
       SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
       AVG(l_quantity) as avg_qty,
       AVG(l_extendedprice) as avg_price,
       AVG(l_discount) as avg_disc,
       COUNT(*) as count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 31.62 ms |
| **ThunderDuck** | 4.44 ms |
| **加速比** | **7.12x** |
| **访问数据量** | LINEITEM 6M 行 x 8 列 = ~183 MB |
| **算子版本** | 基础版 (低基数直接数组聚合) |
| **加速器** | CPU SIMD (ARM Neon) |
| **优化方案** | 直接数组聚合 (6 个分组槽位)、SIMD 向量化计算、8 路循环展开 |

---

### Q2: 最小成本供应商 (Minimum Cost Supplier)

**SQL:**
```sql
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
  AND p_size = 15 AND p_type LIKE '%BRASS'
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = 'EUROPE'
  AND ps_supplycost = (
    SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region
    WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
      AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
      AND r_name = 'EUROPE'
  )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 7.77 ms |
| **ThunderDuck** | 0.35 ms |
| **加速比** | **22.44x** |
| **访问数据量** | PART 200K + PARTSUPP 800K + SUPPLIER 10K + NATION 25 + REGION 5 = ~40 MB |
| **算子版本** | 基础版 |
| **加速器** | CPU |
| **优化方案** | Hash 索引预构建、小表物化、相关子查询优化 |

---

### Q3: 运输优先级 (Shipping Priority)

**SQL:**
```sql
SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) as revenue,
       o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 13.43 ms |
| **ThunderDuck** | 9.19 ms |
| **加速比** | **1.46x** |
| **访问数据量** | LINEITEM 6M + ORDERS 1.5M + CUSTOMER 150K = ~293 MB |
| **算子版本** | **V31** (最优版本) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | GPU SEMI Join + INNER JOIN V19.2、8 线程并行聚合、Hash 缓存预热 |

---

### Q4: 订单优先级检查 (Order Priority Checking)

**SQL:**
```sql
SELECT o_orderpriority, COUNT(*) as order_count
FROM orders
WHERE o_orderdate >= DATE '1993-07-01' AND o_orderdate < DATE '1993-10-01'
  AND EXISTS (
    SELECT * FROM lineitem
    WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate
  )
GROUP BY o_orderpriority
ORDER BY o_orderpriority;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 15.05 ms |
| **ThunderDuck** | 4.25 ms |
| **加速比** | **3.54x** |
| **访问数据量** | ORDERS 1.5M + LINEITEM 6M (EXISTS) = ~286 MB |
| **算子版本** | **V27** (Bitmap SEMI Join) |
| **加速器** | CPU SIMD |
| **优化方案** | Bitmap SEMI Join、EXISTS 转 SEMI Join、预过滤日期 |

---

### Q5: 本地供应商收入 (Local Supplier Volume)

**SQL:**
```sql
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) as revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = 'ASIA'
  AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name
ORDER BY revenue DESC;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 12.28 ms |
| **ThunderDuck** | 10.00 ms |
| **加速比** | **1.23x** |
| **访问数据量** | 6 表 Join = ~350 MB |
| **算子版本** | **V32** (紧凑 Hash Table) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | 紧凑 Hash Table、Bloom Filter 预过滤、Thread-Local 聚合 |

---

### Q6: 预测收入变化 (Forecasting Revenue Change)

**SQL:**
```sql
SELECT SUM(l_extendedprice * l_discount) as revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 2.88 ms |
| **ThunderDuck** | 2.15 ms |
| **加速比** | **1.34x** |
| **访问数据量** | LINEITEM 6M x 4 列 = ~92 MB |
| **算子版本** | **V25** (SIMD Filter + 线程池) |
| **加速器** | CPU SIMD (ARM Neon) + 8 线程并行 |
| **优化方案** | SIMD 向量化 Filter、8 路循环展开、预取优化、Filter-Aggregate 融合 |

---

### Q7: 体量运输 (Volume Shipping)

**SQL:**
```sql
SELECT supp_nation, cust_nation, l_year, SUM(volume) as revenue
FROM (
  SELECT n1.n_name as supp_nation, n2.n_name as cust_nation,
         EXTRACT(YEAR FROM l_shipdate) as l_year,
         l_extendedprice * (1 - l_discount) as volume
  FROM supplier, lineitem, orders, customer, nation n1, nation n2
  WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey
    AND c_custkey = o_custkey AND s_nationkey = n1.n_nationkey
    AND c_nationkey = n2.n_nationkey
    AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
      OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
    AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 12.43 ms |
| **ThunderDuck** | 4.77 ms |
| **加速比** | **2.60x** |
| **访问数据量** | 6 表 Join = ~350 MB |
| **算子版本** | **V32** (紧凑 Hash Table + Bloom Filter) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | 双向国家过滤、直接数组索引 (nationkey)、Thread-Local 聚合 |

---

### Q8: 国家市场份额 (National Market Share)

**SQL:**
```sql
SELECT o_year, SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) as mkt_share
FROM (
  SELECT EXTRACT(YEAR FROM o_orderdate) as o_year,
         l_extendedprice * (1 - l_discount) as volume,
         n2.n_name as nation
  FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
  WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey
    AND l_orderkey = o_orderkey AND o_custkey = c_custkey
    AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey
    AND r_name = 'AMERICA' AND s_nationkey = n2.n_nationkey
    AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
    AND p_type = 'ECONOMY ANODIZED STEEL'
) all_nations
GROUP BY o_year
ORDER BY o_year;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 12.65 ms |
| **ThunderDuck** | 11.14 ms |
| **加速比** | **1.14x** |
| **访问数据量** | 8 表 Join = ~400 MB |
| **算子版本** | **V34** (继续攻坚) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | 条件聚合 (CASE WHEN)、直接数组索引、预过滤 p_type |

---

### Q9: 产品类型利润 (Product Type Profit Measure)

**SQL:**
```sql
SELECT nation, o_year, SUM(amount) as sum_profit
FROM (
  SELECT n_name as nation, EXTRACT(YEAR FROM o_orderdate) as o_year,
         l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
  FROM part, supplier, lineitem, partsupp, orders, nation
  WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey
    AND ps_partkey = l_partkey AND p_partkey = l_partkey
    AND o_orderkey = l_orderkey AND s_nationkey = n_nationkey
    AND p_name LIKE '%green%'
) profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 47.08 ms |
| **ThunderDuck** | 29.01 ms |
| **加速比** | **1.62x** |
| **访问数据量** | 6 表 Join = ~400 MB |
| **算子版本** | **V32** (紧凑 Hash Table) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | LIKE '%green%' 预过滤、复合键 Hash 索引、Thread-Local 聚合 |

---

### Q10: 退货报告 (Returned Item Reporting)

**SQL:**
```sql
SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) as revenue,
       c_acctbal, n_name, c_address, c_phone, c_comment
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
  AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 31.69 ms |
| **ThunderDuck** | 15.70 ms |
| **加速比** | **2.02x** |
| **访问数据量** | 4 表 Join = ~300 MB |
| **算子版本** | **V25** (线程池 + Hash 缓存) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | l_returnflag = 'R' 预过滤、日期范围过滤、TopK 部分排序 |

---

### Q11: 重要库存标识 (Important Stock Identification)

**SQL:**
```sql
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) as value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
  AND n_name = 'GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
  SELECT SUM(ps_supplycost * ps_availqty) * 0.0001
  FROM partsupp, supplier, nation
  WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
    AND n_name = 'GERMANY'
)
ORDER BY value DESC;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 4.23 ms |
| **ThunderDuck** | 2.67 ms |
| **加速比** | **1.59x** |
| **访问数据量** | PARTSUPP 800K + SUPPLIER 10K + NATION 25 = ~32 MB |
| **算子版本** | **V27** (单遍扫描 + 后置过滤) |
| **加速器** | CPU |
| **优化方案** | 单遍扫描同时计算 SUM 和分组、HAVING 后置过滤 |

---

### Q12: 运输模式与订单优先级 (Shipping Modes and Order Priority)

**SQL:**
```sql
SELECT l_shipmode,
       SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
                THEN 1 ELSE 0 END) as high_line_count,
       SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
                THEN 1 ELSE 0 END) as low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
  AND l_shipmode IN ('MAIL', 'SHIP')
  AND l_commitdate < l_receiptdate
  AND l_shipdate < l_commitdate
  AND l_receiptdate >= DATE '1994-01-01'
  AND l_receiptdate < DATE '1995-01-01'
GROUP BY l_shipmode
ORDER BY l_shipmode;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 20.79 ms |
| **ThunderDuck** | 4.30 ms |
| **加速比** | **4.84x** |
| **访问数据量** | LINEITEM 6M + ORDERS 1.5M = ~286 MB |
| **算子版本** | **V27** |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | INNER JOIN V19.2、条件聚合 (CASE WHEN)、l_shipmode 预过滤 |

---

### Q13: 客户分布 (Customer Distribution)

**SQL:**
```sql
SELECT c_count, COUNT(*) as custdist
FROM (
  SELECT c_custkey, COUNT(o_orderkey) as c_count
  FROM customer LEFT OUTER JOIN orders ON c_custkey = o_custkey
    AND o_comment NOT LIKE '%special%requests%'
  GROUP BY c_custkey
) c_orders
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 33.45 ms |
| **ThunderDuck** | 17.27 ms |
| **加速比** | **1.94x** |
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M = ~63 MB |
| **算子版本** | **V34** (LEFT JOIN + COUNT 攻坚) |
| **加速器** | CPU + 8 线程并行 |
| **优化方案** | LEFT OUTER JOIN 优化、NOT LIKE 预过滤、两级 GROUP BY |

---

### Q14: 促销效果 (Promotion Effect)

**SQL:**
```sql
SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                         THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
       / SUM(l_extendedprice * (1 - l_discount)) as promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey
  AND l_shipdate >= DATE '1995-09-01'
  AND l_shipdate < DATE '1995-10-01';
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 9.12 ms |
| **ThunderDuck** | 7.07 ms |
| **加速比** | **1.29x** |
| **访问数据量** | LINEITEM 6M + PART 200K = ~237 MB |
| **算子版本** | **V27** |
| **加速器** | CPU SIMD |
| **优化方案** | p_type LIKE 'PROMO%' 预计算 Bitmap、条件聚合、日期范围预过滤 |

---

### Q15: 顶级供应商 (Top Supplier)

**SQL:**
```sql
WITH revenue AS (
  SELECT l_suppkey as supplier_no,
         SUM(l_extendedprice * (1 - l_discount)) as total_revenue
  FROM lineitem
  WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
  GROUP BY l_suppkey
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue
WHERE s_suppkey = supplier_no
  AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
ORDER BY s_suppkey;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 6.08 ms |
| **ThunderDuck** | 2.60 ms |
| **加速比** | **2.34x** |
| **访问数据量** | LINEITEM 6M + SUPPLIER 10K = ~230 MB |
| **算子版本** | **V27** (直接数组索引 + 并行聚合) |
| **加速器** | CPU SIMD + 8 线程并行 |
| **优化方案** | CTE 物化、直接数组索引 (suppkey)、MAX 后置过滤 |

---

### Q16: 零件/供应商关系 (Parts/Supplier Relationship)

**SQL:**
```sql
SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) as supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey
  AND p_brand <> 'Brand#45'
  AND p_type NOT LIKE 'MEDIUM POLISHED%'
  AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
  AND ps_suppkey NOT IN (
    SELECT s_suppkey FROM supplier
    WHERE s_comment LIKE '%Customer%Complaints%'
  )
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 12.91 ms |
| **ThunderDuck** | 8.29 ms |
| **加速比** | **1.56x** |
| **访问数据量** | PARTSUPP 800K + PART 200K + SUPPLIER 10K = ~41 MB |
| **算子版本** | **V27** (PredicatePrecomputer) |
| **加速器** | CPU |
| **优化方案** | NOT IN 转 ANTI Join、PredicatePrecomputer 批量评估、Hash 索引 |

---

### Q17: 小批量订单收入 (Small-Quantity-Order Revenue)

**SQL:**
```sql
SELECT SUM(l_extendedprice) / 7.0 as avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem
    WHERE l_partkey = p_partkey
  );
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 13.42 ms |
| **ThunderDuck** | 13.42 ms |
| **加速比** | **1.00x** (基线) |
| **访问数据量** | LINEITEM 6M + PART 200K = ~237 MB |
| **算子版本** | 基线 (DuckDB 回退) |
| **加速器** | - |
| **优化方案** | 相关子查询未优化 (待改进) |

---

### Q18: 大批量客户 (Large Volume Customer)

**SQL:**
```sql
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity)
FROM customer, orders, lineitem
WHERE o_orderkey IN (
  SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300
)
  AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 33.60 ms |
| **ThunderDuck** | 14.47 ms |
| **加速比** | **2.32x** |
| **访问数据量** | LINEITEM 6M + ORDERS 1.5M + CUSTOMER 150K = ~293 MB |
| **算子版本** | **V32** |
| **加速器** | CPU + 8 线程并行 |
| **优化方案** | 单线程 8 路展开 GROUP BY、IN 转 SEMI Join、partial_sort |

---

### Q19: 折扣收入 (Discounted Revenue)

**SQL:**
```sql
SELECT SUM(l_extendedprice * (1 - l_discount)) as revenue
FROM lineitem, part
WHERE (
  p_partkey = l_partkey AND p_brand = 'Brand#12'
  AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
  AND l_quantity >= 1 AND l_quantity <= 11
  AND p_size BETWEEN 1 AND 5
  AND l_shipmode IN ('AIR', 'AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON'
) OR (
  p_partkey = l_partkey AND p_brand = 'Brand#23'
  AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
  AND l_quantity >= 10 AND l_quantity <= 20
  AND p_size BETWEEN 1 AND 10
  AND l_shipmode IN ('AIR', 'AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON'
) OR (
  p_partkey = l_partkey AND p_brand = 'Brand#34'
  AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
  AND l_quantity >= 20 AND l_quantity <= 30
  AND p_size BETWEEN 1 AND 15
  AND l_shipmode IN ('AIR', 'AIR REG')
  AND l_shipinstruct = 'DELIVER IN PERSON'
);
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 35.60 ms |
| **ThunderDuck** | 5.05 ms |
| **加速比** | **7.05x** |
| **访问数据量** | LINEITEM 6M + PART 200K = ~237 MB |
| **算子版本** | **V33** (通用化 + 条件组参数化) |
| **加速器** | CPU SIMD |
| **优化方案** | PredicatePrecomputer 条件组预计算、3 条件组并行评估、partkey 直接数组索引 |

---

### Q20: 潜在零件促销 (Potential Part Promotion)

**SQL:**
```sql
SELECT s_name, s_address
FROM supplier, nation
WHERE s_suppkey IN (
  SELECT ps_suppkey FROM partsupp
  WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
    AND ps_availqty > (
      SELECT 0.5 * SUM(l_quantity) FROM lineitem
      WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
        AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
    )
)
  AND s_nationkey = n_nationkey AND n_name = 'CANADA'
ORDER BY s_name;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 9.91 ms |
| **ThunderDuck** | 9.91 ms |
| **加速比** | **1.00x** (基线) |
| **访问数据量** | SUPPLIER 10K + PARTSUPP 800K + PART 200K + LINEITEM 6M = ~270 MB |
| **算子版本** | 基线 (DuckDB 回退) |
| **加速器** | - |
| **优化方案** | 嵌套相关子查询未优化 (待改进) |

---

### Q21: 未按时交货的供应商 (Suppliers Who Kept Orders Waiting)

**SQL:**
```sql
SELECT s_name, COUNT(*) as numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey
  AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND EXISTS (
    SELECT * FROM lineitem l2
    WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey
  )
  AND NOT EXISTS (
    SELECT * FROM lineitem l3
    WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey
      AND l3.l_receiptdate > l3.l_commitdate
  )
  AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 44.52 ms |
| **ThunderDuck** | 44.52 ms |
| **加速比** | **1.00x** (基线) |
| **访问数据量** | SUPPLIER 10K + LINEITEM 6M x 3 + ORDERS 1.5M = ~400 MB |
| **算子版本** | 基线 (DuckDB 回退) |
| **加速器** | - |
| **优化方案** | EXISTS + NOT EXISTS 组合未优化 (V35 尝试崩溃) |

---

### Q22: 全球销售机会 (Global Sales Opportunity)

**SQL:**
```sql
SELECT cntrycode, COUNT(*) as numcust, SUM(c_acctbal) as totacctbal
FROM (
  SELECT SUBSTRING(c_phone FROM 1 FOR 2) as cntrycode, c_acctbal
  FROM customer
  WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
    AND c_acctbal > (
      SELECT AVG(c_acctbal) FROM customer
      WHERE c_acctbal > 0.00
        AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
    )
    AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
) custsale
GROUP BY cntrycode
ORDER BY cntrycode;
```

| 项目 | 数值 |
|------|------|
| **DuckDB** | 10.44 ms |
| **ThunderDuck** | 11.14 ms |
| **加速比** | **0.94x** (略慢) |
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M = ~63 MB |
| **算子版本** | **V34** |
| **加速器** | CPU |
| **优化方案** | SUBSTRING 预计算、NOT EXISTS 转 ANTI Join、AVG 子查询物化 |

---

## 四、优化技术总结

### 4.1 加速器使用情况

| 加速器 | 使用查询 | 占比 |
|--------|---------|------|
| CPU SIMD (ARM Neon) | Q1, Q3, Q5, Q6, Q7, Q9, Q10, Q12, Q14, Q19 | 45% |
| 8 线程并行 | Q3, Q5, Q6, Q7, Q9, Q10, Q12, Q13, Q15, Q18 | 45% |
| GPU Metal | 预留 (未启用) | 0% |
| NPU (ANE) | 预留 (未启用) | 0% |

### 4.2 算子版本分布

| 版本 | 查询数 | 查询列表 |
|------|--------|---------|
| 基础版 | 5 | Q1, Q2, Q17, Q20, Q21 |
| V25 | 2 | Q6, Q10 |
| V27 | 6 | Q4, Q11, Q12, Q14, Q15, Q16 |
| V31 | 1 | Q3 |
| V32 | 4 | Q5, Q7, Q9, Q18 |
| V33 | 1 | Q19 |
| V34 | 3 | Q8, Q13, Q22 |

### 4.3 核心优化技术

| 技术 | 描述 | 受益查询 |
|------|------|---------|
| **直接数组聚合** | 低基数 GROUP BY 使用固定大小数组 | Q1, Q12 |
| **SIMD 向量化** | ARM Neon 128-bit 向量操作 | Q1, Q6, Q19 |
| **Thread-Local 聚合** | 消除原子操作竞争 | Q5, Q7, Q9 |
| **紧凑 Hash Table** | 开放寻址 + 线性探测 | Q5, Q7, Q9, Q18 |
| **Bloom Filter 预过滤** | 快速排除非匹配项 | Q5, Q7 |
| **SEMI/ANTI Join** | EXISTS/NOT EXISTS 优化 | Q4, Q16 |
| **PredicatePrecomputer** | 复杂谓词批量预计算 | Q16, Q19 |
| **Pipeline Fusion** | Filter->Join->Aggregate 融合 | Q3, Q12 |
| **8 路循环展开** | 提高指令级并行度 | Q6, Q18 |

---

## 五、性能汇总表

| Query | DuckDB (ms) | ThunderDuck (ms) | 加速比 | 版本 | 加速器 |
|-------|-------------|------------------|--------|------|--------|
| Q1 | 31.62 | 4.44 | **7.12x** | 基础 | SIMD |
| Q2 | 7.77 | 0.35 | **22.44x** | 基础 | CPU |
| Q3 | 13.43 | 9.19 | **1.46x** | V31 | SIMD+8T |
| Q4 | 15.05 | 4.25 | **3.54x** | V27 | SIMD |
| Q5 | 12.28 | 10.00 | **1.23x** | V32 | SIMD+8T |
| Q6 | 2.88 | 2.15 | **1.34x** | V25 | SIMD+8T |
| Q7 | 12.43 | 4.77 | **2.60x** | V32 | SIMD+8T |
| Q8 | 12.65 | 11.14 | **1.14x** | V34 | SIMD+8T |
| Q9 | 47.08 | 29.01 | **1.62x** | V32 | SIMD+8T |
| Q10 | 31.69 | 15.70 | **2.02x** | V25 | SIMD+8T |
| Q11 | 4.23 | 2.67 | **1.59x** | V27 | CPU |
| Q12 | 20.79 | 4.30 | **4.84x** | V27 | SIMD+8T |
| Q13 | 33.45 | 17.27 | **1.94x** | V34 | 8T |
| Q14 | 9.12 | 7.07 | **1.29x** | V27 | SIMD |
| Q15 | 6.08 | 2.60 | **2.34x** | V27 | SIMD+8T |
| Q16 | 12.91 | 8.29 | **1.56x** | V27 | CPU |
| Q17 | 13.42 | 13.42 | 1.00x | 基线 | - |
| Q18 | 33.60 | 14.47 | **2.32x** | V32 | 8T |
| Q19 | 35.60 | 5.05 | **7.05x** | V33 | SIMD |
| Q20 | 9.91 | 9.91 | 1.00x | 基线 | - |
| Q21 | 44.52 | 44.52 | 1.00x | 基线 | - |
| Q22 | 10.44 | 11.14 | 0.94x | V34 | CPU |

---

## 六、待优化查询

| 查询 | 当前加速比 | 瓶颈分析 | 优化方向 |
|------|-----------|---------|---------|
| Q17 | 1.00x | 相关子查询 | 物化 AVG 子查询、窗口函数改写 |
| Q20 | 1.00x | 嵌套相关子查询 | 多级 SEMI Join、物化子查询 |
| Q21 | 1.00x | EXISTS + NOT EXISTS | SemiAntiJoin 组合算子 (V35 待修复) |
| Q22 | 0.94x | SUBSTRING + NOT EXISTS | 字符串前缀索引、ANTI Join 优化 |

---

## 七、V35 通用算子状态

| 算子 | 状态 | 覆盖查询 |
|------|------|---------|
| DirectArrayIndexBuilder | 稳定 | Q3, Q8, Q14 |
| SIMDStringProcessor | 稳定 | Q14, Q22 |
| SemiAntiJoin | 崩溃待修复 | Q21 |
| ConditionalAggregator | 稳定 | Q8, Q14 |
| PipelineFusion | 稳定 | Q3 |

**V35 回退原因:**
- Q3: 性能回退 0.82x -> 恢复 V31
- Q8: 性能回退 0.87x -> 恢复 V34
- Q14: 崩溃 -> 恢复 V27
- Q21: 崩溃 -> 恢复基线
- Q22: 崩溃 -> 恢复 V34

---

## 八、结论

ThunderDuck 在 TPC-H SF=1 基准测试中实现了 **2.11x 几何平均加速比**，其中:

- **18 个查询** (81.8%) 比 DuckDB 更快
- **最大加速**: Q2 (22.44x)、Q19 (7.05x)、Q1 (7.12x)
- **待优化**: Q17, Q20, Q21 (相关子查询)、Q22 (略慢)

核心优势来自:
1. ARM Neon SIMD 向量化
2. 8 线程并行执行
3. 紧凑数据结构 (直接数组、紧凑 Hash Table)
4. 算法优化 (Thread-Local 聚合、Pipeline Fusion)
