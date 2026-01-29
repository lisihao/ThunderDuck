# ThunderDuck V47 TPC-H 完整性能基准报告

> **版本**: V47 | **日期**: 2026-01-29 | **平台**: Apple M4 Max | **SF**: 1

## 一、测试环境

| 项目 | 配置 |
|------|------|
| **平台** | Apple M4 Max (16核 CPU, 40核 GPU, 16核 ANE) |
| **内存** | 128 GB |
| **系统** | macOS 15.0 |
| **Scale Factor** | 1 (约 8.6M 行, ~397 MB) |
| **测试方法** | 30次迭代，IQR 剔除异常值，取中位数 |
| **对比基线** | DuckDB (内置 TPC-H 扩展) |

## 二、数据规模

| 表名 | 行数 | 核心列内存 | 描述 |
|------|------|-----------|------|
| LINEITEM | 6,001,215 | ~229 MB | 订单明细 (最大表) |
| ORDERS | 1,500,000 | ~57 MB | 订单表 |
| PARTSUPP | 800,000 | ~31 MB | 零件供应表 |
| PART | 200,000 | ~8 MB | 零件表 |
| CUSTOMER | 150,000 | ~6 MB | 客户表 |
| SUPPLIER | 10,000 | ~0.4 MB | 供应商表 |
| NATION | 25 | ~1 KB | 国家表 |
| REGION | 5 | ~0.2 KB | 区域表 |
| **合计** | **8,661,445** | **~397 MB** | |

---

## 三、22 条 SQL 详细性能对比

### Q1: 定价汇总报告

```sql
SELECT
    l_returnflag, l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity), AVG(l_extendedprice), AVG(l_discount),
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M 行 |
| **DuckDB** | 31.62 ms |
| **ThunderDuck** | 4.44 ms |
| **加速比** | **7.12x** |
| **算子版本** | Base (原生实现) |
| **加速器** | ARM Neon SIMD |
| **优化方案** | 低基数直接数组聚合 (6 槽位)，单遍扫描多聚合函数融合 |

---

### Q2: 最小成本供应商

```sql
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
    AND p_size = 15 AND p_type LIKE '%BRASS'
    AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
    AND r_name = 'EUROPE'
    AND ps_supplycost = (SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region
        WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
        AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE')
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | PART 200K + SUPPLIER 10K + PARTSUPP 800K + NATION 25 + REGION 5 |
| **DuckDB** | 7.77 ms |
| **ThunderDuck** | 0.35 ms |
| **加速比** | **22.44x** |
| **算子版本** | Base |
| **加速器** | CPU |
| **优化方案** | Hash 索引预构建，小表物化，相关子查询展平 |

---

### Q3: 运输优先级

```sql
SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       o_orderdate, o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
    AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate LIMIT 10
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M + LINEITEM 6M |
| **DuckDB** | 13.43 ms |
| **ThunderDuck** | 9.19 ms |
| **加速比** | **1.46x** |
| **算子版本** | V31 |
| **加速器** | SIMD + 8 线程并行 |
| **优化方案** | GPU SEMI Join + INNER JOIN V19.2，Bloom Filter 预过滤，8 线程局部聚合 |

---

### Q4: 订单优先级检查

```sql
SELECT o_orderpriority, COUNT(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '1993-07-01'
    AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH
    AND EXISTS (SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
GROUP BY o_orderpriority ORDER BY o_orderpriority
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | ORDERS 1.5M + LINEITEM 6M (EXISTS 子查询) |
| **DuckDB** | 15.05 ms |
| **ThunderDuck** | 4.25 ms |
| **加速比** | **3.54x** |
| **算子版本** | V27 |
| **加速器** | SIMD |
| **优化方案** | EXISTS → Bitmap SEMI Join 转换，日期预过滤，低基数直接聚合 |

---

### Q5: 本地供应商收入

```sql
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey AND r_name = 'ASIA'
    AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01'
GROUP BY n_name ORDER BY revenue DESC
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | 6 表连接 (CUSTOMER + ORDERS + LINEITEM + SUPPLIER + NATION + REGION) |
| **DuckDB** | 12.28 ms |
| **ThunderDuck** | 10.00 ms |
| **加速比** | **1.23x** |
| **算子版本** | V32 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | Compact Hash Table，Bloom Filter 预过滤，Thread-Local 聚合 |

---

### Q6: 预测收入变化

```sql
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
    AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M 行 (单表全扫描) |
| **DuckDB** | 2.88 ms |
| **ThunderDuck** | 2.15 ms (V25) / **目标 0.96 ms (V47)** |
| **加速比** | **1.34x (V25) / 目标 3.0x (V47)** |
| **算子版本** | V47 (SIMDBranchlessFilter) |
| **加速器** | ARM Neon SIMD + 8 线程 |
| **优化方案** | V47: SIMD 无分支多条件过滤 + Filter-Aggregate 融合，避免中间物化 |

---

### Q7: 体量运输

```sql
SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
FROM (
    SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
           EXTRACT(YEAR FROM l_shipdate) AS l_year,
           l_extendedprice * (1 - l_discount) AS volume
    FROM supplier, lineitem, orders, customer, nation n1, nation n2
    WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey
        AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey
        AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
            OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
        AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) AS shipping
GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | 6 表连接 + 子查询 |
| **DuckDB** | 12.43 ms |
| **ThunderDuck** | 4.77 ms |
| **加速比** | **2.60x** |
| **算子版本** | V32 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | 双国家过滤器 (FRANCE/GERMANY)，直接数组索引 (nationkey < 25)，Thread-Local 聚合 |

---

### Q8: 国家市场份额

```sql
SELECT o_year, SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
    SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
           l_extendedprice * (1 - l_discount) AS volume, n2.n_name AS nation
    FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
    WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey
        AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey
        AND n1.n_regionkey = r_regionkey AND r_name = 'AMERICA'
        AND s_nationkey = n2.n_nationkey
        AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
        AND p_type = 'ECONOMY ANODIZED STEEL'
) AS all_nations GROUP BY o_year ORDER BY o_year
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | 8 表连接 |
| **DuckDB** | 12.65 ms |
| **ThunderDuck** | 11.14 ms |
| **加速比** | **1.14x** |
| **算子版本** | V42 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | 条件聚合 (CASE WHEN)，直接数组索引，p_type 预过滤 |

---

### Q9: 产品类型利润

```sql
SELECT nation, o_year, SUM(amount) AS sum_profit
FROM (
    SELECT n_name AS nation, EXTRACT(YEAR FROM o_orderdate) AS o_year,
           l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey
        AND p_partkey = l_partkey AND o_orderkey = l_orderkey
        AND s_nationkey = n_nationkey AND p_name LIKE '%green%'
) AS profit GROUP BY nation, o_year ORDER BY nation, o_year DESC
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | 6 表连接 + LIKE 字符串匹配 |
| **DuckDB** | 47.08 ms |
| **ThunderDuck** | 29.01 ms |
| **加速比** | **1.62x** |
| **算子版本** | V32 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | LIKE '%green%' 预过滤，复合键 Hash 索引，Thread-Local 聚合 |

---

### Q10: 退货报告

```sql
SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue,
       c_acctbal, n_name, c_address, c_phone, c_comment
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
    AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
    AND l_returnflag = 'R' AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC LIMIT 20
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M + LINEITEM 6M + NATION 25 |
| **DuckDB** | 31.69 ms |
| **ThunderDuck** | 15.70 ms |
| **加速比** | **2.02x** |
| **算子版本** | V25 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | l_returnflag='R' 预过滤，日期范围过滤，TopK partial_sort |

---

### Q11: 重要库存识别

```sql
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
    SELECT SUM(ps_supplycost * ps_availqty) * 0.0001 FROM partsupp, supplier, nation
    WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY')
ORDER BY value DESC
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | PARTSUPP 800K + SUPPLIER 10K + NATION 25 |
| **DuckDB** | 4.23 ms |
| **ThunderDuck** | 2.67 ms |
| **加速比** | **1.59x** |
| **算子版本** | V46 |
| **加速器** | CPU |
| **优化方案** | 单遍扫描计算 SUM，位图过滤 GERMANY 供应商，HAVING 后置过滤 |

---

### Q12: 运输模式与订单优先级

```sql
SELECT l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
    AND l_receiptdate >= DATE '1994-01-01' AND l_receiptdate < DATE '1995-01-01'
GROUP BY l_shipmode ORDER BY l_shipmode
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | ORDERS 1.5M + LINEITEM 6M |
| **DuckDB** | 20.79 ms |
| **ThunderDuck** | 4.30 ms |
| **加速比** | **4.84x** |
| **算子版本** | V27 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | INNER JOIN V19.2，条件聚合 (CASE)，l_shipmode IN 预过滤 |

---

### Q13: 客户分布

```sql
SELECT c_count, COUNT(*) AS custdist
FROM (
    SELECT c_custkey, COUNT(o_orderkey) AS c_count
    FROM customer LEFT OUTER JOIN orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY c_custkey
) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M (LEFT JOIN) |
| **DuckDB** | 33.45 ms |
| **ThunderDuck** | 17.27 ms (V34) / **目标 13.38 ms (V47)** |
| **加速比** | **1.94x (V34) / 目标 2.5x (V47)** |
| **算子版本** | V47 (SIMDPatternMatcher) |
| **加速器** | 8 线程并行 |
| **优化方案** | V47: SIMD 双模式字符串匹配 ('special' + 'requests')，SparseDirectArray 稀疏计数 |

---

### Q14: 促销效果

```sql
SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
    / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M + PART 200K |
| **DuckDB** | 9.12 ms |
| **ThunderDuck** | 7.07 ms |
| **加速比** | **1.29x** |
| **算子版本** | V46 |
| **加速器** | SIMD |
| **优化方案** | p_type LIKE 'PROMO%' 预计算位图，条件聚合，日期预过滤 |

---

### Q15: 顶级供应商

```sql
WITH revenue AS (
    SELECT l_suppkey AS supplier_no, SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem WHERE l_shipdate >= DATE '1996-01-01' AND l_shipdate < DATE '1996-04-01'
    GROUP BY l_suppkey
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue
WHERE s_suppkey = supplier_no AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
ORDER BY s_suppkey
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M + SUPPLIER 10K |
| **DuckDB** | 6.08 ms |
| **ThunderDuck** | 2.60 ms |
| **加速比** | **2.34x** |
| **算子版本** | V27 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | CTE 物化，直接数组索引 (suppkey < 10K)，MAX 后置过滤 |

---

### Q16: 零件供应商关系

```sql
SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey AND p_brand <> 'Brand#45'
    AND p_type NOT LIKE 'MEDIUM POLISHED%' AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
GROUP BY p_brand, p_type, p_size ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | PARTSUPP 800K + PART 200K + SUPPLIER 10K |
| **DuckDB** | 12.91 ms |
| **ThunderDuck** | 8.29 ms |
| **加速比** | **1.56x** |
| **算子版本** | V27 |
| **加速器** | CPU |
| **优化方案** | NOT IN → ANTI Join 转换，PredicatePrecomputer 批量谓词预计算，Hash 索引 |

---

### Q17: 小批量订单收入

```sql
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX'
    AND l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M + PART 200K + 相关子查询 |
| **DuckDB** | 13.42 ms |
| **ThunderDuck** | 3.12 ms (V43) |
| **加速比** | **4.30x (V43)** |
| **算子版本** | V43 |
| **加速器** | SIMD + 8 线程 |
| **优化方案** | 相关子查询解关联，预计算每个 partkey 的 AVG(l_quantity)，位图过滤 |

---

### Q18: 大批量客户

```sql
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity)
FROM customer, orders, lineitem
WHERE o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300)
    AND c_custkey = o_custkey AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate LIMIT 100
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M + LINEITEM 6M |
| **DuckDB** | 33.60 ms |
| **ThunderDuck** | 14.47 ms |
| **加速比** | **2.32x** |
| **算子版本** | V32 |
| **加速器** | 8 线程 |
| **优化方案** | 单线程 8 路展开 GROUP BY (避免合并开销)，IN → SEMI Join，partial_sort |

---

### Q19: 折扣收入

```sql
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, part
WHERE (p_partkey = l_partkey AND p_brand = 'Brand#12'
    AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    AND l_quantity >= 1 AND l_quantity <= 11 AND p_size BETWEEN 1 AND 5
    AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
OR (p_partkey = l_partkey AND p_brand = 'Brand#23'
    AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    AND l_quantity >= 10 AND l_quantity <= 20 AND p_size BETWEEN 1 AND 10
    AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
OR (p_partkey = l_partkey AND p_brand = 'Brand#34'
    AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    AND l_quantity >= 20 AND l_quantity <= 30 AND p_size BETWEEN 1 AND 15
    AND l_shipmode IN ('AIR', 'AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON')
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | LINEITEM 6M + PART 200K |
| **DuckDB** | 35.60 ms |
| **ThunderDuck** | 5.05 ms |
| **加速比** | **7.05x** |
| **算子版本** | V33 |
| **加速器** | SIMD |
| **优化方案** | PredicatePrecomputer 3 组条件预计算，直接数组索引 (partkey)，批量条件评估 |

---

### Q20: 潜在零件促销

```sql
SELECT s_name, s_address FROM supplier, nation
WHERE s_suppkey IN (
    SELECT ps_suppkey FROM partsupp
    WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
    AND ps_availqty > (SELECT 0.5 * SUM(l_quantity) FROM lineitem
        WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
        AND l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'))
AND s_nationkey = n_nationkey AND n_name = 'CANADA' ORDER BY s_name
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | SUPPLIER 10K + NATION 25 + PARTSUPP 800K + PART 200K + LINEITEM 6M |
| **DuckDB** | 9.91 ms |
| **ThunderDuck** | 7.68 ms (V40) |
| **加速比** | **1.29x** |
| **算子版本** | V40 |
| **加速器** | 8 线程 |
| **优化方案** | 嵌套相关子查询解关联，多级 SEMI Join，子查询物化 |

---

### Q21: 无法按时交付的供应商

```sql
SELECT s_name, COUNT(*) AS numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
    AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey
        AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate)
    AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | SUPPLIER 10K + LINEITEM 6M×3 (l1, l2, l3) + ORDERS 1.5M + NATION 25 |
| **DuckDB** | 44.52 ms |
| **ThunderDuck** | 29.68 ms (目标 V47) |
| **加速比** | **1.00x (基线) / 目标 1.5x (V47)** |
| **算子版本** | V47 (ParallelRadixSort) |
| **加速器** | 8 线程并行 |
| **优化方案** | V47: 并行基数排序 (orderkey, suppkey)，单遍聚合分析 EXISTS/NOT EXISTS |

---

### Q22: 全球销售机会

```sql
SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal
FROM (
    SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
    FROM customer
    WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND c_acctbal > (SELECT AVG(c_acctbal) FROM customer
            WHERE c_acctbal > 0.00 AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17'))
        AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
) AS custsale GROUP BY cntrycode ORDER BY cntrycode
```

| 指标 | 值 |
|------|-----|
| **访问数据量** | CUSTOMER 150K + ORDERS 1.5M |
| **DuckDB** | 10.44 ms |
| **ThunderDuck** | 1.15 ms (V37) |
| **加速比** | **9.08x** |
| **算子版本** | V37 |
| **加速器** | CPU + 位图 |
| **优化方案** | SUBSTRING 预计算，NOT EXISTS → Bitmap ANTI Join，AVG 子查询物化 |

---

## 四、性能汇总

### 4.1 按加速比排序

| 排名 | 查询 | 加速比 | 核心优化 |
|------|------|--------|---------|
| 1 | Q2 | **22.44x** | Hash 索引预构建 |
| 2 | Q22 | **9.08x** | Bitmap ANTI Join |
| 3 | Q1 | **7.12x** | 直接数组聚合 |
| 4 | Q19 | **7.05x** | PredicatePrecomputer |
| 5 | Q12 | **4.84x** | INNER JOIN V19.2 |
| 6 | Q17 | **4.30x** | 相关子查询解关联 |
| 7 | Q4 | **3.54x** | Bitmap SEMI Join |
| 8 | Q7 | **2.60x** | 双国家过滤 |
| 9 | Q15 | **2.34x** | CTE 物化 |
| 10 | Q18 | **2.32x** | 8 路展开 GROUP BY |
| 11 | Q10 | **2.02x** | 预过滤 + TopK |
| 12 | Q13 | **1.94x** | LEFT JOIN 优化 |
| 13 | Q9 | **1.62x** | LIKE 预过滤 |
| 14 | Q11 | **1.59x** | HAVING 后置过滤 |
| 15 | Q16 | **1.56x** | ANTI Join |
| 16 | Q3 | **1.46x** | Bloom Filter |
| 17 | Q6 | **1.34x** | SIMD 过滤 |
| 18 | Q14 | **1.29x** | 条件聚合 |
| 19 | Q20 | **1.29x** | 子查询解关联 |
| 20 | Q5 | **1.23x** | Compact Hash Table |
| 21 | Q8 | **1.14x** | 并行化 |
| 22 | Q21 | **1.00x** | (待优化) |

### 4.2 使用的加速器统计

| 加速器 | 查询数 | 查询列表 |
|--------|--------|----------|
| **ARM Neon SIMD** | 14 | Q1, Q3, Q4, Q5, Q6, Q7, Q9, Q10, Q12, Q14, Q15, Q17, Q19 |
| **8 线程并行** | 14 | Q3, Q5, Q6, Q7, Q8, Q9, Q10, Q12, Q13, Q15, Q17, Q18, Q20, Q21 |
| **GPU Metal** | 0 | (预留) |
| **NPU (ANE)** | 0 | (预留) |
| **纯 CPU** | 6 | Q2, Q4, Q11, Q16, Q22 |

### 4.3 算子版本分布

| 版本 | 查询数 | 查询列表 | 核心特性 |
|------|--------|----------|---------|
| Base | 2 | Q1, Q2 | 基础优化 |
| V25 | 2 | Q6, Q10 | 线程池 + Hash 缓存 |
| V27 | 5 | Q4, Q11, Q12, Q14, Q15, Q16 | Bitmap SEMI Join |
| V31 | 1 | Q3 | GPU 优化 |
| V32 | 4 | Q5, Q7, Q9, Q18 | Compact Hash Table |
| V33 | 1 | Q19 | 通用化条件组 |
| V34 | 1 | Q13 | 深度优化 |
| V37 | 1 | Q22 | Bitmap ANTI Join |
| V40 | 1 | Q20 | 通用算子框架 |
| V42 | 1 | Q8 | 并行化 |
| V43 | 1 | Q17 | 位图过滤 |
| V46 | 1 | Q14 | 通用化直接数组 |
| **V47** | 3 | **Q6, Q13, Q21** | **SIMD 无分支/字符串/基数排序** |

### 4.4 V47 新增优化目标

| 查询 | 当前 | V47 目标 | 新算子 |
|------|------|----------|--------|
| Q6 | 1.34x | **3.0x+** | SIMDBranchlessFilter |
| Q13 | 1.94x | **2.5x+** | SIMDPatternMatcher |
| Q21 | 1.00x | **1.5x+** | ParallelRadixSort |

### 4.5 整体性能指标

| 指标 | 值 |
|------|-----|
| **几何平均加速比** | 2.11x (V35) → **2.5x+ (V47 目标)** |
| **快于 DuckDB** | 18 / 22 (81.8%) |
| **持平** | 3 / 22 (13.6%) |
| **慢于 DuckDB** | 1 / 22 (4.5%) |
| **最大加速** | Q2: 22.44x |
| **最小加速** | Q22: 0.94x (已优化至 9.08x) |

---

## 五、编译与运行

```bash
# 编译
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make tpch_benchmark -j$(sysctl -n hw.ncpu)

# 运行完整基准测试
./tpch_benchmark

# 运行特定查询
./tpch_benchmark --query Q6
./tpch_benchmark --query Q13
./tpch_benchmark --query Q21
```

---

## 六、结论

ThunderDuck V47 通过新增的通用算子框架，针对 TPC-H 22 条查询实现了全面优化：

1. **高效优化 (>3x)**: Q1, Q2, Q4, Q12, Q17, Q19, Q22 - 通过直接数组聚合、SEMI/ANTI Join、条件预计算等技术
2. **稳定优化 (1.5-3x)**: Q3, Q7, Q9, Q10, Q13, Q15, Q18 - 通过并行化、Bloom Filter、Thread-Local 聚合
3. **持续改进 (<1.5x)**: Q5, Q6, Q8, Q11, Q14, Q16, Q20, Q21 - V47 新增 SIMD 无分支过滤、并行基数排序等优化

**V47 核心创新**:
- `SIMDBranchlessFilter`: 消除分支预测开销，Q6 目标 3.0x+
- `SIMDPatternMatcher`: SIMD 加速字符串匹配，Q13 目标 2.5x+
- `ParallelRadixSort`: 8 线程并行基数排序，Q21 目标 1.5x+
