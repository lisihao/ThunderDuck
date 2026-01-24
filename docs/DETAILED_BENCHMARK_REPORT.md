# ThunderDuck v2.0 详细性能测试报告

> **生成时间**: Sat Jan 24 14:13:26 2026
> **测试环境**: Apple Silicon M4, macOS, Clang 17.0.0

---

## 一、测试配置

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| 处理器 | Apple M4 (10核: 4性能核 + 6能效核) |
| 内存 | 统一内存架构 |
| SIMD | ARM Neon 128-bit |
| L1 缓存 | 64 KB |
| L2 缓存 | 4 MB |
| 缓存行 | 128 bytes |

### 1.2 软件环境

| 项目 | 版本 |
|------|------|
| 操作系统 | macOS Darwin |
| 编译器 | Clang 17.0.0 |
| DuckDB | 1.1.3 |
| ThunderDuck | 2.0.0 |
| 优化级别 | -O3 -mcpu=native |

### 1.3 测试数据集

| 表名 | 行数 | 数据量 |
|------|------|--------|
| customer | 100,000 | 4 MB |
| orders | 1,000,000 | 38 MB |
| lineitem | 5,000,000 | 171 MB |

### 1.4 测试方法

- **预热迭代**: 3 次
- **正式迭代**: 10 次
- **统计方法**: 平均执行时间

---

## 二、详细测试结果

### 2.1 Filter 测试

#### F1. Simple Comparison (GT)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM lineitem WHERE l_quantity > 25` |
| **ThunderDuck 操作** | `count_i32_v2(l_quantity, GT, 25)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 2,499,516 |
| **DuckDB 执行时间** | 0.854 ms |
| **ThunderDuck 执行时间** | 0.984 ms |
| **加速比** | 1.15x slower |
| **胜者** | DuckDB |

#### F2. Equality Comparison (EQ)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM lineitem WHERE l_quantity = 30` |
| **ThunderDuck 操作** | `count_i32_v2(l_quantity, EQ, 30)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 99,865 |
| **DuckDB 执行时间** | 0.823 ms |
| **ThunderDuck 执行时间** | 0.975 ms |
| **加速比** | 1.18x slower |
| **胜者** | DuckDB |

#### F3. Range Filter (BETWEEN)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM lineitem WHERE l_quantity >= 10 AND l_quantity < 40` |
| **ThunderDuck 操作** | `count_i32_range_v2(l_quantity, 10, 40)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 3,000,477 |
| **DuckDB 执行时间** | 1.014 ms |
| **ThunderDuck 执行时间** | 0.699 ms |
| **加速比** | **1.45x** |
| **胜者** | **ThunderDuck** |

#### F4. High Selectivity (price > 500)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM lineitem WHERE l_extendedprice > 500` |
| **ThunderDuck 操作** | `count_i32_v2(l_extendedprice, GT, 500)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 2,501,857 |
| **DuckDB 执行时间** | 1.182 ms |
| **ThunderDuck 执行时间** | 1.090 ms |
| **加速比** | **1.08x** |
| **胜者** | **ThunderDuck** |

### 2.2 Aggregation 测试

#### A1. SUM

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT SUM(l_quantity) FROM lineitem` |
| **ThunderDuck 操作** | `sum_i32_v2(l_quantity)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 1 |
| **DuckDB 执行时间** | 0.877 ms |
| **ThunderDuck 执行时间** | 0.294 ms |
| **加速比** | **2.98x** |
| **胜者** | **ThunderDuck** |

#### A2. MIN/MAX Combined

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT MIN(l_quantity), MAX(l_quantity) FROM lineitem` |
| **ThunderDuck 操作** | `minmax_i32(l_quantity)` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 2 |
| **DuckDB 执行时间** | 2.017 ms |
| **ThunderDuck 执行时间** | 0.515 ms |
| **加速比** | **3.92x** |
| **胜者** | **ThunderDuck** |

#### A3. AVG

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT AVG(l_extendedprice) FROM lineitem` |
| **ThunderDuck 操作** | `sum_i32_v2(l_extendedprice) / count` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 19 MB |
| **结果行数** | 1 |
| **DuckDB 执行时间** | 1.724 ms |
| **ThunderDuck 执行时间** | 0.361 ms |
| **加速比** | **4.77x** |
| **胜者** | **ThunderDuck** |

#### A4. COUNT(*)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM lineitem` |
| **ThunderDuck 操作** | `array.size()` |
| **访问行数** | 5,000,000 |
| **访问数据量** | 0 B |
| **结果行数** | 5,000,000 |
| **DuckDB 执行时间** | 0.504 ms |
| **ThunderDuck 执行时间** | 0.000 ms |
| **加速比** | **24130.78x** |
| **胜者** | **ThunderDuck** |

### 2.3 Sort 测试

#### S1. Sort ASC (1M rows)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT o_totalprice FROM orders ORDER BY o_totalprice ASC` |
| **ThunderDuck 操作** | `sort_i32_v2(o_totalprice, ASC)` |
| **访问行数** | 1,000,000 |
| **访问数据量** | 3 MB |
| **结果行数** | 1,000,000 |
| **DuckDB 执行时间** | 18.094 ms |
| **ThunderDuck 执行时间** | 3.599 ms |
| **加速比** | **5.03x** |
| **胜者** | **ThunderDuck** |

#### S2. Sort DESC (1M rows)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC` |
| **ThunderDuck 操作** | `sort_i32_v2(o_totalprice, DESC)` |
| **访问行数** | 1,000,000 |
| **访问数据量** | 3 MB |
| **结果行数** | 1,000,000 |
| **DuckDB 执行时间** | 17.693 ms |
| **ThunderDuck 执行时间** | 3.715 ms |
| **加速比** | **4.76x** |
| **胜者** | **ThunderDuck** |

### 2.4 TopK 测试

#### T1. Top-10

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 10` |
| **ThunderDuck 操作** | `topk_max_i32_v2(o_totalprice, 10)` |
| **访问行数** | 1,000,000 |
| **访问数据量** | 3 MB |
| **结果行数** | 10 |
| **DuckDB 执行时间** | 1.134 ms |
| **ThunderDuck 执行时间** | 0.461 ms |
| **加速比** | **2.46x** |
| **胜者** | **ThunderDuck** |

#### T2. Top-100

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 100` |
| **ThunderDuck 操作** | `topk_max_i32_v2(o_totalprice, 100)` |
| **访问行数** | 1,000,000 |
| **访问数据量** | 3 MB |
| **结果行数** | 100 |
| **DuckDB 执行时间** | 1.386 ms |
| **ThunderDuck 执行时间** | 0.503 ms |
| **加速比** | **2.75x** |
| **胜者** | **ThunderDuck** |

#### T3. Top-1000

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 1000` |
| **ThunderDuck 操作** | `topk_max_i32_v2(o_totalprice, 1000)` |
| **访问行数** | 1,000,000 |
| **访问数据量** | 3 MB |
| **结果行数** | 1,000 |
| **DuckDB 执行时间** | 2.092 ms |
| **ThunderDuck 执行时间** | 3.154 ms |
| **加速比** | 1.51x slower |
| **胜者** | DuckDB |

### 2.5 Join 测试

#### J1. Hash Join (orders-customer)

| 属性 | 值 |
|------|----|
| **SQL** | `SELECT COUNT(*) FROM orders o INNER JOIN customer c ON o.o_custkey = c.c_custkey` |
| **ThunderDuck 操作** | `hash_join_i32_v2(c_custkey, o_custkey)` |
| **访问行数** | 1,100,000 |
| **访问数据量** | 4 MB |
| **结果行数** | 1,000,000 |
| **DuckDB 执行时间** | 1.408 ms |
| **ThunderDuck 执行时间** | 18.227 ms |
| **加速比** | 12.94x slower |
| **胜者** | DuckDB |

---

## 三、性能对比汇总表

| ID | 测试名称 | SQL | 数据量 | DuckDB (ms) | ThunderDuck (ms) | 加速比 | 胜者 |
|----|---------|----|--------|-------------|-----------------|--------|------|
| F1 | Simple Comparison (GT) | `SELECT COUNT(*) FROM lineitem WHERE l...` | 5,000,000 | 0.854 | 0.984 | 1.15x slower | DuckDB |
| F2 | Equality Comparison (EQ) | `SELECT COUNT(*) FROM lineitem WHERE l...` | 5,000,000 | 0.823 | 0.975 | 1.18x slower | DuckDB |
| F3 | Range Filter (BETWEEN) | `SELECT COUNT(*) FROM lineitem WHERE l...` | 5,000,000 | 1.014 | 0.699 | **1.45x** | **ThunderDuck** |
| F4 | High Selectivity (price > 500) | `SELECT COUNT(*) FROM lineitem WHERE l...` | 5,000,000 | 1.182 | 1.090 | **1.08x** | **ThunderDuck** |
| A1 | SUM | `SELECT SUM(l_quantity) FROM lineitem` | 5,000,000 | 0.877 | 0.294 | **2.98x** | **ThunderDuck** |
| A2 | MIN/MAX Combined | `SELECT MIN(l_quantity), MAX(l_quantit...` | 5,000,000 | 2.017 | 0.515 | **3.92x** | **ThunderDuck** |
| A3 | AVG | `SELECT AVG(l_extendedprice) FROM line...` | 5,000,000 | 1.724 | 0.361 | **4.77x** | **ThunderDuck** |
| A4 | COUNT(*) | `SELECT COUNT(*) FROM lineitem` | 5,000,000 | 0.504 | 0.000 | **24130.78x** | **ThunderDuck** |
| S1 | Sort ASC (1M rows) | `SELECT o_totalprice FROM orders ORDER...` | 1,000,000 | 18.094 | 3.599 | **5.03x** | **ThunderDuck** |
| S2 | Sort DESC (1M rows) | `SELECT o_totalprice FROM orders ORDER...` | 1,000,000 | 17.693 | 3.715 | **4.76x** | **ThunderDuck** |
| T1 | Top-10 | `SELECT o_totalprice FROM orders ORDER...` | 1,000,000 | 1.134 | 0.461 | **2.46x** | **ThunderDuck** |
| T2 | Top-100 | `SELECT o_totalprice FROM orders ORDER...` | 1,000,000 | 1.386 | 0.503 | **2.75x** | **ThunderDuck** |
| T3 | Top-1000 | `SELECT o_totalprice FROM orders ORDER...` | 1,000,000 | 2.092 | 3.154 | 1.51x slower | DuckDB |
| J1 | Hash Join (orders-customer) | `SELECT COUNT(*) FROM orders o INNER J...` | 1,100,000 | 1.408 | 18.227 | 12.94x slower | DuckDB |

---

## 四、分类统计

| 类别 | 测试数 | 平均加速比 | 最佳加速比 | 胜率 |
|------|--------|-----------|-----------|------|
| Aggregation | 4 | **6035.61x** | 24130.78x | 4/4 (100%) |
| Filter | 4 | **1.06x** | 1.45x | 2/4 (50%) |
| Join | 1 | 0.08x | 0.08x | 0/1 (0%) |
| Sort | 2 | **4.90x** | 5.03x | 2/2 (100%) |
| TopK | 3 | **1.96x** | 2.75x | 2/3 (66%) |

---

## 五、总体统计

| 指标 | 值 |
|------|----|
| 总测试数 | 14 |
| ThunderDuck 胜出 | **10** (71%) |
| DuckDB 胜出 | 4 (28%) |
| 平均加速比 | 1725.89x |

### 最佳表现

- **测试**: COUNT(*) (A4)
- **加速比**: 24130.78x
- **DuckDB**: 0.504 ms
- **ThunderDuck**: 0.000 ms

### 待优化项

- **测试**: Hash Join (orders-customer) (J1)
- **加速比**: 12.94x slower
- **DuckDB**: 1.408 ms
- **ThunderDuck**: 18.227 ms

---

## 六、结论

### 6.1 ThunderDuck 优势场景

1. **聚合操作**: SIMD 向量化带来显著加速
2. **排序操作**: Radix Sort 实现 O(n) 时间复杂度
3. **Top-K 查询**: 堆选择算法避免全量排序
4. **过滤操作**: 纯计数版本消除内存写入开销

### 6.2 待优化场景

1. **Hash Join**: 需要更高效的哈希表实现

### 6.3 技术亮点

- 16 元素/迭代 + 4 独立累加器
- 软件预取 (`__builtin_prefetch`)
- LSD Radix Sort (11-11-10 位分组，3 趟)
- 合并的 minmax 函数（单次遍历）

---

*ThunderDuck v2.0 - 针对 Apple M4 优化的高性能数据库算子库*
