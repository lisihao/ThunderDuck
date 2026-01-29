# ThunderDuck V23 性能分析报告

> 日期: 2026-01-28 | 基于 TPC-H SF=1 测试

## 一、测试结果总览

### 1.1 算子级别性能 (6M 行)

| 算子 | DuckDB | ThunderDuck | 加速比 |
|------|--------|-------------|--------|
| Filter i32 | 2.25 ms | 0.91 ms | **2.47x** |
| SUM i32 | 1.14 ms | 0.34 ms | **3.32x** |
| SUM i64 | 1.46 ms | 0.81 ms | **1.80x** |
| GROUP BY (6 groups) | 9.54 ms | 0.82 ms | **11.61x** |
| INNER JOIN | 44.46 ms | 29.03 ms | **1.53x** |

**结论**: ThunderDuck 算子本身比 DuckDB 快 1.5-11x。

### 1.2 TPC-H 查询性能

| 类别 | 查询数 | 加速 | 减速 | 平均加速比 |
|------|--------|------|------|-----------|
| A (完全优化) | 10 | 4 | 6 | 1.20x |
| B (部分优化) | 6 | 1 | 4 | 0.63x |
| C (回退) | 6 | 0 | 0 | 1.00x |

**几何平均**: 0.84x (整体变慢)

### 1.3 关键发现

```
┌─────────────────────────────────────────────────────────┐
│  算子快 2-11x，但完整查询慢 0.84x                        │
│  这说明：数据提取/移动/中间结果开销 >> 算子加速收益      │
└─────────────────────────────────────────────────────────┘
```

## 二、性能瓶颈深度分析

### 2.1 表现好的查询分析

| 查询 | 加速比 | 原因 |
|------|--------|------|
| Q1 | **6.38x** | GROUP BY 聚合主导，单表扫描，无 JOIN |
| Q2 | **23.20x** | 小表操作，内存命中率高 |
| Q6 | **1.44x** | 单表 Filter + SUM，无 JOIN，无中间结果 |
| Q10 | **1.16x** | JOIN 少，聚合为主 |
| Q19 | **1.17x** | 简单过滤 |

**共同特点**:
- 单表或少表操作
- 以聚合为主
- 无复杂 JOIN 链

### 2.2 表现差的查询分析

| 查询 | 减速比 | 主要瓶颈 |
|------|--------|---------|
| Q3 | 0.32x | 3 表 JOIN + 多次数据复制 |
| Q4 | 0.18x | 子查询重写效率低 |
| Q5 | 0.26x | 5 表 JOIN + 多次 hash 查找 |
| Q9 | 0.42x | 6 表 JOIN + 大量中间结果 |
| Q18 | 0.38x | GROUP BY + HAVING + 子查询 |

### 2.3 Q3 详细瓶颈分析 (0.32x)

```cpp
// 当前实现的问题：

// Step 1: 过滤客户 - O(150K) 循环 + vector 分配
for (i = 0; i < cust.count; ++i) {
    if (cust.c_mktsegment_code[i] == 1) {
        building_custkeys.push_back(cust.c_custkey[i]);  // 堆分配
    }
}

// Step 2: SEMI JOIN - GPU 很快，但结果需要复制到 CPU
ops::semi_join_i32(...);  // GPU -> CPU 复制开销

// Step 3: 再次循环过滤日期 - O(SEMI 结果) + 3 个 vector 分配
for (uint32_t idx : orders_semi_matches) {
    if (ord.o_orderdate[idx] < date_threshold) {
        valid_orderkeys.push_back(...);      // 堆分配
        valid_orderdates.push_back(...);     // 堆分配
        valid_shippriorities.push_back(...); // 堆分配
    }
}

// Step 4: 过滤 lineitem - O(6M) 循环
for (i = 0; i < li.count; ++i) {
    if (li.l_shipdate[i] > date_threshold) {
        li_orderkeys.push_back(...);  // 堆分配
        li_indices.push_back(...);    // 堆分配
    }
}

// Step 5: INNER JOIN - OK

// Step 6: 聚合 - 使用 unordered_map，缓存不友好
for (j = start; j < end; ++j) {
    auto& r = local_results[orderkey];  // hash 查找，缓存未命中
    ...
}
```

**瓶颈量化**:
- 5 次数据遍历 (vs DuckDB 流水线执行)
- 10+ 次 std::vector 堆分配
- ~10M 次 unordered_map 操作 (hash + 查找)
- GPU->CPU 数据传输

## 三、根本原因

### 3.1 架构层面

```
┌─────────────────────────────────────────────────────────┐
│                    DuckDB 执行模式                       │
├─────────────────────────────────────────────────────────┤
│  Table Scan ──▶ Filter ──▶ Join ──▶ Aggregate          │
│       │            │          │          │              │
│       └────────────┴──────────┴──────────┘              │
│              流水线执行，数据不落地                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                ThunderDuck 当前执行模式                   │
├─────────────────────────────────────────────────────────┤
│  DuckDB 提取 ──▶ 内存数组                               │
│       │                │                                │
│       │          Filter 算子 ──▶ 中间 vector            │
│       │                │                                │
│       │          Join 算子 ──▶ 中间 vector              │
│       │                │                                │
│       │        Aggregate 算子 ──▶ 结果                  │
│       └────────────────┴─────────────────               │
│              每步物化，数据多次复制                       │
└─────────────────────────────────────────────────────────┘
```

### 3.2 具体开销

| 开销类型 | 估算占比 | 说明 |
|---------|---------|------|
| 数据提取 (DuckDB→内存) | 20-30% | 6M 行 × 10 列 ≈ 240MB 复制 |
| 中间结果物化 | 30-40% | 每个算子输出需要新分配内存 |
| Hash 表操作 | 20-30% | unordered_map 缓存不友好 |
| ThunderDuck 算子 | 10-20% | 实际计算时间 |

## 四、优化建议

### 4.1 短期优化 (1-2 周)

#### 优化 1: 消除不必要的中间 vector

```cpp
// 当前: 每个过滤步骤创建新 vector
std::vector<int32_t> filtered_keys;
for (i = 0; i < n; ++i) {
    if (condition) filtered_keys.push_back(key[i]);
}

// 优化: 使用选择向量，避免数据复制
SelectionVector sel(n);
size_t count = filter_op->Execute(input, condition, sel.data());
sel.resize(count);
// 后续算子直接使用 sel，不复制实际数据
```

**预期收益**: 减少 30-50% 的内存分配

#### 优化 2: 替换 unordered_map 为数组

```cpp
// 当前: hash 表聚合
std::unordered_map<int32_t, int64_t> results;
results[key] += value;  // hash 查找 + 可能的 rehash

// 优化: 对于小基数分组，使用直接数组
// Q1: 6 个分组 -> int64_t results[6]
// Q3: orderkey 作为 key -> 预分配数组 + 排序后二分
std::vector<int64_t> results(max_key + 1);
results[key] += value;  // 直接索引
```

**预期收益**: 聚合阶段 2-5x 加速

#### 优化 3: 融合 Filter + Join

```cpp
// 当前: 先 Filter 创建中间数组，再 Join
auto filtered = filter(orders, date_condition);  // 创建中间数组
auto joined = join(filtered, lineitem);

// 优化: 在 Join 内部做条件过滤
auto joined = join_with_filter(orders, lineitem, date_condition);
// Join 时跳过不满足条件的行，无中间数组
```

**预期收益**: 减少一次 O(n) 遍历

### 4.2 中期优化 (1-2 月)

#### 优化 4: 实现真正的晚期物化

```
┌─────────────────────────────────────────────────────────┐
│                   晚期物化流水线                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   原始数据 (指针)                                        │
│       │                                                  │
│   Filter ──▶ SelectionVector (只存索引)                 │
│       │                                                  │
│   Join ──▶ SelectionPair (左右索引对)                   │
│       │                                                  │
│   Aggregate ──▶ 根据索引访问原始数据                    │
│       │                                                  │
│   最终物化 (仅需要的列)                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 优化 5: DuckDB Extension 深度集成

```cpp
// 注册 ThunderDuck 执行钩子
class ThunderDuckExtension : public duckdb::Extension {
    void OnOperatorExecution(PhysicalOperator* op, ExecutionContext& ctx) {
        if (CanAccelerate(op)) {
            // 直接在 DuckDB 的 Vector 上执行 ThunderDuck 算子
            // 零拷贝，无中间结果
            ExecuteWithThunderDuck(op, ctx);
        }
    }
};
```

### 4.3 长期优化 (3+ 月)

#### 优化 6: 查询级别代码生成

针对高频查询模式生成优化代码：

```cpp
// Q6 专用融合算子
int64_t q6_fused_kernel(
    const int32_t* shipdate,
    const int64_t* discount,
    const int64_t* quantity,
    const int64_t* extprice,
    size_t n
) {
    // 单遍扫描，SIMD 并行
    // 无任何中间结果
}
```

## 五、优先级排序

| 优先级 | 优化项 | 预期收益 | 工作量 |
|--------|--------|---------|--------|
| P0 | 选择向量替换中间 vector | 1.3-1.5x | 3 天 |
| P1 | 数组替换 hash 表 | 1.2-1.5x | 2 天 |
| P2 | Filter + Join 融合 | 1.1-1.3x | 5 天 |
| P3 | 晚期物化流水线 | 1.5-2.0x | 2 周 |
| P4 | DuckDB Extension | 2.0-3.0x | 1 月 |

## 六、预期目标

完成 P0-P2 优化后:
- Category A 平均加速比: 1.5x → 2.0x
- 几何平均加速比: 0.84x → 1.3x
- Q3/Q5/Q9 从 0.3x 提升到 1.2x+

---

*ThunderDuck V23 性能分析 - 优化路线图*
