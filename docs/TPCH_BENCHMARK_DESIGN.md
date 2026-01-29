# ThunderDuck TPC-H 完整基准测试系统设计

> 版本: 1.0 | 日期: 2026-01-28

## 一、目标

构建独立的 ThunderDuck TPC-H 基准测试系统：
1. 完整运行所有 22 条 TPC-H SQL
2. 对比 DuckDB 原生执行 vs ThunderDuck 加速执行
3. 使用 DuckDB 作为存储和 SQL 解析，ThunderDuck 算子做计算加速

## 二、架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    TPC-H Benchmark                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ DataLoader  │───▶│  Executor   │───▶│   Report    │ │
│  │ (DuckDB)    │    │ (Hybrid)    │    │ (Markdown)  │ │
│  └─────────────┘    └──────┬──────┘    └─────────────┘ │
│                            │                            │
│         ┌──────────────────┼──────────────────┐        │
│         ▼                  ▼                  ▼        │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   DuckDB    │    │ ThunderDuck │    │  Verify   │  │
│  │  Baseline   │    │  Optimized  │    │  Results  │  │
│  └─────────────┘    └─────────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 三、查询分类

### A. 完全可优化 (10 条) - 预期加速 1.5-3x
| 查询 | 优化算子 | 说明 |
|------|----------|------|
| Q1 | GROUP BY V15 + SUM | 分组聚合 |
| Q3 | JOIN V14 + GROUP BY | 多表连接+聚合 |
| Q5 | JOIN V14 + GROUP BY | 区域销售 |
| Q6 | Filter V19.1 + SUM V21 | 简单过滤聚合 |
| Q7 | JOIN V14 + GROUP BY | 国家间贸易 |
| Q9 | JOIN V14 + GROUP BY | 产品利润 |
| Q10 | JOIN V14 + GROUP BY | 客户退货 |
| Q12 | JOIN V14 + CASE | 运输模式 |
| Q14 | Filter + 条件聚合 | 促销效果 |
| Q18 | GROUP BY + HAVING | 大订单客户 |

### B. 部分可优化 (6 条) - 预期加速 1.0-1.5x
Q2, Q4, Q11, Q15, Q16, Q19 - 包含子查询，部分算子可优化

### C. DuckDB 回退 (6 条) - 预期 ~1.0x
Q8, Q13, Q17, Q20, Q21, Q22 - 复杂子查询/EXISTS，使用 DuckDB 执行

## 四、文件结构

```
benchmark/tpch/
├── tpch_benchmark_main.cpp     # 主入口
├── tpch_data_loader.h/.cpp     # 数据加载器
├── tpch_executor.h/.cpp        # 查询执行器
├── tpch_operators.h/.cpp       # 算子封装
├── tpch_queries.h/.cpp         # 22条查询实现
└── tpch_report.h/.cpp          # 报告生成
```

## 五、关键实现

### 5.1 数据加载器

```cpp
class TPCHDataLoader {
public:
    void generate_data(int scale_factor);  // dbgen(sf)
    void extract_all_tables();             // 提取到内存数组

    // 8 张表的列式数据
    LineitemColumns lineitem;   // 6M × SF 行
    OrdersColumns orders;       // 1.5M × SF 行
    CustomerColumns customer;   // 150K × SF 行
    // ... part, supplier, partsupp, nation, region
};
```

### 5.2 查询执行模式

```cpp
QueryResult run_query(const std::string& qid) {
    // 1. DuckDB 基线
    double duckdb_ms = measure([&]() {
        con.Query(TPCH_SQL[qid]);
    });

    // 2. ThunderDuck 执行
    double td_ms = measure([&]() {
        // 使用 ThunderDuck 算子实现等价逻辑
    });

    // 3. 结果验证
    bool match = validate_results(...);

    return {qid, duckdb_ms, td_ms, duckdb_ms/td_ms, match};
}
```

### 5.3 Q6 示例实现 (最佳案例)

```cpp
// Q6: SELECT SUM(l_extendedprice * l_discount) FROM lineitem
//     WHERE l_shipdate BETWEEN '1994-01-01' AND '1994-12-31'
//       AND l_discount BETWEEN 0.05 AND 0.07
//       AND l_quantity < 24

double run_q6_thunderduck() {
    auto& li = loader.lineitem;
    __int128 revenue = 0;

    #pragma omp parallel for reduction(+:revenue)
    for (size_t i = 0; i < li.count; i += 4) {
        int32x4_t dates = vld1q_s32(&li.l_shipdate[i]);
        // 向量化条件判断 + 选择性聚合
        // ...
    }
    return revenue / 1e8;  // 定点数转换
}
```

## 六、性能测量

### 6.1 IQR 中位数测量

```cpp
template<typename Func>
double measure_median_iqr(Func&& func, size_t iters = 10, size_t warmup = 2) {
    std::vector<double> times;

    // 预热
    for (size_t i = 0; i < warmup; ++i) func();

    // 测量
    for (size_t i = 0; i < iters; ++i) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(end - start).count());
    }

    // IQR 剔除异常值
    std::sort(times.begin(), times.end());
    double q1 = times[n / 4];
    double q3 = times[n * 3 / 4];
    double iqr = q3 - q1;
    // 剔除 < Q1-1.5*IQR 或 > Q3+1.5*IQR 的值

    return median;
}
```

## 七、使用方法

```bash
# 构建
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make tpch_benchmark

# 运行完整测试
./tpch_benchmark --sf 1

# 运行指定类别
./tpch_benchmark --sf 1 --category A

# 运行单个查询
./tpch_benchmark --query Q6

# 输出到指定文件
./tpch_benchmark --sf 10 --output report.md
```

## 八、预期输出

```
===== ThunderDuck TPC-H Benchmark Report =====
Scale Factor: 1 | Date: 2026-01-28

Query    DuckDB(ms)  ThunderDuck(ms)   Speedup  Status
------------------------------------------------------
Q1           53.09            21.15     2.51x      OK
Q2           18.45            18.45     1.00x      OK
Q3           22.78            15.21     1.50x      OK
...
Q22           8.12             8.12     1.00x      OK
------------------------------------------------------

Summary:
- Total Queries: 22
- Faster: 16 (72.7%)
- Same: 6
- Slower: 0
- Geometric Mean Speedup: 1.67x
```

## 九、成功标准

1. 22 条 TPC-H 查询全部运行通过
2. Category A 查询平均加速比 ≥ 1.5x
3. 总体几何平均加速比 ≥ 1.3x
4. SF=1 完整测试 < 60 秒

## 十、后续优化方向

1. **Q6 深度优化**: 使用 ARM Neon SIMD 完全向量化
2. **JOIN 优化**: 探索 GPU INNER JOIN 用于大表
3. **字符串过滤**: SIMD 加速 LIKE 操作
4. **并行执行**: 多线程执行独立的子查询
