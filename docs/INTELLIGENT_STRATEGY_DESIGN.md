# ThunderDuck 智能策略选择系统设计

> **版本**: 2.0 | **日期**: 2026-01-25

## 一、设计目标

将固化的阈值判断升级为基于**数据统计信息**的智能路由系统：

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  SQL 查询       │ ──► │  统计信息收集    │ ──► │  智能策略选择   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        ▼                                 ▼                                 ▼
                 ┌─────────────┐                  ┌─────────────┐                  ┌─────────────┐
                 │ CPU v3/v4/v5│                  │ GPU Metal   │                  │ 组合策略    │
                 │ SIMD/MT     │                  │ UMA         │                  │ Radix+GPU   │
                 └─────────────┘                  └─────────────┘                  └─────────────┘
```

---

## 二、核心数据结构

### 2.1 表统计信息 (TableStatistics)

```cpp
// include/thunderduck/statistics.h

struct ColumnStatistics {
    // 基础统计
    size_t row_count;              // 行数
    size_t null_count;             // NULL 数量
    size_t distinct_count;         // 唯一值数量 (基数)

    // 值域统计
    int64_t min_value;             // 最小值
    int64_t max_value;             // 最大值
    double avg_value;              // 平均值

    // 分布统计
    float cardinality_ratio;       // distinct_count / row_count
    float skewness;                // 数据倾斜度 (-1 到 1)

    // 直方图 (等宽)
    static constexpr int HISTOGRAM_BUCKETS = 64;
    uint32_t histogram[HISTOGRAM_BUCKETS];
    int64_t histogram_min;
    int64_t histogram_max;

    // 采样 Top-N 频繁值
    static constexpr int TOP_N = 16;
    int64_t frequent_values[TOP_N];
    uint32_t frequent_counts[TOP_N];

    // 元数据
    uint64_t last_update_time;     // 最后更新时间
    size_t sample_size;            // 采样大小
};

struct TableStatistics {
    std::string table_name;
    size_t row_count;
    size_t column_count;
    std::vector<ColumnStatistics> columns;

    // 表级统计
    size_t avg_row_size;           // 平均行大小
    bool is_sorted;                // 是否有序
    int sort_column;               // 排序列 (-1 = 无)
};
```

### 2.2 查询特征 (QueryCharacteristics)

```cpp
struct JoinCharacteristics {
    size_t build_count;
    size_t probe_count;
    float build_cardinality;       // build 表基数比
    float probe_cardinality;       // probe 表基数比
    float estimated_selectivity;   // 预估选择率 (匹配率)
    float key_overlap_ratio;       // 键重叠率
    bool build_is_unique;          // build 键是否唯一
    bool keys_are_sequential;      // 键是否连续
};

struct FilterCharacteristics {
    size_t row_count;
    float estimated_selectivity;   // 基于直方图估算
    float value_in_range_ratio;    // 过滤值在值域中的位置
    CompareOp op;
};

struct AggregateCharacteristics {
    size_t row_count;
    size_t group_count;            // 分组数 (0 = 全表聚合)
    float group_cardinality;       // group_count / row_count
    AggregateType agg_type;
};

struct TopKCharacteristics {
    size_t row_count;
    size_t k;
    float cardinality_ratio;       // 数据基数比
    bool data_is_sorted;           // 数据是否已排序
};
```

---

## 三、算子执行路径

### 3.1 Join 执行路径

```cpp
enum class JoinExecutionPath {
    // CPU 路径
    CPU_V3_RADIX16,          // v3: 16 分区, 适合小数据
    CPU_V3_RADIX256,         // v3: 256 分区, 适合中等数据
    CPU_V4_BLOOM,            // v4: Bloom 预过滤, 低选择率
    CPU_V4_RADIX256,         // v4: 256 分区优化版
    CPU_V5_MULTITHREAD,      // v5: 多线程并行

    // GPU 路径
    GPU_UMA_DIRECT,          // GPU: 直接探测, 最佳场景
    GPU_UMA_BATCHED,         // GPU: 分批处理, 超大数据

    // 组合路径
    HYBRID_BLOOM_GPU,        // CPU Bloom 过滤 + GPU 探测
    HYBRID_RADIX_GPU,        // CPU Radix 分区 + GPU 分区探测

    // NPU 路径
    NPU_BLOOM_FILTER,        // NPU 加速 Bloom 计算
};
```

### 3.2 Filter 执行路径

```cpp
enum class FilterExecutionPath {
    CPU_V3_SIMD,             // v3: 单线程 SIMD
    CPU_V5_MULTITHREAD,      // v5: 多线程 SIMD
    GPU_ATOMIC,              // GPU: 原子计数版 (低选择率)
    GPU_SCAN,                // GPU: 前缀和版 (高选择率)
};
```

### 3.3 TopK 执行路径

```cpp
enum class TopKExecutionPath {
    CPU_V3_HEAP,             // v3: 堆排序, 小 K
    CPU_V4_SAMPLE,           // v4: 采样预过滤
    CPU_V5_COUNT,            // v5: 计数排序, 低基数
    GPU_FILTER,              // GPU: 过滤法, 小 K
    GPU_BITONIC,             // GPU: Bitonic 排序, 大 K
};
```

---

## 四、智能选择算法

### 4.1 Join 路径选择

```cpp
JoinExecutionPath selectJoinPath(const JoinCharacteristics& chars) {
    const size_t build = chars.build_count;
    const size_t probe = chars.probe_count;
    const float selectivity = chars.estimated_selectivity;
    const float build_card = chars.build_cardinality;

    // ============================================================
    // 规则 1: 小数据量 → CPU v3
    // ============================================================
    if (probe < 100000) {
        return JoinExecutionPath::CPU_V3_RADIX16;
    }

    // ============================================================
    // 规则 2: 低选择率 (< 10% 匹配) → Bloom 预过滤
    // ============================================================
    if (selectivity > 0 && selectivity < 0.1f) {
        if (probe > 10000000 && gpu_available) {
            return JoinExecutionPath::HYBRID_BLOOM_GPU;  // Bloom + GPU
        }
        return JoinExecutionPath::CPU_V4_BLOOM;
    }

    // ============================================================
    // 规则 3: 中等数据量 + GPU 可用 → GPU 直接探测
    // ============================================================
    if (gpu_available && probe >= 500000 && probe <= 50000000) {
        // 最佳 GPU 场景: 1M-10M probe
        if (probe >= 1000000 && probe <= 10000000) {
            return JoinExecutionPath::GPU_UMA_DIRECT;
        }
        // 次优 GPU 场景
        return JoinExecutionPath::GPU_UMA_DIRECT;
    }

    // ============================================================
    // 规则 4: 超大数据量 → GPU 分批 或 Radix+GPU
    // ============================================================
    if (probe > 50000000) {
        if (gpu_available) {
            // 分批处理减少内存压力
            return JoinExecutionPath::GPU_UMA_BATCHED;
        }
        return JoinExecutionPath::CPU_V5_MULTITHREAD;
    }

    // ============================================================
    // 规则 5: 高基数 build 表 → 256 分区优化
    // ============================================================
    if (build_card > 0.8f && build > 100000) {
        return JoinExecutionPath::CPU_V4_RADIX256;
    }

    // ============================================================
    // 默认: CPU v3 256 分区
    // ============================================================
    return JoinExecutionPath::CPU_V3_RADIX256;
}
```

### 4.2 Filter 路径选择

```cpp
FilterExecutionPath selectFilterPath(const FilterCharacteristics& chars) {
    const size_t rows = chars.row_count;
    const float selectivity = chars.estimated_selectivity;

    // 小数据量: 单线程 SIMD
    if (rows < 5000000) {
        return FilterExecutionPath::CPU_V3_SIMD;
    }

    // 中等数据量: 多线程
    if (rows < 10000000) {
        return FilterExecutionPath::CPU_V5_MULTITHREAD;
    }

    // 大数据量 + GPU 可用
    if (gpu_available) {
        // 低选择率: GPU 原子版更优
        if (selectivity > 0 && selectivity < 0.3f) {
            return FilterExecutionPath::GPU_ATOMIC;
        }
        // 高选择率: GPU 前缀和版
        return FilterExecutionPath::GPU_SCAN;
    }

    // GPU 不可用: 多线程
    return FilterExecutionPath::CPU_V5_MULTITHREAD;
}
```

### 4.3 TopK 路径选择

```cpp
TopKExecutionPath selectTopKPath(const TopKCharacteristics& chars) {
    const size_t n = chars.row_count;
    const size_t k = chars.k;
    const float cardinality = chars.cardinality_ratio;

    // 低基数数据: 计数排序
    if (cardinality > 0 && cardinality < 0.01f) {
        return TopKExecutionPath::CPU_V5_COUNT;
    }

    // 小 K: 采样预过滤
    if (k < 1000) {
        if (n > 50000000 && gpu_available) {
            return TopKExecutionPath::GPU_FILTER;
        }
        return TopKExecutionPath::CPU_V4_SAMPLE;
    }

    // 大 K: 考虑完整排序
    if (k > 10000) {
        if (n > 10000000 && gpu_available) {
            return TopKExecutionPath::GPU_BITONIC;
        }
    }

    // 默认: 采样预过滤
    return TopKExecutionPath::CPU_V4_SAMPLE;
}
```

---

## 五、统计信息收集

### 5.1 采样策略

```cpp
class StatisticsCollector {
public:
    // 采样比例: 根据数据量自适应
    static float getSampleRatio(size_t row_count) {
        if (row_count < 10000) return 1.0f;        // 全量
        if (row_count < 100000) return 0.1f;       // 10%
        if (row_count < 1000000) return 0.01f;     // 1%
        return 0.001f;                              // 0.1%
    }

    // 收集列统计信息
    ColumnStatistics collectColumnStats(
        const int32_t* data, size_t count,
        float sample_ratio = -1.0f);

    // 估算选择率 (基于直方图)
    float estimateSelectivity(
        const ColumnStatistics& stats,
        CompareOp op, int64_t value);

    // 估算 Join 选择率
    float estimateJoinSelectivity(
        const ColumnStatistics& build_stats,
        const ColumnStatistics& probe_stats);
};
```

### 5.2 直方图选择率估算

```cpp
float estimateSelectivity(const ColumnStatistics& stats,
                          CompareOp op, int64_t value) {
    // 值域外检查
    if (value < stats.min_value) {
        return (op == CompareOp::LT || op == CompareOp::LE) ? 0.0f : 1.0f;
    }
    if (value > stats.max_value) {
        return (op == CompareOp::GT || op == CompareOp::GE) ? 0.0f : 1.0f;
    }

    // 计算 value 在直方图中的位置
    int64_t range = stats.histogram_max - stats.histogram_min;
    int bucket = (value - stats.histogram_min) * HISTOGRAM_BUCKETS / range;
    bucket = std::clamp(bucket, 0, HISTOGRAM_BUCKETS - 1);

    // 累积分布
    uint64_t total = 0, cumulative = 0;
    for (int i = 0; i < HISTOGRAM_BUCKETS; i++) {
        total += stats.histogram[i];
        if (i < bucket) cumulative += stats.histogram[i];
    }

    float cdf = (float)cumulative / total;

    switch (op) {
        case CompareOp::LT: return cdf;
        case CompareOp::LE: return cdf + (float)stats.histogram[bucket] / total;
        case CompareOp::GT: return 1.0f - cdf - (float)stats.histogram[bucket] / total;
        case CompareOp::GE: return 1.0f - cdf;
        case CompareOp::EQ: return 1.0f / stats.distinct_count;  // 假设均匀分布
        case CompareOp::NE: return 1.0f - 1.0f / stats.distinct_count;
    }
    return 0.5f;  // 未知
}
```

---

## 六、API 设计

### 6.1 新版策略选择器

```cpp
// include/thunderduck/query_optimizer.h

class QueryOptimizer {
public:
    static QueryOptimizer& instance();

    // ==================== 统计信息管理 ====================

    // 注册表统计信息
    void registerTableStats(const std::string& table,
                            const TableStatistics& stats);

    // 更新列统计信息 (增量)
    void updateColumnStats(const std::string& table, int column,
                           const int32_t* data, size_t count);

    // 清除统计信息
    void clearStats(const std::string& table = "");

    // ==================== 路径选择 ====================

    // Join 路径选择
    JoinExecutionPath selectJoinPath(
        const std::string& build_table, int build_key_col,
        const std::string& probe_table, int probe_key_col);

    // 直接传入特征
    JoinExecutionPath selectJoinPath(const JoinCharacteristics& chars);

    // Filter 路径选择
    FilterExecutionPath selectFilterPath(
        const std::string& table, int column,
        CompareOp op, int64_t value);

    FilterExecutionPath selectFilterPath(const FilterCharacteristics& chars);

    // TopK 路径选择
    TopKExecutionPath selectTopKPath(
        const std::string& table, int column, size_t k);

    TopKExecutionPath selectTopKPath(const TopKCharacteristics& chars);

    // ==================== 执行接口 ====================

    // 自动选择最优路径执行 Join
    size_t executeJoin(
        const int32_t* build_keys, size_t build_count,
        const int32_t* probe_keys, size_t probe_count,
        JoinResult* result,
        const JoinCharacteristics* chars = nullptr);  // 可选传入特征

    // 自动选择最优路径执行 Filter
    size_t executeFilter(
        const int32_t* data, size_t count,
        CompareOp op, int32_t value,
        uint32_t* out_indices,
        const FilterCharacteristics* chars = nullptr);

    // ==================== 调试接口 ====================

    // 获取决策原因
    std::string getLastDecisionReason() const;

    // 获取路径性能统计
    struct PathPerformance {
        uint64_t call_count;
        double total_time_ms;
        double avg_throughput;
    };
    PathPerformance getPathPerformance(JoinExecutionPath path) const;

private:
    std::unordered_map<std::string, TableStatistics> table_stats_;

    // 运行时性能统计 (用于自学习)
    std::unordered_map<int, PathPerformance> join_perf_;
    std::unordered_map<int, PathPerformance> filter_perf_;
};
```

### 6.2 使用示例

```cpp
// 示例 1: 注册统计信息后执行 Join
auto& opt = QueryOptimizer::instance();

// 注册表统计
TableStatistics orders_stats = collectTableStats(orders_table);
TableStatistics customers_stats = collectTableStats(customers_table);
opt.registerTableStats("orders", orders_stats);
opt.registerTableStats("customers", customers_stats);

// 执行 Join (自动选择最优路径)
JoinResult result;
size_t matches = opt.executeJoin(
    customer_ids, customer_count,
    order_customer_ids, order_count,
    &result);

std::cout << "Decision: " << opt.getLastDecisionReason() << std::endl;
// 输出: "Selected GPU_UMA_DIRECT: probe=10M in optimal GPU range,
//        build cardinality=0.95 (high uniqueness)"

// 示例 2: 直接传入特征
JoinCharacteristics chars;
chars.build_count = 1000000;
chars.probe_count = 10000000;
chars.build_cardinality = 0.95f;
chars.estimated_selectivity = 0.1f;

JoinExecutionPath path = opt.selectJoinPath(chars);
// 返回: HYBRID_BLOOM_GPU (低选择率 + 大数据量)
```

---

## 七、实现计划

### Phase 1: 统计信息框架 (1-2 天)

| 任务 | 文件 | 说明 |
|------|------|------|
| 定义统计结构 | `include/thunderduck/statistics.h` | ColumnStatistics, TableStatistics |
| 统计收集器 | `src/core/statistics_collector.cpp` | 采样、直方图、基数估算 |
| 选择率估算 | `src/core/selectivity_estimator.cpp` | 基于直方图 |

### Phase 2: 路径选择器 (1-2 天)

| 任务 | 文件 | 说明 |
|------|------|------|
| 路径枚举 | `include/thunderduck/execution_path.h` | 定义所有执行路径 |
| 查询优化器 | `src/core/query_optimizer.cpp` | 路径选择逻辑 |
| 单元测试 | `tests/test_optimizer.cpp` | 验证选择正确性 |

### Phase 3: 执行路由 (2-3 天)

| 任务 | 文件 | 说明 |
|------|------|------|
| Join 路由 | `src/operators/join/join_router.cpp` | 根据路径调用对应实现 |
| Filter 路由 | `src/operators/filter/filter_router.cpp` | 根据路径调用对应实现 |
| TopK 路由 | `src/operators/sort/topk_router.cpp` | 根据路径调用对应实现 |

### Phase 4: 性能反馈 (可选)

| 任务 | 文件 | 说明 |
|------|------|------|
| 运行时统计 | `src/core/runtime_stats.cpp` | 记录各路径性能 |
| 自适应调整 | `src/core/adaptive_tuning.cpp` | 基于历史调整阈值 |

---

## 八、路径选择决策树

```
                              ┌─────────────────┐
                              │   Join 查询     │
                              └────────┬────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  probe_count < 100K ?   │
                          └────────────┬────────────┘
                                  YES  │  NO
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            CPU_V3_RADIX16                    ┌───────────────────────┐
                                             │ selectivity < 10% ?   │
                                             └───────────┬───────────┘
                                                    YES  │  NO
                                  ┌──────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                      ┌───────────────────┐                        ┌───────────────────────┐
                      │ probe > 10M ?     │                        │ 500K < probe < 50M ?  │
                      │ & GPU available   │                        │ & GPU available       │
                      └─────────┬─────────┘                        └───────────┬───────────┘
                           YES  │  NO                                     YES  │  NO
                    ┌──────────┴──────────┐                     ┌─────────────┴─────────────┐
                    ▼                     ▼                     ▼                           ▼
           HYBRID_BLOOM_GPU      CPU_V4_BLOOM            GPU_UMA_DIRECT            CPU_V4_RADIX256
```

---

## 九、预期收益

| 场景 | 当前方案 | 智能选择方案 | 提升 |
|------|---------|-------------|------|
| 小数据 Join | 可能误用 GPU | CPU v3 | 避免 GPU 开销 |
| 低选择率 Join | GPU 直接探测 | Bloom + GPU | 1.3-1.5x |
| 高基数 Join | 通用路径 | Radix256 优化 | 1.2-1.5x |
| 大数据 Filter | 单线程 | 多线程自动启用 | 2x |
| 低基数 TopK | 堆排序 | 计数排序 | 2-3x |

---

## 十、总结

智能策略选择系统的核心价值：

1. **数据驱动**: 基于实际数据统计而非固定阈值
2. **多路径可选**: 充分利用已实现的 v3/v4/v5/GPU 版本
3. **组合策略**: 支持 Bloom+GPU、Radix+GPU 等混合方案
4. **可扩展**: 新增算子版本只需注册路径
5. **可观测**: 提供决策原因和性能统计
