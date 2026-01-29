# Q17 优化方案设计

> **版本**: 1.0 | **日期**: 2026-01-29 | **目标**: 相关子查询通用优化

## 一、Q17 深度分析

### 1.1 原始 SQL

```sql
SELECT SUM(l_extendedprice) / 7.0 as avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity)
    FROM lineitem
    WHERE l_partkey = p_partkey  -- 相关条件
  );
```

### 1.2 查询结构分解

```
┌─────────────────────────────────────────────────────────────────┐
│                        Q17 执行流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 外层查询                                                     │
│     ├── FROM: lineitem JOIN part ON l_partkey = p_partkey       │
│     ├── WHERE: p_brand = 'Brand#23' AND p_container = 'MED BOX' │
│     └── WHERE: l_quantity < [相关子查询结果]                     │
│                                                                 │
│  2. 相关子查询 (对每个匹配的 partkey 执行一次)                    │
│     └── SELECT 0.2 * AVG(l_quantity)                            │
│         FROM lineitem                                           │
│         WHERE l_partkey = [外层的 p_partkey]                     │
│                                                                 │
│  3. 聚合                                                         │
│     └── SUM(l_extendedprice) / 7.0                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 性能瓶颈分析

| 瓶颈 | 描述 | 影响 |
|------|------|------|
| **重复扫描** | 对每个匹配的 part，重新扫描 lineitem 计算 AVG | O(P × L) 复杂度 |
| **相关性** | 子查询依赖外层的 p_partkey，无法独立执行 | 阻止并行化 |
| **延迟物化** | 无法预先计算子查询结果 | 每次都要重算 |

**数据规模估算 (SF=1)**:
- PART 匹配 `Brand#23 + MED BOX`: ~200,000 / (25 brands × 40 containers) ≈ 200 行
- 每个 partkey 平均有 ~30 条 lineitem
- 朴素执行: 200 × 6M = 12 亿次比较 (不可接受)

---

## 二、优化策略

### 2.1 核心思路: 子查询解关联 (Decorrelation)

将相关子查询转换为独立的预计算 + Join：

```sql
-- 原始 (相关子查询)
WHERE l_quantity < (SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)

-- 转换后 (解关联)
WITH part_avg AS (
  SELECT l_partkey, 0.2 * AVG(l_quantity) as threshold
  FROM lineitem
  GROUP BY l_partkey
)
SELECT SUM(l.l_extendedprice) / 7.0
FROM lineitem l, part p, part_avg pa
WHERE p.p_partkey = l.l_partkey
  AND p.p_partkey = pa.l_partkey
  AND p.p_brand = 'Brand#23'
  AND p.p_container = 'MED BOX'
  AND l.l_quantity < pa.threshold;
```

### 2.2 执行计划对比

**朴素执行**:
```
Aggregate (SUM)
└── Filter (l_quantity < correlated_subquery)
    └── Hash Join (lineitem × part)
        ├── Seq Scan part (brand='Brand#23', container='MED BOX')
        └── Seq Scan lineitem
            └── [对每行] Subquery Scan lineitem (WHERE l_partkey = ?)
```

**优化执行**:
```
Aggregate (SUM)
└── Hash Join (l_quantity < threshold)
    ├── Hash Join (lineitem × part)
    │   ├── Seq Scan part (brand='Brand#23', container='MED BOX')
    │   └── Seq Scan lineitem
    └── Pre-computed part_avg (一次扫描)
```

---

## 三、通用化设计: CorrelatedSubqueryOptimizer

### 3.1 适用模式

该优化器可处理以下相关子查询模式：

| 模式 | 示例 | 适用查询 |
|------|------|---------|
| **Scalar Aggregate** | `WHERE col < (SELECT AVG(x) FROM T WHERE T.key = outer.key)` | Q17, Q20 |
| **EXISTS/NOT EXISTS** | `WHERE EXISTS (SELECT 1 FROM T WHERE T.key = outer.key)` | Q4, Q21, Q22 |
| **IN/NOT IN** | `WHERE col IN (SELECT x FROM T WHERE T.key = outer.key)` | Q16, Q18 |
| **Quantified Comparison** | `WHERE col > ALL (SELECT x FROM T WHERE ...)` | 少见 |

### 3.2 类设计

```cpp
// include/thunderduck/correlated_subquery.h

namespace thunderduck {
namespace operators {

/**
 * 相关子查询优化器 - 通用化设计
 *
 * 支持的聚合函数: AVG, SUM, COUNT, MIN, MAX
 * 支持的比较运算: <, <=, >, >=, =, <>
 * 支持的相关模式: 等值连接 (outer.key = inner.key)
 */

// 聚合类型枚举
enum class AggregateType {
    AVG,
    SUM,
    COUNT,
    MIN,
    MAX,
    COUNT_DISTINCT
};

// 比较运算符枚举
enum class CompareOp {
    LT,     // <
    LE,     // <=
    GT,     // >
    GE,     // >=
    EQ,     // =
    NE      // <>
};

// 预计算结果存储
template<typename KeyType, typename ValueType>
class PrecomputedAggregates {
public:
    // 构建: 一次扫描计算所有分组的聚合值
    void build(const KeyType* keys,
               const ValueType* values,
               size_t count,
               AggregateType agg_type,
               double scale_factor = 1.0);  // 用于 0.2 * AVG 这种情况

    // 查询: O(1) 获取指定 key 的聚合结果
    std::optional<ValueType> get(KeyType key) const;

    // 批量查询: 返回所有 key 的结果
    void batch_get(const KeyType* keys,
                   size_t count,
                   ValueType* results,
                   ValueType default_value = ValueType{}) const;

    // 获取内部 hash table (用于 Join)
    const auto& get_table() const { return table_; }

private:
    // 使用紧凑 Hash Table 存储
    CompactHashTable<KeyType, ValueType> table_;

    // 对于 AVG，需要同时存储 sum 和 count
    struct AvgState {
        ValueType sum = 0;
        int64_t count = 0;
        ValueType avg() const { return count > 0 ? sum / count : 0; }
    };
    CompactHashTable<KeyType, AvgState> avg_table_;
};

/**
 * 相关子查询转换器
 *
 * 将 WHERE col <op> (SELECT agg(x) FROM T WHERE T.key = outer.key)
 * 转换为预计算 + Join
 */
template<typename OuterKey, typename InnerKey, typename ValueType>
class CorrelatedSubqueryTransformer {
public:
    struct Config {
        AggregateType agg_type;
        CompareOp compare_op;
        double scale_factor;      // 0.2 * AVG 中的 0.2
        bool allow_null_mismatch; // 子查询返回 NULL 时的处理
    };

    CorrelatedSubqueryTransformer(const Config& config);

    // Step 1: 预计算子查询结果
    void precompute(const InnerKey* inner_keys,
                    const ValueType* inner_values,
                    size_t inner_count);

    // Step 2: 应用到外层查询，返回满足条件的行索引
    std::vector<uint32_t> apply(
        const OuterKey* outer_keys,      // 外层的关联键
        const ValueType* outer_values,   // 外层的比较值 (如 l_quantity)
        size_t outer_count
    );

    // Step 3 (可选): 获取每行对应的子查询结果 (用于调试或后续计算)
    void get_subquery_results(
        const OuterKey* outer_keys,
        size_t count,
        ValueType* results
    );

private:
    Config config_;
    PrecomputedAggregates<InnerKey, ValueType> precomputed_;
};

/**
 * Q17 专用优化实现
 *
 * 完全解关联版本，一次扫描完成
 */
class Q17Optimizer {
public:
    struct Result {
        int64_t sum_extendedprice;  // 定点数
        double avg_yearly;          // 最终结果
    };

    // 单次调用完成 Q17
    static Result execute(
        // Part 表
        const int32_t* p_partkey,
        const std::vector<std::string>& p_brand,
        const std::vector<std::string>& p_container,
        size_t part_count,
        // Lineitem 表
        const int32_t* l_partkey,
        const int64_t* l_quantity,
        const int64_t* l_extendedprice,
        size_t lineitem_count,
        // 参数
        const std::string& target_brand,     // "Brand#23"
        const std::string& target_container, // "MED BOX"
        double quantity_factor               // 0.2
    );

private:
    // Phase 1: 找出目标 part 的 partkey 集合
    static std::unordered_set<int32_t> find_target_parts(
        const int32_t* p_partkey,
        const std::vector<std::string>& p_brand,
        const std::vector<std::string>& p_container,
        size_t count,
        const std::string& target_brand,
        const std::string& target_container
    );

    // Phase 2: 计算每个 partkey 的 AVG(l_quantity)
    static CompactHashTable<int32_t, int64_t> compute_quantity_thresholds(
        const int32_t* l_partkey,
        const int64_t* l_quantity,
        size_t count,
        const std::unordered_set<int32_t>& target_parts,
        double quantity_factor
    );

    // Phase 3: 过滤并聚合
    static int64_t filter_and_aggregate(
        const int32_t* l_partkey,
        const int64_t* l_quantity,
        const int64_t* l_extendedprice,
        size_t count,
        const std::unordered_set<int32_t>& target_parts,
        const CompactHashTable<int32_t, int64_t>& thresholds
    );
};

} // namespace operators
} // namespace thunderduck
```

### 3.3 核心算法实现

```cpp
// src/operators/correlated_subquery.cpp

namespace thunderduck {
namespace operators {

// ============================================================================
// PrecomputedAggregates 实现
// ============================================================================

template<typename K, typename V>
void PrecomputedAggregates<K, V>::build(
    const K* keys, const V* values, size_t count,
    AggregateType agg_type, double scale_factor
) {
    switch (agg_type) {
        case AggregateType::AVG: {
            // 两遍扫描: 第一遍计算 sum 和 count，第二遍计算 avg
            avg_table_.reserve(count / 10);  // 估计 10% 的唯一键

            // 第一遍: 累加
            for (size_t i = 0; i < count; ++i) {
                auto& state = avg_table_[keys[i]];
                state.sum += values[i];
                state.count++;
            }

            // 转换为最终结果并应用 scale_factor
            table_.reserve(avg_table_.size());
            for (const auto& [key, state] : avg_table_) {
                V avg = state.count > 0 ? state.sum / state.count : 0;
                table_[key] = static_cast<V>(avg * scale_factor);
            }
            break;
        }

        case AggregateType::SUM: {
            table_.reserve(count / 10);
            for (size_t i = 0; i < count; ++i) {
                table_[keys[i]] += values[i];
            }
            // 应用 scale_factor
            if (scale_factor != 1.0) {
                for (auto& [key, val] : table_) {
                    val = static_cast<V>(val * scale_factor);
                }
            }
            break;
        }

        case AggregateType::COUNT: {
            CompactHashTable<K, int64_t> count_table;
            count_table.reserve(count / 10);
            for (size_t i = 0; i < count; ++i) {
                count_table[keys[i]]++;
            }
            // 转换并应用 scale_factor
            table_.reserve(count_table.size());
            for (const auto& [key, cnt] : count_table) {
                table_[key] = static_cast<V>(cnt * scale_factor);
            }
            break;
        }

        case AggregateType::MIN: {
            table_.reserve(count / 10);
            for (size_t i = 0; i < count; ++i) {
                auto it = table_.find(keys[i]);
                if (it == table_.end()) {
                    table_[keys[i]] = values[i];
                } else if (values[i] < it->second) {
                    it->second = values[i];
                }
            }
            // MIN/MAX 通常不需要 scale_factor，但支持
            if (scale_factor != 1.0) {
                for (auto& [key, val] : table_) {
                    val = static_cast<V>(val * scale_factor);
                }
            }
            break;
        }

        case AggregateType::MAX: {
            table_.reserve(count / 10);
            for (size_t i = 0; i < count; ++i) {
                auto it = table_.find(keys[i]);
                if (it == table_.end()) {
                    table_[keys[i]] = values[i];
                } else if (values[i] > it->second) {
                    it->second = values[i];
                }
            }
            if (scale_factor != 1.0) {
                for (auto& [key, val] : table_) {
                    val = static_cast<V>(val * scale_factor);
                }
            }
            break;
        }
    }
}

// ============================================================================
// CorrelatedSubqueryTransformer 实现
// ============================================================================

template<typename OK, typename IK, typename V>
std::vector<uint32_t> CorrelatedSubqueryTransformer<OK, IK, V>::apply(
    const OK* outer_keys,
    const V* outer_values,
    size_t outer_count
) {
    std::vector<uint32_t> result;
    result.reserve(outer_count / 10);  // 预估 10% 选择率

    const auto& table = precomputed_.get_table();

    // 根据比较运算符选择比较函数
    auto compare = [this](V outer_val, V threshold) -> bool {
        switch (config_.compare_op) {
            case CompareOp::LT: return outer_val < threshold;
            case CompareOp::LE: return outer_val <= threshold;
            case CompareOp::GT: return outer_val > threshold;
            case CompareOp::GE: return outer_val >= threshold;
            case CompareOp::EQ: return outer_val == threshold;
            case CompareOp::NE: return outer_val != threshold;
        }
        return false;
    };

    // 遍历外层并应用条件
    for (size_t i = 0; i < outer_count; ++i) {
        auto it = table.find(outer_keys[i]);
        if (it != table.end()) {
            if (compare(outer_values[i], it->second)) {
                result.push_back(static_cast<uint32_t>(i));
            }
        } else if (config_.allow_null_mismatch) {
            // 子查询返回 NULL (没有匹配行)，根据 SQL 语义决定
            // 默认: NULL 比较返回 UNKNOWN，不满足条件
        }
    }

    return result;
}

// ============================================================================
// Q17Optimizer 实现
// ============================================================================

Q17Optimizer::Result Q17Optimizer::execute(
    const int32_t* p_partkey,
    const std::vector<std::string>& p_brand,
    const std::vector<std::string>& p_container,
    size_t part_count,
    const int32_t* l_partkey,
    const int64_t* l_quantity,
    const int64_t* l_extendedprice,
    size_t lineitem_count,
    const std::string& target_brand,
    const std::string& target_container,
    double quantity_factor
) {
    // ========================================
    // Phase 1: 找出目标 part
    // ========================================
    auto target_parts = find_target_parts(
        p_partkey, p_brand, p_container, part_count,
        target_brand, target_container
    );

    if (target_parts.empty()) {
        return {0, 0.0};
    }

    // ========================================
    // Phase 2: 单遍扫描计算阈值 + 聚合
    // ========================================
    // 优化: 合并 Phase 2 和 Phase 3 为单遍扫描

    // 存储每个 partkey 的 sum 和 count
    struct QuantityState {
        int64_t sum = 0;
        int64_t count = 0;
    };
    CompactHashTable<int32_t, QuantityState> qty_states;
    qty_states.reserve(target_parts.size());

    // 存储满足条件的行 (延迟评估)
    struct PendingRow {
        int64_t quantity;
        int64_t extendedprice;
    };
    CompactHashTable<int32_t, std::vector<PendingRow>> pending_rows;
    pending_rows.reserve(target_parts.size());

    // 单遍扫描 lineitem
    for (size_t i = 0; i < lineitem_count; ++i) {
        int32_t pk = l_partkey[i];

        // 只处理目标 part
        if (target_parts.count(pk) == 0) continue;

        // 累加用于计算 AVG
        auto& state = qty_states[pk];
        state.sum += l_quantity[i];
        state.count++;

        // 记录行数据，稍后评估
        pending_rows[pk].push_back({l_quantity[i], l_extendedprice[i]});
    }

    // ========================================
    // Phase 3: 计算阈值并过滤聚合
    // ========================================
    int64_t total_extendedprice = 0;

    for (auto& [pk, rows] : pending_rows) {
        auto it = qty_states.find(pk);
        if (it == qty_states.end() || it->second.count == 0) continue;

        // 计算阈值: 0.2 * AVG(l_quantity)
        int64_t avg_qty = it->second.sum / it->second.count;
        int64_t threshold = static_cast<int64_t>(avg_qty * quantity_factor);

        // 过滤并聚合
        for (const auto& row : rows) {
            if (row.quantity < threshold) {
                total_extendedprice += row.extendedprice;
            }
        }
    }

    // 返回结果
    Result result;
    result.sum_extendedprice = total_extendedprice;
    result.avg_yearly = static_cast<double>(total_extendedprice) / 7.0 / 10000.0;
    return result;
}

std::unordered_set<int32_t> Q17Optimizer::find_target_parts(
    const int32_t* p_partkey,
    const std::vector<std::string>& p_brand,
    const std::vector<std::string>& p_container,
    size_t count,
    const std::string& target_brand,
    const std::string& target_container
) {
    std::unordered_set<int32_t> result;
    result.reserve(count / 1000);  // 估计 0.1% 选择率

    for (size_t i = 0; i < count; ++i) {
        if (p_brand[i] == target_brand && p_container[i] == target_container) {
            result.insert(p_partkey[i]);
        }
    }

    return result;
}

} // namespace operators
} // namespace thunderduck
```

---

## 四、并行化优化

### 4.1 多线程架构

```cpp
// 8 线程并行版本
class Q17OptimizerParallel {
public:
    static Result execute_parallel(
        /* 参数同上 */
        size_t num_threads = 8
    ) {
        // ========================================
        // Phase 1: 并行查找目标 part (小数据量，单线程即可)
        // ========================================
        auto target_parts = find_target_parts(/* ... */);

        // ========================================
        // Phase 2: 并行扫描 lineitem
        // ========================================

        // 每个线程的局部状态
        struct alignas(128) ThreadLocalState {
            CompactHashTable<int32_t, QuantityState> qty_states;
            CompactHashTable<int32_t, std::vector<PendingRow>> pending_rows;
        };

        std::vector<ThreadLocalState> thread_states(num_threads);
        std::vector<std::thread> threads;

        size_t chunk_size = (lineitem_count + num_threads - 1) / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, lineitem_count);

            threads.emplace_back([&, t, start, end]() {
                auto& state = thread_states[t];

                for (size_t i = start; i < end; ++i) {
                    int32_t pk = l_partkey[i];
                    if (target_parts.count(pk) == 0) continue;

                    auto& qs = state.qty_states[pk];
                    qs.sum += l_quantity[i];
                    qs.count++;

                    state.pending_rows[pk].push_back({l_quantity[i], l_extendedprice[i]});
                }
            });
        }

        for (auto& th : threads) th.join();

        // ========================================
        // Phase 3: 合并线程结果
        // ========================================
        CompactHashTable<int32_t, QuantityState> merged_states;
        CompactHashTable<int32_t, std::vector<PendingRow>> merged_rows;

        for (size_t t = 0; t < num_threads; ++t) {
            for (auto& [pk, qs] : thread_states[t].qty_states) {
                auto& m = merged_states[pk];
                m.sum += qs.sum;
                m.count += qs.count;
            }
            for (auto& [pk, rows] : thread_states[t].pending_rows) {
                auto& m = merged_rows[pk];
                m.insert(m.end(), rows.begin(), rows.end());
            }
        }

        // ========================================
        // Phase 4: 并行过滤聚合
        // ========================================
        std::vector<int32_t> partkeys;
        for (const auto& [pk, _] : merged_rows) {
            partkeys.push_back(pk);
        }

        std::atomic<int64_t> total_price{0};

        size_t pk_chunk = (partkeys.size() + num_threads - 1) / num_threads;
        threads.clear();

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * pk_chunk;
            size_t end = std::min(start + pk_chunk, partkeys.size());

            threads.emplace_back([&, start, end]() {
                int64_t local_sum = 0;

                for (size_t i = start; i < end; ++i) {
                    int32_t pk = partkeys[i];

                    auto qs_it = merged_states.find(pk);
                    if (qs_it == merged_states.end() || qs_it->second.count == 0) continue;

                    int64_t threshold = static_cast<int64_t>(
                        (qs_it->second.sum / qs_it->second.count) * quantity_factor
                    );

                    auto rows_it = merged_rows.find(pk);
                    if (rows_it == merged_rows.end()) continue;

                    for (const auto& row : rows_it->second) {
                        if (row.quantity < threshold) {
                            local_sum += row.extendedprice;
                        }
                    }
                }

                total_price.fetch_add(local_sum, std::memory_order_relaxed);
            });
        }

        for (auto& th : threads) th.join();

        return {total_price.load(), static_cast<double>(total_price.load()) / 7.0 / 10000.0};
    }
};
```

### 4.2 SIMD 优化 (阈值比较)

```cpp
// ARM Neon 向量化阈值比较
#ifdef __aarch64__
#include <arm_neon.h>

inline size_t simd_filter_lt_threshold(
    const int64_t* quantities,
    const int64_t* prices,
    size_t count,
    int64_t threshold,
    int64_t& sum_out
) {
    int64x2_t threshold_vec = vdupq_n_s64(threshold);
    int64x2_t sum_vec = vdupq_n_s64(0);
    size_t match_count = 0;

    size_t i = 0;
    for (; i + 2 <= count; i += 2) {
        int64x2_t qty_vec = vld1q_s64(quantities + i);
        int64x2_t price_vec = vld1q_s64(prices + i);

        // 比较 qty < threshold
        uint64x2_t mask = vcltq_s64(qty_vec, threshold_vec);

        // 条件累加
        int64x2_t masked_price = vandq_s64(vreinterpretq_s64_u64(mask), price_vec);
        sum_vec = vaddq_s64(sum_vec, masked_price);

        // 计数
        match_count += (vgetq_lane_u64(mask, 0) ? 1 : 0) + (vgetq_lane_u64(mask, 1) ? 1 : 0);
    }

    // 水平求和
    sum_out = vgetq_lane_s64(sum_vec, 0) + vgetq_lane_s64(sum_vec, 1);

    // 处理剩余
    for (; i < count; ++i) {
        if (quantities[i] < threshold) {
            sum_out += prices[i];
            match_count++;
        }
    }

    return match_count;
}
#endif
```

---

## 五、通用性验证

### 5.1 适用于其他查询

| 查询 | 模式 | 配置 |
|------|------|------|
| **Q17** | `l_quantity < 0.2 * AVG(l_quantity)` | `{AVG, LT, 0.2}` |
| **Q20** | `ps_availqty > 0.5 * SUM(l_quantity)` | `{SUM, GT, 0.5}` |
| **Q2** | `ps_supplycost = MIN(ps_supplycost)` | `{MIN, EQ, 1.0}` |
| **Q15** | `total_revenue = MAX(total_revenue)` | `{MAX, EQ, 1.0}` |

### 5.2 Q20 适配示例

```cpp
// Q20: ps_availqty > 0.5 * SUM(l_quantity) WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey

// 使用复合键
using Q20Key = std::pair<int32_t, int32_t>;  // (partkey, suppkey)

CorrelatedSubqueryTransformer<Q20Key, Q20Key, int64_t> transformer({
    .agg_type = AggregateType::SUM,
    .compare_op = CompareOp::GT,
    .scale_factor = 0.5,
    .allow_null_mismatch = false
});

// 预计算 SUM(l_quantity) GROUP BY l_partkey, l_suppkey
transformer.precompute(/* ... */);

// 应用到 partsupp
auto matches = transformer.apply(/* ... */);
```

---

## 六、预期性能

### 6.1 复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 朴素执行 | O(P × L) | O(1) |
| 解关联 (单线程) | O(P + L) | O(P) |
| 解关联 (8线程) | O((P + L) / 8) | O(P × 8) |

其中 P = 匹配的 part 数量 (~200)，L = lineitem 行数 (~6M)

### 6.2 预期加速比

| 版本 | 预期时间 | 加速比 |
|------|----------|--------|
| DuckDB | 13.42 ms | 1.00x |
| 解关联 (单线程) | ~8 ms | 1.7x |
| 解关联 (8线程) | ~3 ms | 4.5x |
| 解关联 + SIMD | ~2 ms | 6.7x |

---

## 七、实现计划

### Step 1: 基础实现 (Day 1)
- [ ] 实现 `PrecomputedAggregates` 类
- [ ] 实现 `CorrelatedSubqueryTransformer` 类
- [ ] 实现 `Q17Optimizer::execute` 单线程版本
- [ ] 单元测试

### Step 2: 并行化 (Day 2)
- [ ] 实现 8 线程并行版本
- [ ] 实现 Thread-Local 状态合并
- [ ] 实现 SIMD 阈值比较
- [ ] 性能测试

### Step 3: 集成测试 (Day 2)
- [ ] 集成到 TPC-H 基准测试系统
- [ ] 验证正确性 (对比 DuckDB 结果)
- [ ] 性能回归测试
- [ ] 文档更新

### Step 4: 扩展到 Q20 (Day 3)
- [ ] 适配 Q20 的复合键
- [ ] 测试 Q20 优化效果
- [ ] 通用化验证

---

## 八、文件结构

| 文件 | 内容 |
|------|------|
| `include/thunderduck/correlated_subquery.h` | 通用类定义 |
| `src/operators/correlated_subquery.cpp` | 通用类实现 |
| `benchmark/tpch/tpch_operators_v36.h` | Q17 优化实现 |
| `benchmark/tpch/tpch_operators_v36.cpp` | V36 版本实现 |

---

## 九、总结

Q17 优化的核心是**相关子查询解关联**，将 O(P × L) 的朴素执行转换为 O(P + L) 的单遍扫描。通用化设计支持：

1. **多种聚合**: AVG, SUM, COUNT, MIN, MAX
2. **多种比较**: <, <=, >, >=, =, <>
3. **缩放因子**: 0.2 * AVG, 0.5 * SUM 等
4. **并行执行**: 8 线程 + SIMD

预期将 Q17 从 1.00x 提升到 **4.5x ~ 6.7x** 加速比。
