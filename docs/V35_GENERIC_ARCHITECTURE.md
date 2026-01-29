# V35 通用化架构设计

> **版本**: V35 | **日期**: 2026-01-29 | **原则**: 通用算子优先，多查询复用

## 一、设计原则

1. **通用性**: 每个新算子至少可用于 3+ 个查询
2. **零硬编码**: 所有参数通过 QueryConfig 配置
3. **向后兼容**: 复用 V32/V33 已有组件
4. **性能保证**: 不低于专用实现的 95%

---

## 二、新增通用算子

### 2.1 SIMD 字符串处理器 (SIMDStringProcessor)

**适用查询**: Q8, Q13, Q16, Q19, Q22 等涉及字符串操作的查询

```cpp
namespace ops_v35 {

/**
 * SIMD 字符串处理器
 *
 * 功能:
 * - SUBSTRING 提取 (批量)
 * - 字符串前缀/后缀匹配
 * - LIKE 模式匹配 (%pattern%)
 * - 字符串集合成员检测
 */
class SIMDStringProcessor {
public:
    // ========== SUBSTRING 操作 ==========

    /**
     * 批量提取固定位置子串
     * @param strings 输入字符串数组
     * @param start 起始位置 (0-based)
     * @param length 提取长度
     * @return 提取的子串数组
     *
     * 用途: Q22 SUBSTRING(c_phone FROM 1 FOR 2)
     */
    static std::vector<std::string> substring_batch(
        const std::vector<std::string>& strings,
        size_t start, size_t length);

    /**
     * 批量提取固定长度前缀并转换为整数
     * @param strings 输入字符串数组
     * @param prefix_len 前缀长度
     * @return 前缀的整数值 (-1 表示无效)
     *
     * 用途: Q22 电话号码前2位 → 国家码
     */
    static std::vector<int16_t> prefix_to_int_batch(
        const std::vector<std::string>& strings,
        size_t prefix_len);

    // ========== 模式匹配 ==========

    /**
     * 批量 LIKE 模式匹配 (%pattern%)
     * @param strings 输入字符串数组
     * @param pattern 搜索模式 (不含 %)
     * @return 匹配位图
     *
     * 用途: Q9 p_name LIKE '%green%', Q13 o_comment NOT LIKE '%special%requests%'
     */
    static std::vector<uint64_t> like_contains_batch(
        const std::vector<std::string>& strings,
        const std::string& pattern);

    /**
     * 批量多模式匹配 (pattern1...pattern2)
     * @param strings 输入字符串数组
     * @param pattern1 第一个模式
     * @param pattern2 第二个模式 (必须在 pattern1 之后)
     * @return 匹配位图
     *
     * 用途: Q13 '%special%requests%'
     */
    static std::vector<uint64_t> like_two_patterns_batch(
        const std::vector<std::string>& strings,
        const std::string& pattern1,
        const std::string& pattern2);

    // ========== 前缀/后缀匹配 ==========

    /**
     * 批量前缀匹配
     * @param strings 输入字符串数组
     * @param prefix 前缀
     * @return 匹配位图
     *
     * 用途: Q14 p_type LIKE 'PROMO%'
     */
    static std::vector<uint64_t> starts_with_batch(
        const std::vector<std::string>& strings,
        const std::string& prefix);

    /**
     * 批量多前缀匹配
     * @param strings 输入字符串数组
     * @param prefixes 前缀集合
     * @return 匹配的前缀索引 (-1 表示不匹配)
     *
     * 用途: Q19 p_brand IN ('Brand#12', 'Brand#23', 'Brand#34')
     */
    static std::vector<int8_t> starts_with_any_batch(
        const std::vector<std::string>& strings,
        const std::vector<std::string>& prefixes);

    // ========== 集合成员检测 ==========

    /**
     * 批量字符串集合检测
     * @param strings 输入字符串数组
     * @param target_set 目标集合
     * @return 匹配位图
     *
     * 用途: Q8 r_name = 'AMERICA', Q7 n_name IN ('FRANCE', 'GERMANY')
     */
    static std::vector<uint64_t> in_set_batch(
        const std::vector<std::string>& strings,
        const std::unordered_set<std::string>& target_set);
};

} // namespace ops_v35
```

**复用分析**:

| 查询 | 使用功能 | 当前实现 |
|------|----------|----------|
| Q8 | `in_set_batch` (nation) | 循环比较 |
| Q9 | `like_contains_batch` (%green%) | strstr |
| Q13 | `like_two_patterns_batch` | memmem (已优化) |
| Q14 | `starts_with_batch` (PROMO%) | starts_with |
| Q16 | `starts_with_batch`, `like_contains_batch` | 混合 |
| Q19 | `starts_with_any_batch` | 条件分支 |
| Q22 | `prefix_to_int_batch` (电话码) | 手动提取 |

---

### 2.2 SEMI/ANTI JOIN 算子 (SemiAntiJoin)

**适用查询**: Q4, Q17, Q20, Q21, Q22 等涉及 EXISTS/NOT EXISTS 的查询

```cpp
namespace ops_v35 {

/**
 * SEMI/ANTI JOIN 算子
 *
 * SEMI JOIN: 返回左表中在右表存在匹配的行 (EXISTS)
 * ANTI JOIN: 返回左表中在右表不存在匹配的行 (NOT EXISTS)
 */
class SemiAntiJoin {
public:
    /**
     * 构建右侧 (Build Side)
     * @param keys 右侧键数组
     * @param count 键数量
     */
    void build(const int32_t* keys, size_t count);

    /**
     * 带过滤的构建
     * @param keys 右侧键数组
     * @param count 键数量
     * @param filter 过滤谓词 (返回 true 保留)
     */
    void build_filtered(const int32_t* keys, size_t count,
                        const std::function<bool(size_t)>& filter);

    // ========== SEMI JOIN (EXISTS) ==========

    /**
     * SEMI JOIN: 返回存在匹配的左侧索引
     * @param probe_keys 左侧键数组
     * @param probe_count 左侧数量
     * @return 存在匹配的行索引
     *
     * 用途: Q4 EXISTS (SELECT * FROM lineitem WHERE ...)
     */
    std::vector<uint32_t> semi_join(const int32_t* probe_keys, size_t probe_count);

    /**
     * SEMI JOIN 位图版本
     * @return 匹配位图
     */
    std::vector<uint64_t> semi_join_bitmap(const int32_t* probe_keys, size_t probe_count);

    // ========== ANTI JOIN (NOT EXISTS) ==========

    /**
     * ANTI JOIN: 返回不存在匹配的左侧索引
     * @param probe_keys 左侧键数组
     * @param probe_count 左侧数量
     * @return 不存在匹配的行索引
     *
     * 用途: Q22 NOT EXISTS (SELECT * FROM orders WHERE ...)
     */
    std::vector<uint32_t> anti_join(const int32_t* probe_keys, size_t probe_count);

    /**
     * ANTI JOIN 位图版本
     * @return 不匹配位图
     */
    std::vector<uint64_t> anti_join_bitmap(const int32_t* probe_keys, size_t probe_count);

    // ========== 组合操作 (Q21) ==========

    /**
     * 计数匹配: 返回每个左侧键的匹配数量
     * @param probe_keys 左侧键数组
     * @param probe_count 左侧数量
     * @return 每个键的匹配计数
     *
     * 用途: Q21 需要检查 "存在其他供应商" 和 "不存在其他迟交供应商"
     */
    std::vector<uint32_t> count_matches(const int32_t* probe_keys, size_t probe_count);

private:
    // Bloom Filter 预过滤 (大数据集)
    SingleHashBloomFilter bloom_;
    bool use_bloom_ = false;

    // 精确存储
    std::unordered_set<int32_t> key_set_;

    // 计数存储 (用于 count_matches)
    std::unordered_map<int32_t, uint32_t> key_counts_;
    bool track_counts_ = false;

    void auto_select_strategy(size_t build_count);
};

} // namespace ops_v35
```

**复用分析**:

| 查询 | 使用功能 | 当前实现 |
|------|----------|----------|
| Q4 | `semi_join` (EXISTS lineitem) | 回退 DuckDB |
| Q17 | `semi_join` + 聚合 | 回退 DuckDB |
| Q20 | `semi_join` (多层) | 回退 DuckDB |
| Q21 | `semi_join` + `anti_join` | 回退 DuckDB |
| Q22 | `anti_join` (NOT EXISTS orders) | unordered_set |

---

### 2.3 管道融合框架 (PipelineFusion)

**适用查询**: Q3, Q5, Q7, Q9, Q10, Q12, Q18 等 JOIN + Aggregate 查询

```cpp
namespace ops_v35 {

/**
 * 管道融合框架
 *
 * 将 Filter → JOIN → Aggregate 融合为单遍扫描
 * 避免中间结果物化，减少内存带宽
 */
template<typename AggKey, typename AggValue>
class PipelineFusion {
public:
    // ========== 配置阶段 ==========

    /**
     * 设置过滤条件
     * @param filter 过滤谓词
     */
    void set_filter(std::function<bool(size_t)> filter);

    /**
     * 添加 Hash Join 阶段
     * @param name 阶段名称
     * @param build_keys Build 侧键
     * @param build_count Build 侧数量
     */
    void add_hash_join(const std::string& name,
                       const int32_t* build_keys, size_t build_count);

    /**
     * 添加带值的 Hash Join
     * @param build_values Build 侧值 (用于后续查找)
     */
    template<typename V>
    void add_hash_join_with_value(const std::string& name,
                                  const int32_t* build_keys,
                                  const V* build_values,
                                  size_t build_count);

    /**
     * 设置聚合函数
     * @param key_extractor 从行提取聚合键
     * @param value_extractor 从行提取聚合值
     * @param combiner 合并函数
     */
    void set_aggregation(
        std::function<AggKey(size_t, const JoinContext&)> key_extractor,
        std::function<AggValue(size_t, const JoinContext&)> value_extractor,
        std::function<AggValue(AggValue, AggValue)> combiner = std::plus<AggValue>());

    // ========== 执行阶段 ==========

    /**
     * 执行融合管道
     * @param probe_keys Probe 侧键 (第一个 JOIN 的探测键)
     * @param probe_count Probe 侧数量
     * @param thread_count 线程数 (0 = 自动)
     */
    void execute(const int32_t* probe_keys, size_t probe_count,
                 size_t thread_count = 0);

    /**
     * 获取聚合结果
     */
    template<typename Func>
    void for_each_result(Func&& callback);

private:
    // JOIN 阶段
    struct JoinStage {
        std::string name;
        AdaptiveHashJoin join;
    };
    std::vector<JoinStage> join_stages_;

    // 线程本地聚合
    GenericThreadLocalAggregator<AggValue> aggregator_;

    // 谓词和提取器
    std::function<bool(size_t)> filter_;
    std::function<AggKey(size_t, const JoinContext&)> key_extractor_;
    std::function<AggValue(size_t, const JoinContext&)> value_extractor_;
    std::function<AggValue(AggValue, AggValue)> combiner_;
};

/**
 * JOIN 上下文 - 保存多阶段 JOIN 的中间结果
 */
struct JoinContext {
    std::vector<int32_t> join_values;  // 各阶段 JOIN 返回的值

    int32_t get(size_t stage) const { return join_values[stage]; }
};

} // namespace ops_v35
```

**复用分析**:

| 查询 | JOIN 阶段数 | 聚合类型 |
|------|-------------|----------|
| Q3 | 2 (customer→orders, orders→lineitem) | SUM revenue |
| Q5 | 4 (region→nation→customer→orders) | SUM revenue by nation |
| Q7 | 4 (nation×2→supplier/customer→lineitem) | SUM by nation pair, year |
| Q9 | 4 (nation→supplier→part→lineitem) | SUM profit by nation, year |
| Q10 | 2 (customer→orders→lineitem) | SUM by customer |
| Q12 | 1 (orders→lineitem) | COUNT by ship mode |
| Q18 | 2 (orders→lineitem→customer) | SUM by customer, order |

---

### 2.4 条件聚合器 (ConditionalAggregator)

**适用查询**: Q8, Q12, Q14 等带 CASE WHEN 的聚合查询

```cpp
namespace ops_v35 {

/**
 * 条件聚合器
 *
 * 支持 CASE WHEN condition THEN value ELSE default END 的聚合
 */
template<typename Key, typename Value>
class ConditionalAggregator {
public:
    /**
     * 添加条件分支
     * @param name 分支名称 (用于结果获取)
     * @param condition 条件谓词
     */
    void add_branch(const std::string& name,
                    std::function<bool(size_t)> condition);

    /**
     * 设置默认分支 (ELSE)
     */
    void set_default_branch(const std::string& name);

    /**
     * 执行条件聚合
     * @param keys 聚合键数组
     * @param values 聚合值数组
     * @param count 数据量
     * @param thread_count 线程数
     */
    void aggregate(const Key* keys, const Value* values, size_t count,
                   size_t thread_count = 0);

    /**
     * 批量条件聚合 (SIMD 优化)
     * @param condition_results 预计算的条件结果 (每行的分支索引)
     */
    void aggregate_precomputed(const Key* keys, const Value* values,
                               const uint8_t* condition_results, size_t count,
                               size_t thread_count = 0);

    /**
     * 获取结果
     * @param branch_name 分支名称
     */
    template<typename Func>
    void for_each(const std::string& branch_name, Func&& callback);

    /**
     * 获取所有分支结果
     */
    template<typename Func>
    void for_each_all(Func&& callback);

private:
    struct Branch {
        std::string name;
        std::function<bool(size_t)> condition;
        GenericThreadLocalAggregator<Value> aggregator;
    };
    std::vector<Branch> branches_;
    size_t default_branch_ = SIZE_MAX;
};

} // namespace ops_v35
```

**复用分析**:

| 查询 | 条件 | 聚合 |
|------|------|------|
| Q8 | nation = 'BRAZIL' | SUM(volume) by year |
| Q12 | shipmode = 'high' / 'low' | COUNT by mode |
| Q14 | p_type LIKE 'PROMO%' | SUM(revenue) |

---

### 2.5 直接数组索引构建器 (DirectArrayIndexBuilder)

**适用查询**: 所有需要快速键查找的场景

```cpp
namespace ops_v35 {

/**
 * 直接数组索引构建器
 *
 * 自动检测数据特征，选择最优索引策略:
 * - 直接数组: 键范围小且稠密
 * - 紧凑 Hash: 键范围大或稀疏
 */
template<typename Value>
class DirectArrayIndexBuilder {
public:
    /**
     * 从数据构建索引
     * @param keys 键数组
     * @param values 值数组
     * @param count 数据量
     * @param default_value 默认值 (未找到时返回)
     */
    void build(const int32_t* keys, const Value* values, size_t count,
               Value default_value = Value{});

    /**
     * 仅构建存在性索引 (用于 JOIN)
     */
    void build_existence(const int32_t* keys, size_t count);

    /**
     * 查找
     */
    __attribute__((always_inline))
    const Value* find(int32_t key) const {
        if (use_direct_array_) {
            size_t idx = static_cast<size_t>(key - min_key_);
            if (idx < direct_array_.size() && valid_[idx]) {
                return &direct_array_[idx];
            }
            return nullptr;
        }
        return hash_table_.find(key);
    }

    /**
     * 检查存在性
     */
    __attribute__((always_inline))
    bool contains(int32_t key) const {
        if (use_direct_array_) {
            size_t idx = static_cast<size_t>(key - min_key_);
            return idx < valid_.size() && valid_[idx];
        }
        return hash_table_.find(key) != nullptr;
    }

    /**
     * 批量查找
     */
    void batch_find(const int32_t* keys, size_t count, const Value** results) const;

    /**
     * 获取选中的策略
     */
    bool is_direct_array() const { return use_direct_array_; }

    /**
     * 内存使用统计
     */
    size_t memory_usage() const;

private:
    bool use_direct_array_ = false;
    int32_t min_key_ = 0;

    // 直接数组
    std::vector<Value> direct_array_;
    std::vector<bool> valid_;

    // Hash 表 (备用)
    CompactHashTable<Value> hash_table_;

    void auto_select_strategy(int32_t min_key, int32_t max_key, size_t count);
};

} // namespace ops_v35
```

**复用分析**:

| 场景 | 当前实现 | V35 改进 |
|------|----------|----------|
| supplier → nation | 直接数组 | 自动选择 |
| customer → nation | 直接数组 | 自动选择 |
| part → valid | unordered_set | 自动选择 |
| orders → info | 直接数组 | 自动选择 |

---

## 三、V35 组件复用矩阵

| 组件 | Q3 | Q5 | Q7 | Q8 | Q9 | Q12 | Q13 | Q14 | Q16 | Q19 | Q21 | Q22 |
|------|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|
| SIMDStringProcessor | | | | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ |
| SemiAntiJoin | | | | | | | | | | | ✓ | ✓ |
| PipelineFusion | ✓ | ✓ | ✓ | | ✓ | ✓ | | | | | | |
| ConditionalAggregator | | | | ✓ | | ✓ | | ✓ | | | | |
| DirectArrayIndexBuilder | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**覆盖统计**:
- SIMDStringProcessor: 7 个查询
- SemiAntiJoin: 2+ 个查询 (可扩展到 Q4, Q17, Q20)
- PipelineFusion: 6 个查询
- ConditionalAggregator: 3 个查询
- DirectArrayIndexBuilder: 12 个查询 (全覆盖)

---

## 四、文件结构

```
benchmark/tpch/
├── tpch_operators_v35.h      # V35 通用算子头文件 (~400 行)
├── tpch_operators_v35.cpp    # V35 通用算子实现 (~800 行)
├── tpch_string_simd_v35.cpp  # SIMD 字符串处理实现 (~300 行)
├── tpch_join_v35.cpp         # SEMI/ANTI JOIN 实现 (~250 行)
├── tpch_pipeline_v35.cpp     # 管道融合框架实现 (~400 行)
└── tpch_queries_v35.cpp      # V35 查询入口 (~200 行)
```

---

## 五、预期收益

### 5.1 直接收益

| 查询 | V34 | V35 目标 | 优化组件 |
|------|-----|----------|----------|
| Q3 | 1.12x | 1.40x | PipelineFusion |
| Q8 | 1.06x | 1.30x | ConditionalAggregator + DirectArray |
| Q14 | 1.18x | 1.50x | ConditionalAggregator |
| Q22 | 0.96x | 1.15x | SIMDString + SemiAntiJoin |

### 5.2 间接收益 (其他查询)

| 查询 | V34 | V35 预期 | 优化组件 |
|------|-----|----------|----------|
| Q5 | 1.22x | 1.35x | PipelineFusion |
| Q7 | 2.50x | 2.70x | PipelineFusion |
| Q9 | 1.32x | 1.50x | PipelineFusion + SIMDString |
| Q13 | 1.89x | 2.00x | SIMDString (已优化) |
| Q21 | 1.00x | 1.20x | SemiAntiJoin |

### 5.3 总体目标

- **几何平均加速比**: 2.10x → **2.30x** (+10%)
- **覆盖率**: 18/22 → **19/22** (新增 Q21)

---

## 六、实现优先级

### Phase 1 (P0): 核心组件
1. DirectArrayIndexBuilder - 全覆盖基础组件
2. SIMDStringProcessor - 7 查询受益
3. ConditionalAggregator - Q8/Q14 直接提升

### Phase 2 (P1): 高级组件
4. PipelineFusion - 6 查询受益
5. SemiAntiJoin - Q21/Q22 优化

### Phase 3 (P2): 完善
6. 集成测试
7. 性能回归检测
8. 文档更新

---

## 七、V33 组件复用

V35 复用 V33 的以下组件:
- `QueryConfig` / `ExecutionConfig` - 配置框架
- `DateRange` / `NumericRange` - 范围谓词
- `AdaptiveHashJoin` - 自适应 JOIN
- `GenericThreadLocalAggregator` - 线程本地聚合
- `TaskScheduler` - 并行调度
- `AutoTuner` - 自动调优

---

*ThunderDuck V35 - 通用化架构，多查询复用*
