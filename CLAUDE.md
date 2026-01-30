# ThunderDuck 开发规范

> **版本**: 1.4.0 | **更新日期**: 2026-01-30

## 项目概述

ThunderDuck 是针对 Apple M4 芯片优化的 DuckDB 算子后端。

## 技术栈

- **语言**: C/C++17, AArch64 汇编
- **SIMD**: ARM Neon Intrinsics
- **NPU**: Core ML / Metal Performance Shaders
- **构建**: CMake 3.20+
- **目标平台**: macOS 14.0+, Apple Silicon M4

## 代码规范

### 命名约定

- 函数: `snake_case` (如 `simd_filter_gt_i32`)
- 类: `PascalCase` (如 `FilterOperator`)
- 常量: `UPPER_SNAKE_CASE` (如 `CACHE_LINE_SIZE`)
- 命名空间: `thunderduck::operators`

### SIMD 代码风格

```cpp
// 使用 ARM Neon intrinsics
#include <arm_neon.h>

// 128-bit 向量类型
int32x4_t   // 4 个 int32
float32x4_t // 4 个 float32
uint8x16_t  // 16 个 uint8
```

### 内存对齐

```cpp
// 128 字节对齐（M4 缓存行大小）
alignas(128) int32_t data[1024];
```

## 构建命令

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
ctest
```

## 基准测试规则

性能测试必须遵循以下规则以确保结果可靠：

### 1. 使用中位数而非平均值

```cpp
// 正确: 使用中位数
std::sort(times.begin(), times.end());
double median = times[times.size() / 2];

// 错误: 使用平均值 (受异常值影响大)
double avg = std::accumulate(times.begin(), times.end(), 0.0) / n;
```

### 2. 报告标准差

```cpp
// 计算标准差
double sq_sum = 0;
for (double t : times) sq_sum += (t - median) * (t - median);
double stddev = sqrt(sq_sum / times.size());

// 输出格式: "1.234 ms (σ=0.05)"
cout << median << " ms (σ=" << stddev << ")" << endl;
```

### 3. 剔除异常值

使用 IQR (四分位距) 方法剔除异常值：

```cpp
// 计算 Q1, Q3, IQR
std::sort(times.begin(), times.end());
double q1 = times[times.size() / 4];
double q3 = times[times.size() * 3 / 4];
double iqr = q3 - q1;

// 剔除 < Q1-1.5*IQR 或 > Q3+1.5*IQR 的值
double lower = q1 - 1.5 * iqr;
double upper = q3 + 1.5 * iqr;
times.erase(std::remove_if(times.begin(), times.end(),
    [&](double t) { return t < lower || t > upper; }), times.end());
```

### 4. 最小迭代次数

- 微基准测试: ≥30 次迭代
- 完整算子测试: ≥10 次迭代
- 预热运行: ≥1 次 (不计入统计)

## 系统表架构规则 (强制)

所有开发的算子必须遵循以下规则：

### 1. 算子必须注册到系统表

每个新算子在 `tpch_query_optimizer.cpp` 的 `register_tpch_query_configs()` 中注册：

```cpp
cat.register_operator(
    "V54-NativeDoubleSIMDFilter",  // 算子名称 (版本-功能)
    0.1f,                           // startup_ms: 启动成本
    0.0003f,                        // per_row_us: 每行处理成本
    10000,                          // min_rows: 最小适用行数
    0                               // max_rows: 最大适用行数 (0=无上限)
);
```

### 2. 优化器从系统表读取决策

优化器 **必须** 从 `catalog::catalog()` 内存数据读取算子信息：

```cpp
// 正确: 从系统表读取
auto& cat = catalog::catalog();
auto stats = cat.get_version_stats(query_id, version);
auto cost = cat.estimate_cost(operator_name, row_count);

// 错误: 硬编码在头文件中
if (version == "V54") { cost = 0.1 + rows * 0.0003; }
```

### 3. 头文件保留描述性信息

头文件 (`tpch_operators_v*.h`) 仍需保留：
- 类型定义和接口声明
- 算子功能文档注释
- 适用场景说明

```cpp
/**
 * V54 NativeDoubleSIMDFilter
 *
 * 功能: 原生 double 列 SIMD 过滤 + 聚合
 * 适用: 单表扫描 + 多谓词过滤 (如 Q6)
 * 特点: 8 线程 SIMD, 最低每行成本
 *
 * 成本模型 (记录在系统表):
 *   startup_ms: 0.1
 *   per_row_us: 0.0003
 *   min_rows: 10000
 */
class NativeDoubleSIMDFilter { ... };
```

### 4. 系统表数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│ System Catalog Structure                                        │
├─────────────────────────────────────────────────────────────────┤
│ 1. td_operators     - 算子元数据 (成本模型)                     │
│ 2. td_query_stats   - 查询性能统计 (历史数据)                   │
│ 3. td_sketch_buckets - 时间尺度 Sketch (多分辨率聚合)           │
│ 4. td_metrics       - 性能指标环形缓冲区                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5. 性能数据自动采集

每次基准测试自动写入系统表：
- DuckDB 基线时间
- ThunderDuck 优化版本时间
- 时间尺度 Sketch (自动卷积: 秒→分→时→日)

### 6. 算子注册硬性检查 (MUST)

**所有实现的算子必须注册到优化器系统中才能被使用。**

```
┌─────────────────────────────────────────────────────────────────┐
│ 算子开发完整流程 (缺一不可)                                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. 头文件声明    tpch_operators_vXX.h: void run_qN_vXX(...)     │
│ 2. 实现代码      tpch_operators_vXX.cpp: 算子逻辑                │
│ 3. CMake 添加    benchmarks/CMakeLists.txt: 源文件列表           │
│ 4. 优化器注册    tpch_query_optimizer.cpp: candidates 列表       │
│                                                                 │
│ ⚠️ 未注册到优化器的算子 = 不存在                                  │
└─────────────────────────────────────────────────────────────────┘
```

注册示例：

```cpp
// tpch_query_optimizer.cpp - register_tpch_query_configs()
case 12: {
    config.candidates = {
        {"V57", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v57::run_q12_v57(l); }},
        {"V27", 0.9, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q12_v27(l); }}
    };
    break;
}
```

**检查命令**：

```bash
# 验证声明与注册一致性
grep "void run_q.*_v57" benchmark/tpch/tpch_operators_v57.h
grep "run_q.*_v57" benchmark/tpch/tpch_query_optimizer.cpp
```

## 通用算子设计规则 (铁律)

**禁止硬编码和专用设计**。所有算子必须是通用的、可复用的。

### 1. 禁止硬编码 (NEVER)

```cpp
// ❌ 禁止: 硬编码魔数
static constexpr size_t MAX_SIZE = 300000;
if (partkey < 200000) { ... }

// ✅ 正确: 运行时检测或配置
static size_t max_size() {
    return compute_from_l2_cache();  // 基于硬件自动计算
}
if (partkey <= detected_max_key) { ... }
```

### 2. 禁止查询专用设计 (NEVER)

```cpp
// ❌ 禁止: 查询专用算子
class Q8ParallelMultiJoin { ... };
void run_q17_optimized() { ... };

// ✅ 正确: 通用算子，查询无关
class ParallelMultiJoin { ... };  // 适用于任意多表 JOIN
class TwoPhaseAggregator { ... }; // 适用于任意两阶段聚合
```

### 3. 禁止 std::function 在热路径 (NEVER)

```cpp
// ❌ 禁止: std::function 无法内联
void process(std::function<bool(size_t)> filter) {
    for (size_t i = 0; i < n; ++i) {
        if (filter(i)) { ... }  // 虚调用开销!
    }
}

// ✅ 正确: 模板参数，编译时内联
template<typename FilterFn>
void process(FilterFn&& filter) {
    for (size_t i = 0; i < n; ++i) {
        if (filter(i)) { ... }  // 内联!
    }
}
```

### 4. 自适应存储结构 (MUST)

```cpp
// ✅ 正确: 自动检测 key 范围，选择最优结构
template<typename KeyT, typename ValueT>
class AdaptiveMap {
public:
    void build(const KeyT* keys, size_t count) {
        // 自动检测 key 范围
        KeyT min_key = *std::min_element(keys, keys + count);
        KeyT max_key = *std::max_element(keys, keys + count);
        size_t range = max_key - min_key + 1;

        // 自适应选择: 直接数组 vs Hash 表
        if (range <= compute_l2_friendly_size<ValueT>()) {
            use_direct_array(min_key, range);
        } else {
            use_hash_table(count);
        }
    }
};
```

### 5. 零硬编码检查清单

每次提交前检查:

| 检查项 | 禁止 | 正确做法 |
|--------|------|----------|
| 魔数 | `300000`, `200000` | `compute_from_hardware()` |
| 查询名 | `Q8`, `Q17` | 无查询引用 |
| 表名 | `lineitem`, `orders` | 泛型参数 |
| 列名 | `l_partkey` | 列索引或泛型 |
| 日期常量 | `9131` | 配置参数 |
| 字符串常量 | `"BRAZIL"` | 配置参数 |

### 6. 性能 vs 通用性权衡

**原则**: 通用性优先，但不牺牲核心性能。

```cpp
// ✅ 正确: 通用设计 + 编译时优化
template<typename KeyT, bool UseDirectArray = false>
class GenericJoin {
    // 编译时分支消除，零运行时开销
    auto lookup(KeyT key) const {
        if constexpr (UseDirectArray) {
            return direct_array_[key - offset_];
        } else {
            return hash_table_.find(key);
        }
    }
};

// 使用时: 编译器生成两个特化版本
using FastJoin = GenericJoin<int32_t, true>;   // 直接数组版本
using FlexJoin = GenericJoin<int32_t, false>;  // Hash 表版本
```

## 文档

- 设计文档: `docs/THUNDERDUCK_DESIGN.md`
