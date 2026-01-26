# ThunderDuck 开发规范

> **版本**: 1.1.0 | **更新日期**: 2026-01-26

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

## 文档

- 设计文档: `docs/THUNDERDUCK_DESIGN.md`
