# ThunderDuck 开发规范

> **版本**: 1.0.0 | **更新日期**: 2026-01-24

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

## 文档

- 设计文档: `docs/THUNDERDUCK_DESIGN.md`
