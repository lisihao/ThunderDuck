# ThunderDuck

> **Supercharging DuckDB on Apple M4**

ThunderDuck 是一个针对 Apple M4 芯片优化的 DuckDB 算子后端，通过深度利用 M4 的 SIMD 指令集、Neural Engine (NPU) 和统一内存架构，实现核心算子的极致性能。

## 特性

- **SIMD 加速** - 利用 128-bit ARM Neon 指令并行处理数据
- **NPU 加速** - 探索 Apple Neural Engine 加速大规模聚合计算
- **缓存优化** - 128 字节缓存行对齐，减少内存访问延迟
- **零拷贝** - 利用统一内存架构，CPU/NPU 共享数据
- **无缝集成** - 作为 DuckDB Extension 或编译时替换

## 优化算子

| 算子 | 优化技术 | 目标加速比 |
|------|----------|------------|
| **Filter** | SIMD 批量比较、位掩码压缩 | 2-4x |
| **Aggregation** | SIMD 并行归约、NPU 张量运算 | 2-3x |
| **Join** | SIMD 哈希计算、缓存友好哈希表 | 1.5-2x |
| **Sort** | Bitonic Sort、分块多路合并 | 1.5-2.5x |

## 项目结构

```
ThunderDuck/
├── src/
│   ├── core/           # 核心框架
│   ├── operators/      # 优化算子实现
│   │   ├── filter/
│   │   ├── aggregate/
│   │   ├── join/
│   │   └── sort/
│   ├── npu/            # NPU 加速层
│   └── utils/          # 工具函数
├── include/            # 公共头文件
├── tests/              # 测试
├── benchmarks/         # 性能基准
└── docs/               # 文档
```

## 技术栈

- **语言**: C/C++, AArch64 汇编
- **SIMD**: ARM Neon Intrinsics
- **NPU**: Core ML / Metal Performance Shaders
- **构建**: CMake
- **依赖**: DuckDB (主分支)

## 快速开始

```bash
# 克隆项目
git clone https://github.com/user/ThunderDuck.git
cd ThunderDuck

# 构建
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# 运行测试
ctest
```

## 系统要求

- macOS 14.0+
- Apple Silicon M4 芯片（推荐）
- Xcode 15.0+ 或 Apple Clang
- DuckDB v1.0+

## 文档

- [设计文档](docs/THUNDERDUCK_DESIGN.md) - 完整的技术设计和实现计划

## 许可证

MIT License
