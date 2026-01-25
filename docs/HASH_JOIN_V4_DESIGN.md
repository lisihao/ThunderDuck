# Hash Join v4 设计文档

> 版本: 4.0.0 | 日期: 2026-01-24

## 一、需求概述

Hash Join v4 旨在通过多策略自适应选择，在不同数据规模和硬件配置下实现最优性能。

### 目标

| 策略 | 优化方向 | 预期提升 | 适用场景 |
|------|---------|---------|---------|
| RADIX256 | 256 分区 (8 bits) | 1.3-1.5x | 中等数据量 |
| BLOOMFILTER | CPU Bloom 预过滤 | 1.3-1.5x | 高选择性 |
| NPU | BNNS 加速 Bloom | 1.5-2x | 大数据量 |
| GPU | Metal 并行探测 | 2-3x | 超大数据量 |

### 设计原则

1. **策略独立**: 各策略可独立使用，互不冲突
2. **自动选择**: AUTO 模式根据数据特征自动选择最优策略
3. **优雅回退**: NPU/GPU 不可用时自动回退到 CPU 策略
4. **向后兼容**: 保持 v3 API 可用

## 二、技术方案

### 2.1 策略调度器

```cpp
class StrategyDispatcher {
    JoinStrategy select_strategy(size_t build, size_t probe, const JoinConfigV4& config) {
        // 用户指定策略
        if (config.strategy != AUTO) return config.strategy;

        // AUTO 选择逻辑
        if (build < 10000) return V3_FALLBACK;          // 小表
        if (gpu_available() && probe >= 1M) return GPU; // GPU 并行
        if (npu_available() && build >= 500K) return NPU; // NPU 加速
        if (build >= 100K) return BLOOMFILTER;          // Bloom 预过滤
        return RADIX256;                                 // 默认
    }
};
```

**回退链**: GPU → NPU → BLOOMFILTER → RADIX256 → V3_FALLBACK

### 2.2 RADIX256 策略

**优化点**:
- 256 分区 (8-bit vs v3 的 4-bit)
- 每分区 ~53KB，适配 M4 L1 缓存 (192KB)
- 复用 v3 的 SOA 哈希表和 SIMD 探测

**算法**:
```
1. 计算直方图 (256 分区)
2. 预分配分区内存
3. Scatter 数据到分区
4. 并行处理每个分区 (morsel-driven)
5. 合并结果
```

### 2.3 BloomFilter 策略

**数据结构**:
- 128 字节对齐位数组
- CRC32 双重哈希: `h_i(x) = h1(x) + i * h2(x)`
- 7 个哈希函数，~1% 假阳性率

**算法**:
```
1. 构建 Bloom Filter (从 build keys)
2. 构建哈希表
3. 批量过滤 probe keys (8192 批次)
4. 只探测通过过滤的 keys
```

**优势**:
- 减少 80-90% 的哈希表探测 (高选择性 join)
- SIMD 批量位检查

### 2.4 NPU 策略 (BNNS)

使用 Apple Accelerate 框架的 BNNS 加速批量哈希计算和向量操作。

**特点**:
- 批量处理 8192 keys
- vDSP 向量化位操作
- 自动回退到 CPU Bloom

### 2.5 GPU 策略 (Metal)

**架构**:
```
┌─────────────────────────────────────┐
│ CPU: 构建哈希表                      │
│      准备 Metal 缓冲区              │
└─────────────┬───────────────────────┘
              │ 统一内存 (零拷贝)
┌─────────────▼───────────────────────┐
│ GPU Kernel: hash_join_probe          │
│   - 每线程处理 1 个 probe key        │
│   - 线性探测哈希表                   │
│   - 原子计数器收集匹配               │
└─────────────────────────────────────┘
```

**Metal Shader**:
- `hash_join_probe`: 基本并行探测
- `bloom_hash_join_probe`: Bloom + Hash Join 组合 (减少内存带宽)

## 三、详细设计

### 3.1 文件结构

```
include/thunderduck/
├── join.h                    # v4 API (JoinStrategy, JoinConfigV4)
├── bloom_filter.h            # Bloom Filter 接口

src/operators/join/
├── hash_join_v4.cpp          # v4 核心 + 策略调度器
├── hash_join_v4_radix256.cpp # RADIX256 策略
├── hash_join_v4_bloom.cpp    # BloomFilter 策略
├── bloom_filter.cpp          # Bloom 数据结构

src/npu/
├── bloom_bnns.cpp            # BNNS 加速 Bloom

src/gpu/
├── hash_join_metal.mm        # Metal 集成
├── shaders/
│   └── hash_join.metal       # Metal 着色器
```

### 3.2 API 设计

```cpp
// 策略枚举
enum class JoinStrategy {
    AUTO,           // 自动选择
    RADIX256,       // 256 分区
    BLOOMFILTER,    // CPU Bloom
    NPU,            // NPU 加速
    GPU,            // Metal GPU
    V3_FALLBACK     // 回退到 v3
};

// 配置结构
struct JoinConfigV4 {
    JoinStrategy strategy = AUTO;
    size_t num_threads = 4;
    int radix_bits = 8;
    double bloom_fpr = 0.01;
    bool fallback_to_cpu = true;
    // ...
};

// 主 API
size_t hash_join_i32_v4(
    const int32_t* build_keys, size_t build_count,
    const int32_t* probe_keys, size_t probe_count,
    JoinType join_type, JoinResult* result);

size_t hash_join_i32_v4_config(
    /* 同上 */,
    const JoinConfigV4& config);
```

### 3.3 Bloom Filter 设计

```cpp
class BloomFilter {
    void insert(int32_t key);
    void insert_batch(const int32_t* keys, size_t count);
    bool maybe_contains(int32_t key) const;
    size_t filter_batch(const int32_t* keys, size_t count,
                        uint32_t* out_indices) const;
};
```

**参数选择**:
- 位数: `m = -n * ln(p) / ln(2)^2` (n=元素数, p=FPR)
- 哈希数: `k = (m/n) * ln(2)` ≈ 7 (for 1% FPR)

## 四、实现计划

| 阶段 | 任务 | 状态 |
|------|------|------|
| 1 | API + 调度器框架 | ✅ 完成 |
| 2 | RADIX256 策略 | ✅ 完成 |
| 3 | Bloom 数据结构 | ✅ 完成 |
| 4 | BloomFilter 策略 | ✅ 完成 |
| 5 | NPU 加速 | ✅ 完成 |
| 6 | GPU 加速 | ✅ 完成 |
| 7 | 集成测试 | 待验证 |

## 五、验证方案

### 正确性验证

```cpp
// 对比 v3 和 v4 结果
auto result_v3 = hash_join_i32_v3(build, build_n, probe, probe_n, ...);
auto result_v4 = hash_join_i32_v4(build, build_n, probe, probe_n, ...);
assert(result_v3.count == result_v4.count);
// 比较结果集合 (可能顺序不同)
```

### 性能验证

```bash
# 运行 J1/J2/J3 基准测试
./build/comprehensive_benchmark

# 期望结果:
# J3 (1M×10M): v3 1.11x → v4 1.5-2x
```

### 策略独立性验证

```cpp
// 同时使用不同策略不冲突
std::thread t1([&]{ run_join(RADIX256); });
std::thread t2([&]{ run_join(GPU); });
t1.join(); t2.join();
```

## 六、性能预期

| 场景 | v3 性能 | v4 预期 | 策略 |
|------|--------|--------|------|
| J1 (10K×100K) | 1.24x | 1.3x | V3_FALLBACK |
| J2 (100K×1M) | 1.15x | 1.4x | RADIX256 |
| J3 (1M×10M) | 1.11x | 1.8x | BLOOMFILTER/NPU |
| J3 GPU | - | 2.5x | GPU |

## 七、未来优化

1. **SIMD Bloom Filter**: 使用 Neon 向量化位检查
2. **预编译 Metal Shader**: 生产环境使用 .metallib
3. **多 GPU 支持**: 分布式 GPU 探测
4. **自适应批次大小**: 根据 L1/L2 缓存动态调整
