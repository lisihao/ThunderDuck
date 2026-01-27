# V16 综合优化设计文档

> **版本**: 16.0 | **日期**: 2026-01-27

## 一、优化目标

| 算子 | 当前性能 | V16 目标 | 预期提升 |
|------|---------|----------|----------|
| Hash Join (INNER) | 0.93x | **1.1x+** | 20%+ |
| SEMI Join | 0.68x | **0.9x+** | 35%+ |
| Filter 10M | 2.6x | **4x+** | 50%+ |

---

## 二、问题分析

### P0: Hash Join (0.93x → 1.1x+)

**当前实现瓶颈** (V6):
1. 线性探测在冲突时效率下降
2. 逐键探测无法利用 SIMD 并行性
3. 预取距离固定，不适应不同数据分布

**DuckDB 优势分析**:
- 使用分区并行，cache line 对齐
- 批量操作，向量化处理
- 动态调整策略

### P1: SEMI Join (0.68x → 0.9x+)

**当前实现瓶颈** (V10):
1. 与 INNER JOIN 使用相同的探测逻辑
2. 没有利用 SEMI 的 "一次匹配即可" 特性
3. 没有预过滤，100% 探测哈希表

**优化机会**:
- Bloom Filter 预过滤 (减少 90%+ 哈希表访问)
- 早期终止 (找到一个匹配即跳过整批)

### P2: Filter 10M (2.6x → 4x+)

**当前实现瓶颈** (V3 vs V9):
1. V9 的多级预取在大数据上反而更慢
2. 预取污染 L2/L3 缓存
3. 选择率 50% 时输出带宽成为瓶颈

**优化机会**:
- 自适应预取距离 (根据数据量调整)
- 分块处理 (保持 L2 cache 热度)
- 批量索引写入 (减少内存写操作)

---

## 三、V16 技术方案

### P0: Hash Join V16 - SIMD 批量探测

**核心思想**: 一次探测 4 个键，使用 SIMD 比较 4 个哈希槽

```cpp
// V16: SIMD 4-way 并行探测
size_t probe_simd_4way(const int32_t* probe_keys, size_t probe_count,
                        uint32_t* out_build, uint32_t* out_probe) const {
    size_t match_count = 0;

    for (size_t i = 0; i + 4 <= probe_count; i += 4) {
        // 1. 批量计算 4 个哈希值
        uint32x4_t hashes = hash_batch_4_neon(probe_keys + i);

        // 2. 计算初始探测位置
        uint32x4_t positions = vandq_u32(hashes, vdupq_n_u32(mask_));

        // 3. SIMD 并行比较
        int32x4_t keys_to_find = vld1q_s32(probe_keys + i);

        // 4. 对每个位置，加载哈希表槽并比较
        alignas(16) int32_t slots[4];
        for (int j = 0; j < 4; ++j) {
            uint32_t pos = vgetq_lane_u32(positions, j);
            slots[j] = keys_[pos];
        }

        int32x4_t slot_keys = vld1q_s32(slots);
        uint32x4_t matches = vceqq_s32(slot_keys, keys_to_find);

        // 5. 提取匹配并写入结果
        uint32_t match_mask = extract_mask_4(matches);
        // ... 处理匹配
    }
}
```

**预期收益**:
- 减少分支预测失败
- 提高 ILP (指令级并行)
- 更好的缓存预取模式

### P1: SEMI Join V16 - Bloom Filter 预过滤

**核心思想**: 先用 Bloom Filter 过滤掉大部分非匹配键

```cpp
// V16: Bloom Filter + 早期终止
size_t probe_semi_bloom(const int32_t* probe_keys, size_t probe_count,
                         uint32_t* out_probe) const {
    // 构建阶段: 建立 Bloom Filter
    constexpr size_t BLOOM_SIZE = 65536;  // 64KB，适合 L1 缓存
    alignas(64) uint64_t bloom[BLOOM_SIZE / 64] = {0};

    // 使用双哈希减少假阳性
    for (size_t i = 0; i < build_count_; ++i) {
        uint32_t h1 = hash_key(keys_[i]);
        uint32_t h2 = h1 * 0x9E3779B1;  // 黄金比例哈希

        bloom[(h1 >> 6) % (BLOOM_SIZE/64)] |= (1ULL << (h1 & 63));
        bloom[(h2 >> 6) % (BLOOM_SIZE/64)] |= (1ULL << (h2 & 63));
    }

    // 探测阶段: 先查 Bloom，再查哈希表
    size_t match_count = 0;
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t key = probe_keys[i];
        uint32_t h1 = hash_key(key);
        uint32_t h2 = h1 * 0x9E3779B1;

        // Bloom Filter 快速检查
        bool maybe_exists =
            (bloom[(h1 >> 6) % (BLOOM_SIZE/64)] & (1ULL << (h1 & 63))) &&
            (bloom[(h2 >> 6) % (BLOOM_SIZE/64)] & (1ULL << (h2 & 63)));

        if (!maybe_exists) continue;  // 确定不存在，跳过

        // 可能存在，查哈希表确认
        size_t idx = h1 & mask_;
        while (keys_[idx] != EMPTY_KEY) {
            if (keys_[idx] == key) {
                out_probe[match_count++] = static_cast<uint32_t>(i);
                break;  // SEMI: 找到即可
            }
            idx = (idx + 1) & mask_;
        }
    }

    return match_count;
}
```

**预期收益**:
- 10% 匹配率场景: 过滤掉 ~85% 的探测
- Bloom Filter 在 L1 缓存中，访问极快
- 显著减少哈希表随机访问

### P2: Filter V16 - 自适应流式处理

**核心思想**: 根据数据量和选择率动态调整策略

```cpp
// V16: 自适应 Filter
size_t filter_i32_v16_adaptive(const int32_t* input, size_t count,
                                CompareOp op, int32_t value,
                                uint32_t* out_indices) {
    // 策略选择
    if (count <= 100000) {
        // 小数据: 直接 SIMD，无预取
        return filter_v16_simple(input, count, op, value, out_indices);
    }

    if (count <= 1000000) {
        // 中等数据: 短距离预取 + 单线程
        return filter_v16_prefetch_near(input, count, op, value, out_indices);
    }

    // 大数据: 分块处理，保持 L2 热度
    constexpr size_t BLOCK_SIZE = 256 * 1024;  // 256K 元素 = 1MB
    size_t total_count = 0;

    // 每块独立处理，避免缓存污染
    for (size_t offset = 0; offset < count; offset += BLOCK_SIZE) {
        size_t block_count = std::min(BLOCK_SIZE, count - offset);

        // 无预取处理当前块
        size_t matches = filter_v16_no_prefetch(
            input + offset, block_count, op, value,
            out_indices + total_count, offset);

        total_count += matches;
    }

    return total_count;
}

// 无预取版本: 依赖硬件预取器
size_t filter_v16_no_prefetch(const int32_t* input, size_t count,
                               CompareOp op, int32_t value,
                               uint32_t* out_indices, size_t base_idx) {
    int32x4_t threshold = vdupq_n_s32(value);
    size_t out_count = 0;

    // 16 元素展开，不手动预取
    for (size_t i = 0; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32x4_t m0 = vcgtq_s32(d0, threshold);
        uint32x4_t m1 = vcgtq_s32(d1, threshold);
        uint32x4_t m2 = vcgtq_s32(d2, threshold);
        uint32x4_t m3 = vcgtq_s32(d3, threshold);

        // 组合掩码
        uint32_t bits = extract_mask_4(m0) |
                       (extract_mask_4(m1) << 4) |
                       (extract_mask_4(m2) << 8) |
                       (extract_mask_4(m3) << 12);

        if (bits == 0) continue;

        // CTZ 循环写入索引
        uint32_t base = static_cast<uint32_t>(base_idx + i);
        while (bits) {
            uint32_t pos = __builtin_ctz(bits);
            out_indices[out_count++] = base + pos;
            bits &= bits - 1;
        }
    }

    return out_count;
}
```

**预期收益**:
- 小数据: 无预取开销
- 中等数据: 适度预取
- 大数据: 分块保持缓存热度，避免污染

---

## 四、实现计划

| 阶段 | 任务 | 文件 | 预期收益 |
|------|------|------|----------|
| 1 | SEMI Join Bloom Filter | `hash_join_v16.cpp` | 35%+ |
| 2 | Hash Join SIMD 4-way | `hash_join_v16.cpp` | 20%+ |
| 3 | Filter 自适应 | `simd_filter_v16.cpp` | 50%+ |
| 4 | 基准测试验证 | `v16_benchmark.cpp` | - |

---

## 五、技术细节

### 5.1 Bloom Filter 参数

```
Build 表大小: 100K
Bloom Filter 大小: 64KB (65536 bits)
每元素 bits: 0.65 bits (使用双哈希)
假阳性率: ~15%

实际效果:
- 10% 匹配率: 过滤 ~85% 探测
- 50% 匹配率: 过滤 ~35% 探测
```

### 5.2 SIMD 4-way 探测流程

```
输入: 4 个 probe 键
步骤:
1. SIMD 计算 4 个哈希值 (4x CRC32)
2. SIMD 计算 4 个初始位置 (4x AND mask)
3. 加载 4 个槽位的键
4. SIMD 比较 4 对键
5. 提取匹配掩码
6. 对冲突进行线性探测回退
```

### 5.3 Filter 分块策略

```
块大小: 1MB (256K int32)
原因:
- L2 缓存: 4MB
- 留出 3MB 给其他数据
- 硬件预取器在 1MB 范围内高效

处理流程:
1. 处理第 1 块 → L2 热
2. 处理第 2 块 → 第 1 块逐出
3. ...
```

---

## 六、验证计划

### 6.1 单元测试

- 正确性: 与标量实现对比
- 边界: 空输入、单元素、极端选择率

### 6.2 性能基准

| 测试场景 | 数据量 | 匹配率 |
|----------|--------|--------|
| Small | 100K | 10%/50% |
| Medium | 1M | 10%/50% |
| Large | 10M | 10%/50% |

### 6.3 与 DuckDB 对比

```bash
# 编译
cmake -DCMAKE_BUILD_TYPE=Release ..
make v16_benchmark

# 运行
./v16_benchmark --iterations 30 --warmup 3
```

---

## 七、风险与备选方案

### 7.1 Bloom Filter 假阳性

**风险**: 假阳性率过高导致收益下降

**缓解**:
- 动态调整 Bloom Filter 大小
- 当 build 表过大时跳过 Bloom

### 7.2 SIMD 4-way 冲突

**风险**: 高冲突率时 4-way 不如线性探测

**缓解**:
- 监测冲突率，动态切换策略
- 使用 Robin Hood 哈希减少探测长度

### 7.3 Filter 分块开销

**风险**: 分块边界处理增加复杂度

**缓解**:
- 仅在 10M+ 数据时启用
- 块大小对齐到 SIMD 宽度
