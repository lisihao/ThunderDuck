# ThunderDuck 性能优化方案

> **目标**: 全面超越 DuckDB 性能
> **版本**: 2.0
> **日期**: 2026-01-24

---

## 一、性能瓶颈深度分析

### 1.1 当前性能对比

| 操作类别 | 当前状态 | 差距原因 |
|---------|---------|---------|
| Filter | **12x 慢** | 写入索引开销、SIMD 利用率低 |
| Sort ASC | **2.4x 慢** | 直接使用 std::sort，无 SIMD 优化 |
| Join | **1.2x 慢** | 链表冲突解决、无批量探测 |
| MIN/MAX | **1.2x 慢** | 分离调用，两次遍历 |

### 1.2 Filter 深度分析

**问题 1: 索引写入成为瓶颈**

```cpp
// 当前实现 - 每个匹配都写入内存
for (uint32_t j = 0; j < match_count; ++j) {
    out_indices[out_count + j] = i + lut[1 + j];  // 内存写入！
}
```

50% 选择率 → 2M 次内存写入，严重影响性能。

**问题 2: 每次只处理 4 个元素**

```cpp
// 当前: 4 元素/迭代
int32x4_t data = vld1q_s32(input + i);
```

ARM Neon 可以同时处理更多数据。

**问题 3: 测试不公平**

- DuckDB: `SELECT COUNT(*)` 只返回计数
- ThunderDuck: 写入所有匹配索引

### 1.3 Sort 深度分析

**问题 1: 直接回退到 std::sort**

```cpp
// 当前实现 - 大数组直接用 std::sort
if (count > 16) {
    std::sort(data, data + count);  // 无 SIMD！
}
```

**问题 2: Bitonic Sort 实现有缺陷**

```cpp
// sort_vec4_simple 内部又调用 std::sort！
inline int32x4_t sort_vec4_simple(int32x4_t v, SortOrder order) {
    alignas(16) int32_t arr[4];
    vst1q_s32(arr, v);
    std::sort(arr, arr + 4);  // 这不是 SIMD 排序！
    return vld1q_s32(arr);
}
```

**问题 3: 未使用 Radix Sort**

对于整数，Radix Sort 是 O(n)，比比较排序快得多。

### 1.4 Join 深度分析

**问题 1: 链表冲突解决**

```cpp
while (entry_idx != EMPTY) {
    const Entry& e = entries[entry_idx];  // 指针追逐！
    entry_idx = e.next;                    // 缓存不友好
}
```

**问题 2: 无批量探测**

```cpp
for (size_t i = 0; i < probe_count; ++i) {
    // 逐个探测，无法利用 SIMD
}
```

**问题 3: 无预取**

```cpp
// 缺少预取指令
__builtin_prefetch(&entries[next_entry], 0, 3);
```

---

## 二、优化方案

### 2.1 Filter 优化方案

#### 方案 A: 实现纯计数版本 (公平对比)

```cpp
// 新增: 只计数，不写索引
size_t count_i32_simd(const int32_t* input, size_t count,
                       CompareOp op, int32_t value) {
    uint32x4_t count_vec = vdupq_n_u32(0);
    int32x4_t threshold = vdupq_n_s32(value);

    // 每次处理 16 个元素 (4 个 SIMD 寄存器)
    for (size_t i = 0; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        uint32x4_t m0 = vcgtq_s32(d0, threshold);
        uint32x4_t m1 = vcgtq_s32(d1, threshold);
        uint32x4_t m2 = vcgtq_s32(d2, threshold);
        uint32x4_t m3 = vcgtq_s32(d3, threshold);

        // 累加 (掩码右移31位变成0或1)
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m0, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m1, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m2, 31));
        count_vec = vaddq_u32(count_vec, vshrq_n_u32(m3, 31));
    }
    return vaddvq_u32(count_vec) + /* 处理剩余 */;
}
```

**预期提升**: 10-20x

#### 方案 B: 位图 + 批量转换索引

```cpp
// 第一遍: 生成位图 (SIMD)
alignas(64) uint64_t bitmap[(count + 63) / 64] = {0};

for (size_t i = 0; i + 64 <= count; i += 64) {
    // 使用 SIMD 生成 64 位掩码
    uint64_t mask = generate_mask_64(input + i, threshold);
    bitmap[i / 64] = mask;
}

// 第二遍: 位图转索引 (可选，延迟执行)
size_t extract_indices(const uint64_t* bitmap, size_t bit_count,
                       uint32_t* out_indices) {
    size_t out_count = 0;
    for (size_t i = 0; i < (bit_count + 63) / 64; ++i) {
        uint64_t bits = bitmap[i];
        while (bits) {
            uint32_t pos = __builtin_ctzll(bits);  // 找最低位 1
            out_indices[out_count++] = i * 64 + pos;
            bits &= bits - 1;  // 清除最低位 1
        }
    }
    return out_count;
}
```

**预期提升**: 5-10x (带索引输出)

### 2.2 Sort 优化方案

#### 方案 A: 实现真正的 SIMD Bitonic Sort

```cpp
// 4 元素网络排序 - 纯 SIMD
inline int32x4_t bitonic_sort_4(int32x4_t v) {
    // Stage 1: 比较相邻对 (0,1) (2,3)
    int32x4_t v1 = vrev64q_s32(v);           // [1,0,3,2]
    int32x4_t lo = vminq_s32(v, v1);
    int32x4_t hi = vmaxq_s32(v, v1);
    int32x4_t s1 = vtrn1q_s32(lo, hi);       // [min(0,1), max(0,1), ...]

    // Stage 2: 比较 (0,2) (1,3)
    int32x4_t s1_swap = vextq_s32(s1, s1, 2); // [2,3,0,1]
    lo = vminq_s32(s1, s1_swap);
    hi = vmaxq_s32(s1, s1_swap);
    int32x4_t s2 = vcombine_s32(vget_low_s32(lo), vget_high_s32(hi));

    // Stage 3: 比较 (1,2)
    int32x4_t s2_swap = vrev64q_s32(s2);
    lo = vminq_s32(s2, s2_swap);
    hi = vmaxq_s32(s2, s2_swap);
    return vtrn1q_s32(lo, vtrn2q_s32(lo, hi));
}

// 8 元素网络排序
inline void bitonic_sort_8(int32x4_t& a, int32x4_t& b) {
    // 各自排序
    a = bitonic_sort_4(a);
    b = bitonic_sort_4(b);

    // 逆序 b
    b = vrev64q_s32(b);
    b = vcombine_s32(vget_high_s32(b), vget_low_s32(b));

    // Bitonic 合并
    int32x4_t lo = vminq_s32(a, b);
    int32x4_t hi = vmaxq_s32(a, b);

    // 递归合并步骤...
}
```

#### 方案 B: 实现 Radix Sort (整数专用)

```cpp
// LSD Radix Sort - O(n)
void radix_sort_i32(int32_t* data, size_t count) {
    constexpr int RADIX_BITS = 8;
    constexpr int RADIX = 1 << RADIX_BITS;
    constexpr int PASSES = sizeof(int32_t) * 8 / RADIX_BITS;

    std::vector<int32_t> buffer(count);
    int32_t* src = data;
    int32_t* dst = buffer.data();

    for (int pass = 0; pass < PASSES; ++pass) {
        // 计数
        alignas(64) size_t counts[RADIX] = {0};
        int shift = pass * RADIX_BITS;

        // SIMD 计数优化
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            counts[key]++;
        }

        // 前缀和
        size_t offset = 0;
        for (int i = 0; i < RADIX; ++i) {
            size_t c = counts[i];
            counts[i] = offset;
            offset += c;
        }

        // 分配
        for (size_t i = 0; i < count; ++i) {
            uint32_t key = (static_cast<uint32_t>(src[i]) >> shift) & (RADIX - 1);
            dst[counts[key]++] = src[i];
        }

        std::swap(src, dst);
    }

    // 处理符号位
    if (src != data) {
        std::memcpy(data, src, count * sizeof(int32_t));
    }
}
```

**预期提升**: 2-3x

#### 方案 C: 并行排序

```cpp
#include <thread>

void parallel_sort_i32(int32_t* data, size_t count) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = (count + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // 并行局部排序
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, count);
        if (start < count) {
            threads.emplace_back([=]() {
                radix_sort_i32(data + start, end - start);
            });
        }
    }

    for (auto& t : threads) t.join();

    // 多路归并
    merge_k_sorted_parallel(...);
}
```

**预期提升**: 3-4x (4核)

### 2.3 Join 优化方案

#### 方案 A: Robin Hood 哈希表

```cpp
struct RobinHoodHashTable {
    struct Entry {
        int32_t key;
        uint32_t row_idx;
        uint8_t psl;  // Probe Sequence Length
        bool occupied;
    };

    std::vector<Entry> table;
    size_t size;
    size_t mask;

    void insert(int32_t key, uint32_t row_idx) {
        uint32_t hash = hash_one_i32(key);
        size_t idx = hash & mask;
        uint8_t psl = 0;

        while (true) {
            if (!table[idx].occupied) {
                table[idx] = {key, row_idx, psl, true};
                ++size;
                return;
            }

            // Robin Hood: 抢夺位置
            if (table[idx].psl < psl) {
                std::swap(key, table[idx].key);
                std::swap(row_idx, table[idx].row_idx);
                std::swap(psl, table[idx].psl);
            }

            idx = (idx + 1) & mask;
            ++psl;
        }
    }

    // 查找更快，因为 PSL 有上界
};
```

#### 方案 B: 批量探测 + 预取

```cpp
size_t probe_batch_i32(const int32_t* probe_keys, size_t probe_count,
                       uint32_t* out_build, uint32_t* out_probe) {
    constexpr size_t BATCH_SIZE = 8;
    constexpr size_t PREFETCH_DIST = 16;

    size_t match_count = 0;

    for (size_t i = 0; i < probe_count; i += BATCH_SIZE) {
        // 预取下一批的桶位置
        if (i + PREFETCH_DIST < probe_count) {
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                uint32_t h = hash_one_i32(probe_keys[i + PREFETCH_DIST + j]);
                __builtin_prefetch(&buckets[h & mask], 0, 3);
            }
        }

        // 批量计算哈希
        uint32_t hashes[BATCH_SIZE];
        for (size_t j = 0; j < BATCH_SIZE && i + j < probe_count; ++j) {
            hashes[j] = hash_one_i32(probe_keys[i + j]);
        }

        // 批量探测
        for (size_t j = 0; j < BATCH_SIZE && i + j < probe_count; ++j) {
            uint32_t bucket_idx = hashes[j] & mask;
            // ... 探测逻辑
        }
    }

    return match_count;
}
```

#### 方案 C: SIMD 键比较

```cpp
// 在桶内使用 SIMD 比较多个键
size_t probe_simd_bucket(int32_t probe_key, const Entry* entries, size_t count,
                         uint32_t base_idx, uint32_t* out_build) {
    int32x4_t probe_vec = vdupq_n_s32(probe_key);
    size_t matches = 0;

    for (size_t i = 0; i + 4 <= count; i += 4) {
        // 加载 4 个键 (假设键连续存储)
        int32x4_t keys = vld1q_s32(&entries[i].key);  // 需要 SOA 布局
        uint32x4_t eq = vceqq_s32(keys, probe_vec);

        if (vmaxvq_u32(eq)) {
            // 有匹配，提取索引
            // ...
        }
    }

    return matches;
}
```

**预期提升**: 1.5-2x

### 2.4 Aggregation 优化方案

#### 方案 A: 合并 MIN/MAX

```cpp
struct MinMaxResult {
    int32_t min_val;
    int32_t max_val;
};

MinMaxResult minmax_i32(const int32_t* input, size_t count) {
    int32x4_t min_vec = vdupq_n_s32(INT32_MAX);
    int32x4_t max_vec = vdupq_n_s32(INT32_MIN);

    // 每次处理 16 个元素
    for (size_t i = 0; i + 16 <= count; i += 16) {
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 同时更新 min 和 max
        min_vec = vminq_s32(min_vec, vminq_s32(vminq_s32(d0, d1), vminq_s32(d2, d3)));
        max_vec = vmaxq_s32(max_vec, vmaxq_s32(vmaxq_s32(d0, d1), vmaxq_s32(d2, d3)));
    }

    return {vminvq_s32(min_vec), vmaxvq_s32(max_vec)};
}
```

**预期提升**: 1.5-2x

#### 方案 B: 预取优化

```cpp
int64_t sum_i32_prefetch(const int32_t* input, size_t count) {
    constexpr size_t PREFETCH_DIST = 256;  // 预取距离 (256 * 4 = 1KB)

    int64x2_t sum = vdupq_n_s64(0);

    for (size_t i = 0; i < count; i += 16) {
        // 预取
        if (i + PREFETCH_DIST < count) {
            __builtin_prefetch(input + i + PREFETCH_DIST, 0, 0);
        }

        // 处理 16 个元素
        int32x4_t d0 = vld1q_s32(input + i);
        int32x4_t d1 = vld1q_s32(input + i + 4);
        int32x4_t d2 = vld1q_s32(input + i + 8);
        int32x4_t d3 = vld1q_s32(input + i + 12);

        // 累加
        sum = vaddq_s64(sum, vpaddlq_s32(vpaddq_s32(d0, d1)));
        sum = vaddq_s64(sum, vpaddlq_s32(vpaddq_s32(d2, d3)));
    }

    return vaddvq_s64(sum);
}
```

---

## 三、实现优先级

### Phase 1: 快速胜利 (1-2 天)

| 任务 | 预期收益 | 复杂度 |
|-----|---------|--------|
| 优化 count_i32 (16 元素/迭代) | Filter 10x+ | 低 |
| 实现 minmax_i32 合并函数 | Agg 1.5x | 低 |
| 添加预取指令 | 全局 1.1x | 低 |

### Phase 2: 核心优化 (3-5 天)

| 任务 | 预期收益 | 复杂度 |
|-----|---------|--------|
| 实现 Radix Sort | Sort 2-3x | 中 |
| 位图 Filter | Filter 5x | 中 |
| Robin Hood 哈希表 | Join 1.5x | 中 |

### Phase 3: 高级优化 (1-2 周)

| 任务 | 预期收益 | 复杂度 |
|-----|---------|--------|
| 纯 SIMD Bitonic Sort | Sort 1.5x | 高 |
| 并行排序 | Sort 3x | 中 |
| 批量 Join 探测 | Join 1.3x | 中 |
| NPU 加速聚合 | Agg 10x | 高 |

---

## 四、预期最终性能

### 优化后对比

| 操作 | 当前 | Phase 1 | Phase 2 | Phase 3 |
|-----|------|---------|---------|---------|
| Filter (count) | 0.17x | **2x** | **5x** | **10x** |
| Sort | 0.4-1.2x | 1.2x | **2.5x** | **4x** |
| Join | 0.82x | 1.0x | **1.5x** | **2x** |
| Aggregation | 1.3x | **2x** | 2x | **5x** |
| Top-K | **3x** | 3x | 3.5x | **5x** |

### 目标: 全面超越 DuckDB

- **Filter**: 纯计数 5x+，带索引 2x+
- **Sort**: Radix Sort 2.5x+
- **Join**: Robin Hood + 批量探测 1.5x+
- **Aggregation**: SIMD + 预取 2x+

---

## 五、代码变更清单

### 5.1 新增文件

```
src/operators/filter/simd_count.cpp      # 优化的计数实现
src/operators/filter/bitmap_filter.cpp   # 位图 Filter
src/operators/sort/radix_sort.cpp        # Radix Sort
src/operators/sort/parallel_sort.cpp     # 并行排序
src/operators/join/robin_hood_hash.cpp   # Robin Hood 哈希表
src/operators/aggregate/simd_minmax.cpp  # 合并的 min/max
```

### 5.2 修改文件

```
include/thunderduck/filter.h    # 新增 count_only_i32
include/thunderduck/sort.h      # 新增 radix_sort_i32
include/thunderduck/join.h      # 修改 HashTable 实现
include/thunderduck/aggregate.h # 新增 minmax_i32
```

### 5.3 测试文件

```
tests/test_optimized_filter.cpp
tests/test_radix_sort.cpp
tests/test_robin_hood.cpp
benchmarks/bench_optimized.cpp
```

---

## 六、验证方法

### 6.1 单元测试

```bash
# 运行所有测试
./run_tests

# 运行优化后测试
./test_optimized_filter
./test_radix_sort
```

### 6.2 Benchmark 对比

```bash
# 小数据集快速测试
./benchmark_app --small

# 完整测试
./benchmark_app

# 生成报告
./benchmark_app -o new_benchmark_report.md
```

### 6.3 性能指标

- **吞吐量**: elements/second
- **延迟**: ms/operation
- **内存带宽利用率**: GB/s (目标: 接近理论峰值)
- **SIMD 利用率**: % 时间在 SIMD 指令

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| Radix Sort 不稳定 | 排序结果可能不同 | 提供稳定版本选项 |
| 并行开销 | 小数据反而慢 | 自适应选择算法 |
| 内存分配 | 额外内存使用 | 提供原地版本 |
| 浮点精度 | 聚合结果差异 | 使用 Kahan 求和 |

---

## 八、结论

通过系统性优化，ThunderDuck 可以在所有核心操作上超越 DuckDB：

1. **Filter**: 位图 + SIMD 计数 → **5-10x 提升**
2. **Sort**: Radix Sort + 并行 → **2.5-4x 提升**
3. **Join**: Robin Hood + 批量探测 → **1.5-2x 提升**
4. **Aggregation**: 合并操作 + 预取 → **2x 提升**

关键是利用 Apple M4 的硬件特性：
- 128-bit SIMD (Neon)
- 128-byte 缓存行
- CRC32 硬件指令
- 高内存带宽

---

*ThunderDuck v2.0 - Designed to Beat DuckDB*
