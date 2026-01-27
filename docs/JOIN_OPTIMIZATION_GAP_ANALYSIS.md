# Hash Join 优化差距分析

> 研究文档 vs 当前实现对比 | 日期: 2026-01-26

## 一、总体对比矩阵

| 优化项 | 研究文档提出 | 当前实现状态 | 差距 |
|--------|-------------|-------------|------|
| **SIMD 并行哈希** | ✓ CRC32/分段哈希 | ✓ CRC32 硬件 | 完成 |
| **SIMD 批量比较** | ✓ 4-8元素并行 | ✓ 4元素 Neon | 完成 |
| **Radix 分区** | ✓ 缓存适配 | ✓ 16/256分区 | 完成 |
| **缓存行对齐** | ✓ 128字节 | ✓ alignas(128) | 完成 |
| **多线程并行** | ✓ 分区并行 | ✓ Morsel-Driven | 完成 |
| **Bloom Filter** | ○ 未明确提出 | ✓ 已实现 | 超额 |
| **完美哈希** | ○ 未明确提出 | ✓ 已实现 | 超额 |
| **Sort-Merge Join** | ✓ 明确要求 | ✗ 未实现 | **缺失** |
| **范围连接** | ✓ BETWEEN优化 | ✗ 未实现 | **缺失** |
| **字符串键优化** | ✓ 16字节批处理 | ✗ 未实现 | **缺失** |
| **LEFT/RIGHT/FULL** | ○ 隐含需求 | ✗ TODO状态 | **缺失** |
| **SEMI/ANTI JOIN** | ○ 隐含需求 | ✗ 无优化 | **缺失** |
| **原子无锁插入** | ✓ CAS插入 | △ 分区避免 | 部分 |
| **GPU 实际可用** | ✓ 超大数据 | △ 阈值500M | 需调整 |
| **NPU 加速** | ○ 未明确 | △ 未完成 | 待验证 |

**图例**: ✓完成 △部分 ✗未实现 ○未提及

---

## 二、已完成优化详细分析

### 2.1 SIMD 并行哈希计算 ✅

**研究文档要求**:
> 使用SIMD批量计算哈希，利用ARMv8提供的CRC32指令

**当前实现** (`hash_join_v3.cpp`):
```cpp
inline uint32_t crc32_hash(int32_t key) {
    return __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(key));
}

// 批量 8 元素展开
inline void hash_8_keys(const int32_t* keys, uint32_t* hashes) {
    hashes[0] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[0]));
    hashes[1] = __crc32cw(0xFFFFFFFF, static_cast<uint32_t>(keys[1]));
    // ... 8路展开
}
```

**性能**: 2 hash/周期，优于标量 MurmurHash 的 0.5 hash/周期

**状态**: ✅ 完全匹配研究目标

---

### 2.2 SIMD 批量比较 ✅

**研究文档要求**:
> 将哈希桶中的键值以SIMD寄存器批量载入，与当前探测键进行并行比较

**当前实现** (`hash_join_v3.cpp`):
```cpp
int32x4_t cand_vec = vld1q_s32(candidates + i);
int32x4_t probe_vec = vdupq_n_s32(probe_key);
uint32x4_t eq_mask = vceqq_s32(cand_vec, probe_vec);
```

**性能**: 4元素并行比较，比标量快 2-3x

**状态**: ✅ 完全实现

---

### 2.3 Radix 分区策略 ✅

**研究文档要求**:
> 先按照哈希值高位将build表和probe表分成若干分区，使每个分区能装入CPU缓存

**当前实现**:
```cpp
// v3: 16 分区 (4位)
constexpr int RADIX_BITS = 4;
constexpr size_t NUM_PARTITIONS = 16;

// v4: 自适应 8-256 分区
int radix_bits = select_adaptive_radix_bits(build_count + probe_count);
```

**分区大小计算**:
- 1M rows / 256 分区 = 4K rows/分区 = 16KB < L1 (192KB)

**状态**: ✅ 完成并增强 (自适应分区位数)

---

### 2.4 缓存与内存布局优化 ✅

**研究文档要求**:
> 针对M4的128B缓存行，让哈希桶大小接近缓存行大小

**当前实现**:
```cpp
constexpr size_t M4_CACHE_LINE = 128;

class SOAHashTable {
    alignas(M4_CACHE_LINE) std::vector<int32_t> keys_;
    std::vector<uint32_t> row_indices_;
    // SOA 布局，顺序访问
};
```

**状态**: ✅ 完全实现

---

### 2.5 多线程并行 ✅

**研究文档要求**:
> 开启多线程并行处理不同分区，利用M4的多核性能

**当前实现** (`hash_join_v3.cpp`):
```cpp
std::atomic<size_t> next_partition{0};
auto worker = [&]() {
    while (true) {
        size_t p = next_partition.fetch_add(1);
        if (p >= NUM_PARTITIONS) break;
        // 处理分区 p
    }
};
// 4 个 P-cores
```

**状态**: ✅ Morsel-Driven 动态负载均衡

---

## 三、未实现优化 - 差距分析

### 3.1 Sort-Merge Join ❌ (P1 高优先级)

**研究文档要求**:
> 对排序-合并连接，也可提供 ThunderSortMergeJoin 接口
> 对于排序-合并连接，当两个已排序列表进行比较时，可以每次取出多个元素并行比较大小

**当前状态**: 完全未实现

**实现建议**:
```cpp
// 建议 API
size_t sort_merge_join_i32(
    const int32_t* left_keys, size_t left_count,
    const int32_t* right_keys, size_t right_count,
    JoinType join_type,
    JoinResult* result);

// SIMD 优化点
// 1. 双指针扫描时 SIMD 批量比较
// 2. 利用已排序特性跳跃前进
// 3. 分块处理提高缓存命中
```

**优势场景**:
- 输入已排序
- 范围连接
- 不等值连接
- 内存受限场景

**预期收益**: 特定场景 **2-5x** 优于 Hash Join

---

### 3.2 范围连接 (Range Join) ❌ (P1 高优先级)

**研究文档要求**:
> 如果涉及范围连接（e.g. BETWEEN 或非等值条件），利用SIMD快速判断范围包含

**当前状态**: 完全未实现

**实现建议**:
```cpp
// 范围连接 API
size_t range_join_i32(
    const int32_t* left_keys, size_t left_count,
    const int32_t* right_lo, const int32_t* right_hi, size_t right_count,
    JoinResult* result);

// SIMD 范围检查
int32x4_t key = vld1q_s32(&left_keys[i]);
int32x4_t lo = vld1q_s32(&right_lo[j]);
int32x4_t hi = vld1q_s32(&right_hi[j]);
uint32x4_t ge_lo = vcgeq_s32(key, lo);
uint32x4_t le_hi = vcleq_s32(key, hi);
uint32x4_t in_range = vandq_u32(ge_lo, le_hi);
```

**预期收益**: 比逐个比较快 **3-4x**

---

### 3.3 字符串键 SIMD 优化 ❌ (P2 中优先级)

**研究文档要求**:
> 对于字符串等长键值，SIMD也可用于并行处理多个字符
> Neon每次处理16个字符字节

**当前状态**: 仅支持整数键

**实现建议**:
```cpp
// 16字节批量比较
inline bool simd_strcmp_16(const char* a, const char* b) {
    uint8x16_t va = vld1q_u8((const uint8_t*)a);
    uint8x16_t vb = vld1q_u8((const uint8_t*)b);
    uint8x16_t cmp = vceqq_u8(va, vb);
    return vmaxvq_u8(cmp) == 0xFF;  // 全相等
}

// 批量字符串哈希
inline uint32_t simd_string_hash(const char* str, size_t len) {
    uint32_t h = 0;
    for (size_t i = 0; i + 16 <= len; i += 16) {
        uint8x16_t v = vld1q_u8((const uint8_t*)str + i);
        // 累积哈希...
    }
    return h;
}
```

**预期收益**: 字符串键性能提升 **2-4x**

---

### 3.4 LEFT/RIGHT/FULL JOIN ❌ (P1 高优先级)

**研究文档隐含需求**: 完整的连接语义支持

**当前状态**: 仅 INNER JOIN，其他为 TODO

**实现建议**:
```cpp
// 添加 bitmap 跟踪匹配
std::vector<bool> build_matched(build_count, false);
std::vector<bool> probe_matched(probe_count, false);

// 探测时标记
for (size_t i = 0; i < match_count; i++) {
    build_matched[result->left_indices[i]] = true;
    probe_matched[result->right_indices[i]] = true;
}

// LEFT JOIN: 添加未匹配的 probe 行
if (join_type == JoinType::LEFT || join_type == JoinType::FULL) {
    for (size_t i = 0; i < probe_count; i++) {
        if (!probe_matched[i]) {
            result->left_indices[pos] = NULL_INDEX;
            result->right_indices[pos] = i;
            pos++;
        }
    }
}
```

**预期收益**: 功能完整性，无性能损失 (额外扫描 O(n))

---

### 3.5 SEMI/ANTI JOIN 优化 ❌ (P2 中优先级)

**研究文档隐含需求**: 半连接/反连接优化

**当前状态**: 使用通用 INNER JOIN 实现

**优化点**:
```cpp
// SEMI JOIN: 只需知道是否存在匹配，可提前退出
if (join_type == JoinType::SEMI) {
    for (size_t i = 0; i < probe_count; i++) {
        if (ht.contains(probe_keys[i])) {
            result->right_indices[pos++] = i;
            // 无需找所有匹配
        }
    }
}

// ANTI JOIN: 找不匹配的
if (join_type == JoinType::ANTI) {
    for (size_t i = 0; i < probe_count; i++) {
        if (!ht.contains(probe_keys[i])) {
            result->right_indices[pos++] = i;
        }
    }
}
```

**预期收益**: **30-50%** 减少比较和输出操作

---

### 3.6 原子无锁哈希表插入 △ (P3 低优先级)

**研究文档要求**:
> 采用无锁算法（如基于原子操作的开放地址插入）提高并发下的哈希表构建速度

**当前状态**: 使用分区策略避免并发插入
- 每线程处理独立分区
- 无跨分区写入

**实现建议** (如需真正并发):
```cpp
// CAS 插入
bool try_insert(Entry* entries, size_t idx, Entry new_entry) {
    Entry expected = {EMPTY_KEY, 0};
    return __atomic_compare_exchange_n(
        &entries[idx], &expected, new_entry,
        false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
```

**评估**: 当前分区策略已足够高效，无锁插入可能带来额外复杂性

---

### 3.7 GPU 策略阈值调整 △ (P2 中优先级)

**研究文档要求**:
> GPU 在超大数据场景应发挥作用

**当前状态**: 阈值 500M 过高，实际不触发

**建议调整**:
```cpp
// 当前
constexpr size_t GPU_MIN_TOTAL = 500000000;  // 500M - 太高

// 建议
constexpr size_t GPU_MIN_TOTAL = 10000000;   // 10M
constexpr size_t GPU_MIN_PROBE = 1000000;    // 1M
```

**但需先验证**: GPU 实现是否真的比 CPU 快

---

## 四、优先级排序建议

### P0 - 紧急 (功能缺失)

| 优化项 | 工作量 | 收益 |
|--------|--------|------|
| LEFT/RIGHT/FULL JOIN | 2-3天 | 功能完整 |

### P1 - 高优先级 (性能显著)

| 优化项 | 工作量 | 收益 |
|--------|--------|------|
| Sort-Merge Join | 3-5天 | 特定场景 2-5x |
| Range Join | 2-3天 | 范围查询 3-4x |
| SEMI/ANTI 优化 | 1天 | 30-50% |

### P2 - 中优先级 (扩展能力)

| 优化项 | 工作量 | 收益 |
|--------|--------|------|
| 字符串键 SIMD | 3-5天 | 字符串键 2-4x |
| GPU 阈值调整 | 1天 | 超大数据场景 |
| 验证 NPU 策略 | 2-3天 | 待定 |

### P3 - 低优先级 (精细优化)

| 优化项 | 工作量 | 收益 |
|--------|--------|------|
| 无锁哈希插入 | 3-5天 | 可能 10-20% |
| 混合策略执行 | 5-7天 | 理论最优 |
| 动态阈值学习 | 5-7天 | 5-10% |

---

## 五、实现路线图建议

### 阶段 1: V10 - 完整连接语义 (1周)
- LEFT/RIGHT/FULL JOIN 支持
- SEMI/ANTI JOIN 优化
- 单元测试完善

### 阶段 2: V11 - Sort-Merge Join (1-2周)
- 基础 Sort-Merge Join 实现
- SIMD 优化版本
- 范围连接支持
- 策略选择器集成

### 阶段 3: V12 - 字符串与混合类型 (2周)
- 字符串键 SIMD 哈希
- 字符串键 SIMD 比较
- 混合类型支持

### 阶段 4: V13 - GPU 实际可用 (2周)
- GPU 实现性能验证
- 阈值调整
- NPU 策略完善

---

## 六、总结

**当前实现覆盖率**: 研究文档 **~65%**

**主要差距**:
1. Sort-Merge Join - 完全缺失
2. 范围连接 - 完全缺失
3. 字符串键优化 - 完全缺失
4. LEFT/RIGHT/FULL JOIN - 仅 TODO
5. GPU 实际不可用 - 阈值过高

**已超额实现**:
1. Bloom Filter 预过滤
2. 完美哈希表 (小整数键)
3. 自适应策略选择
4. Robin Hood 哈希

**建议优先实现**:
1. **LEFT/RIGHT/FULL JOIN** - 基础功能
2. **SEMI/ANTI 优化** - 快速收益
3. **Sort-Merge Join** - 新场景支持
