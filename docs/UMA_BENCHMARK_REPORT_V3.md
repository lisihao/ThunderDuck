# ThunderDuck UMA 综合基准测试报告 v3

> **测试日期**: 2026-01-25
> **测试平台**: Apple M4 (macOS 14.0+)
> **内存**: 统一内存架构 (UMA)

---

## 一、测试概述

### 测试目标
- 评估各算子 CPU SIMD / GPU Metal 性能
- 对比 ThunderDuck v3 与 DuckDB 原版
- 识别优化瓶颈和机会点

### 测试环境
| 配置项 | 值 |
|--------|-----|
| CPU | Apple M4 (高性能核心) |
| GPU | Apple M4 GPU (Metal 3) |
| 内存带宽 | ~100 GB/s (理论) |
| L1 Cache | 128KB/核心 |
| L2 Cache | 16MB 共享 |

---

## 二、Filter 算子性能

### SQL 语义
```sql
SELECT * FROM table WHERE column > threshold
```

### 测试结果

| 测试ID | 数据量 | 选择率 | 加速器 | 吞吐量 | 执行时间 | vs DuckDB |
|--------|--------|--------|--------|--------|----------|-----------|
| F1 | 1M | 50% | CPU SIMD | 2,648 M/s | 0.38 ms | 1.2x |
| F2 | 5M | 50% | CPU SIMD | 2,583 M/s | 1.94 ms | 1.3x |
| F3 | 10M | 50% | CPU SIMD | 2,583 M/s | 3.87 ms | 1.2x |
| F4 | 50M | 50% | CPU SIMD | 2,534 M/s | 19.7 ms | 1.1x |
| **F4** | **50M** | **50%** | **GPU Metal** | **6,117 M/s** | **8.2 ms** | **2.7x** |

### 带宽分析
| 加速器 | 实测带宽 | 理论峰值 | 利用率 |
|--------|----------|----------|--------|
| CPU SIMD | 10.1 GB/s | 100 GB/s | 10.1% |
| GPU Metal | 24.5 GB/s | 100 GB/s | 24.5% |

### 优化建议
1. **GPU 启用阈值**: 当前 10M 行时 GPU 开始有优势，建议阈值设为 5M
2. **带宽利用率低**: CPU 仅 10%，可通过多线程并行提升
3. **选择率敏感**: 低选择率时输出写入成为瓶颈

---

## 三、Aggregate 算子性能

### SQL 语义
```sql
SELECT SUM(col), MIN(col), MAX(col), AVG(col), COUNT(*) FROM table
```

### 测试结果

| 测试ID | 数据量 | 聚合函数 | 加速器 | 吞吐量 | 带宽 | vs DuckDB |
|--------|--------|----------|--------|--------|------|-----------|
| A1 | 1M | SUM | CPU SIMD | 29,586 M/s | 118 GB/s | 1.2x |
| A2 | 10M | SUM | CPU SIMD | 25,975 M/s | 104 GB/s | 1.1x |
| A3 | 50M | SUM | CPU SIMD | 24,065 M/s | 96 GB/s | 1.1x |
| A4 | 50M | MIN | CPU SIMD | 23,756 M/s | 95 GB/s | 1.1x |
| A5 | 50M | MAX | CPU SIMD | 23,511 M/s | 94 GB/s | 1.1x |
| **A6** | **50M** | **ALL** | **GPU Fused** | **26,000 M/s** | **104 GB/s** | **1.15x** |

### 带宽分析
| 加速器 | 实测带宽 | 理论峰值 | 利用率 |
|--------|----------|----------|--------|
| CPU SIMD | 94-118 GB/s | 100 GB/s | **94-118%** |
| GPU Fused | 104 GB/s | 100 GB/s | 104% |

### 关键发现
- **CPU 已达内存带宽极限** (94-118%)
- GPU 融合 kernel 提供 SUM+MIN+MAX 单次遍历
- 超过 100% 利用率说明缓存命中率高

### 优化建议
1. **已接近理论极限**: Aggregate 优化空间有限
2. **融合 kernel 有效**: 减少内存访问次数
3. **不建议 GPU 加速**: CPU SIMD 已足够高效

---

## 四、Join 算子性能 (核心优化)

### SQL 语义
```sql
SELECT * FROM probe_table p
JOIN build_table b ON p.key = b.key
```

### 测试结果

| 测试ID | Build | Probe | 匹配数 | 加速器 | 吞吐量 | 执行时间 | vs v3 | vs DuckDB |
|--------|-------|-------|--------|--------|--------|----------|-------|-----------|
| J1 | 10K | 100K | 10K | CPU v3 | 466 M/s | 0.21 ms | 1.0x | 0.9x |
| J2 | 100K | 1M | 100K | CPU v3 | 329 M/s | 3.0 ms | 1.0x | 1.1x |
| **J2** | **100K** | **1M** | **100K** | **GPU Metal** | **575 M/s** | **1.7 ms** | **1.75x** | **1.9x** |
| J3 | 1M | 10M | 1M | CPU v3 | 233 M/s | 47 ms | 1.0x | 1.1x |
| **J3** | **1M** | **10M** | **1M** | **GPU Metal** | **947 M/s** | **12 ms** | **4.07x** | **4.5x** |
| J4 | 5M | 50M | 5M | CPU v3 | 287 M/s | 174 ms | 1.0x | 1.0x |
| **J4** | **5M** | **50M** | **5M** | **GPU Metal** | **367 M/s** | **136 ms** | **1.28x** | **1.3x** |

### GPU 优化分析

| 规模 | GPU 加速比 | 瓶颈分析 |
|------|-----------|---------|
| 100K×1M | 1.75x | 数据量不足，GPU 启动开销 |
| **1M×10M** | **4.07x** | **最佳工作点** |
| 5M×50M | 1.28x | 内存带宽饱和 |

### 关键技术优化
1. **Threadgroup 前缀和**: 减少原子争用从 O(n) 到 O(n/256)
2. **统一内存零拷贝**: 避免 CPU-GPU 数据传输
3. **批量写入**: 本地缓存 16 个结果后批量写入

### 优化建议
1. **J3 是最佳目标**: 4.07x 加速，继续优化可达 5-6x
2. **J4 内存瓶颈**: 考虑分批处理或 radix 分区
3. **哈希表优化**: 当前 chaining，可改为 open addressing

---

## 五、TopK 算子性能

### SQL 语义
```sql
SELECT * FROM table ORDER BY column DESC LIMIT k
```

### 测试结果

| 测试ID | 数据量 | K 值 | 加速器 | 吞吐量 | vs DuckDB |
|--------|--------|------|--------|--------|-----------|
| T1 | 1M | 100 | CPU v5 | 523 M/s | 2.4x |
| T2 | 10M | 100 | CPU v5 | 488 M/s | 3.0x |
| T3 | 50M | 100 | CPU v5 | 421 M/s | 3.3x |
| T4 | 10M | 1000 | CPU v5 | 312 M/s | 2.1x |

### v5 优化技术
- 采样预过滤 (1% 采样率)
- 双路径策略: 低 cardinality 用计数排序，高 cardinality 用堆
- SIMD 8x 展开比较

### 优化建议
1. **GPU 加速未启用**: TopK shader 需内联
2. **大 K 值性能下降**: K=1000 时吞吐量降低 36%
3. **考虑 radix select**: 对于 K > 1000 可能更优

---

## 六、Sort 算子性能

### SQL 语义
```sql
SELECT * FROM table ORDER BY column [ASC|DESC]
```

### 测试结果

| 测试ID | 数据量 | 方向 | 加速器 | 吞吐量 | vs DuckDB |
|--------|--------|------|--------|--------|-----------|
| S1 | 1M | ASC | CPU | 12.5 M/s | 0.95x |
| S2 | 10M | ASC | CPU | 10.8 M/s | 0.92x |
| S3 | 1M | DESC | CPU | 15.6 M/s | 1.25x |
| S4 | 10M | DESC | CPU | 13.2 M/s | 1.18x |

### 分析
- DESC 排序比 ASC 快 ~20% (可能是分支预测)
- 整体与 DuckDB 相当，无明显优势

### 优化建议
1. **优先级低**: Sort 性能与 DuckDB 相当
2. **考虑 GPU radix sort**: 大数据量可获 2-3x 提升
3. **并行归并**: 多线程归并排序

---

## 七、综合性能对比

### ThunderDuck vs DuckDB 加速比

| 算子 | 最佳场景 | 加速比 | 关键技术 |
|------|---------|--------|---------|
| **Join** | 1M×10M | **4.5x** | GPU 并行探测 |
| **TopK** | 50M, K=100 | **3.3x** | 采样预过滤 |
| **Filter** | 50M | **2.7x** | GPU 前缀和 |
| **Aggregate** | 50M | **1.15x** | 融合 kernel |
| Sort | 10M | 1.2x | - |

### 加速器使用建议

| 数据量 | Filter | Aggregate | Join | TopK |
|--------|--------|-----------|------|------|
| < 1M | CPU | CPU | CPU | CPU |
| 1M-10M | CPU | CPU | **GPU** | CPU |
| > 10M | **GPU** | CPU | **GPU** | CPU |

---

## 八、优化机会分析

### 高优先级 (预期收益 > 2x)

| 机会 | 当前状态 | 预期提升 | 实现难度 |
|------|---------|---------|---------|
| Join J3 进一步优化 | 4.07x | 5-6x | 中 |
| TopK GPU 加速 | CPU only | 2-3x | 低 |
| Filter 多线程 | 单线程 | 2-4x | 低 |

### 中优先级 (预期收益 1.5-2x)

| 机会 | 当前状态 | 预期提升 | 实现难度 |
|------|---------|---------|---------|
| Join radix 分区 | 无 | 1.5-2x | 中 |
| Join Bloom 预过滤 | 无 | 1.3-1.5x | 中 |
| Sort GPU radix | CPU | 2-3x | 高 |

### 低优先级 (已接近极限)

| 算子 | 原因 |
|------|------|
| Aggregate | CPU 已达 94-118% 带宽利用率 |
| 小数据量 Join | GPU 启动开销抵消收益 |

---

## 九、下一步优化计划

### 立即执行 (本周)

1. **TopK GPU shader 内联**
   - 文件: `src/gpu/topk_uma.mm`
   - 预期: 2-3x 加速

2. **Filter 多线程**
   - 文件: `src/operators/filter/filter_simd.cpp`
   - 预期: 2-4x 加速

### 短期 (本月)

3. **Join v4 实现**
   - RADIX256 分区
   - Bloom 预过滤
   - NPU 加速 Bloom

4. **Join J4 优化**
   - 分批处理减少内存压力
   - 预期: 1.28x → 2x

### 长期

5. **GPU Sort (radix)**
6. **端到端查询优化**

---

## 十、结论

### 主要成果
1. **Join GPU 优化成功**: J3 场景获得 4.07x 加速
2. **Filter GPU 有效**: 大数据量 2.4x 加速
3. **Aggregate 达极限**: 94-118% 带宽利用率

### 关键瓶颈
1. **内存带宽**: 大数据量时成为主要瓶颈
2. **GPU 启动开销**: 小数据量时 CPU 更优
3. **原子操作争用**: 已通过 threadgroup 前缀和缓解

### 推荐优化路径
```
TopK GPU (2-3x) → Filter 多线程 (2-4x) → Join v4 (1.5-2x)
```

---

## 附录：测试命令

```bash
# 构建
make -C /Users/sihaoli/ThunderDuck -j8

# DuckDB 对比测试
DYLD_LIBRARY_PATH=/Users/sihaoli/ThunderDuck/third_party/duckdb \
  ./build/benchmark_app

# UMA 综合测试
./build/comprehensive_uma_benchmark

# Join 专项测试
make -C /Users/sihaoli/ThunderDuck test-uma-join
```
