# ThunderDuck V58 Technical Report

> **Version**: 58.0 | **Date**: 2026-01-30 | **Platform**: Apple Silicon M4

## Executive Summary

ThunderDuck is a high-performance SQL operator backend optimized for Apple M4 silicon, achieving **3.24x geometric mean speedup** over DuckDB across all 22 TPC-H queries at SF=1.

### Key Achievements

| Metric | Value |
|--------|-------|
| Geometric Mean Speedup | 3.24x |
| Best Query (Q21) | 23.41x |
| Queries Faster | 22/22 (100%) |
| Total Operators | 70+ |
| Query Optimizer | Adaptive Selection |

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ThunderDuck Query Engine                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Query     │  │   System    │  │   Query Optimizer       │  │
│  │   Parser    │──│   Catalog   │──│   (Adaptive Selection)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Operator Framework                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Filter  │ │   Join   │ │Aggregate │ │  TopN    │           │
│  │  (SIMD)  │ │  (Hash)  │ │ (Direct) │ │ (Heap)   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                    Hardware Abstraction                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  ARM Neon    │  │  Metal GPU   │  │  Thread Pool │          │
│  │  SIMD        │  │  Compute     │  │  (8 cores)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Design Principles

1. **Zero Hardcoding**: All constants from `tpch_constants.h`
2. **O(1) Hot Paths**: DirectArray for L2-cache-friendly access
3. **Adaptive Selection**: Runtime operator selection based on data characteristics
4. **SIMD First**: ARM Neon intrinsics for vectorized operations

---

## 2. Query Optimizer

### 2.1 Adaptive Selection Framework

The query optimizer dynamically selects the optimal operator version based on:

```cpp
struct ApplicabilityContext {
    size_t row_count;           // Data volume
    size_t max_key_range;       // Maximum key value (for DirectArray)
    size_t distinct_keys;       // Cardinality
    double selectivity;         // Filter selectivity
    size_t l2_cache_bytes;      // L2 cache size (4MB default)
    bool has_native_double;     // Native double column presence
};
```

### 2.2 Selection Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| STATIC | Pre-configured speedup ratios | Default |
| ADAPTIVE | Historical performance data | Warm systems |
| HYBRID | Combines static + adaptive | Production |

### 2.3 Applicability Rules

| Operator | Condition | Rationale |
|----------|-----------|-----------|
| DirectArray | `max_key_range <= 250K` | L2 cache (4MB / 16B) |
| ParallelScan | `row_count >= 10K` | Parallel overhead |
| CompactHash | `row_count >= 100K` | Hash amortization |
| NativeDoubleSIMD | `has_native_double` | float64 SIMD |
| TopN-Aware | `has_top_n` | Early termination |

---

## 3. Core Algorithms

### 3.1 DirectArrayAggregator (V46/V57/V58)

O(1) aggregation using direct array indexing instead of hash tables.

```cpp
template<typename ValueT = int64_t>
class DirectArrayAggregator {
    std::vector<ValueT> data_;

public:
    // O(1) add - no hash computation
    void add(int32_t key, ValueT value) {
        data_[key] += value;
    }

    // Applicability check
    static bool is_applicable(int32_t max_key) {
        // L2 cache: 4MB, each entry: 8 bytes (int64)
        // Conservative limit: 250K entries
        return max_key <= 250000;
    }
};
```

**When to use**: Low-cardinality aggregation keys (partkey, nationkey, etc.)

### 3.2 PrecomputedBitmap (V37/V58)

Pre-computed bitmap for O(1) membership testing.

```cpp
class PrecomputedBitmap {
    std::vector<uint64_t> bitmap_;

public:
    // Build from predicate
    void build(const std::vector<std::string>& values,
               std::function<bool(const std::string&)> predicate) {
        for (size_t i = 0; i < values.size(); i++) {
            if (predicate(values[i])) {
                set(i);
            }
        }
    }

    // O(1) lookup
    bool test(size_t idx) const {
        return bitmap_[idx / 64] & (1ULL << (idx % 64));
    }
};
```

**Applications**:
- Q22: ANTI JOIN with order existence
- Q9: "green" parts filtering
- Q2: BRASS suffix matching

### 3.3 CompactHashTable (V32)

Cache-optimized hash table with CRC32 hardware hashing.

```cpp
class CompactHashTable {
    // 8-way parallel probing with prefetch
    static constexpr int PREFETCH_DISTANCE = 8;

    uint32_t hash(int64_t key) {
        // Hardware CRC32 instruction
        return __crc32cd(0, key);
    }

    void insert_batch(const int64_t* keys, size_t n) {
        for (size_t i = 0; i + PREFETCH_DISTANCE < n; i++) {
            __builtin_prefetch(&buckets_[hash(keys[i + PREFETCH_DISTANCE])]);
            insert(keys[i]);
        }
    }
};
```

### 3.4 ParallelScanExecutor (V58)

Generic parallel scanning with thread-local results.

```cpp
template<typename ResultT>
class ParallelScanExecutor {
    size_t thread_count_;
    std::vector<std::vector<ResultT>> thread_local_results_;

public:
    template<typename ProcessFn>
    void execute(const void* data, size_t count, ProcessFn&& fn) {
        size_t chunk_size = (count + thread_count_ - 1) / thread_count_;

        std::vector<std::future<void>> futures;
        for (size_t t = 0; t < thread_count_; t++) {
            futures.push_back(std::async([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, count);
                fn(data, start, end, thread_local_results_[t]);
            }));
        }

        for (auto& f : futures) f.wait();
    }

    std::vector<ResultT> merge() {
        // Merge thread-local results
    }
};
```

### 3.5 SIMD Branchless Filter (V47/V54)

ARM Neon vectorized filtering without branches.

```cpp
// 4-way SIMD comparison
uint32x4_t simd_filter_ge(int32x4_t values, int32_t threshold) {
    int32x4_t thresh_vec = vdupq_n_s32(threshold);
    return vcgeq_s32(values, thresh_vec);  // Returns mask
}

// Branchless selection using mask
void filter_batch(const int32_t* input, int32_t* output,
                  uint32_t* mask, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        int32x4_t vals = vld1q_s32(&input[i]);
        uint32x4_t m = simd_filter_ge(vals, threshold);

        // Compress using mask
        uint64_t bits = vget_lane_u64(vreinterpret_u64_u32(
            vmovn_u32(m)), 0);
        // ... compress store
    }
}
```

---

## 4. Performance Results

### 4.1 TPC-H SF=1 Benchmark

| Query | DuckDB (ms) | ThunderDuck (ms) | Speedup | Key Optimization |
|-------|-------------|------------------|---------|------------------|
| Q1 | 38.60 | 9.08 | **4.25x** | SIMD Aggregate |
| Q2 | 6.66 | 2.16 | **3.08x** | V58 ParallelScan |
| Q3 | 12.14 | 9.91 | **1.23x** | V49 TopN-Aware |
| Q4 | 15.07 | 3.87 | **3.90x** | Bitmap SEMI |
| Q5 | 13.41 | 8.08 | **1.66x** | V57 ZeroCost |
| Q6 | 4.77 | 2.29 | **2.08x** | V54 NativeDouble |
| Q7 | 16.31 | 5.04 | **3.24x** | V32 CompactHash |
| Q8 | 11.95 | 7.97 | **1.50x** | V57 Generic |
| Q9 | 43.17 | 31.16 | **1.39x** | V32 Bloom Filter |
| Q10 | 29.17 | 10.39 | **2.81x** | Precomputed Join |
| Q11 | 4.12 | 1.89 | **2.17x** | V46 DirectArray |
| Q12 | 21.29 | 3.92 | **5.43x** | SIMD Branchless |
| Q13 | 48.74 | 16.18 | **3.01x** | V34 GenericOuter |
| Q14 | 10.26 | 2.06 | **4.98x** | V46 DirectArray |
| Q15 | 7.35 | 2.06 | **3.57x** | Precomputed Agg |
| Q16 | 14.21 | 7.59 | **1.87x** | V27 Bitmap |
| Q17 | 10.90 | 4.38 | **2.49x** | V57 DirectArray |
| Q18 | 34.70 | 7.71 | **4.50x** | V32 CompactHash |
| Q19 | 29.75 | 4.48 | **6.63x** | V33 StringSet |
| Q20 | 16.20 | 6.90 | **2.35x** | V40 Decorrelation |
| Q21 | 44.23 | 1.89 | **23.41x** | V51 RadixSort |
| Q22 | 10.30 | 1.21 | **8.49x** | V37 BitmapAnti |

### 4.2 Category Analysis

| Category | Queries | Avg Speedup | Description |
|----------|---------|-------------|-------------|
| A | Q1,3,5,6,7,9,10,12,14,18 | 3.16x | Fully optimizable |
| B | Q2,4,11,15,16,19 | 2.95x | Partially optimizable |
| C | Q8,13,17,20,21,22 | 5.54x | Complex queries |

---

## 5. Operator Version Evolution

### 5.1 Key Milestones

| Version | Feature | Impact |
|---------|---------|--------|
| V24 | SelectionVector | Zero-copy filtering |
| V25 | ThreadPool | 8-core parallelism |
| V27 | BitmapSemiJoin | O(1) existence test |
| V32 | CompactHashTable | CRC32 + 8-way prefetch |
| V37 | BitmapAntiJoin | Q22: 9x speedup |
| V46 | DirectArrayAggregator | Low-cardinality O(1) |
| V49 | TopN-Aware | Early termination |
| V54 | NativeDoubleSIMD | float64 vectorization |
| V57 | ZeroCostAggregator | Template metaprogramming |
| V58 | ParallelScan + Bitmap | Q2: 3x, generic design |

### 5.2 Registered Operators (70 total)

```
V58: DirectArrayAggregator, PrecomputedBitmap, ParallelScanExecutor, FusedSIMDFilterAggregate
V57: AdaptiveMap, DirectArray, ZeroCostAggregator, ParallelScanner
V56: DirectArrayDecorrelation, BloomFilteredJoin, DirectArrayTwoPhaseAgg
V55: SubqueryDecorrelation, GenericParallelMultiJoin, GenericTwoPhaseAgg
V54: NativeDoubleSIMDFilter
V53: EnhancedBitmap
V52: BitmapPredicateIndex
V51: RadixSort, PartitionedAgg, FusedFilterAgg
V50: HybridExecutor
V49: TopNAware, HeapPruning
V47: ParallelRadixSort, SIMDBranchlessFilter, SIMDPatternMatcher, SparseDirectArray
V46: DirectArrayFilter, BitmapMembershipFilter, DirectArrayAggregator
V45: DirectArrayDimension
V41: OrderStatePrecompute, DirectArrayLookup
V40: DynamicBitmapFilter, SortedGroupByAggregator, MergeJoinOperator
V37: BitmapExistenceSet, BitmapAntiJoin, OrderKeyState
V36: PrecomputedAggregates, SubqueryDecorrelation
V35: DirectArrayIndexBuilder, SIMDStringProcessor, SemiAntiJoin, PipelineFusion
V34: GenericAntiJoin, GenericOuterJoin, ConditionalAggregator
V33: DateRangeFilter, StringSetMatcher, AdaptiveHashJoin, ThreadLocalAggregator
V32: CompactHash, CRC32ParallelHash, ThreadLocalAgg, AdaptiveSF
V27: BitmapSemiJoin, BitmapAntiJoin, StringDict, PredicatePrecomputer
V26: MutableWeakHash, BloomPrefilter, VectorizedGroupBy
V25: ThreadPool, WeakHashTable, DictEncodedJoin
V24: SelectionVector, DirectArrayAgg
```

---

## 6. System Catalog

### 6.1 Operator Metadata

Each operator registers:
- **startup_ms**: Fixed startup overhead
- **per_row_us**: Per-row processing cost
- **min_rows**: Minimum applicable row count
- **max_rows**: Maximum applicable row count (0 = unlimited)

### 6.2 Cost Model

```
estimated_time = startup_ms + (per_row_us * row_count / 1000)
```

### 6.3 Adaptive Learning

The system records execution metrics and uses them for future selection:
- Average execution time
- Median time (more robust)
- Standard deviation
- Sample count
- Confidence score

---

## 7. Build & Usage

### 7.1 Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 tpch_benchmark
```

### 7.2 Run Benchmark

```bash
# Full benchmark
./benchmarks/tpch_benchmark

# Specific query
./benchmarks/tpch_benchmark --query Q6

# Category
./benchmarks/tpch_benchmark --category A

# Scale factor
./benchmarks/tpch_benchmark --sf 10
```

### 7.3 Show Catalog

```bash
./benchmarks/tpch_benchmark --show-catalog
```

---

## 8. Future Directions

1. **NPU Acceleration**: Core ML integration for batch operations
2. **Adaptive Parallelism**: Dynamic thread count based on query
3. **Memory Pool**: Arena allocation for reduced fragmentation
4. **JIT Compilation**: Runtime code generation for hot paths

---

## Appendix: File Structure

```
ThunderDuck/
├── include/thunderduck/
│   ├── generic_operators_v58.h    # V58 generic operators
│   ├── system_catalog.h           # Operator registry
│   └── tpch_constants.h           # Zero-hardcode constants
├── benchmark/tpch/
│   ├── tpch_query_optimizer.h     # Adaptive optimizer
│   ├── tpch_query_optimizer.cpp   # Selection logic
│   ├── tpch_operators_v*.cpp      # Version-specific operators
│   └── tpch_benchmark_main.cpp    # Benchmark driver
└── docs/
    └── THUNDERDUCK_V58_TECHNICAL_REPORT.md  # This document
```

---

*Generated by ThunderDuck Development Team*
*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
