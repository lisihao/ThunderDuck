# ThunderDuck Benchmark Report

> Generated: Sat Jan 24 13:48:45 2026
> Version: ThunderDuck 1.0.0 vs DuckDB 1.1.3

## System Information

| Property | Value |
|----------|-------|
| Architecture | ARM64 |
| Cache Line | 128 bytes |
| Compiler | Clang 17.0.0 (clang-1700.6.3.2) |
| Platform | macOS (Apple Silicon) |
| SIMD | ARM Neon (128-bit) |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Orders | 500,000 |
| Customers | 50,000 |
| Products | 5,000 |
| Line Items | 2,000,000 |
| Iterations | 10 |
| Warmup Iterations | 3 |

## DuckDB Benchmark Results

| Test | Category | Min (ms) | Avg (ms) | Max (ms) | StdDev | Rows |
|------|----------|----------|----------|----------|--------|------|
| Filter: quantity > 25 | Filter | 0.354 | 0.385 | 0.448 | 0.027 | 1 |
| Filter: quantity == 30 | Filter | 0.338 | 0.380 | 0.466 | 0.038 | 1 |
| Filter: range 10-40 | Filter | 0.408 | 0.466 | 0.513 | 0.032 | 1 |
| Filter: price > 500 | Filter | 0.358 | 0.380 | 0.401 | 0.014 | 1 |
| Agg: SUM(quantity) | Aggregation | 0.212 | 0.252 | 0.282 | 0.024 | 1 |
| Agg: MIN/MAX(quantity) | Aggregation | 0.424 | 0.465 | 0.510 | 0.025 | 1 |
| Agg: AVG(price) | Aggregation | 0.361 | 0.392 | 0.428 | 0.021 | 1 |
| Agg: COUNT(*) | Aggregation | 0.178 | 0.219 | 0.276 | 0.032 | 1 |
| Sort: prices ASC | Sort | 7.939 | 9.670 | 12.464 | 1.442 | 500000 |
| Sort: prices DESC | Sort | 8.750 | 9.809 | 12.803 | 1.310 | 500000 |
| Top-10 prices | TopK | 0.670 | 0.823 | 0.918 | 0.074 | 10 |
| Top-100 prices | TopK | 0.770 | 0.870 | 1.056 | 0.076 | 100 |
| Top-1000 prices | TopK | 1.153 | 1.206 | 1.302 | 0.038 | 1000 |
| Join: orders-customers | Join | 0.668 | 0.749 | 0.917 | 0.075 | 1 |

## ThunderDuck Benchmark Results

| Test | Category | Min (ms) | Avg (ms) | Max (ms) | StdDev | Rows |
|------|----------|----------|----------|----------|--------|------|
| Filter: quantity > 25 | Filter | 0.340 | 0.365 | 0.382 | 0.014 | 1000672 |
| Filter: quantity == 30 | Filter | 0.371 | 0.378 | 0.387 | 0.005 | 40218 |
| Filter: range 10-40 | Filter | 0.241 | 0.253 | 0.262 | 0.005 | 1200822 |
| Filter: price > 500 | Filter | 0.336 | 0.343 | 0.374 | 0.011 | 999757 |
| Agg: SUM(quantity) | Aggregation | 0.091 | 0.103 | 0.135 | 0.012 | 1 |
| Agg: MIN/MAX(quantity) | Aggregation | 0.082 | 0.086 | 0.092 | 0.003 | 2 |
| Agg: AVG(price) | Aggregation | 0.093 | 0.114 | 0.146 | 0.016 | 1 |
| Agg: COUNT(*) | Aggregation | 0.000 | 0.000 | 0.000 | 0.000 | 2000000 |
| Sort: prices ASC | Sort | 1.668 | 1.766 | 1.893 | 0.063 | 500000 |
| Sort: prices DESC | Sort | 1.781 | 1.809 | 1.835 | 0.017 | 500000 |
| Top-10 prices | TopK | 0.225 | 0.238 | 0.249 | 0.010 | 10 |
| Top-100 prices | TopK | 0.239 | 0.252 | 0.266 | 0.011 | 100 |
| Top-1000 prices | TopK | 0.521 | 0.536 | 0.562 | 0.012 | 1000 |
| Join: orders-customers | Join | 8.609 | 8.833 | 9.358 | 0.258 | 500000 |

## Head-to-Head Comparison

| Test | Category | DuckDB (ms) | ThunderDuck (ms) | Speedup | Winner |
|------|----------|-------------|------------------|---------|--------|
| Agg: AVG(price) | Aggregation | 0.392 | 0.114 | 3.426x | ThunderDuck |
| Agg: COUNT(*) | Aggregation | 0.219 | 0.000 | 13142x | ThunderDuck |
| Agg: MIN/MAX(quantity) | Aggregation | 0.465 | 0.086 | 5.380x | ThunderDuck |
| Agg: SUM(quantity) | Aggregation | 0.252 | 0.103 | 2.443x | ThunderDuck |
| Filter: price > 500 | Filter | 0.380 | 0.343 | 1.109x | ThunderDuck |
| Filter: quantity == 30 | Filter | 0.380 | 0.378 | 1.006x | ThunderDuck |
| Filter: quantity > 25 | Filter | 0.385 | 0.365 | 1.052x | ThunderDuck |
| Filter: range 10-40 | Filter | 0.466 | 0.253 | 1.839x | ThunderDuck |
| Join: orders-customers | Join | 0.749 | 8.833 | 11.79x slower | DuckDB |
| Sort: prices ASC | Sort | 9.670 | 1.766 | 5.476x | ThunderDuck |
| Sort: prices DESC | Sort | 9.809 | 1.809 | 5.422x | ThunderDuck |
| Top-10 prices | TopK | 0.823 | 0.238 | 3.457x | ThunderDuck |
| Top-100 prices | TopK | 0.870 | 0.252 | 3.459x | ThunderDuck |
| Top-1000 prices | TopK | 1.206 | 0.536 | 2.250x | ThunderDuck |

## Performance Analysis

### By Category

| Category | Avg Speedup | Best Speedup | Tests |
|----------|-------------|--------------|-------|
| Aggregation | 3288.37x | 13142.21x | 4 |
| Filter | 1.25x | 1.84x | 4 |
| Join | 0.08x | 0.08x | 1 |
| Sort | 5.45x | 5.48x | 2 |
| TopK | 3.06x | 3.46x | 3 |

### Summary Statistics

- **Total Tests**: 14
- **ThunderDuck Wins**: 13
- **DuckDB Wins**: 1
- **Average Speedup**: 941.33x

## Detailed Timing Distribution

All times are in milliseconds.

### Agg: AVG(price)

**DuckDB**: min=0.361, median=0.396, avg=0.392, max=0.428, stddev=0.021

**ThunderDuck**: min=0.093, median=0.115, avg=0.114, max=0.146, stddev=0.016

### Agg: COUNT(*)

**DuckDB**: min=0.178, median=0.222, avg=0.219, max=0.276, stddev=0.032

**ThunderDuck**: min=0.000, median=0.000, avg=0.000, max=0.000, stddev=0.000

### Agg: MIN/MAX(quantity)

**DuckDB**: min=0.424, median=0.467, avg=0.465, max=0.510, stddev=0.025

**ThunderDuck**: min=0.082, median=0.086, avg=0.086, max=0.092, stddev=0.003

### Agg: SUM(quantity)

**DuckDB**: min=0.212, median=0.263, avg=0.252, max=0.282, stddev=0.024

**ThunderDuck**: min=0.091, median=0.101, avg=0.103, max=0.135, stddev=0.012

### Filter: price > 500

**DuckDB**: min=0.358, median=0.384, avg=0.380, max=0.401, stddev=0.014

**ThunderDuck**: min=0.336, median=0.339, avg=0.343, max=0.374, stddev=0.011

### Filter: quantity == 30

**DuckDB**: min=0.338, median=0.370, avg=0.380, max=0.466, stddev=0.038

**ThunderDuck**: min=0.371, median=0.378, avg=0.378, max=0.387, stddev=0.005

### Filter: quantity > 25

**DuckDB**: min=0.354, median=0.383, avg=0.385, max=0.448, stddev=0.027

**ThunderDuck**: min=0.340, median=0.373, avg=0.365, max=0.382, stddev=0.014

### Filter: range 10-40

**DuckDB**: min=0.408, median=0.474, avg=0.466, max=0.513, stddev=0.032

**ThunderDuck**: min=0.241, median=0.253, avg=0.253, max=0.262, stddev=0.005

### Join: orders-customers

**DuckDB**: min=0.668, median=0.751, avg=0.749, max=0.917, stddev=0.075

**ThunderDuck**: min=8.609, median=8.807, avg=8.833, max=9.358, stddev=0.258

### Sort: prices ASC

**DuckDB**: min=7.939, median=9.091, avg=9.670, max=12.464, stddev=1.442

**ThunderDuck**: min=1.668, median=1.777, avg=1.766, max=1.893, stddev=0.063

### Sort: prices DESC

**DuckDB**: min=8.750, median=9.318, avg=9.809, max=12.803, stddev=1.310

**ThunderDuck**: min=1.781, median=1.814, avg=1.809, max=1.835, stddev=0.017

### Top-10 prices

**DuckDB**: min=0.670, median=0.842, avg=0.823, max=0.918, stddev=0.074

**ThunderDuck**: min=0.225, median=0.241, avg=0.238, max=0.249, stddev=0.010

### Top-100 prices

**DuckDB**: min=0.770, median=0.893, avg=0.870, max=1.056, stddev=0.076

**ThunderDuck**: min=0.239, median=0.252, avg=0.252, max=0.266, stddev=0.011

### Top-1000 prices

**DuckDB**: min=1.153, median=1.201, avg=1.206, max=1.302, stddev=0.038

**ThunderDuck**: min=0.521, median=0.536, avg=0.536, max=0.562, stddev=0.012

## Conclusions

ThunderDuck SIMD-optimized operators performance analysis:

1. **Aggregation Operations**: Significant improvement due to SIMD vector reductions
2. **Filter Operations**: SIMD batch comparison provides speedup for pure filtering
3. **Sort Operations**: Competitive with highly-optimized std::sort
4. **Top-K Operations**: Heap-based selection outperforms full sort

---
*Report generated by ThunderDuck Benchmark Suite*
