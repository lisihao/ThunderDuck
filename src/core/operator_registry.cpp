/**
 * ThunderDuck Operator Registry 实现
 *
 * 注册所有 ThunderDuck 优化算子
 * 包含启动代价分解、并行度建议、数据量区间直方图
 *
 * @version 2.0
 * @date 2026-01-29
 */

#include "thunderduck/operator_registry.h"
#include <iostream>
#include <iomanip>

namespace thunderduck {

// ============================================================================
// 成本模型构建辅助函数
// ============================================================================

namespace {

// 创建 CPU SIMD 算子的成本模型
CostModel make_cpu_simd_cost(
    double init_ms,           // 初始化开销
    double per_row_ns,        // 每行成本
    uint32_t opt_threads = 8, // 最优线程数
    double scaling = 0.85     // 并行扩展因子
) {
    CostModel cost;
    cost.startup.initialization = init_ms;
    cost.startup.warmup = 0.01;
    cost.per_row_cost = per_row_ns;
    cost.parallelism_hint.min_threads = 1;
    cost.parallelism_hint.optimal_threads = opt_threads;
    cost.parallelism_hint.max_threads = opt_threads * 2;
    cost.parallelism_hint.scaling_factor = scaling;
    cost.histogram.init_default();
    cost.finalize();
    return cost;
}

// 创建 GPU 算子的成本模型
CostModel make_gpu_cost(
    double init_ms,           // 初始化开销
    double kernel_launch_ms,  // 内核启动
    double transfer_ms_mb,    // 传输成本 (ms/MB)
    double per_row_ns,        // 每行成本
    uint32_t parallelism = 128
) {
    CostModel cost;
    cost.startup.initialization = init_ms;
    cost.startup.compilation = 0.5;  // shader 编译
    cost.startup.data_transfer = 0.0;  // UMA 下通常为 0
    cost.kernel_launch_cost = kernel_launch_ms;
    cost.transfer_cost = transfer_ms_mb;
    cost.per_row_cost = per_row_ns;
    cost.parallelism_hint.optimal_threads = parallelism;
    cost.parallelism_hint.max_threads = parallelism * 2;
    cost.parallelism_hint.scaling_factor = 0.9;
    // GPU 最优区间: 大数据量
    cost.histogram.buckets = {
        {0, 10000, 50.0, 0.1, false},                    // 太小
        {10000, 50000, 10.0, 0.4, false},                // 小
        {50000, 100000, 2.0, 0.7, false},                // 中等
        {100000, 1000000, 0.8, 1.0, true},               // 最优
        {1000000, 10000000, 0.5, 0.95, true},            // 大
        {10000000, SIZE_MAX, 0.4, 0.85, true}            // 超大
    };
    cost.finalize();
    return cost;
}

// 创建 NPU 算子的成本模型
CostModel make_npu_cost(
    double model_load_ms,     // 模型加载
    double init_ms,           // 初始化
    double per_batch_ns,      // 每批成本
    size_t opt_batch = 64     // 最优批大小
) {
    CostModel cost;
    cost.startup.initialization = init_ms;
    cost.startup.model_load = model_load_ms;
    cost.per_row_cost = per_batch_ns;
    cost.parallelism_hint.min_batch = 1;
    cost.parallelism_hint.optimal_batch = opt_batch;
    cost.parallelism_hint.max_batch = opt_batch * 4;
    cost.parallelism_hint.optimal_threads = 16;
    cost.parallelism_hint.scaling_factor = 0.7;  // NPU 并行扩展较差
    // NPU 最优区间: 批量处理
    cost.histogram.buckets = {
        {0, 16, 100.0, 0.2, false},                      // 批太小
        {16, 64, 20.0, 0.6, false},                      // 小批
        {64, 256, 5.0, 1.0, true},                       // 最优批
        {256, 1024, 3.0, 0.9, true},                     // 大批
        {1024, SIZE_MAX, 2.0, 0.7, false}                // 超大批
    };
    cost.finalize();
    return cost;
}

// 创建向量搜索算子的成本模型
CostModel make_vector_cost(
    double init_ms,
    double per_vector_ns,
    bool is_ann = false       // 是否为近似搜索
) {
    CostModel cost;
    cost.startup.initialization = init_ms;
    cost.per_row_cost = per_vector_ns;
    cost.parallelism_hint.optimal_threads = 8;
    cost.parallelism_hint.max_threads = 16;
    cost.parallelism_hint.scaling_factor = 0.8;

    if (is_ann) {
        // ANN 在大数据量上效率更高
        cost.histogram.buckets = {
            {0, 10000, 5.0, 0.3, false},                 // Brute-force 更好
            {10000, 100000, 1.0, 0.8, true},             // ANN 开始有效
            {100000, 1000000, 0.1, 1.0, true},           // ANN 最优
            {1000000, SIZE_MAX, 0.01, 0.95, true}        // ANN 必需
        };
    } else {
        // Brute-force 在小数据量上更好
        cost.histogram.buckets = {
            {0, 10000, 1.0, 1.0, true},                  // 最优
            {10000, 100000, 1.0, 0.8, true},             // 可用
            {100000, 1000000, 1.0, 0.5, false},          // 不推荐
            {1000000, SIZE_MAX, 1.0, 0.2, false}         // 避免使用
        };
    }
    cost.finalize();
    return cost;
}

} // anonymous namespace

// ============================================================================
// 打印注册表
// ============================================================================

void OperatorRegistry::print_registry() const {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         ThunderDuck Operator Registry v3.0                               ║\n";
    std::cout << "║                   (启动代价 | 并行度建议 | 数据量区间优化)                                ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════════╣\n";

    auto type_name = [](OperatorType t) -> const char* {
        switch (t) {
            // CPU 算子
            case OperatorType::FILTER: return "FILTER";
            case OperatorType::BLOOM_FILTER: return "BLOOM";
            case OperatorType::HASH_JOIN: return "HASH_JOIN";
            case OperatorType::SEMI_JOIN: return "SEMI_JOIN";
            case OperatorType::ANTI_JOIN: return "ANTI_JOIN";
            case OperatorType::HASH_AGGREGATE: return "HASH_AGG";
            case OperatorType::SORT: return "SORT";
            case OperatorType::TOP_N: return "TOP_N";
            // GPU 算子
            case OperatorType::GPU_FILTER: return "GPU_FILT";
            case OperatorType::GPU_HASH_JOIN: return "GPU_JOIN";
            case OperatorType::GPU_SEMI_JOIN: return "GPU_SEMI";
            case OperatorType::GPU_ANTI_JOIN: return "GPU_ANTI";
            case OperatorType::GPU_AGGREGATE: return "GPU_AGG";
            case OperatorType::GPU_TOP_K: return "GPU_TOPK";
            case OperatorType::GPU_SORT: return "GPU_SORT";
            // NPU 算子
            case OperatorType::NPU_INFERENCE: return "NPU_INFER";
            case OperatorType::NPU_EMBEDDING: return "NPU_EMBED";
            case OperatorType::NPU_MATMUL: return "NPU_MAT";
            case OperatorType::NPU_NORMALIZE: return "NPU_NORM";
            case OperatorType::NPU_ATTENTION: return "NPU_ATTN";
            // 向量算子
            case OperatorType::VECTOR_SEARCH: return "VEC_SRCH";
            case OperatorType::VECTOR_ANN: return "VEC_ANN";
            case OperatorType::VECTOR_SIMILARITY: return "VEC_SIM";
            case OperatorType::VECTOR_QUANTIZE: return "VEC_QUANT";
            case OperatorType::VECTOR_RERANK: return "VEC_RANK";
            case OperatorType::VECTOR_INDEX_BUILD: return "VEC_IDX";
            default: return "OTHER";
        }
    };

    auto device_name = [](DeviceType d) -> const char* {
        switch (d) {
            case DeviceType::CPU_SIMD: return "CPU/SIMD";
            case DeviceType::CPU_AMX: return "CPU/AMX";
            case DeviceType::GPU_METAL: return "GPU/Metal";
            case DeviceType::GPU_MPS: return "GPU/MPS";
            case DeviceType::NPU_ANE: return "NPU/ANE";
            case DeviceType::NPU_BNNS: return "NPU/BNNS";
            case DeviceType::AUTO: return "AUTO";
            default: return "UNKNOWN";
        }
    };

    // 获取最优数据范围描述
    auto optimal_range = [](const DataRangeHistogram& h) -> std::string {
        auto [min_r, max_r] = h.optimal_range();
        if (min_r == SIZE_MAX) return "N/A";
        auto format_size = [](size_t s) -> std::string {
            if (s >= 1000000) return std::to_string(s / 1000000) + "M";
            if (s >= 1000) return std::to_string(s / 1000) + "K";
            return std::to_string(s);
        };
        return format_size(min_r) + "-" + format_size(max_r);
    };

    printf("║ %-28s │ %-9s │ %-10s │ %6s │ %6s │ %5s │ %-10s ║\n",
           "Name", "Type", "Device", "Init", "PerRow", "Thr", "OptRange");
    std::cout << "╠══════════════════════════════╪═══════════╪════════════╪════════╪════════╪═══════╪════════════╣\n";

    for (const auto& [name, meta] : operators_) {
        std::string short_name = name;
        // 移除 "ThunderDuck::" 前缀以节省空间
        if (short_name.substr(0, 13) == "ThunderDuck::") {
            short_name = short_name.substr(13);
        }

        printf("║ %-28s │ %-9s │ %-10s │ %4.1fms │ %4.1fns │ %2d-%2d │ %-10s ║\n",
               short_name.substr(0, 28).c_str(),
               type_name(meta.type),
               device_name(meta.device),
               meta.cost.startup.total(),
               meta.cost.per_row_cost,
               meta.cost.parallelism_hint.min_threads,
               meta.cost.parallelism_hint.max_threads,
               optimal_range(meta.cost.histogram).c_str());
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "Total operators: " << operators_.size() << "\n";
    std::cout << "Legend: Init=Startup Cost, PerRow=Per-Row Cost, Thr=Thread Range, OptRange=Optimal Data Range\n\n";
}

namespace operators {

// ============================================================================
// Filter 算子注册 (使用增强元数据)
// ============================================================================

void register_filter_operators() {
    auto& reg = OperatorRegistry::instance();

    // SIMD Filter (INT32) - 使用增强成本模型
    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::SIMDFilterInt32";
        meta.description = "ARM NEON SIMD filter for INT32";
        meta.type = OperatorType::FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32};
        meta.output_types = {DataType::INT32};
        meta.cost = make_cpu_simd_cost(
            0.01,   // init_ms: 极低启动开销
            0.5,    // per_row_ns: ~2x faster than DuckDB
            8,      // opt_threads
            0.9     // scaling: 接近线性扩展
        );
        // 自定义最优区间 (SIMD 在中等数据量最优)
        meta.cost.histogram.buckets = {
            {0, 1000, 5.0, 0.4, false},                   // 太小,启动开销大
            {1000, 10000, 1.0, 0.8, true},                // 小数据可用
            {10000, 100000, 0.5, 1.0, true},              // 最优区间
            {100000, 1000000, 0.5, 0.95, true},           // 大数据高效
            {1000000, SIZE_MAX, 0.5, 0.9, true}           // 超大数据
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 1000
        };
        reg.register_operator(meta);
    }

    // SIMD Filter (INT64)
    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::SIMDFilterInt64";
        meta.description = "ARM NEON SIMD filter for INT64";
        meta.type = OperatorType::FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT64};
        meta.output_types = {DataType::INT64};
        meta.cost = make_cpu_simd_cost(0.01, 0.8, 8, 0.85);
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 10000
        };
        reg.register_operator(meta);
    }

    // Bloom Filter - 大数据预过滤
    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::BloomFilter";
        meta.description = "Bloom filter for fast pre-filtering";
        meta.type = OperatorType::BLOOM_FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32, DataType::INT64};
        meta.output_types = {DataType::BOOLEAN};

        // Bloom Filter 启动代价较高 (需要构建)
        meta.cost.startup.initialization = 0.5;
        meta.cost.startup.warmup = 0.1;
        meta.cost.per_row_cost = 0.1;  // 探测极快
        meta.cost.parallelism_hint.optimal_threads = 8;
        meta.cost.parallelism_hint.scaling_factor = 0.9;
        // Bloom Filter 仅在大数据量有效
        meta.cost.histogram.buckets = {
            {0, 10000, 50.0, 0.1, false},                 // 不推荐
            {10000, 100000, 5.0, 0.5, false},             // 边界效益
            {100000, 1000000, 0.2, 0.9, true},            // 有效
            {1000000, SIZE_MAX, 0.1, 1.0, true}           // 最优
        };
        meta.cost.finalize();

        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 100000
        };
        reg.register_operator(meta);
    }

    // Bitmap Filter - O(1) 存在性检查
    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::BitmapFilter";
        meta.description = "Concurrent bitmap for existence check";
        meta.type = OperatorType::FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32};
        meta.output_types = {DataType::BOOLEAN};

        meta.cost.startup.initialization = 0.1;
        meta.cost.per_row_cost = 0.05;  // O(1) 检查
        meta.cost.memory_factor = 0.125; // 每元素 1 bit
        meta.cost.parallelism_hint.optimal_threads = 8;
        meta.cost.parallelism_hint.contention_factor = 0.1; // 低竞争
        meta.cost.histogram.init_default();
        meta.cost.finalize();

        meta.capabilities = {
            .supports_parallel = true,
            .max_cardinality = 100000000  // 最大 100M 元素
        };
        reg.register_operator(meta);
    }
}

// ============================================================================
// Join 算子注册
// ============================================================================

void register_join_operators() {
    auto& reg = OperatorRegistry::instance();

    // Adaptive Hash Join
    reg.register_operator({
        .name = "ThunderDuck::AdaptiveHashJoin",
        .description = "Auto-select DIRECT_ARRAY/COMPACT_HASH/BLOOM_HASH",
        .type = OperatorType::HASH_JOIN,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.0,
            .per_row_cost = 2.0,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_predicate_pushdown = true,
            .min_rows_for_benefit = 10000
        }
    });

    // Compact Hash Join
    reg.register_operator({
        .name = "ThunderDuck::CompactHashJoin",
        .description = "Memory-efficient hash join (85-97% less memory)",
        .type = OperatorType::HASH_JOIN,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.8,
            .per_row_cost = 2.5,
            .memory_factor = 0.1,  // 10% 内存
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 50000
        }
    });

    // Bloom Hash Join
    reg.register_operator({
        .name = "ThunderDuck::BloomHashJoin",
        .description = "Hash join with Bloom filter pre-filtering",
        .type = OperatorType::HASH_JOIN,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.5,  // 需要构建 Bloom
            .per_row_cost = 1.5,  // 但探测更快
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 100000
        }
    });

    // Bitmap Semi Join
    reg.register_operator({
        .name = "ThunderDuck::BitmapSemiJoin",
        .description = "Semi join using bitmap existence check",
        .type = OperatorType::SEMI_JOIN,
        .input_types = {DataType::INT32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.2,
            .per_row_cost = 0.1,  // 极快
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_parallel = true,
            .max_cardinality = 10000000
        }
    });

    // Bitmap Anti Join
    reg.register_operator({
        .name = "ThunderDuck::BitmapAntiJoin",
        .description = "Anti join using bitmap (Q22: 9x speedup)",
        .type = OperatorType::ANTI_JOIN,
        .input_types = {DataType::INT32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.2,
            .per_row_cost = 0.1,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_parallel = true,
            .max_cardinality = 10000000
        }
    });
}

// ============================================================================
// Aggregate 算子注册
// ============================================================================

void register_aggregate_operators() {
    auto& reg = OperatorRegistry::instance();

    // Direct Array Aggregate (低基数)
    reg.register_operator({
        .name = "ThunderDuck::DirectArrayAggregate",
        .description = "Direct array for low-cardinality grouping (10-50x faster)",
        .type = OperatorType::HASH_AGGREGATE,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::INT64},
        .cost = {
            .startup_cost = 0.05,
            .per_row_cost = 0.2,  // 极快
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .max_cardinality = 100000  // 低基数才适用
        }
    });

    // Thread-Local Aggregate
    reg.register_operator({
        .name = "ThunderDuck::ThreadLocalAggregate",
        .description = "Eliminate atomic contention via thread-local aggregation",
        .type = OperatorType::HASH_AGGREGATE,
        .input_types = {DataType::ANY},
        .output_types = {DataType::INT64},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 1.0,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_parallel = true,
            .min_rows_for_benefit = 100000
        }
    });

    // Top-N Aware Aggregate
    reg.register_operator({
        .name = "ThunderDuck::TopNAwareAggregate",
        .description = "Aggregation with early Top-N pruning (Q3: +9%)",
        .type = OperatorType::HASH_AGGREGATE,
        .input_types = {DataType::ANY},
        .output_types = {DataType::INT64},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.8,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_parallel = true,
            .supports_streaming = true,
            .min_rows_for_benefit = 50000
        }
    });

    // Conditional Aggregate (CASE WHEN)
    reg.register_operator({
        .name = "ThunderDuck::ConditionalAggregate",
        .description = "Multi-branch conditional aggregation",
        .type = OperatorType::HASH_AGGREGATE,
        .input_types = {DataType::ANY},
        .output_types = {DataType::INT64},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 1.5,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_parallel = true
        }
    });
}

// ============================================================================
// Sort 算子注册
// ============================================================================

void register_sort_operators() {
    auto& reg = OperatorRegistry::instance();

    // Radix Sort (整数)
    reg.register_operator({
        .name = "ThunderDuck::RadixSort",
        .description = "Radix sort for integer keys",
        .type = OperatorType::SORT,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.5,
            .per_row_cost = 3.0,
            .parallelism = 4.0
        },
        .capabilities = {
            .supports_parallel = true,
            .min_rows_for_benefit = 10000
        }
    });

    // Top-N Sort (Heap)
    reg.register_operator({
        .name = "ThunderDuck::TopNSort",
        .description = "Heap-based Top-N sort (avoid full sort)",
        .type = OperatorType::TOP_N,
        .input_types = {DataType::ANY},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.5,  // O(log N) 每行
            .parallelism = 1.0   // 难以并行
        },
        .capabilities = {
            .supports_streaming = true
        }
    });
}

// ============================================================================
// GPU 算子注册 (Metal/MPS)
// ============================================================================

void register_gpu_join_operators() {
    auto& reg = OperatorRegistry::instance();

    // GPU Inner Join (Metal)
    reg.register_operator({
        .name = "ThunderDuck::GPU::InnerJoin",
        .description = "Metal GPU parallel inner join (two-phase)",
        .type = OperatorType::GPU_HASH_JOIN,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 2.0,
            .per_row_cost = 0.5,      // GPU 并行极快
            .transfer_cost = 0.1,     // UMA 低传输开销
            .kernel_launch_cost = 0.5,
            .parallelism = 128.0      // GPU 高并行度
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 100000  // GPU 需要大数据量
        }
    });

    // GPU Semi Join (Metal)
    reg.register_operator({
        .name = "ThunderDuck::GPU::SemiJoin",
        .description = "Metal GPU parallel semi join with atomic counters",
        .type = OperatorType::GPU_SEMI_JOIN,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.5,
            .per_row_cost = 0.3,
            .transfer_cost = 0.1,
            .kernel_launch_cost = 0.5,
            .parallelism = 128.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });

    // GPU Anti Join (Metal)
    reg.register_operator({
        .name = "ThunderDuck::GPU::AntiJoin",
        .description = "Metal GPU parallel anti join",
        .type = OperatorType::GPU_ANTI_JOIN,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.5,
            .per_row_cost = 0.3,
            .transfer_cost = 0.1,
            .kernel_launch_cost = 0.5,
            .parallelism = 128.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });

    // GPU Hash Join UMA (零拷贝)
    reg.register_operator({
        .name = "ThunderDuck::GPU::HashJoinUMA",
        .description = "Metal UMA zero-copy hash join",
        .type = OperatorType::GPU_HASH_JOIN,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.0,
            .per_row_cost = 0.4,
            .transfer_cost = 0.0,     // UMA 零拷贝
            .kernel_launch_cost = 0.3,
            .parallelism = 128.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });
}

void register_gpu_aggregate_operators() {
    auto& reg = OperatorRegistry::instance();

    // GPU Group Aggregate V2 (UMA)
    reg.register_operator({
        .name = "ThunderDuck::GPU::GroupAggregateV2",
        .description = "Metal UMA group-by aggregation",
        .type = OperatorType::GPU_AGGREGATE,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64, DataType::FLOAT32},
        .output_types = {DataType::INT64, DataType::FLOAT64},
        .cost = {
            .startup_cost = 1.5,
            .per_row_cost = 0.3,
            .transfer_cost = 0.0,
            .kernel_launch_cost = 0.3,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 100000
        }
    });

    // GPU Group Aggregate V3 (优化版)
    reg.register_operator({
        .name = "ThunderDuck::GPU::GroupAggregateV3",
        .description = "Metal UMA optimized group aggregation",
        .type = OperatorType::GPU_AGGREGATE,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::ANY},
        .output_types = {DataType::INT64, DataType::FLOAT64},
        .cost = {
            .startup_cost = 1.2,
            .per_row_cost = 0.25,
            .transfer_cost = 0.0,
            .kernel_launch_cost = 0.25,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 100000
        }
    });

    // GPU Aggregate UMA (简单聚合)
    reg.register_operator({
        .name = "ThunderDuck::GPU::AggregateUMA",
        .description = "Metal UMA sum/count/avg aggregation",
        .type = OperatorType::GPU_AGGREGATE,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64, DataType::FLOAT32},
        .output_types = {DataType::INT64, DataType::FLOAT64},
        .cost = {
            .startup_cost = 0.8,
            .per_row_cost = 0.2,
            .transfer_cost = 0.0,
            .kernel_launch_cost = 0.2,
            .parallelism = 128.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });
}

void register_gpu_filter_operators() {
    auto& reg = OperatorRegistry::instance();

    // GPU Filter UMA
    reg.register_operator({
        .name = "ThunderDuck::GPU::FilterUMA",
        .description = "Metal UMA parallel predicate evaluation",
        .type = OperatorType::GPU_FILTER,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64, DataType::FLOAT32},
        .output_types = {DataType::BOOLEAN},
        .cost = {
            .startup_cost = 0.5,
            .per_row_cost = 0.1,
            .transfer_cost = 0.0,
            .kernel_launch_cost = 0.2,
            .parallelism = 256.0      // 过滤可以极高并行
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 100000
        }
    });
}

void register_gpu_topk_operators() {
    auto& reg = OperatorRegistry::instance();

    // GPU Top-K V13
    reg.register_operator({
        .name = "ThunderDuck::GPU::TopKV13",
        .description = "Metal GPU bitonic sort Top-K",
        .type = OperatorType::GPU_TOP_K,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::INT32, DataType::INT64, DataType::FLOAT32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.0,
            .per_row_cost = 0.3,
            .transfer_cost = 0.1,
            .kernel_launch_cost = 0.3,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });

    // GPU Top-K UMA
    reg.register_operator({
        .name = "ThunderDuck::GPU::TopKUMA",
        .description = "Metal UMA zero-copy Top-K selection",
        .type = OperatorType::GPU_TOP_K,
        .device = DeviceType::GPU_METAL,
        .input_types = {DataType::ANY},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.8,
            .per_row_cost = 0.25,
            .transfer_cost = 0.0,
            .kernel_launch_cost = 0.25,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_uma = true,
            .requires_metal = true,
            .min_rows_for_benefit = 50000
        }
    });
}

void register_gpu_operators() {
    register_gpu_join_operators();
    register_gpu_aggregate_operators();
    register_gpu_filter_operators();
    register_gpu_topk_operators();
}

// ============================================================================
// NPU 算子注册 (ANE/BNNS/CoreML/MPS)
// ============================================================================

void register_npu_inference_operators() {
    auto& reg = OperatorRegistry::instance();

    // Core ML ANE 推理
    reg.register_operator({
        .name = "ThunderDuck::NPU::CoreMLInference",
        .description = "Apple Neural Engine inference via Core ML",
        .type = OperatorType::NPU_INFERENCE,
        .device = DeviceType::NPU_ANE,
        .input_types = {DataType::TENSOR, DataType::VECTOR_F32},
        .output_types = {DataType::TENSOR, DataType::VECTOR_F32},
        .cost = {
            .startup_cost = 5.0,       // 模型加载开销
            .per_row_cost = 0.1,       // 批处理极快
            .model_load_cost = 50.0,   // 首次加载
            .parallelism = 16.0
        },
        .capabilities = {
            .supports_npu = true,
            .supports_ane = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_batch_size = 1,
            .max_batch_size = 128
        }
    });

    // MPS Neural Network
    reg.register_operator({
        .name = "ThunderDuck::NPU::MPSNeuralNetwork",
        .description = "Metal Performance Shaders neural network",
        .type = OperatorType::NPU_INFERENCE,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::TENSOR, DataType::VECTOR_F32, DataType::VECTOR_F16},
        .output_types = {DataType::TENSOR, DataType::VECTOR_F32, DataType::VECTOR_F16},
        .cost = {
            .startup_cost = 3.0,
            .per_row_cost = 0.2,
            .model_load_cost = 20.0,
            .parallelism = 32.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_batch_size = 1,
            .max_batch_size = 256
        }
    });

    // BNNS CPU 推理 (AMX 加速)
    reg.register_operator({
        .name = "ThunderDuck::NPU::BNNSInference",
        .description = "BNNS CPU inference with AMX acceleration",
        .type = OperatorType::NPU_INFERENCE,
        .device = DeviceType::NPU_BNNS,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::VECTOR_F32},
        .cost = {
            .startup_cost = 1.0,
            .per_row_cost = 0.5,
            .model_load_cost = 5.0,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true,
            .min_batch_size = 1,
            .max_batch_size = 64
        }
    });
}

void register_npu_matmul_operators() {
    auto& reg = OperatorRegistry::instance();

    // BNNS Matrix Multiply (BLAS/AMX)
    reg.register_operator({
        .name = "ThunderDuck::NPU::BNNSMatmul",
        .description = "BNNS matrix multiply with AMX (C = A @ B)",
        .type = OperatorType::NPU_MATMUL,
        .device = DeviceType::CPU_AMX,
        .input_types = {DataType::FLOAT32, DataType::FLOAT16},
        .output_types = {DataType::FLOAT32, DataType::FLOAT16},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.01,      // AMX 极快
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_batch_size = 1
        }
    });

    // MPS Matrix Multiply
    reg.register_operator({
        .name = "ThunderDuck::NPU::MPSMatmul",
        .description = "MPS GPU matrix multiply",
        .type = OperatorType::NPU_MATMUL,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16},
        .output_types = {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16},
        .cost = {
            .startup_cost = 0.5,
            .per_row_cost = 0.005,     // GPU 矩阵乘极快
            .kernel_launch_cost = 0.2,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });

    // MPS Batch Matrix Multiply
    reg.register_operator({
        .name = "ThunderDuck::NPU::MPSBatchMatmul",
        .description = "MPS GPU batch matrix multiply (BMM)",
        .type = OperatorType::NPU_MATMUL,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::TENSOR},
        .output_types = {DataType::TENSOR},
        .cost = {
            .startup_cost = 0.5,
            .per_row_cost = 0.003,
            .kernel_launch_cost = 0.2,
            .parallelism = 64.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });
}

void register_npu_embedding_operators() {
    auto& reg = OperatorRegistry::instance();

    // BNNS L2 Normalize
    reg.register_operator({
        .name = "ThunderDuck::NPU::BNNSL2Normalize",
        .description = "BNNS L2 vector normalization",
        .type = OperatorType::NPU_NORMALIZE,
        .device = DeviceType::NPU_BNNS,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::VECTOR_F32},
        .cost = {
            .startup_cost = 0.05,
            .per_row_cost = 0.2,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_batch = true
        }
    });

    // BNNS Layer Norm
    reg.register_operator({
        .name = "ThunderDuck::NPU::BNNSLayerNorm",
        .description = "BNNS layer normalization",
        .type = OperatorType::NPU_NORMALIZE,
        .device = DeviceType::NPU_BNNS,
        .input_types = {DataType::VECTOR_F32, DataType::TENSOR},
        .output_types = {DataType::VECTOR_F32, DataType::TENSOR},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.3,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_batch = true
        }
    });

    // MPS Layer Norm
    reg.register_operator({
        .name = "ThunderDuck::NPU::MPSLayerNorm",
        .description = "MPS GPU layer normalization",
        .type = OperatorType::NPU_NORMALIZE,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::TENSOR, DataType::VECTOR_F32, DataType::VECTOR_F16},
        .output_types = {DataType::TENSOR, DataType::VECTOR_F32, DataType::VECTOR_F16},
        .cost = {
            .startup_cost = 0.3,
            .per_row_cost = 0.1,
            .kernel_launch_cost = 0.2,
            .parallelism = 32.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });

    // BNNS Dense Layer (MLP)
    reg.register_operator({
        .name = "ThunderDuck::NPU::BNNSDenseLayer",
        .description = "BNNS fully connected layer with activation",
        .type = OperatorType::NPU_EMBEDDING,
        .device = DeviceType::NPU_BNNS,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::VECTOR_F32},
        .cost = {
            .startup_cost = 0.2,
            .per_row_cost = 0.5,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_batch = true
        }
    });

    // MPS Dense Layer
    reg.register_operator({
        .name = "ThunderDuck::NPU::MPSDenseLayer",
        .description = "MPS GPU dense layer",
        .type = OperatorType::NPU_EMBEDDING,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::VECTOR_F32, DataType::VECTOR_F16},
        .output_types = {DataType::VECTOR_F32, DataType::VECTOR_F16},
        .cost = {
            .startup_cost = 0.3,
            .per_row_cost = 0.2,
            .kernel_launch_cost = 0.2,
            .parallelism = 32.0
        },
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });
}

void register_npu_operators() {
    register_npu_inference_operators();
    register_npu_matmul_operators();
    register_npu_embedding_operators();
}

// ============================================================================
// 向量数据库算子注册
// ============================================================================

void register_vector_search_operators() {
    auto& reg = OperatorRegistry::instance();

    // Brute-Force 向量搜索 (SIMD)
    reg.register_operator({
        .name = "ThunderDuck::Vector::BruteForceSearch",
        .description = "SIMD brute-force KNN search",
        .type = OperatorType::VECTOR_SEARCH,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::INT32, DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.5,       // 取决于维度
            .parallelism = 8.0
        },
        .metric = DistanceMetric::L2_SQUARED,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true,
            .max_dimension = 4096
        }
    });

    // Brute-Force 搜索 (BLAS/AMX)
    reg.register_operator({
        .name = "ThunderDuck::Vector::BruteForceSearchBLAS",
        .description = "BLAS/AMX accelerated brute-force search",
        .type = OperatorType::VECTOR_SEARCH,
        .device = DeviceType::CPU_AMX,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::INT32, DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.2,
            .per_row_cost = 0.2,       // AMX 更快
            .parallelism = 8.0
        },
        .metric = DistanceMetric::L2_SQUARED,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true,
            .max_dimension = 8192
        }
    });

    // HNSW 近似最近邻
    reg.register_operator({
        .name = "ThunderDuck::Vector::HNSWSearch",
        .description = "HNSW approximate nearest neighbor search",
        .type = OperatorType::VECTOR_ANN,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::INT32, DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.01,      // ANN 极快 (sublinear)
            .parallelism = 8.0
        },
        .metric = DistanceMetric::L2_SQUARED,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true,
            .supports_incremental = true,
            .max_dimension = 4096
        }
    });

    // FP16 量化向量搜索
    reg.register_operator({
        .name = "ThunderDuck::Vector::BruteForceSearchFP16",
        .description = "FP16 quantized brute-force search (50% memory)",
        .type = OperatorType::VECTOR_SEARCH,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F16},
        .output_types = {DataType::INT32, DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.4,
            .memory_factor = 0.5,      // 50% 内存
            .parallelism = 8.0
        },
        .metric = DistanceMetric::L2_SQUARED,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .max_dimension = 4096
        }
    });
}

void register_vector_similarity_operators() {
    auto& reg = OperatorRegistry::instance();

    // 余弦相似度
    reg.register_operator({
        .name = "ThunderDuck::Vector::CosineSimilarity",
        .description = "SIMD cosine similarity computation",
        .type = OperatorType::VECTOR_SIMILARITY,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.05,
            .per_row_cost = 0.3,
            .parallelism = 8.0
        },
        .metric = DistanceMetric::COSINE,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true
        }
    });

    // BNNS 余弦相似度
    reg.register_operator({
        .name = "ThunderDuck::Vector::BNNSCosineSimilarity",
        .description = "BNNS cosine similarity with vDSP",
        .type = OperatorType::VECTOR_SIMILARITY,
        .device = DeviceType::NPU_BNNS,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.05,
            .per_row_cost = 0.2,
            .parallelism = 8.0
        },
        .metric = DistanceMetric::COSINE,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true
        }
    });

    // MPS 向量相似度
    reg.register_operator({
        .name = "ThunderDuck::Vector::MPSSimilarity",
        .description = "MPS GPU vector similarity (cosine/dot/L2)",
        .type = OperatorType::VECTOR_SIMILARITY,
        .device = DeviceType::GPU_MPS,
        .input_types = {DataType::VECTOR_F32, DataType::VECTOR_F16},
        .output_types = {DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.3,
            .per_row_cost = 0.05,
            .kernel_launch_cost = 0.2,
            .parallelism = 64.0
        },
        .metric = DistanceMetric::COSINE,
        .capabilities = {
            .supports_gpu = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_rows_for_benefit = 10000
        }
    });

    // 内积相似度
    reg.register_operator({
        .name = "ThunderDuck::Vector::InnerProductSimilarity",
        .description = "SIMD inner product similarity",
        .type = OperatorType::VECTOR_SIMILARITY,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::FLOAT32},
        .cost = {
            .startup_cost = 0.05,
            .per_row_cost = 0.25,
            .parallelism = 8.0
        },
        .metric = DistanceMetric::INNER_PRODUCT,
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true
        }
    });

    // 学习型相似度 (Cross-encoder)
    reg.register_operator({
        .name = "ThunderDuck::Vector::LearnedSimilarity",
        .description = "Neural cross-encoder similarity",
        .type = OperatorType::VECTOR_SIMILARITY,
        .device = DeviceType::NPU_ANE,
        .input_types = {DataType::VECTOR_F32, DataType::TENSOR},
        .output_types = {DataType::FLOAT32},
        .cost = {
            .startup_cost = 5.0,
            .per_row_cost = 1.0,       // 神经网络较慢
            .model_load_cost = 50.0,
            .parallelism = 16.0
        },
        .capabilities = {
            .supports_npu = true,
            .supports_ane = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_batch_size = 1,
            .max_batch_size = 64
        }
    });
}

void register_vector_quantizer_operators() {
    auto& reg = OperatorRegistry::instance();

    // Scalar 量化
    reg.register_operator({
        .name = "ThunderDuck::Vector::ScalarQuantizer",
        .description = "Scalar quantization (FP32 -> INT8/FP16)",
        .type = OperatorType::VECTOR_QUANTIZE,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::UINT8, DataType::FLOAT16},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.2,
            .memory_factor = 0.25,     // 75% 压缩
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_int8 = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });

    // Product Quantization (PQ)
    reg.register_operator({
        .name = "ThunderDuck::Vector::ProductQuantizer",
        .description = "Product quantization for high compression",
        .type = OperatorType::VECTOR_QUANTIZE,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::UINT8},
        .cost = {
            .startup_cost = 1.0,       // 需要训练 codebook
            .per_row_cost = 0.5,
            .memory_factor = 0.0625,   // 16x 压缩
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_batch = true
        }
    });

    // Neural Quantizer (Autoencoder)
    reg.register_operator({
        .name = "ThunderDuck::Vector::NeuralQuantizer",
        .description = "Neural autoencoder quantization",
        .type = OperatorType::VECTOR_QUANTIZE,
        .device = DeviceType::NPU_ANE,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::VECTOR_F32},
        .cost = {
            .startup_cost = 5.0,
            .per_row_cost = 1.0,
            .model_load_cost = 50.0,
            .memory_factor = 0.1,      // 高压缩比
            .parallelism = 16.0
        },
        .capabilities = {
            .supports_npu = true,
            .supports_ane = true,
            .supports_fp16 = true,
            .supports_batch = true
        }
    });
}

void register_vector_reranker_operators() {
    auto& reg = OperatorRegistry::instance();

    // Cross-Encoder Reranker
    reg.register_operator({
        .name = "ThunderDuck::Vector::CrossEncoderReranker",
        .description = "Neural cross-encoder reranking",
        .type = OperatorType::VECTOR_RERANK,
        .device = DeviceType::NPU_ANE,
        .input_types = {DataType::VECTOR_F32, DataType::TENSOR},
        .output_types = {DataType::FLOAT32, DataType::INT32},
        .cost = {
            .startup_cost = 5.0,
            .per_row_cost = 2.0,       // 精细排序较慢
            .model_load_cost = 50.0,
            .parallelism = 16.0
        },
        .capabilities = {
            .supports_npu = true,
            .supports_ane = true,
            .supports_fp16 = true,
            .supports_batch = true,
            .min_batch_size = 1,
            .max_batch_size = 64
        }
    });

    // Two-Stage Retriever (粗排 + 精排)
    reg.register_operator({
        .name = "ThunderDuck::Vector::TwoStageRetriever",
        .description = "Two-stage retrieval: vector similarity + cross-encoder",
        .type = OperatorType::VECTOR_RERANK,
        .device = DeviceType::AUTO,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::FLOAT32, DataType::INT32},
        .cost = {
            .startup_cost = 6.0,
            .per_row_cost = 0.1,       // 只对 Top-K 精排
            .model_load_cost = 50.0,
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_npu = true,
            .supports_batch = true
        }
    });
}

void register_vector_index_operators() {
    auto& reg = OperatorRegistry::instance();

    // HNSW Index Build
    reg.register_operator({
        .name = "ThunderDuck::Vector::HNSWIndexBuild",
        .description = "HNSW index construction",
        .type = OperatorType::VECTOR_INDEX_BUILD,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 1.0,
            .per_row_cost = 5.0,       // 索引构建较慢
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_incremental = true,
            .max_dimension = 4096
        }
    });

    // Flat Index (用于小数据集)
    reg.register_operator({
        .name = "ThunderDuck::Vector::FlatIndex",
        .description = "Flat index with precomputed norms",
        .type = OperatorType::VECTOR_INDEX_BUILD,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::VECTOR_F32, DataType::VECTOR_F16},
        .output_types = {DataType::ANY},
        .cost = {
            .startup_cost = 0.1,
            .per_row_cost = 0.1,       // 简单存储
            .parallelism = 8.0
        },
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_fp16 = true,
            .supports_incremental = true
        }
    });
}

void register_vector_operators() {
    register_vector_search_operators();
    register_vector_similarity_operators();
    register_vector_quantizer_operators();
    register_vector_reranker_operators();
    register_vector_index_operators();
}

// ============================================================================
// TPC-H 专用算子版本注册 (供优化器选择)
// ============================================================================

void register_tpch_versioned_operators() {
    auto& reg = OperatorRegistry::instance();

    // ========================================================================
    // V27: Bitmap Join 系列 (基础版本)
    // ========================================================================

    reg.register_operator({
        .name = "ThunderDuck::TPCH::BitmapSemiJoinV27",
        .description = "V27 Bitmap SEMI Join for EXISTS queries",
        .type = OperatorType::SEMI_JOIN,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::INT32},
        .output_types = {DataType::ANY},
        .cost = make_cpu_simd_cost(0.2, 0.1, 8, 0.9),
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .max_cardinality = 10000000,
            .min_rows_for_benefit = 10000
        }
    });

    reg.register_operator({
        .name = "ThunderDuck::TPCH::BloomFilterJoinV27",
        .description = "V27 Bloom Filter pre-filtering for large joins",
        .type = OperatorType::HASH_JOIN,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = make_cpu_simd_cost(1.5, 1.5, 8, 0.85),
        .capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 100000
        }
    });

    // ========================================================================
    // V32: 紧凑 Hash + 自适应策略 (性能基线)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::CompactHashJoinV32";
        meta.description = "V32 Compact Hash Join with adaptive strategy (SF<5: direct, SF>=5: compact)";
        meta.type = OperatorType::HASH_JOIN;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32, DataType::INT64};
        meta.output_types = {DataType::ANY};
        meta.cost = make_cpu_simd_cost(0.8, 2.0, 8, 0.9);
        meta.cost.memory_factor = 0.15;  // 85% 内存节省
        // 最优区间: 中大数据量
        meta.cost.histogram.buckets = {
            {0, 10000, 5.0, 0.5, false},
            {10000, 100000, 2.0, 0.9, true},
            {100000, 1000000, 1.5, 1.0, true},   // 最优
            {1000000, SIZE_MAX, 2.0, 0.9, true}
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 10000
        };
        reg.register_operator(meta);
    }

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::ThreadLocalAggV32";
        meta.description = "V32 Thread-local aggregation eliminating atomic contention";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::ANY};
        meta.output_types = {DataType::INT64, DataType::FLOAT64};
        meta.cost = make_cpu_simd_cost(0.1, 1.0, 8, 0.95);  // 近乎线性扩展
        meta.cost.parallelism_hint.contention_factor = 0.05;  // 极低竞争
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 50000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V37: Bitmap Anti-Join (Q22 最优: 9.08x)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::BitmapAntiJoinV37";
        meta.description = "V37 Bitmap Anti-Join for NOT EXISTS (Q22: 9.08x speedup)";
        meta.type = OperatorType::ANTI_JOIN;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32};
        meta.output_types = {DataType::ANY};
        meta.cost = make_cpu_simd_cost(0.2, 0.05, 8, 0.95);
        meta.cost.memory_factor = 0.01;  // 极低内存 (bitmap)
        // 最优区间: 中小基数
        meta.cost.histogram.buckets = {
            {0, 10000, 0.1, 0.9, true},
            {10000, 100000, 0.05, 1.0, true},   // 最优
            {100000, 1000000, 0.05, 0.95, true},
            {1000000, 10000000, 0.1, 0.8, false},
            {10000000, SIZE_MAX, 1.0, 0.3, false}  // bitmap 过大
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .max_cardinality = 10000000,  // 最大 10M 元素
            .min_rows_for_benefit = 1000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V42: 并行化聚合 (Q8 最优: 1.85x)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::ParallelAggV42";
        meta.description = "V42 Parallel aggregation with thread-local accumulation (Q8: 1.85x)";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::ANY};
        meta.output_types = {DataType::INT64, DataType::FLOAT64};
        meta.cost = make_cpu_simd_cost(0.15, 0.8, 8, 0.92);
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 100000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V43: 两阶段聚合 (Q17 最优: 4.30x)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::TwoStageAggV43";
        meta.description = "V43 Two-stage aggregation with bitmap filtering (Q17: 4.30x)";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::ANY};
        meta.output_types = {DataType::INT64, DataType::FLOAT64};
        meta.cost = make_cpu_simd_cost(0.3, 0.5, 8, 0.88);
        // 适用于有预过滤的聚合
        meta.cost.histogram.buckets = {
            {0, 10000, 2.0, 0.5, false},
            {10000, 100000, 0.8, 0.9, true},
            {100000, 1000000, 0.5, 1.0, true},   // 最优
            {1000000, SIZE_MAX, 0.5, 0.9, true}
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_predicate_pushdown = true,
            .min_rows_for_benefit = 50000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V46: 直接数组 (Q11: 4.14x, Q14: 2.91x)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::DirectArrayAggV46";
        meta.description = "V46 Direct array aggregation for low-cardinality (Q11: 4.14x, Q14: 2.91x)";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32};
        meta.output_types = {DataType::INT64, DataType::FLOAT64};
        meta.cost = make_cpu_simd_cost(0.05, 0.15, 8, 0.98);  // 极快
        meta.cost.memory_factor = 0.01;  // 极低内存
        // 仅适用于低基数
        meta.cost.histogram.buckets = {
            {0, 100000, 0.1, 1.0, true},        // 最优
            {100000, 1000000, 0.2, 0.7, false},
            {1000000, SIZE_MAX, 1.0, 0.2, false}  // 不适用
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .max_cardinality = 1000000,  // 最大 1M 基数
            .min_rows_for_benefit = 1000
        };
        reg.register_operator(meta);
    }

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::BitmapMembershipV46";
        meta.description = "V46 Bitmap membership filter for existence check";
        meta.type = OperatorType::FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32};
        meta.output_types = {DataType::BOOLEAN};
        meta.cost = make_cpu_simd_cost(0.1, 0.03, 8, 0.95);
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .max_cardinality = 100000000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V47: SIMD 无分支 (Q6 目标: 3.0x+)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::SIMDBranchlessFilterV47";
        meta.description = "V47 SIMD branchless multi-condition filter (Q6 target: 3.0x+)";
        meta.type = OperatorType::FILTER;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::INT32, DataType::INT64, DataType::FLOAT32};
        meta.output_types = {DataType::BOOLEAN};
        meta.cost = make_cpu_simd_cost(0.02, 0.2, 8, 0.92);
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .min_rows_for_benefit = 10000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V48: Group-then-Filter (Q21 通用重写)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::GroupThenFilterV48";
        meta.description = "V48 Group-then-filter rewrite for complex EXISTS/NOT EXISTS";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::ANY};
        meta.output_types = {DataType::INT64};
        meta.cost = make_cpu_simd_cost(0.5, 1.5, 8, 0.85);
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_streaming = true,
            .min_rows_for_benefit = 50000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // V49: Top-N Aware 局部聚合 (Q3: +9%)
    // ========================================================================

    {
        OperatorMeta meta;
        meta.name = "ThunderDuck::TPCH::TopNAwareAggV49";
        meta.description = "V49 Top-N aware local aggregation with early pruning (Q3: +9%)";
        meta.type = OperatorType::HASH_AGGREGATE;
        meta.device = DeviceType::CPU_SIMD;
        meta.input_types = {DataType::ANY};
        meta.output_types = {DataType::INT64, DataType::FLOAT64};
        meta.cost = make_cpu_simd_cost(0.1, 0.7, 8, 0.9);
        // 适用于有 LIMIT 的聚合
        meta.cost.histogram.buckets = {
            {0, 10000, 1.0, 0.6, false},
            {10000, 100000, 0.7, 0.9, true},
            {100000, 1000000, 0.7, 1.0, true},   // 最优
            {1000000, SIZE_MAX, 0.7, 0.95, true}
        };
        meta.capabilities = {
            .supports_simd = true,
            .supports_parallel = true,
            .supports_streaming = true,
            .min_rows_for_benefit = 10000
        };
        reg.register_operator(meta);
    }

    // ========================================================================
    // 通用组件 (跨版本复用)
    // ========================================================================

    reg.register_operator({
        .name = "ThunderDuck::TPCH::ThreadPool",
        .description = "Shared thread pool with prewarming",
        .type = OperatorType::PROJECTION,  // 辅助组件
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::ANY},
        .output_types = {DataType::ANY},
        .cost = make_cpu_simd_cost(0.5, 0.0, 8, 1.0),
        .capabilities = {
            .supports_parallel = true
        }
    });

    reg.register_operator({
        .name = "ThunderDuck::TPCH::GenerationDeduplicator",
        .description = "Generation counter based deduplication",
        .type = OperatorType::DISTINCT,
        .device = DeviceType::CPU_SIMD,
        .input_types = {DataType::INT32, DataType::INT64},
        .output_types = {DataType::ANY},
        .cost = make_cpu_simd_cost(0.1, 0.05, 8, 0.95),
        .capabilities = {
            .supports_parallel = true,
            .supports_incremental = true
        }
    });
}

// ============================================================================
// 注册所有算子
// ============================================================================

void register_all_operators() {
    // CPU 算子 (SIMD)
    register_filter_operators();
    register_join_operators();
    register_aggregate_operators();
    register_sort_operators();

    // GPU 算子 (Metal/MPS)
    register_gpu_operators();

    // NPU 算子 (ANE/BNNS/CoreML)
    register_npu_operators();

    // 向量数据库算子
    register_vector_operators();

    // TPC-H 专用版本算子 (供优化器选择)
    register_tpch_versioned_operators();
}

} // namespace operators
} // namespace thunderduck
