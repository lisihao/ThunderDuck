/**
 * ThunderDuck TPC-H Query Optimizer Implementation
 *
 * 注册所有 TPC-H 查询的算子版本配置
 * 成本模型参数基于 SF=1 实测数据 (2026-01-31)
 *
 * 基准测试结果摘要 (SF=1, 10 iterations, M4 Mac):
 * - 几何平均加速比: 3.42x (vs DuckDB)
 * - 最高: Q21 24.86x, Q1 9.30x, Q22 9.22x
 * - 最低: Q9 1.36x, Q3 1.43x, Q6 1.77x
 * - 全部 22 查询均快于 DuckDB
 */

// 系统表必须在优化器之前 include
#include "../../include/thunderduck/system_catalog.h"
#include "tpch_query_optimizer.h"
#include "tpch_data_loader.h"
#include "tpch_operators_v24.h"
#include "tpch_operators_v25.h"
#include "tpch_operators_v27.h"
#include "tpch_operators_v32.h"
#include "tpch_operators_v33.h"
#include "tpch_operators_v34.h"
#include "tpch_operators_v35.h"
#include "tpch_operators_v36.h"
#include "tpch_operators_v37.h"
#include "tpch_operators_v40.h"
// V42, V43 专用算子已删除，由 V57 通用算子替代
#include "tpch_operators_v46.h"
#include "tpch_operators_v47.h"
#include "tpch_operators_v48.h"
#include "tpch_operators_v49.h"
#include "tpch_operators_v50.h"
#include "tpch_operators_v51.h"
#include "tpch_operators_v52.h"
#include "tpch_operators_v53.h"
#include "tpch_operators_v54.h"
#include "tpch_operators_v55.h"
#include "tpch_operators_v56.h"
#include "tpch_operators_v57.h"
#include "tpch_operators_v58.h"
#include "tpch_operators_v59.h"
#include "tpch_operators_v60.h"
#include "tpch_operators_v61.h"
#include "tpch_operators_v62.h"
#include "tpch_operators_v63.h"
#include "tpch_operators_v64.h"
#include "tpch_operators_v65.h"
#include "tpch_operators_v66.h"
#include "tpch_operators_v67.h"
#include "tpch_operators_v68.h"
#include "tpch_operators_v69.h"
#include "tpch_operators_v70.h"
#include "tpch_operators_v71.h"
#include "tpch_operators_v88.h"
#include "tpch_operators_v92.h"

namespace thunderduck {
namespace tpch {

// 前向声明基础查询实现 (仅声明已存在的)
namespace queries {
    void run_q1(TPCHDataLoader& loader);
    void run_q2(TPCHDataLoader& loader);
    void run_q3(TPCHDataLoader& loader);
    void run_q4(TPCHDataLoader& loader);
    void run_q5(TPCHDataLoader& loader);
    void run_q6(TPCHDataLoader& loader);
    void run_q7(TPCHDataLoader& loader);
    // Q8 无基础实现，使用优化版本
    void run_q9(TPCHDataLoader& loader);
    void run_q10(TPCHDataLoader& loader);
    void run_q11(TPCHDataLoader& loader);
    void run_q12(TPCHDataLoader& loader);
    // Q13 无基础实现，使用优化版本
    void run_q14(TPCHDataLoader& loader);
    void run_q15(TPCHDataLoader& loader);
    void run_q16(TPCHDataLoader& loader);
    // Q17 无基础实现，使用优化版本
    void run_q18(TPCHDataLoader& loader);
    void run_q19(TPCHDataLoader& loader);
    // Q20, Q21, Q22 无基础实现，使用优化版本
}

// 前向声明
void register_operator_metadata();

void register_tpch_query_configs() {
    auto& opt = TPCHQueryOptimizer::instance();
    auto& cat = catalog::catalog();

    // 注册 GPU 算子 (V68, V69) - 需要预热
    register_operator_metadata();

    // ========================================================================
    // 注册所有有效算子到系统表 (V24-V54)
    //
    // 成本模型: startup_ms (启动开销) + per_row_us * rows (每行处理成本)
    // min_rows: 最小适用行数 (低于此值可能不如基准版本)
    // max_rows: 最大适用行数 (0 = 无上限)
    // ========================================================================

    // ------------------------------------------------------------------------
    // V24: SelectionVector + DirectArrayAgg (选择向量零复制)
    // 适用: Q3, Q5, Q6, Q9 - 避免中间数据复制
    // ------------------------------------------------------------------------
    cat.register_operator("V24-SelectionVector", 0.2f, 0.025f, 50000, 0);
    cat.register_operator("V24-DirectArrayAgg", 0.15f, 0.02f, 10000, 0);

    // ------------------------------------------------------------------------
    // V25: ThreadPool + WeakHashTable (线程池复用 + Key Hash缓存)
    // 适用: Q3, Q5, Q6, Q7, Q9, Q10, Q12, Q14, Q18 - 并行+预计算
    // ------------------------------------------------------------------------
    cat.register_operator("V25-ThreadPool", 0.5f, 0.01f, 100000, 0);
    cat.register_operator("V25-WeakHashTable", 0.3f, 0.015f, 50000, 0);
    cat.register_operator("V25-DictEncodedJoin", 0.1f, 0.005f, 10000, 0);

    // ------------------------------------------------------------------------
    // V26: MutableWeakHashTable + BloomFilter (原地更新 + Bloom预过滤)
    // 适用: Q3, Q18 - SIMD批量hash, 分区聚合
    // ------------------------------------------------------------------------
    cat.register_operator("V26-MutableWeakHash", 0.25f, 0.012f, 50000, 0);
    cat.register_operator("V26-BloomPrefilter", 0.05f, 0.001f, 10000, 0);
    cat.register_operator("V26-VectorizedGroupBy", 0.2f, 0.008f, 100000, 0);

    // ------------------------------------------------------------------------
    // V27: Bitmap + StringDict + PredicatePrecomputer
    // 适用: Q3-Q19 (除Q8,Q13,Q17) - 位图O(1)测试, 字符串编码
    // ------------------------------------------------------------------------
    cat.register_operator("V27-BitmapSemiJoin", 0.1f, 0.002f, 10000, 0);
    cat.register_operator("V27-BitmapAntiJoin", 0.1f, 0.002f, 10000, 0);
    cat.register_operator("V27-StringDict", 0.05f, 0.001f, 1000, 0);
    cat.register_operator("V27-PredicatePrecomputer", 0.08f, 0.0005f, 5000, 0);

    // ------------------------------------------------------------------------
    // V32: CompactHashTable + 自适应策略 (CRC32硬件hash + 8路并行预取)
    // 适用: Q1-Q19 - 自适应SF阈值, Thread-Local聚合
    // ------------------------------------------------------------------------
    cat.register_operator("V32-CompactHash", 0.2f, 0.02f, 50000, 0);
    cat.register_operator("V32-CRC32ParallelHash", 0.15f, 0.008f, 100000, 0);
    cat.register_operator("V32-ThreadLocalAgg", 0.1f, 0.01f, 50000, 0);
    cat.register_operator("V32-AdaptiveSF", 0.3f, 0.015f, 100000, 0);

    // ------------------------------------------------------------------------
    // V33: 通用算子框架 Layer 2-4 (配置驱动, 零硬编码)
    // 适用: Q5, Q7, Q9, Q18, Q19 - 可自由组合
    // ------------------------------------------------------------------------
    cat.register_operator("V33-DateRangeFilter", 0.05f, 0.0008f, 10000, 0);
    cat.register_operator("V33-StringSetMatcher", 0.08f, 0.001f, 5000, 0);
    cat.register_operator("V33-AdaptiveHashJoin", 0.3f, 0.025f, 50000, 0);
    cat.register_operator("V33-ThreadLocalAggregator", 0.1f, 0.012f, 50000, 0);

    // ------------------------------------------------------------------------
    // V34: 通用化扩展 (ANTI/OUTER JOIN, CASE WHEN条件聚合)
    // 适用: Q8, Q13, Q22 - 复杂查询通用实现
    // ------------------------------------------------------------------------
    cat.register_operator("V34-GenericAntiJoin", 0.2f, 0.018f, 20000, 0);
    cat.register_operator("V34-GenericOuterJoin", 0.25f, 0.022f, 30000, 0);
    cat.register_operator("V34-ConditionalAggregator", 0.1f, 0.008f, 10000, 0);

    // ------------------------------------------------------------------------
    // V35: 通用高阶算子 (SIMD字符串处理, 管道融合)
    // 适用: Q3, Q5, Q7, Q8, Q9, Q13, Q14, Q21, Q22
    // ------------------------------------------------------------------------
    cat.register_operator("V35-DirectArrayIndexBuilder", 0.2f, 0.005f, 50000, 0);
    cat.register_operator("V35-SIMDStringProcessor", 0.1f, 0.003f, 10000, 0);
    cat.register_operator("V35-SemiAntiJoin", 0.15f, 0.01f, 20000, 0);
    cat.register_operator("V35-PipelineFusion", 0.05f, 0.002f, 100000, 0);

    // ------------------------------------------------------------------------
    // V36: 相关子查询解关联 (预计算聚合结果)
    // 适用: Q17, Q20 - 消除相关子查询开销
    // ------------------------------------------------------------------------
    cat.register_operator("V36-PrecomputedAggregates", 0.4f, 0.005f, 50000, 0);
    cat.register_operator("V36-SubqueryDecorrelation", 0.5f, 0.008f, 100000, 0);

    // ------------------------------------------------------------------------
    // V37: Bitmap Anti-Join (Bitmap O(1)存在性测试)
    // 适用: Q8, Q17, Q20, Q21, Q22 - 并行SIMD预过滤
    // ------------------------------------------------------------------------
    cat.register_operator("V37-BitmapExistenceSet", 0.1f, 0.001f, 10000, 0);
    cat.register_operator("V37-BitmapAntiJoin", 0.12f, 0.0015f, 20000, 0);
    cat.register_operator("V37-OrderKeyState", 0.15f, 0.002f, 50000, 0);

    // ------------------------------------------------------------------------
    // V40: 通用算子框架 (DynamicBitmapFilter, MergeJoin)
    // 适用: Q20 - 消除硬编码, 通用排序后聚合
    // ------------------------------------------------------------------------
    cat.register_operator("V40-DynamicBitmapFilter", 0.08f, 0.0012f, 10000, 0);
    cat.register_operator("V40-SortedGroupByAggregator", 0.3f, 0.02f, 50000, 0);
    cat.register_operator("V40-MergeJoinOperator", 0.2f, 0.015f, 50000, 0);

    // ------------------------------------------------------------------------
    // V41: 单遍预计算 + 直接数组 (O(1)订单状态查找)
    // 适用: Q21 - 消除排序开销
    // ------------------------------------------------------------------------
    cat.register_operator("V41-OrderStatePrecompute", 0.4f, 0.003f, 100000, 0);
    cat.register_operator("V41-DirectArrayLookup", 0.05f, 0.0005f, 10000, 0);

    // ------------------------------------------------------------------------
    // V45: 通用直接数组维度算子
    // 注: V42-Q8, V43-Q17, V44-Q3 专用算子已删除，由 V57 通用算子替代
    // ------------------------------------------------------------------------
    cat.register_operator("V45-DirectArrayDimension", 0.08f, 0.004f, 10000, 0);

    // ------------------------------------------------------------------------
    // V46: 通用直接数组 (自动检测key范围)
    // 适用: Q5, Q11, Q14 - L2缓存友好
    // ------------------------------------------------------------------------
    cat.register_operator("V46-DirectArrayFilter", 0.05f, 0.0008f, 5000, 0);
    cat.register_operator("V46-BitmapMembershipFilter", 0.03f, 0.0005f, 1000, 0);
    cat.register_operator("V46-DirectArrayAggregator", 0.1f, 0.006f, 20000, 0);

    // ------------------------------------------------------------------------
    // V47: 高性能通用算子 (并行基数排序, SIMD无分支)
    // 适用: Q6, Q13, Q21 - 自适应稀疏数组
    // ------------------------------------------------------------------------
    cat.register_operator("V47-ParallelRadixSort", 0.4f, 0.04f, 100000, 0);
    cat.register_operator("V47-SIMDBranchlessFilter", 0.05f, 0.0003f, 10000, 0);
    cat.register_operator("V47-SIMDPatternMatcher", 0.08f, 0.001f, 5000, 0);
    cat.register_operator("V47-SparseDirectArray", 0.1f, 0.005f, 10000, 0);

    // ------------------------------------------------------------------------
    // V49: TopN感知优化 (堆排序剪枝)
    // 适用: LIMIT N 查询 - 提前终止
    // ------------------------------------------------------------------------
    cat.register_operator("V49-TopNAware", 0.15f, 0.015f, 100000, 0);
    cat.register_operator("V49-HeapPruning", 0.1f, 0.01f, 50000, 0);

    // ------------------------------------------------------------------------
    // V50: 混合执行框架
    // ------------------------------------------------------------------------
    cat.register_operator("V50-HybridExecutor", 0.3f, 0.02f, 50000, 0);

    // ------------------------------------------------------------------------
    // V51: 高级算子套件 (RadixSort, PartitionedAgg, FusedFilterAgg)
    // ------------------------------------------------------------------------
    cat.register_operator("V51-RadixSort", 0.5f, 0.05f, 100000, 0);
    cat.register_operator("V51-PartitionedAgg", 0.3f, 0.03f, 50000, 0);
    cat.register_operator("V51-FusedFilterAgg", 0.1f, 0.01f, 100000, 0);
    cat.register_operator("V51", 0.3f, 0.03f, 50000, 0);

    // ------------------------------------------------------------------------
    // V52: BitmapPredicateIndex (位图谓词索引)
    // ------------------------------------------------------------------------
    cat.register_operator("V52-BitmapPredicateIndex", 0.05f, 0.0001f, 1000, 0);

    // ------------------------------------------------------------------------
    // V53: 增强型 Bitmap
    // ------------------------------------------------------------------------
    cat.register_operator("V53-EnhancedBitmap", 0.06f, 0.00015f, 2000, 0);

    // ------------------------------------------------------------------------
    // V54: NativeDoubleSIMDFilter (原生double列SIMD过滤)
    // 适用: Q6 - 8线程SIMD, 最低每行成本
    // ------------------------------------------------------------------------
    cat.register_operator("V54-NativeDoubleSIMDFilter", 0.1f, 0.0003f, 10000, 0);

    // ------------------------------------------------------------------------
    // V55: 通用算子框架 (SubqueryDecorrelation, GenericParallelMultiJoin, GenericTwoPhaseAgg)
    // 注: V55-AdaptiveQ12 专用算子已删除
    // ------------------------------------------------------------------------
    cat.register_operator("V55-SubqueryDecorrelation", 0.3f, 0.004f, 10000, 0);
    cat.register_operator("V55-GenericParallelMultiJoin", 0.5f, 0.006f, 100000, 0);
    cat.register_operator("V55-GenericTwoPhaseAgg", 0.2f, 0.003f, 50000, 0);

    // ------------------------------------------------------------------------
    // V56: 优化通用算子 (Direct Array + Bloom Filter + 预计算维度)
    // 注: V56-OptimizedQ5 专用算子已删除，由 V57 通用算子替代
    // ------------------------------------------------------------------------
    cat.register_operator("V56-DirectArrayDecorrelation", 0.2f, 0.002f, 5000, 0);
    cat.register_operator("V56-BloomFilteredJoin", 0.15f, 0.003f, 50000, 0);
    cat.register_operator("V56-DirectArrayTwoPhaseAgg", 0.15f, 0.002f, 30000, 0);

    // ------------------------------------------------------------------------
    // V57: 零开销通用算子框架 (AdaptiveMap + ZeroCostAggregator)
    // 适用: Q5, Q8, Q17 - 零硬编码 + 模板参数消除虚调用
    // ------------------------------------------------------------------------
    cat.register_operator("V57-AdaptiveMap", 0.1f, 0.002f, 5000, 0);
    cat.register_operator("V57-DirectArray", 0.05f, 0.001f, 1000, 0);
    cat.register_operator("V57-ZeroCostAggregator", 0.08f, 0.002f, 10000, 0);
    cat.register_operator("V57-ParallelScanner", 0.2f, 0.001f, 100000, 0);

    // ------------------------------------------------------------------------
    // V58: 深度优化算子 (DirectArrayAggregator, PrecomputedBitmap, ParallelScan)
    // 适用: Q3, Q9, Q2 - O(1) 热路径 + SIMD 批处理
    // ------------------------------------------------------------------------
    cat.register_operator("V58-DirectArrayAggregator", 0.05f, 0.0008f, 50000, 0);
    cat.register_operator("V58-PrecomputedBitmap", 0.03f, 0.0001f, 10000, 0);
    cat.register_operator("V58-ParallelScanExecutor", 0.1f, 0.0005f, 10000, 0);
    cat.register_operator("V58-FusedSIMDFilterAggregate", 0.08f, 0.0006f, 100000, 0);

    // ------------------------------------------------------------------------
    // V59: 延迟 JOIN + SIMD 字符串匹配
    // 适用: Q3, Q9 - 核心突破: 先聚合再 JOIN (减少 95% hash lookup)
    // ------------------------------------------------------------------------
    cat.register_operator("V59-DeferredJoin", 0.02f, 0.0003f, 100000, 0);  // 延迟 JOIN
    cat.register_operator("V59-MinHeapTopK", 0.01f, 0.0001f, 1000, 0);      // O(n log k) Top-K
    cat.register_operator("V59-SIMDStringMatch", 0.05f, 0.0002f, 10000, 0); // SIMD 批量字符串

    // ========================================================================
    // Q1: 定价汇总报告 - 单表聚合 (V69: GPU 分组聚合)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q1";
        config.description = "Pricing Summary Report (single table aggregation)";
        config.estimated_rows = 6000000;
        config.join_count = 0;
        config.has_aggregation = true;

        config.candidates = {
            // V69: GPU 分组聚合 (Block-local hash + Two-stage reduction)
            {"V69-GPUGroupAggregate", 10.0, 100000, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v69::is_v69_applicable("Q1", l.lineitem().count)) {
                     ops_v69::run_q1_v69(l);
                 } else {
                     queries::run_q1(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v69::is_gpu_available() && ctx.row_count >= 100000;
             }},
            // 基础版本作为回退
            {"Base", 9.15, 0, 0, queries::run_q1}
        };
        config.fallback = queries::run_q1;

        opt.register_query(config);
    }

    // ========================================================================
    // Q2: 最低成本供应商 - 4 表 JOIN (V56: Direct Array 解关联)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q2";
        config.description = "Minimum Cost Supplier (4-way join)";
        config.estimated_rows = 800000;
        config.join_count = 4;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V58: ParallelScan + PrecomputedBitmap + DirectArray (预期 2.0x)
            // 适用性: 需要足够数据量触发并行收益 (>= 10000 行)
            {"V58", 2.0, 0, 0,
             [](TPCHDataLoader& l) { ops_v58::run_q2_v58(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // V58 ParallelScan 需要足够数据量才能体现并行优势
                 return ctx.row_count >= 10000;
             }},
            // V56: Direct Array 解关联 + O(1) 查找
            // 适用性: 中等数据量，适合单线程直接数组
            {"V56", 1.5, 0, 0,
             [](TPCHDataLoader& l) { ops_v56::run_q2_v56(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // V56 适合较小数据集
                 return ctx.row_count >= 1000 && ctx.row_count < 1000000;
             }},
            // V55: SubqueryDecorrelation 预计算
            {"V55", 1.4, 0, 0,
             [](TPCHDataLoader& l) { ops_v55::run_q2_v55(l); },
             nullptr},  // 无特殊限制
            // Base fallback
            {"Base", 1.0, 0, 0, queries::run_q2, nullptr}
        };
        config.fallback = queries::run_q2;

        opt.register_query(config);
    }

    // ========================================================================
    // Q3: 运输优先级 - 3 表 JOIN + Top-N (V49: TopN-Aware 聚合)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q3";
        config.description = "Shipping Priority (3-way join with Top-N)";
        config.estimated_rows = 7500000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_top_n = true;

        config.candidates = {
            // V68.1: One-Pass GPU Fused + Cache (缓存后第二次运行更快)
            {"V68", 2.50, 100000, 0,
             [](TPCHDataLoader& l) { ops_v68::run_q3_v68(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v68::is_v68_q3_applicable(ctx.row_count, 10);
             }},
            // V63: 位图 + 直接数组 (CPU fallback)
            {"V63", 1.70, 0, 0,
             [](TPCHDataLoader& l) { ops_v63::run_q3_v63(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V49: Top-N Aware (实测 1.43x, V60 分片锁更慢)
            {"V49", 1.43, 0, 0,
             [](TPCHDataLoader& l) { ops_v49::run_q3_v49(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000 ||
                        ctx.max_key_range > 250000;
             }},
            // V31: GPU SEMI + V19.2 JOIN (中等数据量)
            {"V31", 1.14, 100000, 1000000,
             [](TPCHDataLoader& l) { ops_v27::run_q3_v31(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // GPU 在中等数据量有优势
                 return ctx.row_count >= 100000 && ctx.row_count <= 1000000;
             }},
            // V27: 小数据集
            {"V27", 0.82, 10000, 100000,
             [](TPCHDataLoader& l) { ops_v27::run_q3_v27(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count < 100000;
             }}
        };
        config.fallback = queries::run_q3;

        opt.register_query(config);
    }

    // ========================================================================
    // Q4: 订单优先级 - 2 表 SEMI JOIN (1.2x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q4";
        config.description = "Order Priority Checking (SEMI join)";
        config.estimated_rows = 3000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V27", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q4_v27(l); }}
        };
        config.fallback = queries::run_q4;

        opt.register_query(config);
    }

    // ========================================================================
    // Q5: 本地供应商收入 - 6 表 JOIN (V56: 预计算 order→nation)
    // V56 核心优化: 预计算 orderkey → cust_nation，消除热路径第 3 次 hash 查找
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q5";
        config.description = "Local Supplier Volume (6-way join)";
        config.estimated_rows = 8000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            // V61: SIMD Bitmask 批量过滤 (实测 2.05x)
            {"V61", 2.05, 0, 0,
             [](TPCHDataLoader& l) { ops_v61::run_q5_v61(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V57: 零开销通用算子 (实测 1.98x, V60 位图更慢)
            {"V57", 1.98, 0, 0,
             [](TPCHDataLoader& l) { ops_v57::run_q5_v57(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V56: 预计算 order→nation
            {"V56", 1.7, 0, 0,
             [](TPCHDataLoader& l) { ops_v56::run_q5_v56(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 50000;
             }},
            // V53: ChunkedDirectArray 优化
            {"V53", 1.6, 0, 0,
             [](TPCHDataLoader& l) { ops_v53::run_q5_v53(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 50000;
             }},
            // V52: BitmapPredicateIndex 优化
            {"V52", 1.5, 0, 0,
             [](TPCHDataLoader& l) { ops_v52::run_q5_v52(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 10000;
             }},
            // V51: PartitionedAggregation 优化
            {"V51", 1.4, 0, 0,
             [](TPCHDataLoader& l) { ops_v51::run_q5_v51(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 10000;
             }},
            // V32: 紧凑 Hash + 批量优化 (通用回退)
            {"V32", 1.27, 0, 0,
             [](TPCHDataLoader& l) { ops_v32::run_q5_v32(l); },
             nullptr}  // 无限制，作为回退
        };
        config.fallback = queries::run_q5;

        opt.register_query(config);
    }

    // ========================================================================
    // Q6: 预测收入变化 - 单表聚合 (V54: 通用 NativeDoubleSIMDFilter)
    // ========================================================================
    // 适用通用算子: V54-NativeDoubleSIMDFilter
    // 条件: join_count == 0, has_native_double == true, rows >= 10000
    {
        QueryOperatorConfig config;
        config.query_id = "Q6";
        config.description = "Forecasting Revenue Change (single table scan)";
        config.estimated_rows = 6000000;
        config.join_count = 0;  // 单表查询 - 适用 V54
        config.has_aggregation = true;

        config.candidates = {
            // V88: GPU Filter-Aggregate 融合 (Metal GPU 加速)
            {"V88-GPUFilterAgg", 4.0, 100000, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v88::is_v88_q6_available()) {
                     ops_v88::run_q6_v88(l);
                 } else {
                     ops_v66::run_q6_v66(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v88::is_v88_q6_available() && ctx.row_count >= 100000;
             }},
            // V66: GPU 融合过滤聚合 (真正 GPU 加速 + 性能收集)
            {"V66-GPUFusedFilterAgg", 3.0, 100000, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v66::is_v66_applicable("Q6", l.lineitem().count)) {
                     ops_v66::run_q6_v66(l);
                 } else {
                     ops_v64::run_q6_v64(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V64: SIMDVectorAggregator 通用框架
            // 原生 int32 SIMD 谓词 + int64 聚合
            {"V64-SIMDVectorAggregator", 2.5, 100000, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v64::is_v64_applicable("Q6", l.lineitem().count)) {
                     ops_v64::run_q6_v64(l);
                 } else {
                     ops_v54::run_q6_v54(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V54: 通用 NativeDoubleSIMDFilter 算子
            // 适用条件: 单表扫描 + 原生 double 列 + 多谓词过滤
            // 成本模型: 0.1ms 启动 + 0.0003us/行
            {"V54-NativeDoubleSIMDFilter", 2.0, 10000, 0,
             [](TPCHDataLoader& l) {
                 // 检查适用性
                 if (ops_v54::is_v54_applicable("Q6", l.lineitem().count, true)) {
                     ops_v54::run_q6_v54(l);
                 } else {
                     ops_v47::run_q6_v47(l);  // 回退到 V47
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // V54 适用条件:
                 // 1. 需要原生 double 列 (SIMD float64 操作)
                 // 2. 数据量足够大 (>= 10000 行)
                 return ctx.has_native_double && ctx.row_count >= 10000;
             }},
            // V47: SIMD 无分支过滤 (小数据量或无原生 double 时使用)
            {"V47-SIMDBranchless", 1.8, 0, 10000,
             [](TPCHDataLoader& l) { ops_v47::run_q6_v47(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // V47 作为小数据量回退，或非原生 double 场景
                 return ctx.row_count < 10000 || !ctx.has_native_double;
             }}
        };
        config.fallback = queries::run_q6;

        opt.register_query(config);
    }

    // ========================================================================
    // Q7: 国家间运输量 - 6 表 JOIN (2.63x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q7";
        config.description = "Volume Shipping (6-way join with date filter)";
        config.estimated_rows = 6000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            // V32: CompactHash + 8 路并行预取
            // 适用性: 多表 JOIN 大数据量
            {"V32", 2.63, 0, 0,
             [](TPCHDataLoader& l) { ops_v32::run_q7_v32(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V25: 较轻量实现 (小数据量)
            {"V25", 1.5, 0, 0,
             [](TPCHDataLoader& l) { ops_v25::run_q7_v25(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count < 100000;
             }}
        };
        config.fallback = queries::run_q7;

        opt.register_query(config);
    }

    // ========================================================================
    // Q8: 市场份额 - 8 表 JOIN (V42: 专用并行多表 JOIN)
    // 注: V56 测试结果 1.62x，V42 1.78x 更优
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q8";
        config.description = "National Market Share (8-way join)";
        config.estimated_rows = 8000000;
        config.join_count = 7;
        config.has_aggregation = true;

        config.candidates = {
            // V86: GPU Fused Multi-Probe (Metal GPU 加速)
            {"V86-GPUFusedProbe", 2.2, 0, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v88::is_v86_q8_available()) {
                     ops_v88::run_q8_v86(l);
                 } else {
                     ops_v57::run_q8_v57(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v88::is_v86_q8_available();
             }},
            // V57: 零开销通用算子 (通用设计，无专用代码)
            {"V57", 1.65, 0, 0, [](TPCHDataLoader& l) { ops_v57::run_q8_v57(l); }},
            // V34: 基础版本
            {"V34", 1.05, 0, 0, [](TPCHDataLoader& l) { ops_v34::run_q8_v34(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v34::run_q8_v34(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q9: 产品利润 - 6 表 JOIN (V32: CompactHash + Bloom)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q9";
        config.description = "Product Type Profit Measure (6-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 5;
        config.has_aggregation = true;

        config.candidates = {
            // V62: CRC32 + CompactHashTable64 (bitmap + 预取) (实测 1.94x)
            {"V62", 1.94, 50000, 0,
             [](TPCHDataLoader& l) { ops_v62::run_q9_v62(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 50000;
             }},
            // V32: 紧凑 Hash + CompactHashTable (实测 1.36x)
            {"V32", 1.36, 50000, 0,
             [](TPCHDataLoader& l) { ops_v32::run_q9_v32(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 50000;
             }},
            // V25: 较轻量实现 (小数据量)
            {"V25", 0.9, 0, 0,
             [](TPCHDataLoader& l) { ops_v25::run_q9_v25(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count < 50000;
             }}
        };
        config.fallback = queries::run_q9;

        opt.register_query(config);
    }

    // ========================================================================
    // Q10: 退货客户 - 4 表 JOIN (1.7x V25)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q10";
        config.description = "Returned Item Reporting (4-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_top_n = true;

        config.candidates = {
            {"V71", 2.5, 0, 0, [](TPCHDataLoader& l) { ops_v71::run_q10_v71(l); }},
            {"V25", 1.7, 0, 0, [](TPCHDataLoader& l) { ops_v25::run_q10_v25(l); }}
        };
        config.fallback = queries::run_q10;

        opt.register_query(config);
    }

    // ========================================================================
    // Q11: 重要库存 - 3 表 JOIN (4.14x V46)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q11";
        config.description = "Important Stock Identification (3-way join)";
        config.estimated_rows = 800000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V46: DirectArray 聚合
            // 适用性: partkey 范围需要适合 L2 缓存 (< 250K 个 int64)
            {"V46", 4.14, 0, 0,
             [](TPCHDataLoader& l) { ops_v46::run_q11_v46(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // DirectArray 适用条件:
                 // max_key_range * sizeof(int64_t) < L2 cache (4MB)
                 // => max_key_range < 500K (保守估计 250K 以留余量)
                 constexpr size_t L2_CACHE_ENTRY_LIMIT = 250000;
                 return ctx.max_key_range <= L2_CACHE_ENTRY_LIMIT ||
                        ctx.max_key_range == 0;  // 0 表示未知，允许尝试
             }},
            // V27: Hash 聚合 (通用回退)
            {"V27", 2.0, 0, 0,
             [](TPCHDataLoader& l) { ops_v27::run_q11_v27(l); },
             nullptr}  // 无限制
        };
        config.fallback = queries::run_q11;

        opt.register_query(config);
    }

    // ========================================================================
    // Q12: 运输方式统计 - 2 表 JOIN (V57 SIMD Branchless)
    // 分析: V57 使用 ZeroCostBranchlessFilter 解决分支预测问题
    // 之前的问题: 4 个过滤条件导致分支预测失败
    // V57 方案: SIMD 批量处理 + 无分支掩码合并
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q12";
        config.description = "Shipping Modes (SIMD Branchless Filter)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V57", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v57::run_q12_v57(l); }},
            {"V27", 0.9, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q12_v27(l); }}
        };
        config.fallback = queries::run_q12;

        opt.register_query(config);
    }

    // ========================================================================
    // Q13: 客户订单分布 - 2 表 LEFT OUTER JOIN (1.96x V34)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q13";
        config.description = "Customer Distribution (LEFT OUTER join)";
        config.estimated_rows = 1500000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V34", 1.96, 0, 0, [](TPCHDataLoader& l) { ops_v34::run_q13_v34(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v34::run_q13_v34(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q14: 促销效果 - 2 表 JOIN (2.91x V46)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q14";
        config.description = "Promotion Effect (2-way join)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            // V46: DirectArray 聚合 (O(1) 查找)
            // 适用性: partkey 范围需要适合 L2 缓存
            {"V46", 2.91, 0, 0,
             [](TPCHDataLoader& l) { ops_v46::run_q14_v46(l); },
             [](const QueryOperatorConfig::ApplicabilityContext&) {
                 // Q14 使用 partkey 作为聚合键
                 // TPC-H partkey 范围: SF*200K (SF=1 时 ~200K)
                 // 远小于 L2 缓存限制 (250K)，总是使用 DirectArray
                 return true;
             }},
            // V25: Hash 聚合 (大 key 范围回退)
            {"V25", 1.8, 0, 0,
             [](TPCHDataLoader& l) { ops_v25::run_q14_v25(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // 当 DirectArray 不适用时使用 Hash
                 return ctx.max_key_range > 250000;
             }}
        };
        config.fallback = queries::run_q14;

        opt.register_query(config);
    }

    // ========================================================================
    // Q15: 顶级供应商 - 2 表 JOIN (1.3x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q15";
        config.description = "Top Supplier Query (2-way join with subquery)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            {"V27", 1.3, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q15_v27(l); }}
        };
        config.fallback = queries::run_q15;

        opt.register_query(config);
    }

    // ========================================================================
    // Q16: 零件供应商关系 - 3 表 JOIN (1.2x V27)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q16";
        config.description = "Parts/Supplier Relationship (3-way join)";
        config.estimated_rows = 800000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V92: 并行扫描 + 基数排序 (目标 2.5x+)
            {"V92-ParallelRadix", 2.5, 100000, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v92::is_v92_q16_applicable(l.partsupp().count)) {
                     ops_v92::run_q16_v92(l);
                 } else {
                     ops_v27::run_q16_v27(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 100000;
             }},
            // V84: GPU HyperLogLog COUNT(DISTINCT)
            {"V84-GPUHyperLogLog", 2.0, 0, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v88::is_v84_q16_available()) {
                     ops_v88::run_q16_v84(l);
                 } else {
                     ops_v27::run_q16_v27(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v88::is_v84_q16_available();
             }},
            {"V27", 1.2, 0, 0, [](TPCHDataLoader& l) { ops_v27::run_q16_v27(l); }}
        };
        config.fallback = queries::run_q16;

        opt.register_query(config);
    }

    // ========================================================================
    // Q17: 小订单收入 - 2 表 JOIN (V57: 零开销通用算子)
    // 注: V56 测试结果 0.57x，V57 2.9x 更优
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q17";
        config.description = "Small-Quantity Order Revenue (2-way join with AVG)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V70: 解关联子查询优化 + GPU 启动成本优化 (最优)
            {"V70", 1.50, 0, 0,
             [](TPCHDataLoader& l) { ops_v70::run_q17_v70(l); },
             [](const QueryOperatorConfig::ApplicabilityContext&) {
                 return true;  // 总是适用
             }},
            // V36: SubqueryDecorrelation (回退)
            {"V36", 1.16, 0, 0,
             [](TPCHDataLoader& l) { ops_v36::run_q17_v36(l); },
             [](const QueryOperatorConfig::ApplicabilityContext&) {
                 return true;
             }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v70::run_q17_v70(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q18: 大订单客户 - 3 表 JOIN (4.27x V32)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q18";
        config.description = "Large Volume Customer (3-way join with HAVING)";
        config.estimated_rows = 7500000;
        config.join_count = 2;
        config.has_aggregation = true;
        config.has_top_n = true;
        config.has_subquery = true;

        config.candidates = {
            // V32: CompactHash + Bloom Filter + 8 路并行预取
            // 适用性: 大数据量时 Hash 开销可接受，Bloom 过滤效果好
            {"V32", 4.27, 0, 0,
             [](TPCHDataLoader& l) { ops_v32::run_q18_v32(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // V32 CompactHash 适合大规模数据
                 return ctx.row_count >= 100000;
             }},
            // V25: 较轻量的 Hash 实现 (小数据量)
            {"V25", 2.0, 0, 0,
             [](TPCHDataLoader& l) { ops_v25::run_q18_v25(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count < 100000;
             }}
        };
        config.fallback = queries::run_q18;

        opt.register_query(config);
    }

    // ========================================================================
    // Q19: 折扣收入 - 2 表 JOIN (2.0x V33)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q19";
        config.description = "Discounted Revenue (2-way join with OR conditions)";
        config.estimated_rows = 6000000;
        config.join_count = 1;
        config.has_aggregation = true;

        config.candidates = {
            {"V33", 2.0, 0, 0, [](TPCHDataLoader& l) { ops_v33::run_q19_v33(l); }}
        };
        config.fallback = queries::run_q19;

        opt.register_query(config);
    }

    // ========================================================================
    // Q20: 潜在零件促销 - 4 表 JOIN (1.29x V40)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q20";
        config.description = "Potential Part Promotion (4-way join with subquery)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V87: GPU Late Materialization
            {"V87-GPULateMaterialize", 2.0, 0, 0,
             [](TPCHDataLoader& l) {
                 if (ops_v88::is_v87_q20_available()) {
                     ops_v88::run_q20_v87(l);
                 } else {
                     ops_v40::run_q20_v40(l);
                 }
             },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ops_v88::is_v87_q20_available();
             }},
            {"V40", 1.29, 0, 0, [](TPCHDataLoader& l) { ops_v40::run_q20_v40(l); }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v40::run_q20_v40(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q21: 延迟供应商 - 4 表 JOIN (V51: ParallelRadixSort)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q21";
        config.description = "Suppliers Who Kept Orders Waiting (4-way join with EXISTS)";
        config.estimated_rows = 6000000;
        config.join_count = 3;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V51 最优: ParallelRadixSort - 两级基数排序 + 单遍 EXISTS 分析
            // 适用性: 大数据量时并行基数排序优势明显
            {"V51", 1.5, 0, 0,
             [](TPCHDataLoader& l) { ops_v51::run_q21_v51(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // ParallelRadixSort 需要足够数据量
                 return ctx.row_count >= 100000;
             }},
            // V50: HybridExecutor (中等数据量)
            {"V50", 1.2, 0, 0,
             [](TPCHDataLoader& l) { ops_v50::run_q21_v50(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count >= 50000 && ctx.row_count < 100000;
             }},
            // V48: 基础优化 (小数据量)
            {"V48", 1.0, 0, 0,
             [](TPCHDataLoader& l) { ops_v48::run_q21_v48(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 return ctx.row_count < 50000;
             }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v48::run_q21_v48(l); };

        opt.register_query(config);
    }

    // ========================================================================
    // Q22: 全球销售机会 - ANTI JOIN (9.08x V37)
    // ========================================================================
    {
        QueryOperatorConfig config;
        config.query_id = "Q22";
        config.description = "Global Sales Opportunity (ANTI join)";
        config.estimated_rows = 1500000;
        config.join_count = 1;
        config.has_aggregation = true;
        config.has_subquery = true;

        config.candidates = {
            // V37: Bitmap Anti-Join (O(1) 存在性测试)
            // 适用性: orderkey 范围适合 Bitmap (custkey 通常较小)
            {"V37", 9.08, 0, 0,
             [](TPCHDataLoader& l) { ops_v37::run_q22_v37(l); },
             [](const QueryOperatorConfig::ApplicabilityContext& ctx) {
                 // Bitmap Anti-Join 适用条件:
                 // orderkey 范围可以用 Bitmap 高效存储
                 // 对于 ANTI JOIN，Bitmap 几乎总是最优选择
                 return ctx.row_count >= 1000;  // 最小数据量阈值
             }}
        };
        config.fallback = [](TPCHDataLoader& l) { ops_v37::run_q22_v37(l); };

        opt.register_query(config);
    }
}

// ============================================================================
// 系统表集成实现
// ============================================================================

void TPCHQueryOptimizer::record_execution(
    const std::string& query_id,
    const std::string& version,
    double execution_time_ms,
    size_t rows_processed
) {
    // 使用轻量级 API 记录性能数据
    catalog::catalog().record_metric(query_id, version, execution_time_ms, rows_processed);

    // 记录版本选择
    catalog::catalog().record_selection(query_id, version, rows_processed);
}

TPCHQueryOptimizer::VersionPerformance TPCHQueryOptimizer::get_version_performance(
    const std::string& query_id,
    const std::string& version
) const {
    VersionPerformance perf{};
    perf.version = version;

    auto stats = catalog::catalog().get_version_stats(query_id, version);
    perf.avg_time_ms = stats.avg_time_ms;
    perf.median_time_ms = stats.median_time_ms;
    perf.stddev_ms = stats.stddev_ms;
    perf.sample_count = stats.sample_count;
    perf.confidence = catalog::catalog().get_version_confidence(query_id, version);

    return perf;
}

std::vector<TPCHQueryOptimizer::VersionPerformance> TPCHQueryOptimizer::get_all_version_performance(
    const std::string& query_id
) const {
    std::vector<VersionPerformance> result;

    auto it = configs_.find(query_id);
    if (it == configs_.end()) {
        return result;
    }

    for (const auto& candidate : it->second.candidates) {
        result.push_back(get_version_performance(query_id, candidate.version));
    }

    return result;
}

SelectionResult TPCHQueryOptimizer::select_adaptive(
    const std::string& query_id,
    const QueryOperatorConfig::ApplicabilityContext& ctx,
    const QueryOperatorConfig& config
) const {
    SelectionResult result;
    result.strategy_used = SelectionStrategy::ADAPTIVE;

    // 收集所有适用的候选版本
    std::vector<std::string> candidates;
    for (const auto& c : config.candidates) {
        // 检查行数范围
        if (ctx.row_count < c.min_rows) continue;
        if (c.max_rows > 0 && ctx.row_count > c.max_rows) continue;

        // 检查适用性函数 (如果提供)
        if (c.is_applicable && !c.is_applicable(ctx)) continue;

        candidates.push_back(c.version);
    }

    if (candidates.empty()) {
        result.selected_version = "Fallback";
        result.executor = config.fallback;
        result.confidence = 0.5;
        result.selection_reason = "No candidates for context";
        return result;
    }

    // 使用系统表选择最优版本
    auto [best_version, predicted_time] =
        catalog::catalog().select_best_version(query_id, ctx.row_count, candidates);

    if (best_version.empty()) {
        // 没有历史数据，回退到静态选择
        return select_static(query_id, ctx, config);
    }

    // 查找对应的执行器
    for (const auto& c : config.candidates) {
        if (c.version == best_version) {
            result.selected_version = best_version;
            result.executor = c.executor;
            result.predicted_time_ms = predicted_time;
            result.confidence = catalog::catalog().get_version_confidence(query_id, best_version);
            result.selection_reason = "Adaptive (predicted=" +
                std::to_string(static_cast<int>(predicted_time)) + "ms, conf=" +
                std::to_string(static_cast<int>(result.confidence * 100)) + "%)";

            if (logging_enabled_) {
                log_selection(query_id, result);
            }

            return result;
        }
    }

    // 未找到，回退
    result.selected_version = "Fallback";
    result.executor = config.fallback;
    result.confidence = 0.5;
    result.selection_reason = "Version not found in candidates";
    return result;
}

SelectionResult TPCHQueryOptimizer::select_hybrid(
    const std::string& query_id,
    const QueryOperatorConfig::ApplicabilityContext& ctx,
    const QueryOperatorConfig& config
) const {
    SelectionResult result;
    result.strategy_used = SelectionStrategy::HYBRID;

    // 首先尝试自适应选择
    auto adaptive_result = select_adaptive(query_id, ctx, config);

    // 如果置信度足够高，使用自适应结果
    constexpr double CONFIDENCE_THRESHOLD = 0.7;

    if (adaptive_result.confidence >= CONFIDENCE_THRESHOLD) {
        adaptive_result.strategy_used = SelectionStrategy::HYBRID;
        adaptive_result.selection_reason = "Hybrid/Adaptive (" +
            adaptive_result.selection_reason.substr(adaptive_result.selection_reason.find('(') + 1);
        return adaptive_result;
    }

    // 否则，混合静态配置
    auto static_result = select_static(query_id, ctx, config);

    // 如果有历史数据但置信度低，加权混合
    if (adaptive_result.predicted_time_ms > 0 && !adaptive_result.selected_version.empty()) {
        // 比较两种选择
        double static_speedup = 0.0;
        for (const auto& c : config.candidates) {
            if (c.version == static_result.selected_version) {
                static_speedup = c.speedup;
                break;
            }
        }

        // 如果自适应选择的版本预测性能更好，即使置信度低也优先使用
        // (预测时间与静态加速比反相关)
        if (adaptive_result.predicted_time_ms < 1000.0 / static_speedup) {
            adaptive_result.strategy_used = SelectionStrategy::HYBRID;
            adaptive_result.selection_reason = "Hybrid/Low-conf adaptive";
            return adaptive_result;
        }
    }

    static_result.strategy_used = SelectionStrategy::HYBRID;
    static_result.selection_reason = "Hybrid/Static (low historical conf)";

    if (logging_enabled_) {
        log_selection(query_id, static_result);
    }

    return static_result;
}

// ============================================================================
// 算子元数据注册
// ============================================================================

void register_operator_metadata() {
    auto& cat = catalog::catalog();

    // 注册 V68 GPU 算子 (专用注册函数)
    ops_v68::register_v68_operators();

    // 注册 V69 GPU 分组聚合算子
    ops_v69::register_v69_operators();

    // 注册各版本算子的元数据 (使用轻量级 API)
    // 参数: version, startup_ms, per_row_us, min_rows, max_rows

    // Filter 算子
    cat.register_operator("V19", 0.1f, 0.005f, 0, 0);         // 8线程并行 SIMD Filter
    cat.register_operator("V19.1", 0.05f, 0.003f, 0, 0);      // 无分支 SIMD Filter

    // Join 算子
    cat.register_operator("V19.2", 0.5f, 0.02f, 10000, 0);    // 激进预取 + SIMD Hash Join
    cat.register_operator("GPU", 2.0f, 0.001f, 100000, 0);    // Metal GPU SEMI Join

    // Aggregate 算子
    cat.register_operator("V15", 0.2f, 0.01f, 0, 0);          // 8线程 + 8路展开聚合
    cat.register_operator("V46", 0.05f, 0.001f, 0, 0);        // 低基数直接数组聚合

    // Top-N 算子
    cat.register_operator("V49", 0.1f, 0.008f, 0, 0);         // Top-N Aware Partial Aggregation

    // Bitmap 算子
    cat.register_operator("V37", 0.3f, 0.002f, 0, 0);         // Bitmap Anti-Join

    // TPC-H 查询版本
    cat.register_operator("V25", 0.2f, 0.015f, 0, 0);
    cat.register_operator("V27", 0.2f, 0.012f, 0, 0);
    cat.register_operator("V31", 0.3f, 0.01f, 100000, 0);
    cat.register_operator("V32", 0.3f, 0.008f, 100000, 0);
    cat.register_operator("V33", 0.2f, 0.01f, 0, 0);
    cat.register_operator("V34", 0.3f, 0.009f, 0, 0);
    cat.register_operator("V36", 0.2f, 0.008f, 0, 0);
    cat.register_operator("V40", 0.3f, 0.007f, 0, 0);
    cat.register_operator("V42", 0.3f, 0.006f, 0, 0);
    cat.register_operator("V43", 0.2f, 0.005f, 0, 0);
    cat.register_operator("V47", 0.1f, 0.003f, 0, 0);
    cat.register_operator("V48", 0.3f, 0.008f, 0, 0);
    cat.register_operator("V50", 0.2f, 0.006f, 0, 0);
    cat.register_operator("V52", 0.15f, 0.004f, 0, 0);  // DirectArrayJoin + SIMDBranchlessFilter
    cat.register_operator("V53", 0.12f, 0.003f, 0, 0);  // QueryArena + ChunkedDirectArray + TypeLifted
    cat.register_operator("Base", 0.1f, 0.02f, 0, 0);

    // ========================================================================
    // V68: One-Pass GPU Fused Q3 (Apple Silicon Metal)
    // ========================================================================
    // 架构: 单遍历 lineitem + Block-local hash 聚合 + Two-stage Top-N
    // 特性: UMA 零拷贝 + orders_flags 缓存 + GPU/CPU 自适应
    // 适用: Q3 查询，lineitem >= 100K 行
    // ========================================================================

    // V68 GPU Fused 算子 (完整管道)
    // startup: GPU 预热 + Metal 编译 (~7ms 冷启动, 0ms 缓存后)
    // per_row: GPU 并行处理 (~0.7µs/row for 6M rows = ~4ms)
    cat.register_operator("V68-GPUFusedQ3", 7.0f, 0.0007f, 100000, 0);

    // V68 子算子 (用于细粒度成本分析)
    // Phase 0: orders_flags 构建 (GPU 或 CPU)
    cat.register_operator("V68-OrdersFlagsBuilder-GPU", 7.0f, 0.004f, 100000, 0);
    cat.register_operator("V68-OrdersFlagsBuilder-CPU", 6.0f, 0.004f, 10000, 0);
    cat.register_operator("V68-OrdersFlagsCache", 0.0f, 0.0f, 0, 0);  // 缓存命中时零成本

    // Phase 1: GPU 扫描聚合
    // Block-local hash (1024 entries) + SIMD 累加 + Two-stage Top-N
    cat.register_operator("V68-GPUScanAggregate", 0.5f, 0.0006f, 100000, 0);

    // Phase 2: CPU 合并
    // 多 block TopK 合并到最终 Top-N
    cat.register_operator("V68-CPUMerge", 0.4f, 0.0001f, 1000, 0);

    // V68 缓存预热版本 (第二次及以后运行)
    // 缓存命中: Phase 0 = 0ms, 只有 Phase 1 + 2
    cat.register_operator("V68-GPUFusedQ3-Warmed", 0.5f, 0.0007f, 100000, 0);

    // ========================================================================
    // V63: CPU Bitmap + DirectArray (V68 的 CPU 回退)
    // ========================================================================
    cat.register_operator("V63-BitmapDirectArray", 2.0f, 0.001f, 50000, 0);
}

} // namespace tpch
} // namespace thunderduck
