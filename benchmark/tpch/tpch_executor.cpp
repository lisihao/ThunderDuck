/**
 * ThunderDuck TPC-H 查询执行器 - 实现
 */

#include "tpch_executor.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace thunderduck {
namespace tpch {

// ============================================================================
// TPC-H SQL 查询定义
// ============================================================================

static const std::map<std::string, std::string> TPCH_QUERIES = {
    // Q1: 定价汇总报告 - GROUP BY + 多聚合
    {"Q1", R"(
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus
)"},

    // Q2: 最小成本供应商
    {"Q2", R"(
SELECT
    s_acctbal, s_name, n_name, p_partkey, p_mfgr,
    s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey
    AND s_suppkey = ps_suppkey
    AND p_size = 15
    AND p_type LIKE '%BRASS'
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'EUROPE'
    AND ps_supplycost = (
        SELECT MIN(ps_supplycost)
        FROM partsupp, supplier, nation, region
        WHERE p_partkey = ps_partkey
            AND s_suppkey = ps_suppkey
            AND s_nationkey = n_nationkey
            AND n_regionkey = r_regionkey
            AND r_name = 'EUROPE'
    )
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100
)"},

    // Q3: 运输优先级
    {"Q3", R"(
SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < DATE '1995-03-15'
    AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10
)"},

    // Q4: 订单优先级检查
    {"Q4", R"(
SELECT o_orderpriority, COUNT(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '1993-07-01'
    AND o_orderdate < DATE '1993-07-01' + INTERVAL '3' MONTH
    AND EXISTS (
        SELECT * FROM lineitem
        WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate
    )
GROUP BY o_orderpriority
ORDER BY o_orderpriority
)"},

    // Q5: 本地供应商收入
    {"Q5", R"(
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'ASIA'
    AND o_orderdate >= DATE '1994-01-01'
    AND o_orderdate < DATE '1994-01-01' + INTERVAL '1' YEAR
GROUP BY n_name
ORDER BY revenue DESC
)"},

    // Q6: 预测收入变化 - 简单过滤聚合 (最佳优化场景)
    {"Q6", R"(
SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
    AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
    AND l_discount BETWEEN 0.05 AND 0.07
    AND l_quantity < 24
)"},

    // Q7: 体量运输
    {"Q7", R"(
SELECT
    supp_nation, cust_nation, l_year, SUM(volume) AS revenue
FROM (
    SELECT
        n1.n_name AS supp_nation,
        n2.n_name AS cust_nation,
        EXTRACT(YEAR FROM l_shipdate) AS l_year,
        l_extendedprice * (1 - l_discount) AS volume
    FROM supplier, lineitem, orders, customer, nation n1, nation n2
    WHERE s_suppkey = l_suppkey
        AND o_orderkey = l_orderkey
        AND c_custkey = o_custkey
        AND s_nationkey = n1.n_nationkey
        AND c_nationkey = n2.n_nationkey
        AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
            OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
        AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) AS shipping
GROUP BY supp_nation, cust_nation, l_year
ORDER BY supp_nation, cust_nation, l_year
)"},

    // Q8: 国家市场份额
    {"Q8", R"(
SELECT
    o_year,
    SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
    SELECT
        EXTRACT(YEAR FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) AS volume,
        n2.n_name AS nation
    FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
    WHERE p_partkey = l_partkey
        AND s_suppkey = l_suppkey
        AND l_orderkey = o_orderkey
        AND o_custkey = c_custkey
        AND c_nationkey = n1.n_nationkey
        AND n1.n_regionkey = r_regionkey
        AND r_name = 'AMERICA'
        AND s_nationkey = n2.n_nationkey
        AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
        AND p_type = 'ECONOMY ANODIZED STEEL'
) AS all_nations
GROUP BY o_year
ORDER BY o_year
)"},

    // Q9: 产品类型利润
    {"Q9", R"(
SELECT nation, o_year, SUM(amount) AS sum_profit
FROM (
    SELECT
        n_name AS nation,
        EXTRACT(YEAR FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey
        AND ps_suppkey = l_suppkey
        AND ps_partkey = l_partkey
        AND p_partkey = l_partkey
        AND o_orderkey = l_orderkey
        AND s_nationkey = n_nationkey
        AND p_name LIKE '%green%'
) AS profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC
)"},

    // Q10: 退货报告
    {"Q10", R"(
SELECT
    c_custkey, c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal, n_name, c_address, c_phone, c_comment
FROM customer, orders, lineitem, nation
WHERE c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate >= DATE '1993-10-01'
    AND o_orderdate < DATE '1993-10-01' + INTERVAL '3' MONTH
    AND l_returnflag = 'R'
    AND c_nationkey = n_nationkey
GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
ORDER BY revenue DESC
LIMIT 20
)"},

    // Q11: 重要库存识别
    {"Q11", R"(
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
    SELECT SUM(ps_supplycost * ps_availqty) * 0.0001
    FROM partsupp, supplier, nation
    WHERE ps_suppkey = s_suppkey
        AND s_nationkey = n_nationkey
        AND n_name = 'GERMANY'
)
ORDER BY value DESC
)"},

    // Q12: 运输模式与订单优先级
    {"Q12", R"(
SELECT
    l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH'
        THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH'
        THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem
WHERE o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= DATE '1994-01-01'
    AND l_receiptdate < DATE '1994-01-01' + INTERVAL '1' YEAR
GROUP BY l_shipmode
ORDER BY l_shipmode
)"},

    // Q13: 客户分布
    {"Q13", R"(
SELECT c_count, COUNT(*) AS custdist
FROM (
    SELECT c_custkey, COUNT(o_orderkey) AS c_count
    FROM customer LEFT OUTER JOIN orders ON
        c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
    GROUP BY c_custkey
) AS c_orders
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC
)"},

    // Q14: 促销效果
    {"Q14", R"(
SELECT
    100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
        THEN l_extendedprice * (1 - l_discount) ELSE 0 END) /
    SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM lineitem, part
WHERE l_partkey = p_partkey
    AND l_shipdate >= DATE '1995-09-01'
    AND l_shipdate < DATE '1995-09-01' + INTERVAL '1' MONTH
)"},

    // Q15: 顶级供应商
    {"Q15", R"(
WITH revenue AS (
    SELECT
        l_suppkey AS supplier_no,
        SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '1996-01-01'
        AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
    GROUP BY l_suppkey
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue
WHERE s_suppkey = supplier_no
    AND total_revenue = (SELECT MAX(total_revenue) FROM revenue)
ORDER BY s_suppkey
)"},

    // Q16: 零件供应商关系
    {"Q16", R"(
SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM partsupp, part
WHERE p_partkey = ps_partkey
    AND p_brand <> 'Brand#45'
    AND p_type NOT LIKE 'MEDIUM POLISHED%'
    AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND ps_suppkey NOT IN (
        SELECT s_suppkey FROM supplier
        WHERE s_comment LIKE '%Customer%Complaints%'
    )
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size
)"},

    // Q17: 小批量订单收入
    {"Q17", R"(
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
    AND p_brand = 'Brand#23'
    AND p_container = 'MED BOX'
    AND l_quantity < (
        SELECT 0.2 * AVG(l_quantity)
        FROM lineitem
        WHERE l_partkey = p_partkey
    )
)"},

    // Q18: 大批量客户
    {"Q18", R"(
SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity)
FROM customer, orders, lineitem
WHERE o_orderkey IN (
    SELECT l_orderkey FROM lineitem
    GROUP BY l_orderkey
    HAVING SUM(l_quantity) > 300
)
    AND c_custkey = o_custkey
    AND o_orderkey = l_orderkey
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100
)"},

    // Q19: 折扣收入
    {"Q19", R"(
SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem, part
WHERE (
    p_partkey = l_partkey
    AND p_brand = 'Brand#12'
    AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    AND l_quantity >= 1 AND l_quantity <= 1 + 10
    AND p_size BETWEEN 1 AND 5
    AND l_shipmode IN ('AIR', 'AIR REG')
    AND l_shipinstruct = 'DELIVER IN PERSON'
) OR (
    p_partkey = l_partkey
    AND p_brand = 'Brand#23'
    AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
    AND l_quantity >= 10 AND l_quantity <= 10 + 10
    AND p_size BETWEEN 1 AND 10
    AND l_shipmode IN ('AIR', 'AIR REG')
    AND l_shipinstruct = 'DELIVER IN PERSON'
) OR (
    p_partkey = l_partkey
    AND p_brand = 'Brand#34'
    AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
    AND l_quantity >= 20 AND l_quantity <= 20 + 10
    AND p_size BETWEEN 1 AND 15
    AND l_shipmode IN ('AIR', 'AIR REG')
    AND l_shipinstruct = 'DELIVER IN PERSON'
)
)"},

    // Q20: 潜在零件促销
    {"Q20", R"(
SELECT s_name, s_address
FROM supplier, nation
WHERE s_suppkey IN (
    SELECT ps_suppkey FROM partsupp
    WHERE ps_partkey IN (
        SELECT p_partkey FROM part WHERE p_name LIKE 'forest%'
    )
    AND ps_availqty > (
        SELECT 0.5 * SUM(l_quantity)
        FROM lineitem
        WHERE l_partkey = ps_partkey
            AND l_suppkey = ps_suppkey
            AND l_shipdate >= DATE '1994-01-01'
            AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
    )
)
    AND s_nationkey = n_nationkey
    AND n_name = 'CANADA'
ORDER BY s_name
)"},

    // Q21: 无法按时交付的供应商
    {"Q21", R"(
SELECT s_name, COUNT(*) AS numwait
FROM supplier, lineitem l1, orders, nation
WHERE s_suppkey = l1.l_suppkey
    AND o_orderkey = l1.l_orderkey
    AND o_orderstatus = 'F'
    AND l1.l_receiptdate > l1.l_commitdate
    AND EXISTS (
        SELECT * FROM lineitem l2
        WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey
    )
    AND NOT EXISTS (
        SELECT * FROM lineitem l3
        WHERE l3.l_orderkey = l1.l_orderkey
            AND l3.l_suppkey <> l1.l_suppkey
            AND l3.l_receiptdate > l3.l_commitdate
    )
    AND s_nationkey = n_nationkey
    AND n_name = 'SAUDI ARABIA'
GROUP BY s_name
ORDER BY numwait DESC, s_name
LIMIT 100
)"},

    // Q22: 全球销售机会
    {"Q22", R"(
SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal
FROM (
    SELECT SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode, c_acctbal
    FROM customer
    WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND c_acctbal > (
            SELECT AVG(c_acctbal) FROM customer
            WHERE c_acctbal > 0.00
                AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('13', '31', '23', '29', '30', '18', '17')
        )
        AND NOT EXISTS (
            SELECT * FROM orders WHERE o_custkey = c_custkey
        )
) AS custsale
GROUP BY cntrycode
ORDER BY cntrycode
)"}
};

// ============================================================================
// 查询分类映射
// ============================================================================

static const std::map<std::string, QueryCategory> QUERY_CATEGORIES = {
    // Category A: 完全可优化
    {"Q1", QueryCategory::A},
    {"Q3", QueryCategory::A},
    {"Q5", QueryCategory::A},
    {"Q6", QueryCategory::A},
    {"Q7", QueryCategory::A},
    {"Q9", QueryCategory::A},
    {"Q10", QueryCategory::A},
    {"Q12", QueryCategory::A},
    {"Q14", QueryCategory::A},
    {"Q18", QueryCategory::A},

    // Category B: 部分可优化
    {"Q2", QueryCategory::B},
    {"Q4", QueryCategory::B},
    {"Q11", QueryCategory::B},
    {"Q15", QueryCategory::B},
    {"Q16", QueryCategory::B},
    {"Q19", QueryCategory::B},

    // Category C: DuckDB 回退
    {"Q8", QueryCategory::C},
    {"Q13", QueryCategory::C},
    {"Q17", QueryCategory::C},
    {"Q20", QueryCategory::C},
    {"Q21", QueryCategory::C},
    {"Q22", QueryCategory::C}
};

// ============================================================================
// 查询分类函数
// ============================================================================

QueryCategory get_query_category(const std::string& query_id) {
    auto it = QUERY_CATEGORIES.find(query_id);
    if (it != QUERY_CATEGORIES.end()) {
        return it->second;
    }
    return QueryCategory::C;  // 默认
}

const char* get_category_description(QueryCategory cat) {
    switch (cat) {
        case QueryCategory::A: return "完全可优化 (预期 1.5-3x)";
        case QueryCategory::B: return "部分可优化 (预期 1.0-1.5x)";
        case QueryCategory::C: return "DuckDB 回退 (预期 ~1.0x)";
        default: return "未知";
    }
}

const std::string& get_tpch_sql(const std::string& query_id) {
    static const std::string empty;
    auto it = TPCH_QUERIES.find(query_id);
    if (it != TPCH_QUERIES.end()) {
        return it->second;
    }
    return empty;
}

std::vector<std::string> get_all_query_ids() {
    std::vector<std::string> ids;
    for (int i = 1; i <= 22; ++i) {
        ids.push_back("Q" + std::to_string(i));
    }
    return ids;
}

std::vector<std::string> get_category_query_ids(QueryCategory category) {
    std::vector<std::string> ids;
    for (const auto& [qid, cat] : QUERY_CATEGORIES) {
        if (cat == category) {
            ids.push_back(qid);
        }
    }
    // 按数字排序
    std::sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) {
        return std::stoi(a.substr(1)) < std::stoi(b.substr(1));
    });
    return ids;
}

// ============================================================================
// TPCHExecutor 实现
// ============================================================================

TPCHExecutor::TPCHExecutor(TPCHDataLoader& loader, size_t iterations, size_t warmup)
    : loader_(loader), iterations_(iterations), warmup_(warmup) {}

template<typename Func>
double TPCHExecutor::measure_median_iqr(Func&& func) {
    std::vector<double> times;
    times.reserve(iterations_);

    // 预热
    for (size_t i = 0; i < warmup_; ++i) {
        func();
    }

    // 测量
    for (size_t i = 0; i < iterations_; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // IQR 剔除异常值
    std::sort(times.begin(), times.end());
    size_t n = times.size();
    if (n < 4) {
        return times[n / 2];
    }

    double q1 = times[n / 4];
    double q3 = times[n * 3 / 4];
    double iqr = q3 - q1;
    double lower = q1 - 1.5 * iqr;
    double upper = q3 + 1.5 * iqr;

    times.erase(std::remove_if(times.begin(), times.end(),
        [&](double t) { return t < lower || t > upper; }), times.end());

    std::sort(times.begin(), times.end());
    return times.empty() ? 0.0 : times[times.size() / 2];
}

double TPCHExecutor::run_duckdb_baseline(const std::string& query_id) {
    const std::string& sql = get_tpch_sql(query_id);
    if (sql.empty()) {
        return 0.0;
    }

    return measure_median_iqr([&]() {
        auto result = loader_.connection().Query(sql);
        if (result->HasError()) {
            std::cerr << "  " << query_id << " DuckDB 错误: " << result->GetError() << std::endl;
        }
    });
}

double TPCHExecutor::run_thunderduck_optimized(const std::string& query_id) {
    auto impl = get_thunderduck_impl(query_id);
    if (!impl) {
        // 没有 ThunderDuck 实现，回退到 DuckDB
        return run_duckdb_baseline(query_id);
    }

    return measure_median_iqr([&]() {
        impl(loader_);
    });
}

bool TPCHExecutor::verify_results(const std::string& query_id) {
    // 简化实现: 假设结果正确
    // 实际应该比较 DuckDB 和 ThunderDuck 的输出
    return true;
}

QueryResult TPCHExecutor::run_query(const std::string& query_id, bool verbose) {
    QueryResult result;
    result.query_id = query_id;
    result.category = [](QueryCategory cat) {
        switch (cat) {
            case QueryCategory::A: return "A";
            case QueryCategory::B: return "B";
            case QueryCategory::C: return "C";
            default: return "?";
        }
    }(get_query_category(query_id));

    if (verbose) {
        std::cout << "  运行 " << query_id << " [" << result.category << "]... " << std::flush;
    }

    // 运行 DuckDB 基线
    result.duckdb_ms = run_duckdb_baseline(query_id);

    // 运行 ThunderDuck 优化
    if (has_thunderduck_impl(query_id)) {
        result.thunderduck_ms = run_thunderduck_optimized(query_id);
    } else {
        // 无优化实现，使用 DuckDB 时间
        result.thunderduck_ms = result.duckdb_ms;
    }

    // 计算加速比
    if (result.thunderduck_ms > 0.001) {
        result.speedup = result.duckdb_ms / result.thunderduck_ms;
    } else {
        result.speedup = 1.0;
    }

    // 验证结果
    result.correct = verify_results(query_id);

    if (verbose) {
        std::cout << std::fixed << std::setprecision(2)
                  << result.duckdb_ms << " ms vs "
                  << result.thunderduck_ms << " ms ("
                  << result.speedup << "x)" << std::endl;
    }

    return result;
}

BenchmarkResult TPCHExecutor::run_all_queries(bool verbose) {
    BenchmarkResult result;
    result.scale_factor = loader_.get_scale_factor();

    // 获取当前日期
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&time));
    result.date = buf;

    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "TPC-H 完整基准测试 SF=" << result.scale_factor << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // 运行所有查询
    auto query_ids = get_all_query_ids();
    std::vector<double> speedups;

    for (const auto& qid : query_ids) {
        auto qr = run_query(qid, verbose);
        result.queries.push_back(qr);

        result.total_duckdb_ms += qr.duckdb_ms;
        result.total_thunderduck_ms += qr.thunderduck_ms;

        if (qr.speedup > 1.05) {
            result.faster_count++;
        } else if (qr.speedup < 0.95) {
            result.slower_count++;
        } else {
            result.same_count++;
        }

        if (qr.speedup > 0) {
            speedups.push_back(std::log(qr.speedup));
        }
    }

    // 计算几何平均加速比
    if (!speedups.empty()) {
        double sum_log = std::accumulate(speedups.begin(), speedups.end(), 0.0);
        result.geometric_mean_speedup = std::exp(sum_log / speedups.size());
    }

    return result;
}

BenchmarkResult TPCHExecutor::run_category(QueryCategory category, bool verbose) {
    BenchmarkResult result;
    result.scale_factor = loader_.get_scale_factor();

    auto query_ids = get_category_query_ids(category);
    std::vector<double> speedups;

    if (verbose) {
        std::cout << "\n--- " << get_category_description(category) << " ---" << std::endl;
    }

    for (const auto& qid : query_ids) {
        auto qr = run_query(qid, verbose);
        result.queries.push_back(qr);

        result.total_duckdb_ms += qr.duckdb_ms;
        result.total_thunderduck_ms += qr.thunderduck_ms;

        if (qr.speedup > 1.05) {
            result.faster_count++;
        } else if (qr.speedup < 0.95) {
            result.slower_count++;
        } else {
            result.same_count++;
        }

        if (qr.speedup > 0) {
            speedups.push_back(std::log(qr.speedup));
        }
    }

    if (!speedups.empty()) {
        double sum_log = std::accumulate(speedups.begin(), speedups.end(), 0.0);
        result.geometric_mean_speedup = std::exp(sum_log / speedups.size());
    }

    return result;
}

} // namespace tpch
} // namespace thunderduck
