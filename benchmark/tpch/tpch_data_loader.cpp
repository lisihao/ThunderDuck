/**
 * ThunderDuck TPC-H 数据加载器 - 实现
 */

#include "tpch_data_loader.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <iomanip>

namespace thunderduck {
namespace tpch {

// ============================================================================
// TPCHDataLoader 实现
// ============================================================================

TPCHDataLoader::TPCHDataLoader(duckdb::Connection& con) : con_(con) {}

bool TPCHDataLoader::generate_data(int scale_factor) {
    scale_factor_ = scale_factor;

    std::cout << "生成 TPC-H SF=" << scale_factor << " 数据..." << std::endl;

    // 加载 TPC-H 扩展
    auto load_result = con_.Query("INSTALL tpch; LOAD tpch;");
    if (load_result->HasError()) {
        std::cerr << "  加载 TPC-H 扩展失败: " << load_result->GetError() << std::endl;
        return false;
    }

    // 生成数据
    std::stringstream ss;
    ss << "CALL dbgen(sf=" << scale_factor << ");";
    auto gen_result = con_.Query(ss.str());
    if (gen_result->HasError()) {
        std::cerr << "  生成 TPC-H 数据失败: " << gen_result->GetError() << std::endl;
        return false;
    }

    std::cout << "  TPC-H 数据生成完成" << std::endl;
    return true;
}

void TPCHDataLoader::extract_all_tables() {
    std::cout << "提取 TPC-H 表到内存..." << std::endl;

    extract_nation();
    extract_region();
    extract_supplier();
    extract_customer();
    extract_part();
    extract_partsupp();
    extract_orders();
    extract_lineitem();

    std::cout << "  所有表提取完成" << std::endl;
}

void TPCHDataLoader::print_stats() const {
    std::cout << "\n=== TPC-H 数据统计 (SF=" << scale_factor_ << ") ===" << std::endl;
    std::cout << "  LINEITEM: " << lineitem_.count << " 行" << std::endl;
    std::cout << "  ORDERS:   " << orders_.count << " 行" << std::endl;
    std::cout << "  CUSTOMER: " << customer_.count << " 行" << std::endl;
    std::cout << "  PART:     " << part_.count << " 行" << std::endl;
    std::cout << "  SUPPLIER: " << supplier_.count << " 行" << std::endl;
    std::cout << "  PARTSUPP: " << partsupp_.count << " 行" << std::endl;
    std::cout << "  NATION:   " << nation_.count << " 行" << std::endl;
    std::cout << "  REGION:   " << region_.count << " 行" << std::endl;

    // 估算内存占用
    size_t lineitem_mem = lineitem_.count * (
        sizeof(int32_t) * 4 +   // keys
        sizeof(int64_t) * 4 +   // numeric
        sizeof(int32_t) * 3 +   // dates
        sizeof(int8_t) * 4      // flags
    );
    size_t orders_mem = orders_.count * (
        sizeof(int32_t) * 3 +
        sizeof(int64_t) * 1 +
        sizeof(int8_t) * 2
    );
    size_t total_mem = lineitem_mem + orders_mem;

    std::cout << "  估算核心列内存: " << (total_mem / (1024*1024)) << " MB" << std::endl;
}

// ============================================================================
// 表提取方法
// ============================================================================

void TPCHDataLoader::extract_lineitem() {
    auto result = con_.Query(R"(
        SELECT
            l_orderkey, l_partkey, l_suppkey, l_linenumber,
            l_quantity, l_extendedprice, l_discount, l_tax,
            l_shipdate, l_commitdate, l_receiptdate,
            l_returnflag, l_linestatus, l_shipinstruct, l_shipmode, l_comment
        FROM lineitem
    )");

    if (result->HasError()) {
        std::cerr << "  提取 LINEITEM 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    lineitem_.count = n;

    // 预分配
    lineitem_.l_orderkey.resize(n);
    lineitem_.l_partkey.resize(n);
    lineitem_.l_suppkey.resize(n);
    lineitem_.l_linenumber.resize(n);
    lineitem_.l_quantity.resize(n);
    lineitem_.l_extendedprice.resize(n);
    lineitem_.l_discount.resize(n);
    lineitem_.l_tax.resize(n);
    lineitem_.l_shipdate.resize(n);
    lineitem_.l_commitdate.resize(n);
    lineitem_.l_receiptdate.resize(n);
    lineitem_.l_returnflag.resize(n);
    lineitem_.l_linestatus.resize(n);
    lineitem_.l_shipinstruct.resize(n);
    lineitem_.l_shipmode.resize(n);
    lineitem_.l_returnflag_str.resize(n);
    lineitem_.l_linestatus_str.resize(n);
    lineitem_.l_shipinstruct_str.resize(n);
    lineitem_.l_shipmode_str.resize(n);
    lineitem_.l_comment.resize(n);

    // 逐行提取
    for (size_t i = 0; i < n; ++i) {
        lineitem_.l_orderkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        lineitem_.l_partkey[i] = result->GetValue(1, i).GetValue<int32_t>();
        lineitem_.l_suppkey[i] = result->GetValue(2, i).GetValue<int32_t>();
        lineitem_.l_linenumber[i] = result->GetValue(3, i).GetValue<int32_t>();

        // 转换为定点数
        lineitem_.l_quantity[i] = to_fixed(result->GetValue(4, i).GetValue<double>());
        lineitem_.l_extendedprice[i] = to_fixed(result->GetValue(5, i).GetValue<double>());
        lineitem_.l_discount[i] = to_fixed(result->GetValue(6, i).GetValue<double>());
        lineitem_.l_tax[i] = to_fixed(result->GetValue(7, i).GetValue<double>());

        // 日期 (DuckDB 返回 int32_t epoch days)
        lineitem_.l_shipdate[i] = result->GetValue(8, i).GetValue<int32_t>();
        lineitem_.l_commitdate[i] = result->GetValue(9, i).GetValue<int32_t>();
        lineitem_.l_receiptdate[i] = result->GetValue(10, i).GetValue<int32_t>();

        // 字符列
        auto rf = result->GetValue(11, i).GetValue<std::string>();
        auto ls = result->GetValue(12, i).GetValue<std::string>();
        auto si = result->GetValue(13, i).GetValue<std::string>();
        auto sm = result->GetValue(14, i).GetValue<std::string>();

        lineitem_.l_returnflag_str[i] = rf;
        lineitem_.l_linestatus_str[i] = ls;
        lineitem_.l_shipinstruct_str[i] = si;
        lineitem_.l_shipmode_str[i] = sm;
        lineitem_.l_comment[i] = result->GetValue(15, i).GetValue<std::string>();

        lineitem_.l_returnflag[i] = encode_returnflag(rf);
        lineitem_.l_linestatus[i] = encode_linestatus(ls);
        lineitem_.l_shipinstruct[i] = encode_shipinstruct(si);
        lineitem_.l_shipmode[i] = encode_shipmode(sm);
    }

    std::cout << "  LINEITEM: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_orders() {
    auto result = con_.Query(R"(
        SELECT
            o_orderkey, o_custkey, o_orderstatus, o_totalprice,
            o_orderdate, o_orderpriority, o_clerk, o_shippriority, o_comment
        FROM orders
    )");

    if (result->HasError()) {
        std::cerr << "  提取 ORDERS 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    orders_.count = n;

    orders_.o_orderkey.resize(n);
    orders_.o_custkey.resize(n);
    orders_.o_orderstatus.resize(n);
    orders_.o_totalprice.resize(n);
    orders_.o_orderdate.resize(n);
    orders_.o_orderpriority.resize(n);
    orders_.o_clerk.resize(n);
    orders_.o_shippriority.resize(n);
    orders_.o_comment.resize(n);
    orders_.o_orderstatus_str.resize(n);
    orders_.o_orderpriority_str.resize(n);

    for (size_t i = 0; i < n; ++i) {
        orders_.o_orderkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        orders_.o_custkey[i] = result->GetValue(1, i).GetValue<int32_t>();

        auto status = result->GetValue(2, i).GetValue<std::string>();
        orders_.o_orderstatus_str[i] = status;
        orders_.o_orderstatus[i] = encode_orderstatus(status);

        orders_.o_totalprice[i] = to_fixed(result->GetValue(3, i).GetValue<double>());
        orders_.o_orderdate[i] = result->GetValue(4, i).GetValue<int32_t>();

        auto priority = result->GetValue(5, i).GetValue<std::string>();
        orders_.o_orderpriority_str[i] = priority;
        orders_.o_orderpriority[i] = encode_orderpriority(priority);

        orders_.o_clerk[i] = result->GetValue(6, i).GetValue<std::string>();
        orders_.o_shippriority[i] = result->GetValue(7, i).GetValue<int32_t>();
        orders_.o_comment[i] = result->GetValue(8, i).GetValue<std::string>();
    }

    std::cout << "  ORDERS: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_customer() {
    auto result = con_.Query(R"(
        SELECT
            c_custkey, c_name, c_address, c_nationkey,
            c_phone, c_acctbal, c_mktsegment, c_comment
        FROM customer
    )");

    if (result->HasError()) {
        std::cerr << "  提取 CUSTOMER 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    customer_.count = n;

    customer_.c_custkey.resize(n);
    customer_.c_name.resize(n);
    customer_.c_address.resize(n);
    customer_.c_nationkey.resize(n);
    customer_.c_phone.resize(n);
    customer_.c_acctbal.resize(n);
    customer_.c_mktsegment.resize(n);
    customer_.c_mktsegment_code.resize(n);
    customer_.c_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        customer_.c_custkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        customer_.c_name[i] = result->GetValue(1, i).GetValue<std::string>();
        customer_.c_address[i] = result->GetValue(2, i).GetValue<std::string>();
        customer_.c_nationkey[i] = result->GetValue(3, i).GetValue<int32_t>();
        customer_.c_phone[i] = result->GetValue(4, i).GetValue<std::string>();
        customer_.c_acctbal[i] = to_fixed(result->GetValue(5, i).GetValue<double>());

        auto segment = result->GetValue(6, i).GetValue<std::string>();
        customer_.c_mktsegment[i] = segment;
        customer_.c_mktsegment_code[i] = encode_mktsegment(segment);

        customer_.c_comment[i] = result->GetValue(7, i).GetValue<std::string>();
    }

    std::cout << "  CUSTOMER: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_part() {
    auto result = con_.Query(R"(
        SELECT
            p_partkey, p_name, p_mfgr, p_brand,
            p_type, p_size, p_container, p_retailprice, p_comment
        FROM part
    )");

    if (result->HasError()) {
        std::cerr << "  提取 PART 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    part_.count = n;

    part_.p_partkey.resize(n);
    part_.p_name.resize(n);
    part_.p_mfgr.resize(n);
    part_.p_brand.resize(n);
    part_.p_type.resize(n);
    part_.p_size.resize(n);
    part_.p_container.resize(n);
    part_.p_retailprice.resize(n);
    part_.p_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        part_.p_partkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        part_.p_name[i] = result->GetValue(1, i).GetValue<std::string>();
        part_.p_mfgr[i] = result->GetValue(2, i).GetValue<std::string>();
        part_.p_brand[i] = result->GetValue(3, i).GetValue<std::string>();
        part_.p_type[i] = result->GetValue(4, i).GetValue<std::string>();
        part_.p_size[i] = result->GetValue(5, i).GetValue<int32_t>();
        part_.p_container[i] = result->GetValue(6, i).GetValue<std::string>();
        part_.p_retailprice[i] = to_fixed(result->GetValue(7, i).GetValue<double>());
        part_.p_comment[i] = result->GetValue(8, i).GetValue<std::string>();
    }

    std::cout << "  PART: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_supplier() {
    auto result = con_.Query(R"(
        SELECT
            s_suppkey, s_name, s_address, s_nationkey,
            s_phone, s_acctbal, s_comment
        FROM supplier
    )");

    if (result->HasError()) {
        std::cerr << "  提取 SUPPLIER 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    supplier_.count = n;

    supplier_.s_suppkey.resize(n);
    supplier_.s_name.resize(n);
    supplier_.s_address.resize(n);
    supplier_.s_nationkey.resize(n);
    supplier_.s_phone.resize(n);
    supplier_.s_acctbal.resize(n);
    supplier_.s_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        supplier_.s_suppkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        supplier_.s_name[i] = result->GetValue(1, i).GetValue<std::string>();
        supplier_.s_address[i] = result->GetValue(2, i).GetValue<std::string>();
        supplier_.s_nationkey[i] = result->GetValue(3, i).GetValue<int32_t>();
        supplier_.s_phone[i] = result->GetValue(4, i).GetValue<std::string>();
        supplier_.s_acctbal[i] = to_fixed(result->GetValue(5, i).GetValue<double>());
        supplier_.s_comment[i] = result->GetValue(6, i).GetValue<std::string>();
    }

    std::cout << "  SUPPLIER: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_partsupp() {
    auto result = con_.Query(R"(
        SELECT
            ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment
        FROM partsupp
    )");

    if (result->HasError()) {
        std::cerr << "  提取 PARTSUPP 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    partsupp_.count = n;

    partsupp_.ps_partkey.resize(n);
    partsupp_.ps_suppkey.resize(n);
    partsupp_.ps_availqty.resize(n);
    partsupp_.ps_supplycost.resize(n);
    partsupp_.ps_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        partsupp_.ps_partkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        partsupp_.ps_suppkey[i] = result->GetValue(1, i).GetValue<int32_t>();
        partsupp_.ps_availqty[i] = result->GetValue(2, i).GetValue<int32_t>();
        partsupp_.ps_supplycost[i] = to_fixed(result->GetValue(3, i).GetValue<double>());
        partsupp_.ps_comment[i] = result->GetValue(4, i).GetValue<std::string>();
    }

    std::cout << "  PARTSUPP: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_nation() {
    auto result = con_.Query(R"(
        SELECT n_nationkey, n_name, n_regionkey, n_comment FROM nation
    )");

    if (result->HasError()) {
        std::cerr << "  提取 NATION 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    nation_.count = n;

    nation_.n_nationkey.resize(n);
    nation_.n_name.resize(n);
    nation_.n_regionkey.resize(n);
    nation_.n_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        nation_.n_nationkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        nation_.n_name[i] = result->GetValue(1, i).GetValue<std::string>();
        nation_.n_regionkey[i] = result->GetValue(2, i).GetValue<int32_t>();
        nation_.n_comment[i] = result->GetValue(3, i).GetValue<std::string>();
    }

    std::cout << "  NATION: " << n << " 行" << std::endl;
}

void TPCHDataLoader::extract_region() {
    auto result = con_.Query(R"(
        SELECT r_regionkey, r_name, r_comment FROM region
    )");

    if (result->HasError()) {
        std::cerr << "  提取 REGION 失败: " << result->GetError() << std::endl;
        return;
    }

    size_t n = result->RowCount();
    region_.count = n;

    region_.r_regionkey.resize(n);
    region_.r_name.resize(n);
    region_.r_comment.resize(n);

    for (size_t i = 0; i < n; ++i) {
        region_.r_regionkey[i] = result->GetValue(0, i).GetValue<int32_t>();
        region_.r_name[i] = result->GetValue(1, i).GetValue<std::string>();
        region_.r_comment[i] = result->GetValue(2, i).GetValue<std::string>();
    }

    std::cout << "  REGION: " << n << " 行" << std::endl;
}

// ============================================================================
// 编码函数
// ============================================================================

int8_t TPCHDataLoader::encode_returnflag(const std::string& s) {
    if (s.empty()) return 0;
    switch (s[0]) {
        case 'A': return 0;
        case 'N': return 1;
        case 'R': return 2;
        default: return 0;
    }
}

int8_t TPCHDataLoader::encode_linestatus(const std::string& s) {
    if (s.empty()) return 0;
    switch (s[0]) {
        case 'F': return 0;
        case 'O': return 1;
        default: return 0;
    }
}

int8_t TPCHDataLoader::encode_orderstatus(const std::string& s) {
    if (s.empty()) return 0;
    switch (s[0]) {
        case 'F': return 0;
        case 'O': return 1;
        case 'P': return 2;
        default: return 0;
    }
}

int8_t TPCHDataLoader::encode_orderpriority(const std::string& s) {
    // 1-URGENT, 2-HIGH, 3-MEDIUM, 4-NOT SPECIFIED, 5-LOW
    if (s.find("1-URGENT") != std::string::npos) return 0;
    if (s.find("2-HIGH") != std::string::npos) return 1;
    if (s.find("3-MEDIUM") != std::string::npos) return 2;
    if (s.find("4-NOT SPECIFIED") != std::string::npos) return 3;
    if (s.find("5-LOW") != std::string::npos) return 4;
    return 3;  // default
}

int8_t TPCHDataLoader::encode_shipinstruct(const std::string& s) {
    // DELIVER IN PERSON, COLLECT COD, NONE, TAKE BACK RETURN
    if (s.find("DELIVER IN PERSON") != std::string::npos) return 0;
    if (s.find("COLLECT COD") != std::string::npos) return 1;
    if (s.find("NONE") != std::string::npos) return 2;
    if (s.find("TAKE BACK RETURN") != std::string::npos) return 3;
    return 2;  // default
}

int8_t TPCHDataLoader::encode_shipmode(const std::string& s) {
    // REG AIR, AIR, RAIL, SHIP, TRUCK, MAIL, FOB
    if (s == "REG AIR") return 0;
    if (s == "AIR") return 1;
    if (s == "RAIL") return 2;
    if (s == "SHIP") return 3;
    if (s == "TRUCK") return 4;
    if (s == "MAIL") return 5;
    if (s == "FOB") return 6;
    return 0;  // default
}

int8_t TPCHDataLoader::encode_mktsegment(const std::string& s) {
    // AUTOMOBILE, BUILDING, FURNITURE, HOUSEHOLD, MACHINERY
    if (s == "AUTOMOBILE") return 0;
    if (s == "BUILDING") return 1;
    if (s == "FURNITURE") return 2;
    if (s == "HOUSEHOLD") return 3;
    if (s == "MACHINERY") return 4;
    return 0;  // default
}

// ============================================================================
// 日期转换
// ============================================================================

int32_t date_to_epoch_days(const std::string& date_str) {
    // 简化实现: 假设格式为 YYYY-MM-DD
    int year, month, day;
    sscanf(date_str.c_str(), "%d-%d-%d", &year, &month, &day);

    // 简化的日期计算 (不考虑闰年等细节)
    int days = (year - 1970) * 365 + (year - 1969) / 4;  // 闰年近似
    static const int month_days[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    days += month_days[month - 1] + day - 1;

    return days;
}

std::string epoch_days_to_date(int32_t days) {
    // 简化实现
    int year = 1970 + days / 365;
    int day_of_year = days % 365;

    static const int month_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int month = 0;
    while (month < 12 && day_of_year >= month_days[month]) {
        day_of_year -= month_days[month];
        month++;
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d", year, month + 1, day_of_year + 1);
    return buf;
}

} // namespace tpch
} // namespace thunderduck
