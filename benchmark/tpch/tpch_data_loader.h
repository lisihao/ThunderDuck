/**
 * ThunderDuck TPC-H 数据加载器
 *
 * 从 DuckDB 加载 TPC-H 数据并转换为列式存储
 *
 * @version 1.0
 * @date 2026-01-28
 */

#ifndef TPCH_DATA_LOADER_H
#define TPCH_DATA_LOADER_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>

// DuckDB 头文件
#include "duckdb.hpp"

// 统一常量定义
#include "tpch_constants.h"

namespace thunderduck {
namespace tpch {

// ============================================================================
// TPC-H 表的列式存储结构
// ============================================================================

/**
 * LINEITEM 表 - TPC-H 最大表 (SF=1 约 6M 行)
 * 列使用定点数存储以提高 SIMD 性能
 */
struct LineitemColumns {
    size_t count = 0;

    // 主键/外键
    std::vector<int32_t> l_orderkey;      // 订单键
    std::vector<int32_t> l_partkey;       // 零件键
    std::vector<int32_t> l_suppkey;       // 供应商键
    std::vector<int32_t> l_linenumber;    // 行号

    // 数值列 (定点数 x10000) - 保持兼容性
    std::vector<int64_t> l_quantity;       // 数量 (x10000)
    std::vector<int64_t> l_extendedprice;  // 扩展价格 (x10000)
    std::vector<int64_t> l_discount;       // 折扣 (x10000, 0.00-0.10)
    std::vector<int64_t> l_tax;            // 税率 (x10000)

    // 数值列 (原生 double) - V54+ SIMD 优化使用
    std::vector<double> l_quantity_d;       // 数量 (原生 double)
    std::vector<double> l_extendedprice_d;  // 扩展价格 (原生 double)
    std::vector<double> l_discount_d;       // 折扣 (原生 double)

    // 日期列 (epoch days)
    std::vector<int32_t> l_shipdate;       // 发货日期
    std::vector<int32_t> l_commitdate;     // 承诺日期
    std::vector<int32_t> l_receiptdate;    // 收货日期

    // 字符列 (编码为整数)
    std::vector<int8_t> l_returnflag;      // 退货标志 (A=0, N=1, R=2)
    std::vector<int8_t> l_linestatus;      // 行状态 (F=0, O=1)
    std::vector<int8_t> l_shipinstruct;    // 发货指令 (编码)
    std::vector<int8_t> l_shipmode;        // 发货模式 (编码)

    // 原始字符串 (用于验证)
    std::vector<std::string> l_returnflag_str;
    std::vector<std::string> l_linestatus_str;
    std::vector<std::string> l_shipinstruct_str;
    std::vector<std::string> l_shipmode_str;
    std::vector<std::string> l_comment;
};

/**
 * ORDERS 表 (SF=1 约 1.5M 行)
 */
struct OrdersColumns {
    size_t count = 0;

    std::vector<int32_t> o_orderkey;       // 订单键
    std::vector<int32_t> o_custkey;        // 客户键
    std::vector<int8_t> o_orderstatus;     // 订单状态 (F=0, O=1, P=2)
    std::vector<int64_t> o_totalprice;     // 总价 (x10000)
    std::vector<int32_t> o_orderdate;      // 订单日期
    std::vector<int8_t> o_orderpriority;   // 订单优先级 (编码)
    std::vector<std::string> o_clerk;      // 职员
    std::vector<int32_t> o_shippriority;   // 发货优先级
    std::vector<std::string> o_comment;
    std::vector<std::string> o_orderstatus_str;
    std::vector<std::string> o_orderpriority_str;
};

/**
 * CUSTOMER 表 (SF=1 约 150K 行)
 */
struct CustomerColumns {
    size_t count = 0;

    std::vector<int32_t> c_custkey;        // 客户键
    std::vector<std::string> c_name;       // 客户名称
    std::vector<std::string> c_address;    // 地址
    std::vector<int32_t> c_nationkey;      // 国家键
    std::vector<std::string> c_phone;      // 电话
    std::vector<int64_t> c_acctbal;        // 账户余额 (x10000)
    std::vector<std::string> c_mktsegment; // 市场细分
    std::vector<int8_t> c_mktsegment_code; // 市场细分编码
    std::vector<std::string> c_comment;
};

/**
 * PART 表 (SF=1 约 200K 行)
 */
struct PartColumns {
    size_t count = 0;

    std::vector<int32_t> p_partkey;        // 零件键
    std::vector<std::string> p_name;       // 零件名称
    std::vector<std::string> p_mfgr;       // 制造商
    std::vector<std::string> p_brand;      // 品牌
    std::vector<std::string> p_type;       // 类型
    std::vector<int32_t> p_size;           // 尺寸
    std::vector<std::string> p_container;  // 容器
    std::vector<int64_t> p_retailprice;    // 零售价 (x10000)
    std::vector<std::string> p_comment;
};

/**
 * SUPPLIER 表 (SF=1 约 10K 行)
 */
struct SupplierColumns {
    size_t count = 0;

    std::vector<int32_t> s_suppkey;        // 供应商键
    std::vector<std::string> s_name;       // 供应商名称
    std::vector<std::string> s_address;    // 地址
    std::vector<int32_t> s_nationkey;      // 国家键
    std::vector<std::string> s_phone;      // 电话
    std::vector<int64_t> s_acctbal;        // 账户余额 (x10000)
    std::vector<std::string> s_comment;
};

/**
 * PARTSUPP 表 (SF=1 约 800K 行)
 */
struct PartsuppColumns {
    size_t count = 0;

    std::vector<int32_t> ps_partkey;       // 零件键
    std::vector<int32_t> ps_suppkey;       // 供应商键
    std::vector<int32_t> ps_availqty;      // 可用数量
    std::vector<int64_t> ps_supplycost;    // 供应成本 (x10000)
    std::vector<std::string> ps_comment;
};

/**
 * NATION 表 (25 行)
 */
struct NationColumns {
    size_t count = 0;

    std::vector<int32_t> n_nationkey;      // 国家键
    std::vector<std::string> n_name;       // 国家名称
    std::vector<int32_t> n_regionkey;      // 区域键
    std::vector<std::string> n_comment;
};

/**
 * REGION 表 (5 行)
 */
struct RegionColumns {
    size_t count = 0;

    std::vector<int32_t> r_regionkey;      // 区域键
    std::vector<std::string> r_name;       // 区域名称
    std::vector<std::string> r_comment;
};

// ============================================================================
// 数据加载器
// ============================================================================

/**
 * TPC-H 数据加载器
 *
 * 功能:
 * - 使用 DuckDB TPC-H 扩展生成数据
 * - 将数据提取到列式内存结构
 * - 支持多种数据类型转换
 */
class TPCHDataLoader {
public:
    explicit TPCHDataLoader(duckdb::Connection& con);
    ~TPCHDataLoader() = default;

    /**
     * 生成 TPC-H 数据
     * @param scale_factor 缩放因子 (1, 10, 100, ...)
     * @return 是否成功
     */
    bool generate_data(int scale_factor);

    /**
     * 提取所有表到内存
     */
    void extract_all_tables();

    /**
     * 获取缩放因子
     */
    int get_scale_factor() const { return scale_factor_; }

    /**
     * 获取各表数据
     */
    const LineitemColumns& lineitem() const { return lineitem_; }
    const OrdersColumns& orders() const { return orders_; }
    const CustomerColumns& customer() const { return customer_; }
    const PartColumns& part() const { return part_; }
    const SupplierColumns& supplier() const { return supplier_; }
    const PartsuppColumns& partsupp() const { return partsupp_; }
    const NationColumns& nation() const { return nation_; }
    const RegionColumns& region() const { return region_; }

    /**
     * 获取 DuckDB 连接 (用于基线查询)
     */
    duckdb::Connection& connection() { return con_; }

    /**
     * 打印统计信息
     */
    void print_stats() const;

private:
    duckdb::Connection& con_;
    int scale_factor_ = 0;

    // 各表数据
    LineitemColumns lineitem_;
    OrdersColumns orders_;
    CustomerColumns customer_;
    PartColumns part_;
    SupplierColumns supplier_;
    PartsuppColumns partsupp_;
    NationColumns nation_;
    RegionColumns region_;

    // 内部提取方法
    void extract_lineitem();
    void extract_orders();
    void extract_customer();
    void extract_part();
    void extract_supplier();
    void extract_partsupp();
    void extract_nation();
    void extract_region();

    // 字符串编码辅助
    static int8_t encode_returnflag(const std::string& s);
    static int8_t encode_linestatus(const std::string& s);
    static int8_t encode_orderstatus(const std::string& s);
    static int8_t encode_orderpriority(const std::string& s);
    static int8_t encode_shipinstruct(const std::string& s);
    static int8_t encode_shipmode(const std::string& s);
    static int8_t encode_mktsegment(const std::string& s);
};

// ============================================================================
// 日期转换工具
// ============================================================================

/**
 * 将日期字符串转换为 epoch days
 * @param date_str 格式 "YYYY-MM-DD"
 * @return epoch days (从 1970-01-01 开始)
 */
int32_t date_to_epoch_days(const std::string& date_str);

/**
 * 将 epoch days 转换为日期字符串
 */
std::string epoch_days_to_date(int32_t days);

/**
 * 常用日期常量 (epoch days)
 *
 * 向后兼容别名 - 实际定义在 tpch_constants.h
 */
namespace dates {
    using namespace constants::dates;

    // Q1: 1998-12-01 - 90 days = 1998-09-02
    constexpr int32_t Q1_THRESHOLD = constants::query_params::q1::SHIP_DATE_THRESHOLD;

    // Q6: 1994-01-01 to 1995-01-01
    constexpr int32_t Q6_DATE_LO = constants::query_params::q6::DATE_LO;
    constexpr int32_t Q6_DATE_HI = constants::query_params::q6::DATE_HI;

    // Q3: 1995-03-15
    constexpr int32_t Q3_DATE = constants::query_params::q3::ORDER_DATE_THRESHOLD;

    // 通用日期常量别名 (向后兼容)
    constexpr int32_t DATE_1993_07_01 = D1993_07_01;
    constexpr int32_t DATE_1993_10_01 = D1993_10_01;
    constexpr int32_t DATE_1994_01_01 = D1994_01_01;
    constexpr int32_t DATE_1995_01_01 = D1995_01_01;
    constexpr int32_t DATE_1995_09_01 = D1995_09_01;
    constexpr int32_t DATE_1995_10_01 = D1995_10_01;
    constexpr int32_t DATE_1996_01_01 = D1996_01_01;
    constexpr int32_t DATE_1996_04_01 = D1996_04_01;
    constexpr int32_t DATE_1996_12_31 = D1996_12_31;
    constexpr int32_t DATE_1998_12_01 = D1998_12_01;
}

// ============================================================================
// 定点数转换工具
// ============================================================================

/**
 * 将 double 转换为定点数 (x10000)
 */
inline int64_t to_fixed(double val) {
    return static_cast<int64_t>(val * 10000.0 + 0.5);
}

/**
 * 将定点数转换为 double
 */
inline double from_fixed(int64_t val) {
    return static_cast<double>(val) / 10000.0;
}

/**
 * 定点数乘法 (结果 x10000)
 * a (x10000) * b (x10000) = result (x10000)
 */
inline int64_t fixed_mul(int64_t a, int64_t b) {
    return (a * b) / 10000;
}

/**
 * 定点数除法 (结果 x10000)
 */
inline int64_t fixed_div(int64_t a, int64_t b) {
    return (a * 10000) / b;
}

} // namespace tpch
} // namespace thunderduck

#endif // TPCH_DATA_LOADER_H
