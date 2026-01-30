/**
 * ThunderDuck TPC-H 通用常量定义
 *
 * 所有 TPC-H 查询相关的常量必须在此文件中定义，禁止在其他文件中硬编码。
 *
 * @version 1.0.0
 * @date 2026-01-30
 */

#ifndef TPCH_CONSTANTS_H
#define TPCH_CONSTANTS_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {
namespace tpch {
namespace constants {

// ============================================================================
// 日期常量 (epoch days from 1970-01-01)
// ============================================================================

namespace dates {
    // 1993 年
    constexpr int32_t D1993_01_01 = 8401;
    constexpr int32_t D1993_07_01 = 8582;
    constexpr int32_t D1993_10_01 = 8674;

    // 1994 年
    constexpr int32_t D1994_01_01 = 8766;
    constexpr int32_t D1994_03_01 = 8825;

    // 1995 年
    constexpr int32_t D1995_01_01 = 9131;
    constexpr int32_t D1995_03_01 = 9190;
    constexpr int32_t D1995_03_15 = 9204;
    constexpr int32_t D1995_09_01 = 9374;
    constexpr int32_t D1995_10_01 = 9404;

    // 1996 年
    constexpr int32_t D1996_01_01 = 9497;
    constexpr int32_t D1996_04_01 = 9588;
    constexpr int32_t D1996_12_31 = 9862;

    // 1997 年
    constexpr int32_t D1997_01_01 = 9862;
    constexpr int32_t D1997_03_01 = 9921;

    // 1998 年
    constexpr int32_t D1998_01_01 = 10227;
    constexpr int32_t D1998_09_02 = 10471;
    constexpr int32_t D1998_12_01 = 10561;
}

// ============================================================================
// 地区/国家名称常量
// ============================================================================

namespace regions {
    constexpr const char* AFRICA = "AFRICA";
    constexpr const char* AMERICA = "AMERICA";
    constexpr const char* ASIA = "ASIA";
    constexpr const char* EUROPE = "EUROPE";
    constexpr const char* MIDDLE_EAST = "MIDDLE EAST";
}

namespace nations {
    constexpr const char* BRAZIL = "BRAZIL";
    constexpr const char* CANADA = "CANADA";
    constexpr const char* CHINA = "CHINA";
    constexpr const char* FRANCE = "FRANCE";
    constexpr const char* GERMANY = "GERMANY";
    constexpr const char* INDIA = "INDIA";
    constexpr const char* INDONESIA = "INDONESIA";
    constexpr const char* JAPAN = "JAPAN";
    constexpr const char* SAUDI_ARABIA = "SAUDI ARABIA";
    constexpr const char* UNITED_KINGDOM = "UNITED KINGDOM";
    constexpr const char* UNITED_STATES = "UNITED STATES";
    constexpr const char* VIETNAM = "VIETNAM";
}

// ============================================================================
// 运输方式编码 (l_shipmode 枚举值)
// ============================================================================

namespace shipmodes {
    constexpr int8_t REG_AIR = 0;
    constexpr int8_t AIR = 1;
    constexpr int8_t RAIL = 2;
    constexpr int8_t SHIP = 3;
    constexpr int8_t TRUCK = 4;
    constexpr int8_t MAIL = 5;
    constexpr int8_t FOB = 6;
}

// ============================================================================
// 订单优先级编码 (o_orderpriority 枚举值)
// ============================================================================

namespace priorities {
    constexpr int8_t URGENT = 0;       // "1-URGENT"
    constexpr int8_t HIGH = 1;         // "2-HIGH"
    constexpr int8_t MEDIUM = 2;       // "3-MEDIUM"
    constexpr int8_t NOT_SPECIFIED = 3; // "4-NOT SPECIFIED"
    constexpr int8_t LOW = 4;          // "5-LOW"
}

// ============================================================================
// 退货标志编码 (l_returnflag 枚举值)
// ============================================================================

namespace returnflags {
    constexpr int8_t RETURNED = 0;     // 'R'
    constexpr int8_t ACCEPTED = 1;     // 'A'
    constexpr int8_t NONE = 2;         // 'N'
}

// ============================================================================
// 行项目状态编码 (l_linestatus 枚举值)
// ============================================================================

namespace linestatuses {
    constexpr int8_t OPEN = 0;         // 'O'
    constexpr int8_t FILLED = 1;       // 'F'
}

// ============================================================================
// 定点数常量 (TPC-H 使用 4 位小数定点数)
// ============================================================================

namespace fixedpoint {
    constexpr int64_t SCALE = 10000;   // 定点数缩放因子

    // 折扣范围 (Q6: discount between 0.05 and 0.07)
    constexpr int64_t DISCOUNT_05 = 500;   // 0.05 * 10000
    constexpr int64_t DISCOUNT_06 = 600;   // 0.06 * 10000
    constexpr int64_t DISCOUNT_07 = 700;   // 0.07 * 10000

    // 数量阈值 (Q6: quantity < 24)
    constexpr int64_t QTY_24 = 240000;     // 24 * 10000

    // Q19 数量范围
    constexpr int64_t QTY_1 = 10000;       // 1 * 10000
    constexpr int64_t QTY_10 = 100000;     // 10 * 10000
    constexpr int64_t QTY_11 = 110000;     // 11 * 10000
    constexpr int64_t QTY_20 = 200000;     // 20 * 10000
    constexpr int64_t QTY_21 = 210000;     // 21 * 10000
    constexpr int64_t QTY_30 = 300000;     // 30 * 10000

    // Q18 聚合数量阈值 (sum(l_quantity) > 300)
    constexpr int64_t QTY_THRESHOLD_300 = 3000000;  // 300 * 10000
}

// ============================================================================
// TPC-H 查询默认参数
// ============================================================================

namespace query_params {

    // Q1: 定价汇总报告
    namespace q1 {
        constexpr int32_t SHIP_DATE_THRESHOLD = dates::D1998_09_02;  // shipdate <= 1998-09-02
    }

    // Q3: 运输优先级
    namespace q3 {
        constexpr int32_t ORDER_DATE_THRESHOLD = dates::D1995_03_15; // orderdate < 1995-03-15
        constexpr const char* MARKET_SEGMENT = "BUILDING";
    }

    // Q4: 订单优先级检查
    namespace q4 {
        constexpr int32_t DATE_LO = dates::D1993_07_01;  // 1993-07-01
        constexpr int32_t DATE_HI = dates::D1993_10_01;  // 1993-10-01
    }

    // Q5: 本地供应商收入
    namespace q5 {
        constexpr const char* REGION = regions::ASIA;
        constexpr int32_t DATE_LO = dates::D1994_01_01;
        constexpr int32_t DATE_HI = dates::D1995_01_01;
    }

    // Q6: 预测收入变化
    namespace q6 {
        constexpr int32_t DATE_LO = dates::D1994_01_01;
        constexpr int32_t DATE_HI = dates::D1995_01_01;
        constexpr int64_t DISCOUNT_LO = fixedpoint::DISCOUNT_05;
        constexpr int64_t DISCOUNT_HI = fixedpoint::DISCOUNT_07;
        constexpr int64_t QUANTITY_THRESHOLD = fixedpoint::QTY_24;
    }

    // Q7: 国家间运输量
    namespace q7 {
        constexpr const char* NATION1 = nations::FRANCE;
        constexpr const char* NATION2 = nations::GERMANY;
        constexpr int32_t DATE_LO = dates::D1995_01_01;
        constexpr int32_t DATE_HI = dates::D1996_12_31;
    }

    // Q8: 市场份额
    namespace q8 {
        constexpr const char* NATION = nations::BRAZIL;
        constexpr const char* REGION = regions::AMERICA;
        constexpr int32_t DATE_LO = dates::D1995_01_01;
        constexpr int32_t DATE_HI = dates::D1996_12_31;
        constexpr const char* PART_TYPE = "ECONOMY ANODIZED STEEL";
    }

    // Q10: 退货客户
    namespace q10 {
        constexpr int32_t DATE_LO = dates::D1993_10_01;
        constexpr int32_t DATE_HI = dates::D1994_01_01;
    }

    // Q12: 运输方式统计
    namespace q12 {
        constexpr int8_t SHIPMODE1 = shipmodes::MAIL;
        constexpr int8_t SHIPMODE2 = shipmodes::SHIP;
        constexpr int32_t DATE_LO = dates::D1994_01_01;
        constexpr int32_t DATE_HI = dates::D1995_01_01;
    }

    // Q14: 促销效果
    namespace q14 {
        constexpr int32_t DATE_LO = dates::D1995_09_01;
        constexpr int32_t DATE_HI = dates::D1995_10_01;
    }

    // Q15: 顶级供应商
    namespace q15 {
        constexpr int32_t DATE_LO = dates::D1996_01_01;
        constexpr int32_t DATE_HI = dates::D1996_04_01;
    }

    // Q17: 小订单收入
    namespace q17 {
        constexpr const char* BRAND = "Brand#23";
        constexpr const char* CONTAINER = "MED BOX";
    }

    // Q18: 大订单客户
    namespace q18 {
        constexpr int64_t QTY_THRESHOLD = fixedpoint::QTY_THRESHOLD_300;
        constexpr size_t RESULT_LIMIT = 100;
    }

    // Q20: 潜在零件促销
    namespace q20 {
        constexpr const char* COLOR_PREFIX = "forest";
        constexpr const char* NATION = nations::CANADA;
        constexpr int32_t DATE_LO = dates::D1994_01_01;
        constexpr int32_t DATE_HI = dates::D1995_01_01;
    }

    // Q21: 延迟供应商
    namespace q21 {
        constexpr const char* NATION = nations::SAUDI_ARABIA;
    }

    // Q22: 全球销售机会
    namespace q22 {
        // 国家电话前缀
        constexpr const char* COUNTRY_CODES[] = {"13", "31", "23", "29", "30", "18", "17"};
        constexpr size_t NUM_COUNTRY_CODES = 7;
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 将日期字符串转换为 epoch days
 * @param year 年份 (如 1994)
 * @param month 月份 (1-12)
 * @param day 日期 (1-31)
 * @return epoch days
 */
constexpr int32_t make_date(int year, int month, int day) {
    // 简化计算: 从 1970 年开始
    int32_t days = 0;
    for (int y = 1970; y < year; ++y) {
        bool leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
        days += leap ? 366 : 365;
    }
    constexpr int month_days[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (int m = 1; m < month; ++m) {
        days += month_days[m];
        if (m == 2) {
            bool leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if (leap) days += 1;
        }
    }
    days += day - 1;
    return days;
}

}  // namespace constants
}  // namespace tpch
}  // namespace thunderduck

#endif  // TPCH_CONSTANTS_H
