/**
 * ThunderDuck TPC-H 报告生成器
 *
 * 生成 Markdown 格式的性能报告
 *
 * @version 1.0
 * @date 2026-01-28
 */

#ifndef TPCH_REPORT_H
#define TPCH_REPORT_H

#include <string>
#include "tpch_executor.h"

namespace thunderduck {
namespace tpch {

/**
 * 生成 TPC-H 基准测试报告
 *
 * @param result 测试结果
 * @param filename 输出文件路径
 * @return 是否成功
 */
bool generate_report(const BenchmarkResult& result, const std::string& filename);

/**
 * 打印汇总到控制台
 */
void print_summary(const BenchmarkResult& result);

/**
 * 生成报告内容 (不写入文件)
 */
std::string generate_report_content(const BenchmarkResult& result);

} // namespace tpch
} // namespace thunderduck

#endif // TPCH_REPORT_H
