/**
 * 显示 ThunderDuck 系统表数据
 */

#include "../include/thunderduck/system_catalog.h"
#include "tpch/tpch_query_optimizer.h"
#include <iostream>

using namespace thunderduck;

int main() {
    // 初始化优化器 (会注册所有算子到系统表)
    tpch::register_tpch_query_configs();

    // 获取系统表实例
    auto& cat = catalog::catalog();

    // 设置持久化路径
    cat.set_persistence_path(".solar/system_catalog.json");

    // 显示所有数据
    cat.print_all();

    // 保存到文件
    if (cat.save_to_file(".solar/system_catalog.json")) {
        std::cout << "✓ System catalog saved to .solar/system_catalog.json\n";
    } else {
        std::cout << "✗ Failed to save system catalog\n";
    }

    return 0;
}
