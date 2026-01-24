/**
 * ThunderDuck - Main Engine
 * 
 * ThunderDuck 主入口，负责初始化和管理优化后端
 */

#include "thunderduck/thunderduck.h"
#include "thunderduck/memory.h"
#include <iostream>
#include <atomic>

namespace thunderduck {

namespace {

std::atomic<bool> g_initialized{false};
bool g_m4_available = false;
bool g_neon_available = false;
bool g_npu_available = false;

} // anonymous namespace

// 外部声明（在 platform_detect.cpp 中实现）
int get_cpu_cores();
int get_performance_cores();
size_t get_l1_cache_size();
size_t get_l2_cache_size();
size_t get_cache_line_size();

/**
 * 初始化 ThunderDuck
 */
bool initialize() {
    if (g_initialized.exchange(true)) {
        // 已经初始化
        return true;
    }

    // 检测硬件特性
    g_m4_available = is_m4_available();
    g_neon_available = is_neon_available();
    g_npu_available = is_npu_available();

    // 打印初始化信息
    std::cout << "ThunderDuck v" << VERSION_MAJOR << "." 
              << VERSION_MINOR << "." << VERSION_PATCH << " initialized\n";
    std::cout << "  Platform: Apple Silicon "
              << (g_m4_available ? "M4" : "(generic)") << "\n";
    std::cout << "  SIMD (Neon): " << (g_neon_available ? "enabled" : "disabled") << "\n";
    std::cout << "  NPU: " << (g_npu_available ? "available" : "not available") << "\n";
    std::cout << "  CPU Cores: " << get_cpu_cores() 
              << " (P-cores: " << get_performance_cores() << ")\n";
    std::cout << "  L1 Cache: " << (get_l1_cache_size() / 1024) << " KB\n";
    std::cout << "  L2 Cache: " << (get_l2_cache_size() / 1024 / 1024) << " MB\n";
    std::cout << "  Cache Line: " << get_cache_line_size() << " bytes\n";

    return true;
}

/**
 * 关闭 ThunderDuck
 */
void shutdown() {
    if (!g_initialized.exchange(false)) {
        return;
    }
    std::cout << "ThunderDuck shutdown\n";
}

/**
 * 获取版本字符串
 */
const char* get_version() {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d", 
             VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return version;
}

/**
 * 检查是否已初始化
 */
bool is_initialized() {
    return g_initialized.load();
}

} // namespace thunderduck
