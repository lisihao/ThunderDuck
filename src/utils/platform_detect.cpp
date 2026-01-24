/**
 * ThunderDuck - Platform Detection
 * 
 * 检测 Apple Silicon M4 芯片特性
 */

#include "thunderduck/thunderduck.h"
#include <sys/sysctl.h>
#include <string>
#include <cstring>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

namespace thunderduck {

namespace {

// 获取 sysctl 字符串值
std::string get_sysctl_string(const char* name) {
    char buffer[256];
    size_t size = sizeof(buffer);
    if (sysctlbyname(name, buffer, &size, nullptr, 0) == 0) {
        return std::string(buffer);
    }
    return "";
}

// 获取 sysctl 整数值
int64_t get_sysctl_int(const char* name) {
    int64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) {
        return value;
    }
    return -1;
}

} // anonymous namespace

/**
 * 检测是否为 Apple M4 芯片
 */
bool is_m4_available() {
#if defined(__APPLE__) && defined(__aarch64__)
    std::string brand = get_sysctl_string("machdep.cpu.brand_string");
    // 检查是否包含 M4 标识
    if (brand.find("M4") != std::string::npos) {
        return true;
    }
    // 也检查 Apple Silicon 通用标识
    if (brand.find("Apple") != std::string::npos) {
        // M4 的 CPU family 检测
        int64_t cpu_family = get_sysctl_int("hw.cpufamily");
        // M4 family ID (需要根据实际值更新)
        // 暂时返回 true 对于所有 Apple Silicon
        return true;
    }
    return false;
#else
    return false;
#endif
}

/**
 * 检测 ARM Neon SIMD 是否可用
 */
bool is_neon_available() {
#if defined(__aarch64__) || defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

/**
 * 检测 Apple Neural Engine 是否可用
 */
bool is_npu_available() {
#if defined(__APPLE__) && defined(__aarch64__)
    // 检查是否有 ANE (Apple Neural Engine)
    // M1 及以后的芯片都有 NPU
    std::string brand = get_sysctl_string("machdep.cpu.brand_string");
    if (brand.find("Apple") != std::string::npos) {
        return true;
    }
    return false;
#else
    return false;
#endif
}

/**
 * 获取 CPU 核心数
 */
int get_cpu_cores() {
    int64_t cores = get_sysctl_int("hw.ncpu");
    return cores > 0 ? static_cast<int>(cores) : 1;
}

/**
 * 获取性能核心数
 */
int get_performance_cores() {
    int64_t pcores = get_sysctl_int("hw.perflevel0.physicalcpu");
    return pcores > 0 ? static_cast<int>(pcores) : get_cpu_cores() / 2;
}

/**
 * 获取 L1 数据缓存大小
 */
size_t get_l1_cache_size() {
    int64_t size = get_sysctl_int("hw.l1dcachesize");
    return size > 0 ? static_cast<size_t>(size) : 64 * 1024; // 默认 64KB
}

/**
 * 获取 L2 缓存大小
 */
size_t get_l2_cache_size() {
    int64_t size = get_sysctl_int("hw.l2cachesize");
    return size > 0 ? static_cast<size_t>(size) : 4 * 1024 * 1024; // 默认 4MB
}

/**
 * 获取缓存行大小
 */
size_t get_cache_line_size() {
    int64_t size = get_sysctl_int("hw.cachelinesize");
    // M4 使用 128 字节缓存行
    return size > 0 ? static_cast<size_t>(size) : 128;
}

} // namespace thunderduck
