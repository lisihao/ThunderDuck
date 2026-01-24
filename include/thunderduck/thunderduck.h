#ifndef THUNDERDUCK_H
#define THUNDERDUCK_H

#include <cstdint>
#include <cstddef>

namespace thunderduck {

// 版本信息
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// ============================================================================
// 平台检测
// ============================================================================

bool is_m4_available();
bool is_neon_available();
bool is_npu_available();

// 硬件信息
int get_cpu_cores();
int get_performance_cores();
size_t get_l1_cache_size();
size_t get_l2_cache_size();
size_t get_cache_line_size();

// ============================================================================
// 生命周期管理
// ============================================================================

bool initialize();
void shutdown();
bool is_initialized();
const char* get_version();

// ============================================================================
// 配置选项
// ============================================================================

struct Config {
    bool enable_simd = true;        // 启用 SIMD 优化
    bool enable_npu = true;         // 启用 NPU 加速
    bool enable_prefetch = true;    // 启用内存预取
    int num_threads = 0;            // 线程数（0=自动）
    size_t block_size = 0;          // 块大小（0=自动）
};

void set_config(const Config& config);
Config get_config();

} // namespace thunderduck

#endif // THUNDERDUCK_H
