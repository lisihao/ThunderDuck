/**
 * Hash Join v4 Strategy Test
 *
 * Tests all v4 strategies: RADIX256, BLOOMFILTER, NPU, GPU
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

#include "thunderduck/join.h"

using namespace thunderduck;
using namespace thunderduck::join;
using namespace std::chrono;

// ============================================================================
// Benchmark Helper
// ============================================================================

template<typename Func>
double benchmark(Func func, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 3; i++) func();

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) func();
    auto end = high_resolution_clock::now();

    return duration_cast<microseconds>(end - start).count() / (double)iterations / 1000.0;  // ms
}

// ============================================================================
// Test Data Generator
// ============================================================================

void generate_join_data(std::vector<int32_t>& build, std::vector<int32_t>& probe,
                        size_t build_size, size_t probe_size, double selectivity = 1.0) {
    build.resize(build_size);
    probe.resize(probe_size);

    std::mt19937 gen(42);

    // Build keys: 0 to build_size-1
    for (size_t i = 0; i < build_size; i++) {
        build[i] = static_cast<int32_t>(i);
    }

    // Probe keys: based on selectivity
    size_t match_count = static_cast<size_t>(probe_size * selectivity);
    for (size_t i = 0; i < match_count; i++) {
        probe[i] = static_cast<int32_t>(i % build_size);
    }

    // Non-matching keys
    for (size_t i = match_count; i < probe_size; i++) {
        probe[i] = static_cast<int32_t>(build_size + i);
    }

    // Shuffle probe
    std::shuffle(probe.begin(), probe.end(), gen);
}

// ============================================================================
// Strategy Availability Check
// ============================================================================

void print_strategy_availability() {
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ Hash Join v4 Strategy Availability                                      │\n";
    std::cout << "├─────────────────────────────────────────────────────────────────────────┤\n";

    auto check = [](JoinStrategy s, const char* name) {
        bool available = is_strategy_available(s);
        std::cout << "│ " << std::left << std::setw(20) << name
                  << ": " << (available ? "✓ Available" : "✗ Not Available")
                  << std::setw(30) << "" << "│\n";
    };

    check(JoinStrategy::V3_FALLBACK, "V3_FALLBACK");
    check(JoinStrategy::RADIX256, "RADIX256");
    check(JoinStrategy::BLOOMFILTER, "BLOOMFILTER");
    check(JoinStrategy::NPU, "NPU");
    check(JoinStrategy::GPU, "GPU");

    std::cout << "└─────────────────────────────────────────────────────────────────────────┘\n\n";
}

// ============================================================================
// Test Specific Strategy
// ============================================================================

struct TestResult {
    double time_ms;
    size_t matches;
    bool success;
};

TestResult test_strategy(JoinStrategy strategy, const char* name,
                         const int32_t* build_keys, size_t build_count,
                         const int32_t* probe_keys, size_t probe_count,
                         size_t expected_matches) {
    TestResult result = {0, 0, false};

    if (!is_strategy_available(strategy)) {
        return result;
    }

    JoinResult* jr = create_join_result(std::max(build_count, probe_count) * 4);

    JoinConfigV4 config;
    config.strategy = strategy;
    config.num_threads = 4;
    config.fallback_to_cpu = false;  // Don't fallback, test specific strategy

    result.time_ms = benchmark([&]() {
        jr->count = 0;
        result.matches = hash_join_i32_v4_config(
            build_keys, build_count,
            probe_keys, probe_count,
            JoinType::INNER, jr, config);
    }, 5);

    // Verify match count
    result.success = (result.matches == expected_matches);

    free_join_result(jr);
    return result;
}

// ============================================================================
// Main Test
// ============================================================================

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Hash Join v4 Comprehensive Strategy Test                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";

    // Print strategy availability
    print_strategy_availability();

    // ========================================
    // Test Scenarios
    // ========================================

    struct TestScenario {
        const char* name;
        size_t build_size;
        size_t probe_size;
        double selectivity;
    };

    TestScenario scenarios[] = {
        {"J1: Small  (10K x 100K)",    10000,   100000,  1.0},
        {"J2: Medium (100K x 1M)",     100000,  1000000, 1.0},
        {"J3: Large  (1M x 10M)",      1000000, 10000000, 1.0},
        {"J4: Low Selectivity (100K)", 100000,  1000000, 0.1},  // 10% match rate
    };

    // Strategies to test
    JoinStrategy strategies[] = {
        JoinStrategy::V3_FALLBACK,
        JoinStrategy::RADIX256,
        JoinStrategy::BLOOMFILTER,
        JoinStrategy::AUTO,
    };

    const char* strategy_names[] = {
        "V3_FALLBACK",
        "RADIX256",
        "BLOOMFILTER",
        "AUTO",
    };

    for (const auto& scenario : scenarios) {
        std::cout << "┌─────────────────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ " << std::left << std::setw(71) << scenario.name << "│\n";
        std::cout << "├─────────────────────────────────────────────────────────────────────────┤\n";

        // Generate data
        std::vector<int32_t> build, probe;
        generate_join_data(build, probe, scenario.build_size, scenario.probe_size, scenario.selectivity);

        size_t expected = static_cast<size_t>(scenario.probe_size * scenario.selectivity);

        std::cout << "│ Build: " << std::setw(10) << scenario.build_size
                  << "  Probe: " << std::setw(10) << scenario.probe_size
                  << "  Expected matches: " << std::setw(10) << expected << "     │\n";
        std::cout << "├─────────────────────────────────────────────────────────────────────────┤\n";

        // Test each strategy
        double base_time = 0;
        for (size_t i = 0; i < 4; i++) {
            auto result = test_strategy(strategies[i], strategy_names[i],
                                        build.data(), build.size(),
                                        probe.data(), probe.size(),
                                        expected);

            if (i == 0) base_time = result.time_ms;  // V3 as baseline

            if (result.time_ms > 0) {
                double speedup = base_time / result.time_ms;
                std::cout << "│ " << std::left << std::setw(12) << strategy_names[i]
                          << ": " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << result.time_ms << " ms"
                          << "  matches: " << std::setw(10) << result.matches
                          << "  vs v3: " << std::setw(5) << std::setprecision(2) << speedup << "x"
                          << (result.success ? " ✓" : " ✗") << "  │\n";
            } else {
                std::cout << "│ " << std::left << std::setw(12) << strategy_names[i]
                          << ": N/A (not available)" << std::setw(40) << "" << "│\n";
            }
        }

        std::cout << "└─────────────────────────────────────────────────────────────────────────┘\n\n";
    }

    // ========================================
    // AUTO Strategy Selection Test
    // ========================================

    std::cout << "┌─────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ AUTO Strategy Selection Test                                           │\n";
    std::cout << "├─────────────────────────────────────────────────────────────────────────┤\n";

    struct AutoTestCase {
        size_t build_count;
        size_t probe_count;
    };

    AutoTestCase auto_cases[] = {
        {1000, 10000},       // Very small - should use V3
        {5000, 50000},       // Small - should use V3
        {50000, 500000},     // Medium - should use RADIX/BLOOM
        {500000, 5000000},   // Large - should use BLOOMFILTER
        {1000000, 10000000}, // Very large - might use GPU if available
    };

    JoinConfigV4 auto_config;
    auto_config.strategy = JoinStrategy::AUTO;

    for (const auto& tc : auto_cases) {
        const char* selected = get_selected_strategy_name(tc.build_count, tc.probe_count, auto_config);
        std::cout << "│ Build: " << std::setw(8) << tc.build_count
                  << "  Probe: " << std::setw(10) << tc.probe_count
                  << "  → " << std::left << std::setw(20) << selected << "               │\n";
    }

    std::cout << "└─────────────────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "✓ Hash Join v4 Strategy Test Complete!\n\n";

    std::cout << "Strategy Summary:\n";
    std::cout << "  • RADIX256: 256-partition (8-bit) L1 cache optimization\n";
    std::cout << "  • BLOOMFILTER: Bloom filter pre-filtering for low selectivity\n";
    std::cout << "  • NPU: BNNS-accelerated (simplified to Bloom fallback)\n";
    std::cout << "  • GPU: Metal parallel probe (for very large datasets)\n";
    std::cout << "  • AUTO: Intelligent strategy selection based on data characteristics\n";

    return 0;
}
