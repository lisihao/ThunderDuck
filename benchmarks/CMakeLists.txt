# ThunderDuck Benchmarks

add_executable(bench_filter bench_filter.cpp)
target_link_libraries(bench_filter thunderduck)

add_executable(bench_aggregate bench_aggregate.cpp)
target_link_libraries(bench_aggregate thunderduck)

# TopK v4 benchmark
add_executable(bench_topk_v4 ${CMAKE_SOURCE_DIR}/benchmark/topk_v4_benchmark.cpp)
target_link_libraries(bench_topk_v4 thunderduck)

# Comprehensive benchmark
add_executable(bench_comprehensive ${CMAKE_SOURCE_DIR}/benchmark/comprehensive_benchmark.cpp)
target_link_libraries(bench_comprehensive thunderduck)

# V9.2 Group Aggregate benchmark
add_executable(bench_group_agg_v92 ${CMAKE_SOURCE_DIR}/benchmark/test_group_aggregate_v92.cpp)
target_link_libraries(bench_group_agg_v92 thunderduck)

# V9.3 Intelligent Strategy benchmark
add_executable(bench_group_agg_v93 ${CMAKE_SOURCE_DIR}/benchmark/test_group_aggregate_v93.cpp)
target_link_libraries(bench_group_agg_v93 thunderduck)

# V14 Complete benchmark
add_executable(v14_complete_benchmark ${CMAKE_SOURCE_DIR}/benchmark/v14_complete_benchmark.cpp)
target_link_libraries(v14_complete_benchmark thunderduck duckdb)
target_include_directories(v14_complete_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/third_party/duckdb)
set_target_properties(v14_complete_benchmark PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
)

# V18 benchmark (性能基线 V2)
add_executable(v18_benchmark ${CMAKE_SOURCE_DIR}/benchmark/v18_benchmark.cpp)
target_link_libraries(v18_benchmark thunderduck duckdb)
target_include_directories(v18_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/third_party/duckdb)
set_target_properties(v18_benchmark PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
)

# V19.2 benchmark (激进预取 + SIMD 槽位比较 + 128 元素批次)
add_executable(v19_2_benchmark ${CMAKE_SOURCE_DIR}/benchmark/v19_2_benchmark.cpp)
target_link_libraries(v19_2_benchmark thunderduck duckdb)
target_include_directories(v19_2_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/third_party/duckdb)
set_target_properties(v19_2_benchmark PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
)

# TPC-H 完整基准测试系统
set(TPCH_SOURCES
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_benchmark_main.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_data_loader.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_executor.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v24.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v25.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v26.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v27.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v32.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_config_v33.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v33.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v34.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v35.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v36.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v37.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v38.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v39.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v40.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v41.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v42.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v43.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v44.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v45.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v46.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v47.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_operators_v48.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_queries.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/tpch/tpch_report.cpp
)
add_executable(tpch_benchmark ${TPCH_SOURCES})
target_link_libraries(tpch_benchmark thunderduck duckdb)
target_include_directories(tpch_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/duckdb
    ${CMAKE_SOURCE_DIR}/benchmark/tpch
    ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(tpch_benchmark PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/third_party/duckdb"
)
# 启用 C++17 (用于 structured bindings)
target_compile_features(tpch_benchmark PRIVATE cxx_std_17)
