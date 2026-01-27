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
