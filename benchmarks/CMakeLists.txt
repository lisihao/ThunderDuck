# ThunderDuck Benchmarks

add_executable(bench_filter bench_filter.cpp)
target_link_libraries(bench_filter thunderduck)

add_executable(bench_aggregate bench_aggregate.cpp)
target_link_libraries(bench_aggregate thunderduck)
