# ThunderDuck Tests

add_executable(test_filter test_filter.cpp)
target_link_libraries(test_filter thunderduck)
add_test(NAME test_filter COMMAND test_filter)

add_executable(test_aggregate test_aggregate.cpp)
target_link_libraries(test_aggregate thunderduck)
add_test(NAME test_aggregate COMMAND test_aggregate)

add_executable(test_join test_join.cpp)
target_link_libraries(test_join thunderduck)
add_test(NAME test_join COMMAND test_join)

add_executable(test_sort test_sort.cpp)
target_link_libraries(test_sort thunderduck)
add_test(NAME test_sort COMMAND test_sort)

# V23 Framework Test
add_executable(test_framework framework_test.cpp)
target_include_directories(test_framework PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_framework thunderduck)
add_test(NAME test_framework COMMAND test_framework)
