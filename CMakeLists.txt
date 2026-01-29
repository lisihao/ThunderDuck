cmake_minimum_required(VERSION 3.20)
project(ThunderDuck VERSION 1.0.0 LANGUAGES C CXX ASM OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Apple Silicon 优化选项
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(STATUS "Building for Apple Silicon")
    # 使用通用 Apple Silicon 优化，让编译器自动检测
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=native -O3")
    add_definitions(-DTHUNDERDUCK_ARM64)
    add_definitions(-DTHUNDERDUCK_NEON)

    # 启用 CRC32 扩展（用于哈希）
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crc")
endif()

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp")
file(GLOB_RECURSE OPERATOR_SOURCES "src/operators/**/*.cpp")
file(GLOB_RECURSE NPU_SOURCES "src/npu/*.cpp")
file(GLOB_RECURSE NPU_SOURCES_MM "src/npu/*.mm")  # V19.5: NPU Objective-C++ files
file(GLOB_RECURSE UTIL_SOURCES "src/utils/*.cpp")
file(GLOB_RECURSE ASM_SOURCES "src/operators/**/*.S")

# GPU 源文件 (Metal/Objective-C++)
file(GLOB_RECURSE GPU_SOURCES "src/gpu/*.mm")

# 主库
add_library(thunderduck STATIC
    ${CORE_SOURCES}
    ${OPERATOR_SOURCES}
    ${NPU_SOURCES}
    ${NPU_SOURCES_MM}
    ${UTIL_SOURCES}
    ${ASM_SOURCES}
    ${GPU_SOURCES}
)

target_include_directories(thunderduck PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Apple 框架链接 (Metal, Accelerate, MPS)
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

    if(METAL_FRAMEWORK)
        message(STATUS "Found Metal framework")
        add_definitions(-DTHUNDERDUCK_HAS_METAL)
        target_link_libraries(thunderduck ${METAL_FRAMEWORK})
    endif()

    if(FOUNDATION_FRAMEWORK)
        target_link_libraries(thunderduck ${FOUNDATION_FRAMEWORK})
    endif()

    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Found Accelerate framework (BNNS)")
        add_definitions(-DTHUNDERDUCK_HAS_BNNS)
        target_link_libraries(thunderduck ${ACCELERATE_FRAMEWORK})
    endif()

    if(MPS_FRAMEWORK)
        message(STATUS "Found Metal Performance Shaders framework")
        add_definitions(-DTHUNDERDUCK_HAS_MPS)
        target_link_libraries(thunderduck ${MPS_FRAMEWORK})
    endif()

    if(MPSGRAPH_FRAMEWORK)
        message(STATUS "Found Metal Performance Shaders Graph framework (V19.5)")
        add_definitions(-DTHUNDERDUCK_HAS_MPSGRAPH)
        target_link_libraries(thunderduck ${MPSGRAPH_FRAMEWORK})
    endif()
endif()

# 测试
enable_testing()
add_subdirectory(tests)

# 基准测试
add_subdirectory(benchmarks)
