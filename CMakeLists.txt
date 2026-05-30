cmake_minimum_required(VERSION 3.20)

project(cuda_kernel_lab
        LANGUAGES CXX CUDA
)

# =====================================
# Find Packages
# =====================================
find_package(fmt REQUIRED)

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# =====================================
# Library
# =====================================

add_library(
    cuda_kernels
    src/vector_add.cu
    src/reduction.cu
)

target_include_directories(
    cuda_kernels
    PUBLIC
    include)

target_compile_features(
    cuda_kernels
    PUBLIC
    cxx_std_20
    cuda_std_20
)

target_link_libraries(
    cuda_kernels
    PUBLIC
    fmt::fmt
    OpenMP::OpenMP_CXX
)

target_compile_options(
    cuda_kernels
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

# =====================================
# Benchmark executables
# =====================================

# Vector Add
add_executable(
    bench_vector_add
    benchmarks/bench_vector_add.cu
)

target_link_libraries(
    bench_vector_add
    PRIVATE
    cuda_kernels
)

set_target_properties(
    bench_vector_add
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)

# Reduction
add_executable(
    bench_reduction
    benchmarks/bench_reduction.cu
)

target_link_libraries(
    bench_reduction
    PRIVATE
    cuda_kernels
)

set_target_properties(
    bench_reduction
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)



# =====================================
# Ctest
# =====================================

enable_testing()

add_test(
    NAME vector_add_test
    COMMAND bench_vector_add
)

add_test(
    NAME reduction_test
    COMMAND bench_reduction
)

