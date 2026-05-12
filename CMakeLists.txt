cmake_minimum_required(VERSION 3.20)
project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(fmt REQUIRED)

# =====================================
# Library
# =====================================

add_library(
    cuda_kernels
    src/vector_add.cu
)

target_include_directories(
    cuda_kernels
    PUBLIC
    include)

# =====================================
# Benchmark executable
# =====================================

add_executable(
    bench_vector_add
    benchmarks/bench_vector_add.cu
)

target_link_libraries(
    bench_vector_add
    PRIVATE
    cuda_kernels
)

