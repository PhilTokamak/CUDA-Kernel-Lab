#include "bench_utils.hpp"
#include "cuda_utils.cuh"
#include "transpose.hpp"

#include <cuda/cmath>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>

void copy_cpu_baseline(const float* A, float* B, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        B[i] = A[i];
    }
}

__global__ void copy_cuda_baseline(const float* A, float* B, size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        B[idx] = A[idx];
    }
}