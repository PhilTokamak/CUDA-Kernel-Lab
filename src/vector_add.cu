#include "vector_add.hpp"
#include "cuda_utils.cuh"
#include <cuda/cmath>

void vector_add_cpu(const float* a, const float* b, float* c, int num_elem)
{
    for(int i = 0; i < num_elem; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int num_elem)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < num_elem)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add(const float* a_dev, const float* b_dev, float* c_dev, int num_elem)
{
    constexpr int threads = 256;
    int blocks = cuda::ceil_div(num_elem, threads);

    vector_add_kernel<<<blocks, threads>>>(a_dev, b_dev, c_dev, num_elem);

    gpu::cuda_check_last();
}