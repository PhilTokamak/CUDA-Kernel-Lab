#include "vector_add.hpp"
#include "cuda_utils.cuh"
#include <cuda/cmath>

void vector_add_cpu(const float* a, const float* b, float* c, size_t num_elem)
{
    for(size_t i = 0; i < num_elem; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t num_elem)
{
    size_t idx = static_cast<size_t>(threadIdx.x) + static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x);
    if(idx < num_elem)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add(const float* a_dev, const float* b_dev, float* c_dev, size_t num_elem)
{
    constexpr int threads = 256;
    int blocks = static_cast<int>(cuda::ceil_div(num_elem, static_cast<size_t>(threads)));

    vector_add_kernel<<<blocks, threads>>>(a_dev, b_dev, c_dev, num_elem);

    gpu::cuda_check_last();
}