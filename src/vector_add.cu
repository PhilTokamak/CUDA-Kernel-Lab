#include "vector_add.hpp"

#include "cuda_utils.cuh"

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