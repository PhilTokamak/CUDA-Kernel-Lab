#include "reduction.hpp"
#include "cuda_utils.cuh"
#include <cuda/cmath>

void reduce_cpu(const float* x, double& out, size_t n)
{
    out = double{};
    for(size_t i = 0; i < n; ++i)
    {
        out += x[i];
    }
}

/**
 * Atomic Add version
 */
__global__ void reduce_atomic_kernel(const float* input, float* out, size_t n)
{
    size_t idx = static_cast<size_t>(threadIdx.x) +
                 static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if(idx < n)
    {
        atomicAdd(out, input[idx]);
    }
}

/**
 * @brief This function launches atomic add version of reduction.
 *
 * @note Before launching, the user need to manually call
 *            `cudaMemset` to the initial value for out_dev
 */
void launch_reduce_atomic(const float* x_dev, float* out_dev, size_t n, int block_size)
{
    int blocks = static_cast<int>(cuda::ceil_div(n, static_cast<size_t>(block_size)));
    reduce_atomic_kernel<<<blocks, block_size>>>(x_dev, out_dev, n);

    gpu::cuda_check_last();
}

/**
 * @brief Block shared memory version
 *
 * @note Each block compute a partial sum and let cpu handle the dummation og partial sums
 */
__global__ void reduce_block_kernel(const float* input, float* partial_sum_out, size_t n)
{
    // `extern __shared__` because is means dynamic-size block-shared memory,
    // the size is dynamically allocated when launchung the kernel by passing in
    // the third param in the triple chavron <<<grid_size, block_size, smem_size_in_bytes>>>
    extern __shared__ float local_array[];

    size_t tid = static_cast<size_t>(threadIdx.x);

    size_t idx = static_cast<size_t>(threadIdx.x) +
                 static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);

    float val{};
    if(idx < n)
    {
        val = input[idx];
    }

    local_array[tid] = val;
    // Synchronize all threads in the block
    __syncthreads();

    for(unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            local_array[tid] += local_array[tid + stride];
        }

        __syncthreads();
    }

    if(tid == 0)
    {
        partial_sum_out[blockIdx.x] = local_array[0];
    }
}

/**
 * @brief Launch atomic-add reduction kernel
 *
 * @param[in] x_dev            Input array to reduce
 * @param[out] partial_sum_dev  Partial sums from GPU kernel
 * @param[in] n                Length of input array
 */
void launch_reduce_block(const float* x_dev, float* partial_sum_dev, size_t n, int block_size)
{
    int blocks = cuda::ceil_div(n, block_size);
    // Calculate shared bytes for specifying the size of block-shared memory
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    reduce_block_kernel<<<blocks, block_size, shared_bytes>>>(x_dev, partial_sum_dev, n);

    gpu::cuda_check_last();
}