#include "cuda_utils.cuh"
#include "reduction.hpp"
#include <bit>
#include <cuda/cmath>
#include <omp.h>

void reduce_cpu(const float* x, float& out, size_t n)
{
    out = float{};
    for (size_t i = 0; i < n; ++i)
    {
        out += x[i];
    }
}

void reduce_cpu_omp(const float* x, float& out, size_t n)
{
    // create a double res variable to avoid that the largest continuous integer fp32 can represent
    // is 1 << 24
    double res = float{};
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; ++i)
    {
        res += x[i];
    }

    out = res;
}

void reduce_cpu_fp64_ref(const float* x, double& out, size_t n)
{
    out = double{};
    for (size_t i = 0; i < n; ++i)
    {
        out += static_cast<double>(x[i]);
    }
}

/**
 * Atomic Add version
 */
__global__ void reduce_atomic_kernel(const float* input, float* out, size_t n)
{
    size_t idx = static_cast<size_t>(threadIdx.x) +
                 static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);
    if (idx < n)
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
    if (idx < n)
    {
        val = input[idx];
    }

    local_array[tid] = val;
    // Synchronize all threads in the block
    __syncthreads();

    // block reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            local_array[tid] += local_array[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        partial_sum_out[blockIdx.x] = local_array[0];
    }
}

/**
 * @brief Launch block_share memory reduction kernel
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

/**
 * Grid-stride loop version
 */
__global__ void reduce_grid_stride_block_kernel(const float* input, float* partial_sum_out,
                                                size_t n)
{
    extern __shared__ float local_array[];

    size_t tid = static_cast<size_t>(threadIdx.x);

    size_t idx = static_cast<size_t>(threadIdx.x) +
                 static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x);

    float val{};
    size_t grid_stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);

    // Grid-stride loop
    for (size_t i = idx; i < n; i += grid_stride)
    {
        val += input[i];
    }

    // Put partial sum of each thread into local array and synchronize block threads
    local_array[tid] = val;
    // Synchronize all threads in the block
    __syncthreads();

    // block reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            local_array[tid] += local_array[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        partial_sum_out[blockIdx.x] = local_array[0];
    }
}

/**
 * @brief Launch grid-stride-loop block-shared memory reduction kernel
 *
 * @param[in] x_dev            Input array to reduce
 * @param[out] partial_sum_dev  Partial sums from GPU kernel
 * @param[in] n                Length of input array
 */
void launch_reduce_grid_stride_block(const float* x_dev, float* partial_sum_dev, size_t n,
                                     int block_size, int grid_size)
{
    // int blocks = cuda::ceil_div(n, block_size);
    // Calculate shared bytes for specifying the size of block-shared memory
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    reduce_grid_stride_block_kernel<<<grid_size, block_size, shared_bytes>>>(x_dev, partial_sum_dev,
                                                                             n);

    gpu::cuda_check_last();
}

/**
 * Grid-stride two-pass version
 */
__global__ void reduce_grid_stride_two_pass_kernel(const float* input_partial_sum, float* out,
                                                   size_t n)
{
    extern __shared__ float local_array[];

    unsigned int tid = threadIdx.x;

    // Put all elements of partial sum into local array in one block and synchronize block threads
    if (tid < n)
    {
        local_array[tid] = input_partial_sum[tid];
    }
    else
    {
        local_array[tid] = 0.0f;
    }

    // Synchronize all threads in the block
    __syncthreads();

    // block reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            local_array[tid] += local_array[tid + stride];
        }
        __syncthreads();
    }

    // Result is now in local_array[0]
    if (tid == 0)
    {
        out[0] = local_array[0];
    }
}

/**
 * @brief Launch grid-stride two pass reduction kernel
 *
 * @param[in] x_dev            Input array to reduce
 * @param[in] partial_sum_dev  Allocated memory for partial sum array from GPU kernel
 * @param[out] out_dev         Final sum from GPU kernel
 * @param[in] n                Length of input array
 */
void launch_reduce_grid_stride_two_pass(const float* x_dev, float* partial_sum_dev, float* out_dev,
                                        size_t n, int block_size, int grid_size)
{
    // int blocks = cuda::ceil_div(n, block_size);
    // Calculate shared bytes for specifying the size of block-shared memory
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    reduce_grid_stride_block_kernel<<<grid_size, block_size, shared_bytes>>>(x_dev, partial_sum_dev,
                                                                             n);

    gpu::cuda_check_last();

    size_t second_pass_block_size     = std::bit_ceil(static_cast<size_t>(grid_size));
    size_t shared_bytes_second_pass   = static_cast<size_t>(second_pass_block_size) * sizeof(float);
    size_t num_elem_input_second_pass = grid_size;


    // In CUDA, the absolute maximum number of threads in a single block is 1024. This limit is
    // enforced by the hardware.
    assert(second_pass_block_size <= 1024);

    reduce_grid_stride_two_pass_kernel<<<1, second_pass_block_size, shared_bytes_second_pass>>>(
        partial_sum_dev, out_dev, num_elem_input_second_pass);

    gpu::cuda_check_last();
}

/**
 * @brief Launch multi-pass reduction kernel
 *
 * @param[in] x_dev            Input array to reduce
 * @param[in] partial_sum_dev  Allocated memory for partial sum array from GPU kernel
 * @param[in] scratch_dev      Allocated memory for multi pass in GPU
 * @param[out] out_dev         Final sum from GPU kernel
 * @param[in] n                Length of input array
 */
void launch_reduce_multi_pass(const float* x_dev, float* partial_sum_dev, float* scratch_dev,
                              float* out_dev, size_t n, int block_size, int grid_size)
{
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    // pass 1 - original array might be very large, use grid-stride version to reduce number of
    // threads
    reduce_grid_stride_block_kernel<<<grid_size, block_size, shared_bytes>>>(x_dev, partial_sum_dev,
                                                                             n);

    gpu::cuda_check_last();

    size_t current_n = static_cast<size_t>(grid_size);

    float* current_in  = partial_sum_dev;
    float* current_out = scratch_dev;

    while (current_n > 1)
    {
        // pass 2, 3, ...
        size_t pass_block_size =
            1024; // Absolute maximum number of threads in a single block is 1024
        if (current_n < pass_block_size)
        {
            pass_block_size = std::bit_ceil(current_n);
        }

        int pass_grid_size = static_cast<int>(cuda::ceil_div(current_n, pass_block_size));

        // Unlike for grid stride reduction, now the shared array length must not be smaller than
        // block_size, since the whole partial sum array is handled by one block
        size_t pass_shared_bytes = pass_block_size * sizeof(float);

        reduce_block_kernel<<<pass_grid_size, static_cast<int>(pass_block_size),
                              pass_shared_bytes>>>(current_in, current_out, current_n);

        gpu::cuda_check_last();

        current_n = static_cast<size_t>(pass_grid_size);

        std::swap(current_in, current_out);
    }

    gpu::cuda_check(cudaMemcpy(out_dev, current_in, sizeof(float), cudaMemcpyDeviceToDevice));
}