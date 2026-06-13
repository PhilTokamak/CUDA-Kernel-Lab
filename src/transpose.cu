#include "cuda_utils.cuh"
#include "transpose.hpp"
#include <cuda/cmath>
#include <omp.h>

void transpose_cpu(const float* mat, float* matT, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            matT[j * rows + i] = mat[i * cols + j];
        }
    }
}

void transpose_cpu_omp(const float* mat, float* matT, size_t rows, size_t cols)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            matT[j * rows + i] = mat[i * cols + j];
        }
    }
}

__global__ void transpose_naive_kernel(const float* mat, float* matT, size_t rows, size_t cols)
{
    size_t col = threadIdx.x + blockIdx.x * blockDim.x;
    size_t row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < rows && col < cols)
    {
        matT[col * rows + row] = mat[row * cols + col];
    }
}

/**
 * @brief Launch naive transpose kernel
 *
 * @param[in] in_dev            Input matrix to transpose
 * @param[out] matT_dev         Output matrix on device
 * @param[in] rows              num of rows of input matrix
 * @param[in] cols              num of cols of input matrix
 */
void launch_transpose_naive(const float* mat_dev, float* matT_dev, size_t rows, size_t cols)
{
    dim3 block(16, 16);

    dim3 grid(cuda::ceil_div(rows, block.x), cuda::ceil_div(cols, block.y));

    transpose_naive_kernel<<<grid, block>>>(mat_dev, matT_dev, rows, cols);

    gpu::cuda_check_last();
}