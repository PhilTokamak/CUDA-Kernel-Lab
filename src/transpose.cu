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

template <unsigned int TILE_DIM, unsigned int BLOCK_ROWS>
__global__ void transpose_tiled_kernel(const float* mat, float* matT, size_t rows, size_t cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Location in mat
    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load mat into tile
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        auto row = y + j;
        auto col = x;

        if (row < rows && col < cols)
        {
            tile[threadIdx.y + j][threadIdx.x] = mat[row * cols + col];
        }
    }

    __syncthreads();

    // Location in matT, note that here block are at tranposed locations
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed tile to matT
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        auto row = y + j;
        auto col = x;

        // matT has shape cols x rows since it is the transpose of A which
        // has the shape rows x cols
        if (row < cols && col < rows)
        {
            matT[row * rows + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/**
 * @brief Launch tiled transpose kernel
 *
 * @param[in] in_dev            Input matrix to transpose
 * @param[out] matT_dev         Output matrix on device
 * @param[in] rows              num of rows of input matrix
 * @param[in] cols              num of cols of input matrix
 *
 */
template <unsigned int TILE_DIM, unsigned int BLOCK_ROWS>
void launch_transpose_tiled_kernel(const float* mat_dev, float* matT_dev, size_t rows, size_t cols,
                                   dim3 block_size, dim3 grid_size)
{
    static_assert(TILE_DIM % BLOCK_ROWS == 0, "TILE_DIM must be divisible by BLOCK_ROWS");

    transpose_tiled_kernel<TILE_DIM, BLOCK_ROWS>
        <<<grid_size, block_size>>>(mat_dev, matT_dev, rows, cols);

    gpu::cuda_check_last();
}