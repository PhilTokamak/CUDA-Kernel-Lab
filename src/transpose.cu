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