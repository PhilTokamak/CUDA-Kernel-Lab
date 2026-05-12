#pragma once
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

class CudaTimer {
public:
    CudaTimer() {
        gpu::cuda_check(cudaEventCreate(&_start));
        gpu::cuda_check(cudaEventCreate(&_stop));
    }

    ~CudaTimer() {
        gpu::cuda_check(cudaEventDestroy(_start));
        gpu::cuda_check(cudaEventDestroy(_stop));
    }

    void start() {
        gpu::cuda_check(cudaEventRecord(_start));
    }

    float stop() {
        gpu::cuda_check(cudaEventRecord(_stop));
        gpu::cuda_check(cudaEventSynchronize(_stop));
        float ms = 0.0f;
        gpu::cuda_check(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }

private:
    cudaEvent_t _start{},
    cudaEvent_t _stop{};
};