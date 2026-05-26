#pragma once
#include "cuda_utils.cuh"
#include <chrono>
#include <cuda_runtime.h>

class CudaTimer
{
  public:
    CudaTimer()
    {
        gpu::cuda_check(cudaEventCreate(&_start));
        gpu::cuda_check(cudaEventCreate(&_stop));
    }

    ~CudaTimer()
    {
        gpu::cuda_check(cudaEventDestroy(_start));
        gpu::cuda_check(cudaEventDestroy(_stop));
    }

    void start()
    {
        gpu::cuda_check(cudaEventRecord(_start));
    }

    [[nodiscard]]
    float stop()
    {
        gpu::cuda_check(cudaEventRecord(_stop));
        gpu::cuda_check(cudaEventSynchronize(_stop));
        float ms = 0.0f;
        gpu::cuda_check(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }

  private:
    cudaEvent_t _start{};
    cudaEvent_t _stop{};
};

class CpuTimer
{
  public:
    using clock = std::chrono::steady_clock;
    CpuTimer()  = default;

    void start()
    {
        _start = std::chrono::steady_clock::now();
    }

    [[nodiscard]]
    double stop()
    {
        _end                                         = clock::now();
        std::chrono::duration<double, std::milli> ms = _end - _start;
        return ms.count();
    }

  private:
    clock::time_point _start;
    clock::time_point _end;
};