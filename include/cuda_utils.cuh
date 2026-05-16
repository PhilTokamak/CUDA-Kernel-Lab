#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <source_location>
#include <stdexcept>
#include <memory>
#include <fmt/core.h>
#include <fmt/format.h>

namespace gpu{

inline void cuda_check(
    cudaError_t err, std::source_location loc =
    std::source_location::current())
{
    if(err != cudaSuccess) [[unlikely]]
    {
        throw std::runtime_error(
            fmt::format(
                "CUDA error at {}:{}:{}\n"
                "code={} ({})",
                loc.file_name(),
                loc.line(),
                loc.column(),
                static_cast<int>(err),
                cudaGetErrorString(err)
            )
        );
    }
}

inline void cuda_check_last(
    std::source_location loc = std::source_location::current())
{
    cuda_check(cudaGetLastError(), loc);
}

} // namespace gpu


inline bool check_result(const float* a, const float* b, size_t num_elem, float tol = 1e-5f)
{
    for(size_t i = 0; i < num_elem; ++i)
    {
        float diff = std::abs(a[i] - b[i]);
        if(diff > tol)
        {
            fmt::print("Mismatch at index {}, {} /= {}.\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

struct CudaHostDeleter
{
    void operator()(float* ptr) const
    {
        if(ptr)
        {
            cudaFreeHost(ptr);
        }
    }
};

using host_ptr = std::unique_ptr<float[], CudaHostDeleter>;

template<typename T>
T* cuda_malloc_host(size_t n)
{
    T* ptr = nullptr;

    gpu::cuda_check(cudaMallocHost(
        reinterpret_cast<void**>(&ptr),
        n * sizeof(T)
    ));

    return ptr;
}