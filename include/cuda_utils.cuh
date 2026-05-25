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

/**
 * This function gets the number of SM for the GPU
 */
inline int get_num_sms()
{
    int device{};
    cuda_check(cudaGetDevice(&device));

    int num_sms{};
    cuda_check(cudaDeviceGetAttribute(
        &num_sms,
        cudaDevAttrMultiProcessorCount,
        device));

    return num_sms;
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

    // cudaMallocHost is a CUDA runtime function used to
    // allocate pinned (or page-locked) memory on the CPU
    gpu::cuda_check(cudaMallocHost(
        reinterpret_cast<void**>(&ptr),
        n * sizeof(T)
    ));

    return ptr;
}


/**
 * This is a class to manage device memory allocation based
 * on RAII and Rule of Five
 */
template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count)
    {
        allocate(count);
    }

    ~DeviceBuffer()
    {
        release();
    }

    // Delete copy constructor
    DeviceBuffer(const DeviceBuffer&) = delete;
    // Delete copy assignment operator
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _ptr(other._ptr),
          _count(other._count)
    {
        other._ptr = nullptr;
        other._count = 0;
    }

    // Move assignement operator
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();

            _ptr = other._ptr;
            _count = other._count;

            other._ptr = nullptr;
            other._count = 0;
        }
        return *this;
    }

    // Swap
    void swap(DeviceBuffer& other) noexcept
    {
        std::swap(_ptr, other._ptr);
        std::swap(_count, other._count);
    }

    T* get()
    {
        return _ptr;
    }

    const T* get() const
    {
        return _ptr;
    }

    size_t count() const
    {
        return _count;
    }

    size_t bytes() const
    {
        return _count * sizeof(T);
    }

    void allocate(size_t count)
    {
        release();

        _count = count;

        if(_count > 0)
        {
            gpu::cuda_check(cudaMalloc(
                reinterpret_cast<void**>(&_ptr),
                _count * sizeof(T)
            ));
        }
    }

    void release()
    {
        if (_ptr != nullptr)
        {
            cudaFree(_ptr);
            _ptr = nullptr;
            _count = 0;
        }
    }

private:
    T* _ptr{nullptr};
    size_t _count{0};
};