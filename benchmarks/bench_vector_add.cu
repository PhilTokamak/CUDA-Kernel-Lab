#include "vector_add.hpp"
#include "cuda_utils.cuh"
#include "timer.hpp"

#include <random>
#include <cuda/cmath>

void init_array(float* a, int num_elem)
{
    // Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i = 0; i < num_elem; ++i)
    {
        a[i] = dis(gen);
    }
}

int main() {
    int n = 1 << 26;

    int bytes = n * sizeof(float);

    fmt::print("Vector length n = {}\nArray size = {} MiB per vector\n", n, bytes/static_cast<double>(1ULL << 20));

    // Initilize CUDA timer
    CudaTimer cuda_timer{CudaTimer()};
    CpuTimer cpu_timer{CpuTimer()};

    // Pointers to host memory
    host_ptr a_host(cuda_malloc_host<float>(n));
    host_ptr b_host(cuda_malloc_host<float>(n));
    host_ptr result_host(cuda_malloc_host<float>(n));
    host_ptr result_from_dev(cuda_malloc_host<float>(n));

    // Pointers to device memory
    float* a_dev = nullptr;
    float* b_dev = nullptr;
    float* result_dev = nullptr;

    // Initialize vectors on the host
    init_array(a_host.get(), n);
    init_array(b_host.get(), n);

    // Allocate device memory
    gpu::cuda_check(cudaMalloc(&a_dev, bytes));
    gpu::cuda_check(cudaMalloc(&b_dev, bytes));
    gpu::cuda_check(cudaMalloc(&result_dev, bytes));

    cpu_timer.start();
    // Copy data to the GPU
    gpu::cuda_check(cudaMemcpy(a_dev, a_host.get(), bytes, cudaMemcpyDefault));
    gpu::cuda_check(cudaMemcpy(b_dev, b_host.get(), bytes, cudaMemcpyDefault));
    float h2d_ms = cpu_timer.stop();

    int repeat_cpu = 3;
    cpu_timer.start();
    for (int i = 0; i < repeat_cpu; ++i)
    {
        vector_add_cpu(a_host.get(), b_host.get(), result_host.get(), n);
    }
    float avg_cpu_ms = cpu_timer.stop() / repeat_cpu;

    int repeat_gpu = 10;
    cuda_timer.start();
    // Launch the kernel
    for (int i = 0; i < repeat_gpu; ++i)
    {
        launch_vector_add(a_dev, b_dev, result_dev, n);
    }

    float avg_kernel_ms = cuda_timer.stop() / repeat_gpu;

    // Wait kernel to finish execution
    gpu::cuda_check(cudaDeviceSynchronize());

    cpu_timer.start();
    // Copy kernel results back to host
    gpu::cuda_check(cudaMemcpy(result_from_dev.get(), result_dev, bytes, cudaMemcpyDefault));
    float d2h_ms = cpu_timer.stop();

    if(check_result(result_host.get(), result_from_dev.get(), n))
    {
        fmt::print("Vector add: CPU and GPU results match\n");
    }
    else
    {
        fmt::print("Vector add: Error - CPU and GPU results do not match\n");
    }

    // Effective bandwidth
    // GPU: read a + read b + write c = 3 * n * sizeof(float);
    double gpu_bytes = 3.0 * bytes;
    double gpu_bandwidth_GB_s = gpu_bytes / (avg_kernel_ms / 1000.0) / 1e9;

    // CPU: assume that arrays are large enough that they can not be put into the last-level cache
    // thus the DRAM bandwidth is measured (roughly): read a + read b + write c = 3 * n * sizeof(float);
    double cpu_bytes = 3.0 * bytes;
    double cpu_bandwidth_GB_s = cpu_bytes / (avg_cpu_ms / 1000.0) / 1e9;

    // H2D bandwidth
    double h2d_bytes = 2.0 * bytes;
    double h2d_bandwidth_GB_s = h2d_bytes / (h2d_ms / 1000.0) / 1e9;

    // D2H bandwidth
    double d2h_bytes = 1.0 * bytes;
    double d2h_bandwidth_GB_s = d2h_bytes / (d2h_ms / 1000.0) / 1e9;

    // End-to-end effective bandwidth
    double e2e_bytes = 3.0 * bytes; // H2D two arrays + D2H one array
    double e2e_ms = h2d_ms + avg_kernel_ms + d2h_ms;
    double e2e_bandwidth_GB_s = e2e_bytes / (e2e_ms / 1000.0) / 1e9;

    fmt::print("Average cpu serial time: {} ms\n", avg_cpu_ms);
    fmt::print("H2D copy time: {} ms\n", h2d_ms);
    fmt::print("Average kernel-only time: {} ms\n", avg_kernel_ms);
    fmt::print("D2H copy time: {} ms\n", d2h_ms);
    fmt::print("GPU End-to-end time: {} ms\n", e2e_ms);
    fmt::print("Effevtive CPU bandwidth = {} GB/s\n", cpu_bandwidth_GB_s);
    fmt::print("Effevtive GPU bandwidth = {} GB/s\n", gpu_bandwidth_GB_s);
    fmt::print("H2D bandwidth: {} GB/s\n", h2d_bandwidth_GB_s);
    fmt::print("D2H bandwidth: {} GB/s\n", d2h_bandwidth_GB_s);
    fmt::print("GPU End-to-end effective bandwidth: {} GB/s\n", e2e_bandwidth_GB_s);


    gpu::cuda_check(cudaFree(a_dev));
    gpu::cuda_check(cudaFree(b_dev));
    gpu::cuda_check(cudaFree(result_dev));

    return 0;
}