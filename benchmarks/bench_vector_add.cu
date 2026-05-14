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

    // Copy data to the GPU
    gpu::cuda_check(cudaMemcpy(a_dev, a_host.get(), bytes, cudaMemcpyDefault));
    gpu::cuda_check(cudaMemcpy(b_dev, b_host.get(), bytes, cudaMemcpyDefault));

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

    // Copy kernel results back to host
    gpu::cuda_check(cudaMemcpy(result_from_dev.get(), result_dev, bytes, cudaMemcpyDefault));

    if(check_result(result_host.get(), result_from_dev.get(), n))
    {
        fmt::print("Vector add: CPU and GPU results match\n");
    }
    else
    {
        fmt::print("Vector add: Error - CPU and GPU results do not match\n");
    }

    // Effective bandwidth
    // read a + read b + write c = 3 * n * sizeof(float);
    double transferred_bytes = 3.0 * bytes;
    double bandwidth_GB_s = transferred_bytes / (avg_kernel_ms / 1000.0) / 1e9;

    fmt::print("Average cpu time: {} ms\n", avg_cpu_ms);
    fmt::print("Average kernel time: {} ms\n", avg_kernel_ms);
    fmt::print("Effevtive bandwidth = {} GB/s\n", bandwidth_GB_s);

    gpu::cuda_check(cudaFree(a_dev));
    gpu::cuda_check(cudaFree(b_dev));
    gpu::cuda_check(cudaFree(result_dev));

    return 0;
}