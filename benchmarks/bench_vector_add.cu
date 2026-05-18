#include "vector_add.hpp"
#include "cuda_utils.cuh"
#include "bench_utils.hpp"

#include <random>
#include <fstream>
#include <string>
#include <vector>
#include <fmt/ranges.h>
#include <fmt/ostream.h>
#include <filesystem>
#include <cuda/cmath>

void init_array(float* a, size_t num_elem)
{
    // Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(size_t i = 0; i < num_elem; ++i)
    {
        a[i] = dis(gen);
    }
}

template<typename T>
inline size_t bytes_per_vector(size_t num_elem)
{
    return num_elem * sizeof(T);
}

inline size_t bytes_vector_add(size_t num_elem)
{
    // read a + read b + write c
    return 3 * bytes_per_vector<float>(num_elem);
}

inline size_t h2d_bytes_vector_add(size_t num_elem)
{
    // copy a and b from host to device
    return 2 * bytes_per_vector<float>(num_elem);
}

inline size_t d2h_bytes_vector_add(size_t num_elem)
{
    // copy c from deivec to host
    return 1 * bytes_per_vector<float>(num_elem);
}

inline size_t num_flop_vector_add(size_t num_elem)
{
    // 1 addition per element
    return bytes_per_vector<float>(num_elem);
}

void write_csv_header(std::ofstream& out)
{
    std::vector<std::string> headers
    {
        "kernel",    // kernel name, e.g. `vector_add`
        "version",   // implementation version, e.g. `cpu_serial`, `cuda_naive`, `pinned memcpy`
        "mode",      // measure mode, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`
        "dtype",     // data type, e.g. fp32
        "n",         // vector length
        "block_size",// CUDA block size (0 for CPU/H2D/D2H)
        "repeat",    // number of measurement repetitions
        "avg_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "bytes",     // bytes used for bandwidth calculation
        "bw_GB_s",   // Effective bandwidth
        "gflops",
        "correct"
    };

    fmt::print(out, "{}\n", fmt::join(headers, ","));
}

void write_csv_row(std::ofstream& out,
                   const std::string& kernel,
                   const std::string& version,
                   const std::string& mode,
                   const std::string& dtype,
                   size_t n,
                   int block_size,
                   int repeat,
                   const TimeStats& time_stats,
                   size_t bytes,
                   bool correct)
{
    double bw_GB_s = bandwidth_GB_s(bytes, time_stats.avg_ms);
    double gflops = calculate_gflops(num_flop_vector_add(n), time_stats.avg_ms);

    fmt::print(out,
               "{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:3f},{},{:.3f},{:.3f},{}\n",
               kernel,
               version,
               mode,
               dtype,
               n,
               block_size,
               repeat,
               time_stats.avg_ms,
               time_stats.min_ms,
               time_stats.max_ms,
               time_stats.std_ms,
               bytes,
               bw_GB_s,
               gflops,
               (correct ? "true" : "false"));
}

int main()
{
    // Initilize timers
    CudaTimer cuda_timer{CudaTimer()};
    CpuTimer cpu_timer{CpuTimer()};

    // Create directory to save benchmark results csv file
    std::error_code ec;
    std::filesystem::create_directories("results/data");
    if (ec)
    {
        fmt::print("{}\n", ec.message());
    }

    std::ofstream out("results/data/vector_add.csv");
    write_csv_header(out);

    // Vector size to sweep
    std::vector<size_t> sizes_vec{1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                                  1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27};

    // GPU block size to sweep
    std::vector<int> block_sizes{64, 128, 256, 512, 1024};

    // Number of measurement repetitions
    int repeat_cpu  = 10;
    int repeat_gpu  = 100;
    int repeat_copy = 10;

    // Sweep over vector lengths
    for (size_t n : sizes_vec)
    {
        size_t bytes = bytes_per_vector<float>(n);
        fmt::print("Running vector_add benchmark for n = {} (Array size = {} MiB)\n",
                   n, static_cast<double>(bytes) / static_cast<double>(1ULL << 20));

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

        // CPU benchmark
        auto time_stats_cpu = benchmark_cpu(
            [&]() {
                vector_add_cpu(a_host.get(), b_host.get(), result_host.get(), n);
            },
        repeat_cpu, cpu_timer);

        // Write CPU benchmark results
        write_csv_row(out,
            "vector_add",
            "cpu_serial",
            "cpu",
            "fp32",
            n,
            0,
            repeat_cpu,
            time_stats_cpu,
            bytes_vector_add(n),
            true);

        // Allocate device memory
        gpu::cuda_check(cudaMalloc(&a_dev, bytes));
        gpu::cuda_check(cudaMalloc(&b_dev, bytes));
        gpu::cuda_check(cudaMalloc(&result_dev, bytes));

        // H2D benchmark
        auto time_stats_h2d = benchmark_cpu(
            [&]() {
                // Copy data to the GPU
                gpu::cuda_check(cudaMemcpy(a_dev, a_host.get(), bytes, cudaMemcpyDefault));
                gpu::cuda_check(cudaMemcpy(b_dev, b_host.get(), bytes, cudaMemcpyDefault));
            },
        repeat_copy, cpu_timer);

        // Write H2D benchmark results
        write_csv_row(out,
            "vector_add",
            "pinned_memcpy",
            "h2d",
            "fp32",
            n,
            0,
            repeat_copy,
            time_stats_h2d,
            h2d_bytes_vector_add(n),
            true);

        // CUDA kernel sweep over block sizes
        for (size_t block_size : block_sizes)
        {
            auto time_stats_kernel = benchmark_cuda_kernel([&]() {
                // Launch the kernel
                launch_vector_add(a_dev, b_dev, result_dev, n, block_size);
            },
            repeat_gpu, cuda_timer);

            // Wait kernel to finish execution
            gpu::cuda_check(cudaDeviceSynchronize());

            // Copy kernel results back to host
            gpu::cuda_check(cudaMemcpy(result_from_dev.get(), result_dev, bytes, cudaMemcpyDefault));

            // check results correctness
            bool correct = check_result(result_host.get(), result_from_dev.get(), n);

            // Write CPU benchmark results
            write_csv_row(out,
                "vector_add",
                "cuda_naive",
                "cuda_kernel",
                "fp32",
                n,
                block_size,
                repeat_gpu,
                time_stats_kernel,
                bytes_vector_add(n),
                correct);
        }

        // D2H benchmark
        auto time_stats_d2h = benchmark_cpu(
            [&]() {
                // Copy kernel results back to host
                gpu::cuda_check(cudaMemcpy(result_from_dev.get(), result_dev,
                                           bytes, cudaMemcpyDefault));
            },
        repeat_copy, cpu_timer);

        // Write D2H benchmark results
        write_csv_row(out,
            "vector_add",
            "pinned_memcpy",
            "d2h",
            "fp32",
            n,
            0,
            repeat_copy,
            time_stats_d2h,
            d2h_bytes_vector_add(n),
            true);

        gpu::cuda_check(cudaFree(a_dev));
        gpu::cuda_check(cudaFree(b_dev));
        gpu::cuda_check(cudaFree(result_dev));
    }

    return 0;
}