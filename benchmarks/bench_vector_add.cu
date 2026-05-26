#include "bench_utils.hpp"
#include "cuda_utils.cuh"
#include "vector_add.hpp"

#include <cuda/cmath>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct VectorAddResult
{
    std::string kernel{}; // kernel name, e.g. `vector_add`
    std::string
        version{};       // implementation version, e.g. `cpu_serial`, `cuda_naive`, `pinned memcpy`
    std::string mode{};  // measure mode, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`
    std::string dtype{}; // data type, e.g. fp32

    size_t n       = 0; // vector length
    int block_size = 0; // CUDA block size (0 for CPU/H2D/D2H)
    int repeat     = 0; // number of measurement repetitions

    TimeStats time_stats{};
    size_t bytes = 0; // bytes used for bandwidth calculation

    double bw_GB_s = 0.0; // Effective bandwidth
    double gflops  = 0.0;

    bool correct = false;
};

struct BenchmarkConfig
{
    std::string kernel{"vector_add"};

    std::filesystem::path output_dir{"results/data"};
    std::filesystem::path output_csv{"results/data/vector_add.csv"};

    DTypeInfo dtype{FP32_DTYPE};

    // Vector size to sweep
    std::vector<size_t> sizes_vec{1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                                  1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27};

    // GPU block size to sweep
    std::vector<int> block_sizes{64, 128, 256, 512, 1024};

    // Number of measurement repetitions
    int repeat_cpu  = 10;
    int repeat_gpu  = 100;
    int repeat_copy = 10;
};

void init_array(float* a, size_t num_elem)
{
    // Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < num_elem; ++i)
    {
        a[i] = dis(gen);
    }
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
    return num_elem;
}

VectorAddResult make_vector_add_result(const std::string& kernel, const std::string& version,
                                       const std::string& mode, const DTypeInfo& dtype, size_t n,
                                       int block_size, int repeat)
{
    VectorAddResult r{};

    r.kernel  = kernel;
    r.version = version;
    r.mode    = mode;
    r.dtype   = dtype.name;

    r.n          = n;
    r.block_size = block_size;
    r.repeat     = repeat;

    return r;
}

/**
 * This function calculates bandwidth (GB/s) amd GFlops for an input result
 */
void cal_bw_and_gflops_for_result(VectorAddResult& r)
{
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops  = calculate_gflops(num_flop_vector_add(r.n), r.time_stats.avg_ms);
}

void write_csv_header(std::ofstream& out)
{
    std::vector<std::string> headers{
        "kernel",     // kernel name, e.g. `vector_add`
        "version",    // implementation version, e.g. `cpu_serial`, `cuda_naive`, `pinned memcpy`
        "mode",       // measure mode, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`
        "dtype",      // data type, e.g. fp32
        "n",          // vector length
        "block_size", // CUDA block size (0 for CPU/H2D/D2H)
        "repeat",     // number of measurement repetitions
        "avg_ms",     "min_ms", "max_ms", "std_ms",
        "bytes",   // bytes used for bandwidth calculation
        "bw_GB_s", // Effective bandwidth
        "gflops",     "correct"};

    fmt::print(out, "{}\n", fmt::join(headers, ","));
}

void write_csv_row(std::ofstream& out, const VectorAddResult& r)
{
    fmt::print(out, "{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:3f},{},{:.3f},{:.3f},{}\n",
               r.kernel, r.version, r.mode, r.dtype, r.n, r.block_size, r.repeat,
               r.time_stats.avg_ms, r.time_stats.min_ms, r.time_stats.max_ms, r.time_stats.std_ms,
               r.bytes, r.bw_GB_s, r.gflops, (r.correct ? "true" : "false"));
}

VectorAddResult bench_vector_add_cpu(const BenchmarkConfig& config, const float* a_host,
                                     const float* b_host, float* result_host, size_t n,
                                     CpuTimer& cpu_timer)
{
    VectorAddResult r = make_vector_add_result(config.kernel, "cpu_serial", "cpu", config.dtype, n,
                                               0, config.repeat_cpu);

    r.time_stats = benchmark_cpu([&]() { vector_add_cpu(a_host, b_host, result_host, n); },
                                 config.repeat_cpu, cpu_timer);

    r.bytes   = bytes_vector_add(n);
    r.correct = true;

    cal_bw_and_gflops_for_result(r);

    return r;
}

VectorAddResult bench_vector_add_h2d(const BenchmarkConfig& config, const float* a_host,
                                     const float* b_host, float* a_dev, float* b_dev, size_t n,
                                     CpuTimer& cpu_timer)
{
    VectorAddResult r = make_vector_add_result(config.kernel, "pinned_memcpy", "h2d", config.dtype,
                                               n, 0, config.repeat_copy);

    size_t bytes = bytes_per_vector<float>(n);

    r.time_stats = benchmark_cpu(
        [&]()
        {
            // Copy data to the GPU
            gpu::cuda_check(cudaMemcpy(a_dev, a_host, bytes, cudaMemcpyDefault));

            gpu::cuda_check(cudaMemcpy(b_dev, b_host, bytes, cudaMemcpyDefault));
        },
        config.repeat_copy, cpu_timer);

    r.bytes   = h2d_bytes_vector_add(n);
    r.correct = true;

    cal_bw_and_gflops_for_result(r);

    return r;
}

VectorAddResult bench_vector_add_kernel(const BenchmarkConfig& config, const float* a_dev,
                                        const float* b_dev, float* result_dev,
                                        const float* result_host, float* result_from_dev, size_t n,
                                        int block_size, CudaTimer& cuda_timer)
{
    VectorAddResult r = make_vector_add_result(config.kernel, "cuda_naive", "cuda_kernel",
                                               config.dtype, n, block_size, config.repeat_gpu);

    size_t bytes = bytes_per_vector<float>(n);

    r.time_stats = benchmark_cuda_kernel(
        [&]()
        {
            // Launch the kernel
            launch_vector_add(a_dev, b_dev, result_dev, n, block_size);
        },
        config.repeat_gpu, cuda_timer);

    // Wait kernel to finish execution
    gpu::cuda_check(cudaDeviceSynchronize());

    // Copy kernel results back to host
    gpu::cuda_check(cudaMemcpy(result_from_dev, result_dev, bytes, cudaMemcpyDefault));

    // check results correctness
    r.correct = check_result(result_host, result_from_dev, n);

    r.bytes = bytes_vector_add(n);

    cal_bw_and_gflops_for_result(r);

    return r;
}

VectorAddResult bench_vector_add_d2h(const BenchmarkConfig& config, const float* result_dev,
                                     float* result_from_dev, size_t n, CpuTimer& cpu_timer)
{
    VectorAddResult r = make_vector_add_result(config.kernel, "pinned_memcpy", "d2h", config.dtype,
                                               n, 0, config.repeat_copy);

    size_t bytes = bytes_per_vector<float>(n);

    r.time_stats = benchmark_cpu(
        [&]()
        {
            // Copy kernel results back to host
            gpu::cuda_check(cudaMemcpy(result_from_dev, result_dev, bytes, cudaMemcpyDefault));
        },
        config.repeat_copy, cpu_timer);

    r.bytes   = d2h_bytes_vector_add(n);
    r.correct = true;

    cal_bw_and_gflops_for_result(r);

    return r;
}

void run_kernel_block_size_sweep(std::ofstream& out, const BenchmarkConfig& config,
                                 const float* a_dev, const float* b_dev, float* result_dev,
                                 const float* result_host, float* result_from_dev, size_t n,
                                 CudaTimer& cuda_timer)
{
    // CUDA kernel sweep over block sizes
    for (int block_size : config.block_sizes)
    {
        VectorAddResult result_kernel =
            bench_vector_add_kernel(config, a_dev, b_dev, result_dev, result_host, result_from_dev,
                                    n, block_size, cuda_timer);

        // Write GPU benchmark results
        write_csv_row(out, result_kernel);
    }
}

void run_single_size_benchmark(std::ofstream& out, const BenchmarkConfig& config, size_t n,
                               CudaTimer& cuda_timer, CpuTimer& cpu_timer)
{
    size_t bytes = bytes_per_vector<float>(n);

    fmt::print("Running vector_add benchmark for n = {} (Array size = {} MiB)\n", n,
               static_cast<double>(bytes) / static_cast<double>(1ULL << 20));

    // Pointers to host memory
    host_ptr a_host(cuda_malloc_host<float>(n));
    host_ptr b_host(cuda_malloc_host<float>(n));
    host_ptr result_host(cuda_malloc_host<float>(n));
    host_ptr result_from_dev(cuda_malloc_host<float>(n));

    // Initialize vectors on the host
    init_array(a_host.get(), n);
    init_array(b_host.get(), n);

    // CPU benchmark
    VectorAddResult result_cpu =
        bench_vector_add_cpu(config, a_host.get(), b_host.get(), result_host.get(), n, cpu_timer);

    // Write CPU benchmark results
    write_csv_row(out, result_cpu);

    // Allocate device memory
    DeviceBuffer<float> a_dev(n);
    DeviceBuffer<float> b_dev(n);
    DeviceBuffer<float> result_dev(n);

    // H2D benchmark
    VectorAddResult result_h2d = bench_vector_add_h2d(config, a_host.get(), b_host.get(),
                                                      a_dev.get(), b_dev.get(), n, cpu_timer);

    // Write H2D benchmark results
    write_csv_row(out, result_h2d);

    run_kernel_block_size_sweep(out, config, a_dev.get(), b_dev.get(), result_dev.get(),
                                result_host.get(), result_from_dev.get(), n, cuda_timer);

    // D2H benchmark
    VectorAddResult result_d2h =
        bench_vector_add_d2h(config, result_dev.get(), result_from_dev.get(), n, cpu_timer);

    // Write D2H benchmark results
    write_csv_row(out, result_d2h);

    // GPU allocated memory is released automatically by DeviceBuffer.
}

void run_benchmark(const BenchmarkConfig& config)
{
    // Initialize timers
    CudaTimer cuda_timer{CudaTimer()};
    CpuTimer cpu_timer{CpuTimer()};

    create_output_directory(config.output_dir);

    std::ofstream out(config.output_csv);
    write_csv_header(out);

    // Sweeep over vectoe lengths
    for (size_t n : config.sizes_vec)
    {
        run_single_size_benchmark(out, config, n, cuda_timer, cpu_timer);
    }
}

int main()
{
    BenchmarkConfig config{};
    run_benchmark(config);

    return 0;
}