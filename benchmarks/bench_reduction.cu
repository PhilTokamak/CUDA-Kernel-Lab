#include "reduction.hpp"
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

struct ReductionResult
{
    std::string kernel{}; // kernel name, e.g. `reduction`
    std::string version{}; // implementation version, e.g. `cpu_serial`, `cuda_atomicAdd`, `pinned memcpy`
    std::string mode{}; // measure mode / execution stage, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`
    std::string dtype{}; // data type, e.g. fp32
    size_t size_of_dtype{};

    size_t n = 0; // vector length
    int block_size = 0; // CUDA block size (0 for CPU/H2D/D2H)
    int grid_size = 0;
    size_t interm_stage_elems = 0; // temporary (partial sum) vector size

    int repeat{};
    size_t bytes = 0;
    TimeStats time_stats;

    float result = 0.0f;
    float ref = 0.0f;

    double abs_error = 0.0;
    double rel_error = 0.0;
    bool correct = false;

    double bw_GB_s = 0.0;
    double gflops = 0.0;
};

inline size_t num_flop_reduction(size_t n)
{
    return n> 0? n - 1 : 0;
}

inline size_t num_bytes_reduction(size_t n, size_t interm_stage_elems)
{
    return bytes_per_vector<float>(n + interm_stage_elems);
}

void calc_error(ReductionResult& r, float abs_tol = 1e-3f, float rel_tol = 1e-4f)
{
    r.abs_error = std::abs(static_cast<double>(r.result) - static_cast<double>(r.ref));
    r.rel_error = r.abs_error / (std::abs(static_cast<double>(r.ref)) + 1e-12);

    r.correct = (r.abs_error < abs_tol) || (r.rel_error < rel_tol);
}

void init_array(float* a, size_t num_elem)
{
    // Initialize to 1 to check corrrectness
    for(size_t i = 0; i < num_elem; ++i)
    {
        a[i] = 1.0f;
    }
}

void write_csv_header(std::ofstream& out)
{
    std::vector<std::string> headers
    {
        "kernel",     // kernel name, e.g. `vector_add`
        "version",    // implementation version, e.g. `cpu_serial`, `cuda_naive`, `pinned memcpy`
        "mode",       // measure mode, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`, `cpu_finalize`
        "dtype",      // data type, e.g. fp32
        "size_of_dtype", // size of dtype (in bytes)

        "n",          // vector length
        "block_size", // CUDA block size (0 for CPU/H2D/D2H)
        "grid_size", // CUDA grid size (0 for CPU/H2D/D2H)
        "interm_stage_elems", // intermediate stage sum array length
        "repeat",    // number of measurement repetitions

        "avg_ms",
        "min_ms",
        "max_ms",
        "std_ms",

        "bytes",     // bytes used for bandwidth calculation
        "bw_GB_s",   // Effective bandwidth
        "gflops",

        "result",
        "ref",
        "abs_error",
        "rel_error",
        "correct"
    };

    fmt::print(out, "{}\n", fmt::join(headers, ","));
}

void write_csv_row(std::ofstream& out,
                   const ReductionResult& r)
{
    fmt::print(out,
                "{},{},{},{},{},"
                "{},{},{},{},{},"
                "{:.6f},{:.6f},{:.6f},{:.3f},"
                "{},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{}\n",
                r.kernel,
                r.version,
                r.mode,
                r.dtype,
                r.size_of_dtype,

                r.n,
                r.block_size,
                r.grid_size,
                r.interm_stage_elems,
                r.repeat,

                r.time_stats.avg_ms,
                r.time_stats.min_ms,
                r.time_stats.max_ms,
                r.time_stats.std_ms,

                r.bytes,
                r.bw_GB_s,
                r.gflops,

                r.result,
                r.ref,
                r.abs_error,
                r.rel_error,
                (r.correct ? "true" : "false"));
}

ReductionResult bench_reduction_cpu(const float* x_host, size_t n, int repeat, CpuTimer& cpu_timer)
{
    ReductionResult r{};

    r.kernel = "reduction";
    r.version = "cpu_serial";
    r.mode = "cpu";
    r.dtype = "fp32";
    r.size_of_dtype = sizeof(float);
    r.n = n;
    r.block_size = 0;
    r.grid_size = 0;
    r.interm_stage_elems = 0; // no intermediate stage
    r.repeat = repeat;

    r.time_stats = benchmark_cpu(
        [&]() {
            reduce_cpu(x_host, r.result, n);
        }, repeat, cpu_timer);
    r.bytes = num_bytes_reduction(n, 0);
    r.ref = r.result;
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops = calculate_gflops(num_flop_reduction(n), r.time_stats.avg_ms);

    r.abs_error = 0.0;
    r.rel_error = 0.0;
    r.correct = true;

    return r;
}

template <typename LaunchKernel, typename ReadResult>
ReductionResult bench_reduction_kernel_only(
    const std::string& kernel,
    const std::string& version,
    const std::string& mode,
    const std::string& dtype,
    size_t n,
    int block_size,
    int grid_size,
    size_t interm_stage_elems, // Still need to determine if this is needed or not
    int repeat,
    float ref,
    LaunchKernel launch_kernel,
    ReadResult read_result,
    CudaTimer& cuda_timer)
{
    ReductionResult r{};
    r.kernel = kernel;
    r.version = version;
    r.mode = mode;
    r.dtype = dtype;
    r.size_of_dtype = sizeof(float);
    r.n = n;
    r.block_size = block_size;
    r.grid_size = grid_size;
    r.interm_stage_elems = interm_stage_elems;
    r.repeat = repeat;
    r.bytes = num_bytes_reduction(n, interm_stage_elems);
    r.ref = ref;

    r.time_stats = benchmark_cuda_kernel(
        [&]() {
            launch_kernel();
        }, repeat, cuda_timer
    );

    // Wait kernel to finish execution
    gpu::cuda_check(cudaDeviceSynchronize());

    r.result = read_result();
    calc_error(r);

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops = calculate_gflops(num_flop_reduction(n), r.time_stats.avg_ms);

    return r;
}

ReductionResult bench_reduction_h2d(
    const std::string& kernel,
    const std::string& version,
    const std::string& mode,
    const std::string& dtype,
    size_t n,
    int block_size,
    int grid_size,
    const float* in_host,
    float* in_dev,
    int repeat,
    CpuTimer& cpu_timer
)
{
    ReductionResult r{};
    r.kernel = kernel;
    r.version = version;
    r.mode = mode;
    r.dtype = dtype;
    r.size_of_dtype = sizeof(float);
    r.n = n;
    r.block_size = block_size;
    r.grid_size = grid_size;
    r.interm_stage_elems = 0;
    r.repeat = repeat;
    r.bytes = bytes_per_vector<float>(n);

    r.time_stats = benchmark_cpu(
        [&]() {
            // Copy kernel results back to host
            gpu::cuda_check(cudaMemcpy(
                in_dev, in_host,
                n * sizeof(float),
                cudaMemcpyDefault));
        }, repeat, cpu_timer);

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops = 0.0;

    r.abs_error = 0.0,
    r.rel_error = 0.0,
    r.correct = true;
    return r;
}

ReductionResult bench_reduction_d2h(
    const std::string& kernel,
    const std::string& version,
    const std::string& mode,
    const std::string& dtype,
    size_t n,
    int block_size,
    int grid_size,
    size_t interm_stage_elems,
    const float* out_dev,
    float* out_host,
    int repeat,
    CpuTimer& cpu_timer
)
{
    ReductionResult r{};
    r.kernel = kernel;
    r.version = version;
    r.mode = mode;
    r.dtype = dtype;
    r.size_of_dtype = sizeof(float);
    r.n = n;
    r.block_size = block_size;
    r.grid_size = grid_size;
    r.interm_stage_elems = interm_stage_elems;
    r.repeat = repeat;
    r.bytes = bytes_per_vector<float>(interm_stage_elems);

    r.time_stats = benchmark_cpu(
        [&]() {
            // Copy kernel results back to host
            gpu::cuda_check(cudaMemcpy(
                out_host, out_dev,
                interm_stage_elems * sizeof(float),
                cudaMemcpyDefault));
        }, repeat, cpu_timer);

    r.correct = true;
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops = 0.0;

    return r;
}

ReductionResult bench_cpu_finalize(
    const std::string& kernel,
    const std::string& version,
    const std::string& mode,
    const std::string& dtype,
    size_t n,
    int block_size,
    int grid_size,
    size_t interm_stage_elems, // Still need to determine if this is needed or not
    const float* partial_sum_host,
    int repeat,
    float ref,
    CpuTimer& cpu_timer
)
{
    ReductionResult r{};
    r.kernel = kernel;
    r.version = version;
    r.mode = mode;
    r.dtype = dtype;
    r.size_of_dtype = sizeof(float);
    r.n = n;
    r.block_size = block_size;
    r.grid_size = grid_size;
    r.interm_stage_elems = interm_stage_elems;
    r.repeat = repeat;
    r.bytes = num_bytes_reduction(interm_stage_elems, 1);
    r.ref = ref;

    float final_sum{};
    r.time_stats = benchmark_cpu(
        [&]() {
            reduce_cpu(partial_sum_host, final_sum, interm_stage_elems);
        }, repeat, cpu_timer);

    r.result = final_sum;
    calc_error(r);

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);
    r.gflops = calculate_gflops(num_flop_reduction(interm_stage_elems), r.time_stats.avg_ms);

    return r;
}


int main()
{
    // Initilize CUDA timer
    CudaTimer cuda_timer{CudaTimer()};
    CpuTimer cpu_timer{CpuTimer()};

    const std::string kernel = "reduction";

    // Create directory to save benchmark results csv file
    std::error_code ec;
    std::filesystem::create_directories("results/data");
    if (ec)
    {
        fmt::print("{}\n", ec.message());
    }

    std::ofstream out("results/data/reduction.csv");
    write_csv_header(out);

    // Vector size to sweep
    std::vector<size_t> sizes_vec{1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                                  1 << 23, 1 << 24, 1 << 25, 1 << 26, 1 << 27};

    // GPU block size to sweep
    std::vector<int> block_sizes{64, 128, 256, 512, 1024};

    // Number of measurement repetitions
    int repeat_cpu  = 10;
    int repeat_kernel  = 100;
    int repeat_cpu_finalize = 50;
    int repeat_copy = 10;

    // Sweep over vector lengths
    for (size_t n : sizes_vec)
    {
        size_t bytes = bytes_per_vector<float>(n);
        fmt::print("Running vector_add benchmark for n = {} (Array size = {} MiB)\n",
                   n, static_cast<double>(bytes) / static_cast<double>(1ULL << 20));

        // Pointers to host memory
        host_ptr x_host(cuda_malloc_host<float>(n));
        host_ptr final_gpu_sum_host(cuda_malloc_host<float>(1));

        // Pointers to device memory
        float* x_dev = nullptr;
            // used by kernels without interm stage or need cpu finalization
        float* final_gpu_sum_dev = nullptr;

        // Initialize vectors on the host
        init_array(x_host.get(), n);

        // CPU benchmark
        ReductionResult result_cpu{};
        result_cpu = bench_reduction_cpu(
            x_host.get(),
            n,
            repeat_cpu,
            cpu_timer
        );

        write_csv_row(out, result_cpu);

        float ref = result_cpu.result;

        // Allocate device memory
        gpu::cuda_check(cudaMalloc(&x_dev, bytes));

        // H2D benchmark
        ReductionResult result_h2d{};
        result_h2d = bench_reduction_h2d(
            kernel,
            "all_gpu_version",
            "h2d",
            "fp32",
            n,
            0,
            0,
            x_host.get(),
            x_dev,
            repeat_copy,
            cpu_timer
        );
        write_csv_row(out, result_h2d);

        // CUDA kernel sweep over block sizes
        for (size_t block_size : block_sizes)
        {
            // 1. Benchmark Atomic Add version
            {
                std::string version = "cuda_atomic_add";
                int grid_size_atomic = static_cast<int>((n + block_size - 1) / block_size);
                gpu::cuda_check(cudaMalloc(&final_gpu_sum_dev, 1 * sizeof(float)));
                ReductionResult result_atomic{};
                result_atomic = bench_reduction_kernel_only(
                    kernel,
                    version,
                    "cuda_kernel",
                    "fp32",
                    n,
                    block_size,
                    grid_size_atomic,
                    1,
                    repeat_kernel,
                    ref,
                    [&]() {
                        // IMPORTANT: before launching this kernel, set value output array to zero
                        gpu::cuda_check(cudaMemset(final_gpu_sum_dev, 0, sizeof(float)));
                        // Launch kernel
                        launch_reduce_atomic(
                            x_dev,
                            final_gpu_sum_dev,
                            n,
                            block_size=block_size);
                    },
                    [&]() -> float {
                        gpu::cuda_check(cudaMemcpy(final_gpu_sum_host.get(), final_gpu_sum_dev, sizeof(float), cudaMemcpyDefault));
                        return final_gpu_sum_host[0];
                    }, cuda_timer
                );
                write_csv_row(out, result_atomic);

                // benchmark d2h
                ReductionResult result_d2h_atomic{};
                result_d2h_atomic = bench_reduction_d2h(
                    kernel,
                    version,
                    "d2h",
                    "fp32",
                    n,
                    0,
                    0,
                    0,
                    final_gpu_sum_host.get(),
                    final_gpu_sum_dev,
                    repeat_copy,
                    cpu_timer
                );
                write_csv_row(out, result_d2h_atomic);
            }

            // 2. Benchmark block shared memory version
            {
                std::string version = "cuda_block_shared_mem";
                int grid_size_block_version = static_cast<int>((n + block_size - 1) / block_size);
                int interm_stage_elems = grid_size_block_version;

                // Allocate interm stage (partial sum) host array memory
                host_ptr partial_sum_host(cuda_malloc_host<float>(interm_stage_elems));

                // Allocate interm stage (partial sum) device memory
                float* partial_sum_dev = nullptr;
                gpu::cuda_check(cudaMalloc(&partial_sum_dev, interm_stage_elems * sizeof(float)));

                ReductionResult result_kernel{};
                result_kernel = bench_reduction_kernel_only(
                    kernel,
                    version,
                    "cuda_kernel",
                    "fp32",
                    n,
                    block_size,
                    grid_size_block_version,
                    interm_stage_elems,
                    repeat_kernel,
                    ref,
                    [&]() {
                        launch_reduce_block(x_dev, partial_sum_dev, n, block_size);
                    },
                    [&]() -> float {
                        gpu::cuda_check(cudaMemcpy(partial_sum_host.get(), partial_sum_dev, interm_stage_elems * sizeof(float), cudaMemcpyDefault));
                        reduce_cpu(partial_sum_host.get(), final_gpu_sum_host[0], interm_stage_elems);
                        return final_gpu_sum_host[0];
                    },
                    cuda_timer
                );
                write_csv_row(out, result_kernel);

                // benchmark d2h
                ReductionResult result_d2h{};
                result_d2h = bench_reduction_d2h(
                    kernel,
                    version,
                    "d2h",
                    "fp32",
                    n,
                    block_size,
                    grid_size_block_version,
                    interm_stage_elems,
                    partial_sum_dev,
                    partial_sum_host.get(),
                    repeat_copy,
                    cpu_timer
                );
                write_csv_row(out, result_d2h);

                // benchmark cpu finalization
                ReductionResult result_cpu_final{};
                result_cpu_final = bench_cpu_finalize(
                    kernel,
                    version,
                    "cpu_finalize",
                    "fp32",
                    n,
                    block_size,
                    grid_size_block_version,
                    interm_stage_elems,
                    partial_sum_host.get(),
                    repeat_cpu_finalize,
                    ref,
                    cpu_timer
                );
                write_csv_row(out, result_cpu_final);

                // free gpu allocated memory
                gpu::cuda_check(cudaFree(partial_sum_dev));
            }

            // 3. Benchmark grid-stride-loop block-shared memory version
            {
                std::string version = "cuda_grid_stride_block_shared";
                // Number of SM for the GPU
                int num_sms;
                cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
                int grid_size_grid_stride_version = 4 * num_sms;
                int interm_stage_elems = grid_size_grid_stride_version;

                // Allocate interm stage (partial sum) host array memory
                host_ptr partial_sum_host(cuda_malloc_host<float>(interm_stage_elems));

                // Allocate interm stage (partial sum) device memory
                float* partial_sum_dev = nullptr;
                gpu::cuda_check(cudaMalloc(&partial_sum_dev, interm_stage_elems * sizeof(float)));

                ReductionResult result_kernel{};
                result_kernel = bench_reduction_kernel_only(
                    kernel,
                    version,
                    "cuda_kernel",
                    "fp32",
                    n,
                    block_size,
                    grid_size_grid_stride_version,
                    interm_stage_elems,
                    repeat_kernel,
                    ref,
                    [&]() {
                        launch_reduce_grid_stride_block(x_dev, partial_sum_dev, n, block_size, grid_size_grid_stride_version);
                    },
                    [&]() -> float {
                        gpu::cuda_check(cudaMemcpy(partial_sum_host.get(), partial_sum_dev, interm_stage_elems * sizeof(float), cudaMemcpyDefault));
                        reduce_cpu(partial_sum_host.get(), final_gpu_sum_host[0], interm_stage_elems);
                        return final_gpu_sum_host[0];
                    },
                    cuda_timer
                );
                write_csv_row(out, result_kernel);

                // benchmark d2h
                ReductionResult result_d2h{};
                result_d2h = bench_reduction_d2h(
                    kernel,
                    version,
                    "d2h",
                    "fp32",
                    n,
                    block_size,
                    grid_size_grid_stride_version,
                    interm_stage_elems,
                    partial_sum_dev,
                    partial_sum_host.get(),
                    repeat_copy,
                    cpu_timer
                );
                write_csv_row(out, result_d2h);

                // benchmark block shared memory version cpu finalization
                ReductionResult result_cpu_final{};
                result_cpu_final = bench_cpu_finalize(
                    kernel,
                    version,
                    "cpu_finalize",
                    "fp32",
                    n,
                    block_size,
                    grid_size_grid_stride_version,
                    interm_stage_elems,
                    partial_sum_host.get(),
                    repeat_cpu_finalize,
                    ref,
                    cpu_timer
                );
                write_csv_row(out, result_cpu_final);

                // free gpu allocated memory
                gpu::cuda_check(cudaFree(partial_sum_dev));
            }

        };

        gpu::cuda_check(cudaFree(x_dev));
        gpu::cuda_check(cudaFree(final_gpu_sum_dev));

    }
}