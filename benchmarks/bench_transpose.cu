#include "bench_utils.hpp"
#include "cuda_utils.cuh"
#include "transpose.hpp"

#include <cuda/cmath>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>

void copy_cpu_baseline(const float* A, float* B, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        B[i] = A[i];
    }
}

__global__ void copy_cuda_baseline(const float* A, float* B, size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        B[idx] = A[idx];
    }
}

struct MatrixShape
{
    int rows;
    int cols;

    constexpr size_t num_elem() const noexcept
    {
        return static_cast<size_t>(rows) * cols;
    }

    constexpr size_t bytes() const noexcept
    {
        return num_elem() * sizeof(float);
    }
};

struct CheckResult
{
    bool correct;

    double max_abs_error{};
    double max_rel_error{};
};

CheckResult check_array_close(const float* a, const float* ref, size_t n, double abs_tol = 1e-6,
                              double rel_tol = 1e-6)
{
    CheckResult result;
    result.correct = true;

    for (size_t i = 0; i < n; ++i)
    {
        double val = static_cast<double>(a[i]);
        double r   = static_cast<double>(ref[i]);

        double abs_err = std::abs(val - r);
        double rel_err = abs_err / (std::abs(r) + 1e-12);

        if (abs_err > abs_tol + rel_err * std::abs(r))
        {
            result.correct = false;
        }

        result.max_abs_error = std::max(result.max_abs_error, abs_err);
        result.max_rel_error = std::max(result.max_rel_error, rel_err);
    }

    return result;
}

struct TransposeResult
{
    std::string kernel{}; // kernel name, e.g. `reduction`
    std::string
        version{}; // implementation version, e.g. `cpu_serial`, `cuda_atomicAdd`, `pinned memcpy`
    std::string mode{};  // measure mode / execution stage, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`
    std::string dtype{}; // data type, e.g. fp32
    size_t size_of_dtype{};

    MatrixShape shape = {0, 0}; // shape of matrix
    dim3 block_size{0, 0, 0};   // CUDA block size (0 for CPU/H2D/D2H)
    dim3 grid_size{0, 0, 0};    // CUDA grid size (0 for CPU/H2D/D2H)

    int repeat{};
    size_t bytes{};
    TimeStats time_stats;

    double bw_GB_s{};

    double max_abs_error{};
    double max_rel_error{};
    bool correct = false;
};

struct BenchmarkConfig
{
    std::string kernel{"transpose"};

    std::filesystem::path output_dir{"results/data"};
    std::filesystem::path output_csv{"results/data/transpose.csv"};

    // Vector size to sweep
    const std::vector<MatrixShape> shapes{
        {1024, 1024}, {2048, 2048}, {4096, 4096}, {4096, 2048}, {2048, 4096}};

    // GPU block size to sweep
    const std::vector<dim3> block_sizes{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 32}};

    // Number of measurement repetitions
    int repeat_cpu    = 10;
    int repeat_kernel = 100;
    int repeat_copy   = 10;
};

size_t num_bytes_transpose(MatrixShape shape)
{
    return 2 * shape.num_elem() * sizeof(float);
}

void init_matrix(float* mat, MatrixShape shape, unsigned seed = 12345)
{
    // Random number engine
    std::mt19937 gen(seed);

    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    int rows = shape.rows;
    int cols = shape.cols;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat[static_cast<size_t>(i) * cols + j] = dis(gen);
        }
    }
}

TransposeResult make_transpose_result(const std::string& kernel, const std::string& version,
                                      const std::string& mode, const DTypeInfo& dtype_info,
                                      MatrixShape shape, dim3 block_size, dim3 grid_size,
                                      int repeat)
{
    TransposeResult r{};

    r.kernel        = kernel;
    r.version       = version;
    r.mode          = mode;
    r.dtype         = dtype_info.name;
    r.size_of_dtype = dtype_info.size;

    r.shape      = shape;
    r.block_size = block_size;
    r.grid_size  = grid_size;
    r.repeat     = repeat;

    return r;
}

void write_csv_header(std::ofstream& out)
{
    std::vector<std::string> headers{
        "kernel",        // kernel name, e.g. `vector_add`
        "version",       // implementation version, e.g. `cpu_serial`, `cuda_naive`, `pinned memcpy`
        "mode",          // measure mode, e.g. `cpu`, `cuda_kernel`, `h2d`, `d2h`, `cpu_finalize`
        "dtype",         // data type, e.g. fp32
        "size_of_dtype", // size of dtype (in bytes)

        "rows",         // num of rows
        "cols",         // num of cols
        "block_size_x", // CUDA block size in x (0 for CPU/H2D/D2H)
        "block_size_y", // CUDA block size in y (0 for CPU/H2D/D2H)
        "grid_size_x",  // CUDA grid size in x (0 for CPU/H2D/D2H)
        "grid_size_y",  // CUDA grid size in y (0 for CPU/H2D/D2H)

        "repeat", // number of measurement repetitions
        "avg_ms", // average time in ms
        "min_ms", // min time in ms
        "max_ms", // max time in ms
        "std_ms", // standard time in ms

        "bytes",   // bytes used for bandwidth calculation
        "bw_GB_s", // Effective bandwidth

        "max_abs_error", "max_rel_error", "correct"};

    fmt::print(out, "{}\n", fmt::join(headers, ","));
}

void write_csv_row(std::ofstream& out, const TransposeResult& r)
{
    fmt::print(out,
               "{},{},{},{},{},"
               "{},{},{},{},{},{},"
               "{},{:.6f},{:.6f},{:.6f},{:.3f},"
               "{},{:.3f},"
               "{:.6e},{:.6e},{}\n",
               r.kernel, r.version, r.mode, r.dtype, r.size_of_dtype,

               r.shape.rows, r.shape.cols, r.block_size.x, r.block_size.y, r.grid_size.x,
               r.grid_size.y,

               r.repeat, r.time_stats.avg_ms, r.time_stats.min_ms, r.time_stats.max_ms,
               r.time_stats.std_ms,

               r.bytes, r.bw_GB_s,

               r.max_abs_error, r.max_rel_error, (r.correct ? "true" : "false"));
}

TransposeResult bench_reduction_cpu_serial(const float* mat_host, float* matT_host,
                                           MatrixShape shape, int repeat, CpuTimer& cpu_timer)
{
    TransposeResult r = make_transpose_result("transpose", "cpu_serial", "cpu", FP32_DTYPE, shape,
                                              dim3(0, 0, 0), dim3(0, 0, 0), repeat);

    r.time_stats = benchmark_cpu(
        [&]() { transpose_cpu(mat_host, matT_host, shape.rows, shape.cols); }, repeat, cpu_timer);

    r.bytes   = num_bytes_transpose(shape);
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    r.correct = true;

    return r;
}

TransposeResult bench_cpu_copy_baseline(const float* mat_host, float* mat_copy_host,
                                        MatrixShape shape, int repeat, CpuTimer& cpu_timer)
{
    TransposeResult r =
        make_transpose_result("transpose", "cpu_serial", "copy_baseline", FP32_DTYPE, shape,
                              dim3(0, 0, 0), dim3(0, 0, 0), repeat);

    r.time_stats = benchmark_cpu(
        [&]() { copy_cpu_baseline(mat_host, mat_copy_host, shape.num_elem()); }, repeat, cpu_timer);

    r.bytes   = shape.bytes() * 2;
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    r.correct = true;

    return r;
}

TransposeResult bench_reduction_cpu_omp(const float* mat_host, float* matT_host, MatrixShape shape,
                                        int repeat, const float* ref, CpuTimer& cpu_timer)
{
    TransposeResult r = make_transpose_result("transpose", "cpu_omp", "cpu", FP32_DTYPE, shape,
                                              dim3(0, 0, 0), dim3(0, 0, 0), repeat);

    r.time_stats =
        benchmark_cpu([&]() { transpose_cpu_omp(mat_host, matT_host, shape.rows, shape.cols); },
                      repeat, cpu_timer);

    r.bytes   = num_bytes_transpose(shape);
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    auto check = check_array_close(matT_host, ref, shape.num_elem());

    r.correct       = check.correct;
    r.max_abs_error = check.max_abs_error;
    r.max_rel_error = check.max_rel_error;

    return r;
}

TransposeResult bench_transpose_h2d(const std::string& kernel, const std::string& version,
                                    const std::string& mode, dim3 block_size, dim3 grid_size,
                                    const float* mat_host, float* mat_dev, MatrixShape shape,
                                    int repeat, CpuTimer& cpu_timer)
{
    TransposeResult r = make_transpose_result(kernel, version, mode, FP32_DTYPE, shape,
                                              dim3(0, 0, 0), dim3(0, 0, 0), repeat);

    r.bytes      = shape.bytes();
    r.time_stats = benchmark_cpu(
        [&]() { gpu::cuda_check(cudaMemcpy(mat_dev, mat_host, shape.bytes(), cudaMemcpyDefault)); },
        repeat, cpu_timer);

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    r.correct = true;

    return r;
}

template <typename LaunchKernel, typename ReadResult>
TransposeResult bench_transpose_kernel_only(const std::string& kernel, const std::string& version,
                                            const std::string& mode, MatrixShape shape,
                                            dim3 block_size, dim3 grid_size, int repeat, float* ref,
                                            LaunchKernel launch_kernel, ReadResult read_result,
                                            CudaTimer& cuda_timer)
{
    TransposeResult r = make_transpose_result(kernel, version, mode, FP32_DTYPE, shape, block_size,
                                              grid_size, repeat);

    r.bytes = num_bytes_transpose(shape);

    r.time_stats = benchmark_cuda_kernel([&]() { launch_kernel(); }, repeat, cuda_timer);

    // Wait kernel to finish execution
    gpu::cuda_check(cudaDeviceSynchronize());

    CheckResult check = read_result();
    r.correct         = check.correct;
    r.max_abs_error   = check.max_abs_error;
    r.max_rel_error   = check.max_rel_error;

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    return r;
}

TransposeResult bench_transpose_d2h(const std::string& kernel, const std::string& version,
                                    const std::string& mode, dim3 block_size, dim3 grid_size,
                                    const float* matT_dev, float* matT_host, MatrixShape shape,
                                    int repeat, CpuTimer& cpu_timer)
{
    TransposeResult r = make_transpose_result(kernel, version, mode, FP32_DTYPE, shape,
                                              dim3(0, 0, 0), dim3(0, 0, 0), repeat);

    r.bytes      = shape.bytes();
    r.time_stats = benchmark_cpu(
        [&]()
        { gpu::cuda_check(cudaMemcpy(matT_host, matT_dev, shape.bytes(), cudaMemcpyDefault)); },
        repeat, cpu_timer);

    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    r.correct = true;

    return r;
}

TransposeResult bench_cuda_copy_baseline(const float* mat_dev, float* mat_copy_dev,
                                         MatrixShape shape, int repeat, CudaTimer& cuda_timer)
{
    dim3 block_size{256};
    dim3 grid_size_copy_baseline{
        static_cast<unsigned int>((shape.num_elem() + block_size.x - 1) / block_size.x)};

    TransposeResult r = bench_transpose_kernel_only(
        "tranpose", "cuda_copy", "copy_baseline", shape, block_size, grid_size_copy_baseline,
        repeat, nullptr,
        [&]()
        {
            copy_cuda_baseline<<<grid_size_copy_baseline, block_size>>>(mat_dev, mat_copy_dev,
                                                                        shape.num_elem());
        },
        [&]() -> CheckResult { return CheckResult{}; }, cuda_timer);

    r.bytes   = shape.bytes() * 2;
    r.bw_GB_s = bandwidth_GB_s(r.bytes, r.time_stats.avg_ms);

    r.correct = true;

    return r;
}

void run_naive_kernel_benchmark(std::ofstream& out, const BenchmarkConfig& config, float* mat_dev,
                                float* matT_dev, MatrixShape shape, dim3 block_size, float* ref,
                                CudaTimer& cuda_timer, CpuTimer& cpu_timer)
{
    const std::string version = "cuda_naive";
    dim3 grid_size_naive_kernel{(shape.cols + block_size.x - 1) / block_size.x,
                                (shape.rows + block_size.y - 1) / block_size.y};

    host_ptr matT_host(cuda_malloc_host<float>(shape.num_elem()));

    TransposeResult result_atomic = bench_transpose_kernel_only(
        config.kernel, version, "cuda_kernel", shape, block_size, grid_size_naive_kernel,
        config.repeat_kernel, ref,
        [&]()
        {
            // Launch kernel
            launch_transpose_naive(mat_dev, matT_dev, shape.rows, shape.cols);
        },
        [&]() -> CheckResult
        {
            gpu::cuda_check(
                cudaMemcpy(matT_host.get(), matT_dev, shape.bytes(), cudaMemcpyDefault));

            return check_array_close(matT_host.get(), ref, shape.num_elem());
        },
        cuda_timer);

    write_csv_row(out, result_atomic);

    // GPU allocated memory is released automatically by DeviceBuffer.
}

void run_block_size_sweep(std::ofstream& out, const BenchmarkConfig& config, float* mat_dev,
                          float* matT_dev, MatrixShape shape, float* ref, CudaTimer& cuda_timer,
                          CpuTimer& cpu_timer)
{
    // CDUA kernel sweep over block sizes
    for (auto block_size : config.block_sizes)
    {
        run_naive_kernel_benchmark(out, config, mat_dev, matT_dev, shape, block_size, ref,
                                   cuda_timer, cpu_timer);
    }
}

void run_single_size_benchmark(std::ofstream& out, const BenchmarkConfig& config, MatrixShape shape,
                               CudaTimer& cuda_timer, CpuTimer& cpu_timer)
{
    size_t n     = shape.num_elem();
    size_t bytes = shape.bytes();

    fmt::print("Running transpose benchmark for shape = {}x{} (Matrix size = {} MiB)\n", shape.rows,
               shape.cols, static_cast<double>(bytes) / static_cast<double>(1ULL << 20));

    // Pointers to host memory
    host_ptr mat_host(cuda_malloc_host<float>(n));
    host_ptr matT_host(cuda_malloc_host<float>(n));

    // Initialize vectors on the host
    init_matrix(mat_host.get(), shape);

    // CPU copy baseline
    host_ptr mat_copy_host(cuda_malloc_host<float>(n));
    TransposeResult result_cpu_copy_baseline = bench_cpu_copy_baseline(
        mat_host.get(), mat_copy_host.get(), shape, config.repeat_copy, cpu_timer);

    write_csv_row(out, result_cpu_copy_baseline);

    // CPU serial benchmark
    TransposeResult result_cpu_serial = bench_reduction_cpu_serial(
        mat_host.get(), matT_host.get(), shape, config.repeat_cpu, cpu_timer);

    write_csv_row(out, result_cpu_serial);

    auto ref = std::make_unique<float[]>(n);
    std::copy(matT_host.get(), matT_host.get() + n, ref.get());

    auto check = check_array_close(matT_host.get(), ref.get(), n);

    // CPU OMP benchmark
    TransposeResult result_cpu_omp = bench_reduction_cpu_omp(
        mat_host.get(), matT_host.get(), shape, config.repeat_cpu, ref.get(), cpu_timer);

    write_csv_row(out, result_cpu_omp);

    // Allocate device memory
    DeviceBuffer<float> mat_dev(n);
    DeviceBuffer<float> matT_dev(n);

    // H2D benchmark
    TransposeResult result_h2d =
        bench_transpose_h2d(config.kernel, "all_gpu_version", "h2d", {0, 0, 0}, {0, 0, 0},
                            mat_host.get(), mat_dev.get(), shape, config.repeat_copy, cpu_timer);

    write_csv_row(out, result_h2d);

    // GPU copy baseline
    DeviceBuffer<float> mat_copy_dev(n);
    TransposeResult result_cuda_copy_baseline = bench_cuda_copy_baseline(
        mat_dev.get(), mat_copy_dev.get(), shape, config.repeat_copy, cuda_timer);
    write_csv_row(out, result_cuda_copy_baseline);

    // Block size sweep
    run_block_size_sweep(out, config, mat_dev.get(), matT_dev.get(), shape, ref.get(), cuda_timer,
                         cpu_timer);

    // benchmark d2h
    TransposeResult result_d2h_atomic =
        bench_transpose_d2h(config.kernel, "all_gpu_version", "d2h", {0, 0, 0}, {0, 0, 0},
                            matT_dev.get(), matT_host.get(), shape, config.repeat_copy, cpu_timer);
    write_csv_row(out, result_d2h_atomic);

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

    // Sweep over matrix shape
    for (MatrixShape shape : config.shapes)
    {
        run_single_size_benchmark(out, config, shape, cuda_timer, cpu_timer);
    }
}

int main()
{
    BenchmarkConfig config{};
    run_benchmark(config);
}