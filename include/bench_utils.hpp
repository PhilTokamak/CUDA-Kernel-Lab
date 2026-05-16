#pragma once
#include <vector>
#include <numeric>

#include "timer.hpp"

struct TimeStats
{
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double std_ms = 0.0;
};

[[nodiscard]]
inline TimeStats compute_time_stats(const std::vector<double>& samples)
{
    TimeStats s;

    if (samples.empty())
    {
        return s;
    }

    s.min_ms = *std::min_element(samples.begin(), samples.end());
    s.max_ms = *std::max_element(samples.begin(), samples.end());

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.avg_ms = sum / static_cast<double>(samples.size());

    double var = 0.0;
    for (double x : samples) {
        double d = x - s.avg_ms;
        var += d * d;
    }
    var /= static_cast<double>(samples.size());

    s.std_ms = std::sqrt(var);

    return s;
}

template <typename Func>
[[nodiscard]]
TimeStats benchmark_cpu(Func&& func, int repeat, CpuTimer& cpu_timer)
{
    std::vector<double> samples;
    samples.reserve(repeat);

    for (int r = 0; r < repeat; ++r)
    {
        cpu_timer.start();

        func();

        double ms = cpu_timer.stop();
        samples.push_back(ms);
    }

    return compute_time_stats(samples);
}

template <typename Func>
[[nodiscard]]
TimeStats benchmark_cuda_kernel(Func&& launch_func, int repeat, CudaTimer& cuda_timer)
{
    std::vector<double> samples;
    samples.reserve(repeat);

    for (int r = 0; r < repeat; ++r)
    {
        cuda_timer.start();

        launch_func();

        double ms = cuda_timer.stop();
        samples.push_back(ms);
    }

    return compute_time_stats(samples);
}

inline double bandwidt_GB_s(size_t bytes, double ms)
{
    return static_cast<double>(bytes) / (ms / 1000.0 * 1e9);
}

inline double gflops(size_t n_flop, double ms)
{
    return static_cast<double>(n_flop) / (ms / 1000.0 * 1e9);
}