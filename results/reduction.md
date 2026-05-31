<!--
  AUTO-GENERATED FILE. DO NOT EDIT DIRECTLY.

  Edit this template instead:

    results/templates/reduction_report.md.j2

  Regenerate with:

    python scripts/generate_reduction_report.py
  -->

  # Reduction Benchmark Report

  ## 1. Goal

  This benchmark studies several CUDA implementations of a one-dimensional FP32 reduction:

  ```cpp
  sum = x[0] + x[1] + ... + x[n - 1]
  ```

The goal is to compare different reduction strategies in terms of:

- kernel execution time
- post-H2D time
- end-to-end time
- effective bandwidth
- correctness
- sensitivity to block size
- scaling with problem size

The benchmark data source is: [`reduction.csv`](data/reduction.csv)

The tested implementations in this run are:


  - `cpu_serial`
  
  - `cpu_omp`
  
  - `cuda_atomic_add`
  
  - `cuda_block_shared_mem`
  
  - `cuda_grid_stride_block_shared`
  
  - `cuda_warp_schuffle`
  
  - `cuda_grid_stride_two_pass`
  
  - `cuda_multi_pass`
  

---

## 2. Timing Definitions

For reduction, different GPU kernels may produce different kinds of outputs.

Some kernels directly produce the final scalar result, while others only produce partial sums that still need to be copied back to the CPU and finalized.

Therefore, the benchmark distinguishes between several timing components:

```text
  H2D time        = host-to-device copy of input vector
  Kernel time     = CUDA kernel execution time
  D2H time        = device-to-host copy of scalar or partial sums
  CPU finalize    = CPU reduction over partial sums, if needed
  Post-H2D time   = kernel + D2H + CPU finalize
  E2E time        = H2D + Post-H2D time
  ```

For partial-output kernels, kernel-only time is not sufficient to represent full algorithm performance. The main comparison therefore uses `Post-H2D time` and `E2E time`.

---

## 3. Metric Definitions

For reduction, the useful input data size is defined as:

```text
  useful bytes = N * sizeof(dtype)
  ```

For this benchmark:

```text
  dtype = fp32
  sizeof(dtype) = 4 bytes
  ```

The useful floating-point work is estimated as:

```text
  useful FLOPs = N - 1
  ```

The effective bandwidth numbers are algorithm-level throughput estimates:

```text
  Post-H2D BW = useful bytes / Post-H2D time
  E2E BW      = useful bytes / E2E time
  ```

These are not direct measurements of physical DRAM traffic. They are useful throughput metrics for comparing implementations.




---

## 4. Main Benchmark Setup

The main benchmark uses:

```text
  N = 67,108,864
  DType = fp32
  Single vector size = 256.00 MiB
  Input pattern = ones
  ```

The CPU serial baseline takes:

```text
  CPU serial time = 113.094 ms
  CPU effective bandwidth = 2.37 GB/s
  CPU GFLOP/s = 0.59
  ```

---

## 5. Main Results at Large N

For `N = 67,108,864`, the fastest GPU implementation in terms of **post-H2D time** is:

```text
  cuda_warp_schuffle
  ```

with:

```text
  block size = 1024
  grid size  = 432
  post-H2D time = 0.254 ms
  post-H2D speedup = 444.46x
  post-H2D effective bandwidth = 1054.96 GB/s
  ```

The fastest GPU implementation in terms of **end-to-end time** is:

```text
  cuda_warp_schuffle
  ```

with:

```text
  block size = 1024
  grid size  = 432
  E2E time = 10.827 ms
  E2E speedup = 10.45x
  E2E effective bandwidth = 24.79 GB/s
  ```

---

## 6. Interpretation of Main Results


  ### 6.1 Atomic reduction

The `cuda_atomic_add` implementation has:

```text
  kernel time = 161.768 ms
  post-H2D time = 161.776 ms
  E2E time = 172.349 ms
  post-H2D speedup = 0.70x
  E2E speedup = 0.66x
  ```

This version is expected to perform poorly because many GPU threads contend for the same global scalar output through:

```cpp
  atomicAdd(out, x[idx]);
  ```

Although the operation launches many GPU threads, updates to the same global output location are heavily serialized.


  In this benchmark, `cuda_atomic_add` is slower than the CPU serial baseline in E2E terms.
  
  


  ### 6.2 Block-shared-memory reduction

The `cuda_block_shared_mem` implementation reduces values within each block using shared memory and writes one partial sum per block.

For the main benchmark size, the selected configuration is:

```text
  block size = 128
  grid size  = 524288
```

It achieves:

```text
  kernel time = 0.535 ms
  D2H time = 0.088 ms
  CPU finalize time = 0.877 ms
  post-H2D time = 1.501 ms
  E2E time = 12.073 ms
  post-H2D speedup = 75.36x
  E2E speedup = 9.37x
```

This is much faster than the atomic version because most of the summation is performed locally inside each block, avoiding global atomic contention.

However, this version still produces many partial sums.
These partial sums must be copied back to the CPU and finalized there,
so D2H and CPU-finalization costs are part of the full algorithm time.
  



### 6.3 Grid-stride shared-memory reduction

#### Grid size selection

Unlike the basic block-shared-memory reduction, the grid-stride version does not use:

```cpp
grid_size = ceil(N / block_size)
```

Instead, it uses a fixed grid size based on the number of SMs on the GPU:

```cpp
grid_size = num_sms * blocks_per_sm
```

In this benchmark:

```text
num_sms = 108
blocks_per_sm = 4
grid_size = 432
```

The reason is that each block uses a grid-stride loop to process multiple input
elements:

```cpp
for (idx = global_thread_id; idx < n; idx += gridDim.x * blockDim.x) {
    local_sum += x[idx];
}
```

Therefore, the number of partial sums is equal to `grid_size`, not
`ceil(N / block_size)`.

This significantly reduces the number of partial sums that must be copied back
to the CPU and finalized there.

For example, at `N = 67,108,864`:

```text
basic block reduction:
    block size = 128
    grid size  = 524288
    partial sums = 524288

grid-stride reduction:
    block size = 1024
    grid size  = 432
    partial sums = 432
```

This reduction in partial sums explains why the grid-stride version has much
smaller D2H and CPU-finalization overhead.

Note that this is a heuristic choice, not a guaranteed optimum. A more complete benchmark
could sweep `blocks_per_sm` to study the trade-off between available parallelism
and the number of partial sums.

---

#### Grid-stride version results interpretation

The `cuda_grid_stride_block_shared` implementation uses a grid-stride loop so that each block processes multiple input elements.

For the main benchmark size, the selected configuration is:

```text
  block size = 1024
  grid size  = 432
  ```

It achieves:

```text
  kernel time = 0.247 ms
  D2H time = 0.008 ms
  CPU finalize time = 0.001 ms
  post-H2D time = 0.256 ms
  E2E time = 10.828 ms
  post-H2D speedup = 442.48x
  E2E speedup = 10.44x
  ```

This implementation reduces the number of partial sums compared with the basic block-level implementation. As a result, both D2H transfer time and CPU finalization time become much smaller.




  

---



### 6.4 Grid-stride reduction with warp-shuffle final reduction

The `cuda_warp_schuffle` implementation is based on the grid-stride shared-memory reduction, but optimizes the final part of the block-local reduction.

The high-level structure is still:

```text
grid-stride accumulation
    -> shared-memory reduction
    -> warp-shuffle final reduction
    -> one partial sum per block
```

Compared with the basic `cuda_grid_stride_block_shared` version, the main difference is how the last stage of the block-local reduction is performed.

In the basic grid-stride shared-memory version, the reduction continues through shared memory until only one value remains:

```cpp
for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        local_array[tid] += local_array[tid + stride];
    }
    __syncthreads();
}
```

In the warp-shuffle version, the shared-memory reduction stops once 64 values remain:

```cpp
for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
        local_array[tid] += local_array[tid + stride];
    }
    __syncthreads();
}
```

The final reduction is then performed using shuffle instructions:

```cpp
if (tid < 32) {
    float warp_val = local_array[tid];
    warp_val += local_array[tid + 32];

    warp_val += __shfl_down_sync(0xffffffff, warp_val, 16);
    warp_val += __shfl_down_sync(0xffffffff, warp_val, 8);
    warp_val += __shfl_down_sync(0xffffffff, warp_val, 4);
    warp_val += __shfl_down_sync(0xffffffff, warp_val, 2);
    warp_val += __shfl_down_sync(0xffffffff, warp_val, 1);

    if (tid == 0) {
        partial_sum_out[blockIdx.x] = warp_val;
    }
}
```

The purpose of this optimization is to avoid the final rounds of shared-memory traffic and
block-wide synchronization. The shuffle operations allow values in the final reduction stage
to be exchanged directly inside the participating execution group.

The mask:

```cpp
0xffffffff
```

is used because this final shuffle sequence is intended to involve the full set of 32 participating
threads in that stage. In this implementation, the shuffle code is guarded by:

```cpp
if (tid < 32)
```

so the full mask is appropriate for this reduction step. If the final shuffle stage involved only
a partial set of participating threads, then a more precise active mask would be needed.

For the main benchmark size, the selected configuration is:

```text
block size = 1024
grid size  = 432
```

It achieves:

```text
kernel time = 0.246 ms
D2H time = 0.008 ms
CPU finalize time = 0.001 ms
post-H2D time = 0.254 ms
E2E time = 10.827 ms
post-H2D speedup = 444.46x
E2E speedup = 10.45x
```


Compared with the basic grid-stride shared-memory version:

```text
basic grid-stride kernel time = 0.247 ms
warp-shuffle kernel time      = 0.246 ms
```

The kernel-time difference is approximately:

```text
1.000 us
```

The post-H2D times are:

```text
basic grid-stride post-H2D time = 0.256 ms
warp-shuffle post-H2D time      = 0.254 ms
```

The post-H2D difference is approximately:

```text
2.000 us
```


In this benchmark, the improvement from the warp-shuffle final reduction is small, typically
on the order of a few microseconds. This is expected because the optimization only affects the
final part of the block-local reduction. The dominant cost is still the grid-stride accumulation
over the full input array.

Nevertheless, this version is useful because it represents a standard CUDA reduction optimization:
use shared memory for the larger block-level reduction, then switch to shuffle instructions for the final
reduction stage.



---



### 6.5 Two-pass GPU reduction

The `cuda_grid_stride_two_pass` implementation performs the reduction entirely on the GPU.

Conceptually, it works as follows:

```text
pass 1:
    input x[0:N] -> partial_sums[0:num_blocks]

pass 2:
    partial_sums[0:num_blocks] -> final scalar sum
```

The first pass is based on the grid-stride reduction strategy: each block processes multiple input
elements and writes one partial sum.

The second pass reduces the partial sums into a single scalar result on the GPU.
In this implementation, the reduction stages use the warp-shuffle optimized final reduction
instead of reducing all the way down through shared memory.

This means that, unlike the `cuda_grid_stride_block_shared` and `cuda_warp_schuffle` versions
with CPU finalization, the final scalar result is produced directly on the GPU.

For the main benchmark size, the selected configuration is:

```text
block size = 1024
grid size  = 432
```

It achieves:

```text
kernel time = 0.250 ms
D2H time = 0.008 ms
CPU finalize time = 0.000 ms
post-H2D time = 0.258 ms
E2E time = 10.831 ms
post-H2D speedup = 438.45x
E2E speedup = 10.44x
```


Compared with the basic grid-stride shared-memory version with CPU finalization:

```text
basic grid-stride post-H2D time = 0.256 ms
two-pass post-H2D time          = 0.258 ms
```

The post-H2D difference is approximately:

```text
2.000 us
```



Compared with the warp-shuffle CPU-finalized version:

```text
warp-shuffle CPU-finalized post-H2D time = 0.254 ms
two-pass GPU-resident post-H2D time      = 0.258 ms
```

The post-H2D difference is approximately:

```text
4.000 us
```


In the current benchmark, the two-pass version is very close to the fastest CPU-finalized grid-stride variants.
The remaining difference is on the order of microseconds.

This is expected. The first pass dominates the cost because it reads and accumulates the full input array.
The second pass only reduces a small partial-sum array, so optimizing it helps, but the total effect is limited.

The main value of the two-pass version is not necessarily that it is much faster in this standalone benchmark.
Its main value is that the final scalar result is produced on the GPU. This is important for GPU-resident
workflows where the reduction result is consumed by later GPU kernels.

Examples include:

```text
norm computation
loss reduction
residual/error calculation
adaptive timestep criteria
normalization statistics
iterative solver convergence checks
```

If the reduction result stays on the GPU, then the scalar D2H transfer can be avoided entirely.
In that case, the relevant time is closer to the GPU computation time rather than the host-result E2E time.

Therefore, even if the two-pass implementation is only approximately as fast as the hybrid CPU-finalized
version in this isolated benchmark, it represents a more GPU-native execution model.




---




### 6.6 Multi-pass GPU reduction

The `cuda_multi_pass` implementation generalizes the two-pass idea.

Instead of assuming that the first pass produces a small enough partial-sum array to be reduced in
one additional kernel, it repeatedly applies a block-level reduction until only one value remains.

Conceptually:

```text
input x[0:N]
    -> partial_sums_level_1
    -> partial_sums_level_2
    -> ...
    -> final scalar sum
```

Each pass reduces the current input array into a smaller partial-sum array. The process continues
until only a single value remains.

In the current implementation, the multi-pass version does not use the warp-shuffle final reduction
optimization. This is intentional for now: the multi-pass path is mainly included as a general
fallback for cases where the partial-sum array is too large to be reduced in a single second pass.

For the current grid-stride configuration, the number of first-pass partial sums is already small,
so the multi-pass version is rarely needed in normal runs.

For the main benchmark size, the selected configuration is:

```text
block size = 1024
grid size  = 432
```

It achieves:

```text
kernel time = 0.254 ms
D2H time = 0.008 ms
CPU finalize time = 0.000 ms
post-H2D time = 0.262 ms
E2E time = 10.834 ms
post-H2D speedup = 432.01x
E2E speedup = 10.44x
```

The value of this version is mainly architectural:

1. It supports larger or more general intermediate reduction sizes.
2. It keeps the reduction process on the GPU.
3. It provides a more general reduction structure.
4. It can be optimized later with the same warp-shuffle final reduction strategy.

In this benchmark, however, the multi-pass version is not expected to outperform the specialized
two-pass version, because the current grid-stride configuration already produces only a small number
of partial sums.




---




### 6.6 OpenMP CPU reduction

The `cpu_omp` implementation is an optimized parallel CPU baseline using OpenMP.

Conceptually, it parallelizes the reduction across CPU threads:

```cpp
#pragma omp parallel for reduction(+:sum)
for (std::size_t i = 0; i < n; ++i) {
    sum += x[i];
}
```

Each thread accumulates a private partial sum, and OpenMP combines these partial sums at the end of
the parallel region. This avoids a shared global accumulator and gives the CPU implementation access
to much more memory bandwidth than a single-threaded loop.

For the main benchmark size:

```text
N = 67,108,864
```

the `cpu_omp` implementation achieves:

```text
time = 6.396 ms
effective bandwidth = 41.97 GB/s
GFLOP/s = 10.49
speedup vs cpu_serial = 17.68x
```

This is a very strong CPU result. With 18 threads, the OpenMP version reaches almost ideal scaling
relative to the single-threaded CPU baseline.

In fact, its E2E speedup can exceed the E2E speedup of the fastest GPU version.


For comparison, the fastest GPU E2E implementation is:

```text
cuda_warp_schuffle
```

with:

```text
GPU E2E speedup = 10.45x
GPU E2E time    = 10.827 ms
```

while `cpu_omp` achieves:

```text
CPU OpenMP speedup = 17.68x
CPU OpenMP time    = 6.396 ms
```



This does not mean that the GPU kernels are slow. Rather, it highlights the difference between
post-H2D and E2E performance.

For the fastest GPU version:

```text
post-H2D speedup = 444.46x
E2E speedup      = 10.45x
```

The post-H2D speedup is much larger because it measures the case where the input data is already
resident on the GPU. The E2E speedup is limited by the H2D transfer of the entire input vector.

In this benchmark, the GPU E2E speedup is limited to roughly the range imposed by host-device
transfer bandwidth, while the OpenMP version operates directly on CPU memory and does not pay any
PCIe/NVLink transfer cost.

This comparison is important:

- If the data starts and ends on the CPU, a strong multi-threaded CPU baseline can be very competitive.
- If the data is already on the GPU, the GPU reduction is much faster than even the optimized CPU version.
- If the reduction result will be consumed by later GPU kernels, GPU-resident reductions such as the two-pass or multi-pass versions become more meaningful.
- GPU-vs-CPU comparisons must always specify whether host-device transfer is included.

Thus, `cpu_omp` is not merely a stronger baseline; it changes the interpretation of the GPU results.
Compared with `cpu_serial`, the GPU post-H2D performance can show hundreds of times speedup.
Compared with a well-optimized CPU OpenMP baseline and including H2D transfer,
the advantage is much smaller and may disappear for a standalone reduction.


---

## 7. Host-Device Transfer Effects

For the largest benchmark size, the fastest E2E implementation is:

```text
  cuda_warp_schuffle
  ```

Its timing breakdown is:

```text
  H2D time = 10.573 ms
  kernel time = 0.246 ms
  D2H time = 0.008 ms
  CPU finalize time = 0.001 ms
  E2E time = 10.827 ms
  ```

The H2D fraction of the E2E time is approximately:

```text
  97.6%
  ```

This shows an important GPU performance principle:

> For low-arithmetic-intensity operations such as reduction, host-device transfer can dominate end-to-end runtime.

If the input data is already resident on the GPU, the relevant metric is post-H2D performance.
If the input starts on the CPU, the relevant metric is E2E performance.

---

## 8. Block Size Sweep Analysis

The block-size sweep measures kernel-stage performance at:

```text
N = 67,108,864
```

These results should be interpreted as kernel-stage diagnostics.
For partial-output reduction kernels, the kernel stage alone does not include D2H transfer or CPU finalization.


### 1. `cuda_atomic_add`

For `cuda_atomic_add`, the best kernel-stage configuration is:

```text
block size = 256
grid size  = 262144
kernel time = 161.768 ms
kernel effective bandwidth = 1.66 GB/s
kernel GFLOP/s = 0.41
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 161.768 ms with block size = 256
max kernel time = 161.782 ms with block_size = 64
max/min ratio   = 1.00x
```


The atomic version shows limited sensitivity to block size. This suggests that
the dominant bottleneck is global atomic contention rather than block-size-dependent occupancy or memory coalescing.



### 2. `cuda_block_shared_mem`

For `cuda_block_shared_mem`, the best kernel-stage configuration is:

```text
block size = 128
grid size  = 524288
kernel time = 0.535 ms
kernel effective bandwidth = 505.60 GB/s
kernel GFLOP/s = 125.42
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.535 ms with block size = 128
max kernel time = 0.849 ms with block_size = 64
max/min ratio   = 1.59x
```


The shared-memory block reduction has a clear block-size optimum. Very small blocks create too many partial sums,
while very large blocks may increase synchronization cost or reduce scheduling flexibility.



### 3. `cuda_grid_stride_block_shared`

For `cuda_grid_stride_block_shared`, the best kernel-stage configuration is:

```text
block size = 1024
grid size  = 432
kernel time = 0.247 ms
kernel effective bandwidth = 1087.02 GB/s
kernel GFLOP/s = 271.75
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.247 ms with block size = 1024
max kernel time = 1.236 ms with block_size = 64
max/min ratio   = 5.00x
```


The grid-stride version benefits from larger block sizes up to the best observed configuration.
This is likely because each block performs more local work before writing a partial sum, reducing overhead while maintaining enough parallelism.



### 4. `cuda_warp_schuffle`

For `cuda_warp_schuffle`, the best kernel-stage configuration is:

```text
block size = 1024
grid size  = 432
kernel time = 0.246 ms
kernel effective bandwidth = 1091.83 GB/s
kernel GFLOP/s = 272.96
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.246 ms with block size = 1024
max kernel time = 1.233 ms with block_size = 64
max/min ratio   = 5.02x
```




### 5. `cuda_grid_stride_two_pass`

For `cuda_grid_stride_two_pass`, the best kernel-stage configuration is:

```text
block size = 1024
grid size  = 432
kernel time = 0.250 ms
kernel effective bandwidth = 1073.05 GB/s
kernel GFLOP/s = 268.26
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.250 ms with block size = 1024
max kernel time = 1.237 ms with block_size = 64
max/min ratio   = 4.94x
```


The grid-stride version benefits from larger block sizes up to the best observed configuration.
This is likely because each block performs more local work before writing a partial sum, reducing overhead while maintaining enough parallelism.



### 6. `cuda_multi_pass`

For `cuda_multi_pass`, the best kernel-stage configuration is:

```text
block size = 1024
grid size  = 432
kernel time = 0.254 ms
kernel effective bandwidth = 1056.90 GB/s
kernel GFLOP/s = 264.22
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.254 ms with block size = 1024
max kernel time = 1.243 ms with block_size = 64
max/min ratio   = 4.89x
```





---

## 9. Kernel-Only Timing Caveat

The block-size sweep reports kernel-stage timing and kernel-stage throughput.

For kernels that only produce partial sums, such as shared-memory block reductions, kernel-only time does not represent full algorithm time.

The full algorithm must also include:

```text
  D2H partial-sum copy
  CPU finalization
  ```

Therefore, kernel-only speedups in the block-size sweep should be interpreted as diagnostic metrics, not as complete algorithm speedups.

---

## 10. Problem Size Scaling Analysis

The problem-size sweep evaluates how each implementation behaves as the input size grows.


### 1. `cuda_atomic_add`

For `cuda_atomic_add`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.647 ms
E2E time = 0.703 ms
post-H2D speedup = 0.68x
E2E speedup = 0.63x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 262144
post-H2D time = 647.167 ms
E2E time = 689.498 ms
post-H2D speedup = 0.70x
E2E speedup = 0.66x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 1.03
E2E speedup growth      = 1.05
```


The atomic version scales poorly because increasing the input size also increases the number
of contending atomic updates to the same global scalar.



### 2. `cuda_block_shared_mem`

For `cuda_block_shared_mem`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.020 ms
E2E time = 0.076 ms
post-H2D speedup = 22.24x
E2E speedup = 5.82x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 128
grid size = 2097152
post-H2D time = 5.957 ms
E2E time = 48.288 ms
post-H2D speedup = 75.94x
E2E speedup = 9.37x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 3.41
E2E speedup growth      = 1.61
```


The shared-memory block version improves significantly over CPU serial,
but as `N` grows it also creates more partial sums, which increases D2H and CPU-finalization overhead.



### 3. `cuda_grid_stride_block_shared`

For `cuda_grid_stride_block_shared`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.016 ms
E2E time = 0.072 ms
post-H2D speedup = 27.56x
E2E speedup = 6.13x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 0.994 ms
E2E time = 43.325 ms
post-H2D speedup = 455.16x
E2E speedup = 10.44x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 16.52
E2E speedup growth      = 1.70
```


The grid-stride version scales well in post-H2D terms because the amount of GPU work
increases while the number of partial sums remains relatively small.



### 4. `cuda_warp_schuffle`

For `cuda_warp_schuffle`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.015 ms
E2E time = 0.071 ms
post-H2D speedup = 28.56x
E2E speedup = 6.18x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 0.991 ms
E2E time = 43.322 ms
post-H2D speedup = 456.41x
E2E speedup = 10.44x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 15.98
E2E speedup growth      = 1.69
```




### 5. `cuda_grid_stride_two_pass`

For `cuda_grid_stride_two_pass`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.018 ms
E2E time = 0.073 ms
post-H2D speedup = 24.90x
E2E speedup = 5.99x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 0.994 ms
E2E time = 43.326 ms
post-H2D speedup = 454.93x
E2E speedup = 10.44x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 18.27
E2E speedup growth      = 1.74
```


The grid-stride version scales well in post-H2D terms because the amount of GPU work
increases while the number of partial sums remains relatively small.



### 6. `cuda_multi_pass`

For `cuda_multi_pass`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.021 ms
E2E time = 0.077 ms
post-H2D speedup = 20.70x
E2E speedup = 5.71x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 1.001 ms
E2E time = 43.332 ms
post-H2D speedup = 452.06x
E2E speedup = 10.44x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 21.83
E2E speedup growth      = 1.83
```







### CPU Optimized Problem Size Scaling



### `cpu_omp`

At the largest tested size:

```text
N = 268,435,456
CPU serial time = 452.375 ms
CPU optimized time = 40.396 ms
CPU optimized bandwidth = 26.58 GB/s
speedup = 11.20x
```




---

## 11. Correctness

All implementations included in the result tables report correctness status.



The input pattern used in this report is:

```text
input_pattern = ones
```

  
For the `ones` input pattern, every input element is:

```cpp
x[i] = 1.0f;
```

Mathematically, the exact reduction result should be:

```text
sum = N
```

FP32 can exactly represent consecutive integers only up to:

```text
2^24 (= 1 << 24) = 16777216
```

Beyond this value, adding `1.0f` to an FP32 accumulator may no longer change the accumulator. For example:

```cpp
float s = 16777216.0f;
s += 1.0f;  // s may still be 16777216.0f
```

Therefore, for `N > 2^24`, the `cpu_serial` and `AtomicAdd` versions are no longer reliable for an all-ones reduction.

This is especially important for parallel reduction kernels. A tree-based or block-based parallel
reduction changes the order of additions, and may produce a result closer to the mathematical
value `N` than the serial FP32 result.

  


---

## 12. Key Observations

1. **Atomic reduction is simple but usually not performant.**

   The atomic implementation is useful, but it is
   not a good high-performance reduction strategy. All threads contend for
   updates to a shared global scalar, so the computation becomes
   heavily serialized around the atomic operation.

1. **Shared-memory reduction is the first important optimization step.**

   Reducing values locally within each block avoids global atomic
   contention and exposes the tree-structured nature of parallel reduction.
   This is the first version where the implementation starts to use GPU memory
   hierarchy and thread cooperation meaningfully.

1. **The number of partial sums matters.**

   A partial-output reduction kernel is not finished after the GPU kernel returns.
   If it produces many partial sums, those partial sums must either be copied back to
   the CPU and finalized there, or reduced further on the GPU. Therefore, the output
   size of each reduction stage is an important part of the algorithm design.

1. **Grid-stride loops reduce partial-sum overhead.**

   A grid-stride reduction lets each block process multiple input elements. This decouples
   the number of partial sums from `ceil(N / block_size)` and allows the implementation to
   use a fixed, hardware-informed number of blocks. This can greatly reduce D2H transfer and
   CPU-finalization overhead when compared with a basic one-block-per-chunk reduction.

1. **Warp-shuffle final reduction gives a small but measurable improvement.**

   Replacing the final shared-memory reduction steps with shuffle operations reduces
   shared-memory traffic and removes the need for block-wide synchronization in the final reduction stage.
   In this benchmark, the improvement is modest because the dominant cost is still reading
   and accumulating the full input array.

1. **CPU-finalized GPU reductions can be fast but are not fully GPU-resident.**

   A GPU kernel followed by D2H transfer and CPU finalization may perform very well in a
   standalone benchmark. However, it breaks GPU residency: the final scalar is produced on the CPU.
   This matters if the result is needed by later GPU kernels.

1. **Two-pass and multi-pass reductions are important even when they are not faster in this standalone benchmark.**

   Fully GPU-resident reductions produce the final result on the GPU. This is important for
   larger GPU pipelines where the reduction result can be consumed by subsequent GPU kernels without
   returning to the CPU. Their value is architectural as well as performance-related.

1. **Kernel-only speedup can be misleading.**

   For partial-output kernels, kernel time alone does not include all work needed to obtain the
   final scalar result. A fair algorithm-level comparison must include D2H transfer and CPU finalization
   if those stages are required.

1. **Post-H2D and E2E answer different questions.**

   - Post-H2D performance answers: “How fast is reduction if data is already on GPU?”
   - E2E performance answers: “How fast is reduction if data starts on CPU?”

   Both metrics are useful, but they describe different execution scenarios.

1. **A strong CPU baseline changes the interpretation of GPU speedups.**

   Comparing GPU kernels only against a single-threaded CPU baseline can overstate practical speedups.
   A multi-threaded CPU implementation can be very competitive for standalone memory-bound reductions,
   especially when GPU E2E time includes host-device transfer.

1. **Reduction is more about memory movement and execution structure than arithmetic throughput.**

    FP32 reduction performs very little arithmetic per byte of input data. As a result, performance
    is usually governed by memory bandwidth, synchronization, atomic contention, launch overhead,
    partial-sum handling, and host-device transfer rather than raw floating-point throughput.

---

## 13. Current Best Implementation

The current best E2E kernel implementation is:

```text
  cuda_warp_schuffle
  ```

with:

```text
  N = 67,108,864
  block size = 1024
  grid size = 432
  kernel time = 0.246 ms
  D2H time = 0.008 ms
  CPU finalize time = 0.001 ms
  post-H2D time = 0.254 ms
  E2E time = 10.827 ms
  post-H2D speedup = 444.46x
  E2E speedup = 10.45x
  ```



### CPU Optimized Baseline

The fastest optimized CPU implementation at `N = 67,108,864` is:

```text
cpu_omp
```
with
```text
time = 6.396 ms
bandwidth = 41.97 GB/s
speedup vs cpu_serial = 17.68x
```



---

## 14. Limitations

The current benchmark has several limitations:

1. **The benchmark is still a microbenchmark.**

   The reduction is measured as an isolated operation. In real applications,
   reductions are usually part of a larger GPU or CPU pipeline. Whether the
   input is already resident on the GPU, and whether the result is consumed on the
   GPU or CPU, can change the interpretation of the results.

1. **E2E performance depends strongly on host-device transfer.**

   GPU E2E timings include host-to-device input transfer. For a standalone reduction,
   this transfer can dominate runtime. In a GPU-resident workload, this cost may be amortized or avoided entirely.

1. **Some GPU implementations still use CPU finalization.**

   Partial-output kernels that copy partial sums back to the CPU are hybrid algorithms, not
   fully GPU-resident reductions. They can be fast, but they are not always appropriate when
   subsequent computation remains on the GPU.

1. **The fully GPU-resident implementations are not yet highly optimized.**

   Two-pass and multi-pass versions are useful algorithmic baselines, but they may still
   have avoidable overheads such as extra kernel launches, inefficient partial-sum handling,
   or suboptimal memory access patterns.

1. **Only one data type is currently tested.**

   Current data type:

   ```text
   fp32
   ```

---

## 16. Auto-Generated Benchmark Tables

The following tables are generated automatically from the benchmark CSV.

Generated tabls can be found in [`reduction_bench_tables.md`](markdown_tables/reduction_bench_tables.md)