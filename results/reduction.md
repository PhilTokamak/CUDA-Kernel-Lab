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
  N = 16,777,216
  DType = fp32
  Single vector size = 64.00 MiB
  Input pattern = ones
  ```

The CPU serial baseline takes:

```text
  CPU serial time = 28.190 ms
  CPU effective bandwidth = 2.38 GB/s
  CPU GFLOP/s = 0.59
  ```

---

## 5. Main Results at Large N

For `N = 16,777,216`, the fastest GPU implementation in terms of **post-H2D time** is:

```text
  cuda_grid_stride_block_shared
  ```

with:

```text
  block size = 512
  grid size  = 432
  post-H2D time = 0.075 ms
  post-H2D speedup = 374.40x
  post-H2D effective bandwidth = 891.28 GB/s
  ```

The fastest GPU implementation in terms of **end-to-end time** is:

```text
  cuda_grid_stride_block_shared
  ```

with:

```text
  block size = 512
  grid size  = 432
  E2E time = 2.725 ms
  E2E speedup = 10.35x
  E2E effective bandwidth = 24.63 GB/s
  ```

---

## 6. Interpretation of Main Results


  ### 6.1 Atomic reduction

The `cuda_atomic_add` implementation has:

```text
  kernel time = 40.437 ms
  post-H2D time = 40.445 ms
  E2E time = 43.094 ms
  post-H2D speedup = 0.70x
  E2E speedup = 0.65x
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
  grid size  = 131072
```

It achieves:

```text
  kernel time = 0.139 ms
  D2H time = 0.028 ms
  CPU finalize time = 0.219 ms
  post-H2D time = 0.386 ms
  E2E time = 3.036 ms
  post-H2D speedup = 73.03x
  E2E speedup = 9.29x
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

For example, at `N = 16,777,216`:

```text
basic block reduction:
    block size = 128
    grid size  = 131072
    partial sums = 131072

grid-stride reduction:
    block size = 512
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
  block size = 512
  grid size  = 432
  ```

It achieves:

```text
  kernel time = 0.067 ms
  D2H time = 0.007 ms
  CPU finalize time = 0.001 ms
  post-H2D time = 0.075 ms
  E2E time = 2.725 ms
  post-H2D speedup = 374.40x
  E2E speedup = 10.35x
  ```

This implementation reduces the number of partial sums compared with the basic block-level implementation. As a result, both D2H transfer time and CPU finalization time become much smaller.


  In this benchmark, `cuda_grid_stride_block_shared` is the fastest implementation in terms of E2E time.
  


  It is also the fastest implementation in terms of post-H2D time.
  
  

---



### 6.4 Two-pass GPU reduction

The `cuda_grid_stride_two_pass` implementation performs the reduction entirely on the GPU in two stages.

Conceptually, it works as follows:

```text
pass 1:
    input x[0:N] -> partial_sums[0:num_blocks]

pass 2:
    partial_sums[0:num_blocks] -> final scalar sum
```

The first pass is similar to the Grid-stride shared-memory reduction:
each CUDA block reduces a subset of the input and writes one partial sum.
The second pass then launches another CUDA kernel to reduce these partial sums into a single scalar result on the GPU.

This means that, unlike the `cuda_block_shared_mem` and `cuda_grid_stride_block_shared` versions with CPU finalization, the final result is produced directly on the GPU.

For the main benchmark size, the selected configuration is:

```text
block size = 512
grid size  = 432
```

It achieves:

```text
kernel time = 0.071 ms
D2H time = 0.007 ms
CPU finalize time = 0.000 ms
post-H2D time = 0.078 ms
E2E time = 2.728 ms
post-H2D speedup = 360.03x
E2E speedup = 10.33x
```

In the current benchmark, the two-pass version does not clearly outperform the grid-stride shared-memory version with CPU finalization.


For comparison, the grid-stride shared-memory version achieves:

```text
grid-stride post-H2D time = 0.075 ms
grid-stride E2E time      = 2.725 ms
```

while the two-pass version achieves:

```text
two-pass post-H2D time = 0.078 ms
two-pass E2E time      = 2.728 ms
```



If we consider the time from the start of computation on the device until the result is obtained,
the two-pass version is approximately 3 us faster than the grid-stride loop version with a final finalize step.
In this scenario, the two-pass version effectively does not need to account for D2H time,
whereas the version with CPU finalize must account for D2H time. Furthermore,
if subsequent computations are still performed on the GPU, the time required to copy the results
to the device must also be considered, making the version with CPU finalize slower.

Furthermore, the significance of the two-pass version is not only its raw runtime in this isolated benchmark.
Its main advantage is that the final scalar result is produced on the GPU. This is important because
in many real GPU-resident workflows, the reduction result may be consumed by later GPU kernels without
being copied back to the CPU.

Examples include:

```text
norm computation
loss reduction
residual/error calculation
adaptive timestep criteria
normalization statistics
iterative solver convergence checks
```

If the reduction result can stay on the GPU and be used by subsequent GPU kernels, then the D2H transfer can often be avoided entirely. In that situation, a fully GPU-resident two-pass reduction can be more appropriate than a faster-looking hybrid version that relies on CPU finalization.

Therefore, even though the two-pass implementation is not substantially faster than the grid-stride CPU-finalized version in this standalone benchmark, it represents a more GPU-native execution model.



---




### 6.5 Multi-pass GPU reduction

The `cuda_multi_pass` implementation generalizes the two-pass idea.

Instead of assuming that the first pass produces a small enough partial-sum array to be reduced in one additional kernel, it repeatedly applies a block-level reduction until only one value remains.

Conceptually:

```text
input x[0:N]
    -> partial_sums_level_1
    -> partial_sums_level_2
    -> ...
    -> final scalar sum
```

Each pass reduces the current input array into a smaller partial-sum array. The process continues until the number of remaining elements is small enough to produce a single final scalar.

For the main benchmark size, the selected configuration is:

```text
block size = 512
grid size  = 432
```

It achieves:

```text
kernel time = 0.074 ms
D2H time = 0.007 ms
CPU finalize time = 0.000 ms
post-H2D time = 0.082 ms
E2E time = 2.731 ms
post-H2D speedup = 345.47x
E2E speedup = 10.32x
```

In the current benchmark, the multi-pass version also does not clearly outperform the two-pass or grid-stride versions.


For comparison:

```text
two-pass post-H2D time   = 0.078 ms
multi-pass post-H2D time = 0.082 ms

two-pass E2E time   = 2.728 ms
multi-pass E2E time = 2.731 ms
```



This is expected for the current problem sizes and implementation style. Multi-pass reduction introduces additional kernel launches and intermediate global-memory traffic. If the number of partial sums after the first pass is already small enough, a fixed two-pass strategy may be sufficient and cheaper.

The value of the multi-pass version is mainly architectural:

1. It supports arbitrary large inputs without assuming that one second-pass kernel is enough.
2. It keeps the entire reduction process on the GPU.
3. It provides a general building block for GPU-resident reductions.
4. It is closer to how a reusable reduction primitive would be structured.
5. It prepares the codebase for later optimizations such as warp-level reduction, cooperative groups, CUB-like primitives, or persistent temporary buffers.

In other words, the multi-pass implementation is not necessarily the fastest current version, but it is a more general and robust GPU-only reduction strategy.



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

Each thread accumulates a private partial sum, and OpenMP combines these partial sums at the end of the parallel region. This avoids a shared global accumulator and gives the CPU implementation access to much more memory bandwidth than a single-threaded loop.

For the main benchmark size:

```text
N = 16,777,216
```

the `cpu_omp` implementation achieves:

```text
time = 1.596 ms
effective bandwidth = 42.04 GB/s
GFLOP/s = 10.51
speedup vs cpu_serial = 17.66x
```

This is a very strong CPU result. With 18 threads, the OpenMP version reaches almost ideal scaling relative to the single-threaded CPU baseline.

In fact, its E2E speedup can exceed the E2E speedup of the fastest GPU version.


For comparison, the fastest GPU E2E implementation is:

```text
cuda_grid_stride_block_shared
```

with:

```text
GPU E2E speedup = 10.35x
GPU E2E time    = 2.725 ms
```

while `cpu_omp` achieves:

```text
CPU OpenMP speedup = 17.66x
CPU OpenMP time    = 1.596 ms
```



This does not mean that the GPU kernels are slow. Rather, it highlights the difference between post-H2D and E2E performance.

For the fastest GPU version:

```text
post-H2D speedup = 374.40x
E2E speedup      = 10.35x
```

The post-H2D speedup is much larger because it measures the case where the input data is already resident on the GPU. The E2E speedup is limited by the H2D transfer of the entire input vector.

In this benchmark, the GPU E2E speedup is limited to roughly the range imposed by host-device transfer bandwidth, while the OpenMP version operates directly on CPU memory and does not pay any PCIe/NVLink transfer cost.

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
  cuda_grid_stride_block_shared
  ```

Its timing breakdown is:

```text
  H2D time = 2.650 ms
  kernel time = 0.067 ms
  D2H time = 0.007 ms
  CPU finalize time = 0.001 ms
  E2E time = 2.725 ms
  ```

The H2D fraction of the E2E time is approximately:

```text
  97.2%
  ```

This shows an important GPU performance principle:

> For low-arithmetic-intensity operations such as reduction, host-device transfer can dominate end-to-end runtime.

If the input data is already resident on the GPU, the relevant metric is post-H2D performance.
If the input starts on the CPU, the relevant metric is E2E performance.

---

## 8. Block Size Sweep Analysis

The block-size sweep measures kernel-stage performance at:

```text
N = 16,777,216
```

These results should be interpreted as kernel-stage diagnostics.
For partial-output reduction kernels, the kernel stage alone does not include D2H transfer or CPU finalization.


### 1. `cuda_atomic_add`

For `cuda_atomic_add`, the best kernel-stage configuration is:

```text
block size = 256
grid size  = 65536
kernel time = 40.437 ms
kernel effective bandwidth = 1.66 GB/s
kernel GFLOP/s = 0.41
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 40.437 ms with block size = 256
max kernel time = 40.447 ms with block_size = 128
max/min ratio   = 1.00x
```


The atomic version shows limited sensitivity to block size. This suggests that
the dominant bottleneck is global atomic contention rather than block-size-dependent occupancy or memory coalescing.



### 2. `cuda_block_shared_mem`

For `cuda_block_shared_mem`, the best kernel-stage configuration is:

```text
block size = 128
grid size  = 131072
kernel time = 0.139 ms
kernel effective bandwidth = 486.49 GB/s
kernel GFLOP/s = 120.68
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.139 ms with block size = 128
max kernel time = 0.217 ms with block_size = 64
max/min ratio   = 1.56x
```


The shared-memory block reduction has a clear block-size optimum. Very small blocks create too many partial sums,
while very large blocks may increase synchronization cost or reduce scheduling flexibility.



### 3. `cuda_grid_stride_block_shared`

For `cuda_grid_stride_block_shared`, the best kernel-stage configuration is:

```text
block size = 512
grid size  = 432
kernel time = 0.067 ms
kernel effective bandwidth = 1000.74 GB/s
kernel GFLOP/s = 250.18
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.067 ms with block size = 512
max kernel time = 0.315 ms with block_size = 64
max/min ratio   = 4.69x
```


The grid-stride version benefits from larger block sizes up to the best observed configuration.
This is likely because each block performs more local work before writing a partial sum, reducing overhead while maintaining enough parallelism.



### 4. `cuda_grid_stride_two_pass`

For `cuda_grid_stride_two_pass`, the best kernel-stage configuration is:

```text
block size = 512
grid size  = 432
kernel time = 0.071 ms
kernel effective bandwidth = 947.12 GB/s
kernel GFLOP/s = 236.78
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.071 ms with block size = 512
max kernel time = 0.319 ms with block_size = 64
max/min ratio   = 4.50x
```


The grid-stride version benefits from larger block sizes up to the best observed configuration.
This is likely because each block performs more local work before writing a partial sum, reducing overhead while maintaining enough parallelism.



### 5. `cuda_multi_pass`

For `cuda_multi_pass`, the best kernel-stage configuration is:

```text
block size = 512
grid size  = 432
kernel time = 0.074 ms
kernel effective bandwidth = 903.58 GB/s
kernel GFLOP/s = 225.89
```

Across the tested block sizes, the kernel time ranges from:

```text
min kernel time = 0.074 ms with block size = 512
max kernel time = 0.322 ms with block_size = 64
max/min ratio   = 4.33x
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
best block size = 64
grid size = 4194304
post-H2D time = 647.170 ms
E2E time = 689.540 ms
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
post-H2D time = 0.018 ms
E2E time = 0.073 ms
post-H2D speedup = 24.93x
E2E speedup = 6.01x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 128
grid size = 2097152
post-H2D time = 5.954 ms
E2E time = 48.323 ms
post-H2D speedup = 75.90x
E2E speedup = 9.35x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 3.04
E2E speedup growth      = 1.55
```


The shared-memory block version improves significantly over CPU serial,
but as `N` grows it also creates more partial sums, which increases D2H and CPU-finalization overhead.



### 3. `cuda_grid_stride_block_shared`

For `cuda_grid_stride_block_shared`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.016 ms
E2E time = 0.071 ms
post-H2D speedup = 28.20x
E2E speedup = 6.19x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 0.993 ms
E2E time = 43.363 ms
post-H2D speedup = 454.95x
E2E speedup = 10.42x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 16.13
E2E speedup growth      = 1.68
```


The grid-stride version scales well in post-H2D terms because the amount of GPU work
increases while the number of partial sums remains relatively small.



### 4. `cuda_grid_stride_two_pass`

For `cuda_grid_stride_two_pass`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.018 ms
E2E time = 0.074 ms
post-H2D speedup = 24.37x
E2E speedup = 5.98x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 0.997 ms
E2E time = 43.366 ms
post-H2D speedup = 453.18x
E2E speedup = 10.42x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 18.59
E2E speedup growth      = 1.74
```


The grid-stride version scales well in post-H2D terms because the amount of GPU work
increases while the number of partial sums remains relatively small.



### 5. `cuda_multi_pass`

For `cuda_multi_pass`, the smallest tested problem size is:

```text
N = 262,144
single vector size = 1.00 MiB
post-H2D time = 0.021 ms
E2E time = 0.076 ms
post-H2D speedup = 21.09x
E2E speedup = 5.76x
```

The largest tested problem size is:

```text
N = 268,435,456
single vector size = 1024.00 MiB
best block size = 1024
grid size = 432
post-H2D time = 1.000 ms
E2E time = 43.369 ms
post-H2D speedup = 452.04x
E2E speedup = 10.42x
```

From the smallest to the largest tested problem size:

```text
post-H2D speedup growth = 21.44
E2E speedup growth      = 1.81
```







### CPU Optimized Problem Size Scaling



### `cpu_omp`

At the largest tested size:

```text
N = 268,435,456
CPU serial time = 451.881 ms
CPU optimized time = 25.245 ms
CPU optimized bandwidth = 42.53 GB/s
speedup = 17.90x
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

This is especially important for parallel reduction kernels. A tree-based or block-based parallel reduction changes the order of additions, and may produce a result closer to the mathematical value `N` than the serial FP32 result.

  


---

## 12. Key Observations

1. **Atomic reduction is simple but usually not performant.**

   The atomic implementation is useful, but it is
   not a good high-performance reduction strategy. All threads contend for
   updates to a shared global scalar, so the computation becomes
   heavily serialized around the atomic operation.

2. **Shared-memory reduction is the first important optimization step.**

   Reducing values locally within each block avoids global atomic
   contention and exposes the tree-structured nature of parallel reduction.
   This is the first version where the implementation starts to use GPU memory
   hierarchy and thread cooperation meaningfully.

3. **The number of partial sums matters.**

   A partial-output reduction kernel is not finished after the GPU kernel returns.
   If it produces many partial sums, those partial sums must either be copied back to
   the CPU and finalized there, or reduced further on the GPU. Therefore, the output
   size of each reduction stage is an important part of the algorithm design.

4. **Grid-stride loops reduce partial-sum overhead.**

   A grid-stride reduction lets each block process multiple input elements. This decouples
   the number of partial sums from `ceil(N / block_size)` and allows the implementation to
   use a fixed, hardware-informed number of blocks. This can greatly reduce D2H transfer and
   CPU-finalization overhead when compared with a basic one-block-per-chunk reduction.

5. **CPU-finalized GPU reductions can be fast but are not fully GPU-resident.**

   A GPU kernel followed by D2H transfer and CPU finalization may perform very well in a
   standalone benchmark. However, it breaks GPU residency: the final scalar is produced on the CPU.
   This matters if the result is needed by later GPU kernels.

6. **Two-pass and multi-pass reductions are important even when they are not faster in this standalone benchmark.**

   Fully GPU-resident reductions produce the final result on the GPU. This is important for
   larger GPU pipelines where the reduction result can be consumed by subsequent GPU kernels without
   returning to the CPU. Their value is architectural as well as performance-related.

7. **Kernel-only speedup can be misleading.**

   For partial-output kernels, kernel time alone does not include all work needed to obtain the
   final scalar result. A fair algorithm-level comparison must include D2H transfer and CPU finalization
   if those stages are required.

8. **Post-H2D and E2E answer different questions.**

   - Post-H2D performance answers: “How fast is reduction if data is already on GPU?”
   - E2E performance answers: “How fast is reduction if data starts on CPU?”

   Both metrics are useful, but they describe different execution scenarios.

9. **A strong CPU baseline changes the interpretation of GPU speedups.**

   Comparing GPU kernels only against a single-threaded CPU baseline can overstate practical speedups.
   A multi-threaded CPU implementation can be very competitive for standalone memory-bound reductions,
   especially when GPU E2E time includes host-device transfer.

10. **Reduction is more about memory movement and execution structure than arithmetic throughput.**

    FP32 reduction performs very little arithmetic per byte of input data. As a result, performance
    is usually governed by memory bandwidth, synchronization, atomic contention, launch overhead,
    partial-sum handling, and host-device transfer rather than raw floating-point throughput.

---

## 13. Current Best Implementation

The current best E2E kernel implementation is:

```text
  cuda_grid_stride_block_shared
  ```

with:

```text
  N = 16,777,216
  block size = 512
  grid size = 432
  kernel time = 0.067 ms
  D2H time = 0.007 ms
  CPU finalize time = 0.001 ms
  post-H2D time = 0.075 ms
  E2E time = 2.725 ms
  post-H2D speedup = 374.40x
  E2E speedup = 10.35x
  ```



### CPU Optimized Baseline

The fastest optimized CPU implementation at `N = 16,777,216` is:

```text
cpu_omp
```
with
```text
time = 1.596 ms
bandwidth = 42.04 GB/s
speedup vs cpu_serial = 17.66x
```



---

## 14. Limitations

The current benchmark has several limitations:

1. **The benchmark is still a microbenchmark.**

   The reduction is measured as an isolated operation. In real applications,
   reductions are usually part of a larger GPU or CPU pipeline. Whether the
   input is already resident on the GPU, and whether the result is consumed on the
   GPU or CPU, can change the interpretation of the results.

2. **E2E performance depends strongly on host-device transfer.**

   GPU E2E timings include host-to-device input transfer. For a standalone reduction,
   this transfer can dominate runtime. In a GPU-resident workload, this cost may be amortized or avoided entirely.

3. **Some GPU implementations still use CPU finalization.**

   Partial-output kernels that copy partial sums back to the CPU are hybrid algorithms, not
   fully GPU-resident reductions. They can be fast, but they are not always appropriate when
   subsequent computation remains on the GPU.

4. **The fully GPU-resident implementations are not yet highly optimized.**

   Two-pass and multi-pass versions are useful algorithmic baselines, but they may still
   have avoidable overheads such as extra kernel launches, inefficient partial-sum handling,
   or suboptimal memory access patterns.

5. **Only one data type is currently tested.**

   Current data type:

   ```text
   fp32
   ```

---

## 16. Auto-Generated Benchmark Tables

The following tables are generated automatically from the benchmark CSV.

Generated tabls can be found in [`reduction_bench_tables.md`](markdown_tables/reduction_bench_tables.md)