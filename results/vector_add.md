# Vector Add Results

Environment: see [`environment.md`](environment.md)

## Generated Tables

Generated tabls can be found in [`vector_add_bench_tables.md`](markdown_tables/vector_add_bench_tables.md)

For CPU cache test, the tables can be found in [`vector_add_bench_tables_cpu_cache.md`](markdown_tables/vector_add_bench_tables_cpu_cache.md)

## Observations

### 1. CUDA kernel-only performance is much faster than CPU single-thread

For a large problem size case,

```text
N = 134,217,728
single vector size = 512 MiB
effective data moved = 1536 MiB
```

the single-thread CPU implementation takes 110.239 ms, while the naive CUDA kernel takes 1.176 ms. This corresponds to a **kernel-only speedup** of approximately **93.7x**. The CUDA kernel reaches an effective bandwidth of approximately 1369 GB/s, whereas the CPU single-thread version reaches about 24.6 GB/s. This confirms that when the data is already resident on the GPU, even a simple one-thread-per-element CUDA implementation can achieve much higher streaming bandwidth than a single CPU thread.



### 2. End-to-end speedup is much smaller because host-to-device/device-to-host transfers dominate

Although the CUDA kernel itself is about 94x faster than the CPU single-thread baseline, the **end-to-end speedup** is only about **1.7x** for large ($\gtrsim$ 16 millions) problem size. The measured times are:

```text
H2D copy:  42.412 ms
kernel:    1.176 ms
D2H copy:  20.395 ms
total:    63.983 ms
```

The kernel accounts for only a small fraction of the end-to-end runtime. Most of the time is spent transferring data between host and device memory.

This is expected for vector addition because the operation has very low arithmatic intensity. Each element performs only one floating-point addition, but the end-to-end path requires transferring two input arrays to the GPU and one output array back to the CPU. This demonstrates an important GPU programming principle:

> A fast GPU kernel does not necessarily imply a large end-to-end speedup id data must be transferred between CPU and GPU for every operation.

### 3. H2D and D2H bandwidth are around 25~26 GB/s

For a large problem size, the measured host-device copy bandwidth is

```text
H2D: 25.3 GB/s
D2H: 26.3 GB/s
```

The H2D copy transfers two arrays, while the D2H copy transfers one array. The H2D time is therefore roughly twice the D2H time, which is consistent with the amount of data transferred.

The end-to-end speedup is limited by this host-device transfer bandwidth. For large problem sizes, the CPU single-thread bandwidth is about 14.6 GB/s, while the effective host-device transfer bandwidth is about 25 GB/s. This ($25 / 14.6 \approx 1.71$) explains why the end-to-end speedup stabilizes around 1.7x.

### 4. CUDA block size matters, but only up to a point

For large problem size, block size 64 is significantly slower, which is around 953 GB/s, while block sizes 128, 256, and 1024 all achives around: 1360 GB/s. The best result in this experiment is obtained with block size 256, but the difference between 128, 256, 512 and 1024 is small. This suggests that once the block size is large enough to provide sufficient parallelism, the vector add kernel becomes primarily limited by global memory bandwidth rather than by the exact block size.

### 5. Small problem are dominated by overhead and transfer cost

For small vectors, the CUDA kernel-only speedup is already significant, but the end-to-end speedup can be less than 1. For example:

```text
N = 262,144
CPU time:      					0.120 ms
GPU kernel:    					0.0068 ms
Kernel-only speedup: 		17.72x
GPU E2E:       					0.158 ms
E2E speedup:   					0.76x
```

In this regime, the GPU kernel is faster than the CPU computation, but the fixed overheads and host-device transfers dominate the total runtime. The GPU end-to-end performance starts to become competitive only when the vector size is large enough. In this experiment, the **break-even** point is around: $$N\approx 2$$ million elements.





### 6. Large problem sizes give more statble bandwidth estimates

For large vectors, the CPU bandwidth stabilizes around:

```text
14.6 ~ 14.8 GB/s
```

and the CUDA kernel effective bandwidth stabilizes around:

```text
1.3 ~ 1.37 TB/s
```

These large-size results are more representative of streaming memory bandwidth.

For small and medium problem sizes, the measured effective bandwidth can be affected by the cache effects, launch overhead, timing overhead, and repeated access to the same data. Therefore, small-size bandwidth numvers for GPU should not be over-interpreted.

### 7. Gflops/s is not the most meaningful metric for vector add

Vector addition performs only one floating-point addition per element. Its arithmetic intensity is very low: 1 Flop / 12 bytes $$\approx$$ 0.083 Flop/byte. Therefore, vector add is memory-bandwidth is a more meaningful performance metric than Gflop/s.



### 8. CPU bandwidth reflects the cache hierarchy

The host CPU used in this experiment has the follwing cache hierarchy:

```text
L1: 64 kB or 62.5 KiB per core
L2: 1024 kB or 976.6 KiB per core
L3: 54 MB or 51.5 MiB shared
```

For vector addition, the appriximate working set is `3 * N * sizeof(float)`. The CPU bandwidth trend is consistent with this cache hierarchy.

For very small vectors, the working set fits in the private cache levels. For example, when `N=4096`, each vector is 16kiB and he total working set is about 48 KiB, which fits within the L1 cache. The measured effective bandwidth is high:

```text
N = 4096
working set = 48 kiB
CPU effective BW ~ 83 GB/s
```

As the working set grows beyond L1 and L2, the CPU bandwidth drops. For example"

```text
N = 16,384
working set = 192 kiB
CPU effective BW ~ 53 GB/s
```

It can still fit within the L2 cache. However, when:

```text
N = 131,072
working set ~ 1.5 MiB
CPU effective BW ~ 40 GB/s
```

It exceeds the private L2 cache size of about 1 MiB, so the effective bandwidth decreases.

For medium-size vectors, the working set canstill fit in the shared L3 cache, and the bandwidth remains around 29 GB/s:

```text
N = 1,048,576
working set = 12 MiB
CPU effective BW ~ 29 GB/s
```

When the working set approaches or exceeds the 51.5 MiB shared L3 cache, the bandwidth drops further:

```text
N = 4,194,304
working set = 48 MiB
CPU effective BW ~ 21.6 GB/s

N = 8,388,608
working set = 96 MiB
CPU effective BW ~ 16.1 GB/s
```

For larger vectors, the CPU bandwidth stabilizes around 14 ~ 16 GB/s, suggesting that the benchmark has entered a DRAM streaming regime. Overall, the CPU results show a clear transition from cache-affected performance for small arrays to DRAM-bandwidth-limited performance for large arrays.

### 9. Summary

The naive CUDA vector add kernel achieves about 1.37 TB/s effective memory bandwidth on the largest problem size and is about 94x faster than the single-thread CPU baseline when measuring kernel execution only.

However, when host-device transfers are included, the end-to-end speedup drops to about 1.7x. This is because vector addition has very low arithmetic intensity, and the total runtime becomes dominated by H2D/D2H transfer time.

This benchmark demonstrates the difference between kernel-only performance and end-to-end application performance. It also motivates common GPU performance strategies such as keeping data resident on the GPU, running many kernels before copying results back, increasing arithmetic intensity, and fusing operations.



## Bandwidth Calculation

For vector addition,

```cpp
c[i] = a[i] + b[i]
```

the approximate data movement per element is:

- read `a[i]`: 4 bytes
- read `b[i]`: 4 bytes
- write `c[i]`: 4 bytes

Therefore, the effective bandwidth is estimated as:

```cpp
bandwidth = 3 * n * sizeof(float) / time
```

This is a simplified effective bandwidth model.

## Caveats

The CPU bandwidth number should be interpreted carefully:

- This bandwidth number should be interpreted as an application-level effective bandwidth, not as a direct measurement of peak DRAM bandwidth.
- The formula does not account for possible write-allocate traffic.
- The compiler may auto-vectorize the CPU loop using SIMD instructions.
- Single-thread CPU bandwidth ususlly does not saturate the full memory bandwidth of the system.
- CPU frequency scaling, NUMA placement, and cache state can affect the result.
