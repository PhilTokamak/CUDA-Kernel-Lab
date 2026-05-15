# Vector Add Results

Environment: see [`environment.md`](environment.md)

## Main Benchmark Results

| Version    | Device |        N | Single Vector Size | Working Set | DType | Block Size | Repeat | Avg Time [ms]* | Min Time [ms] | Effective BW [GB/s] | Speedup vs CPU serial* |
| ---------- | ------ | -------: | -----------------: | ----------: | ----- | ---------: | -----: | -------------: | ------------: | ------------------: | ---------------------: |
| cpu_serial | CPU    | 67108864 |             256MiB |     768 MiB | FP32  |        N/A |      3 |         282.33 |             / |               2.852 |                  1.00x |
| cuda_naive | GPU    | 67108864 |             256MiB |      768MiB | FP32  |        256 |     10 |           0.64 |             / |            1250.053 |                441.14x |

*Note that the time (and speedup for GPU) is kernel only and doesn't include H2D/D2H copy.

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
- If the arrays fit into cache, the measured bandwidth may reflect cache bandwidth rather than DRAM bandwidth.
- The compiler may auto-vectorize the CPU loop using SIMD instructions.
- Single-thread CPU bandwidth ususlly does not saturate the full memory bandwidth of the system.
- CPU frequency scaling, NUMA placement, and cache state can affect the result.

The GPU timing only measures kernel execution tie and does not include H2D/D2H transfers.