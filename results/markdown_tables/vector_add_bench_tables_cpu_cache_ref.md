# Vector Add Benchmark Tables (mainly for CPU cache test)

This file is generated from `results/data/vector_add_cpu_cache_test_ref.csv`. In problem size sweep, N starts from very small value, in order to fit into L1/L2 and L3 caches respectively.

As a reference, the CPU used in this experiment has the following cache hierarchy:

```text
L1: 64 kB or 62.5 KiB per core
L2: 1024 kB or 976.6 KiB per core
L3: 54 MB or 51.5 MiB shared
```



## Problem Size Sweep

Version = cpu_serial and cuda_naive

| N          | Single Vector Size | CPU Time [ms] | CPU BW [GB/s] | Best GPU Block | GPU Kernel Time [ms] | GPU Kernel Effective BW [GB/s] | Kernel Speedup |
| :--------- | :----------------- | ------------: | ------------: | -------------: | -------------------: | -----------------------------: | :------------- |
| 512        | 2.0 kiB            |       9.6e-05 |        64.207 |            256 |             0.005751 |                          1.068 | 0.02x          |
| 1,024      | 4.0 kiB            |      0.000174 |        70.482 |            512 |             0.005767 |                          2.131 | 0.03x          |
| 2,048      | 8.0 kiB            |       0.00031 |        79.208 |            512 |             0.005803 |                          4.235 | 0.05x          |
| 4,096      | 16.0 kiB           |      0.000592 |        83.053 |            512 |             0.005828 |                          8.434 | 0.10x          |
| 8,192      | 32.0 kiB           |      0.001853 |        53.058 |            256 |             0.005847 |                         16.814 | 0.32x          |
| 16,384     | 64.0 kiB           |      0.003705 |         53.07 |            256 |             0.005919 |                         33.214 | 0.63x          |
| 32,768     | 128.0 kiB          |      0.007386 |         53.24 |            512 |             0.005912 |                         66.512 | 1.25x          |
| 65,536     | 256.0 kiB          |      0.015739 |        49.966 |           1024 |             0.006028 |                        130.467 | 2.61x          |
| 131,072    | 512.0 kiB          |       0.03954 |        39.779 |            512 |             0.006237 |                        252.165 | 6.34x          |
| 262,144    | 1.0 MiB            |      0.107446 |        29.277 |            512 |             0.006729 |                        467.491 | 15.97x         |
| 524,288    | 2.0 MiB            |      0.215081 |        29.252 |            512 |             0.007557 |                        832.485 | 28.46x         |
| 1,048,576  | 4.0 MiB            |      0.431037 |        29.192 |            512 |             0.009188 |                        1369.42 | 46.91x         |
| 2,097,152  | 8.0 MiB            |      0.875002 |        28.761 |            512 |             0.013314 |                        1890.23 | 65.72x         |
| 4,194,304  | 16.0 MiB           |       2.33492 |        21.556 |            128 |             0.043639 |                        1153.37 | 53.51x         |
| 8,388,608  | 32.0 MiB           |       6.25419 |        16.095 |            256 |             0.080131 |                        1256.24 | 78.05x         |
| 16,777,216 | 64.0 MiB           |       13.1692 |        15.288 |            256 |             0.153235 |                        1313.84 | 85.94x         |

