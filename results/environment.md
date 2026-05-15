# Benchmark Environment

Unless otherwise stated, the benchmarks were run with one NVIDIA A100 GPU, 8 CPU cores, and 32 GB host memory.

| Item                    | Value                                           |
| ----------------------- | ----------------------------------------------- |
| Platform                | Linux x86_64                                    |
| GPU                     | NVIDIA A100 40GB                                |
| GPU architecture        | Ampere                                          |
| CUDA compute capability | 8.0                                             |
| CUDA toolkit            | 13.0                                            |
| Host compiler           | GCC 15.1                                        |
| CUDA architecture flag  | `sm_80`                                         |
| CMAKE_BUILD_TYPE        | `Release`                                       |
| GPU timing method       | CUDA events                                     |
| GPU timing scope        | Kernel-only unless otherwise stated             |
| CPU baseline            | Single-thread reference unless otherwise stated |
| Default data type       | FP32                                            |

Notes:

- Reported GPU bandwidth values are intended as approximate performance measurements for learning and comparison, not as formal reproducibility claims.