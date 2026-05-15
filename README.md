# CUDA Kernel Lab

A modern CUDA/C++ playground for experimenting with GPU kernels and learning GPU performance engineering.

## Kernels
- Vector Add

## Features

- CUDA kernel experiments
- CPU/GPU result verification
- CUDA event-based timing utilities
- Modern CMake setup
- Error handling with `std::source_location`
- `fmtlib`-based diagnostics

## Goals

This project is intended as a learning and experimentation infrastructure for:

- CUDA kernel optimization
- Benchmarking infrastructure
- Modern C++ in HPC/GPU programming

## Build
Example configure and build commands for build type `release` is shown below:
```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```
Similar for build type `debug` or `relwithdebinfo`.

If `fmt` library is manually installed to a user-defnied directory, e.g. `$HOME/local/lib64/cmake/fmt`, then the cmake configure command should be
```bash
cmake --preset release -Dfmt_DIR=$HOME/local/lib64/cmake/fmt
```