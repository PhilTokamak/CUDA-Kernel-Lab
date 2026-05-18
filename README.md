# CUDA Kernel Lab

A modern CUDA/C++ playground for experimenting with GPU kernels and learning GPU performance engineering.

## Current Kernels

- Vector Add
- Reduction

## Features

- CUDA C++20 kernel experiments
- CPU/GPU result verification
- CUDA event-based timing utilities
- Modern CMake + Ninja preset-based build workflow
- Error handling with `std::source_location`
- `fmtlib`-based diagnostics

## Goals

This project is intended as a learning and experimentation infrastructure for:

- CUDA kernel optimization
- Benchmarking infrastructure
- Modern C++ in HPC/GPU programming

## Build
Configure the project:
```bash
cmake --preset release
```

Build the project:
```bash
cmake --build --preset release
```

Run ctest:
```bash
ctest --preset release
```
Similar for build type `debug` or `relwithdebinfo`.

If `fmt` library is manually installed to a user-defnied directory, e.g. `$HOME/local/lib64/cmake/fmt`, then the cmake configure command should be
```bash
cmake --preset release -Dfmt_DIR=$HOME/local/lib64/cmake/fmt
```

Alternatively, after configuration, you can build directly with Ninja from the binary directory:
```bash
cd build/release
ninja
```