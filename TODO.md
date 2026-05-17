# TODO list

## Current Focus: Vector Add

Goal: build the full CUDA kernel development workflow

- [x] Implement CPU reference vector add
- [x] Implement naive CUDA vector add
- [x] Add correctness check
- [x] Add `CudaTimer`
- [x] Add `CpuTimer`
- [x] Benchmark CPU version
- [x] Benchmark GPU version
- [x] Compute CPU effective bandwidth
- [x] Compute GPU effective bandwidth
- [x] Test different vector sizes
- [x] Test different block sizes
- [ ] Compare pageable host memory and pinned host memory for H2D/D2H transfer

## Refactoring Ideas

- [x] Add generic CPU benchmark helper
- [x] Add generic CUDA benchmark helper

## Possible Improvements for Vector Add

- [ ] Add OpenMP CPU baseline

## Backlog

## Next Kernels

- [ ] Reduction
- [ ] Matrix transpose
- [ ] Matrix multiplication
- [ ] Softmax

