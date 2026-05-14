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
- [ ] Compute CPU effective bandwidth
- [x] Compute GPU effective bandwidth
- [ ] Test different vector sizes
- [ ] Test different block sizes

## Refactoring Ideas

- [ ] Add generic CPU benchmark helper
- [ ] Add generic CUDA benchmark helper

## Possible Improvements for Vector Add

- [ ] Add OpenMP CPU baseline

## Next Kernels

- [ ] Reduction
- [ ] Matrix transpose
- [ ] Matrix multiplication
- [ ] Softmax

