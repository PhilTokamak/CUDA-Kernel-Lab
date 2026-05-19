# TODO list

## Current Focus: Reduction

Goal: implement and benchmark several CUDA reduction strategies

- [x] Implement CPU serial reduction
- [x] Implement CUDA atomic reduction
- [x] Implement CUDA block-level shared memory
- [ ] Add correctness check with absolute and relative error
- [ ] Add benchmark sweep over `n`
- [ ] Add benchmark sweep over block size
- [ ] Output benchmark data to `results/data/reduction`
- [ ] Generate Markdown tables
- [ ] Write `results/reduction.md`

## Possible Improvements

- [ ]

## Backlog

- [ ] Compare pageable host memory and pinned host memory for H2D/D2H transfer
- [ ] Add OpenMP CPU baseline

## Next Kernels

- [ ] Matrix transpose
- [ ] Matrix multiplication
- [ ] Softmax

