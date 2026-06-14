# TODO list

## Current Focus: Matrix Transpose

Goal: study 1D memory access, understand global memory coalescing, shared memory tiling, and bank conflicts

- [x] Implement CPU reference transpose
- [x] Implement CPU copy baseline
- [x] Implement CUDA copy baseline
- [x] Implement CUDA naive transpose
- [x] Implement CUDA shared memory tiled transpose
- [ ] Implement CUDA padded shared memory tiled transpose
- [x] Add OpenMP CPU baseline
- [x] Add correctness check
- [ ] Add benchmark sweep over square matrix size
- [ ] Add benchmark sweep over tile size
- [ ] Compare transpose bandwidth against device-to-device copy bandwidth
- [ ] Compare effective bandwidth to copy baseline
- [ ] Inspect global memory load/store efficiency
- [ ] Inspect shared memory bank conflicts
- [ ] Output benchmark data to `results/data/transpose`
- [ ] Generate Markdown tables
- [ ] Generate Jinja benchmark report template
- [ ] Generate report `results/transpose.md`

## Possible Improvements

- [ ]

## Backlog

- [ ]

## Next Kernels

- [ ] Matrix multiplication
- [ ] Softmax
- [ ] LayerNorm
- [ ] 2D Stencil
- [ ] SpMV

