# Profiling Notes

## CPU Profiling

### Check hardware counters with `perf`

Example command:
```bash
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bench_vector_add
```
- High LLC miss rate suggests data is coming from DRAM.
- Low LLC miss rate may indicate cache-resident data.
- Hardware counter interpretation is CPU-specific

### Check compiler vectorization
**TODO:** vectorization reports? missed vectorization?


## GPU Profiling
Useful command:
```bash
nsys profile ./bench_vector_add
```