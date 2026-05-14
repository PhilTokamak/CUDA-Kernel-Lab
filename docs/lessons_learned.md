# Lessons Learned

## CPU Effective Bandwidth

In performance engineering, "effective bandwidth" often means: estimated useful data movement / measured runtime. It is not necessarily equal to the physical memory traffic seen by the hardware.

In a simple streaming kernel such as vector addition,

```cpp
c[i] = a[i] + b[i];
```

a first-order estimate of memory traffic per element is

```text
2 loads + 1 store = 3 * sizeof(float) 
```

However, this is only effective bandwidth estimate. On real CPUs, several factors complicate the interpretation:

### Cache effects

If the working set fits in L1, L2, or L3 cache, the measured bandwidth may represent cache bandwidth rather than DRAM bandwidth.

To approximate DRAM bandwidth, the total working set should be much larger than the last-lavel cache.

### Write allocate

For normal cached stores, the CPU may first load the target cache line before writing it. This can introduce additional memory traffic.

Therefore, the actual memory traddic may be closer to:

```text
read a + read b + read c cache ine + write c
```

rather than just

```text
read a + read b + write c
```

### SIMD auto-vectorization

With compiler optimization flags such as `-O3`, the compiler may vectorize the loop using SIMD instructions such as AVX2 or AVX-512. This affects CPU performance, but the kernel may still remain memory-bandwidth bound for large arrays.

### Single-thread vs. multi-thread bandwidth

A single CPU thread often cannot saturate the full memory bandwidth of a socket. An OpenMP version may achieve much higher bandwidth.

### Prefetching

Hardware prefetchers can hide some memory latency.

### NUMA effects

Memory placement may matter on multi-socket systems.



## Modern CMake

### 1. Modern CMake Is Target-Centric

Modern CMake revolves around the concept of **targets**.

Targets are created using:

```cmake
add_library(...)
add_executable(...)
```

The first parameter is the **target name**.

Example:

```cmake
add_library(cuda_kernels ...)
add_executable(bench_vector_add ...)
```

A target is not just a file or binary. It is a collection of:

- source files
- include directories
- compile options
- compile features
- linked libraries
- usage requirements

Modern CMake treats the build system as a **dependency graph of targets**.

------

### 2. `target_\*` Commands Attach Properties to Targets

After a target is created, properties can be attached using:

```cmake
target_include_directories(...)
target_compile_features(...)
target_link_libraries(...)
target_compile_options(...)
```

These commands modify the target’s behavior.

Example:

```cmake
target_include_directories(
    cuda_kernels
    PUBLIC
    include
)
```

adds the `include/` directory to the target.

------

### 3. Target Must Exist Before `target_\*`

This is required:

```cmake
add_library(cuda_kernels ...)
```

before:

```cmake
target_include_directories(cuda_kernels ...)
```

Otherwise CMake errors because the target does not exist yet.

------

### 4. Order Between `target_\*` Commands Usually Does Not Matter

These are generally equivalent:

```cmake
target_include_directories(...)
target_link_libraries(...)
```

and

```cmake
target_link_libraries(...)
target_include_directories(...)
```

because they simply add properties to the target.

The important requirement is only:

```text
target must already exist
```

------

### 5. PUBLIC / PRIVATE / INTERFACE

Modern CMake uses propagation semantics.

#### PRIVATE

Used only by the current target.

Does not propagate.

------

#### PUBLIC

Used by the current target and propagated to dependent targets.

------

#### INTERFACE

Not used by the current target.

Only propagated to dependents.

------

Example:

```cmake
target_include_directories(
    cuda_kernels
    PUBLIC
    include
)
```

means:

- `cuda_kernels` uses `include/`
- anything linking `cuda_kernels` also gets `include/`

------

### 6. Usage Requirements Propagate Through Dependencies

Example:

```cmake
target_compile_features(
    cuda_kernels
    PUBLIC
    cxx_std_20
    cuda_std_20
)
```

and:

```cmake
target_link_libraries(
    bench_vector_add
    PRIVATE
    cuda_kernels
)
```

Result:

```text
bench_vector_add automatically inherits:
- C++20
- CUDA20
```

This is why `bench_vector_add` does not need its own
 `target_compile_features(...)`.

------

### 7. Prefer `target_compile_features` Over Global Standards

Older style:

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
```

Modern style:

```cmake
target_compile_features(
    cuda_kernels
    PUBLIC
    cxx_std_20
    cuda_std_20
)
```

Reason:

- target-local
- composable
- dependency-aware
- avoids global configuration pollution

Modern CMake prefers target properties over global variables.

------

### 8. CMakePresets.json vs CMakeLists.txt

#### **`CMakeLists.txt`**

Defines:

- project structure
- targets
- dependencies
- include directories
- compile features

It describes the **build graph**.

------

#### `CMakePresets.json`

Defines:

- Debug/Release
- generator
- architecture
- compile_commands.json
- user/environment configuration

It describes the **build configuration**.

------

### 9. Presets Are the Modern Configure Interface

Example:

```json
"configurePresets": [
  {
      "name": "release",
      "inherits": "base",

      "cacheVariables": {
          "CMAKE_BUILD_TYPE": "Release"
      }
  }
],

"buildPresets": [
        {
            "name": "release",
            "configurePreset": "release"
        }
]
```

Usage:

```bash
cmake --preset release
cmake --build --preset release
```

This avoids long configure commands.

------

### 10. Cache Variables vs Normal Variables

Preset variables are usually:

```text
cache variables
```

Equivalent to:

```bash
-DVARIABLE=value
```

or:

```cmake
set(VARIABLE value CACHE STRING "")
```

These persist inside:

```text
CMakeCache.txt
```

and usually override normal `set(...)` assignments.

------

### 11. Build Types Control Optimization and Debugging

Typical build types:

| **Build Type** | **Typical Flags** |
| -------------- | ----------------- |
| Debug          | `-O0 -g`          |
| Release        | `-O3 -DNDEBUG`    |
| RelWithDebInfo | `-O2 -g`          |
| MinSizeRel     | `-Os`             |

------

### 12. CUDA Build Types

CUDA behaves similarly.

Debug builds may enable:

```text
-G
```

which generates device debug information and disables many optimizations.

Release builds prioritize performance.

------

### 13. `compile_commands.json`

Enable with:

```json
"CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
```

This generates:

```text
compile_commands.json
```

used by:

- clangd
- ccls
- VSCode
- CLion

------

### 14. Library Targets Are the Core of Modern CMake

Executables are usually thin wrappers.

Core logic should live in libraries:

```cmake
add_library(cuda_kernels ...)
```

Benchmarks/tests/apps then link against the library.

This improves:

- modularity
- reuse
- dependency management
- scalability

------

### 15. Modern CMake Philosophy

Old CMake:

```cmake
include_directories(...)
add_definitions(...)
set(CMAKE_...)
```

was largely global-state-based.



Modern CMake prefers:

```cmake
target_link_libraries(...)
target_compile_features(...)
target_include_directories(...)
```

This creates a cleaner dependency graph and avoids global pollution.



The central idea is: **Everything should be attached to targets.**
