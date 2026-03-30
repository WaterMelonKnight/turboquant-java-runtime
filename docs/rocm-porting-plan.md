# ROCm / HIP Porting Plan

This document explains what must change — and crucially, what stays the same —
when upgrading `tq-backend-hip` from a mock placeholder to a real ROCm/HIP
implementation.

---

## Why the shared C ABI matters

`include/tq_native_api.h` defines a backend-agnostic stable ABI that both
`libtq_cuda.so` and `libtq_hip.so` implement identically.  Java never calls
CUDA or HIP APIs directly; it only calls `tq_runtime_create`,
`tq_session_infer`, etc.  This layering means:

- The entire Java stack (API, SPI, Core, Spring starter, bench CLI) compiles
  once and runs against either backend with no source changes.
- Unit tests written for `tq-backend-cpu-stub` exercise the same interfaces
  that CUDA and HIP satisfy — no new test infrastructure is needed.
- Benchmarks (`tq-bench-cli`) run unmodified against real HIP kernels;
  pass `--backend hip` and set the system property.

---

## What stays unchanged

### Java layer — zero changes required

| Component | Why no change is needed |
|-----------|------------------------|
| `tq-runtime-api` | Pure Java interfaces and value types; no vendor imports |
| `tq-runtime-spi` | `BackendProvider` SPI; vendor-neutral |
| `tq-runtime-core` | `BackendRegistry`, `DefaultTurboQuantRuntime`; discovers backends via `ServiceLoader` |
| `tq-backend-cpu-stub` | Independent CPU reference; used for baseline benchmarks |
| `tq-spring-boot-starter` | Auto-configuration; no backend-specific code |
| `tq-bench-cli` | Runs against any backend selected by `--backend hip` |
| `HipNativeBridge.java` | Method signatures already match the stable ABI |
| `HipBackend.java` | Uses `tqRuntimeCreate` / `tqRuntimeDescribe` / `tqRuntimeDestroy` |
| `HipComputeSession.java` | Uses `tqSessionCreate` / `tqSessionInfer` / `tqSessionDestroy` |
| `HipTensorHandle.java` | Uses `tqMalloc` / `tqFree` / `tqDownloadFloat` |

### C JNI bridge — one-line difference from CUDA

`native/hip/tq_jni_bridge.c` is identical to `native/cuda/tq_jni_bridge.c`
except for one line:

```c
// CUDA
#define TQ_JNI(name)  Java_com_turboquant_backend_cuda_CudaNativeBridge_##name

// HIP
#define TQ_JNI(name)  Java_com_turboquant_backend_hip_HipNativeBridge_##name
```

The JNI bridge itself requires **no further changes** when real kernels land;
it delegates everything to `tq_native_mock.c` (placeholder) or its real
replacement.

---

## What must be implemented in HIP native code

All work is contained in `native/hip/`.  The file `tq_native_mock.c` is the
placeholder; replace or supplement it with real HIP implementations.

### Step-by-step checklist

#### 1. Toolchain

- Install ROCm ≥ 6.0 and `hipcc`.
- Add to `native/hip/CMakeLists.txt`:

```cmake
find_package(hip REQUIRED)
target_sources(tq_hip PRIVATE tq_hip_impl.cpp)
target_link_libraries(tq_hip PRIVATE hip::device)
```

- Change the source language from C99 to C++17 (required by HIP):

```cmake
set_target_properties(tq_hip PROPERTIES
    LANGUAGE CXX
    CXX_STANDARD 17)
```

#### 2. Replace `tq_native_mock.c` with `tq_hip_impl.cpp`

Implement each function from `tq_native_api.h` using the HIP runtime API.
The CUDA mock is the reference; the mechanical translation is:

| CUDA call | HIP equivalent |
|-----------|---------------|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cudaStreamSynchronize` | `hipStreamSynchronize` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaRuntimeGetVersion` | `hipRuntimeGetVersion` |

`hipify-perl` (bundled with ROCm) automates this translation:

```bash
hipify-perl native/cuda/tq_native_mock.c > native/hip/tq_hip_impl.cpp
# Review and adjust; hipify misses some edge cases
```

#### 3. Implement `tq_session_infer` with real kernels

The mock uses a CPU LCG loop.  The real implementation must:

1. Accept `input_token_ids` on the host.
2. Upload the token-embedding lookup to an `hipMalloc`'d buffer.
3. Launch the attention + FFN kernels on the session's `hipStream_t`.
4. Run sampling (greedy or top-p) to produce `generated_token_ids`.
5. Download the logit vector for the last generated token.
6. Populate `tq_infer_result_t` and return `TQ_OK`.

No Java changes are needed; `HipComputeSession.infer()` already calls
`tqSessionInfer` and reads all fields via the result-handle getters.

#### 4. Implement `tq_quantise_int8` / `tq_dequantise_int8` with HIP kernels

Replace the scalar CPU loops in the mock with:

```cpp
// INT8 quantisation — one thread per element
__global__ void quantise_int8_kernel(int8_t* dst, const float* src,
                                      size_t n, float inv_scale) { … }
```

The function signature in `tq_native_api.h` remains unchanged; only the
implementation differs.

#### 5. Runtime version string

ROCm encodes versions as `major*10000 + minor*100 + patch`.
`tq_runtime_describe` in the mock already formats this correctly:

```c
int ver   = tq_runtime_version();   // e.g. 60300
int major = ver / 10000;            // 6
int minor = (ver % 10000) / 100;    // 3
```

Replace the hard-coded `MOCK_ROCM_VERSION` with a real `hipRuntimeGetVersion`
call in `tq_hip_impl.cpp`.

---

## How benchmarks and tests are reused

### Bench CLI

```bash
# Build the real libtq_hip.so
cd native/hip/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Run the bench against HIP
java -Dtq.backend.hip.enabled=true \
     -Djava.library.path=native/hip/build \
     -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
     --backend hip --warmup 3 --iters 20 --prompt-len 256 --gen-len 64
```

No CLI changes are needed; `--backend hip` already routes to `HipBackend`
via `BackendRegistry`.

### Unit tests

The existing test suites in `tq-runtime-core` and `tq-backend-cpu-stub` run
against the shared Java interfaces and remain valid.

To add HIP-specific integration tests, add a test module
`tq-backend-hip-it` that:

1. Skips all tests when `tq.backend.hip.enabled` is not set (CI without GPU).
2. Otherwise instantiates `HipBackend`, calls `init()`, and asserts that
   `infer()` returns a valid `InferenceResult`.

The test body is identical to the existing `CpuBackendTest` — no new
infrastructure required.

---

## Native folder layout after porting

```
native/
├── cuda/
│   ├── CMakeLists.txt
│   ├── tq_native_mock.c      ← kept for CI without GPU
│   ├── tq_jni_bridge.c
│   └── tq_cuda_impl.cu       ← real CUDA kernels (future)
└── hip/
    ├── CMakeLists.txt
    ├── tq_native_mock.c      ← kept for CI without GPU
    ├── tq_jni_bridge.c       ← one-line diff from CUDA bridge
    └── tq_hip_impl.cpp       ← real HIP kernels (this is the porting target)
```

The mock files are never removed — they allow the full Java test suite to
run in CI environments without a GPU.

---

## Summary of porting cost

| Category | Effort |
|----------|--------|
| Java layer | **0 changes** |
| JNI bridge (`tq_jni_bridge.c`) | **0 changes** |
| C ABI header (`tq_native_api.h`) | **0 changes** |
| Mock → real runtime lifecycle | ~50 lines of C++ |
| `tq_session_infer` with real kernels | Bulk of the work |
| Quantisation kernels | Per-kernel effort; interface is stable |
| CMakeLists.txt | ~5-line addition for HIP toolchain |
| Benchmark / test harness | **0 changes** |
