# Architecture: turboquant-java-runtime

> **Status:** Pre-alpha. The architecture described here is implemented as a working skeleton. The CPU stub path is fully functional. CUDA and HIP paths are placeholders with correct structure but no real GPU kernels.

---

## The one-sentence design goal

> The Java application layer should never need to change when the GPU vendor changes.

Everything else in this document follows from that constraint.

---

## The main flow

```
Your application code
        │  calls TurboQuantRuntime.infer(request)
        ▼
  tq-runtime-core
        │  routes to the selected backend via Backend/ComputeSession
        ▼
  tq-backend-cpu-stub   (pure Java, always available)
  tq-backend-cuda       (JNI shim → libtq_cuda.so)
  tq-backend-hip        (JNI shim → libtq_hip.so)
        │  calls C functions declared in tq_native_api.h
        ▼
  Native shared library
        │  tq_session_infer(), tq_malloc(), tq_quantise_int8(), …
        ▼
  Actual computation (CPU mock / CUDA kernels / HIP kernels)
```

At every step, only the interfaces from `tq-runtime-api` cross layer boundaries. No vendor class, no CUDA type, no HIP header ever appears above the dashed line between the JNI bridge and the native library.

---

## The four Java layers

### 1. `tq-runtime-api` — stable public interfaces

This module defines what the outside world sees. It contains:

| Type | Role |
|------|------|
| `TurboQuantRuntime` | Entry point; `infer(request)` + `kvCacheStats()` |
| `Backend` | Lifecycle: `init(config)`, `newSession()`, `close()` |
| `ComputeSession` | Per-request work: `infer()`, `allocate()`, `upload()`, `synchronize()`, `close()` |
| `TensorHandle` | Opaque handle to device-resident memory; `AutoCloseable` |
| `InferenceRequest` | Immutable request value: token IDs, max new tokens, temperature |
| `InferenceResult` | Immutable result value: generated IDs, logits, timing, backend name |
| `KvCacheStats` | Java 17 record: cached tokens, capacity, bytes, hit rate |
| `BackendCapability` | Enum of optional features (INT8_MATMUL, MULTI_STREAM, …) |
| `BackendConfig` | Immutable config: device index, extra properties |

**Rule:** This module has zero GPU vendor imports. It never will. Java engineers who write application code only need to import types from here.

### 2. `tq-runtime-spi` — extension point

This module defines `BackendProvider`, the interface that each backend implements to register itself:

```java
public interface BackendProvider {
    String  backendName();   // e.g. "cpu-stub", "cuda", "hip"
    int     priority();      // 0=CPU, 80=GPU; higher wins
    boolean isAvailable();   // checked before create()
    Backend create();        // factory, no init yet
}
```

This is the only thing a new backend author needs to implement to plug in to the runtime.

### 3. `tq-runtime-core` — orchestration

This module contains no GPU code. It provides:

- `BackendRegistry` — loads all `BackendProvider` implementations via `ServiceLoader.load()`, sorts by descending priority, filters by `isAvailable()`. Zero compile-time dependency on any backend JAR.
- `DefaultTurboQuantRuntime` — creates a `Backend`, calls `init()`, keeps a long-lived `ComputeSession`, delegates `infer()` to it. Thread-safe, idempotent `close()`.

**Why `ServiceLoader`?** It is the standard Java extension mechanism (used by JDBC drivers, JCE providers, etc.) and requires no reflection hacks or Spring context. Each backend ships a file at `META-INF/services/com.turboquant.runtime.spi.BackendProvider` that lists its provider class. `tq-runtime-core` discovers backends at runtime by scanning the classpath — adding or removing a backend JAR is the only change needed.

### 4. `tq-backend-*` — implementations

Each backend module implements `Backend`, `ComputeSession`, and `BackendProvider`:

| Module | What it does |
|--------|-------------|
| `tq-backend-cpu-stub` | Pure Java. LCG token generator, simulated KV cache. Always available. Used for development and CI. |
| `tq-backend-cuda` | JNI shim. Loads `libtq_cuda.so`. Currently a placeholder skeleton. |
| `tq-backend-hip` | JNI shim. Loads `libtq_hip.so`. Disabled by default. Currently a placeholder skeleton. |

---

## The native C ABI

### Why a C ABI at all?

Without an explicit ABI contract, each backend's JNI bridge would contain vendor-specific types (JCuda classes, hipJava classes, etc.) and the bridges would diverge structurally. Adding a third backend (Metal, Vulkan, …) would require inventing a new bridge shape from scratch.

The file `include/tq_native_api.h` is a small, stable set of C-linkage functions that **all** native backends implement:

```c
// High-level handles (opaque 64-bit integers to Java)
typedef uint64_t tq_runtime_t;
typedef uint64_t tq_session_t;

// Lifecycle
tq_status_t tq_runtime_create(int32_t device_index, tq_runtime_t* rt_out);
void        tq_runtime_destroy(tq_runtime_t rt);
tq_status_t tq_session_create(tq_runtime_t rt, tq_session_t* session_out);
void        tq_session_destroy(tq_session_t session);

// Inference
tq_status_t tq_session_infer(tq_session_t session,
                              const int32_t* input_token_ids, int32_t input_count,
                              int32_t max_new_tokens, tq_infer_result_t* result_out);

// Low-level tensor ops
tq_status_t tq_malloc(tq_device_ptr_t* ptr_out, size_t bytes);
tq_status_t tq_quantise_int8(tq_device_ptr_t dst, tq_device_ptr_t src,
                               size_t numel, tq_device_ptr_t scale_out, tq_stream_t stream);
// …
```

Java sees only `long` handles and `int` status codes. No vendor header ever leaks into the JVM.

### How CUDA and HIP share the same ABI

Both `tq-backend-cuda` and `tq-backend-hip` call the same set of function names through their respective JNI bridges. The JNI bridge for CUDA is:

```c
#define TQ_JNI(name)  Java_com_turboquant_backend_cuda_CudaNativeBridge_##name
```

The JNI bridge for HIP is:

```c
#define TQ_JNI(name)  Java_com_turboquant_backend_hip_HipNativeBridge_##name
```

That one-line macro difference is the **only** structural difference between the two bridges. The function bodies are identical.

The C implementation files (`tq_native_mock.c` for now, real kernels later) implement the same `tq_*` function names. CUDA uses `cudaMalloc`; HIP uses `hipMalloc`. The mechanical translation tool `hipify-perl` handles ~99% of this.

### The result-handle pattern

`tq_session_infer` populates a heap-allocated `tq_infer_result_t` and returns a pointer (as `jlong`). Individual getter functions extract fields from that pointer. `tq_result_free` releases it. This avoids marshaling a complex struct across the JNI boundary in one call and keeps each JNI function trivially testable.

---

## Why the HIP backend is disabled by default

`HipNativeBridge` triggers `System.loadLibrary("tq_hip")` in its static initializer. If the library is absent the JVM throws `UnsatisfiedLinkError`. On a CUDA-only host that has the HIP JAR on the classpath — for example in a fat JAR bundling all backends — this throw would happen at class-load time and could confuse startup diagnostics.

The flag `-Dtq.backend.hip.enabled=true` gates the load attempt entirely. Unless the flag is set, the HIP backend never attempts to load the native library and `isAvailable()` returns `false` silently.

---

## Key design decisions

### `long` for device pointers

JNI maps Java `long` ↔ C `jlong` ↔ C `int64_t`. A 64-bit integer is wide enough for any device address on current and foreseeable hardware. Using `long` avoids object allocation on every memory operation. The null device pointer is `0L`.

### `BackendConfig` is immutable

Configuration is frozen at `Backend.init()`. Mutable config would require either defensive copying or synchronisation across concurrent sessions, and there is no use case that requires changing device config at runtime.

### `TensorHandle` is `AutoCloseable`

Device memory is a finite resource. Making handles `AutoCloseable` makes try-with-resources the idiomatic usage pattern and makes leaks visible in code review. `close()` is idempotent — safe to call twice.

### `ComputeSession` maps to one stream

Each `ComputeSession` owns one CUDA/HIP stream internally (one `hipStream_t` / `cudaStream_t` in the real implementation). Creating a session per request allows concurrent requests to run on independent streams without synchronizing against each other.

### No `module-info.java` yet

JPMS modules would be the natural next step but add friction during the skeleton phase: every transitive dependency needs `opens` and `requires` declarations. Module descriptors will be added once the public API stabilises.

---

## ROCm porting cost summary

| Layer | Changes for ROCm port |
|-------|-----------------------|
| `tq-runtime-api` | None |
| `tq-runtime-spi` | None |
| `tq-runtime-core` | None |
| `tq-backend-cpu-stub` | None |
| `tq-backend-cuda` | None |
| `tq-backend-hip` | None — Java structure already correct |
| `tq_native_api.h` | None — shared header |
| `native/hip/tq_jni_bridge.c` | None — structure already correct |
| `native/hip/tq_native_mock.c` | Replace with real HIP kernels |
| Application / Spring Boot | None |

The entire porting effort is concentrated in one file: the native HIP implementation that calls `hipMalloc`, `hipLaunchKernelGGL`, etc. Everything above it is reused unchanged.

See [docs/rocm-porting-plan.md](rocm-porting-plan.md) for the step-by-step porting checklist.
