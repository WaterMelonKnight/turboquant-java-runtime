# Architecture: TurboQuant Java Runtime

## Overview

TurboQuant Java Runtime is designed around one central constraint:

> **The Java layer must never contain vendor-specific GPU code.**

All GPU operations are pushed behind a stable C ABI (`tq_native_api.h`),
then exposed to Java via thin JNI bridges.  The Java API layer is entirely
backend-agnostic.  Swapping CUDA for HIP requires only a new `.so` —
zero Java changes.

---

## Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Application / Spring Boot                   │
│           (only imports tq-runtime-api types)                 │
└────────────────────────┬─────────────────────────────────────┘
                         │  Backend / ComputeSession / TensorHandle
┌────────────────────────▼─────────────────────────────────────┐
│                    tq-runtime-core                            │
│   BackendRegistry  ──  ServiceLoader  ──  RuntimeEngine       │
└────────────────────────┬─────────────────────────────────────┘
                         │  BackendProvider SPI
          ┌──────────────┼──────────────┐
          │              │              │
┌─────────▼───┐  ┌───────▼───┐  ┌──────▼────────┐
│ cpu-stub    │  │  cuda     │  │  hip          │
│ (pure Java) │  │  (JNI)    │  │  (JNI)        │
└─────────────┘  └─────┬─────┘  └──────┬────────┘
                        │               │
                ┌───────▼───────────────▼──────────┐
                │        C ABI: tq_native_api.h     │
                └───────┬───────────────┬──────────┘
                        │               │
               ┌────────▼──┐   ┌────────▼──┐
               │libtq_cuda │   │libtq_hip  │
               │  (.so)    │   │  (.so)    │
               └───────────┘   └───────────┘
```

---

## Why a C ABI?

### Problem: vendor lock-in at the Java boundary

Without an ABI contract, the CUDA backend JAR would contain
`import jcuda.*` or similar vendor classes.  Adding HIP would mean
duplicating the entire JNI bridge code.

### Solution: thin C ABI as the portability seam

`tq_native_api.h` defines a small, stable set of C-linkage functions:

```c
tq_status_t tq_init(int32_t device_index);
tq_status_t tq_malloc(tq_device_ptr_t* ptr_out, size_t bytes);
tq_status_t tq_quantise_int8(tq_device_ptr_t dst, ...);
// ...
```

The Java JNI bridge calls these symbols by name.  The **only** difference
between the CUDA and HIP backends is:

1. The shared library name (`libtq_cuda.so` vs `libtq_hip.so`).
2. The C implementation file, which can be produced mechanically:

```bash
hipify-perl libtq_cuda.cu > libtq_hip.cu
# ~99% of the diff is s/cuda/hip/g and s/CUDA/HIP/g
```

**No Java changes are needed when porting to HIP.**

### Why not use an existing Java GPU binding?

Options like JCuda, JOCL, or TorchScript have their own object models and
lifecycle assumptions.  TurboQuant kernels are specialised enough that a
thin bespoke ABI is simpler and more controllable.

---

## Why ServiceLoader for backend discovery?

### Requirement

`tq-runtime-core` must not have a compile-time dependency on any backend JAR.
Adding a new backend must not require changes to core.

### Solution: Java ServiceLoader

Each backend JAR ships a service descriptor:

```
META-INF/services/com.turboquant.runtime.spi.BackendProvider
```

`BackendRegistry` calls `ServiceLoader.load(BackendProvider.class)` at startup.
All registered providers are ranked by `priority()` and filtered by
`isAvailable()`.  The highest-priority available backend is selected
automatically.

This is the standard Java extension mechanism (used by JDBC, JCE, etc.).
It requires no reflection hacks, no Spring context, and no configuration files
beyond the service descriptor that each backend already owns.

### Provider contract

```java
public interface BackendProvider {
    String  backendName();   // stable identifier
    int     priority();      // 0 = CPU, 80 = GPU
    boolean isAvailable();   // checked before create()
    Backend create();        // factory; no init yet
}
```

---

## Why the HIP backend is disabled by default

The HIP JNI bridge class-loads `libtq_hip.so` in its static initialiser.
If the library is absent, the JVM throws `UnsatisfiedLinkError` and the
backend's `isAvailable()` returns `false` — safe to ignore.

However, on a CUDA-only host that also has the HIP JAR on the classpath
(e.g. a fat-jar that bundles all backends), attempting to load `libtq_hip.so`
may cause a spurious warning or even a crash if a partial ROCm installation
is present.  The flag `-Dtq.backend.hip.enabled=true` gates the load attempt,
so the HIP path is completely inert unless explicitly requested.

---

## ROCm porting cost analysis

| Artifact             | Change needed for ROCm port   | Effort |
|----------------------|-------------------------------|--------|
| `tq-runtime-api`     | None                          | 0      |
| `tq-runtime-spi`     | None                          | 0      |
| `tq-runtime-core`    | None                          | 0      |
| `tq-backend-cpu-stub`| None                          | 0      |
| `tq-backend-cuda`    | None                          | 0      |
| `tq-backend-hip`     | Implement `libtq_hip.so`      | medium |
| `tq_native_api.h`    | None (shared between backends)| 0      |
| App / Spring Boot    | None                          | 0      |

The only real work is writing `libtq_hip.cu` (or `.cpp` with hipify output)
that implements the ~12 functions declared in `tq_native_api.h`.  Every other
layer is reused unchanged.

---

## Key design decisions

### 1. No `module-info.java` (yet)

JPMS modules would be a natural next step but add friction during the
skeleton phase (every transitive dependency needs `opens`/`requires`).
Adding module descriptors is a safe incremental change once the API
stabilises.

### 2. `long` for device pointers

JNI maps `jlong` ↔ Java `long` ↔ C `int64_t`.  Using `long` for device
pointers avoids object allocation and works on all 64-bit platforms.
The null device pointer is `0L`.

### 3. `BackendConfig` is immutable

Configuration is frozen at `Backend#init()` time.  Mutating config at
runtime would require synchronisation across sessions and is not needed
for the current use cases.

### 4. Sessions are lightweight

`ComputeSession` maps 1:1 to a CUDA stream / HIP stream.
Creating and destroying sessions is cheap (stream creation is ~µs).
Applications should create one session per concurrent request rather
than sharing a single session.

### 5. `TensorHandle` is `AutoCloseable`

Device memory is a scarce resource.  Making handles `AutoCloseable`
encourages try-with-resources usage and makes leaks visible in code review.
