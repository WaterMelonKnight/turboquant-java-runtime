# turboquant-java-runtime

Experimental Java 17 runtime for TurboQuant-oriented inference backend experiments.

This project aims to provide a **backend-agnostic Java runtime** for future TurboQuant-style inference integration, with an architecture that can evolve across:

- CPU stub
- CUDA backends
- HIP / ROCm backends

The current focus is **runtime architecture and backend extensibility**, not a full production TurboQuant implementation.

---

## Status

**Project stage:** Pre-alpha / experimental

### What works today

- Java 17 multi-module runtime structure
- Backend-agnostic public API
- `ServiceLoader`-based backend discovery
- Pure Java `cpu-stub` backend for local development
- CUDA backend placeholder with native bridge shape
- HIP / ROCm backend placeholder with matching structure
- Bench CLI skeleton
- Stable shared native C ABI header: `tq_native_api.h`

### What does *not* exist yet

- Full TurboQuant algorithm implementation
- Production-ready CUDA kernels
- Production-ready HIP / ROCm kernels
- Real model inference integration
- Verified end-to-end TurboQuant benchmarks
- Production stability guarantees

If you are looking for a ready-to-use TurboQuant implementation, this repository is **not there yet**.

---

## Why this project exists

Most Java AI projects stop at the application layer.

This repository explores a different direction:

- a Java-callable runtime layer
- pluggable native backends
- unified API across CPU / CUDA / HIP
- future support for quantisation-oriented inference experiments
- an architecture that can later be integrated into larger systems such as **Halo Cloud**

The goal is to make it easier to experiment with **Java + native inference backends + future TurboQuant-style optimization paths** without binding the Java API to one specific GPU vendor.

---

## Design goals

- **Backend-agnostic Java API**
- **Low-cost future ROCm porting**
- **Stable native ABI**
- **Clean separation between API, SPI, core runtime, and native backends**
- **CPU-first local development path**
- **CUDA-first validation path**
- **HIP / ROCm expansion path later**

---

## Architecture

At a high level, the project is split into four layers:

### 1. Public API
Java-facing interfaces and data models.

Examples:
- `Backend`
- `ComputeSession`
- `TensorHandle`
- `BackendConfig`

### 2. SPI
Backend extension points discovered via `ServiceLoader`.

### 3. Runtime core
Backend selection, discovery, lifecycle, and orchestration.

### 4. Native backends
Vendor-specific or runtime-specific implementations:
- `tq-backend-cpu-stub`
- `tq-backend-cuda`
- `tq-backend-hip`

The Java API should remain stable even as native backends evolve.

---

## Project layout

```text
turboquant-java-runtime/
├── pom.xml
├── tq-runtime-api/
├── tq-runtime-spi/
├── tq-runtime-core/
├── tq-backend-cpu-stub/
├── tq-backend-cuda/
├── tq-backend-hip/
├── tq-bench-cli/
├── tq-spring-boot-starter/
└── include/
    └── tq_native_api.h