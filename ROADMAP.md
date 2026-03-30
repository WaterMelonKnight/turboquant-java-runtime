# Roadmap

This document tracks planned work for `turboquant-java-runtime`.
Items are grouped by phase; estimated scope is noted but no delivery dates are promised.
The project is pre-alpha — phases may be reordered or reprioritised.

---

## v0.1 — Architecture baseline (current)

**Goal:** Establish a working multi-module structure that compiles, tests, and demonstrates the full layering from Java API down to a native C ABI, without requiring any GPU hardware.

- [x] Java 17 multi-module Maven build
- [x] Stable public API (`tq-runtime-api`): `Backend`, `ComputeSession`, `TensorHandle`, `InferenceRequest`, `InferenceResult`, `KvCacheStats`
- [x] `BackendProvider` SPI and `ServiceLoader`-based backend discovery
- [x] `DefaultTurboQuantRuntime` with auto-selection and named-backend lookup
- [x] `tq-backend-cpu-stub`: deterministic LCG inference, simulated KV cache
- [x] `tq-backend-cuda`: placeholder Java shape, JNI bridge, mock C implementation
- [x] `tq-backend-hip`: placeholder Java shape, JNI bridge, mock C implementation (disabled by default)
- [x] Stable C ABI header (`include/tq_native_api.h`) — runtime, session, infer, low-level tensor ops
- [x] Native mock implementations (`native/cuda/`, `native/hip/`) — compilable with CMake, no GPU required
- [x] Bench CLI (`tq-bench-cli`) running against CPU stub
- [x] Spring Boot 3 auto-configuration starter
- [x] Unit tests: 63 tests, 0 failures
- [x] Architecture docs and ROCm porting plan

---

## v0.2 — Real inference path

**Goal:** Replace the CPU stub's LCG placeholder with an actual forward pass — even a trivial one — so the project can demonstrate real model input/output.

- [ ] Tokenizer integration (SentencePiece or HuggingFace tokenizers-java, TBD)
- [ ] Model weight loading (GGUF or SafeTensors reader, TBD)
- [ ] Real CPU forward pass for a small transformer (e.g. TinyStories-class model)
- [ ] `InferenceRequest` driven by real text input, not synthetic token IDs
- [ ] Proper KV cache implementation in the CPU path
- [ ] Update bench CLI to accept text prompts
- [ ] Integration test: load weights → tokenize → infer → detokenize → assert non-empty output

---

## v0.3 — Experimental quantization integration

**Goal:** Demonstrate at least one quantization path working end-to-end on CPU, establishing the pattern for later GPU kernel integration.

- [ ] INT8 weight quantization in `tq-backend-cpu-stub` (symmetric per-tensor)
- [ ] Accuracy comparison: FP32 baseline vs INT8 on a small reference model
- [ ] `BackendCapability` flags wired to actual runtime behavior, not just declared
- [ ] Quantization benchmark in `tq-bench-cli` (latency and accuracy deltas)

---

## v0.4 — Real CUDA path

**Goal:** Replace the CUDA mock with real GPU kernels so `libtq_cuda.so` runs actual computation.

- [ ] CUDA kernel for INT8 matrix multiplication (the core of quantized inference)
- [ ] CUDA memory management using real `cudaMalloc` / `cudaFree`
- [ ] `tq_session_infer` running a real forward pass on GPU
- [ ] CI gate: skip CUDA tests when no GPU is present (`-Pcuda-tests`)
- [ ] Benchmark comparison: CPU stub vs CUDA on common sequence lengths

---

## v0.5 — HIP/ROCm implementation

**Goal:** Demonstrate that the shared C ABI delivers on its ROCm portability promise.

- [ ] Run `hipify-perl` on `native/cuda/` to produce `native/hip/tq_hip_impl.cpp`
- [ ] Build `libtq_hip.so` with a real HIP toolchain
- [ ] Verify token output matches CUDA output for same seed (deterministic test)
- [ ] CI gate: skip HIP tests when no ROCm device is present (`-Phip-tests`)
- [ ] Document the diff between `libtq_cuda.so` and `libtq_hip.so` build

---

## Later — Operations and deployment

- [ ] Structured logging (MDC, backend name, latency per request)
- [ ] Micrometer metrics: inference latency histogram, KV cache occupancy gauge
- [ ] Kubernetes deployment example: Spring Boot app + GPU node selector
- [ ] Graceful multi-backend fallback (CUDA → CPU) if GPU is unavailable at startup
- [ ] `JPMS` (`module-info.java`) descriptors once API stabilises

---

## Possible future — Halo Cloud integration

This project may later serve as the inference runtime tier for Halo Cloud.
No timeline or scope has been defined. Items that would support this:

- [ ] gRPC inference endpoint compatible with Halo Cloud routing layer
- [ ] Multi-tenant session isolation
- [ ] Backend health-check and hot-swap without restart
- [ ] Cost-efficient batching for concurrent inference requests

---

## Not planned

The following are explicitly out of scope for this repository:

- A full production TurboQuant algorithm implementation
- Model training or fine-tuning
- A Python API
- Anything requiring proprietary Google or internal materials

---

> Roadmap items may be added, removed, or reprioritised at any time.
> This is a pre-alpha project; treat this document as intent, not commitment.
