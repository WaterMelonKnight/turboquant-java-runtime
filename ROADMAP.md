# Roadmap

This document tracks planned work for `turboquant-java-runtime`.
Items are grouped by phase; estimated scope is noted but no delivery dates are promised.
The project is pre-alpha — phases may be reordered or reprioritised.

---

## v0.1 — Architecture baseline (done)

**Goal:** Establish a working multi-module structure that compiles, tests, and demonstrates the full layering from Java API down to a native C ABI, without requiring any GPU hardware.

- [x] Java 17 multi-module Maven build
- [x] Stable public API (`tq-runtime-api`): `Backend`, `ComputeSession`, `TensorHandle`, `InferenceRequest`, `InferenceResult`, `KvCacheStats`, `SessionConfig`
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

## v0.2 — Real inference path (in progress)

**Goal:** Provide a working end-to-end inference path against a real language model. The llama.cpp path is the primary experimental real-inference track.

- [x] `tq-backend-llamacpp` module using `de.kherud:llama` Java binding
- [x] `SessionConfig` with `modelPath`, `maxContextTokens`, `maxNewTokens`, `temperature`, `topP`
- [x] `InferenceRequest.fromText(prompt, maxNewTokens)` — text-based requests need no placeholder token IDs
- [x] Two-mode `InferenceRequest` design: token-based (cpu-stub, cuda, hip) and text-based (llama.cpp)
- [x] `InferenceResult.generatedText()` for raw text output
- [x] Bench CLI `--model-path`, `--prompt`, `--context`, `--max-new-tokens` options
- [x] Bench CLI mode labels and summary block for each backend type
- [x] `LlamaCppBackend` file-existence check and clear error messages for common failures
- [x] Optional `LlamaCppSmokeTest` (skipped unless `-Dtq.test.llamacpp.model=...` is set)
- [x] Smoke run validated: Qwen2.5-0.5B-Instruct Q4_0, CPU-only, ~9 tok/s
- [x] `docs/llamacpp-demo.md` — full demo guide for the llama.cpp path
- [ ] GPU offload via llama.cpp `nGpuLayers` (requires a system with a supported GPU)
- [ ] Proper KV cache metrics surfaced from llama.cpp internals
- [ ] Tokenizer integration for prompt token count reporting
- [ ] Integration test with a bundled tiny model (no large file dependency)

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
