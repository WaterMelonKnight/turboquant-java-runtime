# llama.cpp demo path

This document describes the `tq-backend-llamacpp` module — the current experimental real-inference path in `turboquant-java-runtime`.

---

## What this path proves

- The backend-agnostic Java API (`tq-runtime-api`) is sufficient to run a real forward pass through a local language model.
- `InferenceRequest.fromText(prompt, maxNewTokens)` routes correctly through the Java runtime to a native inference backend without any fake token IDs.
- The full call stack — `BenchCli` → `DefaultTurboQuantRuntime` → `LlamaCppBackend` → `LlamaCppSession` → `de.kherud:llama` → native llama.cpp — works end to end.
- The CPU stub and llama.cpp backends coexist in the same fat JAR without interfering with each other.

This is **not** a production-ready path and **not** a TurboQuant algorithm implementation. It is an experimental demo that validates the architecture.

---

## Prerequisites

- Java 17+
- Maven 3.8+
- A local GGUF model file (see below)
- No GPU required — the validated mode is CPU-only

The `de.kherud:llama` JAR bundles native llama.cpp binaries for Linux (x86_64 and aarch64), macOS (x86_64 and arm64), and Windows. No separate llama.cpp installation is required.

---

## Getting a GGUF model

Any GGUF-format model compatible with the bundled llama.cpp version works.
The following models have been used during development:

| Model | Size | Quantization | Source |
|-------|------|--------------|--------|
| Qwen2.5-0.5B-Instruct | ~400 MB | Q4_0 | Hugging Face |
| TinyLlama-1.1B | ~700 MB | Q4_0 | Hugging Face |
| Phi-3-mini | ~2.2 GB | Q4_K_M | Hugging Face |

Download with `huggingface-cli` (requires `pip install huggingface_hub`):

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_0.gguf \
    --local-dir ./models
```

Or download the `.gguf` file directly from the model's **Files** tab on Hugging Face.

Place the file anywhere on your local filesystem. You will pass the absolute path to the CLI or test.

---

## Building the fat JAR

```bash
# From the repository root
mvn clean package -pl tq-bench-cli -am -q
```

This produces `tq-bench-cli/target/tq-bench-cli-*-fat.jar`, which includes all backends.

---

## Running tq-bench-cli with llama.cpp

```bash
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --context 2048 \
  --warmup 1 \
  --iters 3
```

Expected output structure:

```
==============================================================
  TurboQuant Bench CLI  v0.1.0-SNAPSHOT  [llama.cpp mode]
==============================================================
  Backend         : llama.cpp
  Model path      : /path/to/model.gguf
  Context tokens  : 2048
  Max new tokens  : 64
  Prompt          : The capital of France is
  Warmup iters    : 1
  Timed iters     : 3
==============================================================
Warming up (1 iteration)...
Running benchmark (3 iterations)...
--------------------------------------------------------------
  Latency (ms) over 3 iterations:
    mean  :   XXXX.XXX ms
    ...
  Throughput    :       XX.X tok/s (generated)
--------------------------------------------------------------
  Generated tokens : NN
  Backend          : llama.cpp
--------------------------------------------------------------
  Last generated text:
  Paris ...
==============================================================
  DEMO SUMMARY
==============================================================
  backend    : llama.cpp (experimental, CPU-only validated)
  throughput : XX.X tok/s (measured, NN token run)
  note       : CUDA/HIP are separate future backend tracks
==============================================================
```

The throughput figure is measured from actual inference time on your machine. It is not projected or interpolated.

---

## Running the optional smoke test

The smoke test in `tq-backend-llamacpp` is skipped automatically when no model path is provided. It is safe to run in CI without a model file.

```bash
# Skipped by default — no model needed
mvn test -pl tq-backend-llamacpp -am
# Output includes: "Skipping llama.cpp smoke test — set -Dtq.test.llamacpp.model=..."

# Run with a model file
mvn test -pl tq-backend-llamacpp -am \
    -Dtq.test.llamacpp.model=/path/to/model.gguf
```

Optional overrides (all have defaults):

| System property | Default | Description |
|-----------------|---------|-------------|
| `tq.test.llamacpp.model` | *(none — test skips)* | Path to GGUF model file |
| `tq.test.llamacpp.context` | `2048` | Context window size in tokens |
| `tq.test.llamacpp.maxNewTokens` | `32` | Max tokens to generate |
| `tq.test.llamacpp.prompt` | `Once upon a time` | Text prompt |

What the smoke test asserts:

- `InferenceResult` is not null
- `generatedText()` is not null and not blank
- `generatedTokenCount()` > 0
- `inferenceNanos()` > 0
- `backendName()` == `"llama.cpp"`

It prints a brief benchmark summary including latency, throughput, and the generated text.

---

## Current limitations

- **CPU-only validated.** GPU offload via llama.cpp (`nGpuLayers > 0`) is now configurable through `--gpu-layers`, but actual GPU activation depends on the native llama.cpp build bundled in `de.kherud:llama`. A CPU-only build silently ignores the setting. The backend logs a warning when GPU offload is requested but the CPU-only signal is detected.
- **GPU offload status is best-effort.** The backend infers offload status from llama.cpp log output. If no CPU-only warning is detected, `gpuOffloadActive` is reported as `null` (unknown) rather than `true`, because the Java binding does not expose a direct API to confirm GPU utilisation.
- **No tokenizer metrics.** The `de.kherud:llama` binding does not surface prompt token counts, so `promptTokenCount()` is reported as 0.
- **No KV cache metrics.** llama.cpp manages its KV cache internally; the Java layer always receives `KvCacheStats.empty()`.
- **Experimental stability.** Model loading errors, context overflow, and native crashes are possible with edge-case inputs or very large models.

---

## GPU offload configuration

GPU layer offload is requested via `--gpu-layers`. This maps directly to llama.cpp's `nGpuLayers` parameter.

```bash
# CPU-only (default — no GPU offload)
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64

# Request GPU offload for 32 layers
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64 \
  --gpu-layers 32

# Request offload for all layers
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64 \
  --gpu-layers -1
```

### How to tell whether GPU offload actually activated

Check the backend startup logs:

- **CPU-only build (offload unavailable):**
  ```
  WARN  LlamaCppBackend - GPU offload was requested (32 layers) but the native build
        does not support it. Running CPU-only.
  ```
  The exported result will have `gpuOffloadActive: false`.

- **GPU-capable build (offload may be active):**
  ```
  INFO  LlamaCppBackend - GPU offload requested (32 layers); no CPU-only warning detected.
        Actual GPU utilisation depends on hardware and driver availability.
  ```
  The exported result will have `gpuOffloadActive: null` (unknown — the binding does not confirm GPU use directly).

- **CPU-only run (not requested):**
  ```
  INFO  LlamaCppBackend - CPU-only (GPU offload not requested)
  ```
  The exported result will have `gpuOffloadRequested: false`, `gpuOffloadActive: false`.

### Important distinction

`--gpu-layers` controls llama.cpp's internal GPU offload mechanism. This is **not** the same as the future `tq-backend-cuda` or `tq-backend-hip` backends, which use a separate C ABI (`include/tq_native_api.h`) and are entirely independent backend tracks. See the table below.

---

## Relationship to CUDA/HIP backends

The llama.cpp path is entirely independent of `tq-backend-cuda` and `tq-backend-hip`.

| Backend | Status | How it works |
|---------|--------|--------------|
| `cpu-stub` | Working | Deterministic LCG, simulated KV cache. No model file. |
| `llama.cpp` | Working (CPU validated; GPU offload configurable) | Real GGUF model via `de.kherud:llama`. GPU offload via `--gpu-layers`; actual activation depends on native build. |
| `cuda` | Placeholder | Java shape + JNI bridge + mock C ABI. No real GPU kernels. |
| `hip` | Placeholder | Same as CUDA, disabled by default. No real GPU kernels. |

CUDA and HIP backends use a custom C ABI (`include/tq_native_api.h`) and are completely separate backend tracks. Implementing real CUDA or HIP kernels requires no changes to the Java API or to the llama.cpp path. The `--gpu-layers` option has no effect on the cuda or hip backends.

---

## Known-good validation run

The following single run was performed to confirm the path works. It is not a performance benchmark.

| Property | Value |
|----------|-------|
| Model | Qwen2.5-0.5B-Instruct (Q4_0 GGUF) |
| Context | 2048 tokens |
| Max new tokens | 32 |
| Warmup iterations | 2 |
| Timed iterations | 10 |
| Mean latency | 3559.747 ms |
| Throughput | ~9.3 tok/s |
| Execution mode | CPU-only |

Results will vary by machine and model.
