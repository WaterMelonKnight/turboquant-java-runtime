# llama.cpp smoke test

This document explains how to run real inference using the `tq-backend-llamacpp` module against a local GGUF model file.

---

## Prerequisites

- Java 17+
- Maven 3.8+
- A local GGUF model file (see below)
- No GPU required — current validated mode is CPU-only

The `de.kherud:llama` JAR bundles native llama.cpp binaries for Linux (x86_64/aarch64), macOS (x86_64/arm64), and Windows. No separate llama.cpp installation is needed.

---

## Getting a GGUF model

Any GGUF-format model compatible with the bundled llama.cpp version should work.
The models listed here were used during development:

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

Or download directly from the model's Files tab on Hugging Face.

---

## Running the bench CLI manually

```bash
# Build the fat JAR (includes all backends)
mvn clean package -pl tq-bench-cli -am -q

# Run with a local model
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --context 2048 \
  --warmup 2 \
  --iters 5
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
  Warmup iters    : 2
  Timed iters     : 5
==============================================================
Warming up (2 iterations)...
Running benchmark (5 iterations)...
--------------------------------------------------------------
  Latency (ms) over 5 iterations:
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
```

---

## Running the optional smoke test

The smoke test in `tq-backend-llamacpp` is skipped by default when no model path is configured. It is safe to run in CI without a model file.

**Run with a model file:**

```bash
mvn test -pl tq-backend-llamacpp -am \
    -Dtq.test.llamacpp.model=/path/to/model.gguf
```

**Optional overrides:**

| Property | Default | Description |
|----------|---------|-------------|
| `tq.test.llamacpp.model` | *(none — test skips)* | Path to GGUF model file |
| `tq.test.llamacpp.context` | `2048` | Context window size in tokens |
| `tq.test.llamacpp.maxNewTokens` | `32` | Max tokens to generate |
| `tq.test.llamacpp.prompt` | `Once upon a time` | Text prompt |

**What the smoke test asserts:**

- `InferenceResult` is not null
- `generatedText()` is not null and not blank
- `generatedTokenCount()` > 0
- `inferenceNanos()` > 0
- `backendName()` == `"llama.cpp"`

It prints a summary including latency, throughput, and the generated text.

**Running all tests (smoke test skipped automatically):**

```bash
mvn test
# All 63 tests pass; smoke test is skipped with:
# "Skipping llama.cpp smoke test — set -Dtq.test.llamacpp.model=..."
```

---

## Known-good validation

The following run was used to validate the path for the first time:

| Property | Value |
|----------|-------|
| Model | Qwen2.5-0.5B-Instruct (Q4_0 GGUF) |
| Context | 2048 tokens |
| Max new tokens | 32 |
| Warmup iterations | 2 |
| Timed iterations | 10 |
| Mean latency | 3559.747 ms |
| P50 latency | 3499.755 ms |
| Throughput | ~9.3 tok/s |
| Execution mode | CPU-only (no GPU offload) |
| GPU offload | Not used — "Not compiled with GPU offload support" warning expected |

---

## Relationship to CUDA/HIP backends

The llama.cpp path is **independent** of the `tq-backend-cuda` and `tq-backend-hip` modules. Those modules implement the custom `tq_native_api.h` C ABI with JNI bridges and are currently placeholder structures with no real GPU kernels.

| Backend | Status | Notes |
|---------|--------|-------|
| `cpu-stub` | Working | Deterministic LCG, simulated KV cache. No real model. |
| `llama.cpp` | Working (CPU) | Real GGUF model, real text generation. CPU-only validated. |
| `cuda` | Placeholder | Java shape + JNI bridge + mock C ABI. No real GPU kernels. |
| `hip` | Placeholder | Same as CUDA, disabled by default. No real GPU kernels. |

GPU offload via llama.cpp (`nGpuLayers > 0`) is structurally wired but not validated — it requires a system with a compatible GPU and a GPU-enabled native llama.cpp build.
