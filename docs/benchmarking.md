# Benchmarking guide

This document describes the benchmark layer in `tq-bench-cli`, explains what each metric means, and shows how to run, export, and compare results.

---

## Benchmark metric definitions

All metrics are recorded in `BenchmarkMetrics`. Fields that cannot be measured with the current backend binding are always `null` in the output; they are never fabricated.

| Metric | Unit | Definition |
|--------|------|------------|
| `latency_mean_ms` | ms | Mean of per-iteration total inference latency |
| `latency_min_ms` | ms | Minimum per-iteration latency |
| `latency_p50_ms` | ms | 50th-percentile per-iteration latency |
| `latency_p99_ms` | ms | 99th-percentile per-iteration latency |
| `latency_max_ms` | ms | Maximum per-iteration latency |
| `generated_tokens_last_run` | tokens | Actual tokens generated in the **last** timed iteration |
| `generated_tokens_per_second_mean` | tok/s | Mean of per-iteration (tokens / latency_in_seconds) |
| `prompt_tokens_per_second` | tok/s | **Always null** — prompt-eval timing not exposed by `de.kherud:llama` |
| `model_load_time_ms` | ms | **Always null** — included in `wall_clock_ms` instead |
| `wall_clock_ms` | ms | Total elapsed time from before runtime creation to after close |

### What "total inference latency" covers

For **llama.cpp**: `inferenceNanos` is measured from the first call to `model.generate()` until all tokens are returned. This covers both prompt evaluation and token generation in one pass. The two phases cannot be separated without instrumentation inside the native library.

For **cpu-stub**: it is the stub LCG computation only — no real model, no real prompt evaluation.

### Why `generated_tokens_per_second_mean` is computed per-iteration

The throughput is computed as the mean of per-iteration values:

```
tok/s_i = generated_tokens_i / (latency_ms_i / 1000.0)
mean_tok/s = mean(tok/s_1, tok/s_2, ..., tok/s_n)
```

This avoids the bias introduced by `mean_tokens / mean_latency` (Jensen's inequality). Both numerator and denominator are taken from the same iteration.

### What `wall_clock_ms` covers

Starts immediately before `DefaultTurboQuantRuntime.withBackend()` (which loads the model for llama.cpp) and ends after `runtime.close()`. It encompasses:
- Model loading (llama.cpp)
- All warmup iterations
- All timed iterations
- Runtime cleanup

Useful as an upper bound on total end-to-end cost.

---

## Stub runs vs. real inference runs

| Field | cpu-stub | llama.cpp (CPU) | llama.cpp (GPU offload requested) | cuda/hip placeholder |
|-------|----------|-----------------|-----------------------------------|----------------------|
| `is_real_inference` | `false` | `true` | `true` | `false` |
| `backend_mode` | `CPU_STUB` | `LLAMA_CPP_REAL` | `LLAMA_CPP_REAL` | `CUDA_PLACEHOLDER` / `HIP_PLACEHOLDER` |
| `gpu_offload_requested` | `false` | `false` | `true` | `false` |
| `gpu_layers_requested` | null | null | e.g. `32` or `-1` | null |
| `gpu_offload_active` | `false` | `false` | `null` (unknown) or `false` (CPU-only build) | `false` |
| Real model | No | Yes (GGUF) | Yes (GGUF) | No |
| Real GPU computation | No | No | Depends on native build | No |

`gpu_offload_active` is `null` (unknown) when GPU offload was requested and no CPU-only warning was detected in llama.cpp output. It is `false` when the native build is CPU-only. It is never set to `true` because the Java binding does not expose a direct confirmation of GPU utilisation.

Do not compare throughput numbers between `CPU_STUB` and `LLAMA_CPP_REAL` runs — they measure fundamentally different things.

---

## Running a benchmark manually

### cpu-stub (no model required)

```bash
# Build the fat JAR
mvn clean package -pl tq-bench-cli -am -q

# Run with defaults (128-token prompt, 32 generated, 10 iterations)
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar

# Custom parameters
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend cpu-stub \
  --warmup 5 \
  --iters 20 \
  --prompt-len 256 \
  --gen-len 64
```

### llama.cpp (real GGUF model required)

```bash
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-new-tokens 64 \
  --context 2048 \
  --warmup 1 \
  --iters 5
```

See [docs/llamacpp-demo.md](llamacpp-demo.md) for model download instructions.

---

## Exporting results

### JSON export

```bash
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend cpu-stub \
  --output-json /tmp/bench-$(date +%Y%m%d-%H%M%S).json
```

The JSON file is overwritten if it already exists. Sample output:

```json
{
  "benchmarkVersion": "0.2",
  "runId": "a1b2c3d4",
  "timestampUtc": "2026-04-02T10:00:00Z",
  "gitCommit": "abc1234",
  "backend": {
    "name": "cpu-stub",
    "mode": "CPU_STUB",
    "isRealInference": false,
    "gpuOffloadActive": false
  },
  "model": {
    "path": null,
    "basename": null,
    "quantHint": null
  },
  "prompt": {
    "text": null,
    "length": 128,
    "promptTokenCount": 128
  },
  "config": {
    "requestedMaxNewTokens": 32,
    "contextTokens": 0,
    "warmupIterations": 2,
    "timedIterations": 10
  },
  "metrics": {
    "generatedTokensLastRun": 32,
    "totalLatencyMs": {
      "mean": 0.123,
      "min": 0.089,
      "p50": 0.118,
      "p99": 0.201,
      "max": 0.201
    },
    "generatedTokensPerSecondMean": 260160.000,
    "promptTokensPerSecond": null,
    "modelLoadTimeMs": null,
    "wallClockMs": 45.000
  },
  "environmentNotes": [
    "cpu-stub — no real model; deterministic LCG only"
  ]
}
```

### CSV export

```bash
# First run — creates file with header + row
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend cpu-stub --output-csv /tmp/bench-results.csv

# Second run — appends a row (header is NOT duplicated)
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend cpu-stub --prompt-len 256 --output-csv /tmp/bench-results.csv
```

CSV columns (26):

```
run_id, timestamp_utc, git_commit, backend_name, backend_mode,
is_real_inference, gpu_offload_requested, gpu_layers_requested, gpu_offload_active,
model_basename, quant_hint,
prompt_length, prompt_token_count, requested_max_new_tokens, context_tokens,
warmup_iters, timed_iters, generated_tokens_last_run,
latency_mean_ms, latency_min_ms, latency_p50_ms, latency_p99_ms, latency_max_ms,
generated_tokens_per_second_mean, wall_clock_ms
```

Empty cells mean the value was not available (null) for that backend.

---

## Comparing runs

Each row in the CSV has a `run_id` (8-character UUID prefix) and a `timestamp_utc` that uniquely identify the run. The `git_commit` field — populated when `git` is available — ties the result to a specific code state.

To compare two runs in a spreadsheet or with tools like `csvkit`:

```bash
# Compare two runs by latency
csvcut -c run_id,backend_name,latency_mean_ms,generated_tokens_per_second_mean \
       /tmp/bench-results.csv | csvlook
```

### What makes a fair comparison

Only compare runs with the same:
- `backend_mode` (do not compare `CPU_STUB` to `LLAMA_CPP_REAL`)
- `requested_max_new_tokens`
- `prompt_length`
- `model_basename` and `quant_hint` (for llama.cpp)
- Similar `context_tokens`
- Same `gpu_offload_requested` and `gpu_layers_requested` (for CPU vs GPU offload comparisons)

---

## GPU offload configuration (llama.cpp)

GPU layer offload is requested via `--gpu-layers` on the CLI. This maps to llama.cpp's `nGpuLayers` parameter.

| `--gpu-layers` value | Meaning |
|----------------------|---------|
| `0` (default) | CPU-only — no layers offloaded |
| Positive integer | That many transformer layers offloaded to GPU |
| `-1` | All layers offloaded (passed as 999 to llama.cpp) |

### GPU offload status in exported results

| `gpu_offload_requested` | `gpu_layers_requested` | `gpu_offload_active` | Interpretation |
|-------------------------|------------------------|----------------------|----------------|
| `false` | null | `false` | CPU-only run, offload not requested |
| `true` | e.g. `32` | `false` | Offload requested but native build is CPU-only |
| `true` | e.g. `32` | null | Offload requested; no CPU-only warning detected; actual GPU use unconfirmed |

`gpu_offload_active` is never set to `true` — the `de.kherud:llama` binding does not expose a direct API to confirm GPU utilisation. Use the backend startup logs to investigate further.

### CPU vs GPU offload comparison workflow

```bash
# CPU-only baseline
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64 --context 2048 \
  --warmup 2 --iters 10 \
  --output-csv /tmp/gpu-comparison.csv

# GPU offload run (requires GPU-enabled llama.cpp build)
java -jar tq-bench-cli/target/tq-bench-cli-*-fat.jar \
  --backend llama.cpp \
  --model-path /path/to/model.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64 --context 2048 \
  --warmup 2 --iters 10 \
  --gpu-layers -1 \
  --output-csv /tmp/gpu-comparison.csv
```

Both rows land in the same CSV. Compare `generated_tokens_per_second_mean` across rows with matching `model_basename`, `requested_max_new_tokens`, and `context_tokens`.

### Important distinction: llama.cpp GPU offload vs. future CUDA/HIP backends

`--gpu-layers` controls llama.cpp's internal GPU offload. This is **not** the same as the future `tq-backend-cuda` / `tq-backend-hip` backends:

| | llama.cpp `--gpu-layers` | future cuda/hip backends |
|-|--------------------------|--------------------------|
| Backend name | `llama.cpp` | `cuda` / `hip` |
| Mechanism | llama.cpp internal offload | Custom C ABI (`tq_native_api.h`) + JNI |
| Status | Configurable; activation depends on native build | Placeholder — no real GPU kernels yet |
| Java API changes needed | None | None (already defined) |

---

## Current limitations

| Limitation | Impact |
|------------|--------|
| GPU offload status is best-effort | `gpu_offload_active` is `null` (unknown) when offload is requested and no CPU-only warning is detected; never confirmed `true` |
| CPU-only native build in `de.kherud:llama` | GPU offload requires a GPU-enabled llama.cpp build; the bundled JAR may be CPU-only depending on the version |
| No prompt-eval vs. generation breakdown | `prompt_tokens_per_second` is always null; `latency_mean_ms` covers both phases |
| No model load time isolation | Model load is included in `wall_clock_ms` but not broken out separately |
| No KV cache metrics from llama.cpp | `kvCacheStats()` returns `KvCacheStats.empty()` for llama.cpp; KV metrics only populate for cpu-stub |
| cpu-stub is not a real model | `CPU_STUB` throughput numbers reflect LCG computation, not real inference |
| CUDA/HIP are placeholders | `CUDA_PLACEHOLDER` and `HIP_PLACEHOLDER` modes have no real GPU kernels; their numbers are as meaningless as cpu-stub |
| TurboQuant not implemented | No quantization algorithms are active in any backend |

---

## Extensibility

`BenchmarkRunResult` is designed so future fields can be added without breaking existing CSV consumers (new columns append at the end). Fields planned for future milestones:

- `kv_cache_hit_rate`, `kv_cache_occupancy_pct` — when llama.cpp surfaces KV cache stats
- `model_load_time_ms` — once model loading is timed separately
- `gpu_memory_mb` — once GPU memory tracking is added to CUDA/HIP backends
- `quantization_bits` — once TurboQuant quantization is active
- `prompt_tokens_per_second` — if prompt-eval timing becomes available
