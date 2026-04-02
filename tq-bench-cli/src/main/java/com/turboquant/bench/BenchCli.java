package com.turboquant.bench;

import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.SessionConfig;
import com.turboquant.runtime.api.TurboQuantRuntime;
import com.turboquant.runtime.core.BackendRegistry;
import com.turboquant.runtime.core.DefaultTurboQuantRuntime;
import com.turboquant.runtime.spi.BackendProvider;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.concurrent.Callable;

/**
 * TurboQuant Bench CLI — v0.2
 *
 * <p>Runs an inference loop against the selected backend and prints:
 * backend name, version, capabilities, config, per-iteration latency stats,
 * and the final KV cache snapshot.</p>
 *
 * <p>Three backend modes are supported:</p>
 * <ul>
 *   <li><b>cpu-stub</b> — pure Java, deterministic LCG, no real model</li>
 *   <li><b>llama.cpp</b> — real GGUF model, real text generation (CPU-only validated)</li>
 *   <li><b>cuda / hip</b> — placeholder structures only, no real GPU kernels</li>
 * </ul>
 *
 * <h2>Usage examples</h2>
 * <pre>
 *   # auto-select best available backend (cpu-stub when no GPU)
 *   java -jar tq-bench-cli-fat.jar
 *
 *   # list all discovered backends and exit
 *   java -jar tq-bench-cli-fat.jar --list
 *
 *   # cpu-stub: 5 warmup + 20 timed iters, 256-token prompt, 64 generated
 *   java -jar tq-bench-cli-fat.jar --backend cpu-stub --warmup 5 --iters 20 \
 *        --prompt-len 256 --gen-len 64
 *
 *   # llama.cpp: real inference with a local GGUF model
 *   java -jar tq-bench-cli-fat.jar --backend llama.cpp \
 *        --model-path /models/qwen2.5-0.5b-instruct-q4_0.gguf \
 *        --prompt "The capital of France is" \
 *        --max-new-tokens 64
 * </pre>
 */
@Command(
    name        = "tq-bench",
    mixinStandardHelpOptions = true,
    version     = "0.1.0-SNAPSHOT",
    description = "TurboQuant v0.1 inference benchmark — stub kernels, real architecture."
)
public class BenchCli implements Callable<Integer> {

    @Option(names = {"--backend", "-b"},
            description = "Backend name: cpu-stub, llama.cpp, cuda, hip. Default: auto-select.",
            defaultValue = "auto")
    private String backendName;

    @Option(names = {"--device", "-d"},
            description = "GPU device index (default: 0).",
            defaultValue = "0")
    private int deviceIndex;

    @Option(names = {"--warmup", "-w"},
            description = "Warmup iterations (excluded from stats). Default: 2.",
            defaultValue = "2")
    private int warmup;

    @Option(names = {"--iters", "-i"},
            description = "Timed benchmark iterations. Default: 10.",
            defaultValue = "10")
    private int iters;

    @Option(names = {"--prompt-len", "-p"},
            description = "Number of input tokens in each synthetic prompt. Default: 128.",
            defaultValue = "128")
    private int promptLen;

    @Option(names = {"--gen-len", "-g"},
            description = "Max new tokens to generate per iteration. Default: 32.",
            defaultValue = "32")
    private int genLen;

    @Option(names = {"--list", "-l"},
            description = "List all discovered backends and exit.",
            defaultValue = "false")
    private boolean listBackends;

    @Option(names = {"--model-path", "-m"},
            description = "Path to a GGUF model file (required for llama.cpp backend).",
            defaultValue = "")
    private String modelPath;

    @Option(names = {"--prompt"},
            description = "Text prompt to use for llama.cpp inference.",
            defaultValue = "Once upon a time")
    private String prompt;

    @Option(names = {"--context"},
            description = "Context window size in tokens (llama.cpp). Default: 2048.",
            defaultValue = "2048")
    private int contextTokens;

    @Option(names = {"--max-new-tokens"},
            description = "Override max new tokens (overrides --gen-len for llama.cpp).",
            defaultValue = "-1")
    private int maxNewTokensOverride;

    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        int exitCode = new CommandLine(new BenchCli()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() {
        if (listBackends) {
            printBackendList();
            return 0;
        }

        boolean isLlamaCpp = "llama.cpp".equalsIgnoreCase(backendName)
                || (!modelPath.isBlank() && "auto".equalsIgnoreCase(backendName));

        if (isLlamaCpp) {
            return runLlamaCpp();
        }

        BackendConfig config = BackendConfig.builder()
                .deviceIndex(deviceIndex)
                .build();

        try (TurboQuantRuntime runtime = buildRuntime(config)) {
            printHeader(runtime, config);
            runBench(runtime);
        }
        return 0;
    }

    private Integer runLlamaCpp() {
        int effectiveMaxNew = maxNewTokensOverride > 0 ? maxNewTokensOverride : genLen;
        SessionConfig sessionConfig = SessionConfig.builder()
                .modelPath(modelPath.isBlank() ? null : modelPath)
                .maxContextTokens(contextTokens)
                .maxNewTokens(effectiveMaxNew)
                .deviceId(deviceIndex)
                .build();

        String targetBackend = "auto".equalsIgnoreCase(backendName) ? "llama.cpp" : backendName;
        try (TurboQuantRuntime runtime = DefaultTurboQuantRuntime.withBackend(targetBackend, sessionConfig)) {
            printLlamaCppHeader(runtime, sessionConfig);
            runLlamaCppBench(runtime, effectiveMaxNew);
        }
        return 0;
    }

    // -------------------------------------------------------------------------
    // Runtime construction
    // -------------------------------------------------------------------------

    private TurboQuantRuntime buildRuntime(BackendConfig config) {
        if ("auto".equalsIgnoreCase(backendName)) {
            return DefaultTurboQuantRuntime.autoSelect(config);
        }
        return DefaultTurboQuantRuntime.withBackend(backendName, config);
    }

    // -------------------------------------------------------------------------
    // llama.cpp header + bench
    // -------------------------------------------------------------------------

    private void printLlamaCppHeader(TurboQuantRuntime runtime, SessionConfig cfg) {
        System.out.println("=".repeat(62));
        System.out.println("  TurboQuant Bench CLI  v0.1.0-SNAPSHOT  [llama.cpp mode]");
        System.out.println("=".repeat(62));
        System.out.printf("  Backend         : %s%n", runtime.backend().name());
        System.out.printf("  Model path      : %s%n",
                cfg.modelPath() != null ? cfg.modelPath() : "(not set)");
        System.out.printf("  Context tokens  : %d%n", cfg.maxContextTokens());
        System.out.printf("  Max new tokens  : %d%n", cfg.maxNewTokens());
        System.out.printf("  Prompt          : %s%n", prompt);
        System.out.printf("  Warmup iters    : %d%n", warmup);
        System.out.printf("  Timed iters     : %d%n", iters);
        System.out.println("=".repeat(62));
    }

    private void runLlamaCppBench(TurboQuantRuntime runtime, int maxNewTokens) {
        InferenceRequest request = InferenceRequest.fromText(prompt, maxNewTokens);

        if (warmup > 0) {
            System.out.printf("Warming up (%d iteration%s)...%n", warmup, warmup == 1 ? "" : "s");
            for (int i = 0; i < warmup; i++) {
                runtime.infer(request);
            }
        }

        System.out.printf("Running benchmark (%d iteration%s)...%n", iters, iters == 1 ? "" : "s");
        double[] latenciesMs = new double[iters];
        InferenceResult lastResult = null;

        for (int i = 0; i < iters; i++) {
            lastResult = runtime.infer(request);
            latenciesMs[i] = lastResult.inferenceMillis();
        }

        printLlamaCppResults(latenciesMs, lastResult);
    }

    private void printLlamaCppResults(double[] latenciesMs, InferenceResult last) {
        DoubleSummaryStatistics stats = java.util.Arrays.stream(latenciesMs).summaryStatistics();
        double meanMs = stats.getAverage();
        int genCount  = last != null ? last.generatedTokenCount() : 0;
        double tokPerSec = genCount > 0 ? (genCount * 1000.0) / meanMs : 0.0;

        System.out.println("-".repeat(62));
        System.out.println("  Latency (ms) over " + iters + " iterations:");
        System.out.printf("    mean  : %10.3f ms%n", meanMs);
        System.out.printf("    min   : %10.3f ms%n", stats.getMin());
        System.out.printf("    p50   : %10.3f ms%n", percentile(latenciesMs, 50));
        System.out.printf("    p99   : %10.3f ms%n", percentile(latenciesMs, 99));
        System.out.printf("    max   : %10.3f ms%n", stats.getMax());
        System.out.printf("  Throughput    : %,10.1f tok/s (generated)%n", tokPerSec);

        if (last != null) {
            System.out.println("-".repeat(62));
            System.out.printf("  Generated tokens : %d%n", last.generatedTokenCount());
            System.out.printf("  Backend          : %s%n", last.backendName());
            if (last.generatedText() != null) {
                System.out.println("-".repeat(62));
                System.out.println("  Last generated text:");
                System.out.println("  " + last.generatedText().replace("\n", "\n  "));
            }
        }

        System.out.println("=".repeat(62));
        System.out.println("  DEMO SUMMARY");
        System.out.println("=".repeat(62));
        System.out.printf("  backend    : llama.cpp (experimental, CPU-only validated)%n");
        System.out.printf("  throughput : %.1f tok/s (measured, %d token run)%n",
                tokPerSec, genCount);
        System.out.printf("  note       : CUDA/HIP are separate future backend tracks%n");
        System.out.println("=".repeat(62));
    }

    // -------------------------------------------------------------------------
    // Backend listing
    // -------------------------------------------------------------------------

    private void printBackendList() {
        List<BackendProvider> all = new BackendRegistry().allProviders();
        System.out.println("Discovered backend providers (" + all.size() + "):");
        System.out.printf("  %-20s  %-10s  %s%n", "NAME", "PRIORITY", "AVAILABLE");
        System.out.println("  " + "-".repeat(46));
        for (BackendProvider p : all) {
            System.out.printf("  %-20s  %-10d  %s%n",
                    p.backendName(), p.priority(), p.isAvailable() ? "YES" : "no");
        }
    }

    // -------------------------------------------------------------------------
    // Header
    // -------------------------------------------------------------------------

    private void printHeader(TurboQuantRuntime runtime, BackendConfig config) {
        String mode = modeLabel(runtime.backend().name());
        System.out.println("=".repeat(62));
        System.out.printf("  TurboQuant Bench CLI  v0.1.0-SNAPSHOT  %s%n", mode);
        System.out.println("=".repeat(62));
        System.out.printf("  Backend name    : %s%n", runtime.backend().name());
        System.out.printf("  Backend version : %s%n", runtime.backend().version());
        System.out.printf("  Capabilities    : %s%n", runtime.backend().capabilities());
        System.out.printf("  Config          : %s%n", config);
        System.out.printf("  Prompt tokens   : %d%n",  promptLen);
        System.out.printf("  Max new tokens  : %d%n",  genLen);
        System.out.printf("  Warmup iters    : %d%n",  warmup);
        System.out.printf("  Timed iters     : %d%n",  iters);
        System.out.println("=".repeat(62));
    }

    // -------------------------------------------------------------------------
    // Benchmark loop
    // -------------------------------------------------------------------------

    private void runBench(TurboQuantRuntime runtime) {
        InferenceRequest request = InferenceRequest.syntheticPrompt(promptLen, genLen);

        // --- warmup ---
        if (warmup > 0) {
            System.out.printf("Warming up (%d iteration%s)...%n",
                    warmup, warmup == 1 ? "" : "s");
            for (int i = 0; i < warmup; i++) {
                runtime.infer(request);
            }
        }

        // --- timed loop ---
        System.out.printf("Running benchmark (%d iteration%s)...%n",
                iters, iters == 1 ? "" : "s");
        double[] latenciesMs = new double[iters];
        InferenceResult lastResult = null;

        for (int i = 0; i < iters; i++) {
            lastResult = runtime.infer(request);
            latenciesMs[i] = lastResult.inferenceMillis();
        }

        KvCacheStats cacheStats = runtime.kvCacheStats();
        printResults(latenciesMs, lastResult, cacheStats);
    }

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------

    private void printResults(double[] latenciesMs, InferenceResult lastResult, KvCacheStats cache) {
        DoubleSummaryStatistics stats = java.util.Arrays.stream(latenciesMs).summaryStatistics();
        double meanMs  = stats.getAverage();
        double minMs   = stats.getMin();
        double maxMs   = stats.getMax();
        double p50Ms   = percentile(latenciesMs, 50);
        double p99Ms   = percentile(latenciesMs, 99);

        double tokPerSec = (genLen * 1000.0) / meanMs;

        System.out.println("-".repeat(62));
        System.out.println("  Latency (ms) over " + iters + " iterations:");
        System.out.printf("    mean  : %10.3f ms%n", meanMs);
        System.out.printf("    min   : %10.3f ms%n", minMs);
        System.out.printf("    p50   : %10.3f ms%n", p50Ms);
        System.out.printf("    p99   : %10.3f ms%n", p99Ms);
        System.out.printf("    max   : %10.3f ms%n", maxMs);
        System.out.printf("  Throughput    : %,10.1f tok/s (generated, stub)%n", tokPerSec);

        if (lastResult != null) {
            System.out.println("-".repeat(62));
            System.out.println("  Last result:");
            System.out.printf("    prompt tokens    : %d%n", lastResult.promptTokenCount());
            System.out.printf("    generated tokens : %d%n", lastResult.generatedTokenCount());
            System.out.printf("    first generated  : token id %d%n",
                    lastResult.generatedTokenIds().length > 0
                            ? lastResult.generatedTokenIds()[0] : -1);
            System.out.printf("    logit sparsity   : 1 non-zero / %d vocab%n",
                    lastResult.lastLogits().length);
        }

        System.out.println("-".repeat(62));
        System.out.println("  KV Cache (end of run):");
        System.out.printf("    cached tokens    : %,d / %,d  (%s)%n",
                cache.cachedTokens(), cache.capacityTokens(), cache.occupancyPercent());
        System.out.printf("    cache size       : %.2f MiB%n", cache.cacheSizeMiB());
        System.out.printf("    simulated hitRate: %.1f%%%n", cache.hitRate() * 100.0);

        System.out.println("=".repeat(62));
        System.out.printf("  NOTE: %s%n", backendNote(lastResult != null ? lastResult.backendName() : backendName));
        System.out.println("=".repeat(62));
    }

    private static String backendNote(String name) {
        return switch (name) {
            case "cpu-stub" -> "cpu-stub — no real model; deterministic LCG only";
            case "cuda"     -> "cuda placeholder — no real GPU kernels yet";
            case "hip"      -> "hip placeholder — no real GPU kernels yet";
            default         -> name + " — stub backend";
        };
    }

    private static String modeLabel(String backendName) {
        return switch (backendName) {
            case "cpu-stub" -> "[cpu-stub mode]";
            case "cuda"     -> "[cuda placeholder]";
            case "hip"      -> "[hip placeholder]";
            case "llama.cpp" -> "[llama.cpp mode]";
            default          -> "[" + backendName + " mode]";
        };
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static double percentile(double[] sortedCopy, int pct) {
        double[] s = sortedCopy.clone();
        java.util.Arrays.sort(s);
        int idx = (int) Math.ceil(pct / 100.0 * s.length) - 1;
        return s[Math.max(0, Math.min(idx, s.length - 1))];
    }
}
