package com.turboquant.bench;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Immutable record of one complete benchmark run.
 *
 * <p>Captures all run metadata, configuration, and measured metrics in a single
 * object suitable for export to JSON or CSV and comparison across runs.</p>
 *
 * <p>Two request modes are represented:</p>
 * <ul>
 *   <li><b>Token-based</b> (cpu-stub, cuda, hip): {@code promptText} is null;
 *       {@code promptLength} is the synthetic token count.</li>
 *   <li><b>Text-based</b> (llama.cpp): {@code promptText} is the raw prompt string;
 *       {@code promptTokenCount} is null because the Java binding does not surface it.</li>
 * </ul>
 */
public final class BenchmarkRunResult {

    /** Schema version for the export format. Increment when fields change incompatibly. */
    public static final String BENCHMARK_VERSION = "0.2";

    // --- run identity ---
    private final String  runId;
    private final Instant timestampUtc;
    private final String  gitCommit;           // null if unavailable

    // --- backend ---
    private final String      backendName;
    private final BackendMode backendMode;
    private final boolean     isRealInference;
    private final boolean     gpuOffloadRequested; // true when nGpuLayers != 0 was requested
    private final Integer     gpuLayersRequested;  // null for non-llama.cpp backends
    private final Boolean     gpuOffloadActive;    // null = unknown, false = CPU-only, true = confirmed active

    // --- model (null for token-based backends) ---
    private final String modelPath;
    private final String modelBasename;
    private final String quantHint;            // null if not derivable from filename

    // --- prompt ---
    private final String  promptText;          // null for synthetic token-based prompts
    private final int     promptLength;        // token count (synthetic) or 0 (text-based, unknown)
    private final Integer promptTokenCount;    // null when not measurable

    // --- run configuration ---
    private final int requestedMaxNewTokens;
    private final int contextTokens;
    private final int warmupIterations;
    private final int timedIterations;

    // --- measured metrics ---
    private final BenchmarkMetrics metrics;

    // --- environment hints ---
    private final List<String> environmentNotes;

    private BenchmarkRunResult(Builder b) {
        this.runId                 = b.runId;
        this.timestampUtc          = b.timestampUtc;
        this.gitCommit             = b.gitCommit;
        this.backendName           = b.backendName;
        this.backendMode           = b.backendMode;
        this.isRealInference       = b.isRealInference;
        this.gpuOffloadRequested   = b.gpuOffloadRequested;
        this.gpuLayersRequested    = b.gpuLayersRequested;
        this.gpuOffloadActive      = b.gpuOffloadActive;
        this.modelPath             = b.modelPath;
        this.modelBasename         = b.modelBasename;
        this.quantHint             = b.quantHint;
        this.promptText            = b.promptText;
        this.promptLength          = b.promptLength;
        this.promptTokenCount      = b.promptTokenCount;
        this.requestedMaxNewTokens = b.requestedMaxNewTokens;
        this.contextTokens         = b.contextTokens;
        this.warmupIterations      = b.warmupIterations;
        this.timedIterations       = b.timedIterations;
        this.metrics               = b.metrics;
        this.environmentNotes      = Collections.unmodifiableList(new ArrayList<>(b.environmentNotes));
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public String runId()                 { return runId; }
    public Instant timestampUtc()         { return timestampUtc; }
    public String gitCommit()             { return gitCommit; }
    public String backendName()           { return backendName; }
    public BackendMode backendMode()      { return backendMode; }
    public boolean isRealInference()      { return isRealInference; }
    public boolean gpuOffloadRequested()  { return gpuOffloadRequested; }
    public Integer gpuLayersRequested()   { return gpuLayersRequested; }
    public Boolean gpuOffloadActive()     { return gpuOffloadActive; }
    public String modelPath()             { return modelPath; }
    public String modelBasename()         { return modelBasename; }
    public String quantHint()             { return quantHint; }
    public String promptText()            { return promptText; }
    public int promptLength()             { return promptLength; }
    public Integer promptTokenCount()     { return promptTokenCount; }
    public int requestedMaxNewTokens()    { return requestedMaxNewTokens; }
    public int contextTokens()            { return contextTokens; }
    public int warmupIterations()         { return warmupIterations; }
    public int timedIterations()          { return timedIterations; }
    public BenchmarkMetrics metrics()     { return metrics; }
    public List<String> environmentNotes(){ return environmentNotes; }

    // -------------------------------------------------------------------------
    // Static helpers
    // -------------------------------------------------------------------------

    /**
     * Classify a backend name string into a {@link BackendMode}.
     */
    public static BackendMode modeFor(String backendName) {
        return switch (backendName) {
            case "llama.cpp" -> BackendMode.LLAMA_CPP_REAL;
            case "cuda"      -> BackendMode.CUDA_PLACEHOLDER;
            case "hip"       -> BackendMode.HIP_PLACEHOLDER;
            default          -> BackendMode.CPU_STUB;
        };
    }

    /**
     * Extract the file basename from an absolute or relative path string.
     * E.g. {@code "/models/foo.gguf"} → {@code "foo.gguf"}.
     */
    public static String basename(String path) {
        if (path == null || path.isBlank()) return null;
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        return slash >= 0 ? path.substring(slash + 1) : path;
    }

    /**
     * Derive a quantization hint from a GGUF filename.
     * E.g. {@code "qwen2.5-0.5b-instruct-q4_0.gguf"} → {@code "Q4_0"}.
     * Returns {@code null} if no recognizable token is found.
     */
    public static String quantHintFrom(String filename) {
        if (filename == null) return null;
        Matcher m = Pattern.compile("(Q\\d+(?:_[A-Z0-9]+)*|F16|F32|BF16)",
                Pattern.CASE_INSENSITIVE).matcher(filename);
        return m.find() ? m.group(1).toUpperCase() : null;
    }

    public static Builder builder() {
        return new Builder();
    }

    // -------------------------------------------------------------------------
    // BackendMode
    // -------------------------------------------------------------------------

    /**
     * Distinguishes real inference backends from stub and placeholder backends.
     * Used in exported results so comparisons know whether numbers reflect real model output.
     */
    public enum BackendMode {
        /** Pure Java LCG stub — deterministic, no real model weights. */
        CPU_STUB,
        /** Real llama.cpp GGUF model loading and text generation (CPU-only validated). */
        LLAMA_CPP_REAL,
        /** Java JNI bridge + mock C ABI. No real CUDA kernels. */
        CUDA_PLACEHOLDER,
        /** Java JNI bridge + mock C ABI. No real HIP/ROCm kernels. */
        HIP_PLACEHOLDER
    }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static final class Builder {
        private String  runId              = UUID.randomUUID().toString().substring(0, 8);
        private Instant timestampUtc       = Instant.now();
        private String  gitCommit;
        private String  backendName        = "unknown";
        private BackendMode backendMode    = BackendMode.CPU_STUB;
        private boolean isRealInference    = false;
        private boolean gpuOffloadRequested = false;
        private Integer gpuLayersRequested;
        private Boolean gpuOffloadActive;
        private String  modelPath;
        private String  modelBasename;
        private String  quantHint;
        private String  promptText;
        private int     promptLength;
        private Integer promptTokenCount;
        private int     requestedMaxNewTokens;
        private int     contextTokens;
        private int     warmupIterations;
        private int     timedIterations;
        private BenchmarkMetrics metrics      = BenchmarkMetrics.empty();
        private List<String> environmentNotes = new ArrayList<>();

        public Builder runId(String v)                  { this.runId = v; return this; }
        public Builder timestampUtc(Instant v)          { this.timestampUtc = v; return this; }
        public Builder gitCommit(String v)              { this.gitCommit = v; return this; }
        public Builder backendName(String v)            { this.backendName = v; return this; }
        public Builder backendMode(BackendMode v)       { this.backendMode = v; return this; }
        public Builder isRealInference(boolean v)       { this.isRealInference = v; return this; }
        public Builder gpuOffloadRequested(boolean v)   { this.gpuOffloadRequested = v; return this; }
        public Builder gpuLayersRequested(Integer v)    { this.gpuLayersRequested = v; return this; }
        public Builder gpuOffloadActive(Boolean v)      { this.gpuOffloadActive = v; return this; }
        public Builder modelPath(String v)              { this.modelPath = v; return this; }
        public Builder modelBasename(String v)          { this.modelBasename = v; return this; }
        public Builder quantHint(String v)              { this.quantHint = v; return this; }
        public Builder promptText(String v)             { this.promptText = v; return this; }
        public Builder promptLength(int v)              { this.promptLength = v; return this; }
        public Builder promptTokenCount(Integer v)      { this.promptTokenCount = v; return this; }
        public Builder requestedMaxNewTokens(int v)     { this.requestedMaxNewTokens = v; return this; }
        public Builder contextTokens(int v)             { this.contextTokens = v; return this; }
        public Builder warmupIterations(int v)          { this.warmupIterations = v; return this; }
        public Builder timedIterations(int v)           { this.timedIterations = v; return this; }
        public Builder metrics(BenchmarkMetrics v)      { this.metrics = v; return this; }
        public Builder addEnvironmentNote(String v)     { this.environmentNotes.add(v); return this; }
        public Builder environmentNotes(List<String> v) {
            this.environmentNotes = new ArrayList<>(v);
            return this;
        }

        public BenchmarkRunResult build() {
            return new BenchmarkRunResult(this);
        }
    }
}
