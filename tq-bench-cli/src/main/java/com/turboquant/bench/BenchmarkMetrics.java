package com.turboquant.bench;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;

/**
 * Standardized benchmark metric values for one complete run.
 *
 * <h2>Metric definitions</h2>
 *
 * <dl>
 *   <dt>{@link #latencyMeanMs}, {@link #latencyMinMs}, {@link #latencyMaxMs},
 *       {@link #latencyP50Ms}, {@link #latencyP99Ms}</dt>
 *   <dd>Per-iteration inference latency statistics in milliseconds.  Each measurement
 *       is the wall-clock time returned by {@code InferenceResult.inferenceNanos()},
 *       which the backend records from the first token request to the last token
 *       returned.  For llama.cpp this covers both prompt evaluation and token
 *       generation in a single pass; the two phases cannot be separated with the
 *       current Java binding.  For cpu-stub this is the stub LCG computation only.</dd>
 *
 *   <dt>{@link #generatedTokensLastRun}</dt>
 *   <dd>The actual number of tokens produced in the last timed iteration.  For
 *       llama.cpp this may be less than {@code requestedMaxNewTokens} if an EOS token
 *       was emitted early.  For cpu-stub it always equals {@code requestedMaxNewTokens}.</dd>
 *
 *   <dt>{@link #generatedTokensPerSecondMean}</dt>
 *   <dd>Mean of per-iteration throughput values computed as
 *       {@code generatedTokenCount[i] / (latencyMs[i] / 1000.0)}.  Computing the mean
 *       of per-iteration rates avoids the bias introduced by dividing mean-token-count
 *       by mean-latency (Jensen's inequality).  Null when no tokens were generated in
 *       any timed iteration.</dd>
 *
 *   <dt>{@link #promptTokensPerSecond}</dt>
 *   <dd>Always {@code null}.  The current Java binding ({@code de.kherud:llama}) does
 *       not expose separate prompt-evaluation timing, so this metric cannot be measured
 *       without instrumentation inside the native library.</dd>
 *
 *   <dt>{@link #modelLoadTimeMs}</dt>
 *   <dd>Always {@code null}.  Model loading is not timed separately in the CLI; it is
 *       included in {@link #wallClockMs}.  A future iteration could time the
 *       {@code DefaultTurboQuantRuntime.withBackend()} call to populate this field.</dd>
 *
 *   <dt>{@link #wallClockMs}</dt>
 *   <dd>Total elapsed time in milliseconds from just before runtime creation
 *       (which includes model loading for llama.cpp) to just after {@code runtime.close()}.
 *       This encompasses all warmup and timed iterations plus any initialization and
 *       cleanup overhead.  Useful as an upper bound on full end-to-end cost.</dd>
 * </dl>
 */
public final class BenchmarkMetrics {

    // --- per-iteration latency (ms) ---
    private final double latencyMeanMs;
    private final double latencyMinMs;
    private final double latencyMaxMs;
    private final double latencyP50Ms;
    private final double latencyP99Ms;

    // --- token throughput ---
    private final int    generatedTokensLastRun;
    /** Mean of per-iteration tok/s values; null if no tokens generated. */
    private final Double generatedTokensPerSecondMean;
    /** Always null — prompt-eval timing not available in current binding. */
    private final Double promptTokensPerSecond;

    // --- wall-clock and load ---
    /** Always null — model load is not timed separately; it is included in wallClockMs. */
    private final Double modelLoadTimeMs;
    private final double wallClockMs;

    private BenchmarkMetrics(Builder b) {
        this.latencyMeanMs                = b.latencyMeanMs;
        this.latencyMinMs                 = b.latencyMinMs;
        this.latencyMaxMs                 = b.latencyMaxMs;
        this.latencyP50Ms                 = b.latencyP50Ms;
        this.latencyP99Ms                 = b.latencyP99Ms;
        this.generatedTokensLastRun       = b.generatedTokensLastRun;
        this.generatedTokensPerSecondMean = b.generatedTokensPerSecondMean;
        this.promptTokensPerSecond        = null;  // not measurable; always null
        this.modelLoadTimeMs              = null;  // not measured; always null
        this.wallClockMs                  = b.wallClockMs;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public double  latencyMeanMs()                 { return latencyMeanMs; }
    public double  latencyMinMs()                  { return latencyMinMs; }
    public double  latencyMaxMs()                  { return latencyMaxMs; }
    public double  latencyP50Ms()                  { return latencyP50Ms; }
    public double  latencyP99Ms()                  { return latencyP99Ms; }
    public int     generatedTokensLastRun()        { return generatedTokensLastRun; }
    public Double  generatedTokensPerSecondMean()  { return generatedTokensPerSecondMean; }
    public Double  promptTokensPerSecond()         { return promptTokensPerSecond; }
    public Double  modelLoadTimeMs()               { return modelLoadTimeMs; }
    public double  wallClockMs()                   { return wallClockMs; }

    // -------------------------------------------------------------------------
    // Static factories
    // -------------------------------------------------------------------------

    /**
     * Compute metrics from raw per-iteration arrays.
     *
     * @param latenciesMs         per-iteration inference latency in milliseconds
     * @param generatedTokenCounts per-iteration actual generated token count
     * @param wallClockMs         total wall-clock time for the full benchmark run (ms)
     */
    public static BenchmarkMetrics from(double[] latenciesMs,
                                        int[]    generatedTokenCounts,
                                        double   wallClockMs) {
        if (latenciesMs.length == 0) {
            return empty();
        }

        DoubleSummaryStatistics stats = Arrays.stream(latenciesMs).summaryStatistics();

        // Per-iteration throughput: mean of (tokens_i / latency_i_in_seconds)
        // This avoids dividing mean-token-count by mean-latency (Jensen's inequality bias).
        double sumTokPerSec = 0;
        int    validIters   = 0;
        for (int i = 0; i < latenciesMs.length; i++) {
            if (latenciesMs[i] > 0 && generatedTokenCounts[i] > 0) {
                sumTokPerSec += generatedTokenCounts[i] * 1000.0 / latenciesMs[i];
                validIters++;
            }
        }
        Double meanTokPerSec = validIters > 0 ? sumTokPerSec / validIters : null;

        int lastRunTokens = generatedTokenCounts[generatedTokenCounts.length - 1];

        return new Builder()
                .latencyMeanMs(stats.getAverage())
                .latencyMinMs(stats.getMin())
                .latencyMaxMs(stats.getMax())
                .latencyP50Ms(percentile(latenciesMs, 50))
                .latencyP99Ms(percentile(latenciesMs, 99))
                .generatedTokensLastRun(lastRunTokens)
                .generatedTokensPerSecondMean(meanTokPerSec)
                .wallClockMs(wallClockMs)
                .build();
    }

    /** Returns a zeroed-out metrics object for cases where no measurement was taken. */
    public static BenchmarkMetrics empty() {
        return new Builder().build();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static double percentile(double[] data, int pct) {
        double[] s = data.clone();
        Arrays.sort(s);
        int idx = (int) Math.ceil(pct / 100.0 * s.length) - 1;
        return s[Math.max(0, Math.min(idx, s.length - 1))];
    }

    public static Builder builder() {
        return new Builder();
    }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static final class Builder {
        private double latencyMeanMs;
        private double latencyMinMs;
        private double latencyMaxMs;
        private double latencyP50Ms;
        private double latencyP99Ms;
        private int    generatedTokensLastRun;
        private Double generatedTokensPerSecondMean;
        private double wallClockMs;

        public Builder latencyMeanMs(double v)               { this.latencyMeanMs = v; return this; }
        public Builder latencyMinMs(double v)                { this.latencyMinMs = v; return this; }
        public Builder latencyMaxMs(double v)                { this.latencyMaxMs = v; return this; }
        public Builder latencyP50Ms(double v)                { this.latencyP50Ms = v; return this; }
        public Builder latencyP99Ms(double v)                { this.latencyP99Ms = v; return this; }
        public Builder generatedTokensLastRun(int v)         { this.generatedTokensLastRun = v; return this; }
        public Builder generatedTokensPerSecondMean(Double v){ this.generatedTokensPerSecondMean = v; return this; }
        public Builder wallClockMs(double v)                 { this.wallClockMs = v; return this; }

        public BenchmarkMetrics build() {
            return new BenchmarkMetrics(this);
        }
    }
}
