package com.turboquant.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/**
 * Exports {@link BenchmarkRunResult} to JSON and CSV.
 *
 * <p>No external serialization library is used — output is hand-built to keep
 * the {@code tq-bench-cli} dependency footprint minimal.</p>
 *
 * <h2>JSON</h2>
 * <p>One complete JSON object per file. Overwrites if the file already exists.
 * All optional fields that are not available are written as JSON {@code null}.</p>
 *
 * <h2>CSV</h2>
 * <p>One row per run.  If the target file does not exist the header row is written
 * first.  If the file already exists, only the data row is appended — this allows
 * accumulating results across multiple runs without losing earlier data.</p>
 */
public final class BenchmarkExporter {

    // CSV column names — 26 columns (added gpu_offload_requested, gpu_layers_requested after gpu_offload_active)
    public static final String CSV_HEADER =
            "run_id,timestamp_utc,git_commit,backend_name,backend_mode," +
            "is_real_inference,gpu_offload_requested,gpu_layers_requested,gpu_offload_active,model_basename,quant_hint," +
            "prompt_length,prompt_token_count,requested_max_new_tokens,context_tokens," +
            "warmup_iters,timed_iters,generated_tokens_last_run," +
            "latency_mean_ms,latency_min_ms,latency_p50_ms,latency_p99_ms,latency_max_ms," +
            "generated_tokens_per_second_mean,wall_clock_ms";

    private BenchmarkExporter() {}

    // -------------------------------------------------------------------------
    // JSON export
    // -------------------------------------------------------------------------

    /**
     * Serializes {@code result} to JSON and writes it to {@code path}, creating
     * parent directories as needed and overwriting any existing file.
     */
    public static void writeJson(BenchmarkRunResult result, Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        Files.writeString(path, toJson(result), StandardCharsets.UTF_8);
    }

    /**
     * Serializes {@code result} to a JSON string.
     * Package-visible so tests can inspect the output without touching the filesystem.
     */
    public static String toJson(BenchmarkRunResult r) {
        BenchmarkMetrics m = r.metrics();
        StringBuilder sb = new StringBuilder(1024);

        sb.append("{\n");
        sb.append("  \"benchmarkVersion\": ").append(qs(BenchmarkRunResult.BENCHMARK_VERSION)).append(",\n");
        sb.append("  \"runId\": ").append(qs(r.runId())).append(",\n");
        sb.append("  \"timestampUtc\": ").append(qs(r.timestampUtc().toString())).append(",\n");
        sb.append("  \"gitCommit\": ").append(qsNull(r.gitCommit())).append(",\n");

        sb.append("  \"backend\": {\n");
        sb.append("    \"name\": ").append(qs(r.backendName())).append(",\n");
        sb.append("    \"mode\": ").append(qs(r.backendMode().name())).append(",\n");
        sb.append("    \"isRealInference\": ").append(r.isRealInference()).append(",\n");
        sb.append("    \"gpuOffloadRequested\": ").append(r.gpuOffloadRequested()).append(",\n");
        sb.append("    \"gpuLayersRequested\": ").append(intNull(r.gpuLayersRequested())).append(",\n");
        sb.append("    \"gpuOffloadActive\": ").append(boolNull(r.gpuOffloadActive())).append("\n");
        sb.append("  },\n");

        sb.append("  \"model\": {\n");
        sb.append("    \"path\": ").append(qsNull(r.modelPath())).append(",\n");
        sb.append("    \"basename\": ").append(qsNull(r.modelBasename())).append(",\n");
        sb.append("    \"quantHint\": ").append(qsNull(r.quantHint())).append("\n");
        sb.append("  },\n");

        sb.append("  \"prompt\": {\n");
        sb.append("    \"text\": ").append(qsNull(r.promptText())).append(",\n");
        sb.append("    \"length\": ").append(r.promptLength()).append(",\n");
        sb.append("    \"promptTokenCount\": ").append(intNull(r.promptTokenCount())).append("\n");
        sb.append("  },\n");

        sb.append("  \"config\": {\n");
        sb.append("    \"requestedMaxNewTokens\": ").append(r.requestedMaxNewTokens()).append(",\n");
        sb.append("    \"contextTokens\": ").append(r.contextTokens()).append(",\n");
        sb.append("    \"warmupIterations\": ").append(r.warmupIterations()).append(",\n");
        sb.append("    \"timedIterations\": ").append(r.timedIterations()).append("\n");
        sb.append("  },\n");

        sb.append("  \"metrics\": {\n");
        sb.append("    \"generatedTokensLastRun\": ").append(m.generatedTokensLastRun()).append(",\n");
        sb.append("    \"totalLatencyMs\": {\n");
        sb.append("      \"mean\": ").append(dbl(m.latencyMeanMs())).append(",\n");
        sb.append("      \"min\": ").append(dbl(m.latencyMinMs())).append(",\n");
        sb.append("      \"p50\": ").append(dbl(m.latencyP50Ms())).append(",\n");
        sb.append("      \"p99\": ").append(dbl(m.latencyP99Ms())).append(",\n");
        sb.append("      \"max\": ").append(dbl(m.latencyMaxMs())).append("\n");
        sb.append("    },\n");
        sb.append("    \"generatedTokensPerSecondMean\": ").append(dblNull(m.generatedTokensPerSecondMean())).append(",\n");
        sb.append("    \"promptTokensPerSecond\": ").append(dblNull(m.promptTokensPerSecond())).append(",\n");
        sb.append("    \"modelLoadTimeMs\": ").append(dblNull(m.modelLoadTimeMs())).append(",\n");
        sb.append("    \"wallClockMs\": ").append(dbl(m.wallClockMs())).append("\n");
        sb.append("  },\n");

        List<String> notes = r.environmentNotes();
        sb.append("  \"environmentNotes\": [");
        if (notes.isEmpty()) {
            sb.append("]");
        } else {
            sb.append("\n");
            for (int i = 0; i < notes.size(); i++) {
                sb.append("    ").append(qs(notes.get(i)));
                if (i < notes.size() - 1) sb.append(",");
                sb.append("\n");
            }
            sb.append("  ]");
        }
        sb.append("\n}\n");

        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // CSV export
    // -------------------------------------------------------------------------

    /**
     * Writes {@code result} as a CSV row to {@code path}.
     * <ul>
     *   <li>If the file does not exist: writes the header row, then the data row.</li>
     *   <li>If the file exists: appends only the data row (accumulating runs).</li>
     * </ul>
     */
    public static void writeCsv(BenchmarkRunResult result, Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        boolean exists = Files.exists(path);
        StringBuilder sb = new StringBuilder();
        if (!exists) {
            sb.append(CSV_HEADER).append(System.lineSeparator());
        }
        sb.append(toCsvRow(result)).append(System.lineSeparator());

        if (exists) {
            Files.writeString(path, sb.toString(), StandardCharsets.UTF_8,
                    StandardOpenOption.APPEND);
        } else {
            Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
        }
    }

    /**
     * Serializes {@code result} as a single CSV data row (no header).
     * Package-visible so tests can inspect the output without touching the filesystem.
     */
    public static String toCsvRow(BenchmarkRunResult r) {
        BenchmarkMetrics m = r.metrics();
        List<String> cols = new ArrayList<>(24);

        cols.add(csvStr(r.runId()));
        cols.add(csvStr(r.timestampUtc().toString()));
        cols.add(csvStr(r.gitCommit()));
        cols.add(csvStr(r.backendName()));
        cols.add(csvStr(r.backendMode().name()));
        cols.add(csvBool(r.isRealInference()));
        cols.add(csvBool(r.gpuOffloadRequested()));
        cols.add(csvIntNull(r.gpuLayersRequested()));
        cols.add(csvBoolNull(r.gpuOffloadActive()));
        cols.add(csvStr(r.modelBasename()));
        cols.add(csvStr(r.quantHint()));
        cols.add(Integer.toString(r.promptLength()));
        cols.add(csvIntNull(r.promptTokenCount()));
        cols.add(Integer.toString(r.requestedMaxNewTokens()));
        cols.add(Integer.toString(r.contextTokens()));
        cols.add(Integer.toString(r.warmupIterations()));
        cols.add(Integer.toString(r.timedIterations()));
        cols.add(Integer.toString(m.generatedTokensLastRun()));
        cols.add(String.format("%.3f", m.latencyMeanMs()));
        cols.add(String.format("%.3f", m.latencyMinMs()));
        cols.add(String.format("%.3f", m.latencyP50Ms()));
        cols.add(String.format("%.3f", m.latencyP99Ms()));
        cols.add(String.format("%.3f", m.latencyMaxMs()));
        cols.add(csvDblNull(m.generatedTokensPerSecondMean()));
        cols.add(String.format("%.3f", m.wallClockMs()));

        return String.join(",", cols);
    }

    // -------------------------------------------------------------------------
    // JSON helpers
    // -------------------------------------------------------------------------

    /** Quoted, escaped JSON string. Input must not be null. */
    private static String qs(String s) {
        return "\"" + escape(s) + "\"";
    }

    /** JSON string or null literal. */
    private static String qsNull(String s) {
        return s != null ? qs(s) : "null";
    }

    /** JSON number (3 d.p.), guarding against NaN/Infinity. */
    private static String dbl(double v) {
        return (Double.isNaN(v) || Double.isInfinite(v)) ? "null" : String.format("%.3f", v);
    }

    /** JSON number or null literal. */
    private static String dblNull(Double v) {
        return v != null ? dbl(v) : "null";
    }

    /** JSON integer or null literal. */
    private static String intNull(Integer v) {
        return v != null ? v.toString() : "null";
    }

    /** JSON boolean or null literal. */
    private static String boolNull(Boolean v) {
        return v != null ? v.toString() : "null";
    }

    /** Minimal JSON string escaping. */
    private static String escape(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    // -------------------------------------------------------------------------
    // CSV helpers
    // -------------------------------------------------------------------------

    /** CSV-safe string: quotes if needed, empty string for null. */
    private static String csvStr(String v) {
        if (v == null) return "";
        if (v.contains(",") || v.contains("\"") || v.contains("\n")) {
            return "\"" + v.replace("\"", "\"\"") + "\"";
        }
        return v;
    }

    private static String csvBool(boolean v)    { return Boolean.toString(v); }
    private static String csvBoolNull(Boolean v) { return v != null ? v.toString() : ""; }
    private static String csvIntNull(Integer v)  { return v != null ? v.toString() : ""; }
    private static String csvDblNull(Double v)   {
        return v != null ? String.format("%.3f", v) : "";
    }
}
