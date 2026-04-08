package com.turboquant.bench;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BenchmarkRunResult}, {@link BenchmarkMetrics}, and
 * {@link BenchmarkExporter}.
 *
 * <p>No real model file is required — all tests use synthetic data.</p>
 */
class BenchmarkExportTest {

    // -------------------------------------------------------------------------
    // Sample data helpers
    // -------------------------------------------------------------------------

    private static BenchmarkMetrics sampleMetrics() {
        double[] latencies  = {100.0, 110.0, 105.0, 108.0, 102.0};
        int[]    tokens     = {32, 32, 32, 32, 32};
        return BenchmarkMetrics.from(latencies, tokens, 800.0);
    }

    private static BenchmarkRunResult sampleResult() {
        return BenchmarkRunResult.builder()
                .runId("test1234")
                .timestampUtc(Instant.parse("2026-04-02T10:00:00Z"))
                .gitCommit("abc1234")
                .backendName("cpu-stub")
                .backendMode(BenchmarkRunResult.BackendMode.CPU_STUB)
                .isRealInference(false)
                .gpuOffloadActive(false)
                .promptLength(128)
                .promptTokenCount(128)
                .requestedMaxNewTokens(32)
                .contextTokens(0)
                .warmupIterations(2)
                .timedIterations(5)
                .metrics(sampleMetrics())
                .addEnvironmentNote("cpu-stub — no real model; deterministic LCG only")
                .build();
    }

    private static BenchmarkRunResult sampleLlamaCppResult() {
        return BenchmarkRunResult.builder()
                .runId("llama001")
                .timestampUtc(Instant.parse("2026-04-02T10:00:00Z"))
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadActive(null)   // unknown / not tested
                .modelPath("/models/qwen2.5-0.5b-instruct-q4_0.gguf")
                .modelBasename("qwen2.5-0.5b-instruct-q4_0.gguf")
                .quantHint("Q4_0")
                .promptText("Once upon a time")
                .promptLength(0)
                .promptTokenCount(null)   // not measurable
                .requestedMaxNewTokens(32)
                .contextTokens(2048)
                .warmupIterations(2)
                .timedIterations(10)
                .metrics(sampleMetrics())
                .addEnvironmentNote("CPU-only (device index = 0)")
                .build();
    }

    // -------------------------------------------------------------------------
    // BenchmarkRunResult construction
    // -------------------------------------------------------------------------

    @Test
    void resultBuildsSuccessfullyWithNoNulls() {
        BenchmarkRunResult r = sampleResult();
        assertNotNull(r.runId());
        assertNotNull(r.timestampUtc());
        assertNotNull(r.backendName());
        assertNotNull(r.backendMode());
        assertNotNull(r.metrics());
        assertNotNull(r.environmentNotes());
    }

    @Test
    void resultNullableFieldsAreNullWhenNotSet() {
        BenchmarkRunResult r = sampleResult();
        assertNull(r.modelPath());
        assertNull(r.modelBasename());
        assertNull(r.quantHint());
        assertNull(r.promptText());
    }

    // -------------------------------------------------------------------------
    // BenchmarkMetrics
    // -------------------------------------------------------------------------

    @Test
    void metricsFromComputesMeanTokPerSecCorrectly() {
        // 5 iterations each generating 32 tokens in ~100 ms → ~320 tok/s
        double[] latencies = {100.0, 100.0, 100.0, 100.0, 100.0};
        int[]    tokens    = {32, 32, 32, 32, 32};
        BenchmarkMetrics m = BenchmarkMetrics.from(latencies, tokens, 1000.0);

        assertNotNull(m.generatedTokensPerSecondMean());
        assertEquals(320.0, m.generatedTokensPerSecondMean(), 0.1);
    }

    @Test
    void metricsNullableFieldsAlwaysNull() {
        BenchmarkMetrics m = sampleMetrics();
        assertNull(m.promptTokensPerSecond(),
                "promptTokensPerSecond must always be null — not measurable");
        assertNull(m.modelLoadTimeMs(),
                "modelLoadTimeMs must always be null — not measured separately");
    }

    @Test
    void metricsGeneratedTokensLastRunIsFromLastIteration() {
        double[] latencies = {100.0, 100.0, 100.0};
        int[]    tokens    = {30, 31, 28};   // last = 28
        BenchmarkMetrics m = BenchmarkMetrics.from(latencies, tokens, 500.0);
        assertEquals(28, m.generatedTokensLastRun());
    }

    // -------------------------------------------------------------------------
    // JSON export
    // -------------------------------------------------------------------------

    @Test
    void toJsonContainsRequiredFields() {
        String json = BenchmarkExporter.toJson(sampleResult());

        assertTrue(json.contains("\"benchmarkVersion\""), "must contain benchmarkVersion");
        assertTrue(json.contains("\"runId\""),            "must contain runId");
        assertTrue(json.contains("\"timestampUtc\""),     "must contain timestampUtc");
        assertTrue(json.contains("\"backendName\"") || json.contains("\"name\""),
                "must contain backend name");
        assertTrue(json.contains("\"totalLatencyMs\""),   "must contain totalLatencyMs block");
        assertTrue(json.contains("\"wallClockMs\""),      "must contain wallClockMs");
        assertTrue(json.contains("\"generatedTokensPerSecondMean\""),
                "must contain throughput metric");
    }

    @Test
    void toJsonRendersNullFieldsAsJsonNull() {
        String json = BenchmarkExporter.toJson(sampleResult());

        // gitCommit is set in sampleResult but modelPath is null
        assertTrue(json.contains("\"path\": null")
                || json.contains("\"path\":null"),
                "null modelPath must render as JSON null, not omitted or as string 'null'");
        assertTrue(json.contains("\"promptTokensPerSecond\": null")
                || json.contains("\"promptTokensPerSecond\":null"),
                "always-null promptTokensPerSecond must render as null");
        assertTrue(json.contains("\"modelLoadTimeMs\": null")
                || json.contains("\"modelLoadTimeMs\":null"),
                "always-null modelLoadTimeMs must render as null");
    }

    @Test
    void toJsonIsValidStructureStartsAndEndsCorrectly() {
        String json = BenchmarkExporter.toJson(sampleResult()).trim();
        assertTrue(json.startsWith("{"), "JSON must start with {");
        assertTrue(json.endsWith("}"),   "JSON must end with }");
    }

    @Test
    void toJsonIncludesNullableGitCommitWhenAbsent() {
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("cpu-stub")
                .backendMode(BenchmarkRunResult.BackendMode.CPU_STUB)
                .metrics(sampleMetrics())
                .build();
        String json = BenchmarkExporter.toJson(r);
        assertTrue(json.contains("\"gitCommit\": null") || json.contains("\"gitCommit\":null"),
                "absent gitCommit must render as null in JSON");
    }

    // -------------------------------------------------------------------------
    // CSV export
    // -------------------------------------------------------------------------

    @Test
    void csvRowColumnCountMatchesHeader() {
        String header = BenchmarkExporter.CSV_HEADER;
        String row    = BenchmarkExporter.toCsvRow(sampleResult());

        int headerCols = header.split(",", -1).length;
        int rowCols    = row.split(",", -1).length;

        assertEquals(headerCols, rowCols,
                "CSV row column count must match header column count (" + headerCols + ")");
    }

    @Test
    void csvRowContainsBackendName() {
        String row = BenchmarkExporter.toCsvRow(sampleResult());
        assertTrue(row.contains("cpu-stub"), "CSV row must contain backend name");
    }

    @Test
    void csvRowNullFieldsAreEmpty() {
        // sampleResult has no modelBasename — should produce empty field, not "null"
        String row = BenchmarkExporter.toCsvRow(sampleResult());
        assertFalse(row.contains("\"null\""), "null fields must not be rendered as the string null");
    }

    // -------------------------------------------------------------------------
    // File I/O
    // -------------------------------------------------------------------------

    @Test
    void writeJsonCreatesFileWithContent(@TempDir Path dir) throws IOException {
        Path out = dir.resolve("bench.json");
        BenchmarkExporter.writeJson(sampleResult(), out);

        assertTrue(Files.exists(out), "JSON file must be created");
        String content = Files.readString(out);
        assertFalse(content.isBlank(), "JSON file must not be empty");
        assertTrue(content.contains("benchmarkVersion"), "file must contain benchmarkVersion");
    }

    @Test
    void writeCsvCreatesFileWithHeaderAndOneRow(@TempDir Path dir) throws IOException {
        Path out = dir.resolve("bench.csv");
        BenchmarkExporter.writeCsv(sampleResult(), out);

        List<String> lines = Files.readAllLines(out);
        assertEquals(2, lines.size(), "CSV file must have exactly 2 lines: header + 1 data row");
        assertEquals(BenchmarkExporter.CSV_HEADER, lines.get(0),
                "first line must be the CSV header");
    }

    @Test
    void writeCsvAppendsSecondRowWithoutDuplicatingHeader(@TempDir Path dir) throws IOException {
        Path out = dir.resolve("bench.csv");
        BenchmarkExporter.writeCsv(sampleResult(), out);
        BenchmarkExporter.writeCsv(sampleResult(), out);

        List<String> lines = Files.readAllLines(out);
        assertEquals(3, lines.size(),
                "after two runs CSV file must have exactly 3 lines: header + 2 data rows");
        // Header must appear only once
        long headerCount = lines.stream()
                .filter(l -> l.equals(BenchmarkExporter.CSV_HEADER))
                .count();
        assertEquals(1, headerCount, "CSV header must appear exactly once");
    }

    // -------------------------------------------------------------------------
    // Static helpers on BenchmarkRunResult
    // -------------------------------------------------------------------------

    @Test
    void basenameExtractsFilename() {
        assertEquals("foo.gguf",  BenchmarkRunResult.basename("/models/foo.gguf"));
        assertEquals("foo.gguf",  BenchmarkRunResult.basename("foo.gguf"));
        assertEquals("model.bin", BenchmarkRunResult.basename("C:\\models\\model.bin"));
        assertNull(BenchmarkRunResult.basename(null));
        assertNull(BenchmarkRunResult.basename(""));
    }

    @Test
    void quantHintExtractsKnownPatterns() {
        assertEquals("Q4_0",   BenchmarkRunResult.quantHintFrom("qwen2.5-0.5b-q4_0.gguf"));
        assertEquals("Q4_K_M", BenchmarkRunResult.quantHintFrom("phi-3-mini-q4_k_m.gguf"));
        assertEquals("Q8_0",   BenchmarkRunResult.quantHintFrom("model-q8_0.gguf"));
        assertEquals("F16",    BenchmarkRunResult.quantHintFrom("model-f16.gguf"));
        assertNull(BenchmarkRunResult.quantHintFrom("model-with-no-quant.gguf"));
        assertNull(BenchmarkRunResult.quantHintFrom(null));
    }

    @Test
    void modeForMapsBackendNamesCorrectly() {
        assertEquals(BenchmarkRunResult.BackendMode.CPU_STUB,
                BenchmarkRunResult.modeFor("cpu-stub"));
        assertEquals(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL,
                BenchmarkRunResult.modeFor("llama.cpp"));
        assertEquals(BenchmarkRunResult.BackendMode.CUDA_PLACEHOLDER,
                BenchmarkRunResult.modeFor("cuda"));
        assertEquals(BenchmarkRunResult.BackendMode.HIP_PLACEHOLDER,
                BenchmarkRunResult.modeFor("hip"));
        // unknown name falls back to CPU_STUB
        assertEquals(BenchmarkRunResult.BackendMode.CPU_STUB,
                BenchmarkRunResult.modeFor("unknown-backend"));
    }

    // -------------------------------------------------------------------------
    // GPU offload fields
    // -------------------------------------------------------------------------

    @Test
    void gpuOffloadNotRequestedByDefault() {
        BenchmarkRunResult r = sampleResult();
        assertFalse(r.gpuOffloadRequested(), "gpuOffloadRequested must be false when not set");
        assertNull(r.gpuLayersRequested(),   "gpuLayersRequested must be null when not set");
        assertFalse(r.gpuOffloadActive(),    "gpuOffloadActive must be false for cpu-stub");
    }

    @Test
    void gpuOffloadRequestedFieldsRoundTrip() {
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadRequested(true)
                .gpuLayersRequested(32)
                .gpuOffloadActive(null)   // unknown — build may not support it
                .metrics(sampleMetrics())
                .build();

        assertTrue(r.gpuOffloadRequested());
        assertEquals(32, r.gpuLayersRequested());
        assertNull(r.gpuOffloadActive(), "unknown offload status must be null");
    }

    @Test
    void gpuOffloadActiveCanBeFalseWhenCpuOnlyBuildDetected() {
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadRequested(true)
                .gpuLayersRequested(99)
                .gpuOffloadActive(false)  // CPU-only build detected
                .metrics(sampleMetrics())
                .addEnvironmentNote("GPU offload requested but CPU-only build detected")
                .build();

        assertTrue(r.gpuOffloadRequested());
        assertEquals(Boolean.FALSE, r.gpuOffloadActive());
    }

    @Test
    void toJsonContainsGpuOffloadFields() {
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadRequested(true)
                .gpuLayersRequested(32)
                .gpuOffloadActive(null)
                .metrics(sampleMetrics())
                .build();

        String json = BenchmarkExporter.toJson(r);
        assertTrue(json.contains("\"gpuOffloadRequested\""), "JSON must contain gpuOffloadRequested");
        assertTrue(json.contains("\"gpuLayersRequested\""),  "JSON must contain gpuLayersRequested");
        assertTrue(json.contains("\"gpuOffloadActive\""),    "JSON must contain gpuOffloadActive");
        assertTrue(json.contains("true"),  "gpuOffloadRequested must be true");
        assertTrue(json.contains("32"),    "gpuLayersRequested must be 32");
    }

    @Test
    void toJsonRendersUnknownGpuOffloadActiveAsNull() {
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadRequested(true)
                .gpuLayersRequested(32)
                .gpuOffloadActive(null)
                .metrics(sampleMetrics())
                .build();

        String json = BenchmarkExporter.toJson(r);
        assertTrue(json.contains("\"gpuOffloadActive\": null") || json.contains("\"gpuOffloadActive\":null"),
                "unknown gpuOffloadActive must render as JSON null");
    }

    @Test
    void csvRowColumnCountMatchesHeaderWithGpuFields() {
        // Build a result with GPU fields set
        BenchmarkRunResult r = BenchmarkRunResult.builder()
                .backendName("llama.cpp")
                .backendMode(BenchmarkRunResult.BackendMode.LLAMA_CPP_REAL)
                .isRealInference(true)
                .gpuOffloadRequested(true)
                .gpuLayersRequested(32)
                .gpuOffloadActive(null)
                .requestedMaxNewTokens(32)
                .metrics(sampleMetrics())
                .build();

        String header = BenchmarkExporter.CSV_HEADER;
        String row    = BenchmarkExporter.toCsvRow(r);
        assertEquals(header.split(",", -1).length, row.split(",", -1).length,
                "CSV row column count must match header after adding GPU fields");
    }
}
