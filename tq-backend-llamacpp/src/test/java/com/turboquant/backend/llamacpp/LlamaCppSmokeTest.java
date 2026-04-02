package com.turboquant.backend.llamacpp;

import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.SessionConfig;
import com.turboquant.runtime.api.TurboQuantRuntime;
import com.turboquant.runtime.core.DefaultTurboQuantRuntime;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Optional smoke test for the llama.cpp inference backend.
 *
 * <p>This test is <b>skipped automatically</b> in standard CI unless a local GGUF
 * model path is provided via the {@code tq.test.llamacpp.model} system property.</p>
 *
 * <h2>Running locally</h2>
 * <pre>
 *   mvn test -pl tq-backend-llamacpp \
 *       -Dtq.test.llamacpp.model=/path/to/model.gguf
 *
 *   # Optional overrides (defaults shown):
 *       -Dtq.test.llamacpp.context=2048
 *       -Dtq.test.llamacpp.maxNewTokens=32
 *       -Dtq.test.llamacpp.prompt="Once upon a time"
 * </pre>
 *
 * <p>The test does <b>not</b> assert GPU offload — it runs in whatever mode
 * llama.cpp selects (typically CPU-only unless a GPU build is present).</p>
 *
 * <p>See {@code docs/llamacpp-smoke-test.md} for model preparation notes.</p>
 */
class LlamaCppSmokeTest {

    private static final String PROP_MODEL     = "tq.test.llamacpp.model";
    private static final String PROP_CONTEXT   = "tq.test.llamacpp.context";
    private static final String PROP_MAX_NEW   = "tq.test.llamacpp.maxNewTokens";
    private static final String PROP_PROMPT    = "tq.test.llamacpp.prompt";

    @Test
    void smokeInference() {
        String modelPath = System.getProperty(PROP_MODEL);
        Assumptions.assumeTrue(
                modelPath != null && !modelPath.isBlank(),
                "Skipping llama.cpp smoke test — set -D" + PROP_MODEL + "=/path/to/model.gguf to run.");

        int contextTokens = intProp(PROP_CONTEXT, 2048);
        int maxNewTokens  = intProp(PROP_MAX_NEW,  32);
        String prompt     = System.getProperty(PROP_PROMPT, "Once upon a time");

        SessionConfig cfg = SessionConfig.builder()
                .modelPath(modelPath)
                .maxContextTokens(contextTokens)
                .maxNewTokens(maxNewTokens)
                .temperature(0.8f)
                .topP(0.9f)
                .build();

        long wallStart = System.nanoTime();

        try (TurboQuantRuntime runtime =
                     DefaultTurboQuantRuntime.withBackend("llama.cpp", cfg)) {

            assertEquals("llama.cpp", runtime.backend().name(),
                    "Backend name must be 'llama.cpp'");

            InferenceRequest request = InferenceRequest.fromText(prompt, maxNewTokens);
            InferenceResult result   = runtime.infer(request);

            // --- assertions ---
            assertNotNull(result, "InferenceResult must not be null");
            assertNotNull(result.generatedText(),
                    "generatedText() must not be null for llama.cpp backend");
            assertFalse(result.generatedText().isBlank(),
                    "generatedText() must not be blank — backend produced no output");
            assertEquals("llama.cpp", result.backendName());
            assertTrue(result.generatedTokenCount() > 0,
                    "generatedTokenCount() must be > 0");
            assertTrue(result.inferenceNanos() > 0,
                    "inferenceNanos() must be > 0");

            // --- summary log ---
            long wallNanos = System.nanoTime() - wallStart;
            double tokPerSec = result.generatedTokenCount() * 1_000_000_000.0
                    / Math.max(1, result.inferenceNanos());

            System.out.println();
            System.out.println("=== llama.cpp smoke test summary ===");
            System.out.printf("  model path      : %s%n", modelPath);
            System.out.printf("  prompt          : %s%n", prompt);
            System.out.printf("  context tokens  : %d%n", contextTokens);
            System.out.printf("  max new tokens  : %d%n", maxNewTokens);
            System.out.printf("  generated tokens: %d%n", result.generatedTokenCount());
            System.out.printf("  inference time  : %.1f ms%n",
                    result.inferenceNanos() / 1_000_000.0);
            System.out.printf("  throughput      : %.1f tok/s%n", tokPerSec);
            System.out.printf("  wall time       : %.1f ms%n",
                    wallNanos / 1_000_000.0);
            System.out.println("  --- generated text ---");
            System.out.println("  " + result.generatedText().replace("\n", "\n  "));
            System.out.println("=====================================");
        }
    }

    private static int intProp(String key, int defaultValue) {
        String val = System.getProperty(key);
        if (val == null || val.isBlank()) return defaultValue;
        try {
            return Integer.parseInt(val.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}
