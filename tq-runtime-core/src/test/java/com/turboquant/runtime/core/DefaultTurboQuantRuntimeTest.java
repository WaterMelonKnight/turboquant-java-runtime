package com.turboquant.runtime.core;

import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.TurboQuantRuntime;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for {@link DefaultTurboQuantRuntime}.
 *
 * <p>Runs entirely against {@code cpu-stub}; no GPU or native library required.</p>
 */
class DefaultTurboQuantRuntimeTest {

    private static BackendConfig defaultConfig() {
        return BackendConfig.defaultConfig();
    }

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    @Test
    void autoSelectCreatesRuntime() {
        try (DefaultTurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            assertNotNull(rt);
            assertNotNull(rt.backend());
        }
    }

    @Test
    void withBackend_cpuStub_createsRuntime() {
        try (DefaultTurboQuantRuntime rt =
                     DefaultTurboQuantRuntime.withBackend("cpu-stub", defaultConfig())) {
            assertEquals("cpu-stub", rt.backend().name());
        }
    }

    @Test
    void withBackend_unknownName_throwsBackendException() {
        assertThrows(BackendException.class,
                () -> DefaultTurboQuantRuntime.withBackend("nonexistent", defaultConfig()));
    }

    // -------------------------------------------------------------------------
    // Backend introspection
    // -------------------------------------------------------------------------

    @Test
    void cpuStubHasCpuFallbackCapability() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            assertTrue(rt.backend().capabilities().contains(BackendCapability.CPU_FALLBACK),
                    "cpu-stub must advertise CPU_FALLBACK capability");
        }
    }

    @Test
    void backendVersionIsNonBlank() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            assertFalse(rt.backend().version().isBlank(),
                    "backend.version() must not be blank");
        }
    }

    // -------------------------------------------------------------------------
    // Inference basics
    // -------------------------------------------------------------------------

    @Test
    void inferReturnsNonNullResult() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            InferenceResult result = rt.infer(InferenceRequest.syntheticPrompt(8, 4));
            assertNotNull(result);
        }
    }

    @Test
    void inferResultPromptTokenCountMatchesInput() {
        int promptLen = 16;
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            InferenceResult result = rt.infer(InferenceRequest.syntheticPrompt(promptLen, 8));
            assertEquals(promptLen, result.promptTokenCount());
        }
    }

    @Test
    void inferResultGeneratedTokenCountMatchesRequest() {
        int genLen = 12;
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            InferenceResult result = rt.infer(InferenceRequest.syntheticPrompt(8, genLen));
            assertEquals(genLen, result.generatedTokenCount());
            assertEquals(genLen, result.generatedTokenIds().length);
        }
    }

    @Test
    void inferResultBackendNameIsCpuStub() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            InferenceResult result = rt.infer(InferenceRequest.syntheticPrompt(4, 2));
            assertEquals("cpu-stub", result.backendName());
        }
    }

    @Test
    void inferLatencyIsPositive() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            InferenceResult result = rt.infer(InferenceRequest.syntheticPrompt(4, 2));
            assertTrue(result.inferenceNanos() > 0,
                    "inferenceNanos must be > 0");
        }
    }

    @Test
    void inferIsDeterministicForSamePrompt() {
        InferenceRequest req = InferenceRequest.syntheticPrompt(8, 4);
        // Each call creates a fresh session (new runtime), so the stub LCG resets.
        int[] first, second;
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            first = rt.infer(req).generatedTokenIds();
        }
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            second = rt.infer(req).generatedTokenIds();
        }
        assertArrayEquals(first, second,
                "Same prompt must produce same tokens across independent sessions");
    }

    // -------------------------------------------------------------------------
    // KV cache state
    // -------------------------------------------------------------------------

    @Test
    void kvCacheStatsAreEmptyBeforeFirstInfer() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            KvCacheStats stats = rt.kvCacheStats();
            assertEquals(0, stats.cachedTokens(),
                    "cachedTokens must be 0 before any inference");
        }
    }

    @Test
    void kvCacheGrowsAfterInfer() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            rt.infer(InferenceRequest.syntheticPrompt(32, 8));
            assertTrue(rt.kvCacheStats().cachedTokens() > 0,
                    "cachedTokens must grow after inference");
        }
    }

    @Test
    void kvCacheAccumulatesAcrossMultipleCalls() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            rt.infer(InferenceRequest.syntheticPrompt(16, 4));
            int after1 = rt.kvCacheStats().cachedTokens();

            rt.infer(InferenceRequest.syntheticPrompt(16, 4));
            int after2 = rt.kvCacheStats().cachedTokens();

            assertTrue(after2 >= after1,
                    "cachedTokens should be non-decreasing across calls");
        }
    }

    @Test
    void kvCacheSizeBytesIsPositiveAfterInfer() {
        try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig())) {
            rt.infer(InferenceRequest.syntheticPrompt(8, 4));
            assertTrue(rt.kvCacheStats().cacheSizeBytes() > 0,
                    "cacheSizeBytes must be > 0 after inference");
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle / close
    // -------------------------------------------------------------------------

    @Test
    void closedRuntimeThrowsOnSubsequentInfer() {
        DefaultTurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig());
        rt.close();
        assertThrows(BackendException.class,
                () -> rt.infer(InferenceRequest.syntheticPrompt(4, 2)),
                "Calling infer() on a closed runtime must throw BackendException");
    }

    @Test
    void closeIsIdempotent() {
        DefaultTurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(defaultConfig());
        rt.close();
        assertDoesNotThrow(rt::close, "Calling close() a second time must not throw");
    }
}
