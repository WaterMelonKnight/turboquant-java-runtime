package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.TensorHandle;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CpuComputeSession} covering the full session lifecycle,
 * low-level tensor operations, stub inference, and KV cache state tracking.
 */
class CpuSessionLifecycleTest {

    private CpuBackend backend;
    private ComputeSession session;

    @BeforeEach
    void setUp() {
        backend = new CpuBackend();
        backend.init(BackendConfig.defaultConfig());
        session = backend.newSession();
    }

    @AfterEach
    void tearDown() {
        session.close();
        backend.close();
    }

    // -------------------------------------------------------------------------
    // Tensor allocation
    // -------------------------------------------------------------------------

    @Test
    void allocateReturnsTensorWithCorrectShape() {
        try (TensorHandle h = session.allocate(DType.FLOAT32, 4, 8)) {
            assertArrayEquals(new long[]{4, 8}, h.shape());
        }
    }

    @Test
    void allocateReturnsTensorWithCorrectDtype() {
        try (TensorHandle h = session.allocate(DType.INT8, 16)) {
            assertEquals(DType.INT8, h.dtype());
        }
    }

    @Test
    void allocateIsZeroFilled() {
        try (TensorHandle h = session.allocate(DType.FLOAT32, 4)) {
            float[] data = h.toFloatArray();
            for (float v : data) {
                assertEquals(0.0f, v, "allocate() must return zero-filled tensor");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Upload / download round-trip
    // -------------------------------------------------------------------------

    @Test
    void uploadPreservesData() {
        float[] src = {1.0f, 2.0f, 3.0f, 4.0f};
        try (TensorHandle h = session.upload(src, 4)) {
            assertArrayEquals(src, h.toFloatArray(), 1e-6f);
        }
    }

    @Test
    void uploadShapeIsPreserved() {
        float[] src = new float[6];
        try (TensorHandle h = session.upload(src, 2, 3)) {
            assertArrayEquals(new long[]{2, 3}, h.shape());
        }
    }

    @Test
    void toFloatArrayThrowsForNonFloat32() {
        try (TensorHandle h = session.allocate(DType.INT8, 4)) {
            assertThrows(UnsupportedOperationException.class, h::toFloatArray);
        }
    }

    // -------------------------------------------------------------------------
    // Quantise / dequantise (stubs)
    // -------------------------------------------------------------------------

    @Test
    void quantiseInt8ReturnsSameShape() {
        try (TensorHandle input = session.upload(new float[16], 4, 4);
             TensorHandle q = session.quantiseInt8(input)) {
            assertArrayEquals(input.shape(), q.shape());
        }
    }

    @Test
    void quantiseInt8ReturnsInt8Dtype() {
        try (TensorHandle input = session.upload(new float[8], 8);
             TensorHandle q = session.quantiseInt8(input)) {
            assertEquals(DType.INT8, q.dtype());
        }
    }

    @Test
    void dequantiseReturnsSameShape() {
        try (TensorHandle input = session.allocate(DType.INT8, 4, 4);
             TensorHandle dq = session.dequantise(input)) {
            assertArrayEquals(input.shape(), dq.shape());
        }
    }

    @Test
    void dequantiseReturnsFloat32Dtype() {
        try (TensorHandle input = session.allocate(DType.INT8, 4);
             TensorHandle dq = session.dequantise(input)) {
            assertEquals(DType.FLOAT32, dq.dtype());
        }
    }

    // -------------------------------------------------------------------------
    // Synchronize (no-op)
    // -------------------------------------------------------------------------

    @Test
    void synchronizeDoesNotThrow() {
        assertDoesNotThrow(() -> session.synchronize());
    }

    // -------------------------------------------------------------------------
    // Inference stub
    // -------------------------------------------------------------------------

    @Test
    void inferReturnsNonNullResult() {
        InferenceRequest req = InferenceRequest.syntheticPrompt(8, 4);
        InferenceResult result = session.infer(req);
        assertNotNull(result);
    }

    @Test
    void inferResultPromptTokenCountMatchesInput() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(16, 8));
        assertEquals(16, result.promptTokenCount());
    }

    @Test
    void inferResultGeneratedTokenCountMatchesRequest() {
        int genLen = 10;
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(8, genLen));
        assertEquals(genLen, result.generatedTokenCount());
        assertEquals(genLen, result.generatedTokenIds().length);
    }

    @Test
    void inferResultGeneratedTokenIdsAreInVocabRange() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(8, 20));
        for (int tokenId : result.generatedTokenIds()) {
            assertTrue(tokenId >= 0 && tokenId < CpuComputeSession.STUB_VOCAB_SIZE,
                    "Generated token id " + tokenId + " is out of vocab range");
        }
    }

    @Test
    void inferLastLogitsHasCorrectLength() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(4, 3));
        assertEquals(CpuComputeSession.STUB_VOCAB_SIZE, result.lastLogits().length);
    }

    @Test
    void inferLastLogitsHasSingleNonZeroEntry() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(4, 3));
        float[] logits = result.lastLogits();
        int nonZero = 0;
        for (float v : logits) {
            if (v != 0.0f) nonZero++;
        }
        assertEquals(1, nonZero,
                "Stub backend must set exactly one non-zero logit (the last generated token)");
    }

    @Test
    void inferIsDeterministicAcrossCallsOnFreshSession() {
        InferenceRequest req = InferenceRequest.syntheticPrompt(8, 4);
        // Two independent sessions, same prompt → same output
        int[] first;
        try (ComputeSession s1 = backend.newSession()) {
            first = s1.infer(req).generatedTokenIds();
        }
        int[] second;
        try (ComputeSession s2 = backend.newSession()) {
            second = s2.infer(req).generatedTokenIds();
        }
        assertArrayEquals(first, second,
                "Stub LCG must produce the same output for the same prompt on independent sessions");
    }

    @Test
    void inferLatencyIsPositive() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(8, 4));
        assertTrue(result.inferenceNanos() > 0);
    }

    @Test
    void inferBackendNameIsCpuStub() {
        InferenceResult result = session.infer(InferenceRequest.syntheticPrompt(4, 2));
        assertEquals("cpu-stub", result.backendName());
    }

    // -------------------------------------------------------------------------
    // KV cache state
    // -------------------------------------------------------------------------

    @Test
    void kvCacheIsEmptyBeforeAnyInfer() {
        KvCacheStats stats = session.kvCacheStats();
        assertEquals(0, stats.cachedTokens());
        assertEquals(0L, stats.cacheSizeBytes());
        assertEquals(0.0, stats.hitRate());
    }

    @Test
    void kvCacheCapacityIsStubConstant() {
        KvCacheStats stats = session.kvCacheStats();
        // capacity is reported even before any inference
        assertEquals(CpuComputeSession.STUB_CACHE_CAPACITY, stats.capacityTokens());
    }

    @Test
    void kvCacheTokensGrowAfterInfer() {
        session.infer(InferenceRequest.syntheticPrompt(32, 8));
        KvCacheStats stats = session.kvCacheStats();
        assertTrue(stats.cachedTokens() > 0);
    }

    @Test
    void kvCacheSizeBytesGrowsWithCachedTokens() {
        session.infer(InferenceRequest.syntheticPrompt(16, 4));
        KvCacheStats after1 = session.kvCacheStats();

        session.infer(InferenceRequest.syntheticPrompt(16, 4));
        KvCacheStats after2 = session.kvCacheStats();

        assertTrue(after2.cacheSizeBytes() >= after1.cacheSizeBytes(),
                "cacheSizeBytes must be non-decreasing");
    }

    @Test
    void kvCacheHitRateIsZeroOnFirstInfer() {
        session.infer(InferenceRequest.syntheticPrompt(8, 4));
        assertEquals(0.0, session.kvCacheStats().hitRate(),
                "hitRate must be 0.0 after the very first inference");
    }

    @Test
    void kvCacheHitRateGrowsWithSubsequentInferences() {
        session.infer(InferenceRequest.syntheticPrompt(8, 4)); // hit rate = 0.0
        session.infer(InferenceRequest.syntheticPrompt(8, 4)); // hit rate > 0.0
        assertTrue(session.kvCacheStats().hitRate() > 0.0,
                "hitRate must grow after repeated inferences");
    }

    @Test
    void kvCacheHitRateNeverExceedsOne() {
        for (int i = 0; i < 20; i++) {
            session.infer(InferenceRequest.syntheticPrompt(8, 4));
        }
        assertTrue(session.kvCacheStats().hitRate() <= 1.0,
                "hitRate must never exceed 1.0");
    }

    @Test
    void kvCacheCachedTokensDoesNotExceedCapacity() {
        // Flood with tokens that would overflow a real cache
        for (int i = 0; i < 50; i++) {
            session.infer(InferenceRequest.syntheticPrompt(128, 32));
        }
        KvCacheStats stats = session.kvCacheStats();
        assertTrue(stats.cachedTokens() <= stats.capacityTokens(),
                "cachedTokens must never exceed capacityTokens");
    }
}
