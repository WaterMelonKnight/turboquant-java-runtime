package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.TensorHandle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CPU stub implementation of {@link ComputeSession}.
 *
 * <p>All kernel operations are synchronous, pure-Java stubs.
 * No real quantisation or inference is performed; the class is intended
 * for CI, unit tests, and environments without a GPU.</p>
 *
 * <h2>Inference stub behaviour</h2>
 * <ul>
 *   <li>Generates {@code maxNewTokens} tokens using a simple LCG seeded from
 *       the last input token — deterministic and testable.</li>
 *   <li>Returns a sparse {@code lastLogits} array (length {@link #STUB_VOCAB_SIZE})
 *       with a single non-zero entry at the index of the last generated token.</li>
 *   <li>KV cache state accumulates across calls within the same session.</li>
 * </ul>
 */
final class CpuComputeSession implements ComputeSession {

    private static final Logger log = LoggerFactory.getLogger(CpuComputeSession.class);

    /** Placeholder vocabulary size matching a typical LLaMA-family model. */
    static final int STUB_VOCAB_SIZE = 32_000;

    /**
     * Placeholder KV cache capacity in tokens.
     * Represents a 2 048-token context window.
     */
    static final int STUB_CACHE_CAPACITY = 2_048;

    /**
     * Bytes charged per cached token in the placeholder KV cache budget.
     * Derived from 2 (K+V) × 32 (layers) × 128 (head-dim) × 2 (bf16) = 16 384 bytes,
     * rounded down to 256 for brevity.
     */
    private static final long BYTES_PER_CACHED_TOKEN = 256L;

    // -------------------------------------------------------------------------
    // KV cache state — accumulates across infer() calls within this session
    // -------------------------------------------------------------------------

    private int  cachedTokens    = 0;
    private long cacheSizeBytes  = 0L;
    private int  totalInferences = 0;

    // -------------------------------------------------------------------------
    // Low-level tensor operations
    // -------------------------------------------------------------------------

    @Override
    public TensorHandle allocate(DType dtype, long... shape) {
        log.trace("allocate dtype={} shape={}", dtype, shape);
        return new CpuTensorHandle(dtype, shape);
    }

    @Override
    public TensorHandle upload(float[] data, long... shape) {
        log.trace("upload {} floats shape={}", data.length, shape);
        return new CpuTensorHandle(data, shape);
    }

    @Override
    public TensorHandle quantiseInt8(TensorHandle input) {
        // Placeholder: real INT8 symmetric per-tensor quantisation not yet implemented.
        log.debug("quantiseInt8 [stub] shape={}", (Object) input.shape());
        return new CpuTensorHandle(DType.INT8, input.shape());
    }

    @Override
    public TensorHandle dequantise(TensorHandle input) {
        // Placeholder: real dequantisation not yet implemented.
        log.debug("dequantise [stub] shape={}", (Object) input.shape());
        return new CpuTensorHandle(DType.FLOAT32, input.shape());
    }

    @Override
    public void synchronize() {
        // CPU stub: all operations are synchronous — nothing to wait for.
    }

    // -------------------------------------------------------------------------
    // Inference stub
    // -------------------------------------------------------------------------

    /**
     * Runs a stub forward pass.
     *
     * <p>Token generation uses a linear congruential generator seeded from
     * the last input token, so results are deterministic given the same
     * prompt suffix.</p>
     */
    @Override
    public InferenceResult infer(InferenceRequest request) {
        long t0 = System.nanoTime();

        int[] inputIds   = request.inputTokenIds();
        int   maxNew     = request.maxNewTokens();

        // --- token generation (stub LCG) ---
        int[] generated  = new int[maxNew];
        int   lcgState   = inputIds[inputIds.length - 1]; // seed from last prompt token
        for (int i = 0; i < maxNew; i++) {
            // 32-bit Park–Miller LCG — deterministic, no imports required
            lcgState   = (int) ((lcgState * 1_103_515_245L + 12_345L) & 0x7fff_ffffL);
            generated[i] = lcgState % STUB_VOCAB_SIZE;
        }

        // --- sparse logit vector (single spike at last generated token) ---
        float[] lastLogits = new float[STUB_VOCAB_SIZE];
        if (maxNew > 0) {
            lastLogits[generated[maxNew - 1]] = 1.0f;
        }

        // --- update KV cache state ---
        cachedTokens = Math.min(cachedTokens + inputIds.length + maxNew, STUB_CACHE_CAPACITY);
        cacheSizeBytes = (long) cachedTokens * BYTES_PER_CACHED_TOKEN;
        totalInferences++;

        long inferenceNanos = System.nanoTime() - t0;

        log.debug("infer [stub] promptLen={} genLen={} latency={}µs",
                inputIds.length, maxNew, inferenceNanos / 1_000);

        return InferenceResult.builder()
                .generatedTokenIds(generated)
                .lastLogits(lastLogits)
                .promptTokenCount(inputIds.length)
                .generatedTokenCount(maxNew)
                .inferenceNanos(inferenceNanos)
                .backendName("cpu-stub")
                .build();
    }

    // -------------------------------------------------------------------------
    // KV cache snapshot
    // -------------------------------------------------------------------------

    @Override
    public KvCacheStats kvCacheStats() {
        // Hit rate grows with each repeated call (tokens from prior iterations
        // are already cached). Saturates at ~90 % to model a realistic steady state.
        double hitRate = totalInferences > 1
                ? Math.min(0.9, (totalInferences - 1.0) / totalInferences)
                : 0.0;
        return new KvCacheStats(cachedTokens, STUB_CACHE_CAPACITY, cacheSizeBytes, hitRate);
    }

    // -------------------------------------------------------------------------

    @Override
    public void close() {
        log.trace("CpuComputeSession closed (totalInferences={})", totalInferences);
    }
}
