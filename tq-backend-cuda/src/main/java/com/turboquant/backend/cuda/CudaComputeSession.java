package com.turboquant.backend.cuda;

import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.TensorHandle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CUDA implementation of {@link ComputeSession}.
 *
 * <p>Each session is backed by an opaque native session handle obtained from
 * {@code tq_session_create}.  The handle owns an internal command stream;
 * {@link #synchronize()} drains it.  {@link #infer(InferenceRequest)} runs
 * the full stub inference path through the C ABI and returns a populated
 * {@link InferenceResult} together with up-to-date KV cache stats.</p>
 *
 * <p>Tensor-level operations ({@link #allocate}, {@link #upload},
 * {@link #quantiseInt8}, {@link #dequantise}) use the low-level
 * {@code tq_malloc} / {@code tq_upload_float32} / … functions via the
 * default stream (handle 0, which is valid in the mock).</p>
 */
final class CudaComputeSession implements ComputeSession {

    private static final Logger log = LoggerFactory.getLogger(CudaComputeSession.class);

    /** Opaque native session handle from tq_session_create. */
    private final long sessionHandle;

    /** Snapshot of KV cache state; updated after every infer() call. */
    private volatile KvCacheStats lastKvStats = KvCacheStats.empty();

    CudaComputeSession(long sessionHandle) {
        this.sessionHandle = sessionHandle;
    }

    // -------------------------------------------------------------------------
    // Tensor allocation
    // -------------------------------------------------------------------------

    @Override
    public TensorHandle allocate(DType dtype, long... shape) {
        long numel = 1;
        for (long d : shape) numel *= d;
        long bytes = numel * bytesPerElement(dtype);
        long ptr = CudaNativeBridge.tqMalloc(bytes);
        log.trace("allocate dtype={} numel={} ptr=0x{}", dtype, numel, Long.toHexString(ptr));
        return new CudaTensorHandle(ptr, dtype, shape);
    }

    @Override
    public TensorHandle upload(float[] data, long... shape) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = CudaNativeBridge.tqMalloc(bytes);
        CudaNativeBridge.tqUploadFloat(ptr, data, 0L /* default stream */);
        log.trace("upload {} floats ptr=0x{}", data.length, Long.toHexString(ptr));
        return new CudaTensorHandle(ptr, DType.FLOAT32, shape);
    }

    // -------------------------------------------------------------------------
    // Quantisation
    // -------------------------------------------------------------------------

    @Override
    public TensorHandle quantiseInt8(TensorHandle input) {
        CudaTensorHandle src = (CudaTensorHandle) input;
        long numel    = src.numel();
        long dstPtr   = CudaNativeBridge.tqMalloc(numel);           // 1 byte per INT8 element
        long scalePtr = CudaNativeBridge.tqMalloc(Float.BYTES);
        CudaNativeBridge.tqQuantiseInt8(dstPtr, src.devicePtr(), numel, scalePtr, 0L);
        CudaNativeBridge.tqFree(scalePtr); // scale retrieval not wired through yet
        return new CudaTensorHandle(dstPtr, DType.INT8, input.shape());
    }

    @Override
    public TensorHandle dequantise(TensorHandle input) {
        CudaTensorHandle src = (CudaTensorHandle) input;
        long numel  = src.numel();
        long dstPtr = CudaNativeBridge.tqMalloc(numel * Float.BYTES);
        CudaNativeBridge.tqDequantiseInt8(dstPtr, src.devicePtr(), numel, 1.0f, 0L);
        return new CudaTensorHandle(dstPtr, DType.FLOAT32, input.shape());
    }

    // -------------------------------------------------------------------------
    // Stream synchronisation
    // -------------------------------------------------------------------------

    @Override
    public void synchronize() {
        CudaNativeBridge.tqSessionSynchronize(sessionHandle);
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    /**
     * Run a stub inference call through the C ABI.
     *
     * <ol>
     *   <li>Calls {@code tq_session_infer} via JNI to get a native result handle.</li>
     *   <li>Extracts all fields using individual getter methods.</li>
     *   <li>Frees the native handle with {@code tqResultFree}.</li>
     *   <li>Returns a fully populated {@link InferenceResult}.</li>
     * </ol>
     */
    @Override
    public InferenceResult infer(InferenceRequest request) {
        long resultHandle = CudaNativeBridge.tqSessionInfer(
                sessionHandle,
                request.inputTokenIds(),
                request.maxNewTokens());
        if (resultHandle == 0L) {
            throw new BackendException("tq_session_infer returned a null result handle.");
        }
        try {
            int[]   generatedIds   = CudaNativeBridge.tqResultGetGeneratedIds(resultHandle);
            float[] lastLogits     = CudaNativeBridge.tqResultGetLastLogits(resultHandle);
            long    inferenceNanos = CudaNativeBridge.tqResultGetInferenceNanos(resultHandle);
            int     promptCount    = CudaNativeBridge.tqResultGetPromptCount(resultHandle);
            int     kvCached       = CudaNativeBridge.tqResultGetKvCachedTokens(resultHandle);
            int     kvCapacity     = CudaNativeBridge.tqResultGetKvCapacityTokens(resultHandle);
            long    kvSizeBytes    = CudaNativeBridge.tqResultGetKvSizeBytes(resultHandle);
            double  kvHitRate      = CudaNativeBridge.tqResultGetKvHitRate(resultHandle);

            lastKvStats = new KvCacheStats(kvCached, kvCapacity, kvSizeBytes, kvHitRate);

            return InferenceResult.builder()
                    .generatedTokenIds(generatedIds)
                    .lastLogits(lastLogits)
                    .promptTokenCount(promptCount)
                    .generatedTokenCount(generatedIds.length)
                    .inferenceNanos(inferenceNanos)
                    .backendName("cuda")
                    .build();
        } finally {
            CudaNativeBridge.tqResultFree(resultHandle);
        }
    }

    // -------------------------------------------------------------------------
    // KV cache stats
    // -------------------------------------------------------------------------

    @Override
    public KvCacheStats kvCacheStats() {
        return lastKvStats;
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    @Override
    public void close() {
        CudaNativeBridge.tqSessionDestroy(sessionHandle);
        log.trace("CudaComputeSession closed (handle=0x{})", Long.toHexString(sessionHandle));
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static long bytesPerElement(DType dtype) {
        return switch (dtype) {
            case FLOAT32  -> Float.BYTES;
            case BFLOAT16 -> 2;
            case INT8     -> 1;
            case UINT4    -> 1; // two elements per byte — packing handled in kernel
        };
    }
}
