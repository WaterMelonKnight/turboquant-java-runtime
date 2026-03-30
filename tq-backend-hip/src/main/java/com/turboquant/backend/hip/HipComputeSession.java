package com.turboquant.backend.hip;

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
 * HIP/ROCm implementation of {@link ComputeSession}.
 *
 * <p>Structurally identical to {@code CudaComputeSession}: both hold an opaque
 * native session handle ({@code long sessionHandle}) and delegate all work to
 * the C ABI via their respective JNI bridges.  The shared ABI design means
 * this class requires <em>no changes</em> when the HIP kernels are implemented;
 * only {@code native/hip/} changes.</p>
 *
 * <p>Tensor-level operations ({@link #allocate}, {@link #upload},
 * {@link #quantiseInt8}, {@link #dequantise}) use the low-level
 * {@code tq_malloc} / {@code tq_upload_float32} / … functions
 * via the default stream (handle {@code 0}).</p>
 */
final class HipComputeSession implements ComputeSession {

    private static final Logger log = LoggerFactory.getLogger(HipComputeSession.class);

    /** Opaque native session handle from tq_session_create. */
    private final long sessionHandle;

    /** Snapshot of KV cache state; updated after every infer() call. */
    private volatile KvCacheStats lastKvStats = KvCacheStats.empty();

    HipComputeSession(long sessionHandle) {
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
        long ptr = HipNativeBridge.tqMalloc(bytes);
        log.trace("allocate dtype={} numel={} ptr=0x{}", dtype, numel, Long.toHexString(ptr));
        return new HipTensorHandle(ptr, dtype, shape);
    }

    @Override
    public TensorHandle upload(float[] data, long... shape) {
        long bytes = (long) data.length * Float.BYTES;
        long ptr = HipNativeBridge.tqMalloc(bytes);
        HipNativeBridge.tqUploadFloat(ptr, data, 0L /* default stream */);
        log.trace("upload {} floats ptr=0x{}", data.length, Long.toHexString(ptr));
        return new HipTensorHandle(ptr, DType.FLOAT32, shape);
    }

    // -------------------------------------------------------------------------
    // Quantisation
    // -------------------------------------------------------------------------

    @Override
    public TensorHandle quantiseInt8(TensorHandle input) {
        HipTensorHandle src = (HipTensorHandle) input;
        long numel    = src.numel();
        long dstPtr   = HipNativeBridge.tqMalloc(numel);           // 1 byte per INT8 element
        long scalePtr = HipNativeBridge.tqMalloc(Float.BYTES);
        HipNativeBridge.tqQuantiseInt8(dstPtr, src.devicePtr(), numel, scalePtr, 0L);
        HipNativeBridge.tqFree(scalePtr);
        return new HipTensorHandle(dstPtr, DType.INT8, input.shape());
    }

    @Override
    public TensorHandle dequantise(TensorHandle input) {
        HipTensorHandle src = (HipTensorHandle) input;
        long numel  = src.numel();
        long dstPtr = HipNativeBridge.tqMalloc(numel * Float.BYTES);
        HipNativeBridge.tqDequantiseInt8(dstPtr, src.devicePtr(), numel, 1.0f, 0L);
        return new HipTensorHandle(dstPtr, DType.FLOAT32, input.shape());
    }

    // -------------------------------------------------------------------------
    // Stream synchronisation
    // -------------------------------------------------------------------------

    @Override
    public void synchronize() {
        HipNativeBridge.tqSessionSynchronize(sessionHandle);
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
     *
     * <p>When real HIP kernels are available, only the C implementation of
     * {@code tq_session_infer} changes; this method remains untouched.</p>
     */
    @Override
    public InferenceResult infer(InferenceRequest request) {
        long resultHandle = HipNativeBridge.tqSessionInfer(
                sessionHandle,
                request.inputTokenIds(),
                request.maxNewTokens());
        if (resultHandle == 0L) {
            throw new BackendException("tq_session_infer returned a null result handle.");
        }
        try {
            int[]   generatedIds   = HipNativeBridge.tqResultGetGeneratedIds(resultHandle);
            float[] lastLogits     = HipNativeBridge.tqResultGetLastLogits(resultHandle);
            long    inferenceNanos = HipNativeBridge.tqResultGetInferenceNanos(resultHandle);
            int     promptCount    = HipNativeBridge.tqResultGetPromptCount(resultHandle);
            int     kvCached       = HipNativeBridge.tqResultGetKvCachedTokens(resultHandle);
            int     kvCapacity     = HipNativeBridge.tqResultGetKvCapacityTokens(resultHandle);
            long    kvSizeBytes    = HipNativeBridge.tqResultGetKvSizeBytes(resultHandle);
            double  kvHitRate      = HipNativeBridge.tqResultGetKvHitRate(resultHandle);

            lastKvStats = new KvCacheStats(kvCached, kvCapacity, kvSizeBytes, kvHitRate);

            return InferenceResult.builder()
                    .generatedTokenIds(generatedIds)
                    .lastLogits(lastLogits)
                    .promptTokenCount(promptCount)
                    .generatedTokenCount(generatedIds.length)
                    .inferenceNanos(inferenceNanos)
                    .backendName("hip")
                    .build();
        } finally {
            HipNativeBridge.tqResultFree(resultHandle);
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
        HipNativeBridge.tqSessionDestroy(sessionHandle);
        log.trace("HipComputeSession closed (handle=0x{})", Long.toHexString(sessionHandle));
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
