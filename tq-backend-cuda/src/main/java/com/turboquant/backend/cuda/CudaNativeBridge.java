package com.turboquant.backend.cuda;

/**
 * JNI native bridge for the CUDA backend.
 *
 * <p>Wraps the C ABI declared in {@code tq_native_api.h}.  The shared library
 * that implements these methods is {@code libtq_cuda.so} (Linux) /
 * {@code tq_cuda.dll} (Windows), built from {@code native/cuda/}.</p>
 *
 * <h2>Two-layer API</h2>
 * <dl>
 *   <dt>High-level (runtime / session / inference)</dt>
 *   <dd>Java callers create an opaque runtime handle via {@link #tqRuntimeCreate},
 *       derive session handles from it, and run inference through
 *       {@link #tqSessionInfer}.  The inference result is returned as an opaque
 *       {@code long} result-handle; individual fields are retrieved with the
 *       {@code tqResult*} getters and the handle is released with
 *       {@link #tqResultFree}.</dd>
 *   <dt>Low-level (tensor memory / stream / quantisation)</dt>
 *   <dd>Used by {@link CudaComputeSession} for direct tensor operations
 *       ({@code allocate}, {@code upload}, {@code quantiseInt8}, …).
 *       Device pointers are represented as {@code long}.</dd>
 * </dl>
 *
 * <h2>Access control</h2>
 * All methods are package-private; public operations go through the
 * {@link com.turboquant.runtime.api.Backend} /
 * {@link com.turboquant.runtime.api.ComputeSession} interfaces.
 *
 * <h2>JNI name mangling</h2>
 * C function names follow the pattern:
 * {@code Java_com_turboquant_backend_cuda_CudaNativeBridge_<method>}.
 */
final class CudaNativeBridge {

    static {
        try {
            System.loadLibrary("tq_cuda");
        } catch (UnsatisfiedLinkError e) {
            throw new UnsatisfiedLinkError(
                "Failed to load native library 'tq_cuda'. " +
                "Build native/cuda/ with CMake and add the output directory " +
                "to -Djava.library.path. Original error: " + e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // High-level runtime lifecycle
    // -------------------------------------------------------------------------

    /**
     * Create a runtime context for the device at {@code deviceIndex}.
     *
     * @return opaque runtime handle, or 0 on failure.
     */
    static native long tqRuntimeCreate(int deviceIndex);

    /** Destroy a runtime previously created by {@link #tqRuntimeCreate}. */
    static native void tqRuntimeDestroy(long rtHandle);

    /**
     * Return a human-readable description of the runtime.
     * Example: {@code "CUDA mock 12.4 / device[0]: Mock CUDA Device 0"}.
     */
    static native String tqRuntimeDescribe(long rtHandle);

    // -------------------------------------------------------------------------
    // High-level session lifecycle
    // -------------------------------------------------------------------------

    /**
     * Create a new compute session attached to {@code rtHandle}.
     *
     * @return opaque session handle, or 0 on failure.
     */
    static native long tqSessionCreate(long rtHandle);

    /** Destroy a session previously created by {@link #tqSessionCreate}. */
    static native void tqSessionDestroy(long sessionHandle);

    /** Block until all work on the session's internal stream has completed. */
    static native void tqSessionSynchronize(long sessionHandle);

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    /**
     * Run a single forward pass through the model stub.
     *
     * <p>Allocates a native {@code tq_infer_result_t} on the heap and returns
     * a pointer to it as a {@code long} result handle.  The caller <em>must</em>
     * call {@link #tqResultFree} on the handle when done.</p>
     *
     * @param sessionHandle opaque session handle
     * @param inputIds      input token ID array (prompt)
     * @param maxNewTokens  maximum number of tokens to generate
     * @return result handle (non-zero on success, 0 on error)
     */
    static native long tqSessionInfer(
            long sessionHandle, int[] inputIds, int maxNewTokens);

    // -------------------------------------------------------------------------
    // Result accessors  (all accept the handle returned by tqSessionInfer)
    // -------------------------------------------------------------------------

    /** Return the generated token IDs as a Java {@code int[]}. */
    static native int[] tqResultGetGeneratedIds(long resultHandle);

    /** Return the last-step logit vector as a Java {@code float[]}. */
    static native float[] tqResultGetLastLogits(long resultHandle);

    /** Return the wall-clock inference duration in nanoseconds. */
    static native long tqResultGetInferenceNanos(long resultHandle);

    /** Return the number of prompt tokens that were processed. */
    static native int tqResultGetPromptCount(long resultHandle);

    /** Return the number of tokens currently resident in the KV cache. */
    static native int tqResultGetKvCachedTokens(long resultHandle);

    /** Return the total KV cache capacity in tokens. */
    static native int tqResultGetKvCapacityTokens(long resultHandle);

    /** Return the KV cache memory currently in use, in bytes. */
    static native long tqResultGetKvSizeBytes(long resultHandle);

    /** Return the simulated KV cache hit-rate in {@code [0, 1]}. */
    static native double tqResultGetKvHitRate(long resultHandle);

    /**
     * Free the native result handle.
     * Releases the heap-allocated arrays and the struct itself.
     * Safe to call with 0.
     */
    static native void tqResultFree(long resultHandle);

    // -------------------------------------------------------------------------
    // Low-level device memory management
    // -------------------------------------------------------------------------

    /**
     * Allocate {@code bytes} of device memory.
     *
     * @return device pointer, or 0 on allocation failure.
     */
    static native long tqMalloc(long bytes);

    /** Free device memory previously allocated by {@link #tqMalloc}. */
    static native void tqFree(long devicePtr);

    /**
     * Copy {@code data} from the Java heap to device memory asynchronously.
     *
     * @param devicePtr destination device pointer
     * @param data      source host array
     * @param stream    stream handle (0 = default stream in mock)
     */
    static native void tqUploadFloat(long devicePtr, float[] data, long stream);

    /**
     * Copy device memory back to a Java float array (synchronous).
     *
     * @param dst       destination array (pre-allocated, length >= numel)
     * @param devicePtr source device pointer
     * @param numel     number of float32 elements to copy
     */
    static native void tqDownloadFloat(float[] dst, long devicePtr, long numel);

    // -------------------------------------------------------------------------
    // Low-level stream management
    // -------------------------------------------------------------------------

    /** @return opaque stream handle (non-zero on success). */
    static native long tqStreamCreate();

    /** Destroy a stream previously created by {@link #tqStreamCreate}. */
    static native void tqStreamDestroy(long streamHandle);

    /** Block until all operations on the stream have completed. */
    static native void tqStreamSynchronize(long streamHandle);

    // -------------------------------------------------------------------------
    // Low-level quantisation kernels
    // -------------------------------------------------------------------------

    /**
     * INT8 symmetric per-tensor quantisation.
     *
     * @param dst      INT8 output device buffer ({@code numel} bytes)
     * @param src      FLOAT32 input device buffer ({@code numel * 4} bytes)
     * @param numel    number of elements
     * @param scaleOut single-element FLOAT32 device pointer for the scale
     * @param stream   stream handle
     */
    static native void tqQuantiseInt8(
            long dst, long src, long numel, long scaleOut, long stream);

    /**
     * Dequantise INT8 → FLOAT32.
     *
     * @param dst    FLOAT32 output device buffer
     * @param src    INT8 input device buffer
     * @param numel  number of elements
     * @param scale  scale factor (host scalar)
     * @param stream stream handle
     */
    static native void tqDequantiseInt8(
            long dst, long src, long numel, float scale, long stream);

    // Private constructor — static-methods-only utility class.
    private CudaNativeBridge() {}
}
