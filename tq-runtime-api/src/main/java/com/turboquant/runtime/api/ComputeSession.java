package com.turboquant.runtime.api;

/**
 * A scoped compute session on a single backend device.
 *
 * <p>Sessions are created via {@link Backend#newSession()} and are
 * {@link AutoCloseable} — resources (streams, command-queues) are released on close.</p>
 */
public interface ComputeSession extends AutoCloseable {

    /**
     * Allocate a tensor on this device, filled with zeros.
     *
     * @param dtype element data type
     * @param shape dimensions, product must be &gt; 0
     * @return a new handle; caller is responsible for closing it
     */
    TensorHandle allocate(DType dtype, long... shape);

    /**
     * Upload a Java float array as a FLOAT32 tensor.
     *
     * @param data   source data (copied, not retained)
     * @param shape  dimensions; {@code product(shape) == data.length}
     */
    TensorHandle upload(float[] data, long... shape);

    /**
     * Placeholder: quantise {@code input} weights to {@link DType#INT8}.
     * Real kernel not implemented yet — returns a zero-filled handle of the same shape.
     */
    TensorHandle quantiseInt8(TensorHandle input);

    /**
     * Placeholder: dequantise {@code input} back to FLOAT32.
     * Real kernel not implemented yet — returns a zero-filled handle of the same shape.
     */
    TensorHandle dequantise(TensorHandle input);

    /** Wait until all enqueued operations on this session have completed. */
    void synchronize();

    /**
     * Run a model forward pass and return generated tokens with metadata.
     *
     * <p>Backends that have not yet implemented real kernels must still return a
     * non-null {@link InferenceResult}; they may populate it with stub/zero values.</p>
     *
     * @param request the prompt and generation parameters
     * @return a non-null result
     * @throws BackendException if the backend encounters an unrecoverable error
     */
    InferenceResult infer(InferenceRequest request);

    /**
     * Snapshot of this session's KV cache after the most recent {@link #infer} call.
     * Returns {@link KvCacheStats#empty()} if no inference has been performed yet.
     */
    KvCacheStats kvCacheStats();

    @Override
    void close();
}
