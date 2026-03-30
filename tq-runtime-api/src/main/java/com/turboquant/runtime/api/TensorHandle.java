package com.turboquant.runtime.api;

/**
 * Opaque handle to a tensor allocated on the backend device.
 *
 * <p>Callers must {@link #close()} handles to release device memory.
 * Implementations may be zero-copy views or device-side allocations.</p>
 *
 * <p>This interface intentionally carries no CUDA / HIP / native imports.</p>
 */
public interface TensorHandle extends AutoCloseable {

    /** Number of elements in each dimension. */
    long[] shape();

    /** Element data type. */
    DType dtype();

    /**
     * Total number of elements ({@code product of shape}).
     */
    default long numel() {
        long n = 1;
        for (long d : shape()) {
            n *= d;
        }
        return n;
    }

    /**
     * Copy tensor data to a newly allocated Java float array.
     * Only valid for {@link DType#FLOAT32} tensors.
     *
     * @throws UnsupportedOperationException if dtype is not FLOAT32
     */
    float[] toFloatArray();

    @Override
    void close();
}
