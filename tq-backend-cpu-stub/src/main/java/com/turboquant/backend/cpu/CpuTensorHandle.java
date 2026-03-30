package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.TensorHandle;

import java.util.Arrays;

/**
 * Heap-allocated float array backed tensor for the CPU stub backend.
 * All data lives in a {@code float[]} regardless of the declared dtype.
 */
final class CpuTensorHandle implements TensorHandle {

    private final long[] shape;
    private final DType dtype;
    /** Backing store; length == numel(). */
    private final float[] data;
    private boolean closed = false;

    CpuTensorHandle(DType dtype, long[] shape) {
        this.dtype = dtype;
        this.shape = Arrays.copyOf(shape, shape.length);
        long n = 1;
        for (long d : shape) n *= d;
        this.data = new float[(int) n];
    }

    CpuTensorHandle(float[] data, long[] shape) {
        this.dtype = DType.FLOAT32;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.data = Arrays.copyOf(data, data.length);
    }

    float[] rawData() {
        return data;
    }

    @Override
    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public float[] toFloatArray() {
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "toFloatArray() requires FLOAT32 tensor, got " + dtype);
        }
        return Arrays.copyOf(data, data.length);
    }

    @Override
    public void close() {
        closed = true;
    }

    boolean isClosed() {
        return closed;
    }
}
