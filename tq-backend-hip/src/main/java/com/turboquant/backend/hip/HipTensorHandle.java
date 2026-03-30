package com.turboquant.backend.hip;

import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.TensorHandle;

import java.util.Arrays;

/**
 * Device-resident tensor for the HIP backend.
 * <p>Wraps a device pointer ({@code long}) obtained from
 * {@link HipNativeBridge#tqMalloc}.  The device pointer is freed on
 * {@link #close()}.</p>
 */
final class HipTensorHandle implements TensorHandle {

    private final long[] shape;
    private final DType dtype;
    private long devicePtr;
    private final long numel;

    HipTensorHandle(long devicePtr, DType dtype, long[] shape) {
        this.devicePtr = devicePtr;
        this.dtype = dtype;
        this.shape = Arrays.copyOf(shape, shape.length);
        long n = 1;
        for (long d : shape) n *= d;
        this.numel = n;
    }

    long devicePtr()        { return devicePtr; }
    @Override public long numel() { return numel; }

    @Override public long[] shape() { return Arrays.copyOf(shape, shape.length); }
    @Override public DType  dtype() { return dtype; }

    @Override
    public float[] toFloatArray() {
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException(
                    "toFloatArray() requires FLOAT32 tensor, got " + dtype);
        }
        float[] result = new float[(int) numel];
        HipNativeBridge.tqDownloadFloat(result, devicePtr, numel);
        return result;
    }

    @Override
    public void close() {
        if (devicePtr != 0) {
            HipNativeBridge.tqFree(devicePtr);
            devicePtr = 0;
        }
    }
}
