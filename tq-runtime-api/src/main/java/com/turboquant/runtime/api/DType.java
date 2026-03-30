package com.turboquant.runtime.api;

/**
 * Quantization data types supported by TurboQuant kernels.
 * Values mirror the constants defined in {@code tq_native_api.h}.
 */
public enum DType {
    /** 32-bit IEEE-754 float */
    FLOAT32(0),
    /** 16-bit brain float */
    BFLOAT16(1),
    /** 8-bit signed integer (used for weight quantisation) */
    INT8(2),
    /** 4-bit unsigned integer packed two-per-byte */
    UINT4(3);

    /** Wire id matching the C ABI {@code tq_dtype_t} enum value. */
    public final int id;

    DType(int id) {
        this.id = id;
    }
}
