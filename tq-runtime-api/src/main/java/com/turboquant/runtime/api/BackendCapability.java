package com.turboquant.runtime.api;

/**
 * Optional capabilities a backend may advertise.
 * Callers can query {@link Backend#capabilities()} to decide which path to take.
 */
public enum BackendCapability {
    /** Backend supports INT8 quantised matmul kernels. */
    INT8_MATMUL,
    /** Backend supports UINT4 weight-only quantisation kernels. */
    UINT4_WEIGHT_QUANT,
    /** Backend can execute multiple sessions concurrently. */
    MULTI_STREAM,
    /** Backend supports FP8 (E4M3 / E5M2) tensor cores. */
    FP8_TENSOR_CORE,
    /** Backend is a pure-CPU fallback (no GPU required). */
    CPU_FALLBACK,
    /** Backend supports text-in / text-out inference (e.g. llama.cpp). */
    TEXT_GENERATION
}
