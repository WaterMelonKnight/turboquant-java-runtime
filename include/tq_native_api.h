/**
 * tq_native_api.h — TurboQuant stable C ABI
 *
 * This header defines the C-linkage function signatures that are implemented
 * by every TurboQuant native backend (CUDA, HIP, Metal, …).  The Java layer
 * calls these via JNI; the implementations live in separate shared libraries
 * (libtq_cuda.so, libtq_hip.so, etc.).
 *
 * Design goals
 * ────────────
 * 1. Stable ABI — function signatures must never change; add new functions
 *    for new features rather than changing existing ones.
 * 2. Backend-agnostic from the Java side — Java only sees device pointers
 *    (opaque 64-bit integers) and status codes; no vendor headers leak up.
 * 3. Minimal surface — only the operations that the Java SPI actually invokes
 *    are declared here; internal helpers live in backend-private headers.
 * 4. ROCm portability — every function has a direct HIP equivalent obtained
 *    by mechanical s/cuda/hip/g and s/CUDA/HIP/g substitution.
 *
 * Versioning
 * ──────────
 * Increment TQ_NATIVE_API_VERSION for every ABI-breaking change.
 * Non-breaking additions (new functions) increment TQ_NATIVE_API_MINOR.
 *
 * Return codes
 * ────────────
 * All functions that can fail return tq_status_t.
 * TQ_OK == 0; negative values are errors (see tq_status_t below).
 *
 * Device pointer convention
 * ─────────────────────────
 * Device pointers are represented as uint64_t (64-bit unsigned integer).
 * Java maps these to long.  The null device pointer is 0.
 */

#ifndef TQ_NATIVE_API_H
#define TQ_NATIVE_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Version
 * ========================================================================= */

/** Increment on every ABI-breaking change. */
#define TQ_NATIVE_API_VERSION       1
/** Increment on backward-compatible additions. */
#define TQ_NATIVE_API_MINOR         1

/* =========================================================================
 * Status codes
 * ========================================================================= */

typedef int32_t tq_status_t;

#define TQ_OK                       ((tq_status_t)  0)
#define TQ_ERR_NOT_INITIALISED      ((tq_status_t) -1)
#define TQ_ERR_INVALID_DEVICE       ((tq_status_t) -2)
#define TQ_ERR_OUT_OF_MEMORY        ((tq_status_t) -3)
#define TQ_ERR_LAUNCH_FAILED        ((tq_status_t) -4)
#define TQ_ERR_INVALID_ARGUMENT     ((tq_status_t) -5)
#define TQ_ERR_NOT_SUPPORTED        ((tq_status_t) -6)
#define TQ_ERR_INTERNAL             ((tq_status_t) -99)

/** Return a human-readable string for a status code. Never returns NULL. */
const char* tq_status_string(tq_status_t status);

/* =========================================================================
 * Data types
 * ========================================================================= */

/**
 * Element data type.  Values are stable and match DType.id in the Java layer.
 */
typedef enum {
    TQ_DTYPE_FLOAT32  = 0,
    TQ_DTYPE_BFLOAT16 = 1,
    TQ_DTYPE_INT8     = 2,
    TQ_DTYPE_UINT4    = 3
} tq_dtype_t;

/** Return the size in bytes of one element of the given dtype. */
size_t tq_dtype_sizeof(tq_dtype_t dtype);

/* =========================================================================
 * Opaque handle types
 *
 * Represented as uint64_t in the ABI so they can be passed through JNI
 * as Java long without truncation on any supported platform.
 * ========================================================================= */

/** Device memory pointer (backend virtual address space). */
typedef uint64_t tq_device_ptr_t;

/** Opaque stream / command-queue handle. */
typedef uint64_t tq_stream_t;

/* =========================================================================
 * Device lifecycle
 * ========================================================================= */

/**
 * Initialise the backend for the device at zero-based index @p device_index.
 * Must be called before any other tq_* function.
 * Safe to call multiple times; subsequent calls are no-ops.
 *
 * @param device_index  Zero-based GPU device index.
 * @return TQ_OK on success.
 */
tq_status_t tq_init(int32_t device_index);

/**
 * Release all resources held by the backend.
 * After this call, @p tq_init must be called again before using the backend.
 */
void tq_destroy(void);

/**
 * Write at most @p buf_len bytes of the device name into @p buf.
 * Always NUL-terminates.  Example: "NVIDIA H100 SXM5 80GB".
 *
 * @return TQ_OK, or TQ_ERR_INVALID_DEVICE if device_index is out of range.
 */
tq_status_t tq_device_name(int32_t device_index, char* buf, size_t buf_len);

/**
 * Return the backend runtime version as an integer.
 * CUDA encoding: major*1000 + minor*10  (e.g. 12040 for CUDA 12.4).
 * HIP  encoding: major*10000 + minor*100 + patch (e.g. 60300 for ROCm 6.3).
 */
int32_t tq_runtime_version(void);

/* =========================================================================
 * Stream management
 * ========================================================================= */

/**
 * Create a new execution stream / command queue.
 *
 * @param[out] stream_out  Receives the new stream handle on success.
 * @return TQ_OK on success.
 */
tq_status_t tq_stream_create(tq_stream_t* stream_out);

/**
 * Destroy a stream previously created by @p tq_stream_create.
 * Blocks until all pending operations on the stream have completed.
 */
tq_status_t tq_stream_destroy(tq_stream_t stream);

/**
 * Block the calling CPU thread until all operations enqueued on @p stream
 * have completed.
 */
tq_status_t tq_stream_synchronize(tq_stream_t stream);

/* =========================================================================
 * Device memory management
 * ========================================================================= */

/**
 * Allocate @p bytes of device memory.
 *
 * @param[out] ptr_out  Receives the device pointer on success.
 * @param bytes         Number of bytes to allocate.  Must be > 0.
 * @return TQ_OK, or TQ_ERR_OUT_OF_MEMORY.
 */
tq_status_t tq_malloc(tq_device_ptr_t* ptr_out, size_t bytes);

/**
 * Free device memory previously allocated by @p tq_malloc.
 * Passing ptr == 0 is a no-op.
 */
tq_status_t tq_free(tq_device_ptr_t ptr);

/**
 * Asynchronously copy @p num_floats float32 values from a host array to
 * device memory on @p stream.
 *
 * @param dst        Destination device pointer (must be float32-aligned).
 * @param src        Source host pointer (read-only).
 * @param num_floats Number of float32 elements to copy.
 * @param stream     Target stream.
 */
tq_status_t tq_upload_float32(
        tq_device_ptr_t dst,
        const float*    src,
        size_t          num_floats,
        tq_stream_t     stream);

/**
 * Synchronously copy @p num_floats float32 values from device to host.
 *
 * @param dst        Destination host pointer (pre-allocated).
 * @param src        Source device pointer.
 * @param num_floats Number of float32 elements to copy.
 */
tq_status_t tq_download_float32(
        float*          dst,
        tq_device_ptr_t src,
        size_t          num_floats);

/* =========================================================================
 * Quantisation kernels
 *
 * NOTE: These are placeholder signatures.
 *       Real kernel implementations are not yet included in this skeleton.
 * ========================================================================= */

/**
 * INT8 symmetric per-tensor quantisation: dst = round(clamp(src / scale, -127, 127)).
 *
 * The scale factor is computed automatically as max(|src|) / 127.0 and written
 * to @p scale_out (a single float32 on device).
 *
 * @param dst        [out] INT8 device buffer, @p numel bytes.
 * @param src        [in]  FLOAT32 device buffer, @p numel * 4 bytes.
 * @param numel      Number of elements.
 * @param scale_out  [out] Single float32 device pointer for the computed scale.
 * @param stream     Execution stream.
 */
tq_status_t tq_quantise_int8(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        tq_device_ptr_t scale_out,
        tq_stream_t     stream);

/**
 * INT8 → FLOAT32 dequantisation: dst = float(src) * scale.
 *
 * @param dst    [out] FLOAT32 device buffer, @p numel * 4 bytes.
 * @param src    [in]  INT8 device buffer, @p numel bytes.
 * @param numel  Number of elements.
 * @param scale  Scale factor (host scalar).
 * @param stream Execution stream.
 */
tq_status_t tq_dequantise_int8(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        float           scale,
        tq_stream_t     stream);

/**
 * UINT4 weight-only quantisation (placeholder).
 * Packs two 4-bit values per byte in low-nibble-first order.
 *
 * @param dst    [out] UINT8 device buffer (ceil(numel/2) bytes, each byte holds 2 elements).
 * @param src    [in]  FLOAT32 device buffer, @p numel * 4 bytes.
 * @param numel  Number of elements (must be even).
 * @param stream Execution stream.
 */
tq_status_t tq_quantise_uint4(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        tq_stream_t     stream);

/* =========================================================================
 * Backend capability query
 * ========================================================================= */

/**
 * Capability flags returned by @p tq_capabilities.
 */
typedef enum {
    TQ_CAP_INT8_MATMUL        = (1 << 0),
    TQ_CAP_UINT4_WEIGHT_QUANT = (1 << 1),
    TQ_CAP_MULTI_STREAM       = (1 << 2),
    TQ_CAP_FP8_TENSOR_CORE    = (1 << 3),
    TQ_CAP_CPU_FALLBACK       = (1 << 4)
} tq_capability_t;

/**
 * Return a bitmask of @p tq_capability_t flags supported by this backend.
 * Must be callable before @p tq_init.
 */
uint32_t tq_capabilities(void);

/* =========================================================================
 * High-level runtime / session / inference API  (added in MINOR=1)
 *
 * This layer sits on top of the low-level device operations and provides
 * the opaque handles that the Java SPI uses directly.  The Java layer
 * creates one tq_runtime_t per backend instance and one tq_session_t per
 * compute session; it never calls tq_init / tq_malloc / etc. directly.
 *
 * Ownership rules
 * ───────────────
 *  • tq_runtime_create   → caller owns the runtime, must call tq_runtime_destroy
 *  • tq_session_create   → caller owns the session, must call tq_session_destroy
 *  • tq_session_infer    → on success, fills *result_out; caller must call
 *                          tq_infer_result_free on the embedded pointers.
 *                          (tq_infer_result_free does NOT free the struct itself
 *                           when the struct is stack-allocated; see docs below.)
 * ========================================================================= */

/** Opaque runtime handle (one per backend / device). */
typedef uint64_t tq_runtime_t;

/** Opaque session handle (one per logical compute session). */
typedef uint64_t tq_session_t;

/**
 * Result of a single inference call.
 *
 * The two heap-allocated fields (generated_token_ids, last_logits) are freed
 * by tq_infer_result_free().  The struct itself is owned by the caller.
 */
typedef struct {
    /** Heap-allocated array of generated token IDs, length = generated_count. */
    int32_t*  generated_token_ids;
    /** Number of tokens actually generated (≤ max_new_tokens). */
    int32_t   generated_count;
    /** Number of prompt tokens processed. */
    int32_t   prompt_count;
    /** Heap-allocated logit vector for the last generated token, length = vocab_size. */
    float*    last_logits;
    /** Vocabulary size (= length of last_logits). */
    int32_t   vocab_size;
    /** Wall-clock inference duration in nanoseconds. */
    int64_t   inference_nanos;
    /** Tokens currently resident in the KV cache. */
    int32_t   kv_cached_tokens;
    /** Total KV cache capacity in tokens. */
    int32_t   kv_capacity_tokens;
    /** KV cache memory currently in use, in bytes. */
    int64_t   kv_size_bytes;
    /** Simulated cache hit-rate in [0, 1]. */
    double    kv_hit_rate;
} tq_infer_result_t;

/* -------------------------------------------------------------------------
 * Runtime lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a runtime context for the device at @p device_index.
 * The runtime initialises the device (equivalent to tq_init) and stores
 * the resulting state in an opaque handle.
 *
 * @param[in]  device_index  Zero-based device index.
 * @param[out] rt_out        Receives the new runtime handle on success.
 * @return TQ_OK on success.
 */
tq_status_t tq_runtime_create(int32_t device_index, tq_runtime_t* rt_out);

/**
 * Destroy a runtime previously created by tq_runtime_create.
 * All sessions derived from this runtime must be destroyed first.
 */
void tq_runtime_destroy(tq_runtime_t rt);

/**
 * Write a human-readable description of the runtime into @p buf.
 * Example: "CUDA mock 12.4 / device[0]: Mock CUDA Device 0".
 * Always NUL-terminates.
 *
 * @return TQ_OK on success, TQ_ERR_NOT_INITIALISED if rt == 0.
 */
tq_status_t tq_runtime_describe(tq_runtime_t rt, char* buf, size_t buf_len);

/* -------------------------------------------------------------------------
 * Session lifecycle
 * ------------------------------------------------------------------------- */

/**
 * Create a new compute session attached to @p rt.
 * Each session owns an independent command stream / queue.
 *
 * @param[in]  rt           Parent runtime handle.
 * @param[out] session_out  Receives the new session handle on success.
 * @return TQ_OK on success.
 */
tq_status_t tq_session_create(tq_runtime_t rt, tq_session_t* session_out);

/**
 * Destroy a session.  Blocks until all pending work on the session stream
 * has completed, then releases all resources.
 */
void tq_session_destroy(tq_session_t session);

/**
 * Block the calling thread until all operations enqueued on the session's
 * internal stream have completed.
 */
tq_status_t tq_session_synchronize(tq_session_t session);

/* -------------------------------------------------------------------------
 * Inference
 * ------------------------------------------------------------------------- */

/**
 * Run a single forward pass through the model stub.
 *
 * @param[in]  session         Session handle.
 * @param[in]  input_token_ids Array of @p input_count token IDs (host pointer).
 * @param[in]  input_count     Number of input tokens (> 0).
 * @param[in]  max_new_tokens  Maximum number of tokens to generate (> 0).
 * @param[out] result_out      Filled on success.  The caller must pass the
 *                             struct to tq_infer_result_free() when done.
 * @return TQ_OK on success, TQ_ERR_INVALID_ARGUMENT if any parameter is
 *         out of range, TQ_ERR_OUT_OF_MEMORY if allocation fails.
 */
tq_status_t tq_session_infer(
        tq_session_t        session,
        const int32_t*      input_token_ids,
        int32_t             input_count,
        int32_t             max_new_tokens,
        tq_infer_result_t*  result_out);

/**
 * Free the heap-allocated fields inside @p result (generated_token_ids and
 * last_logits).  Does NOT free the struct itself.
 * Safe to call with result == NULL or on an already-freed result.
 */
void tq_infer_result_free(tq_infer_result_t* result);

/**
 * Free a heap-allocated C string returned by any tq_* function.
 * Passing NULL is a no-op.
 */
void tq_string_free(char* str);

#ifdef __cplusplus
}
#endif

#endif /* TQ_NATIVE_API_H */
