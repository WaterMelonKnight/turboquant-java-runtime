/**
 * tq_native_mock.c — TurboQuant mock C ABI implementation
 *
 * Implements every function declared in tq_native_api.h using purely CPU
 * memory and a deterministic LCG token generator.  No GPU, CUDA headers,
 * or JNI dependency.  Designed to be compiled together with tq_jni_bridge.c
 * into libtq_cuda.so so that the Java layer can load and exercise the full
 * call path without a real GPU.
 *
 * KV-cache simulation mirrors CpuComputeSession exactly so benchmark numbers
 * are comparable between the Java-stub and C-stub paths.
 */

#include "tq_native_api.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* =========================================================================
 * Constants (must match CpuComputeSession.STUB_*)
 * ========================================================================= */

#define STUB_VOCAB_SIZE      32000
#define STUB_CACHE_CAPACITY  2048
#define BYTES_PER_TOKEN      256L

/* =========================================================================
 * Implementation structs
 * ========================================================================= */

typedef struct {
    int32_t device_index;
} tq_runtime_impl_t;

typedef struct {
    tq_runtime_impl_t* runtime;
    int32_t  cached_tokens;
    int64_t  cache_size_bytes;
    int32_t  total_inferences;
} tq_session_impl_t;

/* =========================================================================
 * Utility helpers
 * ========================================================================= */

const char* tq_status_string(tq_status_t s) {
    switch (s) {
        case TQ_OK:                   return "TQ_OK";
        case TQ_ERR_NOT_INITIALISED:  return "TQ_ERR_NOT_INITIALISED";
        case TQ_ERR_INVALID_DEVICE:   return "TQ_ERR_INVALID_DEVICE";
        case TQ_ERR_OUT_OF_MEMORY:    return "TQ_ERR_OUT_OF_MEMORY";
        case TQ_ERR_LAUNCH_FAILED:    return "TQ_ERR_LAUNCH_FAILED";
        case TQ_ERR_INVALID_ARGUMENT: return "TQ_ERR_INVALID_ARGUMENT";
        case TQ_ERR_NOT_SUPPORTED:    return "TQ_ERR_NOT_SUPPORTED";
        case TQ_ERR_INTERNAL:         return "TQ_ERR_INTERNAL";
        default:                      return "TQ_ERR_UNKNOWN";
    }
}

size_t tq_dtype_sizeof(tq_dtype_t dtype) {
    switch (dtype) {
        case TQ_DTYPE_FLOAT32:  return 4;
        case TQ_DTYPE_BFLOAT16: return 2;
        case TQ_DTYPE_INT8:     return 1;
        case TQ_DTYPE_UINT4:    return 1; /* two packed elements per byte */
        default:                return 0;
    }
}

/* =========================================================================
 * Low-level device lifecycle (global, for backward-compat with tq_init path)
 * ========================================================================= */

static volatile int g_initialised = 0;

tq_status_t tq_init(int32_t device_index) {
    (void)device_index;
    g_initialised = 1;
    return TQ_OK;
}

void tq_destroy(void) {
    g_initialised = 0;
}

tq_status_t tq_device_name(int32_t device_index, char* buf, size_t buf_len) {
    if (!buf || buf_len == 0) return TQ_ERR_INVALID_ARGUMENT;
    snprintf(buf, buf_len, "Mock CUDA Device %d", device_index);
    return TQ_OK;
}

int32_t tq_runtime_version(void) {
    return 12040; /* CUDA 12.4 mock */
}

/* =========================================================================
 * Low-level stream management (all no-ops on CPU mock)
 * ========================================================================= */

tq_status_t tq_stream_create(tq_stream_t* stream_out) {
    if (!stream_out) return TQ_ERR_INVALID_ARGUMENT;
    *stream_out = 1; /* non-zero dummy handle */
    return TQ_OK;
}

tq_status_t tq_stream_destroy(tq_stream_t stream) {
    (void)stream;
    return TQ_OK;
}

tq_status_t tq_stream_synchronize(tq_stream_t stream) {
    (void)stream;
    return TQ_OK;
}

/* =========================================================================
 * Low-level device memory management
 *
 * In the mock, "device" memory is ordinary heap memory, making all
 * host↔device transfers simple memcpy calls.
 * ========================================================================= */

tq_status_t tq_malloc(tq_device_ptr_t* ptr_out, size_t bytes) {
    if (!ptr_out) return TQ_ERR_INVALID_ARGUMENT;
    void* p = calloc(1, bytes > 0 ? bytes : 1);
    if (!p) return TQ_ERR_OUT_OF_MEMORY;
    *ptr_out = (tq_device_ptr_t)(uintptr_t)p;
    return TQ_OK;
}

tq_status_t tq_free(tq_device_ptr_t ptr) {
    free((void*)(uintptr_t)ptr);
    return TQ_OK;
}

tq_status_t tq_upload_float32(
        tq_device_ptr_t dst,
        const float*    src,
        size_t          num_floats,
        tq_stream_t     stream)
{
    (void)stream;
    if (!dst || !src) return TQ_ERR_INVALID_ARGUMENT;
    memcpy((void*)(uintptr_t)dst, src, num_floats * sizeof(float));
    return TQ_OK;
}

tq_status_t tq_download_float32(
        float*          dst,
        tq_device_ptr_t src,
        size_t          num_floats)
{
    if (!dst || !src) return TQ_ERR_INVALID_ARGUMENT;
    memcpy(dst, (const void*)(uintptr_t)src, num_floats * sizeof(float));
    return TQ_OK;
}

/* =========================================================================
 * Low-level quantisation kernels (CPU stub implementations)
 * ========================================================================= */

tq_status_t tq_quantise_int8(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        tq_device_ptr_t scale_out,
        tq_stream_t     stream)
{
    (void)stream;
    if (!dst || !src || numel == 0) return TQ_ERR_INVALID_ARGUMENT;

    const float* fsrc = (const float*)(uintptr_t)src;
    int8_t*      idst = (int8_t*)(uintptr_t)dst;

    /* Compute max absolute value for symmetric quantisation. */
    float max_abs = 1e-6f;
    for (size_t i = 0; i < numel; i++) {
        float v = fsrc[i] < 0.0f ? -fsrc[i] : fsrc[i];
        if (v > max_abs) max_abs = v;
    }
    float scale = max_abs / 127.0f;

    for (size_t i = 0; i < numel; i++) {
        float q = fsrc[i] / scale;
        if (q >  127.0f) q =  127.0f;
        if (q < -127.0f) q = -127.0f;
        idst[i] = (int8_t)(int32_t)(q >= 0.0f ? q + 0.5f : q - 0.5f);
    }

    if (scale_out) {
        float* sp = (float*)(uintptr_t)scale_out;
        *sp = scale;
    }
    return TQ_OK;
}

tq_status_t tq_dequantise_int8(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        float           scale,
        tq_stream_t     stream)
{
    (void)stream;
    if (!dst || !src || numel == 0) return TQ_ERR_INVALID_ARGUMENT;

    const int8_t* isrc = (const int8_t*)(uintptr_t)src;
    float*        fdst = (float*)(uintptr_t)dst;
    for (size_t i = 0; i < numel; i++) {
        fdst[i] = (float)isrc[i] * scale;
    }
    return TQ_OK;
}

tq_status_t tq_quantise_uint4(
        tq_device_ptr_t dst,
        tq_device_ptr_t src,
        size_t          numel,
        tq_stream_t     stream)
{
    (void)dst; (void)src; (void)numel; (void)stream;
    return TQ_ERR_NOT_SUPPORTED; /* UINT4 placeholder — not yet implemented */
}

/* =========================================================================
 * Backend capability query
 * ========================================================================= */

uint32_t tq_capabilities(void) {
    return (uint32_t)(TQ_CAP_INT8_MATMUL | TQ_CAP_MULTI_STREAM);
}

/* =========================================================================
 * High-level runtime lifecycle
 * ========================================================================= */

tq_status_t tq_runtime_create(int32_t device_index, tq_runtime_t* rt_out) {
    if (!rt_out) return TQ_ERR_INVALID_ARGUMENT;
    tq_runtime_impl_t* impl =
        (tq_runtime_impl_t*)calloc(1, sizeof(tq_runtime_impl_t));
    if (!impl) return TQ_ERR_OUT_OF_MEMORY;
    impl->device_index = device_index;
    *rt_out = (tq_runtime_t)(uintptr_t)impl;
    return TQ_OK;
}

void tq_runtime_destroy(tq_runtime_t rt) {
    free((void*)(uintptr_t)rt);
}

tq_status_t tq_runtime_describe(tq_runtime_t rt, char* buf, size_t buf_len) {
    if (!rt || !buf || buf_len == 0) return TQ_ERR_NOT_INITIALISED;
    tq_runtime_impl_t* impl = (tq_runtime_impl_t*)(uintptr_t)rt;
    int ver = tq_runtime_version();
    snprintf(buf, buf_len,
             "CUDA mock %d.%d / device[%d]: Mock CUDA Device %d",
             ver / 1000, (ver % 1000) / 10,
             impl->device_index, impl->device_index);
    return TQ_OK;
}

/* =========================================================================
 * High-level session lifecycle
 * ========================================================================= */

tq_status_t tq_session_create(tq_runtime_t rt, tq_session_t* session_out) {
    if (!rt || !session_out) return TQ_ERR_INVALID_ARGUMENT;
    tq_session_impl_t* s =
        (tq_session_impl_t*)calloc(1, sizeof(tq_session_impl_t));
    if (!s) return TQ_ERR_OUT_OF_MEMORY;
    s->runtime = (tq_runtime_impl_t*)(uintptr_t)rt;
    *session_out = (tq_session_t)(uintptr_t)s;
    return TQ_OK;
}

void tq_session_destroy(tq_session_t session) {
    free((void*)(uintptr_t)session);
}

tq_status_t tq_session_synchronize(tq_session_t session) {
    (void)session;
    return TQ_OK;
}

/* =========================================================================
 * High-level inference
 * ========================================================================= */

/**
 * LCG token generator — identical seed arithmetic to CpuComputeSession so
 * that Java unit tests can assert deterministic output across both paths.
 *
 * state_{n+1} = (state_n * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
 * token_id    = state % STUB_VOCAB_SIZE
 */
static int32_t lcg_next(int32_t state) {
    return (int32_t)(((int64_t)state * 1103515245LL + 12345LL) & 0x7fffffffLL);
}

tq_status_t tq_session_infer(
        tq_session_t        session,
        const int32_t*      input_token_ids,
        int32_t             input_count,
        int32_t             max_new_tokens,
        tq_infer_result_t*  result_out)
{
    if (!session || !input_token_ids || input_count <= 0 ||
        max_new_tokens <= 0 || !result_out)
        return TQ_ERR_INVALID_ARGUMENT;

    tq_session_impl_t* s = (tq_session_impl_t*)(uintptr_t)session;

    /* Allocate output buffers. */
    int32_t* generated = (int32_t*)malloc(
        (size_t)max_new_tokens * sizeof(int32_t));
    float*   logits    = (float*)calloc(
        (size_t)STUB_VOCAB_SIZE, sizeof(float));
    if (!generated || !logits) {
        free(generated);
        free(logits);
        return TQ_ERR_OUT_OF_MEMORY;
    }

    /* Time the (trivial) stub work. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* LCG seeded from last input token — matches Java CpuComputeSession. */
    int32_t state = input_token_ids[input_count - 1];
    for (int32_t i = 0; i < max_new_tokens; i++) {
        state      = lcg_next(state);
        generated[i] = state % STUB_VOCAB_SIZE;
    }

    /* Set a single non-zero logit for the last generated token. */
    if (max_new_tokens > 0) {
        logits[generated[max_new_tokens - 1]] = 1.0f;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    int64_t nanos =
        (int64_t)(t1.tv_sec  - t0.tv_sec)  * 1000000000LL +
        (int64_t)(t1.tv_nsec - t0.tv_nsec);
    if (nanos <= 0) nanos = 1; /* guarantee positive latency */

    /* Update simulated KV cache — cap at STUB_CACHE_CAPACITY. */
    int32_t new_cached = s->cached_tokens + input_count + max_new_tokens;
    if (new_cached > STUB_CACHE_CAPACITY) new_cached = STUB_CACHE_CAPACITY;
    s->cached_tokens    = new_cached;
    s->cache_size_bytes = (int64_t)s->cached_tokens * BYTES_PER_TOKEN;
    s->total_inferences++;

    /* Hit-rate ramps up toward 0.9 as inferences accumulate. */
    double hit_rate = 0.0;
    if (s->total_inferences > 1) {
        hit_rate = (s->total_inferences - 1.0) / (double)s->total_inferences;
        if (hit_rate > 0.9) hit_rate = 0.9;
    }

    /* Populate result struct. */
    result_out->generated_token_ids  = generated;
    result_out->generated_count      = max_new_tokens;
    result_out->prompt_count         = input_count;
    result_out->last_logits          = logits;
    result_out->vocab_size           = STUB_VOCAB_SIZE;
    result_out->inference_nanos      = nanos;
    result_out->kv_cached_tokens     = s->cached_tokens;
    result_out->kv_capacity_tokens   = STUB_CACHE_CAPACITY;
    result_out->kv_size_bytes        = s->cache_size_bytes;
    result_out->kv_hit_rate          = hit_rate;

    return TQ_OK;
}

void tq_infer_result_free(tq_infer_result_t* result) {
    if (!result) return;
    free(result->generated_token_ids);
    free(result->last_logits);
    result->generated_token_ids = NULL;
    result->last_logits         = NULL;
}

void tq_string_free(char* str) {
    free(str);
}
