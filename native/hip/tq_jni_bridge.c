/**
 * tq_jni_bridge.c — JNI glue between Java HipNativeBridge and the C ABI
 *
 * Structurally identical to native/cuda/tq_jni_bridge.c.
 * The only difference is the TQ_JNI macro, which expands function names to
 * the HIP package prefix:
 *
 *   Java_com_turboquant_backend_hip_HipNativeBridge_<methodName>
 *
 * This is the sole file-level change required when porting the JNI bridge from
 * CUDA to HIP — a one-line diff.  All business logic lives in tq_native_mock.c
 * (placeholder) or the real HIP kernel implementation that replaces it.
 *
 * Result handle convention
 * ─────────────────────────
 * tqSessionInfer() mallocs a tq_infer_result_t, fills it via tq_session_infer,
 * and returns the pointer as a jlong.  Caller must call tqResultFree() when
 * done.  Individual tqResult* getters accept the same jlong handle.
 */

#include "tq_native_api.h"
#include <jni.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Convenience macro: expands to the mangled JNI function name.
 * Change this one line to retarget the bridge to a different class.
 * ========================================================================= */
#define TQ_JNI(name) \
    Java_com_turboquant_backend_hip_HipNativeBridge_##name

/* =========================================================================
 * High-level runtime / session / inference
 * ========================================================================= */

JNIEXPORT jlong JNICALL TQ_JNI(tqRuntimeCreate)
    (JNIEnv* env, jclass cls, jint deviceIndex)
{
    (void)env; (void)cls;
    tq_runtime_t rt = 0;
    tq_status_t rc = tq_runtime_create((int32_t)deviceIndex, &rt);
    if (rc != TQ_OK) return 0L;
    return (jlong)rt;
}

JNIEXPORT void JNICALL TQ_JNI(tqRuntimeDestroy)
    (JNIEnv* env, jclass cls, jlong rtHandle)
{
    (void)env; (void)cls;
    tq_runtime_destroy((tq_runtime_t)rtHandle);
}

JNIEXPORT jstring JNICALL TQ_JNI(tqRuntimeDescribe)
    (JNIEnv* env, jclass cls, jlong rtHandle)
{
    (void)cls;
    char buf[256];
    tq_status_t rc = tq_runtime_describe((tq_runtime_t)rtHandle, buf, sizeof(buf));
    if (rc != TQ_OK) return (*env)->NewStringUTF(env, "unknown");
    return (*env)->NewStringUTF(env, buf);
}

JNIEXPORT jlong JNICALL TQ_JNI(tqSessionCreate)
    (JNIEnv* env, jclass cls, jlong rtHandle)
{
    (void)env; (void)cls;
    tq_session_t session = 0;
    tq_status_t rc = tq_session_create((tq_runtime_t)rtHandle, &session);
    if (rc != TQ_OK) return 0L;
    return (jlong)session;
}

JNIEXPORT void JNICALL TQ_JNI(tqSessionDestroy)
    (JNIEnv* env, jclass cls, jlong sessionHandle)
{
    (void)env; (void)cls;
    tq_session_destroy((tq_session_t)sessionHandle);
}

JNIEXPORT void JNICALL TQ_JNI(tqSessionSynchronize)
    (JNIEnv* env, jclass cls, jlong sessionHandle)
{
    (void)env; (void)cls;
    tq_session_synchronize((tq_session_t)sessionHandle);
}

JNIEXPORT jlong JNICALL TQ_JNI(tqSessionInfer)
    (JNIEnv* env, jclass cls,
     jlong sessionHandle, jintArray inputIds, jint maxNewTokens)
{
    (void)cls;
    jsize input_count = (*env)->GetArrayLength(env, inputIds);
    jint* elems       = (*env)->GetIntArrayElements(env, inputIds, NULL);

    tq_infer_result_t* result =
        (tq_infer_result_t*)calloc(1, sizeof(tq_infer_result_t));
    if (!result) {
        (*env)->ReleaseIntArrayElements(env, inputIds, elems, JNI_ABORT);
        return 0L;
    }

    tq_status_t rc = tq_session_infer(
        (tq_session_t)sessionHandle,
        (const int32_t*)elems,
        (int32_t)input_count,
        (int32_t)maxNewTokens,
        result);

    (*env)->ReleaseIntArrayElements(env, inputIds, elems, JNI_ABORT);

    if (rc != TQ_OK) { free(result); return 0L; }
    return (jlong)(uintptr_t)result;
}

/* =========================================================================
 * Result accessors
 * ========================================================================= */

JNIEXPORT jintArray JNICALL TQ_JNI(tqResultGetGeneratedIds)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)cls;
    tq_infer_result_t* r = (tq_infer_result_t*)(uintptr_t)resultHandle;
    jintArray arr = (*env)->NewIntArray(env, (jsize)r->generated_count);
    (*env)->SetIntArrayRegion(env, arr, 0, (jsize)r->generated_count,
                              (const jint*)r->generated_token_ids);
    return arr;
}

JNIEXPORT jfloatArray JNICALL TQ_JNI(tqResultGetLastLogits)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)cls;
    tq_infer_result_t* r = (tq_infer_result_t*)(uintptr_t)resultHandle;
    jfloatArray arr = (*env)->NewFloatArray(env, (jsize)r->vocab_size);
    (*env)->SetFloatArrayRegion(env, arr, 0, (jsize)r->vocab_size,
                                (const jfloat*)r->last_logits);
    return arr;
}

JNIEXPORT jlong JNICALL TQ_JNI(tqResultGetInferenceNanos)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jlong)((tq_infer_result_t*)(uintptr_t)resultHandle)->inference_nanos;
}

JNIEXPORT jint JNICALL TQ_JNI(tqResultGetPromptCount)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jint)((tq_infer_result_t*)(uintptr_t)resultHandle)->prompt_count;
}

JNIEXPORT jint JNICALL TQ_JNI(tqResultGetKvCachedTokens)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jint)((tq_infer_result_t*)(uintptr_t)resultHandle)->kv_cached_tokens;
}

JNIEXPORT jint JNICALL TQ_JNI(tqResultGetKvCapacityTokens)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jint)((tq_infer_result_t*)(uintptr_t)resultHandle)->kv_capacity_tokens;
}

JNIEXPORT jlong JNICALL TQ_JNI(tqResultGetKvSizeBytes)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jlong)((tq_infer_result_t*)(uintptr_t)resultHandle)->kv_size_bytes;
}

JNIEXPORT jdouble JNICALL TQ_JNI(tqResultGetKvHitRate)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    return (jdouble)((tq_infer_result_t*)(uintptr_t)resultHandle)->kv_hit_rate;
}

JNIEXPORT void JNICALL TQ_JNI(tqResultFree)
    (JNIEnv* env, jclass cls, jlong resultHandle)
{
    (void)env; (void)cls;
    tq_infer_result_t* r = (tq_infer_result_t*)(uintptr_t)resultHandle;
    if (!r) return;
    tq_infer_result_free(r);
    free(r);
}

/* =========================================================================
 * Low-level device memory management
 * ========================================================================= */

JNIEXPORT jlong JNICALL TQ_JNI(tqMalloc)
    (JNIEnv* env, jclass cls, jlong bytes)
{
    (void)env; (void)cls;
    tq_device_ptr_t ptr = 0;
    tq_status_t rc = tq_malloc(&ptr, (size_t)bytes);
    if (rc != TQ_OK) return 0L;
    return (jlong)ptr;
}

JNIEXPORT void JNICALL TQ_JNI(tqFree)
    (JNIEnv* env, jclass cls, jlong devicePtr)
{
    (void)env; (void)cls;
    tq_free((tq_device_ptr_t)devicePtr);
}

JNIEXPORT void JNICALL TQ_JNI(tqUploadFloat)
    (JNIEnv* env, jclass cls, jlong devicePtr, jfloatArray data, jlong stream)
{
    (void)cls;
    jsize   len   = (*env)->GetArrayLength(env, data);
    jfloat* elems = (*env)->GetFloatArrayElements(env, data, NULL);
    tq_upload_float32((tq_device_ptr_t)devicePtr,
                      (const float*)elems, (size_t)len,
                      (tq_stream_t)stream);
    (*env)->ReleaseFloatArrayElements(env, data, elems, JNI_ABORT);
}

JNIEXPORT void JNICALL TQ_JNI(tqDownloadFloat)
    (JNIEnv* env, jclass cls, jfloatArray dst, jlong devicePtr, jlong numel)
{
    (void)cls;
    jfloat* elems = (*env)->GetFloatArrayElements(env, dst, NULL);
    tq_download_float32((float*)elems,
                        (tq_device_ptr_t)devicePtr,
                        (size_t)numel);
    (*env)->ReleaseFloatArrayElements(env, dst, elems, 0);
}

/* =========================================================================
 * Low-level stream management
 * ========================================================================= */

JNIEXPORT jlong JNICALL TQ_JNI(tqStreamCreate)
    (JNIEnv* env, jclass cls)
{
    (void)env; (void)cls;
    tq_stream_t stream = 0;
    tq_stream_create(&stream);
    return (jlong)stream;
}

JNIEXPORT void JNICALL TQ_JNI(tqStreamDestroy)
    (JNIEnv* env, jclass cls, jlong streamHandle)
{
    (void)env; (void)cls;
    tq_stream_destroy((tq_stream_t)streamHandle);
}

JNIEXPORT void JNICALL TQ_JNI(tqStreamSynchronize)
    (JNIEnv* env, jclass cls, jlong streamHandle)
{
    (void)env; (void)cls;
    tq_stream_synchronize((tq_stream_t)streamHandle);
}

/* =========================================================================
 * Low-level quantisation kernels
 * ========================================================================= */

JNIEXPORT void JNICALL TQ_JNI(tqQuantiseInt8)
    (JNIEnv* env, jclass cls,
     jlong dst, jlong src, jlong numel, jlong scaleOut, jlong stream)
{
    (void)env; (void)cls;
    tq_quantise_int8((tq_device_ptr_t)dst, (tq_device_ptr_t)src,
                     (size_t)numel, (tq_device_ptr_t)scaleOut,
                     (tq_stream_t)stream);
}

JNIEXPORT void JNICALL TQ_JNI(tqDequantiseInt8)
    (JNIEnv* env, jclass cls,
     jlong dst, jlong src, jlong numel, jfloat scale, jlong stream)
{
    (void)env; (void)cls;
    tq_dequantise_int8((tq_device_ptr_t)dst, (tq_device_ptr_t)src,
                       (size_t)numel, (float)scale, (tq_stream_t)stream);
}
