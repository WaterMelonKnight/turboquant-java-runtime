package com.turboquant.runtime.api;

/**
 * Top-level runtime façade: owns one backend and exposes high-level
 * inference operations.
 *
 * <p>Obtain an instance from {@code DefaultTurboQuantRuntime} in the
 * {@code tq-runtime-core} module:</p>
 * <pre>{@code
 * try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(BackendConfig.defaultConfig())) {
 *     InferenceResult r = rt.infer(InferenceRequest.syntheticPrompt(128, 32));
 *     System.out.println(r);
 * }
 * }</pre>
 *
 * <p><b>Thread safety:</b> implementations are not required to be thread-safe.
 * Use one runtime per thread or add external synchronisation.</p>
 */
public interface TurboQuantRuntime extends AutoCloseable {

    /** The active backend powering this runtime instance. */
    Backend backend();

    /**
     * Run a forward pass and return the generated tokens and metadata.
     *
     * @param request the prompt and generation parameters
     * @return a non-null result containing generated token IDs, logits, and timing
     * @throws BackendException if the backend encounters an unrecoverable error
     */
    InferenceResult infer(InferenceRequest request);

    /**
     * Snapshot of the KV cache state after the most recent inference call.
     * Returns {@link KvCacheStats#empty()} if no inference has been performed yet.
     */
    KvCacheStats kvCacheStats();

    /** Release the session and backend resources. Idempotent. */
    @Override
    void close();
}
