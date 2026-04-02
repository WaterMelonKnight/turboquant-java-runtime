package com.turboquant.runtime.api;

import java.util.Set;

/**
 * Top-level backend abstraction.
 *
 * <p>Implementations live in separate modules ({@code tq-backend-*}) and are
 * discovered at runtime via {@code ServiceLoader<BackendProvider>} in
 * {@code tq-runtime-spi} / {@code tq-runtime-core}.</p>
 *
 * <p>The Java API is backend-agnostic: this interface must never import or
 * reference any GPU-vendor class.</p>
 */
public interface Backend extends AutoCloseable {

    /** Stable identifier, e.g. {@code "cpu-stub"}, {@code "cuda"}, {@code "hip"}. */
    String name();

    /** Human-readable version string, e.g. {@code "CUDA 12.4 / driver 550.54"}. */
    String version();

    /** Capabilities advertised by this backend. */
    Set<BackendCapability> capabilities();

    /** {@code true} if this backend can be used in the current environment. */
    boolean isAvailable();

    /**
     * Initialise the backend with the supplied config.
     * Must be called once before {@link #newSession()}.
     */
    void init(BackendConfig config);

    /**
     * Initialise the backend with model-level session configuration.
     *
     * <p>Backends that support model loading (e.g. llama.cpp) should override
     * this method.  The default implementation converts {@code config} to a
     * {@link BackendConfig} and delegates to {@link #init(BackendConfig)}.</p>
     */
    default void init(SessionConfig config) {
        init(config.toBackendConfig());
    }

    /** Open a new compute session on this backend. */
    ComputeSession newSession();

    @Override
    void close();
}
