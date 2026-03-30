package com.turboquant.runtime.spi;

import com.turboquant.runtime.api.Backend;

/**
 * SPI contract implemented by every backend module.
 *
 * <p>Backend JARs register their implementation by placing a UTF-8 text file
 * at {@code META-INF/services/com.turboquant.runtime.spi.BackendProvider}
 * containing the fully-qualified class name of their provider.</p>
 *
 * <p>Discovery is performed by {@code BackendRegistry} in {@code tq-runtime-core}
 * using {@link java.util.ServiceLoader}.</p>
 *
 * <h2>Implementing a new backend</h2>
 * <ol>
 *   <li>Implement this interface in your module.</li>
 *   <li>Create {@code Backend} implementation returned by {@link #create()}.</li>
 *   <li>Add the service descriptor file.</li>
 *   <li>That's it — no changes to core or other modules required.</li>
 * </ol>
 */
public interface BackendProvider {

    /**
     * Stable identifier that matches {@link Backend#name()} of the created backend.
     * Examples: {@code "cpu-stub"}, {@code "cuda"}, {@code "hip"}.
     */
    String backendName();

    /**
     * Priority used for automatic selection when multiple backends are available.
     * Higher value wins. Range: 0–100.
     * Convention: CPU stub = 0, CUDA = 80, HIP = 80.
     */
    int priority();

    /**
     * {@code true} if the backend can be used in the current JVM process.
     * Called before {@link #create()} — if {@code false}, the backend is skipped.
     */
    boolean isAvailable();

    /**
     * Instantiate and return the backend.
     * Must not perform device initialisation here; that happens in {@link Backend#init}.
     */
    Backend create();
}
