package com.turboquant.runtime.core;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.SessionConfig;
import com.turboquant.runtime.api.TurboQuantRuntime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default {@link TurboQuantRuntime} implementation.
 *
 * <p>Owns a single long-lived {@link ComputeSession} so that KV cache state
 * accumulates across successive {@link #infer} calls — consistent with how
 * continuous batching engines operate in production.</p>
 *
 * <h2>Obtaining an instance</h2>
 * <pre>{@code
 * // auto-select highest-priority available backend
 * try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.autoSelect(BackendConfig.defaultConfig())) {
 *     InferenceResult r = rt.infer(InferenceRequest.syntheticPrompt(128, 32));
 * }
 *
 * // explicitly choose a backend
 * try (TurboQuantRuntime rt = DefaultTurboQuantRuntime.withBackend("cpu-stub", config)) {
 *     ...
 * }
 * }</pre>
 *
 * <p><b>Thread safety:</b> not thread-safe. Use one instance per thread or
 * add external synchronisation.</p>
 */
public final class DefaultTurboQuantRuntime implements TurboQuantRuntime {

    private static final Logger log = LoggerFactory.getLogger(DefaultTurboQuantRuntime.class);

    private final Backend        backend;
    private final ComputeSession session;
    private volatile boolean     closed = false;

    private DefaultTurboQuantRuntime(Backend backend, BackendConfig config) {
        backend.init(config);
        this.backend = backend;
        this.session = backend.newSession();
        log.info("DefaultTurboQuantRuntime ready — backend='{}', config={}",
                backend.name(), config);
    }

    private DefaultTurboQuantRuntime(Backend backend, SessionConfig config) {
        backend.init(config);
        this.backend = backend;
        this.session = backend.newSession();
        log.info("DefaultTurboQuantRuntime ready — backend='{}', sessionConfig={}",
                backend.name(), config);
    }

    /**
     * Auto-select the highest-priority available backend.
     *
     * @throws BackendException if no backend is available on the classpath
     */
    public static DefaultTurboQuantRuntime autoSelect(BackendConfig config) {
        Backend backend = new BackendRegistry().createBest();
        return new DefaultTurboQuantRuntime(backend, config);
    }

    /**
     * Auto-select the highest-priority available backend, initialised with
     * model-level {@link SessionConfig}.
     *
     * @throws BackendException if no backend is available on the classpath
     */
    public static DefaultTurboQuantRuntime autoSelect(SessionConfig config) {
        Backend backend = new BackendRegistry().createBest();
        return new DefaultTurboQuantRuntime(backend, config);
    }

    /**
     * Use the named backend.
     *
     * @throws BackendException if the backend is not registered or not available
     */
    public static DefaultTurboQuantRuntime withBackend(String backendName, BackendConfig config) {
        Backend backend = new BackendRegistry().createByName(backendName);
        return new DefaultTurboQuantRuntime(backend, config);
    }

    /**
     * Use the named backend, initialised with model-level {@link SessionConfig}.
     *
     * @throws BackendException if the backend is not registered or not available
     */
    public static DefaultTurboQuantRuntime withBackend(String backendName, SessionConfig config) {
        Backend backend = new BackendRegistry().createByName(backendName);
        return new DefaultTurboQuantRuntime(backend, config);
    }

    // -------------------------------------------------------------------------
    // TurboQuantRuntime
    // -------------------------------------------------------------------------

    @Override
    public Backend backend() {
        return backend;
    }

    @Override
    public InferenceResult infer(InferenceRequest request) {
        checkNotClosed();
        return session.infer(request);
    }

    @Override
    public KvCacheStats kvCacheStats() {
        checkNotClosed();
        return session.kvCacheStats();
    }

    // -------------------------------------------------------------------------
    // AutoCloseable
    // -------------------------------------------------------------------------

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            session.close();
        } catch (Exception e) {
            log.warn("Exception closing session for backend '{}': {}", backend.name(), e.getMessage());
        } finally {
            backend.close();
            log.info("DefaultTurboQuantRuntime closed (backend: {})", backend.name());
        }
    }

    // -------------------------------------------------------------------------

    private void checkNotClosed() {
        if (closed) {
            throw new BackendException("This TurboQuantRuntime has already been closed.");
        }
    }
}
