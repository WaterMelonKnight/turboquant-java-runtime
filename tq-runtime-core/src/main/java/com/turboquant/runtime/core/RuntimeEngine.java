package com.turboquant.runtime.core;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.ComputeSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for application code.
 *
 * <p>Wraps a {@link BackendRegistry}, selects a backend, initialises it,
 * and vends {@link ComputeSession} instances.  Implements {@link AutoCloseable}
 * so it can be used in try-with-resources or as a Spring bean.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (var engine = RuntimeEngine.autoSelect(BackendConfig.defaultConfig())) {
 *     try (var session = engine.newSession()) {
 *         var handle = session.upload(weights, 1, 4096);
 *         var q = session.quantiseInt8(handle);
 *         // ...
 *     }
 * }
 * }</pre>
 */
public final class RuntimeEngine implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(RuntimeEngine.class);

    private final Backend backend;

    private RuntimeEngine(Backend backend, BackendConfig config) {
        this.backend = backend;
        backend.init(config);
        log.info("RuntimeEngine started with backend '{}' ({})", backend.name(), backend.version());
    }

    /**
     * Select the highest-priority available backend and initialise it.
     */
    public static RuntimeEngine autoSelect(BackendConfig config) {
        var registry = new BackendRegistry();
        Backend backend = registry.createBest();
        return new RuntimeEngine(backend, config);
    }

    /**
     * Select a backend by name and initialise it.
     */
    public static RuntimeEngine withBackend(String backendName, BackendConfig config) {
        var registry = new BackendRegistry();
        Backend backend = registry.createByName(backendName);
        return new RuntimeEngine(backend, config);
    }

    /** The active backend. */
    public Backend backend() {
        return backend;
    }

    /** Open a new compute session on the active backend. */
    public ComputeSession newSession() {
        return backend.newSession();
    }

    @Override
    public void close() {
        backend.close();
        log.info("RuntimeEngine closed (backend: {})", backend.name());
    }
}
