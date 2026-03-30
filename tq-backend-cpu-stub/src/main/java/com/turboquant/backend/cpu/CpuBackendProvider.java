package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.spi.BackendProvider;

/**
 * {@link BackendProvider} for the CPU stub backend.
 * Registered via {@code META-INF/services/com.turboquant.runtime.spi.BackendProvider}.
 */
public final class CpuBackendProvider implements BackendProvider {

    @Override
    public String backendName() {
        return "cpu-stub";
    }

    @Override
    public int priority() {
        return 0; // lowest priority — GPU backends take precedence
    }

    @Override
    public boolean isAvailable() {
        return true; // always available
    }

    @Override
    public Backend create() {
        return new CpuBackend();
    }
}
