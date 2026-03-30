package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.ComputeSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

/**
 * Pure-Java CPU stub {@link Backend}.
 *
 * <p>Always available — no GPU or native library required.
 * Priority 0 ensures GPU backends are preferred when present.</p>
 */
public final class CpuBackend implements Backend {

    private static final Logger log = LoggerFactory.getLogger(CpuBackend.class);

    private BackendConfig config;

    @Override
    public String name() {
        return "cpu-stub";
    }

    @Override
    public String version() {
        return "0.1.0-SNAPSHOT (Java " + System.getProperty("java.version") + ")";
    }

    @Override
    public Set<BackendCapability> capabilities() {
        return Set.of(BackendCapability.CPU_FALLBACK);
    }

    @Override
    public boolean isAvailable() {
        return true;
    }

    @Override
    public void init(BackendConfig config) {
        this.config = config;
        log.info("CpuBackend initialised with config: {}", config);
    }

    @Override
    public ComputeSession newSession() {
        return new CpuComputeSession();
    }

    @Override
    public void close() {
        log.debug("CpuBackend closed");
    }

    BackendConfig config() {
        return config;
    }
}
