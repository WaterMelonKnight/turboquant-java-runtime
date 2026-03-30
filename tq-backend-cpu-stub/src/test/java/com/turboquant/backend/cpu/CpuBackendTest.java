package com.turboquant.backend.cpu;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CpuBackend}.
 */
class CpuBackendTest {

    private CpuBackend freshBackend() {
        return new CpuBackend();
    }

    @Test
    void nameIsCpuStub() {
        assertEquals("cpu-stub", freshBackend().name());
    }

    @Test
    void isAlwaysAvailable() {
        assertTrue(freshBackend().isAvailable(),
                "CpuBackend.isAvailable() must always return true");
    }

    @Test
    void hasCpuFallbackCapability() {
        Backend backend = freshBackend();
        assertTrue(backend.capabilities().contains(BackendCapability.CPU_FALLBACK),
                "cpu-stub must advertise CPU_FALLBACK");
    }

    @Test
    void doesNotAdvertiseGpuCapabilities() {
        Backend backend = freshBackend();
        assertFalse(backend.capabilities().contains(BackendCapability.INT8_MATMUL),
                "cpu-stub must NOT advertise INT8_MATMUL (stub only)");
        assertFalse(backend.capabilities().contains(BackendCapability.MULTI_STREAM),
                "cpu-stub must NOT advertise MULTI_STREAM");
    }

    @Test
    void initSucceedsWithDefaultConfig() {
        Backend backend = freshBackend();
        assertDoesNotThrow(() -> backend.init(BackendConfig.defaultConfig()),
                "init() with default config must not throw");
    }

    @Test
    void versionContainsJavaKeyword() {
        Backend backend = freshBackend();
        backend.init(BackendConfig.defaultConfig());
        assertTrue(backend.version().toLowerCase().contains("java"),
                "version() should mention Java: " + backend.version());
    }

    @Test
    void newSessionIsNonNull() {
        Backend backend = freshBackend();
        backend.init(BackendConfig.defaultConfig());
        try (var session = backend.newSession()) {
            assertNotNull(session);
        }
    }

    @Test
    void closeIsIdempotent() {
        Backend backend = freshBackend();
        backend.init(BackendConfig.defaultConfig());
        assertDoesNotThrow(() -> {
            backend.close();
            backend.close(); // second call must be harmless
        });
    }
}
