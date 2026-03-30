package com.turboquant.runtime.core;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.spi.BackendProvider;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link BackendRegistry}.
 *
 * <p>{@code tq-backend-cpu-stub} is on the test classpath so at least one
 * provider is always discovered.</p>
 */
class BackendRegistryTest {

    // -------------------------------------------------------------------------
    // Discovery
    // -------------------------------------------------------------------------

    @Test
    void atLeastOneProviderIsDiscovered() {
        BackendRegistry registry = new BackendRegistry();
        assertFalse(registry.allProviders().isEmpty(),
                "Expected at least one BackendProvider on the test classpath");
    }

    @Test
    void cpuStubProviderIsAlwaysPresent() {
        BackendRegistry registry = new BackendRegistry();
        boolean found = registry.allProviders().stream()
                .anyMatch(p -> "cpu-stub".equals(p.backendName()));
        assertTrue(found, "cpu-stub provider must be discoverable via ServiceLoader");
    }

    @Test
    void cpuStubIsAvailable() {
        BackendRegistry registry = new BackendRegistry();
        BackendProvider cpuProvider = registry.findByName("cpu-stub")
                .orElseThrow(() -> new AssertionError("cpu-stub not found"));
        assertTrue(cpuProvider.isAvailable(),
                "cpu-stub must always report isAvailable() == true");
    }

    // -------------------------------------------------------------------------
    // Priority ordering
    // -------------------------------------------------------------------------

    @Test
    void providersAreSortedByDescendingPriority() {
        BackendRegistry registry = new BackendRegistry();
        List<BackendProvider> providers = registry.allProviders();
        for (int i = 1; i < providers.size(); i++) {
            int prev = providers.get(i - 1).priority();
            int curr = providers.get(i).priority();
            assertTrue(prev >= curr,
                    "Providers must be sorted by descending priority; found " + prev + " before " + curr);
        }
    }

    @Test
    void bestProviderIsNonNullWhenCpuStubPresent() {
        BackendRegistry registry = new BackendRegistry();
        assertTrue(registry.bestProvider().isPresent(),
                "bestProvider() must be non-empty when cpu-stub is on the classpath");
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    @Test
    void createBestReturnsCpuStubOnNonGpuHost() {
        BackendRegistry registry = new BackendRegistry();
        Backend backend = registry.createBest();
        assertNotNull(backend);
        // On a host without GPU native libraries, cpu-stub (priority=0) is the
        // only available backend.  We cannot guarantee this in all CI environments
        // but at minimum the backend must be non-null and its name must be stable.
        assertFalse(backend.name().isBlank(), "backend.name() must not be blank");
    }

    @Test
    void createByName_cpuStub_returnsBackend() {
        BackendRegistry registry = new BackendRegistry();
        Backend backend = registry.createByName("cpu-stub");
        assertNotNull(backend);
        assertEquals("cpu-stub", backend.name());
    }

    @Test
    void createByName_unknownBackend_throwsBackendException() {
        BackendRegistry registry = new BackendRegistry();
        assertThrows(BackendException.class,
                () -> registry.createByName("does-not-exist"),
                "Unknown backend name must throw BackendException");
    }

    @Test
    void findByName_returnsEmptyForUnknownName() {
        BackendRegistry registry = new BackendRegistry();
        assertTrue(registry.findByName("no-such-backend").isEmpty());
    }

    @Test
    void findByName_isCaseInsensitive() {
        BackendRegistry registry = new BackendRegistry();
        assertTrue(registry.findByName("CPU-STUB").isPresent(),
                "findByName should be case-insensitive");
    }
}
