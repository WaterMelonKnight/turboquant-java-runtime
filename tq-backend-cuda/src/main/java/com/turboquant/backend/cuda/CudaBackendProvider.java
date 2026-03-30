package com.turboquant.backend.cuda;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.spi.BackendProvider;

/**
 * {@link BackendProvider} for the CUDA backend.
 * Registered via {@code META-INF/services/com.turboquant.runtime.spi.BackendProvider}.
 */
public final class CudaBackendProvider implements BackendProvider {

    @Override
    public String backendName() {
        return "cuda";
    }

    @Override
    public int priority() {
        return 80; // preferred over CPU stub, equals HIP
    }

    @Override
    public boolean isAvailable() {
        // Delegate to the backend itself, which checks for the native library.
        return new CudaBackend().isAvailable();
    }

    @Override
    public Backend create() {
        return new CudaBackend();
    }
}
