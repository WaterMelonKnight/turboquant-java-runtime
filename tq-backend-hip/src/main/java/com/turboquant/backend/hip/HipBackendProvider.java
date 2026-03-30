package com.turboquant.backend.hip;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.spi.BackendProvider;

/**
 * {@link BackendProvider} for the HIP/ROCm backend.
 *
 * <p>Disabled by default — see {@link HipBackend} for the enable mechanism.</p>
 */
public final class HipBackendProvider implements BackendProvider {

    @Override
    public String backendName() {
        return "hip";
    }

    @Override
    public int priority() {
        return 80; // same as CUDA; tie broken alphabetically in BackendRegistry
    }

    @Override
    public boolean isAvailable() {
        return new HipBackend().isAvailable();
    }

    @Override
    public Backend create() {
        return new HipBackend();
    }
}
