package com.turboquant.backend.llamacpp;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.spi.BackendProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link BackendProvider} for the llama.cpp backend.
 *
 * <p>Registered via {@code META-INF/services/com.turboquant.runtime.spi.BackendProvider}.
 * Available whenever the {@code de.kherud:llama} JAR is on the classpath —
 * native binaries are bundled inside the JAR.</p>
 */
public final class LlamaCppBackendProvider implements BackendProvider {

    /**
     * {@code true} when the llama.cpp native library loaded successfully at
     * class-load time.  Checked once so the backend reports clean availability.
     */
    static final boolean NATIVE_AVAILABLE;

    static {
        boolean ok = false;
        try {
            // Trigger native loading by touching the LlamaModel class.
            Class.forName("de.kherud.llama.LlamaModel");
            ok = true;
        } catch (ClassNotFoundException e) {
            LoggerFactory.getLogger(LlamaCppBackendProvider.class)
                    .debug("llama.cpp backend unavailable: de.kherud:llama not on classpath");
        } catch (UnsatisfiedLinkError | ExceptionInInitializerError e) {
            LoggerFactory.getLogger(LlamaCppBackendProvider.class)
                    .debug("llama.cpp backend unavailable: native binding failed to load — {}",
                            e.getMessage());
        }
        NATIVE_AVAILABLE = ok;
    }

    @Override
    public String backendName() {
        return "llama.cpp";
    }

    @Override
    public int priority() {
        // Negative so llama.cpp never wins autoSelect(BackendConfig).
        // It always requires an explicit model path and must be selected via
        // withBackend("llama.cpp", sessionConfig) or --backend llama.cpp.
        return -1;
    }

    @Override
    public boolean isAvailable() {
        return NATIVE_AVAILABLE;
    }

    @Override
    public Backend create() {
        return new LlamaCppBackend();
    }
}
