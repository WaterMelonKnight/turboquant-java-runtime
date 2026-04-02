package com.turboquant.backend.llamacpp;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.SessionConfig;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Set;

/**
 * llama.cpp {@link Backend} implementation.
 *
 * <p>Loads a local GGUF model file via the {@code de.kherud:llama} Java binding.
 * Must be initialised with a {@link SessionConfig} that has {@code modelPath} set.</p>
 *
 * <p>Tensor-level operations ({@code allocate}, {@code upload}, quantisation)
 * are not applicable and will throw {@link UnsupportedOperationException} if called.</p>
 */
public final class LlamaCppBackend implements Backend {

    private static final Logger log = LoggerFactory.getLogger(LlamaCppBackend.class);

    private LlamaModel model;
    private SessionConfig sessionConfig;

    @Override
    public String name() {
        return "llama.cpp";
    }

    @Override
    public String version() {
        return "de.kherud:llama (llama.cpp)";
    }

    @Override
    public Set<BackendCapability> capabilities() {
        return Set.of(BackendCapability.TEXT_GENERATION);
    }

    @Override
    public boolean isAvailable() {
        return LlamaCppBackendProvider.NATIVE_AVAILABLE;
    }

    @Override
    public void init(BackendConfig config) {
        throw new BackendException(
                "LlamaCppBackend requires a SessionConfig with modelPath set. "
                + "Use DefaultTurboQuantRuntime.withBackend(\"llama.cpp\", sessionConfig) "
                + "where sessionConfig includes modelPath.");
    }

    @Override
    public void init(SessionConfig config) {
        if (config.modelPath() == null || config.modelPath().isBlank()) {
            throw new BackendException(
                    "LlamaCppBackend: modelPath must not be null or blank. "
                    + "Pass a path to a local GGUF file via SessionConfig.modelPath().");
        }

        File modelFile = new File(config.modelPath());
        if (!modelFile.exists()) {
            throw new BackendException(
                    "LlamaCppBackend: model file not found: " + config.modelPath());
        }
        if (!modelFile.isFile()) {
            throw new BackendException(
                    "LlamaCppBackend: model path is not a file: " + config.modelPath());
        }

        this.sessionConfig = config;

        int nGpuLayers = config.deviceId() > 0 ? 99 : 0;
        ModelParameters params = new ModelParameters()
                .setModelFilePath(config.modelPath())
                .setNGpuLayers(nGpuLayers)
                .setNCtx(config.maxContextTokens());

        log.info("Loading llama.cpp model: {}", config.modelPath());
        log.info("  context={} tokens, GPU layers={}", config.maxContextTokens(), nGpuLayers);
        try {
            this.model = new LlamaModel(params);
        } catch (Exception e) {
            throw new BackendException(
                    "LlamaCppBackend: failed to load model from " + config.modelPath()
                    + " — " + e.getMessage(), e);
        }
        log.info("llama.cpp model loaded successfully");
    }

    @Override
    public ComputeSession newSession() {
        if (model == null) {
            throw new BackendException("LlamaCppBackend not initialised — call init(SessionConfig) first.");
        }
        return new LlamaCppSession(model, sessionConfig);
    }

    @Override
    public void close() {
        if (model != null) {
            try {
                model.close();
            } catch (Exception e) {
                log.warn("Exception closing LlamaModel: {}", e.getMessage());
            } finally {
                model = null;
            }
        }
    }
}
