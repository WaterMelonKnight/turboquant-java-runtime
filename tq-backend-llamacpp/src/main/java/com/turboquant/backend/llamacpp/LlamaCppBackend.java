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
 * <h2>GPU offload</h2>
 * <p>GPU layer offload is requested via {@link SessionConfig#nGpuLayers()}.
 * Whether it actually activates depends on the native llama.cpp build:
 * a CPU-only build silently ignores the setting.  The backend infers the
 * effective offload status from llama.cpp log output captured at model-load
 * time and exposes it via {@link #gpuOffloadActive()}.</p>
 *
 * <p>Tensor-level operations ({@code allocate}, {@code upload}, quantisation)
 * are not applicable and will throw {@link UnsupportedOperationException} if called.</p>
 */
public final class LlamaCppBackend implements Backend {

    private static final Logger log = LoggerFactory.getLogger(LlamaCppBackend.class);

    private LlamaModel    model;
    private SessionConfig sessionConfig;
    /** Resolved number of GPU layers actually passed to llama.cpp (0 = CPU-only). */
    private int           resolvedGpuLayers;
    /**
     * Best-effort inference of whether GPU offload is active.
     * {@code null} = unknown (e.g. offload was requested but we cannot confirm it).
     * {@code false} = CPU-only (either not requested, or build does not support it).
     * {@code true}  = offload was requested AND the build appears to support it
     *                 (confirmed by absence of the "no GPU offload" warning in logs).
     */
    private Boolean       gpuOffloadActive;

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

        // Resolve nGpuLayers: -1 means "all layers" → pass 999 to llama.cpp.
        // 0 means CPU-only. Any positive value is passed as-is.
        int requestedLayers = config.nGpuLayers();
        this.resolvedGpuLayers = (requestedLayers < 0) ? 999 : requestedLayers;

        logGpuOffloadIntent(requestedLayers, resolvedGpuLayers);

        ModelParameters params = new ModelParameters()
                .setModelFilePath(config.modelPath())
                .setNGpuLayers(resolvedGpuLayers)
                .setNCtx(config.maxContextTokens());

        log.info("Loading llama.cpp model: {}", config.modelPath());
        log.info("  context={} tokens, nGpuLayers={} (requested={})",
                config.maxContextTokens(), resolvedGpuLayers, requestedLayers);

        // Capture llama.cpp stderr to detect whether GPU offload is actually active.
        // The de.kherud:llama binding writes llama.cpp log output to stderr.
        // A CPU-only build emits: "llama_model_load: no GPU offload support"
        // We redirect stderr temporarily to detect this signal.
        GpuOffloadDetector detector = new GpuOffloadDetector();
        detector.install();
        try {
            this.model = new LlamaModel(params);
        } catch (Exception e) {
            throw new BackendException(
                    "LlamaCppBackend: failed to load model from " + config.modelPath()
                    + " — " + e.getMessage(), e);
        } finally {
            detector.restore();
        }

        this.gpuOffloadActive = resolveGpuOffloadActive(resolvedGpuLayers, detector);
        logGpuOffloadResult(requestedLayers, resolvedGpuLayers, gpuOffloadActive);
    }

    /**
     * Returns the best-effort GPU offload status after model loading.
     * {@code null} if the backend has not been initialised yet.
     */
    public Boolean gpuOffloadActive() {
        return gpuOffloadActive;
    }

    /** The number of GPU layers that was actually passed to llama.cpp (0 = CPU-only). */
    public int resolvedGpuLayers() {
        return resolvedGpuLayers;
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

    // -------------------------------------------------------------------------
    // GPU offload helpers
    // -------------------------------------------------------------------------

    private static void logGpuOffloadIntent(int requested, int resolved) {
        if (requested == 0) {
            log.info("GPU offload: not requested (nGpuLayers=0) — running CPU-only");
        } else if (requested < 0) {
            log.info("GPU offload: requested all layers (nGpuLayers=-1 → passing {} to llama.cpp)", resolved);
        } else {
            log.info("GPU offload: requested {} layer(s) — actual activation depends on native build", requested);
        }
    }

    private static Boolean resolveGpuOffloadActive(int resolvedLayers, GpuOffloadDetector detector) {
        if (resolvedLayers == 0) {
            return false;  // not requested — definitively CPU-only
        }
        if (detector.detectedCpuOnlyWarning()) {
            // llama.cpp emitted "no GPU offload support" — build is CPU-only
            return false;
        }
        // Offload was requested and no CPU-only warning was detected.
        // We cannot confirm GPU is actually being used without deeper instrumentation,
        // so we return null (unknown) rather than claiming true.
        return null;
    }

    private static void logGpuOffloadResult(int requested, int resolved, Boolean active) {
        if (requested == 0) {
            log.info("llama.cpp model loaded — CPU-only (GPU offload not requested)");
            return;
        }
        if (Boolean.FALSE.equals(active)) {
            log.warn("llama.cpp model loaded — GPU offload was requested ({} layers) but the native "
                    + "build does not support it. Running CPU-only. "
                    + "To enable GPU offload, use a GPU-enabled llama.cpp build.", resolved);
        } else {
            // null = unknown
            log.info("llama.cpp model loaded — GPU offload requested ({} layers); "
                    + "no CPU-only warning detected. "
                    + "Actual GPU utilisation depends on hardware and driver availability.", resolved);
        }
    }
}
