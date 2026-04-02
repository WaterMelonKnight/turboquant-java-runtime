package com.turboquant.backend.llamacpp;

import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import com.turboquant.runtime.api.DType;
import com.turboquant.runtime.api.InferenceRequest;
import com.turboquant.runtime.api.InferenceResult;
import com.turboquant.runtime.api.KvCacheStats;
import com.turboquant.runtime.api.SessionConfig;
import com.turboquant.runtime.api.TensorHandle;
import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.LlamaOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * llama.cpp {@link ComputeSession} implementation.
 *
 * <p>Delegates inference to a shared {@link LlamaModel}.  Each call to
 * {@link #infer} runs a full prompt + generation pass; KV-cache reuse across
 * calls depends on llama.cpp's internal state.</p>
 *
 * <p>Tensor operations are not supported; calling them throws
 * {@link UnsupportedOperationException}.</p>
 */
public final class LlamaCppSession implements ComputeSession {

    private static final Logger log = LoggerFactory.getLogger(LlamaCppSession.class);

    private final LlamaModel   model;
    private final SessionConfig config;

    LlamaCppSession(LlamaModel model, SessionConfig config) {
        this.model  = model;
        this.config = config;
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    @Override
    public InferenceResult infer(InferenceRequest request) {
        String prompt = resolvePrompt(request);
        int maxNewTokens = request.maxNewTokens();

        InferenceParameters inferParams = new InferenceParameters(prompt)
                .setNPredict(maxNewTokens)
                .setTemperature(request.temperature())
                .setTopP(request.topP());

        long startNanos = System.nanoTime();
        StringBuilder sb = new StringBuilder();
        int tokenCount = 0;

        try {
            for (LlamaOutput output : model.generate(inferParams)) {
                sb.append(output.text);
                tokenCount++;
            }
        } catch (Exception e) {
            throw new BackendException("llama.cpp inference failed: " + e.getMessage(), e);
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        String generatedText = sb.toString();

        log.debug("Generated {} tokens in {}ms", tokenCount,
                String.format("%.1f", elapsedNanos / 1_000_000.0));

        return InferenceResult.builder()
                .generatedTokenIds(new int[0])   // token IDs not surfaced by de.kherud:llama
                .lastLogits(new float[0])
                .promptTokenCount(0)             // prompt token count not surfaced by de.kherud:llama
                .generatedTokenCount(tokenCount)
                .inferenceNanos(elapsedNanos)
                .backendName("llama.cpp")
                .generatedText(generatedText)
                .build();
    }

    private String resolvePrompt(InferenceRequest request) {
        if (request.promptText() != null && !request.promptText().isBlank()) {
            return request.promptText();
        }
        // Fall back: synthesise a numeric token-ID string so the model sees *something*.
        int[] ids = request.inputTokenIds();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(ids[i]);
        }
        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // KV cache — not tracked separately; llama.cpp manages this internally
    // -------------------------------------------------------------------------

    @Override
    public KvCacheStats kvCacheStats() {
        return KvCacheStats.empty();
    }

    // -------------------------------------------------------------------------
    // Tensor ops — not applicable for this backend
    // -------------------------------------------------------------------------

    @Override
    public TensorHandle allocate(DType dtype, long... shape) {
        throw new UnsupportedOperationException(
                "LlamaCppSession does not support raw tensor operations.");
    }

    @Override
    public TensorHandle upload(float[] data, long... shape) {
        throw new UnsupportedOperationException(
                "LlamaCppSession does not support raw tensor operations.");
    }

    @Override
    public TensorHandle quantiseInt8(TensorHandle input) {
        throw new UnsupportedOperationException(
                "LlamaCppSession does not support raw tensor operations.");
    }

    @Override
    public TensorHandle dequantise(TensorHandle input) {
        throw new UnsupportedOperationException(
                "LlamaCppSession does not support raw tensor operations.");
    }

    @Override
    public void synchronize() {
        // llama.cpp inference is synchronous; nothing to wait for.
    }

    @Override
    public void close() {
        // Session holds no resources beyond the shared model reference.
    }
}
