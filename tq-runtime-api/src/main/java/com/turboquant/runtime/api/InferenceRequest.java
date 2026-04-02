package com.turboquant.runtime.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable request object for a single forward-pass through the model.
 *
 * <p>Two request modes are supported:</p>
 * <ul>
 *   <li><b>Token-based:</b> provide {@code inputTokenIds} — used by backends that consume
 *       pre-tokenised input (cpu-stub, cuda, hip).</li>
 *   <li><b>Text-based:</b> provide {@code promptText} — used by text-native backends such
 *       as llama.cpp that handle tokenisation internally.</li>
 * </ul>
 *
 * <p>At least one of {@code promptText} or {@code inputTokenIds} must be set.
 * Neither is required to be present when the other is supplied.</p>
 *
 * <pre>{@code
 * // Token-based (cpu-stub, cuda, hip)
 * InferenceRequest req = InferenceRequest.syntheticPrompt(128, 32);
 *
 * // Text-based (llama.cpp)
 * InferenceRequest req = InferenceRequest.fromText("Once upon a time", 64);
 * }</pre>
 */
public final class InferenceRequest {

    private final int[]  inputTokenIds;
    private final int    maxNewTokens;
    private final float  temperature;
    private final float  topP;
    private final String promptText;

    private InferenceRequest(Builder b) {
        this.inputTokenIds = Arrays.copyOf(b.inputTokenIds, b.inputTokenIds.length);
        this.maxNewTokens  = b.maxNewTokens;
        this.temperature   = b.temperature;
        this.topP          = b.topP;
        this.promptText    = b.promptText;
    }

    /**
     * Token IDs of the prompt.
     * Returns an empty array for text-mode requests (those built via {@link #fromText}).
     */
    public int[] inputTokenIds() {
        return Arrays.copyOf(inputTokenIds, inputTokenIds.length);
    }

    /** Number of new tokens to generate. Must be &gt; 0. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    /**
     * Sampling temperature (default 1.0).
     * Backends may ignore this in stub implementations.
     */
    public float temperature() {
        return temperature;
    }

    /**
     * Top-p (nucleus) sampling parameter (default 1.0 = disabled).
     * Backends may ignore this in stub implementations.
     */
    public float topP() {
        return topP;
    }

    /**
     * Raw prompt text, used by text-native backends (e.g. llama.cpp).
     * May be {@code null} when token IDs are supplied directly.
     */
    public String promptText() {
        return promptText;
    }

    /** Convenience: total tokens that will be processed (prompt + generated). */
    public int totalTokens() {
        return inputTokenIds.length + maxNewTokens;
    }

    @Override
    public String toString() {
        if (promptText != null && !promptText.isBlank()) {
            return "InferenceRequest{mode=text, maxNewTokens=" + maxNewTokens
                    + ", temperature=" + temperature
                    + ", topP=" + topP + '}';
        }
        return "InferenceRequest{mode=tokens, promptLen=" + inputTokenIds.length
                + ", maxNewTokens=" + maxNewTokens
                + ", temperature=" + temperature
                + ", topP=" + topP + '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Convenience factory: sequential token IDs [0, 1, 2, …, promptLen-1]. */
    public static InferenceRequest syntheticPrompt(int promptLen, int maxNewTokens) {
        int[] ids = new int[promptLen];
        for (int i = 0; i < promptLen; i++) ids[i] = i + 1; // avoid token 0 (often PAD)
        return builder().inputTokenIds(ids).maxNewTokens(maxNewTokens).build();
    }

    /** Convenience factory: text prompt for text-native backends (e.g. llama.cpp). */
    public static InferenceRequest fromText(String prompt, int maxNewTokens) {
        Objects.requireNonNull(prompt, "prompt must not be null");
        if (prompt.isBlank()) {
            throw new IllegalArgumentException("prompt must not be blank");
        }
        return builder()
                .promptText(prompt)
                .maxNewTokens(maxNewTokens)
                .build();
    }

    public static final class Builder {
        private int[]  inputTokenIds = new int[0];
        private int    maxNewTokens  = 1;
        private float  temperature   = 1.0f;
        private float  topP          = 1.0f;
        private String promptText    = null;

        public Builder inputTokenIds(int... ids) {
            Objects.requireNonNull(ids, "inputTokenIds must not be null");
            this.inputTokenIds = ids;
            return this;
        }

        public Builder maxNewTokens(int n) {
            if (n <= 0) throw new IllegalArgumentException("maxNewTokens must be > 0");
            this.maxNewTokens = n;
            return this;
        }

        public Builder temperature(float t) {
            if (t <= 0) throw new IllegalArgumentException("temperature must be > 0");
            this.temperature = t;
            return this;
        }

        public Builder topP(float p) {
            if (p <= 0 || p > 1) throw new IllegalArgumentException("topP must be in (0, 1]");
            this.topP = p;
            return this;
        }

        public Builder promptText(String text) {
            this.promptText = text;
            return this;
        }

        public InferenceRequest build() {
            boolean hasText   = promptText != null && !promptText.isBlank();
            boolean hasTokens = inputTokenIds.length > 0;
            if (!hasText && !hasTokens) {
                throw new IllegalStateException(
                        "InferenceRequest requires either promptText or at least one inputTokenId");
            }
            return new InferenceRequest(this);
        }
    }
}
