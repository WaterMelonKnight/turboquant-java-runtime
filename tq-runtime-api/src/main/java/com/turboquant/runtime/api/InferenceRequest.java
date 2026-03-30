package com.turboquant.runtime.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable request object for a single forward-pass through the model.
 *
 * <p>Build via {@link #builder()}:</p>
 * <pre>{@code
 * InferenceRequest req = InferenceRequest.builder()
 *     .inputTokenIds(new int[]{1, 2, 3})
 *     .maxNewTokens(32)
 *     .build();
 * }</pre>
 */
public final class InferenceRequest {

    private final int[] inputTokenIds;
    private final int   maxNewTokens;
    private final float temperature;

    private InferenceRequest(Builder b) {
        this.inputTokenIds = Arrays.copyOf(b.inputTokenIds, b.inputTokenIds.length);
        this.maxNewTokens  = b.maxNewTokens;
        this.temperature   = b.temperature;
    }

    /** Token IDs of the prompt. Must be non-empty. */
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

    /** Convenience: total tokens that will be processed (prompt + generated). */
    public int totalTokens() {
        return inputTokenIds.length + maxNewTokens;
    }

    @Override
    public String toString() {
        return "InferenceRequest{promptLen=" + inputTokenIds.length
                + ", maxNewTokens=" + maxNewTokens
                + ", temperature=" + temperature + '}';
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

    public static final class Builder {
        private int[]  inputTokenIds = new int[0];
        private int    maxNewTokens  = 1;
        private float  temperature   = 1.0f;

        public Builder inputTokenIds(int... ids) {
            Objects.requireNonNull(ids, "inputTokenIds must not be null");
            if (ids.length == 0) throw new IllegalArgumentException("inputTokenIds must be non-empty");
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

        public InferenceRequest build() {
            if (inputTokenIds.length == 0) {
                throw new IllegalStateException("inputTokenIds must be set before building");
            }
            return new InferenceRequest(this);
        }
    }
}
