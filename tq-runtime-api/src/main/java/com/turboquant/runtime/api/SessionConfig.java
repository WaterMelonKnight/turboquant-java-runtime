package com.turboquant.runtime.api;

import java.util.Map;

/**
 * Model-level session configuration passed to {@link Backend#init(SessionConfig)}.
 *
 * <p>Extends the device-selection concern of {@link BackendConfig} with
 * inference parameters: which model to load, how many tokens to generate, etc.</p>
 *
 * <p>Use {@link #builder()} to construct instances:</p>
 * <pre>{@code
 * SessionConfig cfg = SessionConfig.builder()
 *     .modelPath("/models/llama-3-8b-q4.gguf")
 *     .maxContextTokens(4096)
 *     .maxNewTokens(256)
 *     .temperature(0.7f)
 *     .topP(0.9f)
 *     .build();
 * }</pre>
 */
public final class SessionConfig {

    private final String modelPath;
    private final int    maxContextTokens;
    private final int    maxNewTokens;
    private final float  temperature;
    private final float  topP;
    private final int    deviceId;
    private final Map<String, String> extras;

    private SessionConfig(Builder b) {
        this.modelPath        = b.modelPath;
        this.maxContextTokens = b.maxContextTokens;
        this.maxNewTokens     = b.maxNewTokens;
        this.temperature      = b.temperature;
        this.topP             = b.topP;
        this.deviceId         = b.deviceId;
        this.extras           = Map.copyOf(b.extras);
    }

    /** Absolute path to the model file (e.g. a GGUF file). May be {@code null} for stub backends. */
    public String modelPath() {
        return modelPath;
    }

    /** Maximum number of tokens in the context window (prompt + generated). */
    public int maxContextTokens() {
        return maxContextTokens;
    }

    /** Default maximum number of new tokens to generate per inference call. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    /** Sampling temperature (default 1.0). Higher = more random. */
    public float temperature() {
        return temperature;
    }

    /** Top-p (nucleus) sampling parameter (default 1.0 = disabled). */
    public float topP() {
        return topP;
    }

    /** Zero-based device index (GPU ordinal or ignored by CPU backends). */
    public int deviceId() {
        return deviceId;
    }

    /** Provider-specific key/value pairs forwarded verbatim to the backend. */
    public Map<String, String> extras() {
        return extras;
    }

    /**
     * Convert to a {@link BackendConfig} for backends that only implement
     * {@link Backend#init(BackendConfig)}.  Model-level parameters are encoded
     * as {@code extras} entries with well-known keys.
     */
    public BackendConfig toBackendConfig() {
        BackendConfig.Builder b = BackendConfig.builder().deviceIndex(deviceId);
        extras.forEach(b::extra);
        if (modelPath != null) b.extra("model.path", modelPath);
        b.extra("max.context.tokens", String.valueOf(maxContextTokens));
        b.extra("max.new.tokens",     String.valueOf(maxNewTokens));
        b.extra("temperature",        String.valueOf(temperature));
        b.extra("top.p",              String.valueOf(topP));
        return b.build();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public String toString() {
        return "SessionConfig{modelPath=" + modelPath
                + ", maxContextTokens=" + maxContextTokens
                + ", maxNewTokens=" + maxNewTokens
                + ", temperature=" + temperature
                + ", topP=" + topP
                + ", deviceId=" + deviceId + '}';
    }

    public static final class Builder {
        private String modelPath        = null;
        private int    maxContextTokens = 2048;
        private int    maxNewTokens     = 128;
        private float  temperature      = 1.0f;
        private float  topP             = 1.0f;
        private int    deviceId         = 0;
        private final java.util.HashMap<String, String> extras = new java.util.HashMap<>();

        public Builder modelPath(String path) {
            this.modelPath = path;
            return this;
        }

        public Builder maxContextTokens(int n) {
            if (n <= 0) throw new IllegalArgumentException("maxContextTokens must be > 0");
            this.maxContextTokens = n;
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

        public Builder deviceId(int id) {
            this.deviceId = id;
            return this;
        }

        public Builder extra(String key, String value) {
            this.extras.put(key, value);
            return this;
        }

        public SessionConfig build() {
            return new SessionConfig(this);
        }
    }
}
