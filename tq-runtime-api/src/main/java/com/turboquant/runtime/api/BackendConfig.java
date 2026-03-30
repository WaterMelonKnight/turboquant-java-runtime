package com.turboquant.runtime.api;

import java.util.Map;

/**
 * Immutable configuration bag passed to a {@link Backend} at initialisation.
 *
 * <p>Use {@link #builder()} to construct instances.</p>
 */
public final class BackendConfig {

    private final int deviceIndex;
    private final Map<String, String> extras;

    private BackendConfig(Builder b) {
        this.deviceIndex = b.deviceIndex;
        this.extras = Map.copyOf(b.extras);
    }

    /** Zero-based GPU device index (ignored by the CPU stub). */
    public int deviceIndex() {
        return deviceIndex;
    }

    /** Provider-specific key/value pairs forwarded verbatim to the backend. */
    public Map<String, String> extras() {
        return extras;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Default config: device 0, no extras. */
    public static BackendConfig defaultConfig() {
        return builder().build();
    }

    @Override
    public String toString() {
        return "BackendConfig{deviceIndex=" + deviceIndex + ", extras=" + extras + '}';
    }

    public static final class Builder {
        private int deviceIndex = 0;
        private final java.util.HashMap<String, String> extras = new java.util.HashMap<>();

        public Builder deviceIndex(int idx) {
            this.deviceIndex = idx;
            return this;
        }

        public Builder extra(String key, String value) {
            this.extras.put(key, value);
            return this;
        }

        public BackendConfig build() {
            return new BackendConfig(this);
        }
    }
}
