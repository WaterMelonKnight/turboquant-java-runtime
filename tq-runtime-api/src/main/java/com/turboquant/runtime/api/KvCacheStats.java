package com.turboquant.runtime.api;

/**
 * Snapshot of the session's key–value cache state.
 *
 * <p>All fields are placeholders in stub backends.  Real backends will
 * populate these from the GPU allocator and attention-cache metadata.</p>
 *
 * @param cachedTokens    Number of tokens currently stored in the KV cache.
 * @param capacityTokens  Maximum tokens the cache can hold before eviction.
 * @param cacheSizeBytes  Actual device memory consumed by the KV cache, in bytes.
 * @param hitRate         Fraction of attention lookups satisfied by the cache (0.0–1.0).
 *                        Always 0.0 for the first request; increases with cache reuse.
 */
public record KvCacheStats(
        int    cachedTokens,
        int    capacityTokens,
        long   cacheSizeBytes,
        double hitRate
) {
    /** Convenience: cache occupancy as a percentage string, e.g. {@code "62.5%"}. */
    public String occupancyPercent() {
        if (capacityTokens == 0) return "N/A";
        return String.format("%.1f%%", 100.0 * cachedTokens / capacityTokens);
    }

    /** Convenience: {@link #cacheSizeBytes} expressed as mebibytes. */
    public double cacheSizeMiB() {
        return cacheSizeBytes / (1024.0 * 1024.0);
    }

    /** A zero-state snapshot (empty cache, no hits). */
    public static KvCacheStats empty() {
        return new KvCacheStats(0, 0, 0L, 0.0);
    }
}
