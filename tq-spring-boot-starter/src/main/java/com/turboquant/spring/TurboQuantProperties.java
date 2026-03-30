package com.turboquant.spring;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Spring Boot configuration properties for TurboQuant.
 *
 * <p>Bind via {@code application.properties} / {@code application.yml}:</p>
 * <pre>
 * turboquant.backend=auto
 * turboquant.device-index=0
 * </pre>
 */
@ConfigurationProperties(prefix = "turboquant")
public class TurboQuantProperties {

    /**
     * Backend to use.  {@code "auto"} (default) picks the highest-priority
     * available backend.  Other values: {@code "cpu-stub"}, {@code "cuda"},
     * {@code "hip"}.
     */
    private String backend = "auto";

    /**
     * Zero-based GPU device index passed to {@link com.turboquant.runtime.api.BackendConfig}.
     */
    private int deviceIndex = 0;

    public String getBackend() {
        return backend;
    }

    public void setBackend(String backend) {
        this.backend = backend;
    }

    public int getDeviceIndex() {
        return deviceIndex;
    }

    public void setDeviceIndex(int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }
}
