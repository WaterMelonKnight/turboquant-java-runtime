package com.turboquant.backend.hip;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

/**
 * HIP/ROCm {@link Backend} implementation.
 *
 * <p><strong>Disabled by default</strong> — enable with system property
 * {@code -Dtq.backend.hip.enabled=true} or the Maven {@code -Phip} profile.
 * This prevents accidental selection on CUDA-only hosts where
 * {@code libtq_hip.so} is absent.</p>
 *
 * <p>Structurally identical to {@code CudaBackend}: both hold an opaque
 * {@code long rtHandle} obtained from {@code tq_runtime_create} in the shared
 * C ABI ({@code tq_native_api.h}).  The only runtime differences are the
 * library name, the {@code hip.enabled} guard, and ROCm-specific version
 * formatting in {@link #version()}.</p>
 *
 * <h2>ROCm porting cost</h2>
 * Because the Java layer is backend-agnostic, porting this backend from
 * placeholder to a real ROCm implementation requires <em>only</em> C/HIP
 * changes in {@code native/hip/}.  See {@code docs/rocm-porting-plan.md}.
 */
public final class HipBackend implements Backend {

    private static final Logger log = LoggerFactory.getLogger(HipBackend.class);

    /** True only when the HIP system property is set AND the library loaded. */
    private static final boolean NATIVE_AVAILABLE;

    static {
        boolean available = false;
        if (Boolean.getBoolean("tq.backend.hip.enabled")) {
            try {
                Class.forName("com.turboquant.backend.hip.HipNativeBridge");
                available = true;
            } catch (ClassNotFoundException | UnsatisfiedLinkError e) {
                log.warn("HIP native library not found — HIP backend is unavailable. ({})",
                         e.getMessage());
            }
        } else {
            log.debug("HIP backend disabled. Set -Dtq.backend.hip.enabled=true to enable.");
        }
        NATIVE_AVAILABLE = available;
    }

    /** Opaque runtime handle returned by tq_runtime_create; 0 before init(). */
    private long rtHandle = 0L;

    /** Human-readable runtime description populated after init(). */
    private String runtimeDescription = "not initialised";

    @Override
    public String name() {
        return "hip";
    }

    @Override
    public String version() {
        return runtimeDescription;
    }

    @Override
    public Set<BackendCapability> capabilities() {
        return Set.of(
                BackendCapability.INT8_MATMUL,
                BackendCapability.UINT4_WEIGHT_QUANT,
                BackendCapability.MULTI_STREAM
        );
    }

    @Override
    public boolean isAvailable() {
        return NATIVE_AVAILABLE;
    }

    @Override
    public void init(BackendConfig config) {
        if (!NATIVE_AVAILABLE) {
            throw new BackendException(
                "HIP backend is not available. " +
                "Set -Dtq.backend.hip.enabled=true and ensure libtq_hip.so is on java.library.path.");
        }
        long handle = HipNativeBridge.tqRuntimeCreate(config.deviceIndex());
        if (handle == 0L) {
            throw new BackendException(
                    "tq_runtime_create failed for device index " + config.deviceIndex());
        }
        rtHandle           = handle;
        runtimeDescription = HipNativeBridge.tqRuntimeDescribe(rtHandle);
        log.info("HIP backend initialised — {}", runtimeDescription);
    }

    @Override
    public ComputeSession newSession() {
        if (rtHandle == 0L) {
            throw new BackendException("HipBackend.init() must be called before newSession().");
        }
        long sessionHandle = HipNativeBridge.tqSessionCreate(rtHandle);
        if (sessionHandle == 0L) {
            throw new BackendException("tq_session_create returned a null handle.");
        }
        return new HipComputeSession(sessionHandle);
    }

    @Override
    public void close() {
        if (NATIVE_AVAILABLE && rtHandle != 0L) {
            HipNativeBridge.tqRuntimeDestroy(rtHandle);
            rtHandle = 0L;
        }
        log.debug("HipBackend closed");
    }
}
