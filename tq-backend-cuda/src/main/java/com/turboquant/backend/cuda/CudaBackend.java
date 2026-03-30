package com.turboquant.backend.cuda;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendCapability;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.api.ComputeSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

/**
 * CUDA {@link Backend} implementation.
 *
 * <p>Availability is determined at class-load time by attempting to load the
 * native library.  If the library is absent, {@link #isAvailable()} returns
 * {@code false} and the backend is skipped during auto-selection.</p>
 *
 * <p>After a successful {@link #init}, the backend holds an opaque
 * <em>runtime handle</em> ({@code long rtHandle}) created by
 * {@code tq_runtime_create} in the C ABI.  Every new session is derived from
 * this handle via {@code tq_session_create}.</p>
 */
public final class CudaBackend implements Backend {

    private static final Logger log = LoggerFactory.getLogger(CudaBackend.class);

    /** True only when the native library loaded successfully at class-init time. */
    private static final boolean NATIVE_AVAILABLE;

    static {
        boolean available = false;
        try {
            // Trigger the static initialiser in CudaNativeBridge which calls
            // System.loadLibrary("tq_cuda").
            Class.forName("com.turboquant.backend.cuda.CudaNativeBridge");
            available = true;
        } catch (ClassNotFoundException | UnsatisfiedLinkError e) {
            log.warn("CUDA native library not found — CUDA backend is unavailable. ({})",
                     e.getMessage());
        }
        NATIVE_AVAILABLE = available;
    }

    /** Opaque runtime handle returned by tq_runtime_create; 0 before init(). */
    private long rtHandle = 0L;

    /** Human-readable runtime description populated after init(). */
    private String runtimeDescription = "not initialised";

    @Override
    public String name() {
        return "cuda";
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
            throw new BackendException("CUDA backend is not available (native library missing).");
        }
        long handle = CudaNativeBridge.tqRuntimeCreate(config.deviceIndex());
        if (handle == 0L) {
            throw new BackendException(
                    "tq_runtime_create failed for device index " + config.deviceIndex());
        }
        rtHandle             = handle;
        runtimeDescription   = CudaNativeBridge.tqRuntimeDescribe(rtHandle);
        log.info("CUDA backend initialised — {}", runtimeDescription);
    }

    @Override
    public ComputeSession newSession() {
        if (rtHandle == 0L) {
            throw new BackendException("CudaBackend.init() must be called before newSession().");
        }
        long sessionHandle = CudaNativeBridge.tqSessionCreate(rtHandle);
        if (sessionHandle == 0L) {
            throw new BackendException("tq_session_create returned a null handle.");
        }
        return new CudaComputeSession(sessionHandle);
    }

    @Override
    public void close() {
        if (NATIVE_AVAILABLE && rtHandle != 0L) {
            CudaNativeBridge.tqRuntimeDestroy(rtHandle);
            rtHandle = 0L;
        }
        log.debug("CudaBackend closed");
    }
}
