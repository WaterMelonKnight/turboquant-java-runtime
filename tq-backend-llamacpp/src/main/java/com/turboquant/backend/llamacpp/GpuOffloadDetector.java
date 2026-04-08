package com.turboquant.backend.llamacpp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Captures System.err during llama.cpp model loading to detect whether the
 * native build supports GPU offload.
 *
 * <p>llama.cpp writes its log output to stderr. A CPU-only build emits a line
 * containing {@code "no GPU offload support"} when {@code nGpuLayers > 0} is
 * requested but the build was compiled without GPU support. This class
 * intercepts that signal so the backend can report an honest offload status.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * GpuOffloadDetector d = new GpuOffloadDetector();
 * d.install();
 * try {
 *     new LlamaModel(params);
 * } finally {
 *     d.restore();
 * }
 * boolean cpuOnly = d.detectedCpuOnlyWarning();
 * }</pre>
 *
 * <p>This approach is best-effort: it relies on llama.cpp's log text, which
 * could change across versions. If the signal is absent we report {@code null}
 * (unknown) rather than claiming GPU offload is active.</p>
 */
final class GpuOffloadDetector {

    private static final Logger log = LoggerFactory.getLogger(GpuOffloadDetector.class);

    /** Substring emitted by CPU-only llama.cpp builds when GPU layers are requested. */
    private static final String CPU_ONLY_SIGNAL = "no GPU offload support";

    private PrintStream            originalErr;
    private ByteArrayOutputStream  captured;
    private boolean                installed = false;

    /**
     * Redirects {@code System.err} to an internal buffer.
     * Must be paired with a {@link #restore()} call in a finally block.
     */
    void install() {
        originalErr = System.err;
        captured    = new ByteArrayOutputStream(4096);
        // Tee: write to both the capture buffer and the original stderr so
        // llama.cpp output is still visible in the terminal.
        System.setErr(new TeeStream(originalErr, captured));
        installed = true;
    }

    /**
     * Restores {@code System.err} to its original stream.
     * Safe to call even if {@link #install()} was never called.
     */
    void restore() {
        if (installed && originalErr != null) {
            System.setErr(originalErr);
            installed = false;
        }
    }

    /**
     * Returns {@code true} if the captured stderr output contains the
     * CPU-only warning emitted by llama.cpp when GPU offload is unavailable.
     */
    boolean detectedCpuOnlyWarning() {
        if (captured == null) return false;
        String output = captured.toString(StandardCharsets.UTF_8);
        boolean detected = output.contains(CPU_ONLY_SIGNAL);
        if (detected) {
            log.debug("GpuOffloadDetector: detected CPU-only signal in llama.cpp output");
        }
        return detected;
    }

    // -------------------------------------------------------------------------
    // TeeStream — writes to two PrintStreams simultaneously
    // -------------------------------------------------------------------------

    private static final class TeeStream extends PrintStream {

        private final ByteArrayOutputStream sink;

        TeeStream(PrintStream primary, ByteArrayOutputStream sink) {
            super(primary, true);
            this.sink = sink;
        }

        @Override
        public void write(int b) {
            super.write(b);
            sink.write(b);
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            super.write(buf, off, len);
            sink.write(buf, off, len);
        }
    }
}
