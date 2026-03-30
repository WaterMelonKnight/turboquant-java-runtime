# TurboQuant Java Runtime

Backend-agnostic Java 17 runtime for TurboQuant quantisation kernels.
Supports **CUDA**, **HIP/ROCm**, and a pure-Java **CPU stub** — all behind a
single clean Java API.

---

## Quick start

```bash
# Build everything (requires JDK 17+, Maven 3.9+)
mvn clean install -DskipTests

# Run the bench CLI against the CPU stub (always works, no GPU needed)
java -jar tq-bench-cli/target/tq-bench-cli-0.1.0-SNAPSHOT-fat.jar

# List all discovered backends
java -jar tq-bench-cli/target/tq-bench-cli-0.1.0-SNAPSHOT-fat.jar --list

# Force a specific backend
java -jar tq-bench-cli/target/tq-bench-cli-0.1.0-SNAPSHOT-fat.jar --backend cpu-stub

# Enable the HIP backend (requires libtq_hip.so on java.library.path)
java -Dtq.backend.hip.enabled=true \
     -Djava.library.path=/opt/rocm/lib \
     -jar tq-bench-cli/target/tq-bench-cli-0.1.0-SNAPSHOT-fat.jar --backend hip
```

---

## Project layout

```
turboquant-java-runtime/
├── pom.xml                        Parent POM, dependency management
│
├── tq-runtime-api/                Public Java API (no GPU vendor imports)
│   └── com.turboquant.runtime.api
│       ├── Backend                Top-level backend interface
│       ├── ComputeSession         Scoped session (alloc, upload, quantise, …)
│       ├── TensorHandle           Opaque tensor (device or heap)
│       ├── BackendConfig          Immutable config bag
│       ├── BackendCapability      Capability enum
│       ├── DType                  FLOAT32 / BFLOAT16 / INT8 / UINT4
│       └── BackendException       Unchecked runtime exception
│
├── tq-runtime-spi/                ServiceLoader SPI (BackendProvider)
│
├── tq-runtime-core/               Discovery engine + RuntimeEngine facade
│   └── com.turboquant.runtime.core
│       ├── BackendRegistry        ServiceLoader wrapper, priority sort
│       └── RuntimeEngine          Main entry point for app code
│
├── tq-backend-cpu-stub/           Pure-Java stub; always available
├── tq-backend-cuda/               JNI bridge to libtq_cuda.so
├── tq-backend-hip/                JNI bridge to libtq_hip.so (disabled by default)
│
├── tq-bench-cli/                  Picocli bench tool (fat JAR)
├── tq-spring-boot-starter/        Spring Boot 3 auto-configuration
│
└── include/
    └── tq_native_api.h            Stable C ABI implemented by every native backend
```

---

## Module dependency graph

```
tq-runtime-api   (no deps)
      ↑
tq-runtime-spi   (api)
      ↑
tq-runtime-core  (api, spi)          ← ServiceLoader discovery lives here
      ↑                ↑
tq-bench-cli     tq-spring-boot-starter
      ↑
tq-backend-cpu-stub / tq-backend-cuda / tq-backend-hip
                    (api, spi)
```

Backend modules only depend on `tq-runtime-api` + `tq-runtime-spi`.
`tq-runtime-core` has **no** compile-time dependency on any backend —
backends are discovered at runtime via `ServiceLoader`.

---

## Backend selection

Priority order (highest wins):

| Backend        | Priority | Available | Notes                          |
|----------------|----------|-----------|--------------------------------|
| `cuda`         | 80       | if `libtq_cuda.so` found | CUDA 11+ |
| `hip`          | 80       | if `libtq_hip.so` found **and** `-Dtq.backend.hip.enabled=true` | ROCm 6+ |
| `cpu-stub`     | 0        | always    | Pure Java, no GPU needed       |

When two backends have the same priority the one with the lexicographically
smaller name wins (deterministic tie-breaking).

Override at runtime:

```java
// Auto-select
RuntimeEngine engine = RuntimeEngine.autoSelect(BackendConfig.defaultConfig());

// Explicit
RuntimeEngine engine = RuntimeEngine.withBackend("cuda",
    BackendConfig.builder().deviceIndex(1).build());
```

---

## Spring Boot usage

Add to your application POM:

```xml
<dependency>
  <groupId>com.turboquant</groupId>
  <artifactId>tq-spring-boot-starter</artifactId>
  <version>0.1.0-SNAPSHOT</version>
</dependency>
<!-- at least one backend must be on the runtime classpath -->
<dependency>
  <groupId>com.turboquant</groupId>
  <artifactId>tq-backend-cpu-stub</artifactId>
  <version>0.1.0-SNAPSHOT</version>
</dependency>
```

`application.properties`:

```properties
turboquant.backend=auto
turboquant.device-index=0
```

Inject the beans:

```java
@Autowired RuntimeEngine engine;   // initialised, ready to use
@Autowired Backend        backend;  // the active backend
```

---

## Adding a new backend

1. Create a Maven module depending on `tq-runtime-api` + `tq-runtime-spi`.
2. Implement `BackendProvider` and `Backend` / `ComputeSession` / `TensorHandle`.
3. Register the provider in `META-INF/services/com.turboquant.runtime.spi.BackendProvider`.
4. Add the module as an optional dependency in `tq-bench-cli` / your app.

No changes to `tq-runtime-core` or any other existing module are required.

---

## Native library (C ABI)

The header `include/tq_native_api.h` defines the stable C ABI.
See [docs/architecture.md](docs/architecture.md) for the design rationale.

To implement a native backend:

```bash
# Generate JNI header from the Java bridge class
javac -h include/ tq-backend-cuda/src/main/java/com/turboquant/backend/cuda/CudaNativeBridge.java

# Implement the generated header against tq_native_api.h
# Build: produces libtq_cuda.so / libtq_hip.so
```

---

## Requirements

- JDK 17 or later
- Maven 3.9+
- (Optional) CUDA Toolkit 11+ for `tq-backend-cuda`
- (Optional) ROCm 6+ for `tq-backend-hip`
