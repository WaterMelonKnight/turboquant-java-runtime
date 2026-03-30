package com.turboquant.runtime.core;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendException;
import com.turboquant.runtime.spi.BackendProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Discovers {@link BackendProvider} implementations on the classpath via
 * {@link ServiceLoader} and provides lookup utilities.
 *
 * <p>Thread-safety: instances are immutable after construction.
 * Construct once (e.g. at Spring Boot startup) and share.</p>
 */
public final class BackendRegistry {

    private static final Logger log = LoggerFactory.getLogger(BackendRegistry.class);

    private final List<BackendProvider> providers;

    /**
     * Create a registry using the context class loader.
     * All available {@link BackendProvider} implementations on the classpath
     * are discovered and filtered by {@link BackendProvider#isAvailable()}.
     */
    public BackendRegistry() {
        this(Thread.currentThread().getContextClassLoader());
    }

    public BackendRegistry(ClassLoader classLoader) {
        var loaded = new ArrayList<BackendProvider>();
        ServiceLoader.load(BackendProvider.class, classLoader)
                .forEach(provider -> {
                    log.debug("Found backend provider: {} (priority={}, available={})",
                            provider.backendName(), provider.priority(), provider.isAvailable());
                    loaded.add(provider);
                });
        // stable sort: descending priority, then alphabetical name for ties
        loaded.sort(Comparator
                .comparingInt(BackendProvider::priority).reversed()
                .thenComparing(BackendProvider::backendName));
        this.providers = List.copyOf(loaded);
        log.info("BackendRegistry initialised with {} provider(s): {}",
                providers.size(),
                providers.stream().map(BackendProvider::backendName).toList());
    }

    /** All registered providers, sorted by descending priority. */
    public List<BackendProvider> allProviders() {
        return providers;
    }

    /** All available providers (those for which {@link BackendProvider#isAvailable()} is true). */
    public List<BackendProvider> availableProviders() {
        return providers.stream().filter(BackendProvider::isAvailable).toList();
    }

    /**
     * Return the highest-priority available provider, or empty if none is available.
     */
    public Optional<BackendProvider> bestProvider() {
        return availableProviders().stream().findFirst();
    }

    /**
     * Return the provider with the given {@code name}, regardless of availability.
     */
    public Optional<BackendProvider> findByName(String name) {
        return providers.stream()
                .filter(p -> p.backendName().equalsIgnoreCase(name))
                .findFirst();
    }

    /**
     * Create a {@link Backend} from the highest-priority available provider.
     *
     * @throws BackendException if no available backend was found
     */
    public Backend createBest() {
        return bestProvider()
                .map(BackendProvider::create)
                .orElseThrow(() -> new BackendException(
                        "No available TurboQuant backend found on the classpath. " +
                        "Add tq-backend-cpu-stub (or another backend) as a runtime dependency."));
    }

    /**
     * Create a {@link Backend} from the named provider.
     *
     * @throws BackendException if the named backend is not registered or not available
     */
    public Backend createByName(String name) {
        BackendProvider provider = findByName(name)
                .orElseThrow(() -> new BackendException("Backend not registered: " + name));
        if (!provider.isAvailable()) {
            throw new BackendException("Backend '" + name + "' is registered but not available in this environment.");
        }
        return provider.create();
    }
}
