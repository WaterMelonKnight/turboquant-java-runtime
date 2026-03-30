package com.turboquant.spring;

import com.turboquant.runtime.api.Backend;
import com.turboquant.runtime.api.BackendConfig;
import com.turboquant.runtime.core.BackendRegistry;
import com.turboquant.runtime.core.RuntimeEngine;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Spring Boot auto-configuration for TurboQuant.
 *
 * <p>Registered in
 * {@code META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports}.</p>
 *
 * <p>Beans provided:</p>
 * <ul>
 *   <li>{@link BackendRegistry} — all discovered backend providers</li>
 *   <li>{@link RuntimeEngine} — initialised engine (singleton, destroyed on context close)</li>
 *   <li>{@link Backend} — the active backend (delegate to {@code RuntimeEngine#backend()})</li>
 * </ul>
 *
 * <p>All beans use {@link ConditionalOnMissingBean} so applications can
 * override any of them.</p>
 */
@AutoConfiguration
@EnableConfigurationProperties(TurboQuantProperties.class)
public class TurboQuantAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public BackendRegistry turboQuantBackendRegistry() {
        return new BackendRegistry();
    }

    @Bean(destroyMethod = "close")
    @ConditionalOnMissingBean
    public RuntimeEngine turboQuantRuntimeEngine(TurboQuantProperties props) {
        BackendConfig config = BackendConfig.builder()
                .deviceIndex(props.getDeviceIndex())
                .build();

        if ("auto".equalsIgnoreCase(props.getBackend())) {
            return RuntimeEngine.autoSelect(config);
        }
        return RuntimeEngine.withBackend(props.getBackend(), config);
    }

    @Bean
    @ConditionalOnMissingBean
    public Backend turboQuantBackend(RuntimeEngine engine) {
        return engine.backend();
    }
}
