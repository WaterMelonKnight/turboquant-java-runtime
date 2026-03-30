package com.turboquant.runtime.api;

/**
 * Thrown when the requested backend cannot be loaded or initialised.
 */
public class BackendException extends RuntimeException {

    public BackendException(String message) {
        super(message);
    }

    public BackendException(String message, Throwable cause) {
        super(message, cause);
    }
}
