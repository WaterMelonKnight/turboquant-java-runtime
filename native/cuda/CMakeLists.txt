# native/cuda/CMakeLists.txt
#
# Builds libtq_cuda.so (Linux) / tq_cuda.dll (Windows) from the mock C ABI
# implementation and the JNI bridge.  No CUDA toolkit required — the mock
# uses only standard C99 and POSIX APIs.
#
# Usage (from the repo root):
#   mkdir -p native/cuda/build && cd native/cuda/build
#   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   make -j$(nproc)
#
# The resulting library must be on java.library.path when running the CUDA
# backend.  Example:
#   java -Djava.library.path=native/cuda/build -jar tq-bench-cli-fat.jar \
#        --backend cuda
#
# HIP note:
#   Replace the two source files with their HIP equivalents and adjust the
#   library name to tq_hip to build the HIP backend library.  The ABI is
#   identical — only the implementation differs.

cmake_minimum_required(VERSION 3.18)
project(tq_cuda C)

# ---------------------------------------------------------------------------
# Locate JNI headers (required for jni.h / jni_md.h)
# ---------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(NOT JNI_FOUND)
    message(FATAL_ERROR "JNI not found. Set JAVA_HOME or JDK_ROOT.")
endif()

# ---------------------------------------------------------------------------
# Shared library target
# ---------------------------------------------------------------------------
add_library(tq_cuda SHARED
    tq_native_mock.c
    tq_jni_bridge.c
)

# Expose the project-level include/ directory (contains tq_native_api.h).
target_include_directories(tq_cuda PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Strict warnings help catch ABI mismatches early.
target_compile_options(tq_cuda PRIVATE
    -std=c99
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

target_link_libraries(tq_cuda PRIVATE ${JNI_LIBRARIES})

# On Linux, link against librt for clock_gettime (glibc < 2.17 only, but
# harmless on newer toolchains).
if(UNIX AND NOT APPLE)
    target_link_libraries(tq_cuda PRIVATE rt)
endif()

# Strip the "lib" prefix on Windows to match System.loadLibrary("tq_cuda").
set_target_properties(tq_cuda PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tq_cuda"
)

# ---------------------------------------------------------------------------
# Install rule (optional — copy into a well-known location)
# ---------------------------------------------------------------------------
install(TARGETS tq_cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib   # .dll on Windows
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "tq_cuda target configured")
message(STATUS "  JNI include dirs : ${JNI_INCLUDE_DIRS}")
message(STATUS "  Source           : tq_native_mock.c  tq_jni_bridge.c")
message(STATUS "  Header           : ../../include/tq_native_api.h")
