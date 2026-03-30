# native/hip/CMakeLists.txt
#
# Builds libtq_hip.so from the HIP mock C ABI implementation and the JNI
# bridge.  No HIP/ROCm toolkit is required for the placeholder — it compiles
# with any C99 toolchain.
#
# Usage (from the repo root):
#   mkdir -p native/hip/build && cd native/hip/build
#   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   make -j$(nproc)
#
# To load the library at runtime:
#   java -Dtq.backend.hip.enabled=true \
#        -Djava.library.path=native/hip/build \
#        -jar tq-bench-cli-fat.jar --backend hip
#
# Real HIP port
# ─────────────
# Replace tq_native_mock.c with a real HIP implementation built with hipcc:
#
#   find_package(hip REQUIRED)
#   target_sources(tq_hip PRIVATE tq_hip_kernels.cpp)
#   target_link_libraries(tq_hip PRIVATE hip::device)
#
# The JNI bridge (tq_jni_bridge.c) and the Java layer require no changes;
# see docs/rocm-porting-plan.md for the full porting checklist.

cmake_minimum_required(VERSION 3.18)
project(tq_hip C)

# ---------------------------------------------------------------------------
# Locate JNI headers
# ---------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(NOT JNI_FOUND)
    message(FATAL_ERROR "JNI not found. Set JAVA_HOME or JDK_ROOT.")
endif()

# ---------------------------------------------------------------------------
# Shared library target
# ---------------------------------------------------------------------------
add_library(tq_hip SHARED
    tq_native_mock.c
    tq_jni_bridge.c
)

target_include_directories(tq_hip PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_options(tq_hip PRIVATE
    -std=c99
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

target_link_libraries(tq_hip PRIVATE ${JNI_LIBRARIES})

if(UNIX AND NOT APPLE)
    target_link_libraries(tq_hip PRIVATE rt)
endif()

# Strip the "lib" prefix on Windows to match System.loadLibrary("tq_hip").
set_target_properties(tq_hip PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tq_hip"
)

# ---------------------------------------------------------------------------
# Install rule
# ---------------------------------------------------------------------------
install(TARGETS tq_hip
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "tq_hip target configured")
message(STATUS "  JNI include dirs : ${JNI_INCLUDE_DIRS}")
message(STATUS "  Source           : tq_native_mock.c  tq_jni_bridge.c")
message(STATUS "  Header           : ../../include/tq_native_api.h")
