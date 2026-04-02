cmake_minimum_required(VERSION 3.22)
project(rl_cnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── godot-cpp ─────────────────────────────────────────────────────────────────
# Expected as a submodule at: addons/rl-agent-plugin/native/godot-cpp
# Add it with:
#   git submodule add -b 4.x https://github.com/godotengine/godot-cpp.git \
#       addons/rl-agent-plugin/native/godot-cpp
set(GODOT_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp")
if(NOT EXISTS "${GODOT_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "godot-cpp not found at '${GODOT_CPP_DIR}'.\n"
        "Run: git submodule add -b 4.x https://github.com/godotengine/godot-cpp.git "
        "addons/rl-agent-plugin/native/godot-cpp")
endif()
add_subdirectory(${GODOT_CPP_DIR} godot-cpp)

# ── Library ───────────────────────────────────────────────────────────────────
add_library(rl_cnn SHARED
    src/register_types.cpp
    src/RlCnnEncoder.cpp
    src/RlDenseLayer.cpp
    src/RlGruLayer.cpp
    src/RlLayerNormLayer.cpp
    src/RlLstmLayer.cpp
)

target_include_directories(rl_cnn PRIVATE src/)
target_link_libraries(rl_cnn PRIVATE godot-cpp)

# ── SIMD ──────────────────────────────────────────────────────────────────────
# AVX2 + FMA enabled by default on x86-64.  Pass -DDISABLE_AVX2=ON to opt out.
option(DISABLE_AVX2 "Disable AVX2/FMA even if the compiler supports it" OFF)
if(NOT DISABLE_AVX2)
    include(CheckCXXCompilerFlag)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(rl_cnn PRIVATE /arch:AVX2)
            target_compile_definitions(rl_cnn PRIVATE RL_USE_AVX2)
        endif()
    else()
        check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(rl_cnn PRIVATE -mavx2 -mfma)
            target_compile_definitions(rl_cnn PRIVATE RL_USE_AVX2)
        endif()
    endif()
endif()

# ── Output naming (Godot convention) ─────────────────────────────────────────
# Outputs to:  addons/rl-agent-plugin/native/bin/
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set_target_properties(rl_cnn PROPERTIES
    PREFIX "lib"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

# Per-configuration suffixes so debug and release sit side-by-side.
# The .gdextension file references these exact names.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_CONFIGURATION_TYPES)
    set_target_properties(rl_cnn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${OUTPUT_DIR}"
    )
endif()
