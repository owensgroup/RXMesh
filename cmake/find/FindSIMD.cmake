# Find SIMD support (SSE, AVX, FMA)
# This module defines
#  SIMD_FOUND, if false, do not try to use SIMD
#  SIMD_FLAGS, the flags to add to the compiler
#  SIMD_LEVEL, the highest SIMD level supported
#  SIMD_FEATURES, list of all available SIMD features
#  SIMD_OPTIMIZATION_LEVEL, automatic optimization level (NONE, SSE, AVX, AVX2, AVX512)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Include individual SIMD modules
find_package(SSE QUIET)
find_package(AVX QUIET)
find_package(FMA QUIET)

# Set up SIMD variables
set(SIMD_FOUND FALSE)
set(SIMD_FLAGS "")
set(SIMD_LEVEL "NONE")
set(SIMD_FEATURES "")
set(SIMD_OPTIMIZATION_LEVEL "NONE")

# Collect all available SIMD features
if(SSE_FOUND)
    set(SIMD_FOUND TRUE)
    set(SIMD_FLAGS "${SIMD_FLAGS} ${SSE_FLAGS}")
    list(APPEND SIMD_FEATURES "SSE:${SSE_LEVEL}")
    if(SSE_LEVEL STREQUAL "SSE4_2")
        set(SIMD_OPTIMIZATION_LEVEL "SSE4.2")
    elseif(SSE_LEVEL STREQUAL "SSE4_1")
        set(SIMD_OPTIMIZATION_LEVEL "SSE4.1")
    elseif(SSE_LEVEL STREQUAL "SSSE3")
        set(SIMD_OPTIMIZATION_LEVEL "SSSE3")
    elseif(SSE_LEVEL STREQUAL "SSE3")
        set(SIMD_OPTIMIZATION_LEVEL "SSE3")
    elseif(SSE_LEVEL STREQUAL "SSE2")
        set(SIMD_OPTIMIZATION_LEVEL "SSE2")
    elseif(SSE_LEVEL STREQUAL "SSE")
        set(SIMD_OPTIMIZATION_LEVEL "SSE")
    endif()
endif()

if(AVX_FOUND)
    set(SIMD_FOUND TRUE)
    set(SIMD_FLAGS "${SIMD_FLAGS} ${AVX_FLAGS}")
    list(APPEND SIMD_FEATURES "AVX:${AVX_LEVEL}")
    
    # Update optimization level based on AVX support
    if(AVX_LEVEL STREQUAL "AVX512F" OR AVX_LEVEL STREQUAL "AVX512BW" OR AVX_LEVEL STREQUAL "AVX512VL")
        set(SIMD_OPTIMIZATION_LEVEL "AVX-512")
    elseif(AVX_LEVEL STREQUAL "AVX2")
        set(SIMD_OPTIMIZATION_LEVEL "AVX2")
    elseif(AVX_LEVEL STREQUAL "AVX")
        set(SIMD_OPTIMIZATION_LEVEL "AVX")
    endif()
endif()

if(FMA_FOUND)
    set(SIMD_FOUND TRUE)
    set(SIMD_FLAGS "${SIMD_FLAGS} ${FMA_FLAGS}")
    list(APPEND SIMD_FEATURES "FMA")
endif()

# Set the overall SIMD level
if(SIMD_OPTIMIZATION_LEVEL STREQUAL "AVX-512")
    set(SIMD_LEVEL "AVX512")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "AVX2")
    set(SIMD_LEVEL "AVX2")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "AVX")
    set(SIMD_LEVEL "AVX")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSE4.2")
    set(SIMD_LEVEL "SSE4.2")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSE4.1")
    set(SIMD_LEVEL "SSE4.1")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSSE3")
    set(SIMD_LEVEL "SSSE3")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSE3")
    set(SIMD_LEVEL "SSE3")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSE2")
    set(SIMD_LEVEL "SSE2")
elseif(SIMD_OPTIMIZATION_LEVEL STREQUAL "SSE")
    set(SIMD_LEVEL "SSE")
else()
    set(SIMD_LEVEL "NONE")
endif()

# Clean up flags
if(SIMD_FOUND)
    string(STRIP ${SIMD_FLAGS} SIMD_FLAGS)
    message(STATUS "SIMD: Found support for: ${SIMD_FEATURES}")
    message(STATUS "SIMD: Optimization level: ${SIMD_OPTIMIZATION_LEVEL}")
    message(STATUS "SIMD: Using flags: ${SIMD_FLAGS}")
else()
    message(STATUS "SIMD: No SIMD support found")
endif()

# Provide convenience functions for targets
function(target_enable_simd TARGET_NAME)
    if(SIMD_FOUND)
        target_compile_options(${TARGET_NAME} PRIVATE ${SIMD_FLAGS})
        target_compile_definitions(${TARGET_NAME} PRIVATE 
            SIMD_LEVEL="${SIMD_LEVEL}"
            SIMD_OPTIMIZATION_LEVEL="${SIMD_OPTIMIZATION_LEVEL}"
        )
        message(STATUS "SIMD: Enabled for target ${TARGET_NAME} with level ${SIMD_LEVEL}")
    else()
        message(WARNING "SIMD: Cannot enable SIMD for target ${TARGET_NAME} - no support found")
    endif()
endfunction()

# Mark variables as advanced
mark_as_advanced(SIMD_FLAGS SIMD_FOUND SIMD_LEVEL SIMD_FEATURES SIMD_OPTIMIZATION_LEVEL)
