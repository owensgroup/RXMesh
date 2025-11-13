# SIMD Detection Modules

This directory contains enhanced CMake modules for detecting and configuring SIMD instruction sets (SSE, AVX, FMA) and CPU features.

## Modules Overview

### 1. SSE.cmake
Detects SSE instruction set support at multiple levels:
- **SSE**: Basic SSE support
- **SSE2**: Enhanced SSE2 support
- **SSE3**: SSE3 instructions
- **SSSE3**: Supplemental SSE3
- **SSE4.1**: SSE4.1 instructions
- **SSE4.2**: SSE4.2 instructions

**Variables:**
- `SSE_FOUND`: Whether SSE is supported
- `SSE_FLAGS`: Compiler flags to enable SSE
- `SSE_RUNTIME_FOUND`: Runtime SSE support
- `SSE_LEVEL`: Highest supported SSE level

### 2. AVX.cmake
Detects AVX instruction set support:
- **AVX**: Basic AVX support
- **AVX2**: Enhanced AVX2 support
- **AVX512F**: AVX-512 Foundation
- **AVX512BW**: AVX-512 Byte and Word
- **AVX512VL**: AVX-512 Vector Length
- **AVX512DQ**: AVX-512 Double and Quad
- **AVX512CD**: AVX-512 Conflict Detection

**Variables:**
- `AVX_FOUND`: Whether AVX is supported
- `AVX_FLAGS`: Compiler flags to enable AVX
- `AVX_RUNTIME_FOUND`: Runtime AVX support
- `AVX_LEVEL`: Highest supported AVX level
- `AVX_FEATURES`: List of available AVX features

### 3. FMA.cmake
Detects Fused Multiply-Add support:
- **FMA**: FMA instructions

**Variables:**
- `FMA_FOUND`: Whether FMA is supported
- `FMA_FLAGS`: Compiler flags to enable FMA
- `FMA_RUNTIME_FOUND`: Runtime FMA support

### 4. SIMD.cmake (Unified Interface)
Provides a unified interface for all SIMD instruction sets:

**Variables:**
- `SIMD_FOUND`: Whether any SIMD is supported
- `SIMD_FLAGS`: Combined compiler flags
- `SIMD_LEVEL`: Overall SIMD level
- `SIMD_FEATURES`: List of all available features
- `SIMD_OPTIMIZATION_LEVEL`: Automatic optimization level

**Functions:**
- `target_enable_simd(TARGET_NAME)`: Enable SIMD for a target

### 5. CPUFeatures.cmake
Detects CPU characteristics and features:

**Variables:**
- `CPU_FEATURES_FOUND`: Whether CPU detection succeeded
- `CPU_VENDOR`: CPU vendor (Intel, AMD, etc.)
- `CPU_MODEL`: CPU model information
- `CPU_CORES`: Number of CPU cores
- `CPU_FEATURES`: List of CPU features

**Functions:**
- `target_enable_cpu_features(TARGET_NAME)`: Enable CPU features for a target

## Usage Examples

### Basic Usage

```cmake
# Find individual SIMD modules
find_package(SSE)
find_package(AVX)
find_package(FMA)

# Use the unified SIMD interface
find_package(SIMD)

# Find CPU features
find_package(CPUFeatures)
```

### Target Integration

```cmake
# Enable SIMD for a target
target_enable_simd(my_target)

# Enable CPU features for a target
target_enable_cpu_features(my_target)

# Manual SIMD configuration
if(SIMD_FOUND)
    target_compile_options(my_target PRIVATE ${SIMD_FLAGS})
    target_compile_definitions(my_target PRIVATE 
        SIMD_LEVEL="${SIMD_LEVEL}"
        SIMD_OPTIMIZATION_LEVEL="${SIMD_OPTIMIZATION_LEVEL}"
    )
endif()
```

### Conditional Compilation

```cmake
# Check for specific SIMD levels
if(SSE_LEVEL STREQUAL "SSE4_2")
    target_compile_definitions(my_target PRIVATE USE_SSE42)
endif()

if(AVX_LEVEL STREQUAL "AVX2")
    target_compile_definitions(my_target PRIVATE USE_AVX2)
endif()

if(FMA_FOUND)
    target_compile_definitions(my_target PRIVATE USE_FMA)
endif()
```

## Runtime Detection

All modules include runtime detection to ensure that the detected instruction sets are actually available on the target system, not just supported by the compiler.

## Automatic Optimization

The `SIMD.cmake` module automatically determines the optimal SIMD level based on available features:

1. **AVX-512**: If AVX512F/AVX512BW/AVX512VL are available
2. **AVX2**: If AVX2 is available
3. **AVX**: If AVX is available
4. **SSE4.2**: If SSE4.2 is available
5. **SSE4.1**: If SSE4.1 is available
6. **SSSE3**: If SSSE3 is available
7. **SSE3**: If SSE3 is available
8. **SSE2**: If SSE2 is available
9. **SSE**: If SSE is available
10. **NONE**: No SIMD support

## Integration with Existing Build System

These modules are designed to work seamlessly with the existing RXMesh build system. The main `CMakeLists.txt` already includes:

```cmake
find_package(SSE)
find_package(AVX)
find_package(FMA)
```

And uses the detected flags:

```cmake
string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(${PROJECT_NAME}_lib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${SIMD_FLAGS}>)
```

## Troubleshooting

### Common Issues

1. **Runtime detection fails**: This usually means the CPU doesn't support the instruction set at runtime, even if the compiler supports it.

2. **Compiler flags not found**: Ensure you're using a modern compiler (GCC 4.6+, Clang 3.0+, MSVC 2012+).

3. **Cross-compilation**: Runtime detection may not work in cross-compilation scenarios. Consider setting variables manually.

### Manual Override

You can manually override SIMD detection:

```cmake
set(SSE_FOUND TRUE CACHE BOOL "" FORCE)
set(SSE_FLAGS "-msse -msse2" CACHE STRING "" FORCE)
set(AVX_FOUND FALSE CACHE BOOL "" FORCE)
```

## Performance Considerations

- **SSE**: Good for basic vectorization, widely supported
- **AVX**: Better performance for 256-bit operations
- **AVX2**: Enhanced AVX with additional instructions
- **AVX-512**: Best performance but limited hardware support
- **FMA**: Can significantly improve floating-point performance

Choose the appropriate SIMD level based on your target hardware and performance requirements.

