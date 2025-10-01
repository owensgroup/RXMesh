# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.20")
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_ROOT "/usr/local/cuda")
project(RXMesh VERSION "0.2.1" DESCRIPTION "A CUDA-based mesh processing library" LANGUAGES CXX CUDA)

# Enable CUDA language
enable_language(CUDA)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

################################################################################
# Project options
################################################################################

option(${PROJECT_NAME}_WITH_APPS "Add demo applications" ON)
option(${PROJECT_NAME}_WITH_PARTH "Use Parth for fill-reducing orderings" ON)
option(${PROJECT_NAME}_WITH_TESTS "Add unit tests" ON)
option(${PROJECT_NAME}_WITH_POLYSCOPE "Enable Polyscope for visualization" ON)
option(${PROJECT_NAME}_WITH_SUITESPARSE "Use SuiteSparse for benchmark" ON)
option(${PROJECT_NAME}_WITH_CUDSS "Use cuDSS - CUDA Library for Direct Sparse Solvers" ON)
option(${PROJECT_NAME}_WITH_METIS "Use METIS for graph partitioning" ON)
option(${PROJECT_NAME}_WITH_DEV "Add dev folder" ON)

# GPU Architecture options - MANUAL CONFIGURATION REQUIRED
set(GPU_ARCHITECTURE "AMPERE" CACHE STRING "Target GPU architecture (AMPERE, HOPPER, ADA)")
set_property(CACHE GPU_ARCHITECTURE PROPERTY STRINGS AMPERE HOPPER ADA)

# GPU Compute Capability - MANUAL CONFIGURATION REQUIRED
set(GPU_COMPUTE_CAPABILITY "80" CACHE STRING "Target GPU compute capability (80, 86, 87, 89, 90)")
set_property(CACHE GPU_COMPUTE_CAPABILITY PROPERTY STRINGS 80 86 87 89 90)

################################################################################
# Configuration
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/find)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()


# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Get and store git sha1
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_LOCAL_CHANGES_STATUS)

# Generate git_sha1.cpp from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_sha1.cpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp"
    @ONLY
)

################################################################################
# Project source files
################################################################################

# Core library source files
file(GLOB_RECURSE RXMESH_SRC_FILES
    "include/*.cpp"
    "include/*/*.cpp"
    "include/*.cu"
    "include/*/*.cu"
)

message(STATUS "Adding RXMesh source files...")
add_library(${PROJECT_NAME}_lib ${RXMESH_SRC_FILES})

# Enable CUDA separable compilation for the library
set_property(TARGET ${PROJECT_NAME}_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Adding include directories...")
target_include_directories(${PROJECT_NAME}_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


################################################################################
# Compiler options
################################################################################

# CUDA setup
message(STATUS "Configuring CUDA...")
find_package(CUDAToolkit REQUIRED)

# Check CUDA version
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "CUDA 11.0 or higher is required. Found: ${CUDAToolkit_VERSION}")
endif()

# Manual GPU Architecture Configuration
message(STATUS "Manual GPU architecture configuration:")
message(STATUS "  GPU_ARCHITECTURE: ${GPU_ARCHITECTURE}")
message(STATUS "  GPU_COMPUTE_CAPABILITY: ${GPU_COMPUTE_CAPABILITY}")

# Validate GPU architecture and compute capability combination
if(GPU_ARCHITECTURE STREQUAL "HOPPER")
    if(NOT GPU_COMPUTE_CAPABILITY STREQUAL "90")
        message(FATAL_ERROR "Hopper architecture requires compute capability 90. Current: ${GPU_COMPUTE_CAPABILITY}")
    endif()
    set(CUDA_ARCHITECTURES "90a") # SM90a for Hopper
elseif(GPU_ARCHITECTURE STREQUAL "AMPERE")
    if(NOT (GPU_COMPUTE_CAPABILITY STREQUAL "80" OR
            GPU_COMPUTE_CAPABILITY STREQUAL "86" OR
            GPU_COMPUTE_CAPABILITY STREQUAL "87" OR
            GPU_COMPUTE_CAPABILITY STREQUAL "89"))
        message(FATAL_ERROR "Ampere architecture requires compute capability 80, 86, 87, or 89. Current: ${GPU_COMPUTE_CAPABILITY}")
    endif()
    set(CUDA_ARCHITECTURES "${GPU_COMPUTE_CAPABILITY}")
elseif(GPU_ARCHITECTURE STREQUAL "ADA")
    if(NOT (GPU_COMPUTE_CAPABILITY STREQUAL "89"))
        message(FATAL_ERROR "Ada architecture requires compute capability 89. Current: ${GPU_COMPUTE_CAPABILITY}")
    endif()
    set(CUDA_ARCHITECTURES "${GPU_COMPUTE_CAPABILITY}")
else()
    message(FATAL_ERROR "Unsupported GPU architecture: ${GPU_ARCHITECTURE}. Supported: AMPERE, HOPPER, ADA")
endif()

# Set architecture specific compile definitions and properties
if(GPU_ARCHITECTURE STREQUAL "HOPPER")
    message(STATUS "Configuring for Hopper architecture with Tensor Memory Accelerator (TMA)")
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_GPU_ARCH_HOPPER)

    set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    )
elseif(GPU_ARCHITECTURE STREQUAL "AMPERE")
    message(STATUS "Configuring for Ampere architecture (CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES})")
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_GPU_ARCH_AMPERE)

    set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    )
elseif(GPU_ARCHITECTURE STREQUAL "ADA")
    message(STATUS "Configuring for Ada architecture (CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES})")
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_GPU_ARCH_ADA)

    set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
    )
endif()

# CUDA compiler flags
target_compile_options(${PROJECT_NAME}_lib PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        -O3
        --use_fast_math
        --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
        -rdc=true
        --expt-extended-lambda
        --expt-relaxed-constexpr
    >
)

# Debug mode CUDA flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME}_lib PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>: -G -src-in-ptx>
    )
endif()

# C++ compiler flags
target_compile_options(${PROJECT_NAME}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -m64 -fopenmp -O3 -std=c++20>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -m64 -fopenmp -O3 -std=c++20>
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++20 /bigobj>
    >
)

# Use C++20 for CXX files
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add -pthread to compilation and linking
find_package(Threads)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# SIMD options
message(STATUS "Searching for SIMD optimizations...")
find_package(SSE)
find_package(AVX)
find_package(FMA)
string(REPLACE " " ";" SIMD_FLAGS "${SSE_FLAGS} ${AVX_FLAGS} ${FMA_FLAGS}")
target_compile_options(${PROJECT_NAME}_lib PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${SIMD_FLAGS}>)


################################################################################
# Required Libraries
################################################################################

# Include recipe files for external dependencies
include(cmake/recipes/googletest.cmake)
include(cmake/recipes/rapidjson.cmake)
include(cmake/recipes/openmesh.cmake)
include(cmake/recipes/spdlog.cmake)
include(cmake/recipes/glm.cmake)
include(cmake/recipes/cereal.cmake)
include(cmake/recipes/metis.cmake)

# Link core libraries
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
    glm::glm
        OpenMeshCore
    OpenMeshTools
)

# Enable GLM experimental extensions globally
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC GLM_ENABLE_EXPERIMENTAL)

# Add additional include directories for header-only libraries
target_include_directories(${PROJECT_NAME}_lib PUBLIC "${rapidjson_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME}_lib PUBLIC "${spdlog_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME}_lib PUBLIC "${cereal_SOURCE_DIR}/include")

# Polyscope (optional)
if(${PROJECT_NAME}_WITH_POLYSCOPE)
    include(cmake/recipes/polyscope.cmake)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC polyscope::polyscope)
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC USE_POLYSCOPE)
endif()

# METIS
if (${PROJECT_NAME}_WITH_METIS)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC metis)
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_WITH_METIS)
endif ()


# SuiteSparse
if (${PROJECT_NAME}_WITH_SUITESPARSE)
    set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITESPARSE_INC})
    set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITESPARSE_LIB})
    find_package(SuiteSparse REQUIRED)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${SUITESPARSE_LIBRARIES})
    target_include_directories(${PROJECT_NAME}_lib PUBLIC ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_WITH_SUITESPARSE)
endif ()

# cuDSS (optional)
if(${PROJECT_NAME}_WITH_CUDSS)
    find_package(cudss REQUIRED)
    if(cudss_FOUND)
        message(STATUS "Found cuDSS version ${cudss_VERSION}")
        target_link_libraries(${PROJECT_NAME}_lib PUBLIC cudss::cudss)
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_WITH_CUDSS)
    else()
        message(WARNING "cuDSS not found, disabling cuDSS support")
        set(${PROJECT_NAME}_WITH_CUDSS OFF CACHE BOOL "" FORCE)
    endif()
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA libraries
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
    CUDA::cusparse
    CUDA::cusolver
    CUDA::cublas
)

# Eigen
include(cmake/recipes/eigen.cmake)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC Eigen3::Eigen)
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

# Additional compile definitions
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/input/
    OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/output/
)

# Build tests and apps based on options
if(${PROJECT_NAME}_WITH_PARTH)
    include(cmake/recipes/parth.cmake)
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC Parth::parth)
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC RXMESH_WITH_PARTH)
endif()

if(${PROJECT_NAME}_WITH_APPS)
	add_subdirectory(apps)
endif()

if(${PROJECT_NAME}_WITH_DEV)
    add_subdirectory(dev)
endif()

