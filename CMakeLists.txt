# ==============================================================================
# RXMesh - A GPU Mesh Data Structure
# ==============================================================================
#To run with cuDSS:
#>> cmake -DRX_USE_CUDSS=ON -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..

# ==============================================================================
# Project Configuration
# ==============================================================================
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(RXMesh
        VERSION 0.2.1
        LANGUAGES C CXX CUDA)

# Build options
set(RX_USE_POLYSCOPE "ON" CACHE BOOL "Enable Ployscope for visualization")
set(RX_BUILD_TESTS "ON" CACHE BOOL "Build RXMesh unit test")
set(RX_BUILD_APPS "ON" CACHE BOOL "Build RXMesh applications")
set(RX_USE_SUITESPARSE "OFF" CACHE BOOL "Use SuiteSparse for benchmark")
set(RX_USE_CUDSS "ON" CACHE BOOL "Use cuDSS - CUDA Library for Direct Sparse Solvers")

set(RX_WITH_DEV "ON" CACHE BOOL "Add DEV folder to build (for development purposes)")

message(STATUS "Polyscope is ${RX_USE_POLYSCOPE}")
message(STATUS "Build RXMesh unit test is ${RX_BUILD_TESTS}")
message(STATUS "Build RXMesh applications is ${RX_BUILD_APPS}")
message(STATUS "SuiteSparse is ${RX_USE_SUITESPARSE}")
message(STATUS "cuDSS is ${RX_USE_CUDSS}")

# Language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# ==============================================================================
# Build System Setup
# ==============================================================================
# Package Management
# TODO: Consider using CPM for the various libraries above
include(cmake/CPM.cmake)

# Auto-detect GPU architecture
include("cmake/AutoDetectCudaArch.cmake")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_LOCAL_CHANGES_STATUS)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_sha1.cpp.in" "${CMAKE_CURRENT_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp" @ONLY)

# Enable IDE folder support
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ==============================================================================
# RXMesh Library Target
# ==============================================================================
# RXMesh: could think of this as just the header library, so name RXMesh
file(GLOB_RECURSE RXMESH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
add_library(RXMesh INTERFACE)

target_sources(RXMesh
        INTERFACE ${RXMESH_SOURCES}
)

# ==============================================================================
# External Dependencies
# ==============================================================================
# METIS
#set(CMAKE_DISABLE_FIND_PACKAGE_METIS TRUE)
include("cmake/recipes/metis.cmake")
target_link_libraries(RXMesh INTERFACE metis)

# GoogleTest
include(cmake/recipes/googletest.cmake)
include(GoogleTest)

# RapidJSON
include(cmake/recipes/rapidjson.cmake)
target_include_directories(${PROJECT_NAME} INTERFACE "${rapidjson_SOURCE_DIR}/include")

# spdlog
include(cmake/recipes/spdlog.cmake)
target_include_directories(${PROJECT_NAME} INTERFACE "${spdlog_SOURCE_DIR}/include")

# OpenMesh
include(cmake/recipes/openmesh.cmake)
target_compile_definitions(${PROJECT_NAME} INTERFACE -DNO_DECREMENT_DEPRECATED_WARNINGS)

# GLM (note: changed from PRIVATE to PUBLIC based on your CMakeLists.txt)
include(cmake/recipes/glm.cmake)
target_link_libraries(${PROJECT_NAME} INTERFACE glm::glm)
target_compile_definitions(${PROJECT_NAME} INTERFACE GLM_ENABLE_EXPERIMENTAL)

# Cereal
include(cmake/recipes/cereal.cmake)
target_include_directories(${PROJECT_NAME} INTERFACE "${cereal_SOURCE_DIR}/include")

# Polyscope (optional)
if(${RX_USE_POLYSCOPE})
    include(cmake/recipes/polyscope.cmake)
    target_include_directories(${PROJECT_NAME} INTERFACE "${cereal_SOURCE_DIR}/include")
    target_link_libraries(RXMesh INTERFACE polyscope)
endif()

# Compile features and definitions
target_compile_features(RXMesh INTERFACE cxx_std_17)
target_compile_definitions(RXMesh
        INTERFACE INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/input/
        INTERFACE OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/output/
)
if(${RX_USE_POLYSCOPE})
    target_compile_definitions(RXMesh INTERFACE USE_POLYSCOPE)
endif()
target_include_directories(RXMesh
        INTERFACE "include"
        INTERFACE "${rapidjson_SOURCE_DIR}/include"
        INTERFACE "${spdlog_SOURCE_DIR}/include"
        INTERFACE "${cereal_SOURCE_DIR}/include"
        INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ==============================================================================
# Compiler Configuration
# ==============================================================================
# C++ compiler flags
set(cxx_flags
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>  #Add MSVC-specific compiler flags here
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-Wall -m64 -fopenmp -O3 -std=c++20 -Wno-unused-function>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Wall -m64 -fopenmp -O0 -g3 -std=c++20 -Wno-unused-function>
)

# CUDA compiler flags
set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /Zi")

set(cuda_flags
        #Windows stuff
        -Xcompiler=$<$<CXX_COMPILER_ID:MSVC>:${MSVC_XCOMPILER_FLAGS}>
        # Release
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:
        -Xcompiler=-rdynamic,-Wall,-fopenmp,-O3,-Wno-unused-function
        -O3
        >
        # Debug
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:
        -G
        -O0
        -Xptxas=-O0
        -lineinfo
        -Xcompiler=-rdynamic,-Wall,-fopenmp,-O0,-g3,-Wno-unused-function
        >
        # other common CUDA flags here (that are safe in both configs)
        -rdc=true
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -warn-spills -res-usage
        --ptxas-options=-v
)

target_compile_options(RXMesh INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3")
endif()

# ==============================================================================
# Required Libraries
# ==============================================================================

# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh INTERFACE OpenMP::OpenMP_CXX)
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(RXMesh INTERFACE CUDA::cusparse)
target_link_libraries(RXMesh INTERFACE CUDA::cusolver)

# Eigen
include("cmake/recipes/eigen.cmake")
target_link_libraries(RXMesh INTERFACE Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(RXMesh INTERFACE "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

# ==============================================================================
# Optional Libraries
# ==============================================================================
# SuiteSparse
if(${RX_USE_SUITESPARSE})
    include("cmake/recipes/suitesparse.cmake")
    target_compile_definitions(RXMesh INTERFACE USE_SUITESPARSE)
endif()

# cuDSS
if(${RX_USE_CUDSS})
    find_package(cudss QUIET)
    if (cudss_FOUND)
        message(STATUS "Found cuDSS version ${cudss_VERSION}")

        # Try to use modern CMake target first, fallback to manual linking
        if(TARGET cudss::cudss)
            target_link_libraries(RXMesh INTERFACE cudss::cudss)
        else()
            # Manual linking for older cuDSS installations
            target_link_directories(RXMesh INTERFACE ${cudss_LIBRARY_DIR})
            
            if (WIN32)
                target_include_directories(RXMesh INTERFACE ${cudss_INCLUDE_DIR})
                target_link_libraries(RXMesh INTERFACE cudss)
            else()
                target_link_libraries(RXMesh INTERFACE cudss_static)
            endif()
        endif()

        # Handle Windows DLL copying for runtime
        if (WIN32)
            add_custom_target(CopyCUDSSDLL)
            foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                add_custom_command(
                        OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
                        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${cudss_BINARY_DIR}/cudss64_0.dll"
                        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
                        DEPENDS "${cudss_BINARY_DIR}/cudss64_0.dll"
                        COMMENT "Copying cudss64_0.dll for ${CONFIG} configuration"
                        VERBATIM
                )
                add_custom_target("CopyCUDSSDLL${CONFIG}"
                        DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll")
                add_dependencies(CopyCUDSSDLL "CopyCUDSSDLL${CONFIG}")
            endforeach()
        endif()

        target_compile_definitions(RXMesh INTERFACE USE_CUDSS)
    else()
        message(WARNING "cuDSS not found, disabling cuDSS support")
        set(RX_USE_CUDSS "OFF" CACHE BOOL "" FORCE)
    endif()
endif()

# ==============================================================================
# Subdirectories
## ==============================================================================
if(${RX_BUILD_TESTS})
   add_subdirectory(tests)
endif()

if(${RX_BUILD_APPS})
   add_subdirectory(apps)
endif()

if(${RX_WITH_DEV})
   add_subdirectory(dev)
endif()