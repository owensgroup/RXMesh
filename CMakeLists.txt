#To run with cuDSS:
#>> cmake -DRX_USE_CUDSS=ON -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# USE_HIP builds the GPU sources with the HIP toolchain (AMD ROCm). The CUDA path
# is unchanged when USE_HIP is OFF. The HIP arch comes from
# CMAKE_HIP_ARCHITECTURES (defaulted only when unset); never hardcode a literal.
option(USE_HIP "Build the GPU sources with HIP for AMD GPUs" OFF)

if(USE_HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
        set(CMAKE_HIP_ARCHITECTURES "gfx90a")
    endif()
else()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

#So we can use FetchContent_MakeAvailable without warnings
cmake_policy(SET CMP0169 OLD)

if(USE_HIP)
    project(RXMesh
            VERSION 0.2.1
            LANGUAGES C CXX HIP)
    message(STATUS "RXMesh: HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    project(RXMesh
            VERSION 0.2.1
            LANGUAGES C CXX CUDA)
    message(STATUS "RXMesh: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()


set(RX_USE_POLYSCOPE "ON" CACHE BOOL "Enable Ployscope for visualization")
set(RX_BUILD_TESTS "ON" CACHE BOOL "Build RXMesh unit test")
set(RX_BUILD_APPS "ON" CACHE BOOL "Build RXMesh applications")
set(RX_USE_SUITESPARSE "OFF" CACHE BOOL "Use SuiteSparse for benchmark")
set(RX_USE_CUDSS "OFF" CACHE BOOL "Use cuDSS - CUDA Library for Direct Sparse Solvers")
set(RX_USE_DOUBLE "OFF" CACHE BOOL "Use double precision for reading input vertex coordinates")

message(STATUS "RXMesh: Polyscope is ${RX_USE_POLYSCOPE}")
message(STATUS "RXMesh: Build RXMesh unit test is ${RX_BUILD_TESTS}")
message(STATUS "RXMesh: Build RXMesh applications is ${RX_BUILD_APPS}")
message(STATUS "RXMesh: SuiteSparse is ${RX_USE_SUITESPARSE}")
message(STATUS "RXMesh: cuDSS is ${RX_USE_CUDSS}")
message(STATUS "RXMesh: Double precision is ${RX_USE_DOUBLE}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()


# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Allow finding local CMake modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Stable anchor for internal includes when RXMesh is used as a subproject.
set(RXMESH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Third party dependencies 
include(cmake/ThirdParty.cmake)

# Package Management
include(cmake/CPM.cmake)

# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232
include(cmake/GitVersion.cmake)


# Enable IDE folder support
set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

# RXMesh target definition and core wiring.
include(cmake/RXMeshTarget.cmake)
include(cmake/RXMeshInstall.cmake)


if(${RX_BUILD_TESTS})
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/googletest.cmake")
    include(GoogleTest)
    add_subdirectory(tests)
endif()

if(${RX_BUILD_APPS})
	add_subdirectory(apps)
endif()

