#To run with cuDSS:
#>> cmake -DRX_USE_CUDSS=ON -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(RXMesh 
        VERSION 0.2.1 
        LANGUAGES C CXX CUDA)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")


set(RX_USE_POLYSCOPE "ON" CACHE BOOL "Enable Ployscope for visualization")
set(RX_BUILD_TESTS "ON" CACHE BOOL "Build RXMesh unit test")
set(RX_BUILD_APPS "ON" CACHE BOOL "Build RXMesh applications")
set(RX_USE_SUITESPARSE "OFF" CACHE BOOL "Use SuiteSparse for benchmark")
set(RX_USE_CUDSS "OFF" CACHE BOOL "Use cuDSS - CUDA Library for Direct Sparse Solvers")

message(STATUS "Polyscope is ${RX_USE_POLYSCOPE}")
message(STATUS "Build RXMesh unit test is ${RX_BUILD_TESTS}")
message(STATUS "Build RXMesh applications is ${RX_BUILD_APPS}")
message(STATUS "SuiteSparse is ${RX_USE_SUITESPARSE}")
message(STATUS "cuDSS is ${RX_USE_CUDSS}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()


# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

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


if(${RX_BUILD_TESTS})
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/googletest.cmake")
    include(GoogleTest)
    add_subdirectory(tests)
endif()

if(${RX_BUILD_APPS})
	add_subdirectory(apps)
endif()

