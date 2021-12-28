cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 OLD)
endif()

project(RXMesh 
        VERSION 0.1.0 
        LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()


include(FetchContent)
# GoogleTest
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        eaf9a3fd77869cf95befb87455a2e2a2e85044ff
)
FetchContent_MakeAvailable(googletest)

# rapidjson
FetchContent_Declare(rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        8f4c021fa2f1e001d2376095928fc0532adf2ae6
)
FetchContent_Populate(rapidjson)

# OpenMesh
set(OPENMESH_DOCS false CACHE BOOL "Enable or disable building of documentation")
FetchContent_Declare(openmesh
    GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
    GIT_TAG        OpenMesh-8.1
)
FetchContent_MakeAvailable(openmesh)
add_definitions(-DNO_DECREMENT_DEPRECATED_WARNINGS)

# spdlog
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.8.5
)
FetchContent_Populate(spdlog)

# Auto-detect GPU architecture, sets ${CUDA_ARCHS}
include("cmake/AutoDetectCudaArch.cmake")

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Get and store git sha1 https://stackoverflow.com/a/4318642/1608232
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_sha1.cpp.in" "${CMAKE_CURRENT_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp" @ONLY)


# Enable IDE folder support
set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

# RXMesh: could think of this as just the header library, so name RXMesh_header_lib
add_library(RXMesh_header_lib INTERFACE) 
target_compile_features(RXMesh_header_lib INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_definitions(RXMesh_header_lib
    INTERFACE INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/input/
    INTERFACE OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/output/
)
target_include_directories( RXMesh_header_lib 
    INTERFACE "include"
	INTERFACE "${rapidjson_SOURCE_DIR}/include"
	INTERFACE "${spdlog_SOURCE_DIR}/include"
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_sources(RXMesh_header_lib
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/include/rxmesh/util/git_sha1.h"
)

# CUDA and C++ compiler flags
set(cxx_flags 
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp /std:c++17>           #Add MSVC-specific compiler flags here
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> #Add GCC/Clang-specific compiler flags here
	)
set(cuda_flags
    -Xcompiler=$<$<CXX_COMPILER_ID:GNU>:-Wall -fopenmp -O3 -Wno-unused-function>
    #Disables warning
    #177-D "function XXX was declared but never referenced"
    -Xcudafe "--display_error_number --diag_suppress=177"
    ${CUDA_ARCHS}
    -rdc=true
	-lineinfo	
	--expt-extended-lambda	
	-use_fast_math	
    $<$<CXX_COMPILER_ID:GNU>:-O3>
	--expt-relaxed-constexpr	
    #-Xptxas -warn-spills -res-usage    
    #-G    
)


add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

target_link_libraries(RXMesh_header_lib INTERFACE $<BUILD_INTERFACE:developer_flags>)

#OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh_header_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

add_subdirectory("include")



include(GoogleTest)
add_subdirectory(apps)
add_subdirectory(tests)