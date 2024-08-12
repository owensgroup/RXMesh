cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(RXMesh 
        VERSION 0.2.1 
        LANGUAGES C CXX CUDA)


set(USE_POLYSCOPE "ON" CACHE BOOL "Enable Ployscope for visualization")

if(${USE_POLYSCOPE})
	message(STATUS "Polyscope is enabled")
else()
	message(STATUS "Polyscope is disabled")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# glm
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        master
)
FetchContent_Populate(glm)
add_subdirectory(${glm_SOURCE_DIR})

# polyscope
if(USE_POLYSCOPE)
FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        6ec777513d802654801fe77e6c3a54fb9dbb6b1f
)
FetchContent_MakeAvailable(polyscope)
endif()

# cereal
FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
)
FetchContent_Populate(cereal)

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

# RXMesh: could think of this as just the header library, so name RXMesh
file(GLOB_RECURSE RXMESH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
add_library(RXMesh INTERFACE) 

target_sources(RXMesh
    INTERFACE ${RXMESH_SOURCES}
)

# shows headers only once 
#add_custom_target(RXMeshSrc 
#	SOURCES ${RXMESH_SOURCES}
#) 

target_compile_features(RXMesh INTERFACE cxx_std_17)
target_compile_definitions(RXMesh
    INTERFACE INPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/input/
    INTERFACE OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/output/
)
if (USE_POLYSCOPE)
	target_compile_definitions(RXMesh INTERFACE USE_POLYSCOPE)
endif()
target_include_directories(RXMesh 
    INTERFACE "include"
	INTERFACE "${rapidjson_SOURCE_DIR}/include"
	INTERFACE "${spdlog_SOURCE_DIR}/include"
	INTERFACE "${cereal_SOURCE_DIR}/include"
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# CUDA and C++ compiler flags
set(cxx_flags 
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17>  #Add MSVC-specific compiler flags here
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> #Add GCC/Clang-specific compiler flags here
	)

set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /Zi")
set(cuda_flags
    -Xcompiler=$<$<CXX_COMPILER_ID:GNU>:-Wall -fopenmp -O3 -Wno-unused-function>
	-Xcompiler=$<$<CXX_COMPILER_ID:MSVC>:${MSVC_XCOMPILER_FLAGS}>
    #Disables warning
    #177-D "function XXX was declared but never referenced"
    -Xcudafe "--display_error_number --diag_suppress=177"    
    -rdc=true
	-lineinfo	
	--expt-extended-lambda	
	-use_fast_math	
    $<$<CXX_COMPILER_ID:GNU>:-O3>
	--expt-relaxed-constexpr	
    -Xptxas -warn-spills -res-usage   
	#-Xptxas -dlcm=cg -dscm=cg
	--ptxas-options=-v	
)


target_compile_options(RXMesh INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

target_link_libraries(RXMesh INTERFACE glm::glm)
if (USE_POLYSCOPE)
	target_link_libraries(RXMesh INTERFACE polyscope)	
endif()

#OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh INTERFACE OpenMP::OpenMP_CXX)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(RXMesh INTERFACE CUDA::cusparse)
target_link_libraries(RXMesh INTERFACE CUDA::cusolver)

include(GoogleTest)
add_subdirectory(apps)
add_subdirectory(tests)
