cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(RXMesh 
        VERSION 0.2.1 
        LANGUAGES C CXX CUDA)


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
add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})

# polyscope
if(${RX_USE_POLYSCOPE})
FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        30f3995f51a895f0113d3580a889077bdb846e7a
)
FetchContent_MakeAvailable(polyscope)
endif()

# cereal
FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
)
FetchContent_Populate(cereal)

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

# RXMesh: could think of this as just the header library, so name RXMesh
file(GLOB_RECURSE RXMESH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
add_library(RXMesh INTERFACE) 

target_sources(RXMesh
    INTERFACE ${RXMESH_SOURCES}
)


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

# CUDA and C++ compiler flags
set(cxx_flags 
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>  #Add MSVC-specific compiler flags here
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> #Add GCC/Clang-specific compiler flags here
	)

set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /Zi")
set(cuda_flags
    -Xcompiler=$<$<CXX_COMPILER_ID:GNU>:-rdynamic -Wall -fopenmp -O3 -Wno-unused-function>
	-Xcompiler=$<$<CXX_COMPILER_ID:MSVC>:${MSVC_XCOMPILER_FLAGS}>
    #Disables warning
    #177-D "function XXX was declared but never referenced"
    #174-D "expression has no effect"
    #20054-D "dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function"
    -Xcudafe "--display_error_number --diag_suppress=177 --diag_suppress=174 --diag_suppress=20054"    
    -rdc=true
	-lineinfo	
	--expt-extended-lambda	
	#-use_fast_math	
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


#METIS
include("cmake/metis.cmake")
target_link_libraries(RXMesh INTERFACE metis)	

#SuiteSparse
if(${RX_USE_SUITESPARSE})
	include("cmake/suitesparse.cmake")
	target_compile_definitions(RXMesh INTERFACE USE_SUITESPARSE)
endif()
target_link_libraries(RXMesh INTERFACE glm::glm)
if(${RX_USE_POLYSCOPE})
	target_link_libraries(RXMesh INTERFACE polyscope)	
endif()

#OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh INTERFACE OpenMP::OpenMP_CXX)
endif()

#cuDSS
if(${RX_USE_CUDSS})
	find_package(cudss CONFIG QUIET)
	if (cudss_FOUND)		
		message(STATUS "Found cuDSS version ${cudss_VERSION}")		
		
		target_include_directories(RXMesh INTERFACE PUBLIC ${cudss_INCLUDE_DIR})		
		target_link_directories(RXMesh INTERFACE ${cudss_LIBRARY_DIR})
		target_link_libraries(RXMesh INTERFACE cudss)
		
		target_compile_definitions(RXMesh INTERFACE USE_CUDSS)
	else (cudss_FOUND)
		message(WARNING "Can not find cuDSS")	
		set(RX_USE_CUDSS "OFF" CACHE BOOL "")		
	endif()		
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(RXMesh INTERFACE CUDA::cusparse)
target_link_libraries(RXMesh INTERFACE CUDA::cusolver)

#Eigen
include("cmake/eigen.cmake")
target_link_libraries(RXMesh INTERFACE Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(RXMesh INTERFACE "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

include(GoogleTest)	

if(${RX_BUILD_TESTS})	
	add_subdirectory(tests)
endif()

if(${RX_BUILD_APPS})
	add_subdirectory(apps)
endif()

