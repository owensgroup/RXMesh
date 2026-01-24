include_guard(GLOBAL)

# When RXMesh is added via FetchContent, this file can be included from a
# different project. Anchor paths to the RXMesh source tree.
if(NOT DEFINED RXMESH_SOURCE_DIR)
    get_filename_component(RXMESH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

# RXMesh: could think of this as just the header library, so name RXMesh
file(GLOB_RECURSE RXMESH_SOURCES "${RXMESH_SOURCE_DIR}/include/*.*")
add_library(RXMesh INTERFACE)

target_sources(RXMesh
    INTERFACE ${RXMESH_SOURCES}
)


target_compile_features(RXMesh INTERFACE cxx_std_17)
target_compile_definitions(RXMesh
    INTERFACE INPUT_DIR=${RXMESH_SOURCE_DIR}/input/
    INTERFACE OUTPUT_DIR=${RXMESH_SOURCE_DIR}/output/
)

if(${RX_USE_POLYSCOPE})
    target_compile_definitions(RXMesh INTERFACE USE_POLYSCOPE)
endif()

target_include_directories(RXMesh
    INTERFACE "${RXMESH_SOURCE_DIR}/include"
    INTERFACE "${rapidjson_SOURCE_DIR}/include"
    INTERFACE "${spdlog_SOURCE_DIR}/include"
    INTERFACE "${cereal_SOURCE_DIR}/include"
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(RXMesh INTERFACE cuBQL cuBQL_queries)

# CUDA and C++ compiler flags
set(cxx_flags
    $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>  # Add MSVC-specific compiler flags here
    $<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> # Add GCC/Clang-specific compiler flags here
)

set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /Zi")
set(cuda_flags
    -Xcompiler=$<$<CXX_COMPILER_ID:GNU>:-rdynamic -Wall -fopenmp -O3 -Wno-unused-function>
    -Xcompiler=$<$<CXX_COMPILER_ID:MSVC>:${MSVC_XCOMPILER_FLAGS}>
    # Disables warning
    # 177-D "function XXX was declared but never referenced"
    # 174-D "expression has no effect"
    # 20054-D "dynamic initialization is not supported for a function-scope static __shared__ variable within a __device__/__global__ function"
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

#SuiteSparse
if(${RX_USE_SUITESPARSE})
	include("${CMAKE_CURRENT_LIST_DIR}/suitesparse.cmake")
	target_compile_definitions(RXMesh INTERFACE USE_SUITESPARSE)
endif()
target_link_libraries(RXMesh INTERFACE glm::glm)
if(${RX_USE_POLYSCOPE})
    include("${CMAKE_CURRENT_LIST_DIR}/polyscope.cmake")
	target_link_libraries(RXMesh INTERFACE polyscope)	
endif()

#METIS
target_link_libraries(RXMesh INTERFACE metis)	

#OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh INTERFACE OpenMP::OpenMP_CXX)
endif()


# cuDSS
if(${RX_USE_CUDSS})
    include("${CMAKE_CURRENT_LIST_DIR}/cuDSS.cmake")
    rxmesh_enable_cudss(RXMesh)
endif()

#cuSolver and cuSparse
find_package(CUDAToolkit REQUIRED)
target_link_libraries(RXMesh INTERFACE CUDA::cusparse)
target_link_libraries(RXMesh INTERFACE CUDA::cusolver)


#Eigen
include("${CMAKE_CURRENT_LIST_DIR}/eigen.cmake")
target_link_libraries(RXMesh INTERFACE Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(RXMesh INTERFACE "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")