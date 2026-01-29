include_guard(GLOBAL)

# When RXMesh is added via FetchContent, this file can be included from a
# different project. Anchor paths to the RXMesh source tree.
if(NOT DEFINED RXMESH_SOURCE_DIR)
    get_filename_component(RXMESH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

# RXMesh compiled library
set(RXMESH_LIBRARY_SOURCES
    "${RXMESH_SOURCE_DIR}/include/rxmesh/rxmesh.cpp"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/rxmesh_static.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/rxmesh_dynamic.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/query.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/attribute.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patch_stash.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patch_scheduler.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patcher/patcher.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/util/git_sha1.cpp"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/util/MshLoader.cpp"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/util/MshSaver.cpp"
)

file(GLOB_RECURSE RXMESH_LIBRARY_HEADERS CONFIGURE_DEPENDS
    "${RXMESH_SOURCE_DIR}/include/rxmesh/*.h"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/*.hpp"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/*.cuh"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/*.inl"
)

add_library(RXMesh STATIC ${RXMESH_LIBRARY_SOURCES} ${RXMESH_LIBRARY_HEADERS})

source_group(
    TREE ${RXMESH_SOURCE_DIR}
    FILES ${RXMESH_LIBRARY_SOURCES} ${RXMESH_LIBRARY_HEADERS}
)

# Required for targets that compile CUDA sources.
set_property(TARGET RXMesh PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(RXMesh PUBLIC cxx_std_17)
target_compile_definitions(RXMesh
    PUBLIC INPUT_DIR=${RXMESH_SOURCE_DIR}/input/
    PUBLIC OUTPUT_DIR=${RXMESH_SOURCE_DIR}/output/
)

if(${RX_USE_POLYSCOPE})
    target_compile_definitions(RXMesh PUBLIC USE_POLYSCOPE)
endif()

target_include_directories(RXMesh
    PUBLIC "${RXMESH_SOURCE_DIR}/include"
    PUBLIC "${rapidjson_SOURCE_DIR}/include"
    PUBLIC "${spdlog_SOURCE_DIR}/include"
    PUBLIC "${cereal_SOURCE_DIR}/include"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(RXMesh PUBLIC cuBQL cuBQL_queries)

# CUDA and C++ compiler flags
set(cxx_flags
    $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>  # Add MSVC-specific compiler flags here
    $<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17 -Wno-unused-function> # Add GCC/Clang-specific compiler flags here
)

set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /bigobj /Zi")
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

target_compile_options(RXMesh PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

#SuiteSparse
if(${RX_USE_SUITESPARSE})
	include("${CMAKE_CURRENT_LIST_DIR}/suitesparse.cmake")
	target_compile_definitions(RXMesh PUBLIC USE_SUITESPARSE)
endif()
target_link_libraries(RXMesh PUBLIC glm::glm)
if(${RX_USE_POLYSCOPE})
    include("${CMAKE_CURRENT_LIST_DIR}/polyscope.cmake")
	target_link_libraries(RXMesh PUBLIC polyscope)
endif()

#METIS
target_link_libraries(RXMesh PUBLIC metis)

#OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(RXMesh PUBLIC OpenMP::OpenMP_CXX)
endif()


# cuDSS
if(${RX_USE_CUDSS})
    include("${CMAKE_CURRENT_LIST_DIR}/cuDSS.cmake")
    rxmesh_enable_cudss(RXMesh)
endif()

#cuSolver and cuSparse
find_package(CUDAToolkit REQUIRED)
target_link_libraries(RXMesh PUBLIC CUDA::cusparse)
target_link_libraries(RXMesh PUBLIC CUDA::cusolver)


#Eigen
include("${CMAKE_CURRENT_LIST_DIR}/eigen.cmake")
target_link_libraries(RXMesh PUBLIC Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(RXMesh PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")