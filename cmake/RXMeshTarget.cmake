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
    "${RXMESH_SOURCE_DIR}/include/rxmesh/hash_functions.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/lp_hashtable.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patch_info.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patch_lock.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/patch_scheduler.cu"
    "${RXMESH_SOURCE_DIR}/include/rxmesh/reduce_handle.cu"
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

if(USE_HIP)
    # Compile the .cu translation units with the HIP toolchain; host C++ is
    # untouched. Keeps the NVIDIA build intact and the diff minimal.
    foreach(src ${RXMESH_LIBRARY_SOURCES})
        if(src MATCHES "\\.cu$")
            set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
    set_target_properties(RXMesh PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_definitions(RXMesh PUBLIC USE_HIP)
    # RXMesh declares __device__ members in headers and defines them in separate
    # .cu units; that needs relocatable device code so the device linker resolves
    # them across TUs (the CUDA path uses CUDA_SEPARABLE_COMPILATION / -rdc=true).
    # -fgpu-rdc must reach consumers too, so apply it as a PUBLIC compile/link opt.
    set_target_properties(RXMesh PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    target_compile_options(RXMesh PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
    target_link_options(RXMesh PUBLIC $<$<LINK_LANGUAGE:HIP>:-fgpu-rdc> --hip-link)
    # On Windows, CMake's Windows-Clang platform module appends -fuse-ld=lld-link
    # which clang++ (gcc-driver mode) rejects under -fgpu-rdc HIP device link.
    # Also, the HIP cooperative-groups header defines this_cluster() as a
    # non-inline __device__ function; under -fgpu-rdc device-link it appears as a
    # duplicate strong symbol.  Both issues are Windows-only (Linux ELF uses
    # COMDAT for inline __device__ headers; the -fuse-ld= is Linux-absent).
    if(WIN32)
        target_link_options(RXMesh PUBLIC
            $<$<LINK_LANGUAGE:HIP>:-fuse-ld=>
            $<$<LINK_LANGUAGE:HIP>:-Xoffload-linker --allow-multiple-definition>)
    endif()
else()
    # Required for targets that compile CUDA sources.
    set_property(TARGET RXMesh PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
set_property(TARGET RXMesh PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_features(RXMesh PUBLIC cxx_std_17)
target_compile_definitions(RXMesh
    PUBLIC INPUT_DIR=${RXMESH_SOURCE_DIR}/input/
    PUBLIC OUTPUT_DIR=${RXMESH_SOURCE_DIR}/output/
    PUBLIC EIGEN_ALLOW_UNALIGNED_SCALARS 
)

if(${RX_USE_POLYSCOPE})
    target_compile_definitions(RXMesh PUBLIC USE_POLYSCOPE)
endif()

if(${RX_USE_DOUBLE})
    target_compile_definitions(RXMesh PUBLIC RXMESH_USE_DOUBLE)
endif()

target_include_directories(RXMesh
    PUBLIC "${RXMESH_SOURCE_DIR}/include"
    PUBLIC "${rapidjson_SOURCE_DIR}/include"
    PUBLIC "${spdlog_SOURCE_DIR}/include"
    PUBLIC "${cereal_SOURCE_DIR}/include"
)
if(USE_HIP)
    # HIP-only redirect headers so the CUDA include spellings
    # (<cooperative_groups.h>, <cub/...>) resolve to their HIP equivalents
    # without editing every include site. Must precede the system include path.
    target_include_directories(RXMesh BEFORE PUBLIC
        "${RXMESH_SOURCE_DIR}/include/rxmesh/hip_compat")
else()
    target_include_directories(RXMesh
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# cuBQL is linked but never #included by RXMesh sources; it is a CUDA-only BVH
# library, so drop it on HIP.
if(NOT USE_HIP)
    target_link_libraries(RXMesh PUBLIC cuBQL cuBQL_queries)
endif()
target_link_libraries(RXMesh PUBLIC rapidobj::rapidobj)

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
    -use_fast_math
    $<$<CXX_COMPILER_ID:GNU>:-O3>
    --expt-relaxed-constexpr
    -Xptxas -warn-spills
    --resource-usage
    #-Xptxas -dlcm=cg -dscm=cg
    --ptxas-options=-v
)

# HIP (clang) flags for the .cu translation units compiled as HIP. The nvcc
# cuda_flags above (-Xcudafe/-rdc/-Xptxas/-lineinfo/--expt-*) are nvcc-only.
set(hip_flags
    $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-function>
    -ffast-math
    -Wno-unused-result
)

target_compile_options(RXMesh PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<BOOL:${USE_HIP}>>>:${cuda_flags}>
    $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<BOOL:${USE_HIP}>>:${hip_flags}>
)

#SuiteSparse
if(${RX_USE_SUITESPARSE})	
	target_compile_definitions(RXMesh PUBLIC USE_SUITESPARSE)
endif()
target_link_libraries(RXMesh PUBLIC glm::glm)
if(${RX_USE_POLYSCOPE})    
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
    rxmesh_enable_cudss(RXMesh)
endif()

#cuSolver and cuSparse (hipSPARSE/hipSOLVER/hipBLAS on HIP)
if(USE_HIP)
    find_package(hipsparse REQUIRED)
    find_package(hipsolver REQUIRED)
    find_package(hipblas REQUIRED)
    target_link_libraries(RXMesh PUBLIC roc::hipsparse roc::hipsolver roc::hipblas)
else()
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(RXMesh PUBLIC CUDA::cusparse)
    target_link_libraries(RXMesh PUBLIC CUDA::cusolver)
endif()


#Eigen
include("${CMAKE_CURRENT_LIST_DIR}/eigen.cmake")
target_link_libraries(RXMesh PUBLIC Eigen3::Eigen)
# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(RXMesh PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")
