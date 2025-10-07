set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# SECTION 1: Include Required Dependencies
################################################################################
include(${CMAKE_SOURCE_DIR}/cmake/recipes/cli11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/libigl.cmake)

################################################################################
# SECTION 2: Development Build Options
################################################################################
option(${PROJECT_NAME}_WITH_PARTH "Use Parth for fill-reducing orderings" ON)
option(${PROJECT_NAME}_WITH_SUITESPARSE "Use SuiteSparse for benchmark" ON)

################################################################################
# SECTION 3: Configure RXMesh Library with Optional Development Dependencies
################################################################################

# SuiteSparse - Must be configured before Parth since Parth depends on it
if (${PROJECT_NAME}_WITH_SUITESPARSE)
    # Try to find system-installed SuiteSparse first
    set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITESPARSE_INC})
    set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITESPARSE_LIB})
    find_package(SuiteSparse QUIET)

    if(NOT SuiteSparse_FOUND)
        # If system SuiteSparse not found, use FetchContent to build from source
        message(STATUS "System SuiteSparse not found. Building from source...")
        include(${CMAKE_SOURCE_DIR}/cmake/recipes/suitesparse.cmake)
    endif()

    # Add SuiteSparse to the main RXMesh library
    target_link_libraries(${PROJECT_NAME} INTERFACE ${SUITESPARSE_LIBRARIES})
    target_include_directories(${PROJECT_NAME} INTERFACE ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} INTERFACE USE_SUITESPARSE)
endif ()

# Parth for fill-reducing orderings
if(${PROJECT_NAME}_WITH_PARTH)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/parth.cmake)
    # Add Parth to the main RXMesh library
    target_link_libraries(${PROJECT_NAME} INTERFACE Parth::parth)
    target_compile_definitions(${PROJECT_NAME} INTERFACE USE_PARTH)
endif()

################################################################################
# SECTION 4: Create Development Helper Library
################################################################################
# This library contains development utilities for solvers, orderings, and utils
# that extend the base RXMesh library for development and benchmarking purposes

# Create parallel_cholmod library first (needed by rxmesh_dev_helpers)
add_library(parallel_cholmod
        LeftLooking_Parth/common.cpp
        LeftLooking_Parth/executor.cpp
        LeftLooking_Parth/inspector.cpp
        LeftLooking_Parth/solve.cpp
        LeftLooking_Parth/utils/compute_etree_inv.cpp
        LeftLooking_Parth/custom_cholmod_files/analysis/parth_analyze.cpp
        LeftLooking_Parth/custom_cholmod_files/analysis/parth_super_symbolic.cpp
        LeftLooking_Parth/custom_cholmod_files/factorize/parth_factorize.cpp
        LeftLooking_Parth/custom_cholmod_files/factorize/parth_super_numeric.cpp



#        LeftLooking_Parth/custom_cholmod_files/parth_factorize.cpp
#        LeftLooking_Parth/custom_cholmod_files/parth_numeric_supernode.cpp
#        LeftLooking_Parth/custom_cholmod_files/parth_solve.cpp
#        LeftLooking_Parth/custom_cholmod_files/parth_solve.cpp
#        LeftLooking_Parth/custom_cholmod_files/parth_solve_supernode.cpp
#        LeftLooking_Parth/custom_cholmod_files/parth_super_symbolic.cpp
)

if(${PROJECT_NAME}_WITH_SUITESPARSE)
    # Link SuiteSparse libraries - order matters for static linking!
    target_include_directories(parallel_cholmod PUBLIC ${SUITESPARSE_INCLUDE_DIRS})
    
    # Ensure that SuiteSparse BLAS wrapper macros are available
    # This is needed for the lowercase BLAS wrapper definitions (SUITESPARSE_BLAS_dsyrk, etc.)
    # which are defined in SuiteSparse_config.h at line ~1814
    target_compile_definitions(parallel_cholmod PUBLIC SUITESPARSE_BLAS_DEFINITIONS)
    
    # For static library linking with CHOLMOD, we need to use --whole-archive to ensure
    # all symbols (especially utility functions like cholmod_mult_size_t) are extracted
    if(TARGET CHOLMOD_static)
        # Use generator expression to properly wrap the library with linker flags
        target_link_libraries(parallel_cholmod PUBLIC 
            "-Wl,--whole-archive"
            "$<TARGET_FILE:CHOLMOD_static>"
            "-Wl,--no-whole-archive"
        )
        # Link other SuiteSparse dependencies normally
        target_link_libraries(parallel_cholmod PUBLIC 
            AMD_static COLAMD_static CAMD_static CCOLAMD_static SuiteSparseConfig_static
        )
        # Link BLAS and LAPACK libraries
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        target_link_libraries(parallel_cholmod PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    else()
        # Fallback to regular linking if static targets not available
        target_link_libraries(parallel_cholmod PUBLIC ${SUITESPARSE_LIBRARIES})
        if(TARGET SuiteSparse::CHOLMOD)
            target_link_libraries(parallel_cholmod PUBLIC SuiteSparse::CHOLMOD)
        endif()
    endif()
endif()

if(${PROJECT_NAME}_WITH_PARTH)
    target_link_libraries(parallel_cholmod PUBLIC Parth::parth)
endif()

# Add include directories for parallel_cholmod
target_include_directories(parallel_cholmod PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        LeftLooking_Parth/include
        LeftLooking_Parth/utils
        LeftLooking_Parth/custom_cholmod_files/factorize
)

target_link_libraries(parallel_cholmod PRIVATE  spdlog::spdlog)

# Create rxmesh_dev_helpers library
add_library(rxmesh_dev_helpers
        LinSysSolvers/LinSysSolver.cpp
        LinSysSolvers/CUDSSSolver.cu
        LinSysSolvers/CHOLMODSolver.cpp
        LinSysSolvers/ParthSolver.cpp
        utils/get_factor_nnz.cpp
        utils/remove_diagonal.cpp
        utils/check_valid_permutation.cpp
        utils/compute_inverse_perm.cpp
        Ordering/ordering.cpp
        Ordering/metis_ordering.cpp
        Ordering/rxmesh_ordering.cu
        Ordering/neutral_ordering.cpp
)

# Link additional dependencies required by dev helpers
target_link_libraries(rxmesh_dev_helpers PUBLIC
    igl::core
    spdlog::spdlog
    Eigen3::Eigen
    CLI11::CLI11
    parallel_cholmod
)
target_link_libraries(rxmesh_dev_helpers PRIVATE ${PROJECT_NAME})


# Add dev code include directories
target_include_directories(rxmesh_dev_helpers PUBLIC 
    LinSysSolvers
    Ordering
    utils
    LeftLooking_Parth/include
    LeftLooking_Parth/custom_cholmod_files/include
    LeftLooking_Parth/custom_cholmod_files/include/Cholmod_Include
    LeftLooking_Parth/custom_cholmod_files/util
)

# Configure CUDA compilation for the dev helper library
# Enable separable compilation but DON'T resolve device symbols yet
# Device symbols will be resolved in the final executable
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Inherit CUDA architecture from the main library
get_target_property(MAIN_CUDA_ARCHITECTURES ${PROJECT_NAME} CUDA_ARCHITECTURES)
if(MAIN_CUDA_ARCHITECTURES)
    set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_ARCHITECTURES ${MAIN_CUDA_ARCHITECTURES})
endif()

# Apply the same CUDA compiler flags as the main library
target_compile_options(rxmesh_dev_helpers PUBLIC
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

################################################################################
# SECTION 5: Create Development Targets (Executables)
################################################################################
# Development targets use both the RXMesh library and the dev helpers library

# Benchmark executable
add_executable(${PROJECT_NAME}_benchmark benchmark.cpp)

# Configure CUDA device linking for the benchmark executable
# Enable separable compilation and resolve device symbols at the executable level
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link only rxmesh_dev_helpers (which already includes RXMesh)
# DO NOT link RXMesh directly to avoid compiling the same .cu files twice
# This prevents multiple definition errors for patcher.cu, rxmesh_dynamic.cu, etc.
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC rxmesh_dev_helpers)
