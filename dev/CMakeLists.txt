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
    target_compile_definitions(${PROJECT_NAME} INTERFACE RXMESH_WITH_SUITESPARSE)
endif ()

# Parth for fill-reducing orderings
if(${PROJECT_NAME}_WITH_PARTH)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/parth.cmake)
    # Add Parth to the main RXMesh library
    target_link_libraries(${PROJECT_NAME} INTERFACE Parth::parth)
    target_compile_definitions(${PROJECT_NAME} INTERFACE RXMESH_WITH_PARTH)
endif()

################################################################################
# SECTION 4: Create Development Helper Library
################################################################################
# This library contains development utilities for solvers, orderings, and utils
# that extend the base RXMesh library for development and benchmarking purposes

add_library(rxmesh_dev_helpers
        LinSysSolvers/LinSysSolver.cpp
        LinSysSolvers/CUDSSSolver.cu
        LinSysSolvers/CHOLMODSolver.cpp
        utils/get_factor_nnz.cpp
        utils/remove_diagonal.cpp
        utils/check_valid_permutation.cpp
        utils/compute_inverse_perm.cpp
        Ordering/ordering.cpp
        Ordering/metis_ordering.cpp
        Ordering/rxmesh_ordering.cu
        Ordering/neutral_ordering.cpp
)

# Link the base RXMesh library
target_link_libraries(rxmesh_dev_helpers PUBLIC ${PROJECT_NAME})

# Link additional dependencies required by dev helpers
target_link_libraries(rxmesh_dev_helpers PUBLIC 
    igl::core
    spdlog::spdlog
    Eigen3::Eigen
    CLI11::CLI11
)

# Add dev code include directories
target_include_directories(rxmesh_dev_helpers PUBLIC 
    LinSysSolvers
    Ordering
    utils
)

# Configure CUDA compilation for the dev helper library
# Enable separable compilation AND resolve device symbols in the library to prevent
# duplicate symbol definitions when linking executables
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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
# Device symbols are already resolved in rxmesh_dev_helpers, so don't resolve again
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Link both RXMesh library (inherited via rxmesh_dev_helpers) and dev helpers
target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC rxmesh_dev_helpers)
