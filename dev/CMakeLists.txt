set(REQUIRED_CMAKE_VERSION "3.9")

################################################################################
# CLI11
include(cli11)
include(eigen)
include(libigl)

#Create the benchmark library
# Create utils library
add_library(rxmesh_dev_solvers
        LinSysSolvers/LinSysSolver.cpp
        LinSysSolvers/CUDSSSolver.cu
        LinSysSolvers/CHOLMODSolver.cpp
        utils/get_factor_nnz.cpp
        utils/remove_diagonal.cpp
        Ordering/ordering.cpp
        Ordering/metis_ordering.cpp
)
#include libigl
include(libigl)
target_link_libraries(rxmesh_dev_solvers PUBLIC igl::core)
# add spdlog logging
target_link_libraries(rxmesh_dev_solvers PUBLIC spdlog::spdlog)
# Add Eigen
target_link_libraries(rxmesh_dev_solvers PUBLIC Eigen3::Eigen)
# Add CLI11
target_link_libraries(rxmesh_dev_solvers PUBLIC CLI11::CLI11)
# Add base RXMESH
target_link_libraries(rxmesh_dev_solvers PUBLIC ${PROJECT_NAME}_lib)
# Add dev code base
target_include_directories(rxmesh_dev_solvers PUBLIC LinSysSolvers)
target_include_directories(rxmesh_dev_solvers PUBLIC Ordering)
target_include_directories(rxmesh_dev_solvers PUBLIC utils)

# Ensure CUDA separable compilation is properly configured for the solver library
# Enable separable compilation but do NOT resolve device symbols in intermediate library
set_property(TARGET rxmesh_dev_solvers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET rxmesh_dev_solvers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Inherit CUDA architecture from the main library
get_target_property(MAIN_CUDA_ARCHITECTURES ${PROJECT_NAME}_lib CUDA_ARCHITECTURES)
if(MAIN_CUDA_ARCHITECTURES)
    set_property(TARGET rxmesh_dev_solvers PROPERTY CUDA_ARCHITECTURES ${MAIN_CUDA_ARCHITECTURES})
endif()

# Apply the same CUDA compiler flags as the main library
target_compile_options(rxmesh_dev_solvers PUBLIC
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


# Testing Example Creation
add_executable(${PROJECT_NAME}_benchmark benchmark.cpp)

# Configure CUDA device linking for the benchmark executable
# The executable should resolve all device symbols from linked libraries
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(${PROJECT_NAME}_benchmark PUBLIC rxmesh_dev_solvers)
