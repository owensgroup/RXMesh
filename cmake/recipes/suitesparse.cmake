# suitesparse.cmake
# Downloads and builds minimal SuiteSparse components (CHOLMOD, AMD, and dependencies)

if(TARGET SuiteSparse::CHOLMOD)
    return()
endif()

include(FetchContent)

# Download SuiteSparse
FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    GIT_TAG v7.11.0  # Latest stable version as of the search
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(suitesparse)
if(NOT suitesparse_POPULATED)
    message(STATUS "Fetching SuiteSparse...")
    FetchContent_Populate(suitesparse)
    
    # Configure SuiteSparse build options
    # Build only minimal required packages for CHOLMOD
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;colamd;ccolamd;cholmod" CACHE STRING "SuiteSparse projects to build")
    
    # Disable unnecessary CHOLMOD modules
    set(CHOLMOD_GPL ON CACHE BOOL "Enable GPL modules (required for supernodal)")
    set(CHOLMOD_SUPERNODAL ON CACHE BOOL "Enable supernodal factorization")
    set(CHOLMOD_CHOLESKY ON CACHE BOOL "Enable Cholesky module")
    set(CHOLMOD_CAMD ON CACHE BOOL "Enable CAMD support")
    set(CHOLMOD_PARTITION OFF CACHE BOOL "Disable Partition module")
    set(CHOLMOD_MATRIXOPS OFF CACHE BOOL "Disable MatrixOps module")
    set(CHOLMOD_MODIFY OFF CACHE BOOL "Disable Modify module")
    
    # Disable demo/test builds
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "Disable demos")
    
    # Use 64-bit integers if needed (important for large matrices)
    # set(SUITESPARSE_USE_64BIT_BLAS ON CACHE BOOL "Use 64-bit BLAS")
    
    # Disable Fortran if not needed
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran")
    
    # Add SuiteSparse to the build
    add_subdirectory(${suitesparse_SOURCE_DIR} ${suitesparse_BINARY_DIR} EXCLUDE_FROM_ALL)
    
    message(STATUS "SuiteSparse configured with CHOLMOD and AMD")
endif()

# Create interface targets if they don't exist
if(NOT TARGET SuiteSparse::CHOLMOD)
    add_library(SuiteSparse::CHOLMOD ALIAS CHOLMOD)
endif()

if(NOT TARGET SuiteSparse::AMD)
    add_library(SuiteSparse::AMD ALIAS AMD)
endif()

if(NOT TARGET SuiteSparse::COLAMD)
    add_library(SuiteSparse::COLAMD ALIAS COLAMD)
endif()

if(NOT TARGET SuiteSparse::CAMD)
    add_library(SuiteSparse::CAMD ALIAS CAMD)
endif()

if(NOT TARGET SuiteSparse::CCOLAMD)
    add_library(SuiteSparse::CCOLAMD ALIAS CCOLAMD)
endif()

# Export include directories for easier use
set(SUITESPARSE_INCLUDE_DIRS 
    ${suitesparse_SOURCE_DIR}/CHOLMOD/Include
    ${suitesparse_SOURCE_DIR}/AMD/Include
    ${suitesparse_SOURCE_DIR}/COLAMD/Include
    ${suitesparse_SOURCE_DIR}/CAMD/Include
    ${suitesparse_SOURCE_DIR}/CCOLAMD/Include
    ${suitesparse_SOURCE_DIR}/SuiteSparse_config
    CACHE PATH "SuiteSparse include directories"
)

set(SUITESPARSE_LIBRARIES CHOLMOD AMD COLAMD CAMD CCOLAMD SuiteSparse_config CACHE STRING "SuiteSparse libraries")

