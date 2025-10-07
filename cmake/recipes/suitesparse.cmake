# suitesparse.cmake
# Downloads and builds minimal SuiteSparse components (CHOLMOD, AMD, and dependencies)

if(TARGET SuiteSparse::CHOLMOD)
    return()
endif()

include(FetchContent)

# Find BLAS and LAPACK (required by SuiteSparse)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Download SuiteSparse
FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    GIT_TAG v7.11.0  # Using latest stable version with BLAS/LAPACK compatibility header
    GIT_SHALLOW TRUE
)

# Check if already populated to avoid re-fetching
FetchContent_GetProperties(suitesparse)
if(NOT suitesparse_POPULATED)
    message(STATUS "Fetching SuiteSparse...")
    
    # Configure SuiteSparse build options before populating
    # Build only minimal required packages for CHOLMOD
    set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;colamd;ccolamd;cholmod" CACHE STRING "SuiteSparse projects to build")
    
    # Configure CHOLMOD modules
    # Note: Utility module (containing cholmod_mult_size_t, cholmod_add_size_t, etc.)
    # must be explicitly enabled
    set(CHOLMOD_GPL ON CACHE BOOL "Enable GPL modules (required for supernodal)")
    set(CHOLMOD_SUPERNODAL ON CACHE BOOL "Enable supernodal factorization")
    set(CHOLMOD_CHOLESKY ON CACHE BOOL "Enable Cholesky module")
    set(CHOLMOD_CAMD ON CACHE BOOL "Enable CAMD support")
    set(CHOLMOD_UTILITY ON CACHE BOOL "Enable Utility module")
    set(CHOLMOD_PARTITION OFF CACHE BOOL "Disable Partition module")
    set(CHOLMOD_MATRIXOPS OFF CACHE BOOL "Disable MatrixOps module")
    set(CHOLMOD_MODIFY OFF CACHE BOOL "Disable Modify module")
    
    # Disable demo/test builds
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "Disable demos")
    
    # Use 64-bit integers if needed (important for large matrices)
    # set(SUITESPARSE_USE_64BIT_BLAS ON CACHE BOOL "Use 64-bit BLAS")
    
    # Disable Fortran if not needed
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran")
    
    # Enable building static libraries (required for targets to be created)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libraries")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Don't build shared libraries")
    
    # Configure BLAS/LAPACK settings for SuiteSparse
    # This ensures the lowercase BLAS/LAPACK wrapper macros are defined
    set(SUITESPARSE_USE_SYSTEM_BLAS ON CACHE BOOL "Use system BLAS")
    set(SUITESPARSE_USE_SYSTEM_LAPACK ON CACHE BOOL "Use system LAPACK")
    set(BLA_VENDOR "Generic" CACHE STRING "BLAS vendor")
    
    # Configure BLAS naming convention - critical for wrapper macro definitions
    # SuiteSparse uses these to define SUITESPARSE_BLAS_dsyrk, etc.
    # Most modern BLAS libraries use lowercase names with underscores (Fortran convention)
    set(BLAS_UNDERSCORE ON CACHE BOOL "BLAS names have trailing underscore")
    set(BLAS_NO_UNDERSCORE OFF CACHE BOOL "BLAS names have no trailing underscore")
    
    # Populate the content
    FetchContent_Populate(suitesparse)
    
    # Add SuiteSparse to the build with EXCLUDE_FROM_ALL
    add_subdirectory(${suitesparse_SOURCE_DIR} ${suitesparse_BINARY_DIR} EXCLUDE_FROM_ALL)
    
    message(STATUS "SuiteSparse configured with CHOLMOD and AMD")
endif()

# Create interface targets if they don't exist
# Note: We're building static libs, so targets are named *_static
if(NOT TARGET SuiteSparse::CHOLMOD)
    if(TARGET CHOLMOD_static)
        add_library(SuiteSparse::CHOLMOD ALIAS CHOLMOD_static)
    elseif(TARGET CHOLMOD)
        add_library(SuiteSparse::CHOLMOD ALIAS CHOLMOD)
    endif()
endif()

if(NOT TARGET SuiteSparse::AMD)
    if(TARGET AMD_static)
        add_library(SuiteSparse::AMD ALIAS AMD_static)
    elseif(TARGET AMD)
        add_library(SuiteSparse::AMD ALIAS AMD)
    endif()
endif()

if(NOT TARGET SuiteSparse::COLAMD)
    if(TARGET COLAMD_static)
        add_library(SuiteSparse::COLAMD ALIAS COLAMD_static)
    elseif(TARGET COLAMD)
        add_library(SuiteSparse::COLAMD ALIAS COLAMD)
    endif()
endif()

if(NOT TARGET SuiteSparse::CAMD)
    if(TARGET CAMD_static)
        add_library(SuiteSparse::CAMD ALIAS CAMD_static)
    elseif(TARGET CAMD)
        add_library(SuiteSparse::CAMD ALIAS CAMD)
    endif()
endif()

if(NOT TARGET SuiteSparse::CCOLAMD)
    if(TARGET CCOLAMD_static)
        add_library(SuiteSparse::CCOLAMD ALIAS CCOLAMD_static)
    elseif(TARGET CCOLAMD)
        add_library(SuiteSparse::CCOLAMD ALIAS CCOLAMD)
    endif()
endif()

if(NOT TARGET SuiteSparse::SuiteSparseConfig)
    if(TARGET SuiteSparseConfig_static)
        add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig_static)
    elseif(TARGET SuiteSparseConfig)
        add_library(SuiteSparse::SuiteSparseConfig ALIAS SuiteSparseConfig)
    endif()
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

# Set library list based on what was built (static or shared)
if(TARGET CHOLMOD_static)
    set(SUITESPARSE_LIBRARIES CHOLMOD_static AMD_static COLAMD_static CAMD_static CCOLAMD_static SuiteSparseConfig_static CACHE STRING "SuiteSparse libraries")
else()
    set(SUITESPARSE_LIBRARIES CHOLMOD AMD COLAMD CAMD CCOLAMD SuiteSparseConfig CACHE STRING "SuiteSparse libraries")
endif()

# Set variables that FindSuiteSparse.cmake expects
set(SuiteSparse_FOUND TRUE CACHE BOOL "SuiteSparse found")
set(SUITESPARSE_FOUND TRUE CACHE BOOL "SuiteSparse found")

# Set individual component variables that FindSuiteSparse.cmake looks for
set(AMD_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/AMD/Include CACHE PATH "AMD include directory")
set(CAMD_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/CAMD/Include CACHE PATH "CAMD include directory")
set(COLAMD_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/COLAMD/Include CACHE PATH "COLAMD include directory")
set(CCOLAMD_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/CCOLAMD/Include CACHE PATH "CCOLAMD include directory")
set(CHOLMOD_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/CHOLMOD/Include CACHE PATH "CHOLMOD include directory")
set(SUITESPARSE_CONFIG_INCLUDE_DIR ${suitesparse_SOURCE_DIR}/SuiteSparse_config CACHE PATH "SuiteSparse_config include directory")

# Set library variables based on what was built
if(TARGET CHOLMOD_static)
    set(AMD_LIBRARY AMD_static CACHE STRING "AMD library")
    set(CAMD_LIBRARY CAMD_static CACHE STRING "CAMD library")
    set(COLAMD_LIBRARY COLAMD_static CACHE STRING "COLAMD library")
    set(CCOLAMD_LIBRARY CCOLAMD_static CACHE STRING "CCOLAMD library")
    set(CHOLMOD_LIBRARY CHOLMOD_static CACHE STRING "CHOLMOD library")
    set(SUITESPARSE_CONFIG_LIBRARY SuiteSparseConfig_static CACHE STRING "SuiteSparse_config library")
else()
    set(AMD_LIBRARY AMD CACHE STRING "AMD library")
    set(CAMD_LIBRARY CAMD CACHE STRING "CAMD library")
    set(COLAMD_LIBRARY COLAMD CACHE STRING "COLAMD library")
    set(CCOLAMD_LIBRARY CCOLAMD CACHE STRING "CCOLAMD library")
    set(CHOLMOD_LIBRARY CHOLMOD CACHE STRING "CHOLMOD library")
    set(SUITESPARSE_CONFIG_LIBRARY SuiteSparseConfig CACHE STRING "SuiteSparse_config library")
endif()

