# metis.cmake

if(TARGET metis)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    metis
    GIT_REPOSITORY https://github.com/scivision/METIS.git
    GIT_TAG d4a3aac2a3a0efc18e1de24ae97302ed510f43c7
)
FetchContent_MakeAvailable(metis)

# Optionally set IDXTYPEWIDTH and REALTYPEWIDTH
set(IDXTYPEWIDTH 32 CACHE STRING "Width of integer type for METIS")
set(REALTYPEWIDTH 32 CACHE STRING "Width of real type for METIS")

# METIS is added by FetchContent_MakeAvailable

# Include directories and definitions
target_include_directories(metis INTERFACE
    $<BUILD_INTERFACE:${metis_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(metis INTERFACE
    IDXTYPEWIDTH=${IDXTYPEWIDTH}
    REALTYPEWIDTH=${REALTYPEWIDTH}
)

# Install rules
install(DIRECTORY ${metis_SOURCE_DIR}/include/ DESTINATION include)
install(TARGETS metis EXPORT MetisTargets)
install(EXPORT MetisTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metis)