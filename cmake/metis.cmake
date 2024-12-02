# metis.cmake

if(TARGET metis)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    metis
    GIT_REPOSITORY https://github.com/EricYJA/METIS-CMake.git
    GIT_TAG f72497fcc8d634815209e993f19ba5be0c46d090
)
FetchContent_GetProperties(metis)
if(NOT metis_POPULATED)
    FetchContent_Populate(metis)
endif()

# Optionally set IDXTYPEWIDTH and REALTYPEWIDTH
set(IDXTYPEWIDTH 32 CACHE STRING "Width of integer type for METIS")
set(REALTYPEWIDTH 32 CACHE STRING "Width of real type for METIS")

# Add the METIS library
add_subdirectory(${metis_SOURCE_DIR} ${metis_BINARY_DIR})

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
