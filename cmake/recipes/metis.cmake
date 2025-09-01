# METIS recipe
include(FetchContent)

# Try to find system METIS first
find_package(METIS QUIET)

if(METIS_FOUND)
    message(STATUS "Found system METIS")
    # Create metis::metis target from system METIS
    if(NOT TARGET metis::metis)
        add_library(metis::metis SHARED IMPORTED)
        set_target_properties(metis::metis PROPERTIES
            IMPORTED_LOCATION ${METIS_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${METIS_INCLUDES}
        )
    endif()
else()
    message(STATUS "System METIS not found, downloading and building from source")
    
    # Download and build METIS from GitHub
    FetchContent_Declare(metis
        GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
        GIT_TAG        master
    )
    
    # Fetch METIS source
    FetchContent_Populate(metis)
    
    # METIS requires GKlib, so we need to get that too
    FetchContent_Declare(gklib
        GIT_REPOSITORY https://github.com/KarypisLab/GKlib.git
        GIT_TAG        master
    )
    
    FetchContent_Populate(gklib)
    
    # Build GKlib first
    add_subdirectory(${gklib_SOURCE_DIR} ${gklib_BINARY_DIR})
    
    # Configure METIS build options
    set(METIS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/metis-install)
    set(GKLIB_PATH ${gklib_SOURCE_DIR})
    
    # Build METIS using its build system
    include(ExternalProject)
    ExternalProject_Add(metis_build
        SOURCE_DIR ${metis_SOURCE_DIR}
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
            CC=${CMAKE_C_COMPILER}
            ${CMAKE_MAKE_PROGRAM} config
            prefix=${METIS_INSTALL_PREFIX}
            gklib_path=${GKLIB_PATH}
            shared=1
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
        BUILD_IN_SOURCE 1
        DEPENDS gk
    )
    
    # Create METIS target
    add_library(metis::metis SHARED IMPORTED)
    add_dependencies(metis::metis metis_build)
    
    # Set library properties
    set_target_properties(metis::metis PROPERTIES
        IMPORTED_LOCATION ${METIS_INSTALL_PREFIX}/lib/libmetis${CMAKE_SHARED_LIBRARY_SUFFIX}
        INTERFACE_INCLUDE_DIRECTORIES ${METIS_INSTALL_PREFIX}/include
    )
    
    # Make sure the library is built before it's used
    file(MAKE_DIRECTORY ${METIS_INSTALL_PREFIX}/include)
    
    message(STATUS "METIS will be built from source")
endif()
