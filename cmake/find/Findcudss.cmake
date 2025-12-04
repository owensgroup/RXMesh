# Find cuDSS - CUDA Direct Sparse Solver

if (CUDSS_INCLUDES AND CUDSS_LIBRARIES)
    set(CUDSS_FIND_QUIETLY TRUE)
endif (CUDSS_INCLUDES AND CUDSS_LIBRARIES)

find_path(CUDSS_INCLUDES
        NAMES
        cudss.h
        PATHS
        $ENV{CUDSSROOT}
        ${CUDSS_ROOT}
        ${INCLUDE_INSTALL_DIR}
        /usr/include
        /usr/include/libcudss/12
        /usr/local/include
        /usr/local/include/cudss
        PATH_SUFFIXES
        .
        cudss
        libcudss
        libcudss/12
        include
        )

macro(_cudss_check_version)
    file(READ "${CUDSS_INCLUDES}/cudss.h" _cudss_version_header)

    string(REGEX MATCH "define[ \t]+CUDSS_VER_MAJOR[ \t]+([0-9]+)" _cudss_major_version_match "${_cudss_version_header}")
    set(CUDSS_MAJOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+CUDSS_VER_MINOR[ \t]+([0-9]+)" _cudss_minor_version_match "${_cudss_version_header}")
    set(CUDSS_MINOR_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define[ \t]+CUDSS_VER_PATCH[ \t]+([0-9]+)" _cudss_patch_version_match "${_cudss_version_header}")
    set(CUDSS_PATCH_VERSION "${CMAKE_MATCH_1}")
    
    # Alternative version pattern matching
    if(NOT CUDSS_MAJOR_VERSION)
        string(REGEX MATCH "CUDSS_VERSION_MAJOR[ \t]+([0-9]+)" _cudss_major_version_match "${_cudss_version_header}")
        set(CUDSS_MAJOR_VERSION "${CMAKE_MATCH_1}")
        string(REGEX MATCH "CUDSS_VERSION_MINOR[ \t]+([0-9]+)" _cudss_minor_version_match "${_cudss_version_header}")
        set(CUDSS_MINOR_VERSION "${CMAKE_MATCH_1}")
        string(REGEX MATCH "CUDSS_VERSION_PATCH[ \t]+([0-9]+)" _cudss_patch_version_match "${_cudss_version_header}")
        set(CUDSS_PATCH_VERSION "${CMAKE_MATCH_1}")
    endif()
    
    if(NOT CUDSS_MAJOR_VERSION)
        message(STATUS "Could not determine cuDSS version. Assuming version 0.6.0")
        set(CUDSS_VERSION 0.6.0)
    else()
        set(CUDSS_VERSION ${CUDSS_MAJOR_VERSION}.${CUDSS_MINOR_VERSION}.${CUDSS_PATCH_VERSION})
    endif()
    
    if(${CUDSS_VERSION} VERSION_LESS ${cudss_FIND_VERSION})
        set(CUDSS_VERSION_OK FALSE)
    else()
        set(CUDSS_VERSION_OK TRUE)
    endif()

    if(NOT CUDSS_VERSION_OK)
        message(STATUS "cuDSS version ${CUDSS_VERSION} found in ${CUDSS_INCLUDES}, "
                "but at least version ${cudss_FIND_VERSION} is required")
    endif(NOT CUDSS_VERSION_OK)
endmacro(_cudss_check_version)

if(CUDSS_INCLUDES AND cudss_FIND_VERSION)
    _cudss_check_version()
else()
    set(CUDSS_VERSION_OK TRUE)
endif()

# Find the main cuDSS library
find_library(CUDSS_LIBRARIES 
        NAMES cudss libcudss
        PATHS 
        $ENV{CUDSSROOT} 
        ${LIB_INSTALL_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu/libcudss/12
        /usr/local/lib
        /usr/local/lib/cudss
        PATH_SUFFIXES 
        lib
        lib64
        )

# Find additional cuDSS libraries that might be needed
find_library(CUDSS_MTLAYER_LIBRARY 
        NAMES cudss_mtlayer_gomp libcudss_mtlayer_gomp
        PATHS 
        $ENV{CUDSSROOT} 
        ${LIB_INSTALL_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu/libcudss/12
        /usr/local/lib
        /usr/local/lib/cudss
        PATH_SUFFIXES 
        lib
        lib64
        )

find_library(CUDSS_COMMLAYER_LIBRARY 
        NAMES cudss_commlayer_nccl libcudss_commlayer_nccl
        PATHS 
        $ENV{CUDSSROOT} 
        ${LIB_INSTALL_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/lib/x86_64-linux-gnu/libcudss/12
        /usr/local/lib
        /usr/local/lib/cudss
        PATH_SUFFIXES 
        lib
        lib64
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cudss DEFAULT_MSG
        CUDSS_INCLUDES CUDSS_LIBRARIES CUDSS_VERSION_OK)

if(cudss_FOUND)
    set(CUDSS_INCLUDE_DIRS ${CUDSS_INCLUDES})
    set(cudss_INCLUDE_DIRS ${CUDSS_INCLUDES})  # For CMakeLists.txt compatibility
    
    # Set up the libraries list
    set(CUDSS_LIBRARY_LIST ${CUDSS_LIBRARIES})
    if(CUDSS_MTLAYER_LIBRARY)
        list(APPEND CUDSS_LIBRARY_LIST ${CUDSS_MTLAYER_LIBRARY})
    endif()
    if(CUDSS_COMMLAYER_LIBRARY)
        list(APPEND CUDSS_LIBRARY_LIST ${CUDSS_COMMLAYER_LIBRARY})
    endif()
    
    set(CUDSS_LIBRARIES ${CUDSS_LIBRARY_LIST})
    set(cudss_LIBRARIES ${CUDSS_LIBRARY_LIST})  # For CMakeLists.txt compatibility
    
    # Set version variable for CMakeLists.txt compatibility
    set(cudss_VERSION ${CUDSS_VERSION})
    
    # Create imported target with expected name
    if(NOT TARGET cudss::cudss)
        add_library(cudss::cudss INTERFACE IMPORTED)
        set_target_properties(cudss::cudss PROPERTIES
            INTERFACE_LINK_LIBRARIES "${CUDSS_LIBRARY_LIST}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIRS}")
    endif()
endif()

mark_as_advanced(CUDSS_INCLUDES CUDSS_LIBRARIES CUDSS_MTLAYER_LIBRARY CUDSS_COMMLAYER_LIBRARY)
