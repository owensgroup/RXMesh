#
# STRUMPACK CMake Recipe
# Finds STRUMPACK and its dependencies (blaspp, lapackpp, slate, scalapack, zfp)
# and configures RPATH for runtime library discovery
#

if(TARGET STRUMPACK::strumpack_full)
    return()
endif()

message(STATUS "Third-party: configuring STRUMPACK and dependencies")

# Set CMAKE_PREFIX_PATH to help find_package locate the dependencies
list(APPEND CMAKE_PREFIX_PATH
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/blaspp/install"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/lapackpp/install"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/slate/install"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/zfp/install"
    "/home/behrooz/Desktop/Last_Project/STRUMPACK/install"
)


# Alternatively, set individual *_DIR variables for more precise control
set(blaspp_DIR "/home/behrooz/Desktop/Last_Project/strumpack-dep/blaspp/install/lib/cmake/blaspp" CACHE PATH "blaspp CMake config directory")
set(lapackpp_DIR "/home/behrooz/Desktop/Last_Project/strumpack-dep/lapackpp/install/lib/cmake/lapackpp" CACHE PATH "lapackpp CMake config directory")
set(slate_DIR "/home/behrooz/Desktop/Last_Project/strumpack-dep/slate/install/lib/cmake/slate" CACHE PATH "slate CMake config directory")
set(zfp_DIR "/home/behrooz/Desktop/Last_Project/strumpack-dep/zfp/install/lib/cmake/zfp" CACHE PATH "zfp CMake config directory")
set(STRUMPACK_DIR "/home/behrooz/Desktop/Last_Project/STRUMPACK/install/lib/cmake/STRUMPACK" CACHE PATH "STRUMPACK CMake config directory")

# Find MPI (required by STRUMPACK)
find_package(MPI REQUIRED COMPONENTS CXX)
if(MPI_FOUND)
    message(STATUS "  Found MPI for STRUMPACK")
endif()

# Find STRUMPACK dependencies
find_package(blaspp QUIET)
if(blaspp_FOUND)
    message(STATUS "  Found blaspp")
endif()

find_package(lapackpp QUIET)
if(lapackpp_FOUND)
    message(STATUS "  Found lapackpp")
    # Print lapackpp library information
    if(TARGET lapackpp)
        get_target_property(LAPACKPP_LOCATION lapackpp LOCATION)
        get_target_property(LAPACKPP_INTERFACE_LINK_LIBRARIES lapackpp INTERFACE_LINK_LIBRARIES)
        message(STATUS "    lapackpp target exists")
        if(LAPACKPP_INTERFACE_LINK_LIBRARIES)
            message(STATUS "    lapackpp links to: ${LAPACKPP_INTERFACE_LINK_LIBRARIES}")
        endif()
    endif()
endif()

find_package(slate QUIET)
if(slate_FOUND)
    message(STATUS "  Found slate")
endif()

find_package(zfp QUIET)
if(zfp_FOUND)
    message(STATUS "  Found zfp")
endif()

# Find STRUMPACK
find_package(STRUMPACK REQUIRED)
if(STRUMPACK_FOUND)
    message(STATUS "  Found STRUMPACK")
endif()

# Create a unified interface target that includes STRUMPACK and all dependencies
if(NOT TARGET STRUMPACK::strumpack_full)
    add_library(STRUMPACK::strumpack_full INTERFACE IMPORTED)
    
    # Link STRUMPACK
    target_link_libraries(STRUMPACK::strumpack_full INTERFACE STRUMPACK::strumpack)
    
    # Link MPI
    target_link_libraries(STRUMPACK::strumpack_full INTERFACE MPI::MPI_CXX)
    
    # Link dependencies if found
    if(TARGET blaspp)
        target_link_libraries(STRUMPACK::strumpack_full INTERFACE blaspp)
    endif()
    
    if(TARGET lapackpp)
        target_link_libraries(STRUMPACK::strumpack_full INTERFACE lapackpp)
    endif()
    
    if(TARGET slate)
        target_link_libraries(STRUMPACK::strumpack_full INTERFACE slate)
    endif()
    
    if(TARGET zfp::zfp)
        target_link_libraries(STRUMPACK::strumpack_full INTERFACE zfp::zfp)
    elseif(TARGET zfp)
        target_link_libraries(STRUMPACK::strumpack_full INTERFACE zfp)
    endif()
    
    message(STATUS "STRUMPACK configuration complete")
endif()

# Configure RPATH directories for STRUMPACK dependencies
# These paths will be used by executables to find shared libraries at runtime
set(STRUMPACK_RPATH_DIRS
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/blaspp/install/lib"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/lapackpp/install/lib"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/slate/install/lib"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/scalapack-2.2.2/install/lib"
    "/home/behrooz/Desktop/Last_Project/strumpack-dep/zfp/install/lib"
    "/home/behrooz/Desktop/Last_Project/STRUMPACK/install/lib"
    CACHE STRING "RPATH directories for STRUMPACK dependencies"
)

message(STATUS "STRUMPACK RPATH directories configured:")
foreach(rpath_dir ${STRUMPACK_RPATH_DIRS})
    message(STATUS "  ${rpath_dir}")
endforeach()

