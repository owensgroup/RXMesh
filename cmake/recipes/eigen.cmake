# Eigen recipe
include(FetchContent)

# Try to find system Eigen first
find_package(Eigen3 QUIET)

if(Eigen3_FOUND)
    message(STATUS "Found system Eigen3")
    # Use system Eigen3 if available
else()
    message(STATUS "System Eigen3 not found, using FetchContent")
    
    # Fallback to FetchContent if system Eigen3 is not available
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        5.0.0
    )
    
    FetchContent_MakeAvailable(eigen)
    
    # Create Eigen3::Eigen target if it doesn't exist
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE)
        target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
    endif()
endif()
