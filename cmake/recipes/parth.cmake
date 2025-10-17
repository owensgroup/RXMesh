#
# Parth CMake Recipe
# Downloads and builds the Parth library for fill-reducing orderings in sparse Cholesky factorization
#

if(TARGET Parth::parth)
    return()
endif()

message(STATUS "Third-party: creating target 'Parth::parth'")

include(FetchContent)

# Download Parth from GitHub
FetchContent_Declare(
    parth
    GIT_REPOSITORY https://github.com/BehroozZare/Parth.git
    GIT_TAG parth_dev
    GIT_SHALLOW TRUE
)

# Set build options for Parth - disable examples and tests for cleaner build
set(PARTH_WITH_TESTS OFF CACHE BOOL "Disable Parth tests" FORCE)
set(PARTH_WITH_API_DEMO OFF CACHE BOOL "Disable Parth API demos" FORCE)
set(PARTH_WITH_CHOLMOD_DEMO OFF CACHE BOOL "Disable Parth CHOLMOD demos" FORCE)
set(PARTH_WITH_ACCELERATE_DEMO OFF CACHE BOOL "Disable Parth Accelerate demos" FORCE)
set(PARTH_WITH_MKL_DEMO OFF CACHE BOOL "Disable Parth MKL demos" FORCE)
set(PARTH_WITH_IPC_DEMO OFF CACHE BOOL "Disable Parth IPC demos" FORCE)
set(PARTH_WITH_REMESHING_DEMO OFF CACHE BOOL "Disable Parth remeshing demos" FORCE)
set(PARTH_WITH_SOLVER_WRAPPER OFF CACHE BOOL "Disable Parth solver wrapper" FORCE)

# Make Parth available
FetchContent_MakeAvailable(parth)

# Parth::parth target is already created by the Parth CMakeLists.txt
message(STATUS "Parth library configured successfully")