include_guard(GLOBAL)

if(NOT ${RX_USE_POLYSCOPE})
    return()
endif()

include(FetchContent)

# polyscope
FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        0c3dd68b9851417e6b2b976d347adc3250026122
)
FetchContent_MakeAvailable(polyscope)

