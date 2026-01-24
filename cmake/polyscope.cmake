include_guard(GLOBAL)

if(NOT ${RX_USE_POLYSCOPE})
    return()
endif()

include(FetchContent)

# polyscope
FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        30f3995f51a895f0113d3580a889077bdb846e7a
)
FetchContent_MakeAvailable(polyscope)

