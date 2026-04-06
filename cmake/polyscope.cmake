include_guard(GLOBAL)

if(NOT ${RX_USE_POLYSCOPE})
    return()
endif()

include(FetchContent)

# polyscope
FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        d7bcbb426fc550c643736732bd017dd4120b6b61
)
FetchContent_MakeAvailable(polyscope)

