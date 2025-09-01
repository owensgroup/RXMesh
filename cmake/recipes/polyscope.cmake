# Polyscope recipe
include(FetchContent)

FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        30f3995f51a895f0113d3580a889077bdb846e7a
)

FetchContent_MakeAvailable(polyscope)

# Create polyscope::polyscope alias if it doesn't exist
if(NOT TARGET polyscope::polyscope)
    add_library(polyscope::polyscope ALIAS polyscope)
endif()
