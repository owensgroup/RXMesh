# Polyscope recipe
include(FetchContent)

# CMakeLists.txt (top of file)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Allow old third-party packages" FORCE)

FetchContent_Declare(polyscope
    GIT_REPOSITORY https://github.com/Ahdhn/polyscope.git
    GIT_TAG        30f3995f51a895f0113d3580a889077bdb846e7a
)

# Enable GLM experimental extensions before making polyscope available
set(GLM_ENABLE_EXPERIMENTAL ON CACHE BOOL "Enable GLM experimental extensions" FORCE)

FetchContent_MakeAvailable(polyscope)

# Apply GLM experimental definition to polyscope
if(TARGET polyscope)
    target_compile_definitions(polyscope PUBLIC GLM_ENABLE_EXPERIMENTAL)
endif()

# Create polyscope::polyscope alias if it doesn't exist
if(NOT TARGET polyscope::polyscope)
    add_library(polyscope::polyscope ALIAS polyscope)
endif()
