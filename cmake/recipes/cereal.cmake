# cmake/recipes/cereal.cmake
# Ensure subprojects can use the deprecated FindBoost module on CMake 4.0
set(CMAKE_POLICY_DEFAULT_CMP0167 OLD)

include(FetchContent)

FetchContent_Declare(cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG        v1.3.2
)

FetchContent_MakeAvailable(cereal)


