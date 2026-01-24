include_guard(GLOBAL)

include(FetchContent)

# cereal
FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
)
FetchContent_Populate(cereal)

