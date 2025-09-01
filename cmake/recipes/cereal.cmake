# Cereal recipe
include(FetchContent)

FetchContent_Declare(cereal
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG        v1.3.2
)

FetchContent_Populate(cereal)

# Add cereal to the main library
target_include_directories(${PROJECT_NAME}_lib PRIVATE "${cereal_SOURCE_DIR}/include")
