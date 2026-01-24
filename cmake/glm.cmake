include_guard(GLOBAL)

include(FetchContent)

# glm
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        master
)
FetchContent_Populate(glm)
add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})

