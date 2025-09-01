# GLM recipe
include(FetchContent)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        master
)

FetchContent_Populate(glm)

# Add GLM to the main library
add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
target_link_libraries(${PROJECT_NAME}_lib PRIVATE glm::glm)
