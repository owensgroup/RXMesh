# GLM recipe
include(FetchContent)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)

FetchContent_Populate(glm)

# Add GLM to the main library
add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
target_link_libraries(${PROJECT_NAME}_lib PRIVATE glm::glm)

# Enable experimental GLM extensions
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC GLM_ENABLE_EXPERIMENTAL)