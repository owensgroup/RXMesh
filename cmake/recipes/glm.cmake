# GLM recipe
include(FetchContent)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863
)

FetchContent_MakeAvailable(glm)

# Link GLM (note: changed from PRIVATE to PUBLIC based on your CMakeLists.txt)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC glm::glm)

# Enable experimental GLM extensions
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC GLM_ENABLE_EXPERIMENTAL)