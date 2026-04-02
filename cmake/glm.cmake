include_guard(GLOBAL)

if(TARGET glm::glm)
    return()
endif()

include(FetchContent)

# glm
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863 #refs/tags/1.0.1
)
FetchContent_Populate(glm)
add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
target_compile_definitions(glm PUBLIC GLM_ENABLE_EXPERIMENTAL)