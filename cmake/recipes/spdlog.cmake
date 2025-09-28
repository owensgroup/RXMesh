# spdlog recipe
include(FetchContent)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.8.5
)

FetchContent_MakeAvailable(spdlog)

# Add spdlog to the main library
target_include_directories(${PROJECT_NAME}_lib PUBLIC "${spdlog_SOURCE_DIR}/include")
