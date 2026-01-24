include_guard(GLOBAL)

include(FetchContent)

# spdlog
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.8.5
)
FetchContent_Populate(spdlog)

