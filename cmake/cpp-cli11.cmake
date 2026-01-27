include_guard(GLOBAL)

include(FetchContent)

# CLI11 - C++17 compatible argument parser
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.3.2
)
FetchContent_MakeAvailable(CLI11)
