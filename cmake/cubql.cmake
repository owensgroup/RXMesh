include_guard(GLOBAL)

include(FetchContent)

# cuBQL
FetchContent_Declare(cubql
    GIT_REPOSITORY https://github.com/NVIDIA/cuBQL.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(cubql)

