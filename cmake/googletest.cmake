include_guard(GLOBAL)

if(NOT ${RX_BUILD_TESTS})
    return()
endif()

include(FetchContent)

# GoogleTest
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

