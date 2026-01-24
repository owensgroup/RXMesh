include_guard(GLOBAL)

if(NOT ${RX_BUILD_TESTS})
    return()
endif()

include(FetchContent)

# GoogleTest
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        eaf9a3fd77869cf95befb87455a2e2a2e85044ff
)
FetchContent_MakeAvailable(googletest)

