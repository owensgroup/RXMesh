include_guard(GLOBAL)

include(FetchContent)

# rapidobj - fast Wavefront OBJ parser
FetchContent_Declare(rapidobj
    GIT_REPOSITORY https://github.com/guybrush77/rapidobj.git
    GIT_TAG        v1.1
)
FetchContent_MakeAvailable(rapidobj)
