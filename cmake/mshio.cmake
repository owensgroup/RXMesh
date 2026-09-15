include_guard(GLOBAL)

include(FetchContent)

# MshIO
FetchContent_Declare(mshio
    GIT_REPOSITORY https://github.com/qnzhou/MshIO.git
    GIT_TAG        e18c8a6c4140da0c16c0cba6273e40d9d1f2f91f
)
FetchContent_MakeAvailable(mshio)
