include_guard(GLOBAL)

include(FetchContent)

# rapidjson
FetchContent_Declare(rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        8f4c021fa2f1e001d2376095928fc0532adf2ae6
)
FetchContent_Populate(rapidjson)

