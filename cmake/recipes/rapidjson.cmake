# cmake/recipes/rapidjson.cmake
cmake_minimum_required(VERSION 3.20)
include(FetchContent)

# Ensure CMP0175 defaults to OLD for subprojects that don't set it
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD CACHE STRING "Work around RapidJSON add_custom_command policy" FORCE)

# Disable RapidJSON extras
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_THIRDPARTY_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(rapidjson
        GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(rapidjson)
