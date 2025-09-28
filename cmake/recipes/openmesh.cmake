# OpenMesh recipe
include(FetchContent)

set(OPENMESH_DOCS false CACHE BOOL "Enable or disable building of documentation")

FetchContent_Declare(openmesh
    GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
    GIT_TAG        OpenMesh-11.0
)

FetchContent_MakeAvailable(openmesh)

# Add OpenMesh to the main library
# Note: OpenMesh targets are linked in the main CMakeLists.txt

# Add definitions
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC -DNO_DECREMENT_DEPRECATED_WARNINGS)
