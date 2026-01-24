include_guard(GLOBAL)

include(FetchContent)

# OpenMesh
set(OPENMESH_DOCS false CACHE BOOL "Enable or disable building of documentation")
FetchContent_Declare(openmesh
    GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
    GIT_TAG        OpenMesh-8.1
)
FetchContent_MakeAvailable(openmesh)
add_definitions(-DNO_DECREMENT_DEPRECATED_WARNINGS)

