# OpenMesh recipe
include(FetchContent)

set(OPENMESH_DOCS false CACHE BOOL "Enable or disable building of documentation")

FetchContent_Declare(openmesh
    GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
    GIT_TAG        OpenMesh-11.0
)

FetchContent_MakeAvailable(openmesh)


