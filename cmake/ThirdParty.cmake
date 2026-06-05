include_guard(GLOBAL)


include("${CMAKE_CURRENT_LIST_DIR}/rapidjson.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rapidobj.cmake")
# OpenMesh is only used by the apps (as a CPU reference); skip it otherwise.
if(RX_BUILD_APPS)
    include("${CMAKE_CURRENT_LIST_DIR}/openmesh.cmake")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/spdlog.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/glm.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/polyscope.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cereal.cmake")
# cuBQL is a CUDA-only BVH library and is not used on the HIP path (the RXMesh
# target does not link it under USE_HIP), so do not fetch it there.
if(NOT USE_HIP)
    include("${CMAKE_CURRENT_LIST_DIR}/cubql.cmake")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/metis.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cpp-cli11.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/suitesparse.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cuDSS.cmake")

