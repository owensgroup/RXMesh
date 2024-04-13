add_executable(ShortestEdgeCollapse)

set(SOURCE_LIST
    sec.cu  
	sec_rxmesh.cuh
	histogram.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(ShortestEdgeCollapse 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(ShortestEdgeCollapse
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(ShortestEdgeCollapse PROPERTIES FOLDER "apps")

set_property(TARGET ShortestEdgeCollapse PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "ShortestEdgeCollapse" FILES ${SOURCE_LIST})

target_link_libraries(ShortestEdgeCollapse     
    PRIVATE RXMesh
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

#gtest_discover_tests( ShortestEdgeCollapse )