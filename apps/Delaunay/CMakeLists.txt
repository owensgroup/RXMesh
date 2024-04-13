add_executable(DelaunayEdgeFlip)

set(SOURCE_LIST
    delaunay_edge_flip.cu  
	delaunay_rxmesh.cuh	
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(DelaunayEdgeFlip 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(DelaunayEdgeFlip
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(DelaunayEdgeFlip PROPERTIES FOLDER "apps")

set_property(TARGET DelaunayEdgeFlip PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "DelaunayEdgeFlip" FILES ${SOURCE_LIST})

target_link_libraries(DelaunayEdgeFlip     
    PRIVATE RXMesh
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

#gtest_discover_tests( DelaunayEdgeFlip )