add_executable(DelaunayEdgeFlip)

set(SOURCE_LIST
    delaunay_edge_flip.cu  
	delaunay_rxmesh.cuh	
)

target_sources(DelaunayEdgeFlip 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(DelaunayEdgeFlip PROPERTIES FOLDER "apps")

set_property(TARGET DelaunayEdgeFlip PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "DelaunayEdgeFlip" FILES ${SOURCE_LIST})

target_link_libraries(DelaunayEdgeFlip 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( DelaunayEdgeFlip )