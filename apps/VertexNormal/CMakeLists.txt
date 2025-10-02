add_executable(VertexNormal)

set(SOURCE_LIST
    vertex_normal.cu 
    vertex_normal_ref.h
    vertex_normal_kernel.cuh
	vertex_normal_hardwired.cuh
)

target_sources(VertexNormal 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(VertexNormal PROPERTIES FOLDER "apps")

set_property(TARGET VertexNormal PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "VertexNormal" FILES ${SOURCE_LIST})

target_link_libraries( VertexNormal     
    PRIVATE RXMesh
    PRIVATE gtest_main
)

if(WIN32 AND RX_USE_CUDSS)
	add_dependencies(VertexNormal CopyCUDSSDLL)
endif()

#gtest_discover_tests( VertexNormal )