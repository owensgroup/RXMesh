add_executable(NDReorder)

set(SOURCE_LIST
    nd_reorder.cu 
    nd_reorder_kernel.cuh
)

target_sources(NDReorder 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(NDReorder PROPERTIES FOLDER "apps")

set_property(TARGET NDReorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NDReorder" FILES ${SOURCE_LIST})

target_link_libraries( NDReorder 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)
