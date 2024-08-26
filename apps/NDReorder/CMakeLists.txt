add_executable(NDReorder)

set(SOURCE_LIST
    nd_reorder.cu 
    nd_single_patch_ordering.cuh
    nd_partition_manager.cuh
    nd_cross_patch_ordering.cuh	
    check_nnz.h
)

target_sources(NDReorder 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

include("metis.cmake")

set_target_properties(NDReorder PROPERTIES FOLDER "apps")

set_property(TARGET NDReorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NDReorder" FILES ${SOURCE_LIST})

target_link_libraries( NDReorder 
    PRIVATE RXMesh
    PRIVATE gtest_main
    PRIVATE metis
    PRIVATE Eigen3::Eigen
)