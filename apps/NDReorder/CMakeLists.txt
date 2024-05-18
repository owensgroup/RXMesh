add_executable(NDReorder)

set(SOURCE_LIST
    nd_reorder.cu 
    nd_reorder_kernel.cuh
    nd_partition_manager.cuh
)

target_sources(NDReorder 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

include("eigen.cmake")

set_target_properties(NDReorder PROPERTIES FOLDER "apps")

set_property(TARGET NDReorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NDReorder" FILES ${SOURCE_LIST})

target_link_libraries( NDReorder 
    PRIVATE RXMesh
    PRIVATE gtest_main
    PRIVATE Eigen3::Eigen
)

# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(SurfaceTracking PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

