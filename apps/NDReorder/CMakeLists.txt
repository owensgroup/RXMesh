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
include("eigen.cmake")

set_target_properties(NDReorder PROPERTIES FOLDER "apps")

set_property(TARGET NDReorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NDReorder" FILES ${SOURCE_LIST})

# for python3 headers and includes
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries( NDReorder 
    PRIVATE RXMesh
    PRIVATE gtest_main
    PRIVATE metis
    PRIVATE Eigen3::Eigen
    PRIVATE ${PYTHON_LIBRARIES}
)

# https://eigen.tuxfamily.org/dox/TopicCUDA.html
# target_compile_definitions(SurfaceTracking PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

