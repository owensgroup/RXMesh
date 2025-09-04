add_executable(GaussianCurvature)

set(SOURCE_LIST
    gaussian_curvature.cu 
    gaussian_curvature_ref.h
    gaussian_curvature_kernel.cuh
)

target_sources(GaussianCurvature 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(GaussianCurvature PROPERTIES FOLDER "apps")

set_property(TARGET GaussianCurvature PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "GaussianCurvature" FILES ${SOURCE_LIST})

target_link_libraries( GaussianCurvature     
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( GaussianCurvature )