add_executable(NDReorder)

set(SOURCE_LIST
    test_all_permutations.cu     
	count_nnz_fillin.h
	render_permutation.h
)

target_sources(NDReorder 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

set_target_properties(NDReorder PROPERTIES FOLDER "apps")

set_property(TARGET NDReorder PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NDReorder" FILES ${SOURCE_LIST})

target_link_libraries( NDReorder 
    PRIVATE RXMesh 
    PRIVATE metis
    PRIVATE Eigen3::Eigen
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(NDReorder CopyCUDSSDLL)
endif()