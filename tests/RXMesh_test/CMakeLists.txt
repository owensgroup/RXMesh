add_executable( RXMesh_test )

set( SOURCE_LIST
    rxmesh_test_main.cu
	rxmesh_test.h
    test_attribute.cuh	
    test_util.cu
	test_iterator.cu
    test_queries.h
	test_queries_oriented.h
	test_higher_queries.h
	query_kernel.cuh
	higher_query.cuh
	test_for_each.cuh
	test_validate.h
	test_lp_pair.h
	test_dynamic.cuh
	test_ev_diamond.h
	test_sparse_matrix.cuh
	test_patch_lock.cuh
	test_patch_scheduler.cuh
	test_patch_slicing.cuh
	test_multi_queries.cuh
	test_wasted_work.cuh
	test_eigen.cuh
	test_boundary.cuh
	test_dense_matrix.cuh
	test_export.cuh
	test_svd.cuh
)

target_sources( RXMesh_test 
    PRIVATE
	${SOURCE_LIST}    
)

set_target_properties( RXMesh_test PROPERTIES FOLDER "tests")

set_property(TARGET RXMesh_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "RXMesh_test" FILES ${SOURCE_LIST})

target_link_libraries( RXMesh_test    
    PRIVATE RXMesh
	PRIVATE gtest_main
)

#gtest_discover_tests( RXMesh_test )