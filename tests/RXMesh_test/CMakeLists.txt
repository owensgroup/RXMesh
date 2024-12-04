add_executable( RXMesh_test )

set( SOURCE_LIST
    rxmesh_test_main.cu	
	rxmesh_test.h
    test_attribute.cu
    test_util.cu
	test_iterator.cu
    test_queries.h
	test_queries_oriented.cu
	test_higher_queries.cu
	query_kernel.cuh
	higher_query.cuh
	test_for_each.cu
	test_validate.cu
	test_lp_pair.cu
	test_dynamic.cu
	test_ev_diamond.cu
	test_sparse_matrix.cu
	test_patch_lock.cuh
	test_patch_scheduler.cuh
	test_patch_slicing.cu
	test_multi_queries.cu
	test_wasted_work.cuh
	test_eigen.cu
	test_boundary.cu
	test_dense_matrix.cu
	test_export.cu
	test_svd.cu
	test_scalar.cu
	test_diff_attribute.cu
	test_inverse.cu
	test_grad.h	
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