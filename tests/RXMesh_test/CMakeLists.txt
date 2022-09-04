add_executable( RXMesh_test )

set( SOURCE_LIST
    rxmesh_test_main.cu
	rxmesh_test.h
    test_attribute.cuh
	test_vector.cu
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
)

target_sources( RXMesh_test 
    PRIVATE
	${SOURCE_LIST}    
)

set_target_properties( RXMesh_test PROPERTIES FOLDER "tests")

set_property(TARGET RXMesh_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "RXMesh_test" FILES ${SOURCE_LIST})

target_link_libraries( RXMesh_test
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
	PRIVATE gtest_main
)

#gtest_discover_tests( RXMesh_test )