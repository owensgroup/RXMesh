add_executable( RXMesh_test )

set( SOURCE_LIST
    rxmesh_test_main.cu
	rxmesh_test.h
    test_attribute.cu
	test_vector.cu
    test_util.cu        
	test_iterator.cu
    test_queries.h
	#test_higher_queries.h
	query.cuh	
	#higher_query.cuh
)

target_sources( RXMesh_test 
    PRIVATE
	${SOURCE_LIST}    
)

set_target_properties( RXMesh_test PROPERTIES FOLDER "tests")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "RXMesh_test" FILES ${SOURCE_LIST})

target_link_libraries( RXMesh_test
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
	PRIVATE gtest_main
)

#gtest_discover_tests( RXMesh_test )