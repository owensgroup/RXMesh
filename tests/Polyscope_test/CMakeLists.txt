if(${RX_USE_POLYSCOPE})
	add_executable( Polyscope_test )

	set( SOURCE_LIST   
		test_polyscope.cu
	)

	target_sources( Polyscope_test 
	    PRIVATE
		${SOURCE_LIST}    
	)

	set_target_properties( Polyscope_test PROPERTIES FOLDER "tests")

	set_property(TARGET Polyscope_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

	source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Polyscope_test" FILES ${SOURCE_LIST})

	target_link_libraries( Polyscope_test
		PRIVATE polyscope	    
	    PRIVATE RXMesh
		PRIVATE gtest_main
	)

	#gtest_discover_tests( Polyscope_test )
endif()