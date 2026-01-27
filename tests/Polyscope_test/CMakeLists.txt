if(${RX_USE_POLYSCOPE})
	set(SOURCE_LIST
		test_polyscope.cu
	)

	rxmesh_add_app(Polyscope_test
	  SOURCES ${SOURCE_LIST}
	  LIBS polyscope RXMesh gtest_main
	  FOLDER "tests"
	)
endif()
