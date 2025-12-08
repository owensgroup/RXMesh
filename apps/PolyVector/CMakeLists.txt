add_executable(PolyVector)

set(SOURCE_LIST
    polyvector.cu	
	read_raw_field.h
)

target_sources(PolyVector 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(PolyVector PROPERTIES FOLDER "apps")

set_property(TARGET PolyVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "PolyVector" FILES ${SOURCE_LIST})

target_link_libraries(PolyVector     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(PolyVector CopyCUDSSDLL)
endif()

#gtest_discover_tests( PolyVector )