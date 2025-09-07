add_executable(DiffARAP)

set(SOURCE_LIST
    diff_arap.cu	
	arap_kernels.h
)

target_sources(DiffARAP 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(DiffARAP PROPERTIES FOLDER "apps")

set_property(TARGET DiffARAP PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "DiffARAP" FILES ${SOURCE_LIST})

target_link_libraries(DiffARAP     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(DiffARAP CopyCUDSSDLL)
endif()
#gtest_discover_tests( DiffARAP )