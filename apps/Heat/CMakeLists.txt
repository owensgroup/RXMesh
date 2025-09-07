add_executable(Heat)

set(SOURCE_LIST
    heat.cu	
)

target_sources(Heat 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(Heat PROPERTIES FOLDER "apps")

set_property(TARGET Heat PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Heat" FILES ${SOURCE_LIST})

target_link_libraries(Heat     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(Heat CopyCUDSSDLL)
endif()

#gtest_discover_tests( Heat )