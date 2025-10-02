add_executable(XPBD)

set(SOURCE_LIST
    xpbd.cu	
)

target_sources(XPBD 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(XPBD PROPERTIES FOLDER "apps")

set_property(TARGET XPBD PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "XPBD" FILES ${SOURCE_LIST})

target_link_libraries(XPBD     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(XPBD CopyCUDSSDLL)
endif()

#gtest_discover_tests( XPBD )