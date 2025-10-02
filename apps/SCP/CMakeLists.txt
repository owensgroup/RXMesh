add_executable(SCP)

set(SOURCE_LIST
    scp.cu	
)

target_sources(SCP 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(SCP PROPERTIES FOLDER "apps")

set_property(TARGET SCP PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SCP" FILES ${SOURCE_LIST})

target_link_libraries(SCP     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(SCP CopyCUDSSDLL)
endif()

#gtest_discover_tests( SCP )