add_executable(Param)

set(SOURCE_LIST
    param.cu	
)

target_sources(Param 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(Param PROPERTIES FOLDER "apps")

set_property(TARGET Param PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Param" FILES ${SOURCE_LIST})

target_link_libraries(Param     
    PRIVATE RXMesh
)

#gtest_discover_tests( Param )