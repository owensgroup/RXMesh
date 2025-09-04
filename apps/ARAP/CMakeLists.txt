add_executable(ARAP)

set(SOURCE_LIST
    arap.cu	
)

target_sources(ARAP 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(ARAP PROPERTIES FOLDER "apps")

set_property(TARGET ARAP PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "ARAP" FILES ${SOURCE_LIST})

target_link_libraries(ARAP     
    PRIVATE RXMesh_lib
)

#gtest_discover_tests( ARAP )