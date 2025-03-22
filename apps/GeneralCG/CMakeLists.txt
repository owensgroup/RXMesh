add_executable(GeneralCG)

set(SOURCE_LIST
    cg.cu	
)

target_sources(GeneralCG 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(GeneralCG PROPERTIES FOLDER "apps")

set_property(TARGET GeneralCG PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "GeneralCG" FILES ${SOURCE_LIST})

target_link_libraries(GeneralCG     
    PRIVATE RXMesh
)

#gtest_discover_tests( GeneralCG )