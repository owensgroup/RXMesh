add_executable(ManiOpt)

set(SOURCE_LIST
    mani_opt.cu	
)

target_sources(ManiOpt 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(ManiOpt PROPERTIES FOLDER "apps")

set_property(TARGET ManiOpt PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "ManiOpt" FILES ${SOURCE_LIST})

target_link_libraries(ManiOpt     
    PRIVATE RXMesh
)

#gtest_discover_tests( ManiOpt )