add_executable(MassSpring)

set(SOURCE_LIST
    mass_spring.cu	
)

target_sources(MassSpring 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(MassSpring PROPERTIES FOLDER "apps")

set_property(TARGET MassSpring PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "MassSpring" FILES ${SOURCE_LIST})

target_link_libraries(MassSpring     
    PRIVATE RXMesh
)

#gtest_discover_tests( MassSpring )