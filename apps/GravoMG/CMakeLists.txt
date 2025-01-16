add_executable(GravoMG)

set(SOURCE_LIST
    gmg.cu	
)

target_sources(GravoMG 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(GravoMG PROPERTIES FOLDER "apps")

set_property(TARGET GravoMG PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "GravoMG" FILES ${SOURCE_LIST})

target_link_libraries(GravoMG     
    PRIVATE RXMesh
)

target_include_directories(GravoMG PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)

#gtest_discover_tests( GravoMG )