add_executable(Smoothing)

set(SOURCE_LIST
    smoothing.cu
	manual.h
)

target_sources(Smoothing 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(Smoothing PROPERTIES FOLDER "apps")

set_property(TARGET Smoothing PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Smoothing" FILES ${SOURCE_LIST})

target_link_libraries(Smoothing     
    PRIVATE RXMesh
)

#gtest_discover_tests( Smoothing )