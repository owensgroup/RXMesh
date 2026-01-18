add_executable(EmbeddedARAP)

set(SOURCE_LIST
    embedded_arap.cu
	util.h
)

target_sources(EmbeddedARAP 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(EmbeddedARAP PROPERTIES FOLDER "apps")

set_property(TARGET EmbeddedARAP PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "EmbeddedARAP" FILES ${SOURCE_LIST})

target_link_libraries(EmbeddedARAP     
    PRIVATE RXMesh
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(EmbeddedARAP CopyCUDSSDLL)
endif()
#gtest_discover_tests( EmbeddedARAP )