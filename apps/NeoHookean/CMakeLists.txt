add_executable(NeoHookean)

set(SOURCE_LIST
    neo_hookean.cu		
	inertial_energy.h	
	gravity_energy.h
	barrier_energy.h
	draw.h
	init.h	
	neo_hookean_energy.h
	bending_energy.h	
	scene_config.h
	naive_hessian_update.h
)

target_sources(NeoHookean 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(NeoHookean PROPERTIES FOLDER "apps")

set_property(TARGET NeoHookean PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "NeoHookean" FILES ${SOURCE_LIST})

target_link_libraries(NeoHookean     
    PRIVATE RXMesh
)


if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(NeoHookean CopyCUDSSDLL)
endif()

#gtest_discover_tests( NeoHookean )