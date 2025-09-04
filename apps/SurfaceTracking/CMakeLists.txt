add_executable(SurfaceTracking)

set(SOURCE_LIST
    tracking.cu  
	tracking_rxmesh.cuh
	tracking_kernels.cuh
	flipper.cuh
	splitter.cuh
	smoother.cuh
	frame_stepper.h
	simulation.h
	noise.h	
	collapser.cuh
	link_condition.cuh
	util.cuh
)


target_sources(SurfaceTracking 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(SurfaceTracking
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(SurfaceTracking PROPERTIES FOLDER "apps")

set_property(TARGET SurfaceTracking PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SurfaceTracking" FILES ${SOURCE_LIST})

target_link_libraries(SurfaceTracking     
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( SurfaceTracking )