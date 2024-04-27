add_executable(SurfaceTracking)

set(SOURCE_LIST
    tracking.cu  
	tracking_rxmesh.cuh
	tracking_kernels.cuh
	improving_kernels.cuh
	frame_stepper.h
	simulation.h
	noise.h
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

include("eigen.cmake")

set_target_properties(SurfaceTracking PROPERTIES FOLDER "apps")

set_property(TARGET SurfaceTracking PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SurfaceTracking" FILES ${SOURCE_LIST})

target_link_libraries(SurfaceTracking     
    PRIVATE RXMesh
    PRIVATE gtest_main	
	PRIVATE Eigen3::Eigen
)

# https://eigen.tuxfamily.org/dox/TopicCUDA.html
target_compile_definitions(SurfaceTracking PUBLIC "EIGEN_DEFAULT_DENSE_INDEX_TYPE=int")

#gtest_discover_tests( SurfaceTracking )