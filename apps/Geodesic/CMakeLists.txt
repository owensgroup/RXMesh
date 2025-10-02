add_executable(Geodesic)

set(SOURCE_LIST    
    geodesic.cu
	geodesic_kernel.cuh
	geodesic_ptp_openmesh.h	
	geodesic_ptp_rxmesh.h	
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(Geodesic
    PRIVATE
	${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(Geodesic
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties( Geodesic PROPERTIES FOLDER "apps")

set_property(TARGET Geodesic PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Geodesic" FILES ${SOURCE_LIST})

target_link_libraries( Geodesic    
    PRIVATE RXMesh
	PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(Geodesic CopyCUDSSDLL)
endif()

#gtest_discover_tests(Geodesic)