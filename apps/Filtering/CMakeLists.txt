add_executable(Filtering)

set(SOURCE_LIST
	filtering_util.h
    filtering_rxmesh_kernel.cuh
    filtering.cu       
	filtering_openmesh.h
	filtering_rxmesh.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(Filtering 
    PRIVATE
	${SOURCE_LIST} ${COMMON_LIST}       
)

if (WIN32)
    target_compile_definitions(Filtering
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(Filtering PROPERTIES FOLDER "apps")

set_property(TARGET Filtering PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Filtering" FILES ${SOURCE_LIST})

target_link_libraries(Filtering     
    PRIVATE RXMesh
	PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

if(WIN32 AND ${RX_USE_CUDSS})
	add_dependencies(Filtering CopyCUDSSDLL)
endif()

#gtest_discover_tests( Filtering )