add_executable(Remesh)

set(SOURCE_LIST
    remesh.cu  
	remesh_rxmesh.cuh	
	util.cuh	
	split.cuh	
	collapse.cuh	
	flip.cuh	
	smoothing.cuh	
	link_condition.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(Remesh 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(Remesh
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(Remesh PROPERTIES FOLDER "apps")

set_property(TARGET Remesh PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Remesh" FILES ${SOURCE_LIST})

target_link_libraries(Remesh     
    PRIVATE RXMesh
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

#gtest_discover_tests( Remesh )