add_executable(Simplification)

set(SOURCE_LIST
    simplification.cu  
	simplification_rxmesh.cuh	
	simplification_kernels.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(Simplification 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(Simplification
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(Simplification PROPERTIES FOLDER "apps")

set_property(TARGET Simplification PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "Simplification" FILES ${SOURCE_LIST})

target_link_libraries(Simplification 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

#gtest_discover_tests( Simplification )