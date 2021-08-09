add_executable(MCF)

target_include_directories(MCF
    PRIVATE ${openmesh_SOURCE_DIR}
)

set(SOURCE_LIST    
    mcf.cu
	mcf_rxmesh_kernel.cuh
	mcf_openmesh.h
	mcf_rxmesh.h	
	mcf_util.h
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	 ../common/openmesh_report.h
)

target_sources(MCF 
    PRIVATE
	${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(MCF
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties( MCF PROPERTIES FOLDER "apps")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "MCF" FILES ${SOURCE_LIST})


target_link_libraries( MCF 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

#gtest_discover_tests( MCF )