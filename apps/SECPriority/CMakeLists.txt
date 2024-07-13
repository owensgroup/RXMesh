add_executable(SECPriority)

set(SOURCE_LIST
    #main.cu
    secp.cu
	secp_rxmesh.cuh
	secp_kernels.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(SECPriority 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(SECPriority
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(SECPriority PROPERTIES FOLDER "apps")

set_property(TARGET SECPriority PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SECPriority" FILES ${SOURCE_LIST})

target_link_libraries(SECPriority     
    PRIVATE RXMesh
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
    PRIVATE cuco
)

#gtest_discover_tests( SECPriority )