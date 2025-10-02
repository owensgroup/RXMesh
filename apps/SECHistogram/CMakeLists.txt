add_executable(SECHistogram)

set(SOURCE_LIST
    sec.cu  
	sec_rxmesh.cuh
	sec_kernels.cuh	
	link_condition.cuh
)

set(COMMON_LIST    
    ../common/openmesh_trimesh.h
	../common/openmesh_report.h
)

target_sources(SECHistogram 
    PRIVATE
    ${SOURCE_LIST} ${COMMON_LIST}
)

if (WIN32)
    target_compile_definitions(SECHistogram
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(SECHistogram PROPERTIES FOLDER "apps")

set_property(TARGET SECHistogram PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "SECHistogram" FILES ${SOURCE_LIST})

target_link_libraries(SECHistogram     
    PRIVATE RXMesh
    PRIVATE gtest_main
	PRIVATE OpenMeshCore
    PRIVATE OpenMeshTools
)

if(WIN32 AND RX_USE_CUDSS)
	add_dependencies(SECHistogram CopyCUDSSDLL)
endif()

#gtest_discover_tests( SECHistogram )