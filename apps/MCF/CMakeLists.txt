add_executable(MCF)

set(SOURCE_LIST    
    mcf.cu
	mcf_kernels.cuh
	mcf_cg.h	
	mcf_cg_mat_free.h	
	mcf_chol.h	
	mcf_gmg.h	
	mcf_eigen.h
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

set_property(TARGET MCF PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "MCF" FILES ${SOURCE_LIST})


target_link_libraries( MCF     
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)
if (${RX_USE_SUITESPARSE})  
  target_link_libraries(MCF PRIVATE SuiteSparse::CHOLMOD)
endif()
#gtest_discover_tests( MCF )