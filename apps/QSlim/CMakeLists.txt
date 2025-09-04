add_executable(QSlim)

set(SOURCE_LIST
    qslim.cu  
	qslim_rxmesh.cuh	
	qslim_kernels.cuh
	link_condition.cuh
)

target_sources(QSlim 
    PRIVATE
    ${SOURCE_LIST}
)

if (WIN32)
    target_compile_definitions(QSlim
      PRIVATE _USE_MATH_DEFINES 
      PRIVATE NOMINMAX
      PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(QSlim PROPERTIES FOLDER "apps")

set_property(TARGET QSlim PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "QSlim" FILES ${SOURCE_LIST})

target_link_libraries(QSlim     
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( QSlim )