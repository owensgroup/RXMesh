add_executable(SECPriority)

set(SOURCE_LIST
    secp.cu
	secp_rxmesh.cuh
	secp_kernels.cuh
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

# Add cuCollection with priority queue. This should eventually come from
# NVIDIA.
CPMAddPackage(
    NAME cuco
    GITHUB_REPOSITORY andrewbriand/cuCollections
    GIT_TAG d58dd9fedde721a264c8ae960f7393a3a3b08c58
    OPTIONS
       "BUILD_TESTS OFF"
       "BUILD_BENCHMARKS OFF"
       "BUILD_EXAMPLES OFF"
)

target_link_libraries(SECPriority     
    PRIVATE RXMesh
    PRIVATE gtest_main	
    PRIVATE cuco
)

#gtest_discover_tests( SECPriority )