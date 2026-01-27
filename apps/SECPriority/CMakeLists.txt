set(SOURCE_LIST
  secp.cu
  secp_rxmesh.cuh
  secp_kernels.cuh
  secp_pair.h
)

# Add cuCollection with priority queue. This should eventually come from
# NVIDIA.
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY ahdhn/cuCollections
  GIT_TAG 9dc051e5fa7283be991404d63da6250d18574541
  OPTIONS
  "BUILD_TESTS OFF"
  "BUILD_BENCHMARKS OFF"
  "BUILD_EXAMPLES OFF"
)

rxmesh_add_app(SECPriority
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh gtest_main cuco
)
