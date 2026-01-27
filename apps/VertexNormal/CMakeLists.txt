set(SOURCE_LIST
  vertex_normal.cu
  vertex_normal_ref.h
  vertex_normal_kernel.cuh
  vertex_normal_hardwired.cuh
)

rxmesh_add_app(VertexNormal
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh gtest_main
)
