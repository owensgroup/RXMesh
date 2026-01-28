set(SOURCE_LIST
  qslim.cu
  qslim_rxmesh.cuh
  qslim_kernels.cuh
  link_condition.cuh
)

rxmesh_add_app(QSlim
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
