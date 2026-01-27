set(SOURCE_LIST
  diff_arap.cu
  arap_kernels.h
)

rxmesh_add_app(DiffARAP
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
