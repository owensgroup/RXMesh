set(SOURCE_LIST
  arap.cu
)

rxmesh_add_app(ARAP
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
