set(SOURCE_LIST
  heat.cu
)

rxmesh_add_app(Heat
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
