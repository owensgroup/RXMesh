set(SOURCE_LIST
  scp.cu
)

rxmesh_add_app(SCP
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
