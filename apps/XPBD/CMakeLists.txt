set(SOURCE_LIST
  xpbd.cu
)

rxmesh_add_app(XPBD
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
