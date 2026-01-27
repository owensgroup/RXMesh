set(SOURCE_LIST
  param.cu
)

rxmesh_add_app(Param
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh
)
