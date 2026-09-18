set(SOURCE_LIST
  tet_deformation.cu
)

rxmesh_add_app(TetDeformation
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
