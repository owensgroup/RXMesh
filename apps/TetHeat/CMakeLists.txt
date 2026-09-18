set(SOURCE_LIST
  tet_heat.cu
)

rxmesh_add_app(TetHeat
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
