set(SOURCE_LIST
  dgpc.cu
)

rxmesh_add_app(DGPC
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
