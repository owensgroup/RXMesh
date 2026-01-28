set(SOURCE_LIST
  polyvector.cu
  raw_field_io.h
)

rxmesh_add_app(PolyVector
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
