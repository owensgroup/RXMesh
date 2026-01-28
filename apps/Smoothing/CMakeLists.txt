set(SOURCE_LIST
  smoothing.cu
  manual.h
)

rxmesh_add_app(Smoothing
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
