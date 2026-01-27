set(SOURCE_LIST
  embedded_arap.cu
  util.h
)

rxmesh_add_app(EmbeddedARAP
  SOURCES ${SOURCE_LIST}
  LIBS RXMesh CLI11::CLI11
)
